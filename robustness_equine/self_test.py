"""Offline scientific plumbing tests; synthetic features are NOT real results."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from core import fingerprint, restricted_manifest, fit_projection, load_features
from triplet import normalize, loss_grad
from background_controls import two_views
import evaluation as ev

HERE=Path(__file__).resolve().parent

def check_gradient():
    rng=np.random.default_rng(55);x=normalize(rng.normal(size=(8,7)))
    labels=np.repeat(np.arange(4),2);w=rng.normal(size=(7,3))
    _,grad=loss_grad(x,labels,w);numerical=np.zeros_like(w);eps=1e-6
    for idx in np.ndindex(w.shape):
        plus=w.copy();minus=w.copy();plus[idx]+=eps;minus[idx]-=eps
        numerical[idx]=(loss_grad(x,labels,plus)[0]-loss_grad(x,labels,minus)[0])/(2*eps)
    assert np.max(np.abs(grad-numerical))<1e-6

def fixture(root):
    catalog={};by_label={};idx=0
    labels=[f'pony_{i:03}' for i in range(122)]
    for number,label in enumerate(labels):
        by_label[label]=[]
        for j in range(7 if number<115 else 3):
            key=str(idx);catalog[key]={'identity':label,'poney':label,'relative_path':f'{label}/{j}.jpg','sha256':'synthetic'}
            by_label[label].append(idx);idx+=1
    def episode(selected):
        return {'identities':list(selected),'gallery':{str(k):[i for label in selected for i in by_label[label][:k]] for k in [1,3,5]},
                'queries':[i for label in selected for i in by_label[label][5:]]}
    manifests=[];(root/'splits').mkdir()
    for seed in range(42,47):
        rng=np.random.default_rng(seed);main=rng.permutation(labels[:115]).tolist()
        train,val,test=main[:70],main[70:90],main[90:]
        m={'seed':seed,'catalog':catalog,'reserved_train_identities':train,'validation_identities':val,'test_identities':test,
           'closed_115':episode(labels[:115]),'reid_25':episode(test),
           'external_unknown_queries':[i for label in labels[115:] for i in by_label[label]]}
        for name,group in [('open_validation',val),('open_test',test)]:
            m[name]={'known':episode(group[:10]),'unknown_queries':[i for label in group[10:] for i in by_label[label]]}
        (root/'splits'/f'seed_{seed}.json').write_text(json.dumps(m),encoding='utf-8');manifests.append(m)
    ids=sorted(catalog,key=int);rng=np.random.default_rng(10)
    for name,dim in [('dinov2',16),('resnet50',24),('megadescriptor',32),('miewid',40)]:
        centers={label:rng.normal(size=dim) for label in labels}
        x=normalize(np.stack([centers[catalog[i]['identity']]+rng.normal(scale=.3,size=dim) for i in ids]))
        np.savez_compressed(root/f'embeddings_{name}.npz',ids=np.array(ids),features=x,
            fingerprint=fingerprint(catalog),metadata=json.dumps({'model':name,'synthetic':True}))
    return manifests,ids

def main():
    check_gradient()
    array=np.full((224,224,3),[20,40,60],dtype=np.uint8);mask=np.zeros((224,224),bool);mask[:,:112]=True
    fg,bg=two_views(Image.fromarray(array),mask)
    assert np.array_equal(np.asarray(fg)[:,:112],array[:,:112]) and (np.asarray(fg)[:,112:]==127).all()
    assert (np.asarray(bg)[:,:112]==127).all() and np.array_equal(np.asarray(bg)[:,112:],array[:,112:])
    with tempfile.TemporaryDirectory(prefix='equine_tests_') as tmp:
        root=Path(tmp);manifests,ids=fixture(root);m=manifests[0];cat=m['catalog']
        eligible=set(ids);first=m['reid_25']['identities'][0];second=m['reid_25']['identities'][1]
        group=[i for i in ids if cat[i]['identity']==first];eligible.remove(group[2])
        for i in [i for i in ids if cat[i]['identity']==second][:3]:eligible.remove(i)
        sub=restricted_manifest(m,eligible);e=sub['reid_25']
        assert first in e['identities'] and second not in e['identities']
        assert set(e['gallery']['1'])<=set(e['gallery']['3'])<=set(e['gallery']['5'])
        assert not set(e['gallery']['5'])&set(e['queries'])
        assert all(str(i) in eligible for i in e['gallery']['5']+e['queries'])
        args=[sys.executable,str(HERE/'run_robustness.py'),'--stage','projections','--project-root',str(root),
              '--splits',str(root/'splits'),'--output',str(root/'out'),'--epochs','1','--steps-per-epoch','1','--output-dim','4','--threads','1']
        run=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode==0,run.stdout
        doc=json.loads((root/'out/projections/results.json').read_text(encoding='utf-8'))
        assert len(doc['metadata'])==4
        assert len([r for r in doc['results'] if r['protocol']=='reid_25'])==4*5*3*3
        assert all(abs(r['chance_top1']-.04)<1e-12 for r in doc['results'] if r['protocol']=='reid_25')
        assert all(r['validation_photo_ufa']<=.05 for r in doc['results'] if r['protocol']=='open_test')
        xids,x,_=load_features(root/'embeddings_dinov2.npz',cat)
        w,pca,report=fit_projection(m,xids,x,root/'out/projections/dinov2',{'epochs':1,'steps_per_epoch':1,'output_dim':4})
        assert set(report['training_identities'])==set(m['reserved_train_identities'])
        assert not set(report['training_identities'])&set(m['test_identities'])
        assert report['selected_epoch'] in [0,1]
        # A changed budget must never silently reuse a finished checkpoint.
        try:fit_projection(m,xids,x,root/'out/projections/dinov2',{'epochs':2,'steps_per_epoch':1,'output_dim':4})
        except ValueError:pass
        else:raise AssertionError('Checkpoint incompatible accepté')
        scores=np.array([.1,.2,.3,.3,.3,.8,.9])
        t=ev.calibrate_threshold(scores,.2);assert np.mean(scores>=t)<=.2
        plot=subprocess.run([sys.executable,str(HERE/'plot_results.py'),'--results',str(root/'out')],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert plot.returncode==0,plot.stdout
        assert len(list((root/'out/figures').glob('*.png')))>=8
    print('PASS : gradient, masques complémentaires, cohorte commune, exclusion des échecs, séparation des identités, 4 descripteurs × 5 graines, rejet, reprise et figures. Aucun modèle téléchargé.')

if __name__=='__main__':main()
