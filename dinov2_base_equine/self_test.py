"""Offline tests with synthetic features and a mock backbone, never real scores."""
import json,tempfile,sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch
import numpy as np
from PIL import Image
from core import fingerprint,normalize,fit_projection,evaluate
from run_dinov2_base import extract,checked_vectors,verify_sources
import evaluation as ev
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

class Tensor:
    def __init__(self,x):self.x=np.asarray(x)
    def to(self,*args):return self
    def cpu(self):return self
    def numpy(self):return self.x
class Model:
    def eval(self):return self
    def to(self,*args):return self
    def parameters(self):return []
    def state_dict(self):return {}
    def __call__(self,batch):
        x=np.zeros((len(batch.x),768),np.float32)
        x[:,0]=1;x[:,1:4]=batch.x.mean(axis=(2,3))
        return Tensor(x)

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);base=root/'photos';base.mkdir();output=root/'extract';output.mkdir()
        cat={}
        for n in range(60):
            p=base/f'{n}.png';Image.new('RGB',(24,16),(n,40,90)).save(p)
            cat[str(n)]={'relative_path':p.name,'identity':'pony','sha256':ev.file_hash(p)}
        args=SimpleNamespace(base=base,output=output,threads=1,device='cpu',batch_size=4,benchmark_only=True)
        fake=SimpleNamespace(manual_seed=lambda n:None,set_num_threads=lambda n:None,
           cuda=SimpleNamespace(is_available=lambda:False),hub=SimpleNamespace(load=lambda *a,**k:Model()),
           from_numpy=Tensor,stack=lambda tensors:Tensor(np.stack([t.x for t in tensors])),inference_mode=nullcontext)
        with patch.dict(sys.modules,{'torch':fake}):
            assert extract(args,cat) is None
            assert len(list((output/'extraction_cache'/'vectors').glob('*.npy')))==52
            args.benchmark_only=False;ids,x,meta=extract(args,cat)
            assert x.shape==(60,768) and meta['new_images_extracted_this_run']==8
            ids2,x2,_=extract(args,cat);assert ids==ids2 and np.array_equal(x,x2)
        Image.new('RGB',(24,16),'green').save(base/'0.png')
        try:verify_sources(base,cat)
        except ValueError:pass
        else:raise AssertionError('Photo modifiée acceptée')
        vector=output/'extraction_cache'/'vectors'/'0.npy';np.save(vector,np.zeros(768))
        try:checked_vectors(vector.parent,['0'])
        except ValueError:pass
        else:raise AssertionError('Vecteur nul accepté')
        tests=root/'training';tests.mkdir();manifests,ids=fixture(tests);m=manifests[0]
        rng=np.random.default_rng(4);x=normalize(rng.normal(size=(len(ids),20)))
        settings={'epochs':1,'steps_per_epoch':1,'output_dim':8}
        w,pca,doc=fit_projection(m,ids,x,tests/'projections',settings)
        assert w.shape==(20,8) and set(doc['training_identities'])==set(m['reserved_train_identities'])
        w2,p2,d2=fit_projection(m,ids,x,tests/'projections',settings)
        assert np.array_equal(w,w2) and np.array_equal(pca,p2)
        r,c,n=evaluate(m,ids,normalize(x@w),'mock','full_original')
        assert len(r)==9 and len(c)==33 and len(n)==3
        assert all(row['validation_photo_ufa']<=.05 for row in r if row['protocol']=='open_test')
        try:fit_projection(m,ids,x,tests/'projections',dict(settings,epochs=2))
        except ValueError:pass
        else:raise AssertionError('Budget incompatible accepté')
    print('PASS : extraction simulée, benchmark/reprise, cache complet, photo modifiée, vecteurs nuls, train/val/test, PCA/projection, budget et calibration. Aucun téléchargement.')

if __name__=='__main__':main()
