"""Tests scientifiques hors ligne sur données synthétiques uniquement."""
import csv
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from core import fingerprint, load_features, evaluate, normalize
from run_background_transfer import (clean_patch, compose, role_manifest, donor_pools,
                                    assignment, write_outputs, load_weights)
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
    rng=np.random.default_rng(19)
    a=rng.integers(0,256,(224,224,3),dtype=np.uint8)
    mask=np.zeros((224,224),bool);mask[80:144,80:144]=True
    image=Image.fromarray(a);donor=Image.new('RGB',(224,224),(30,60,90))
    out=np.asarray(compose(image,mask,donor))
    assert np.array_equal(out[mask],a[mask])
    assert (out[~mask]==[30,60,90]).all()
    patch,box=clean_patch(image,mask)
    assert patch is not None and not mask[box[1]:box[3],box[0]:box[2]].any()
    assert clean_patch(Image.new('RGB',(224,224)),mask)==(None,None)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);manifests,_=fixture(root);m=manifests[0]
        ids,x,_=load_features(root/'embeddings_dinov2.npz',m['catalog'])
        rm=role_manifest(m);keys=sorted(rm['catalog']);lookup={key:i for i,key in enumerate(ids)}
        rx=x[[lookup[key.split(':',1)[1]] for key in keys]]
        r,c,n=evaluate(m,ids,x,'raw','same')
        rr,cc,nn=evaluate(rm,keys,rx,'raw','same')
        assert r==rr and c==cc and n==nn, 'Le remappage change les métriques'
        records=[{'identity':label,'key':label+'_'+str(i)} for label in m['reserved_train_identities'] for i in range(2)]
        records.append({'identity':m['test_identities'][0],'key':'forbidden'})
        pools=donor_pools(records,m,42)
        labels=[{d['identity'] for d in rows} for rows in pools.values()]
        assert all(not labels[i]&labels[j] for i in range(3) for j in range(i))
        assert set.union(*labels)<=set(m['reserved_train_identities'])
        assert assignment(keys,pools,42,0)==assignment(keys,pools,42,0)
        assert assignment(keys,pools,42,0)!=assignment(keys,pools,42,1)
        bad=json.loads(json.dumps(m));bad['reid_25']['queries'].append(bad['reid_25']['gallery']['1'][0])
        try:role_manifest(bad)
        except ValueError:pass
        else:raise AssertionError('Fuite photo acceptée')
        signature='test';train=m['reserved_train_identities']
        np.savez(root/'projection_seed_42.npz',weights=np.ones((384,128)),train_identities=np.array(train),signature=signature)
        (root/'selection_seed_42.json').write_text(json.dumps({'signature':signature,'training_identities':train,'selected_epoch':65,'settings':{}}))
        w,_=load_weights(root,42,m,384);assert w.shape==(384,128)
        rows=[{'method':'raw','condition':'random','protocol':'reid_25','gallery_size':1,'seed':42,'top1_macro':0} for _ in range(3)]
        rows.append(dict(rows[0],seed=43,top1_macro=1))
        write_outputs(root,rows,[],[],{})
        with (root/'summary_transfer.csv').open(encoding='utf-8-sig',newline='') as f:summary=list(csv.DictReader(f,delimiter=';'))
        assert float(summary[0]['mean'])==.5 and int(summary[0]['n_partitions'])==2
    print('PASS : masques, compositing, métriques identiques après remappage, séparation des décors, tirages reproductibles, fuite photo rejetée, checkpoints et agrégation par partition. Aucun téléchargement.')

if __name__=='__main__':main()
