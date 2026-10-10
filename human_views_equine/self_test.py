import csv,json,tempfile
from pathlib import Path
import numpy as np
from core import fingerprint,normalize,load_features
from run_human_views import load_annotations,analyze,export
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

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);ms,_=fixture(root);m=ms[0];cat=m['catalog']
        path=root/'annotations.csv'
        rows=[{'relative_path':c['relative_path'],'sha256':c['sha256'],'identity':c['poney'],
               'view':['head','body','partial'][int(Path(c['relative_path']).stem)%3],
               'image_id':int(i)+10000,'updated_at':'test'} for i,c in cat.items()]
        def write(records):
            with path.open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,rows[0].keys(),delimiter=';');w.writeheader();w.writerows(records)
        write(rows);labels,mapping,meta=load_annotations(path,cat)
        assert len(mapping)==len(cat) and mapping[0]['annotation_image_id']!=mapping[0]['image_index']
        ids,x,_=load_features(root/'embeddings_dinov2.npz',cat)
        result=analyze(m,labels,ids,x,'raw','original','full_original')
        assert len(result)==22
        allrow=next(r for r in result if r['analysis']=='fixed_gallery' and r['query_view']=='all' and r['gallery_size']==5)
        old=ev.evaluate_closed(m['reid_25'],5,x,{i:n for n,i in enumerate(ids)},cat,m['seed'],[],'test')
        assert allrow['top1_macro']==old['top1_macro']
        output=root/'out';output.mkdir();export(output,result,meta)
        assert (output/'human_view_summary.csv').exists()
        bad=[dict(r) for r in rows];bad[0]['sha256']='wrong';write(bad)
        try:load_annotations(path,cat)
        except ValueError:pass
        else:raise AssertionError('Photo modifiée acceptée')
        write(rows+[rows[0]])
        try:load_annotations(path,cat)
        except ValueError:pass
        else:raise AssertionError('Doublon accepté')
    print('PASS : correspondance par chemin/empreinte, indices distincts, catégories humaines, métriques identiques, exports et rejets des incohérences.')

if __name__=='__main__':main()
