"""Re-evaluate existing frozen features and selected projections without training."""
import argparse,json
from pathlib import Path
import numpy as np
from common import manifests,load_embeddings,evaluate,write_csv,validation_score

def main(a):
    ms,cat=manifests(a.splits);ids,raw=load_embeddings(a.embeddings,cat)
    a.output.mkdir(parents=True,exist_ok=True);results=[];predictions=[];sources=[]
    for m in ms:
        path=a.projections/f"projection_seed_{m['seed']}.npz"
        with np.load(path,allow_pickle=False) as z:
            train=z['train_identities'].tolist()
            if set(train)!=set(m['reserved_train_identities']):raise ValueError('Projection training identities mismatch')
            w=z['weights'].copy();epoch=int(z['selected_epoch']) if 'selected_epoch' in z else None
        if w.shape!=(384,128) or not np.isfinite(w).all():raise ValueError('Expected finite 384x128 projection')
        f=raw@w;f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12)
        sidecar=a.projections/f"selection_seed_{m['seed']}.json"
        if epoch is None and sidecar.exists():
            selection=json.loads(sidecar.read_text(encoding='utf-8'))
            measured=validation_score(m,f,{i:n for n,i in enumerate(ids)})
            if abs(measured-selection['validation_score'])>1e-6:raise ValueError('Projection and selection sidecar disagree')
            epoch=int(selection['selected_epoch'])
        for method,x in [('dinov2',raw),('projection_triplet',f)]:
            r,p=evaluate(m,x,ids,method);results+=r;predictions+=p
        sources.append({'seed':m['seed'],'projection_file':path.name,'selected_epoch':epoch})
    write_csv(a.output/'predictions_reference.csv',predictions)
    (a.output/'results_reference.json').write_text(json.dumps({'sources':sources,'results':results},indent=2))
    mean=np.mean([r['top1_macro'] for r in results if r['method']=='projection_triplet' and r['protocol']=='reid_25' and r['gallery_size']==5])
    print(f'Projected five-reference macro Top-1: {100*mean:.4f}% (paper: 89.75%)')
    print('Saved',a.output,'; check selected epochs against the final 200-epoch runs.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=Path,default=Path(__file__).parent/'splits')
    p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--projections',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('reference_results'))
    main(p.parse_args())
