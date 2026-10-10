import csv, hashlib, json
from pathlib import Path
import numpy as np
from evaluation import evaluate_closed, evaluate_open, unknown_scores, calibrate_threshold, rank

def manifests(folder):
    paths=sorted(Path(folder).glob('seed_*.json'))
    ms=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    if [m['seed'] for m in ms]!=[42,43,44,45,46]:raise ValueError('Expected the five original seeds 42–46')
    cat=ms[0]['catalog']
    for m in ms:
        if m['catalog']!=cat:raise ValueError('Catalogs differ')
        sets=[set(m[k]) for k in ['reserved_train_identities','validation_identities','test_identities']]
        if any(sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)]):raise ValueError('Identity leakage')
    return ms,cat

def fingerprint(cat):return hashlib.sha256(json.dumps(cat,sort_keys=True).encode()).hexdigest()

def write_csv(path,rows):
    if not rows:raise ValueError('No output rows')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,delimiter=';');w.writeheader();w.writerows(rows)

def validation_episode(m):
    cat=m['catalog'];val=set(m['validation_identities']);e=m['closed_115']
    return {'gallery':{k:[i for i in g if cat[str(i)]['identity'] in val] for k,g in e['gallery'].items()},
            'queries':[i for i in e['queries'] if cat[str(i)]['identity'] in val]}

def validation_score(m,features,lookup):
    e=validation_episode(m);cat=m['catalog'];scores=[]
    for k in ['1','3','5']:
        ids,_,order=rank(e['gallery'][k],e['queries'],features,lookup,cat)
        bucket={}
        for n,i in enumerate(e['queries']):
            true=cat[str(i)]['identity'];bucket.setdefault(true,[]).append(ids[order[n,0]]==true)
        scores.append(np.mean([np.mean(v) for v in bucket.values()]))
    return float(np.mean(scores))

def evaluate(m,features,ids,method):
    cat=m['catalog'];lookup={str(i):n for n,i in enumerate(ids)};predictions=[];results=[]
    for k in [1,3,5]:
        results.append(evaluate_closed(m['reid_25'],k,features,lookup,cat,m['seed'],predictions,'reid_25'))
        val=m['open_validation'];s=unknown_scores(val['known'],k,val['unknown_queries'],features,lookup,cat)
        threshold=calibrate_threshold(s,.05)
        if np.mean(s>=threshold)>.05:raise ValueError('Invalid calibration')
        test=m['open_test']
        for protocol,unknown in [('open_test',test['unknown_queries']),('external_7',m['external_unknown_queries'])]:
            result=evaluate_open(test['known'],k,unknown,threshold,features,lookup,cat,m['seed'],predictions,protocol)
            result['validation_photo_ufa']=float(np.mean(s>=threshold));results.append(result)
    for r in results+predictions:r['method']=method
    return results,predictions

def load_embeddings(path,cat):
    with np.load(path,allow_pickle=False) as z:
        if str(z['fingerprint'])!=fingerprint(cat):raise ValueError('Embedding catalog mismatch')
        ids=[str(i) for i in z['ids'].tolist()];features=z['features'].astype(np.float64)
    if ids!=sorted(cat,key=int) or features.shape!=(len(cat),384):raise ValueError('Expected ViT-S/14 384D embeddings')
    if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5):raise ValueError('Invalid normalized features')
    return ids,features
