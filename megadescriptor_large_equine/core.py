"""Common cohort construction, projection fitting and matched evaluation."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import evaluation as ev
from triplet import normalize, loss_grad

KS = (1, 3, 5)

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def csv_write(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=names, delimiter=';')
        writer.writeheader(); writer.writerows(rows)

def fingerprint(catalog):
    return hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()

def load_features(path, catalog, partial=False):
    with np.load(path, allow_pickle=False) as z:
        if str(z['fingerprint']) != fingerprint(catalog):
            raise ValueError(f'Catalogue incompatible : {path}')
        ids = [str(i) for i in z['ids'].tolist()]
        x = z['features'].astype(np.float64)
        meta = json.loads(str(z['metadata']))
    if len(set(ids)) != len(ids) or not set(ids) <= set(catalog):
        raise ValueError(f'Indices incompatibles : {path}')
    if not partial and set(ids) != set(catalog):
        raise ValueError(f'Cache incomplet : {path}')
    if x.ndim != 2 or len(x) != len(ids) or not np.isfinite(x).all():
        raise ValueError(f'Vecteurs invalides : {path}')
    if not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5):
        raise ValueError(f'Vecteurs non normalisés : {path}')
    return ids, x, meta

def validation_episode(m):
    labels = set(m['validation_identities']); cat = m['catalog']; e = m['closed_115']
    return {'gallery': {k:[i for i in rows if cat[str(i)]['identity'] in labels]
                        for k, rows in e['gallery'].items()},
            'queries':[i for i in e['queries'] if cat[str(i)]['identity'] in labels]}

def restrict_episode(e, eligible, cat):
    """Fixed new cohort shared by all conditions, at least five refs + one query.

    Keep the old ordered reference reserve first, then its old query order.
    Failed images disappear; eligible old queries can become references. These
    are NEW episodes, not the old benchmark with unchanged references.
    """
    groups = defaultdict(list)
    for idx in list(e['gallery']['5']) + list(e['queries']):
        if str(idx) in eligible:
            groups[cat[str(idx)]['identity']].append(idx)
    groups = {label: rows for label, rows in groups.items() if len(rows) >= 6}
    return {'identities':sorted(groups),
            'gallery':{str(k):[i for label in sorted(groups) for i in groups[label][:k]] for k in KS},
            'queries':[i for label in sorted(groups) for i in groups[label][5:]]}

def restricted_manifest(m, eligible):
    out = dict(m); cat = m['catalog']
    out['closed_115'] = restrict_episode(m['closed_115'], eligible, cat)
    out['reid_25'] = restrict_episode(m['reid_25'], eligible, cat)
    for key in ['open_validation', 'open_test']:
        out[key] = dict(m[key])
        out[key]['known'] = restrict_episode(m[key]['known'], eligible, cat)
        out[key]['unknown_queries'] = [i for i in m[key]['unknown_queries'] if str(i) in eligible]
    out['external_unknown_queries'] = [i for i in m['external_unknown_queries'] if str(i) in eligible]
    return out

def validation_score(e, x, lookup, cat):
    if not e['queries'] or not e['gallery']['5']:
        raise ValueError('Aucune identité de validation avec 5 références + une requête')
    values=[]
    for k in KS:
        labels, _, order = ev.rank(e['gallery'][str(k)], e['queries'], x, lookup, cat)
        buckets=defaultdict(list)
        for j, idx in enumerate(e['queries']):
            true=cat[str(idx)]['identity'];buckets[true].append(labels[order[j,0]]==true)
        values.append(np.mean([np.mean(v) for v in buckets.values()]))
    return float(np.mean(values))

def fit_projection(m, ids, x, folder, settings, eligible=None):
    folder.mkdir(parents=True, exist_ok=True); seed=m['seed']; cat=m['catalog']
    lookup={i:n for n,i in enumerate(ids)}
    train=set(m['reserved_train_identities']); val=set(m['validation_identities']); test=set(m['test_identities'])
    if train & val or train & test or val & test:
        raise ValueError('Identités entraînement/validation/test non disjointes')
    signature=hashlib.sha256(json.dumps({'settings':settings,'manifest':m,'feature_ids':ids,
        'feature_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
        'eligible':sorted(eligible) if eligible is not None else None},sort_keys=True).encode()).hexdigest()
    target=folder/f'projection_seed_{seed}.npz'; report=folder/f'selection_seed_{seed}.json'
    if target.exists() and report.exists():
        doc=json.loads(report.read_text(encoding='utf-8'))
        if doc['signature']!=signature:
            raise ValueError(f'Résultats existants incompatibles : {folder}. Choisir un nouveau --output.')
        with np.load(target,allow_pickle=False) as z:
            return z['weights'].copy(),z['initial_pca'].copy(),doc
    groups={label:[lookup[i] for i in ids if cat[i]['identity']==label and
                   (eligible is None or i in eligible)] for label in sorted(train)}
    groups={label:rows for label,rows in groups.items() if len(rows)>=4}
    if len(groups)<2:
        raise ValueError('Moins de deux identités entraînement avec quatre images admissibles')
    labels=list(groups); rows=[row for label in labels for row in groups[label]]
    dim=settings['output_dim']
    if dim>min(x.shape[1],len(rows)):
        raise ValueError('Dimension de projection supérieure au rang maximal du sous-ensemble')
    # Thin SVD of TRAINING ONLY features: same uncentered PCA subspace, avoiding
    # a D x D eigendecomposition for ResNet/MiewID.
    _,_,vt=np.linalg.svd(x[rows],full_matrices=False)
    initial=vt[:dim].T.copy();w=initial.copy()
    e=validation_episode(m);rng=np.random.default_rng(seed)
    score=validation_score(e,normalize(x@w),lookup,cat)
    best=w.copy();best_score=score;best_epoch=0
    history=[{'seed':seed,'epoch':0,'training_loss':'','validation_top1_mean_1_3_5':score}]
    am=np.zeros_like(w);av=np.zeros_like(w);step=0
    batch_identities=min(16,len(labels))
    for epoch in range(1,settings['epochs']+1):
        losses=[]
        for _ in range(settings['steps_per_epoch']):
            chosen=rng.choice(len(labels),batch_identities,replace=False)
            batch=np.concatenate([rng.choice(groups[labels[i]],4,replace=False) for i in chosen])
            loss,g=loss_grad(x[batch],np.repeat(chosen,4),w);g+=1e-4*w
            step+=1;am=.9*am+.1*g;av=.999*av+.001*g*g
            w-=3e-4*(am/(1-.9**step))/(np.sqrt(av/(1-.999**step))+1e-8)
            if not np.isfinite(w).all():raise ValueError('Projection non finie')
            losses.append(loss)
        if epoch%5==0 or epoch==settings['epochs']:
            score=validation_score(e,normalize(x@w),lookup,cat)
            history.append({'seed':seed,'epoch':epoch,'training_loss':float(np.mean(losses)),
                            'validation_top1_mean_1_3_5':score})
            csv_write(folder/f'training_history_seed_{seed}.csv',history)
            if score>best_score:best_score=score;best=w.copy();best_epoch=epoch
            print(f'  seed {seed}, époque {epoch}/{settings["epochs"]}, val={score:.4f}, meilleur={best_epoch}',flush=True)
    doc={'signature':signature,'seed':seed,'selected_epoch':best_epoch,'validation_score':best_score,
         'training_identities':labels,'excluded_training_identities':sorted(train-set(labels)),
         'training_images':len(rows),'batch_identities':batch_identities,
         'settings':settings,'pca':'uncentered thin SVD on training images only',
         'optimizer':'Adam, lr=3e-4, coupled L2=1e-4, batch-hard cosine margin=0.2',
         'selection':'validation macro Top-1 averaged over 1/3/5; earliest exact tie',
         'history':history,'last_epoch_selected':best_epoch==settings['epochs']}
    tmp=target.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,weights=best,initial_pca=initial,train_identities=np.array(labels),signature=signature)
    tmp.replace(target);atomic_json(report,doc)
    return best,initial,doc

def macro(values, queries, cat):
    buckets=defaultdict(list)
    for idx,value in zip(queries,values):buckets[cat[str(idx)]['identity']].append(value)
    return float(np.mean([np.mean(v) for v in buckets.values()]))

def evaluate(m, ids, x, method, condition):
    cat=m['catalog'];lookup={i:n for n,i in enumerate(ids)};seed=m['seed']
    results=[];curves=[];counts=[]
    for k in KS:
        e=m['reid_25']
        if e['queries'] and e['gallery'][str(k)]:
            r=ev.evaluate_closed(e,k,x,lookup,cat,seed,[],'reid_25')
            r.update(method=method,condition=condition,chance_top1=1/r['identities'])
            results.append(r)
        val=m['open_validation'];test=m['open_test']
        ready=all([val['known']['gallery'][str(k)],val['unknown_queries'],
                   test['known']['gallery'][str(k)],test['known']['queries'],test['unknown_queries']])
        counts.append({'seed':seed,'method':method,'condition':condition,'gallery_size':k,
            'closed_identities':len({cat[str(i)]['identity'] for i in e['queries']}),
            'closed_queries':len(e['queries']),
            'validation_known_identities':len({cat[str(i)]['identity'] for i in val['known']['gallery'][str(k)]}),
            'validation_unknown_identities':len({cat[str(i)]['identity'] for i in val['unknown_queries']}),
            'test_known_identities':len({cat[str(i)]['identity'] for i in test['known']['gallery'][str(k)]}),
            'test_unknown_identities':len({cat[str(i)]['identity'] for i in test['unknown_queries']}),
            'known_queries':len(test['known']['queries']),'unknown_queries':len(test['unknown_queries']),
            'open_set_evaluable':bool(ready)})
        if not ready:continue
        vs=ev.unknown_scores(val['known'],k,val['unknown_queries'],x,lookup,cat)
        threshold=ev.calibrate_threshold(vs,.05)
        for protocol,unknown in [('open_test',test['unknown_queries']),('external_7',m['external_unknown_queries'])]:
            if not unknown:continue
            r=ev.evaluate_open(test['known'],k,unknown,threshold,x,lookup,cat,seed,[],protocol)
            r.update(method=method,condition=condition,validation_photo_ufa=float(np.mean(vs>=threshold)))
            results.append(r)
        labels,scores,order=ev.rank(test['known']['gallery'][str(k)],test['known']['queries'],x,lookup,cat)
        best=scores.max(axis=1).astype(np.float64)
        correct=np.array([labels[order[j,0]]==cat[str(i)]['identity'] for j,i in enumerate(test['known']['queries'])])
        us=ev.unknown_scores(test['known'],k,test['unknown_queries'],x,lookup,cat)
        # All thresholds derive from validation; no operating point chosen on test.
        for alpha in [0,.01,.025,.05,.075,.1,.15,.2,.3,.4,.5]:
            t=ev.calibrate_threshold(vs,alpha)
            curves.append({'seed':seed,'method':method,'condition':condition,'gallery_size':k,
                'validation_target_photo_ufa':alpha,'threshold':t,'validation_observed_photo_ufa':float(np.mean(vs>=t)),
                'test_cia_macro':macro((best>=t)&correct,test['known']['queries'],cat),
                'test_ufa_macro':macro(us>=t,test['unknown_queries'],cat)})
    return results,curves,counts

def save_evaluations(folder, results, curves, counts, metadata):
    atomic_json(folder/'results.json',{'metadata':metadata,'results':results})
    csv_write(folder/'operating_points.csv',curves);csv_write(folder/'episode_counts.csv',counts)
    groups=defaultdict(list)
    for r in results:
        for key,value in r.items():
            if key.endswith('_macro') or key=='chance_top1':
                groups[(r['method'],r['condition'],r['protocol'],r['gallery_size'],key)].append(float(value))
    summary=[]
    for (method,condition,protocol,k,metric),values in sorted(groups.items()):
        summary.append({'method':method,'condition':condition,'protocol':protocol,'gallery_size':k,
            'metric':metric,'n_partitions':len(values),'mean':float(np.mean(values)),
            'std_across_seeds':float(np.std(values,ddof=1)) if len(values)>1 else ''})
    csv_write(folder/'summary.csv',summary)
