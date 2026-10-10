"""Recalculate frozen view analyses from human annotations and cached features."""
import argparse
import csv
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from core import atomic_json,csv_write,load_features,normalize,restricted_manifest
from utils import find_one,load_weights
from strata import measure,paired_episodes
import evaluation as ev

HERE=Path(__file__).resolve().parent

def load_annotations(path,cat):
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f,delimiter=';'))
    by={}
    for row in rows:
        relative=row['relative_path'].replace('\\','/')
        if relative in by:raise ValueError(f'Annotation en double : {relative}')
        if row['view'] not in ['head','body','partial','uncertain','']:raise ValueError('Catégorie invalide')
        by[relative]=row
    labels={};mapping=[]
    for idx,record in cat.items():
        row=by.get(record['relative_path'])
        if row is None:raise ValueError(f'Photo absente du CSV : {record["relative_path"]}')
        if row['sha256']!=record['sha256']:raise ValueError(f'Empreinte incompatible : {record["relative_path"]}')
        if row['identity']!=record.get('poney',record['identity'].split('/')[-1]):raise ValueError('Nom du poney incompatible')
        view=row['view'] or 'unannotated';labels[idx]={'view':view}
        mapping.append({'image_index':idx,'annotation_image_id':row['image_id'],'relative_path':record['relative_path'],
            'identity':record['identity'],'sha256':record['sha256'],'view':view,'annotation_updated_at':row['updated_at']})
    excluded=[row for relative,row in by.items() if relative not in {r['relative_path'] for r in cat.values()}]
    return labels,mapping,{'annotations_received':len(rows),'received_counts':dict(Counter(r['view'] or 'unannotated' for r in rows)),
            'retained_counts':dict(Counter(r['view'] for r in mapping)),
            'excluded_annotations':[{'relative_path':r['relative_path'],'view':r['view']} for r in excluded],
            'matching':'relative_path and SHA-256; annotation image_id is NOT an experiment index'}

def analyze(m,labels,ids,x,method,condition,cohort):
    rows=[]
    for k in [1,3,5]:
        for view in ['all','head','body','partial','uncertain','unannotated']:
            queries=[str(i) for i in m['reid_25']['queries'] if view=='all' or labels[str(i)]['view']==view]
            rows.append(measure(m,ids,x,queries,m['reid_25']['gallery'][str(k)],method,condition,'fixed_gallery','mixed',view,k))
    galleries,queries=paired_episodes(m,labels)
    for rv in ['head','body']:
        for qv in ['head','body']:
            rows.append(measure(m,ids,x,queries[qv],galleries[rv],method,condition,'controlled_pairs',rv,qv,1))
    for row in rows:row['cohort']=cohort
    return rows

def export(output,rows,metadata):
    atomic_json(output/'human_view_results.json',{'metadata':metadata,'results':rows})
    csv_write(output/'human_view_metrics.csv',rows)
    csv_write(output/'human_view_counts.csv',[{k:v for k,v in r.items() if not k.startswith('top')} for r in rows])
    groups=defaultdict(list)
    names=['cohort','method','condition','analysis','reference_view','query_view','gallery_size']
    for r in rows:
        if r['status']=='ok':groups[tuple(r[k] for k in names)].append(r)
    summary=[]
    for key,group in sorted(groups.items()):
        for metric in ['top1_macro','top3_macro','top5_macro']:
            values=[r[metric] for r in group]
            summary.append(dict(zip(names,key),metric=metric,n_partitions=len(group),mean=float(np.mean(values)),
                std_across_partitions=float(np.std(values,ddof=1)) if len(values)>1 else '',
                min_queries=min(r['queries'] for r in group),max_queries=max(r['queries'] for r in group),
                min_candidate_identities=min(r['candidate_identities'] for r in group),
                max_candidate_identities=max(r['candidate_identities'] for r in group)))
    csv_write(output/'human_view_summary.csv',summary)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,default=HERE.parent/'annotation_vues_equine'/'annotations'/'annotations_vues.csv')
    p.add_argument('--project-root',type=Path,default=HERE.parent)
    p.add_argument('--robustness-results',type=Path,default=HERE.parent/'robustness_equine'/'results')
    p.add_argument('--original-embeddings',type=Path,default=HERE.parent/'background_equine'/'results'/'embeddings_dinov2.npz')
    p.add_argument('--foreground-embeddings',type=Path)
    p.add_argument('--audit',type=Path)
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--raw-only',action='store_true',help='Diagnostic sans poids des projections')
    args=p.parse_args()
    if args.threads<1:p.error('--threads doit être positif')
    args.output.mkdir(parents=True,exist_ok=True)
    full=[json.loads((HERE/'splits'/f'seed_{s}.json').read_text(encoding='utf-8')) for s in range(42,47)]
    cat=full[0]['catalog']
    labels,mapping,annotation_meta=load_annotations(args.annotations,cat)
    csv_write(args.output/'annotation_mapping.csv',mapping)
    pipeline=args.project_root/'background_equine'/'results'
    search=pipeline if pipeline.exists() else args.project_root
    args.audit=args.audit or find_one(search,'segmentation_audit.csv')
    foreground_path=args.foreground_embeddings or find_one(search,'embeddings_dinov2_foreground.npz')
    with args.audit.open(encoding='utf-8-sig',newline='') as f:audit={str(r['image_index']):r for r in csv.DictReader(f,delimiter=';')}
    if set(audit)!=set(cat):raise ValueError('Audit incomplet')
    for idx in cat:
        if audit[idx]['relative_path']!=cat[idx]['relative_path'] or audit[idx]['identity']!=cat[idx]['identity']:raise ValueError('Audit incompatible')
    eligible={i for i,row in audit.items() if row['status']=='segmented'}
    matched=[]
    for manifest in full:
        path=args.robustness_results/'background'/f'matched_manifest_seed_{manifest["seed"]}.json'
        expected=restricted_manifest(manifest,eligible)
        if path.exists():
            saved=json.loads(path.read_text(encoding='utf-8'))
            if saved!=expected:raise ValueError('Épisodes appariés différents du protocole v2')
            matched.append(saved)
        elif args.raw_only:matched.append(expected)
        else:raise ValueError(f'Manifeste apparié absent : {path}')
    oi,ox,om=load_features(args.original_embeddings,cat);fi,fx,fm=load_features(foreground_path,cat)
    if om.get('model')!='dinov2' or fm.get('model')!='dinov2_foreground_with_fallback' or ox.shape[1]!=384 or fx.shape[1]!=384:
        raise ValueError('Caches DINOv2 384 attendus')
    if om['model_state_sha256']!=fm['model_state_sha256']:raise ValueError('Poids DINOv2 différents')
    coverage=[]
    for view in ['head','body','partial','uncertain','unannotated']:
        indices=[i for i in cat if labels[i]['view']==view]
        coverage.append({'view':view,'photos':len(indices),'segmented':sum(i in eligible for i in indices),
                         'mask_success_fraction':sum(i in eligible for i in indices)/len(indices) if indices else ''})
    csv_write(args.output/'human_view_segmentation_coverage.csv',coverage)
    meta={'human_annotations':annotation_meta,'raw_only':args.raw_only,'checkpoints':{},
       'input_sha256':{'annotations':ev.file_hash(args.annotations),'original_features':ev.file_hash(args.original_embeddings),
                      'foreground_features':ev.file_hash(foreground_path),'segmentation_audit':ev.file_hash(args.audit)},
       'notes':['One annotator; view labels are human and not independent multi-rater ground truth.',
        'View analysis added adaptively after inspecting automatic classification coverage.',
        'No re-identification training, model selection or hyperparameter adjustment.',
        'full_original: all Test-25 source episodes, original images only, including segmentation failures.',
        'matched_segmented: exactly the v2 common cohort; all fallback photos excluded.',
        'Fixed-gallery strata may differ in identity composition; no causal comparison of views.',
        'Controlled pairs use one reserved head and one reserved body reference per identity, with a shared candidate cohort.',
        'Controlled-pair cohorts are smaller; query identity composition may differ by view.',
        'Partial views retained separately. Empty/tiny cells reported explicitly.',
        'Mean/std across overlapping partitions, not independent confidence intervals.']}
    results=[]
    with threadpool_limits(limits=args.threads):
        for original,sub in zip(full,matched):
            seed=original['seed'];print(f'Partition {seed}',flush=True)
            weights=[]
            if not args.raw_only:
                wo,io=load_weights(args.robustness_results/'projections'/'dinov2',seed,original,384)
                wf,ifg=load_weights(args.robustness_results/'background'/'foreground_projection',seed,original,384)
                weights=[('projection_trained_original',wo),('projection_trained_foreground',wf)]
                meta['checkpoints'][str(seed)]={'original':io,'foreground':ifg}
            for m,condition,cohort,ids,x in [(original,'original','full_original',oi,ox),
                      (sub,'original','matched_segmented',oi,ox),(sub,'gray_foreground','matched_segmented',fi,fx)]:
                for method,f in [('dinov2_raw',x)]+[(name,normalize(x@w)) for name,w in weights]:
                    results.extend(analyze(m,labels,ids,f,method,condition,cohort))
    export(args.output,results,meta)
    print('Répartition retenue :',annotation_meta['retained_counts'],flush=True)
    print(f'Terminé : {args.output}',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:sys.exit(f'Erreur : {type(exc).__name__}: {exc}')
