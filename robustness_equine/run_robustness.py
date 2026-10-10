"""Contrôles du fond sans fallback et projections 128D de descripteurs gelés."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import evaluation as ev
from core import (load_features, fit_projection, normalize, evaluate, save_evaluations,
                  restricted_manifest, atomic_json, fingerprint)

HERE=Path(__file__).resolve().parent
FILES={'dinov2':'embeddings_dinov2.npz','resnet50':'embeddings_resnet50.npz',
       'megadescriptor':'embeddings_megadescriptor.npz','miewid':'embeddings_miewid.npz',
       'foreground':'embeddings_dinov2_foreground.npz'}

def discover(root, name, override=None):
    if override is not None:
        p=Path(override).resolve()
        if not p.is_file():raise ValueError(f'Cache demandé absent : {p}')
        return p
    matches=sorted(root.rglob(FILES[name]),key=lambda p:(len(p.parts),str(p)))
    if not matches:return None
    # Do not silently choose between divergent copies.
    if len(matches)>1:
        with np.load(matches[0],allow_pickle=False) as z:
            ref_ids=z['ids'].copy();ref=z['features'].copy();fp=str(z['fingerprint'])
        for p in matches[1:]:
            with np.load(p,allow_pickle=False) as z:
                same=np.array_equal(z['ids'],ref_ids) and str(z['fingerprint'])==fp and np.array_equal(z['features'],ref)
            if not same:
                raise ValueError(f'Plusieurs caches {name} différents. Préciser --embedding {name}=CHEMIN')
    return matches[0]

def projection_stage(args, manifests, paths):
    cat=manifests[0]['catalog'];folder=args.output/'projections';folder.mkdir(parents=True,exist_ok=True)
    results=[];curves=[];counts=[];metadata={}
    for name in ['dinov2','resnet50','megadescriptor','miewid']:
        if name not in paths:
            print(f'{name} : cache absent, méthode ignorée et signalée.',flush=True);metadata[name]={'status':'missing_cache'};continue
        print(f'Projection {name} : {paths[name]}',flush=True)
        ids,x,meta=load_features(paths[name],cat)
        if meta.get('model')!=name:raise ValueError(f'Modèle incompatible dans {paths[name]} : {meta.get("model")}')
        metadata[name]={'path':str(paths[name]),'sha256':ev.file_hash(paths[name]),'metadata':meta}
        for m in manifests:
            w,pca,doc=fit_projection(m,ids,x,folder/name,args.settings)
            for label,features in [('raw',x),('pca128',normalize(x@pca)),('triplet128',normalize(x@w))]:
                r,c,n=evaluate(m,ids,features,name+'_'+label,'full_original')
                results+=r;curves+=c;counts+=n
            save_evaluations(folder,results,curves,counts,metadata)
    if not results:raise ValueError('Aucun cache de descripteur original trouvé')

def background_stage(args, manifests, paths):
    from background_controls import create_background
    if 'dinov2' not in paths or 'foreground' not in paths:
        raise ValueError('Les caches DINOv2 original ET foreground sont nécessaires. Utiliser --embedding si besoin.')
    cat=manifests[0]['catalog'];ids,original,orig_meta=load_features(paths['dinov2'],cat)
    fids,foreground,fg_meta=load_features(paths['foreground'],cat)
    if orig_meta.get('model')!='dinov2' or fg_meta.get('model')!='dinov2_foreground_with_fallback':
        raise ValueError('Caches DINOv2 original/foreground attendus')
    if fg_meta.get('model_state_sha256')!=orig_meta.get('model_state_sha256'):
        raise ValueError('Poids DINOv2 original/foreground différents')
    fg_lookup={i:n for n,i in enumerate(fids)}
    foreground=foreground[[fg_lookup[i] for i in ids]]
    if args.audit is None:
        audit_paths=sorted(args.project_root.rglob('segmentation_audit.csv'),key=lambda p:(len(p.parts),str(p)))
        if not audit_paths:raise ValueError('segmentation_audit.csv absent; fournir --audit CHEMIN')
        hashes={ev.file_hash(p) for p in audit_paths}
        if len(hashes)>1:raise ValueError('Plusieurs audits différents : préciser --audit CHEMIN')
        args.audit=audit_paths[0]
    with args.audit.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f,delimiter=';'))
    audit={str(r['image_index']):r for r in rows}
    if len(audit)!=len(rows):raise ValueError('Indices dupliqués dans l’audit')
    import torch
    torch.set_num_threads(args.threads);torch.manual_seed(42)
    args.resolved_device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if args.resolved_device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA indisponible')
    if args.resolved_device=='cuda':
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    bg_path,eligible=create_background(args,cat,audit,fg_meta,orig_meta)
    bids,background,bg_meta=load_features(bg_path,cat,partial=True)
    folder=args.output/'background';results=[];curves=[];counts=[]
    for full in manifests:
        m=restricted_manifest(full,eligible)
        atomic_json(folder/f'matched_manifest_seed_{m["seed"]}.json',m)
        # Original projection trained on original full training images.
        w,pca,selection=fit_projection(full,ids,original,args.output/'projections'/'dinov2',args.settings)
        # Foreground projection trained exclusively on successful foregrounds.
        wf,pf,selection_fg=fit_projection(m,ids,foreground,folder/'foreground_projection',args.settings,eligible)
        conditions=[('original','dinov2_raw',ids,original),
                    ('foreground','dinov2_raw',ids,foreground),
                    ('background','dinov2_raw',bids,background),
                    ('original','projection_trained_original',ids,normalize(original@w)),
                    ('foreground','projection_trained_original',ids,normalize(foreground@w)),
                    ('background','projection_trained_original',bids,normalize(background@w)),
                    ('foreground','pca_trained_foreground',ids,normalize(foreground@pf)),
                    ('foreground','projection_trained_foreground',ids,normalize(foreground@wf))]
        for condition,method,xids,x in conditions:
            r,c,n=evaluate(m,xids,x,method,condition);results+=r;curves+=c;counts+=n
        save_evaluations(folder,results,curves,counts,{'eligible_images':len(eligible),
            'catalog_images':len(cat),'fallback_used':False,'episode_reselection':'5 eligible refs then fixed eligible queries',
            'original':orig_meta,'foreground':fg_meta,'background':bg_meta,
            'limitations':['Automatic masks, not ground truth; largest mask may select a different horse.',
                'Background retains silhouette/position of the gray horse mask.',
                'Subset reference sets/candidate counts differ from the original benchmark.',
                'All conditions share exactly the same newly restricted episodes within each seed.',
                'Exploratory extensions; five overlapping partitions are not independent replications.']})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=HERE.parent,help='Dossier equitrain contenant les expériences précédentes')
    p.add_argument('--base',type=Path,default=ev.DEFAULT_BASE)
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--splits',type=Path,default=HERE/'splits')
    p.add_argument('--stage',choices=['all','projections','background'],default='all')
    p.add_argument('--embedding',action='append',default=[],metavar='MODELE=CHEMIN',help='dinov2/resnet50/megadescriptor/miewid/foreground')
    p.add_argument('--audit',type=Path)
    p.add_argument('--yolo-weights',type=Path)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--steps-per-epoch',type=int,default=10)
    p.add_argument('--output-dim',type=int,default=128)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    for name in ['epochs','steps_per_epoch','output_dim','threads','batch_size']:
        if getattr(args,name)<1:p.error(name+' doit être positif')
    args.project_root=args.project_root.resolve();args.output=args.output.resolve()
    overrides={}
    for item in args.embedding:
        if '=' not in item:p.error('--embedding attend MODELE=CHEMIN')
        name,path=item.split('=',1)
        if name not in FILES or name in overrides:p.error('Modèle inconnu ou fourni deux fois : '+name)
        overrides[name]=path
    manifests=[json.loads(f.read_text(encoding='utf-8')) for f in sorted(args.splits.glob('seed_*.json'))]
    if len(manifests)!=5 or {m['seed'] for m in manifests}!={42,43,44,45,46}:
        raise ValueError('Les cinq partitions originales 42–46 sont attendues')
    if any(m['catalog']!=manifests[0]['catalog'] for m in manifests):raise ValueError('Catalogues divergents')
    paths={}
    for name in FILES:
        if args.stage=='projections' and name=='foreground':continue
        if args.stage=='background' and name not in ['dinov2','foreground']:continue
        path=discover(args.project_root,name,overrides.get(name))
        if path is not None:paths[name]=path;print(f'Cache {name} : {path}',flush=True)
    args.settings={'epochs':args.epochs,'steps_per_epoch':args.steps_per_epoch,'output_dim':args.output_dim}
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/'run_configuration.json',{'settings':args.settings,'stage':args.stage,
        'paths':{k:str(v) for k,v in paths.items()},'catalog_fingerprint':fingerprint(manifests[0]['catalog']),
        'note':'No test-based selection; exploratory extension of previously inspected data.'})
    with threadpool_limits(limits=args.threads):
        if args.stage in ['all','projections']:projection_stage(args,manifests,paths)
        if args.stage in ['all','background']:background_stage(args,manifests,paths)
    print(f'Terminé. Résultats : {args.output}',flush=True)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:sys.exit('Arrêt. Relancer la même commande : caches et graines terminées réutilisés; graine interrompue réentraînée.')
    except Exception as exc:sys.exit(f'Erreur : {type(exc).__name__}: {exc}')
