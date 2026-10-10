"""DINOv2 ViT-B/14 frozen, raw/PCA-128/triplet-128, same five identity splits."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from core import (atomic_json,csv_write,fingerprint,load_features,normalize,
                  fit_projection,evaluate,save_evaluations)
from utils import load_weights
import evaluation as ev

HERE=Path(__file__).resolve().parent

def verify_sources(base,cat):
    ids=sorted(cat,key=int);paths=[]
    for i in ids:
        path=base.joinpath(*cat[i]['relative_path'].split('/'))
        if not path.is_file() or ev.file_hash(path)!=cat[i]['sha256']:
            raise ValueError(f'Photo absente ou modifiée : {path}')
        paths.append(path)
    return ids,paths

def checked_vectors(folder,ids,dimension=768):
    x=np.stack([np.load(folder/f'{i}.npy',allow_pickle=False) for i in ids]).astype(np.float64)
    if x.shape!=(len(ids),dimension) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5):
        raise ValueError('Vecteurs partiels invalides : choisir un nouveau dossier --output')
    return x

def extract(args,cat):
    print('Vérification des photos du corpus…',flush=True)
    ids,paths=verify_sources(args.base,cat);fp=fingerprint(cat)
    output=args.output/'embeddings_dinov2_vitb14.npz'
    if output.exists():
        xids,x,meta=load_features(output,cat)
        if x.shape[1]!=768 or meta.get('model')!='dinov2_vitb14' or meta.get('preprocess')!=ev.PREPROCESS:
            raise ValueError('Cache ViT-B incompatible')
        print(f'Cache ViT-B complet réutilisé : {output}',flush=True)
        return xids,x,meta
    import torch
    torch.manual_seed(42);torch.set_num_threads(args.threads)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA indisponible')
    if device=='cuda':torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    print('Chargement des poids officiels DINOv2 ViT-B/14 (téléchargement si nécessaire)…',flush=True)
    started_loading=time.perf_counter()
    model=torch.hub.load('facebookresearch/dinov2:main','dinov2_vitb14',trust_repo=True).eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    meta={'model':'dinov2_vitb14','weights':'official dinov2_vitb14 pretrained, without registers',
        'model_state_sha256':h.hexdigest(),'preprocess':ev.PREPROCESS,'source_catalog_sha256':fp,
        'number_images':len(ids),'dimension':768,'device':device,'model_training_on_corpus':False,
        'python':platform.python_version(),'packages':{},'load_seconds_this_run':time.perf_counter()-started_loading}
    for name in ['numpy','Pillow','torch','torchvision','threadpoolctl']:
        try:meta['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    signature=hashlib.sha256(json.dumps({'catalog':fp,'model':meta['model'],
        'model_sha256':meta['model_state_sha256'],'preprocess':ev.PREPROCESS},sort_keys=True).encode()).hexdigest()
    cache=args.output/'extraction_cache';cache.mkdir(exist_ok=True);info=cache/'metadata.json'
    if info.exists() and json.loads(info.read_text(encoding='utf-8'))['signature']!=signature:
        raise ValueError('Cache partiel incompatible : choisir un nouveau --output')
    atomic_json(info,{'signature':signature,'metadata':meta})
    vectors=cache/'vectors';vectors.mkdir(exist_ok=True)
    missing=[(i,path) for i,path in zip(ids,paths) if not (vectors/f'{i}.npy').exists()]
    print(f'{len(ids)-len(missing)}/{len(ids)} vecteurs déjà enregistrés ; appareil : {device}',flush=True)
    done=0;durations=[];timings=[];next_notice=50;started=time.perf_counter()
    mean=np.array([.485,.456,.406],dtype=np.float32);std=np.array([.229,.224,.225],dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(missing),args.batch_size):
            group=missing[start:start+args.batch_size];t=time.perf_counter();batch=[]
            for i,path in group:
                a=np.asarray(ev.load_image(path),dtype=np.float32)/255
                batch.append(torch.from_numpy(((a-mean)/std).transpose(2,0,1).copy()))
            features=model(torch.stack(batch).to(device)).cpu().numpy().astype(np.float32)
            features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
            if features.shape!=(len(group),768) or not np.isfinite(features).all():raise ValueError('Extraction ViT-B invalide')
            for (i,path),feature in zip(group,features):
                tmp=vectors/f'{i}.tmp.npy';np.save(tmp,feature);tmp.replace(vectors/f'{i}.npy')
            done+=len(group);duration=time.perf_counter()-t;durations.append((duration,len(group)))
            usable=durations[1:] or durations
            seconds_per_photo=sum(d for d,n in usable)/sum(n for d,n in usable)
            remaining=seconds_per_photo*(len(missing)-done)
            timings.append({'new_images_this_run':done,'batch_seconds':duration,'batch_images':len(group),
                'seconds_per_photo_excluding_first_batch_when_possible':seconds_per_photo,
                'estimated_remaining_extraction_seconds':remaining})
            if done>=next_notice or done==len(missing):
                print(f'ViT-B : {len(ids)-len(missing)+done}/{len(ids)} photos ; estimation extraction restante : {remaining/60:.1f} min',flush=True)
                csv_write(args.output/'extraction_timing.csv',timings);next_notice=done+50
            if args.benchmark_only and done>=50:break
    csv_write(args.output/'extraction_timing.csv',timings)
    meta['new_images_extracted_this_run']=done;meta['extraction_seconds_this_run']=time.perf_counter()-started
    if done<len(missing):
        atomic_json(args.output/'benchmark.json',meta)
        print('Chronométrage terminé. Relancer sans --benchmark-only pour poursuivre les vecteurs enregistrés.',flush=True)
        return None
    x=checked_vectors(vectors,ids)
    temp=output.with_suffix('.tmp.npz')
    np.savez_compressed(temp,ids=np.array(ids),features=x.astype(np.float32),fingerprint=fp,
                        preprocess=ev.PREPROCESS,metadata=json.dumps(meta))
    temp.replace(output)
    # Read back the same float32 archive before fitting: identical to cache reuse.
    return load_features(output,cat)

def projection_experiment(manifests,ids,x,args,meta):
    results=[];curves=[];counts=[];selections={}
    settings={'epochs':args.epochs,'steps_per_epoch':10,'output_dim':128}
    folder=args.output/'projections'/'dinov2_vitb14'
    metadata={'backbone':meta,'settings':settings,'manifest_sha256':{
        p.name:ev.file_hash(p) for p in sorted(args.splits.glob('seed_*.json'))},'selection':selections,
        'notes':['Original photographs only, frozen DINOv2 ViT-B/14 without registers at 224 pixels.',
        'Same fixed identity-disjoint train70/validation20/test25 splits as ViT-S; partitions overlap across seeds.',
        'Raw features 768D; train-only uncentered PCA and supervised projection both 128D.',
        'Same batch-hard cosine/Adam settings and validation selection as previous 200-epoch experiment.',
        'No selection on test scores; open-set thresholds calibrated on validation.',
        'Test false acceptance need not equal the 5% validation target.']}
    with threadpool_limits(limits=args.threads):
        for m in manifests:
            print(f'Projection ViT-B : partition {m["seed"]}',flush=True)
            w,pca,doc=fit_projection(m,ids,x,folder,settings)
            selections[str(m['seed'])]={'selected_epoch':doc['selected_epoch'],
                'validation_score':doc['validation_score'],'last_epoch_selected':doc['last_epoch_selected'],
                'training_identities':len(doc['training_identities']),'training_images':doc['training_images'],
                'projection_sha256':ev.file_hash(folder/f'projection_seed_{m["seed"]}.npz')}
            for method,f in [('dinov2_vitb14_raw',x),('dinov2_vitb14_pca128',normalize(x@pca)),
                             ('dinov2_vitb14_triplet128',normalize(x@w))]:
                r,c,n=evaluate(m,ids,f,method,'full_original');results+=r;curves+=c;counts+=n
            save_evaluations(args.output,results,curves,counts,metadata)
    return results,curves,counts,metadata

def compare_small(args,manifests,bresults,bcurves,bcounts,bmeta):
    if args.skip_small:return
    cat=manifests[0]['catalog'];ids,x,meta=load_features(args.small_embeddings,cat)
    if x.shape[1]!=384 or meta.get('model')!='dinov2':raise ValueError('Cache ViT-S 384 attendu pour comparaison')
    results=list(bresults);curves=list(bcurves);counts=list(bcounts)
    with threadpool_limits(limits=args.threads):
        for m in manifests:
            folder=args.small_projections;w,info=load_weights(folder,m['seed'],m,384)
            if info['settings']!={'epochs':args.epochs,'steps_per_epoch':10,'output_dim':128}:
                raise ValueError('Budget ViT-S différent : comparaison à budget identique impossible. Résultats ViT-B conservés ; utiliser --skip-small au besoin.')
            with np.load(folder/f'projection_seed_{m["seed"]}.npz',allow_pickle=False) as z:pca=z['initial_pca'].copy()
            if pca.shape!=(384,128) or not np.isfinite(pca).all():raise ValueError('PCA ViT-S invalide')
            # Verify provenance using the exact existing projection signature.
            # Same settings/catalog/feature bytes must reproduce its identity.
            expected=hashlib.sha256(json.dumps({'settings':info['settings'],'manifest':m,'feature_ids':ids,
                'feature_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'eligible':None},sort_keys=True).encode()).hexdigest()
            report=json.loads((folder/f'selection_seed_{m["seed"]}.json').read_text(encoding='utf-8'))
            if report['signature']!=expected:raise ValueError('Projection ViT-S incompatible avec ces descripteurs ou partitions')
            for method,f in [('dinov2_vits14_raw',x),('dinov2_vits14_pca128',normalize(x@pca)),
                             ('dinov2_vits14_triplet128',normalize(x@w))]:
                r,c,n=evaluate(m,ids,f,method,'full_original');results+=r;curves+=c;counts+=n
    metadata=dict(bmeta);metadata['small_backbone']={'metadata':meta,'features_sha256':ev.file_hash(args.small_embeddings),
            'note':'ViT-S uses its existing 200-epoch checkpoints; no retraining of ViT-S.'}
    folder=args.output/'comparison';folder.mkdir(exist_ok=True)
    save_evaluations(folder,results,curves,counts,metadata)
    print('Comparaison ViT-S / ViT-B enregistrée.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ev.DEFAULT_BASE)
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--splits',type=Path,default=HERE/'splits')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--benchmark-only',action='store_true',help='Arrêter après environ 50 nouvelles photos, garder le cache')
    p.add_argument('--skip-small',action='store_true',help='Ne pas recalculer la comparaison ViT-S déjà apprise')
    p.add_argument('--small-embeddings',type=Path,default=HERE.parent/'background_equine'/'results'/'embeddings_dinov2.npz')
    p.add_argument('--small-projections',type=Path,default=HERE.parent/'robustness_equine'/'results'/'projections'/'dinov2')
    args=p.parse_args()
    if min(args.batch_size,args.threads,args.epochs)<1:p.error('Entiers positifs attendus')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    manifests=[json.loads((args.splits/f'seed_{s}.json').read_text(encoding='utf-8')) for s in range(42,47)]
    cat=manifests[0]['catalog']
    if any(m['catalog']!=cat for m in manifests):raise ValueError('Catalogues différents')
    if not args.skip_small and not args.benchmark_only:
        if not args.small_embeddings.is_file():raise ValueError('Cache ViT-S absent : préciser --small-embeddings ou utiliser --skip-small')
        for seed in range(42,47):
            if not (args.small_projections/f'projection_seed_{seed}.npz').is_file():raise ValueError('Checkpoints ViT-S absents : préciser --small-projections ou utiliser --skip-small')
    extracted=extract(args,cat)
    if extracted is None:return
    if args.benchmark_only:
        print('Extraction complète disponible ; relancer sans --benchmark-only pour les projections.',flush=True);return
    ids,x,meta=extracted
    r,c,n,doc=projection_experiment(manifests,ids,x,args,meta)
    compare_small(args,manifests,r,c,n,doc)
    print(f'Terminé : {args.output}',flush=True)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:sys.exit('Arrêt. Relancer la même commande : vecteurs et partitions entièrement entraînées sont conservés.')
    except Exception as exc:sys.exit(f'Erreur : {type(exc).__name__}: {exc}')
