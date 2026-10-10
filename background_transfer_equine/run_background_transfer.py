"""Perturbation du décor à checkpoints figés; aucune optimisation sur ce test."""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits
import evaluation as ev
from core import atomic_json, csv_write, load_features, normalize, evaluate, fingerprint

HERE=Path(__file__).resolve().parent
ROLES=('gallery','validation_query','test_query')

def hash_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def find_one(root, filename):
    found=sorted(root.rglob(filename),key=lambda p:(len(p.parts),str(p)))
    if not found:raise ValueError(f'Fichier absent dans {root} : {filename}')
    if len({ev.file_hash(p) for p in found})>1:
        raise ValueError(f'Plusieurs fichiers {filename} différents. Préciser un dossier plus ciblé.')
    return found[0]

def square224(image):
    side=max(image.size);canvas=Image.new('RGB',(side,side),(127,127,127))
    canvas.paste(image,((side-image.width)//2,(side-image.height)//2))
    return canvas.resize((224,224),Image.Resampling.BICUBIC)

def clean_patch(image, mask):
    """Largest square patch outside an expanded automatic mask, deterministic.

    This is an automatic background candidate, not verified background truth.
    Reject uniform patches and avoid interpolating holes filled with gray.
    """
    arr=np.asarray(image);h,w=mask.shape
    if arr.shape[:2]!=(h,w):raise ValueError('Image et masque de tailles différentes')
    margin=max(2,int(.03*min(h,w)))
    # Conservative square dilation via PIL max filter, at downscaled resolution.
    from PIL import ImageFilter
    small=Image.fromarray(mask.astype(np.uint8)*255)
    if max(h,w)>640:
        scale=640/max(h,w);small=small.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.NEAREST)
    radius=max(1,round(margin*small.width/w))
    expanded=np.asarray(small.filter(ImageFilter.MaxFilter(2*radius+1)).resize((w,h),Image.Resampling.NEAREST))>0
    for fraction in [.6,.5,.4,.3,.25,.2]:
        size=int(min(h,w)*fraction)
        if size<32 or size*size/(h*w)<.025:continue
        candidates=[]
        for y in sorted(set(np.linspace(0,h-size,7,dtype=int))):
            for x in sorted(set(np.linspace(0,w-size,7,dtype=int))):
                if expanded[y:y+size,x:x+size].any():continue
                patch=arr[y:y+size,x:x+size]
                variance=float(np.std(patch.astype(float),axis=(0,1)).mean())
                if variance>=5:candidates.append((variance,int(y),int(x)))
        if candidates:
            _,y,x=max(candidates,key=lambda v:(v[0],-v[1],-v[2]))
            return image.crop((x,y,x+size,y+size)).resize((224,224),Image.Resampling.BICUBIC),[x,y,x+size,y+size]
    return None,None

def compose(image, mask, donor):
    """Replace background on native canvas, then use the original padding policy."""
    background=ImageOps.fit(donor,image.size,method=Image.Resampling.BICUBIC)
    a=np.asarray(background).copy();original=np.asarray(image)
    if mask.shape!=original.shape[:2]:raise ValueError('Masque natif incorrect')
    a[mask]=original[mask]
    return square224(Image.fromarray(a))

def role_manifest(m):
    """Different embeddings for the SAME photo when used as ref or open unknown.

    Prevents accidentally reusing gallery-pool backgrounds for unknown queries.
    """
    out=dict(m);cat=m['catalog'];wanted=set()
    def idx(i,role):
        key=role+':'+str(i);wanted.add(key);return key
    def episode(e,qrole):
        queries=set(map(str,e['queries']))
        for refs in e['gallery'].values():
            if queries & set(map(str,refs)):
                raise ValueError('Photo source commune aux références et requêtes connues')
        return {'identities':e.get('identities',[]),
            'gallery':{k:[idx(i,'gallery') for i in g] for k,g in e['gallery'].items()},
            'queries':[idx(i,qrole) for i in e['queries']]}
    out['reid_25']=episode(m['reid_25'],'test_query')
    for name,role in [('open_validation','validation_query'),('open_test','test_query')]:
        out[name]=dict(m[name]);out[name]['known']=episode(m[name]['known'],role)
        out[name]['unknown_queries']=[idx(i,role) for i in m[name]['unknown_queries']]
    out['external_unknown_queries']=[idx(i,'test_query') for i in m['external_unknown_queries']]
    out['catalog']={key:dict(cat[key.split(':',1)[1]],source_image_index=key.split(':',1)[1],role=key.split(':',1)[0]) for key in sorted(wanted)}
    return out

def donor_pools(records, m, seed):
    # Donors from TRAINING only; donor identities disjoint across the three roles.
    train=set(m['reserved_train_identities']);by_label=defaultdict(list)
    for record in records:
        if record['identity'] in train:by_label[record['identity']].append(record)
    names=sorted(by_label)
    if len(names)<9:raise ValueError('Moins de 9 identités avec fonds admissibles; utiliser --background-dir')
    rng=np.random.default_rng(seed+17001);names=rng.permutation(names).tolist()
    splits=np.array_split(np.array(names),3);pools={}
    for role,labels in zip(ROLES,splits):pools[role]=[r for label in labels for r in by_label[str(label)]]
    return pools

def external_pools(records, seed):
    # Whole files assigned to distinct roles; deduplicate pixel-identical files.
    if len(records)<9:raise ValueError('--background-dir nécessite au moins 9 fonds distincts')
    rng=np.random.default_rng(seed+17001);order=rng.permutation(len(records))
    return {role:[records[int(i)] for i in indices] for role,indices in zip(ROLES,np.array_split(order,3))}

def assignment(keys,pools,seed,repeat):
    rng=np.random.default_rng(np.random.SeedSequence([seed,repeat,9821]))
    rows=[]
    for key in sorted(keys):
        role,source=key.split(':',1);pool=pools[role]
        donor=pool[int(rng.integers(len(pool)))]
        rows.append({'role_key':key,'source_image_index':source,'role':role,
                     'donor_key':donor['key'],'donor_identity':donor.get('identity','external')})
    return rows

def load_weights(folder,seed,m,dimension):
    path=folder/f'projection_seed_{seed}.npz';report=folder/f'selection_seed_{seed}.json'
    if not path.exists() or not report.exists():raise ValueError(f'Checkpoint ou sélection absent : {path}')
    doc=json.loads(report.read_text(encoding='utf-8'))
    with np.load(path,allow_pickle=False) as z:
        w=z['weights'].copy();train=set(z['train_identities'].tolist());signature=str(z['signature'])
    if signature!=doc['signature'] or train!=set(doc['training_identities']):raise ValueError('Checkpoint/sélection incohérents')
    if not train<=set(m['reserved_train_identities']) or train&(set(m['test_identities'])|set(m['validation_identities'])):
        raise ValueError('Identités entraînement incompatibles')
    if w.shape!=(dimension,128) or not np.isfinite(w).all():raise ValueError('Projection 384 → 128 attendue')
    return w,{'file_sha256':ev.file_hash(path),'selection_sha256':ev.file_hash(report),
              'selected_epoch':doc['selected_epoch'],'training_identities':len(train),'settings':doc['settings']}

def build_donors(args,cat,audit,cache):
    folder=args.output/'donors';folder.mkdir(parents=True,exist_ok=True)
    records=[];seen=set()
    if args.background_dir:
        paths=sorted(p for p in args.background_dir.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.webp'])
        for path in paths:
            with Image.open(path) as opened:image=ImageOps.exif_transpose(opened).convert('RGB')
            donor=ImageOps.fit(image,(224,224),method=Image.Resampling.BICUBIC)
            pixel_hash=hashlib.sha256(np.asarray(donor).tobytes()).hexdigest()
            if pixel_hash in seen:continue
            seen.add(pixel_hash);key='external_'+pixel_hash;target=folder/(key+'.png');donor.save(target)
            records.append({'key':key,'identity':'external','path':str(target),'source_sha256':ev.file_hash(path),
                            'pixels_sha256':pixel_hash,'source':str(path)})
    else:
        for position,idx in enumerate(sorted(audit,key=int)):
            if position%100==0:print(f'  recherche de fonds : {position}/{len(audit)} photos',flush=True)
            row=audit[idx]
            if row['status']!='segmented' or int(row['horse_detections'])!=1:continue
            image_path=args.base.joinpath(*cat[idx]['relative_path'].split('/'))
            with Image.open(image_path) as opened:image=ImageOps.exif_transpose(opened).convert('RGB')
            with Image.open(cache/f'{idx}_mask.png') as opened:mask=np.asarray(opened.convert('L'))>127
            donor,box=clean_patch(image,mask)
            if donor is None:continue
            pixel_hash=hashlib.sha256(np.asarray(donor).tobytes()).hexdigest()
            if pixel_hash in seen:continue
            seen.add(pixel_hash);key='patch_'+idx;target=folder/(key+'.png');donor.save(target)
            records.append({'key':key,'identity':cat[idx]['identity'],'path':str(target),
                'source_image_index':idx,'crop_xyxy':box,'pixels_sha256':pixel_hash})
    atomic_json(args.output/'donor_candidates.json',records)
    return records

def encode(args,model,key_manifest,assignments,donors,cache,signature,out):
    import torch
    keys=sorted(key_manifest['catalog']);lookup={r['role_key']:r for r in assignments}
    donor_lookup={r['key']:r for r in donors};vectors=out/'vectors';vectors.mkdir(parents=True,exist_ok=True)
    info=out/'signature.json'
    if info.exists() and json.loads(info.read_text(encoding='utf-8'))['signature']!=signature:
        raise ValueError('Cache composite incompatible : changer --output')
    atomic_json(info,{'signature':signature})
    missing=[key for key in keys if not (vectors/(key.replace(':','_')+'.npy')).exists()]
    preview_count=0
    with torch.inference_mode():
        for start in range(0,len(missing),args.batch_size):
            group=missing[start:start+args.batch_size];batch=[]
            for key in group:
                record=key_manifest['catalog'][key];idx=record['source_image_index'];donor=donor_lookup[lookup[key]['donor_key']]
                with Image.open(args.base.joinpath(*record['relative_path'].split('/'))) as opened:
                    image=ImageOps.exif_transpose(opened).convert('RGB')
                with Image.open(cache/f'{idx}_mask.png') as opened:mask=np.asarray(opened.convert('L'))>127
                with Image.open(donor['path']) as opened:composite=compose(image,mask,opened.convert('RGB'))
                if args.previews and preview_count<6:
                    previews=out/'previews';previews.mkdir(exist_ok=True)
                    panel=Image.new('RGB',(448,224));panel.paste(square224(image));panel.paste(composite,(224,0))
                    panel.save(previews/(key.replace(':','_')+'.jpg'),quality=90);preview_count+=1
                arr=np.asarray(composite,dtype=np.float32)/255
                arr=(arr-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
                batch.append(torch.from_numpy(arr.transpose(2,0,1).copy()))
            features=model(torch.stack(batch).to(args.resolved_device)).cpu().numpy().astype(np.float32)
            features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
            for key,feature in zip(group,features):
                target=vectors/(key.replace(':','_')+'.npy');tmp=target.with_suffix('.tmp.npy')
                np.save(tmp,feature);tmp.replace(target)
            print(f'  embeddings composites : {min(start+args.batch_size,len(missing))}/{len(missing)} nouveaux',flush=True)
    x=np.stack([np.load(vectors/(key.replace(':','_')+'.npy'),allow_pickle=False) for key in keys]).astype(np.float64)
    if x.shape!=(len(keys),384) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5):
        raise ValueError('Cache de descripteurs composites invalide')
    return keys,x

def write_outputs(output,results,curves,counts,metadata):
    atomic_json(output/'results_transfer.json',{'metadata':metadata,'results':results})
    csv_write(output/'operating_points_transfer.csv',curves);csv_write(output/'episode_counts_transfer.csv',counts)
    # First average decor replicates WITHIN a partition, then average partitions.
    groups=defaultdict(lambda:defaultdict(list))
    for r in results:
        for metric,value in r.items():
            if metric.endswith('_macro') or metric=='chance_top1':
                groups[(r['method'],r['condition'],r['protocol'],r['gallery_size'],metric)][r['seed']].append(value)
    rows=[]
    for (method,condition,protocol,k,metric),seeds in sorted(groups.items()):
        values=[float(np.mean(v)) for _,v in sorted(seeds.items())]
        rows.append({'method':method,'condition':condition,'protocol':protocol,'gallery_size':k,'metric':metric,
            'n_partitions':len(values),'mean':float(np.mean(values)),
            'std_across_partitions':float(np.std(values,ddof=1)) if len(values)>1 else '',
            'mean_replicates_per_partition':float(np.mean([len(v) for v in seeds.values()]))})
    csv_write(output/'summary_transfer.csv',rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=HERE.parent)
    p.add_argument('--robustness-results',type=Path,default=HERE.parent/'robustness_equine'/'results')
    p.add_argument('--base',type=Path,default=ev.DEFAULT_BASE)
    p.add_argument('--original-embeddings',type=Path)
    p.add_argument('--foreground-embeddings',type=Path)
    p.add_argument('--audit',type=Path)
    p.add_argument('--background-dir',type=Path,help='Facultatif : dossier de photos de décors sans animaux')
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--previews',action='store_true')
    args=p.parse_args()
    if min(args.repeats,args.batch_size,args.threads)<1:p.error('Paramètres entiers strictement positifs attendus')
    args.project_root=args.project_root.resolve();args.robustness_results=args.robustness_results.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    bg=args.robustness_results/'background';cache=bg/'cache'
    paths=[bg/f'matched_manifest_seed_{seed}.json' for seed in range(42,47)]
    manifests=[json.loads(path.read_text(encoding='utf-8')) for path in paths]
    cat=manifests[0]['catalog']
    if any(m['catalog']!=cat for m in manifests):raise ValueError('Catalogues différents')
    original_path=args.original_embeddings or find_one(args.project_root,'embeddings_dinov2.npz')
    foreground_path=args.foreground_embeddings or find_one(args.project_root,'embeddings_dinov2_foreground.npz')
    args.audit=args.audit or find_one(args.project_root,'segmentation_audit.csv')
    with args.audit.open(encoding='utf-8-sig',newline='') as f:audit={str(r['image_index']):r for r in csv.DictReader(f,delimiter=';')}
    if set(audit)!=set(cat):raise ValueError('Audit incomplet')
    if any(audit[i]['relative_path']!=cat[i]['relative_path'] or audit[i]['identity']!=cat[i]['identity'] for i in cat):
        raise ValueError('Identités ou chemins de l’audit incompatibles')
    ids,original,ometa=load_features(original_path,cat);fids,foreground,fmeta=load_features(foreground_path,cat)
    if ometa.get('model')!='dinov2' or fmeta.get('model')!='dinov2_foreground_with_fallback':raise ValueError('Caches DINOv2 attendus')
    if ometa['model_state_sha256']!=fmeta['model_state_sha256']:raise ValueError('Poids originaux/foreground différents')
    if original.shape[1]!=384 or foreground.shape[1]!=384:raise ValueError('Descripteurs DINOv2 384 attendus')
    expected_signature=hash_json({'catalog':fingerprint(cat),'yolo_sha':fmeta['segmentation']['weights_sha256'],
        'dinov2_sha':ometa['model_state_sha256'],'audit':audit,'method':'gray127_native_largest_eligible_horse_v1'})
    if json.loads((cache/'metadata.json').read_text(encoding='utf-8'))['signature']!=expected_signature:
        raise ValueError('Provenance des masques incompatible avec le foreground et l’audit')
    original_lookup={i:n for n,i in enumerate(ids)};foreground_lookup={i:n for n,i in enumerate(fids)}
    eligible={idx for idx,row in audit.items() if row['status']=='segmented'}
    # Verify ALL native masks against the v2 cache and original photographs.
    for idx in sorted(eligible,key=int):
        record=json.loads((cache/f'{idx}.json').read_text(encoding='utf-8'))
        if ev.file_hash(cache/f'{idx}_mask.png')!=record['mask_sha256']:raise ValueError(f'Masque modifié : {idx}')
        path=args.base.joinpath(*cat[idx]['relative_path'].split('/'))
        if not path.is_file() or ev.file_hash(path)!=cat[idx]['sha256']:raise ValueError(f'Photo absente ou modifiée : {path}')
    donors=build_donors(args,cat,audit,cache)
    print(f'{len(donors)} fonds candidats disponibles; aucune image fallback utilisée.',flush=True)
    import torch
    torch.manual_seed(42);torch.set_num_threads(args.threads)
    args.resolved_device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if args.resolved_device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA indisponible')
    if args.resolved_device=='cuda':torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=torch.hub.load('facebookresearch/dinov2:main','dinov2_vits14',trust_repo=True).eval().to(args.resolved_device)
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if h.hexdigest()!=ometa['model_state_sha256']:raise ValueError('Checkpoint DINOv2 différent')
    settings={'repeats':args.repeats,'donor_source':'external' if args.background_dir else 'training_photo_patches',
        'model_state_sha256':h.hexdigest(),'catalog':fingerprint(cat),'version':'decor_transfer_v1'}
    metadata={'settings':settings,'checkpoints':{},'inputs':{
        'original_embeddings_sha256':ev.file_hash(original_path),'foreground_embeddings_sha256':ev.file_hash(foreground_path),
        'audit_sha256':ev.file_hash(args.audit),'mask_cache_signature':expected_signature},'notes':[
        'No retraining, checkpoint selection or hyperparameter tuning in this experiment.',
        'Distinct donor identities/files for gallery, validation query and test query pools.',
        'No fixed pony-to-donor mapping; random draws for every image, role and repeat.',
        'Native masks and source photographs reused; synthetic composites are not real novel-site validation.',
        'Automatic masks may retain contextual pixels or omit animal pixels.',
        'Default donor patches avoid expanded masks and multi-horse detections; not manually verified.',
        'The five identity partitions overlap; decor replicates are not new independent animals.',
        'Open-set thresholds are calibrated separately per method/condition/repeat using validation only.']}
    results=[];curves=[];counts=[]
    with threadpool_limits(limits=args.threads):
        for m in manifests:
            seed=m['seed'];rm=role_manifest(m);keys=sorted(rm['catalog'])
            if any(key.split(':',1)[1] not in eligible for key in keys):raise ValueError('Image fallback dans les épisodes appariés')
            w_orig,info_orig=load_weights(args.robustness_results/'projections'/'dinov2',seed,m,384)
            w_fg,info_fg=load_weights(bg/'foreground_projection',seed,m,384)
            metadata['checkpoints'][str(seed)]={'original':info_orig,'foreground':info_fg}
            pools=external_pools(donors,seed) if args.background_dir else donor_pools(donors,m,seed)
            # Document exactly which donor identities/images are eligible in each role.
            atomic_json(args.output/f'donor_pools_seed_{seed}.json',{role:[r['key'] for r in rows] for role,rows in pools.items()})
            baseline_original=original[[original_lookup[key.split(':',1)[1]] for key in keys]]
            baseline_fg=foreground[[foreground_lookup[key.split(':',1)[1]] for key in keys]]
            def record_eval(condition,repeat,xids,x):
                for method,f in [('dinov2_raw',x),('projection_trained_original',normalize(x@w_orig)),
                                 ('projection_trained_foreground',normalize(x@w_fg))]:
                    r,c,n=evaluate(rm,xids,f,method,condition)
                    for rows in [r,c,n]:
                        for row in rows:row['decor_repeat']=repeat
                    results.extend(r);curves.extend(c);counts.extend(n)
            record_eval('matched_original',-1,keys,baseline_original)
            record_eval('matched_gray_foreground',-1,keys,baseline_fg)
            for repeat in range(args.repeats):
                print(f'Partition {seed}, fonds aléatoires {repeat+1}/{args.repeats}',flush=True)
                assigned=assignment(keys,pools,seed,repeat)
                csv_write(args.output/f'assignments_seed_{seed}_repeat_{repeat}.csv',assigned)
                signature=hash_json({'settings':settings,'seed':seed,'repeat':repeat,'assignments':assigned,
                    'donor_pixels':{r['key']:r['pixels_sha256'] for role in pools for r in pools[role]},
                    'manifest_sha256':ev.file_hash(bg/f'matched_manifest_seed_{seed}.json'),
                    'mask_sha256':{key.split(':',1)[1]:ev.file_hash(cache/(key.split(':',1)[1]+'_mask.png')) for key in keys}})
                out=args.output/'cache'/f'seed_{seed}'/f'repeat_{repeat}';out.mkdir(parents=True,exist_ok=True)
                xids,x=encode(args,model,rm,assigned,donors,cache,signature,out)
                record_eval('randomized_background',repeat,xids,x)
                write_outputs(args.output,results,curves,counts,metadata)
    print(f'Terminé : {args.output}',flush=True)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:sys.exit('Arrêt. Relancer la même commande pour réutiliser les vecteurs enregistrés.')
    except Exception as exc:sys.exit(f'Erreur : {type(exc).__name__}: {exc}')
