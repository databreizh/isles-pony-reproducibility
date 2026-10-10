"""Reconstruct and cache native masks; extract only successful background controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import evaluation as ev
from core import atomic_json, csv_write, fingerprint

def square224(image):
    side=max(image.size);out=Image.new('RGB',(side,side),(127,127,127))
    out.paste(image,((side-image.width)//2,(side-image.height)//2))
    return out.resize((224,224),Image.Resampling.BICUBIC)

def two_views(image, mask):
    array=np.asarray(image)
    if mask.shape!=array.shape[:2]:raise ValueError('Taille du masque incorrecte')
    fg=array.copy();fg[~mask]=127
    bg=array.copy();bg[mask]=127
    return square224(Image.fromarray(fg)),square224(Image.fromarray(bg))

def create_background(args, catalog, audit, foreground_meta, original_meta):
    import torch
    from ultralytics import YOLO
    folder=args.output/'background';folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'cache';cache.mkdir(exist_ok=True)
    if not audit or set(audit)!=set(catalog):raise ValueError('Audit incomplet ou indices incompatibles')
    for idx,row in catalog.items():
        p=args.base.joinpath(*row['relative_path'].split('/'))
        if not p.is_file() or ev.file_hash(p)!=row['sha256']:
            raise ValueError(f'Photo absente ou modifiée : {p}')
        if audit[idx]['relative_path']!=row['relative_path'] or audit[idx]['identity']!=row['identity']:
            raise ValueError(f'Audit incompatible pour {idx}')
    expected_weight=foreground_meta.get('segmentation',{}).get('weights_sha256')
    if not expected_weight:raise ValueError('Empreinte YOLO absente des métadonnées foreground')
    weights=args.yolo_weights
    if weights is None:
        matches=sorted(args.project_root.rglob('yolo11n-seg.pt'),key=lambda p:len(p.parts))
        weights=next((p for p in matches if ev.file_hash(p)==expected_weight),'yolo11n-seg.pt')
    model=YOLO(str(weights))
    if ev.file_hash(Path(model.ckpt_path))!=expected_weight:
        raise ValueError('Poids YOLO différents de la précédente segmentation')
    horse=[int(key) for key,name in model.names.items() if name=='horse']
    if len(horse)!=1:raise ValueError('Classe horse ambiguë')
    signature=hashlib.sha256(json.dumps({'catalog':fingerprint(catalog),'yolo_sha':expected_weight,
        'dinov2_sha':original_meta['model_state_sha256'],
        'audit':audit,'method':'gray127_native_largest_eligible_horse_v1'},sort_keys=True).encode()).hexdigest()
    info_path=cache/'metadata.json'
    if info_path.exists() and json.loads(info_path.read_text(encoding='utf-8'))['signature']!=signature:
        raise ValueError('Cache background incompatible : choisir un autre --output')
    atomic_json(info_path,{'signature':signature,'weights_sha256':expected_weight})
    old_ids=sorted([idx for idx,row in audit.items() if row['status']=='segmented'],key=int)
    if not old_ids:raise ValueError('Aucun masque réussi dans l’ancien audit')
    atomic_json(folder/'coverage.json',{'catalog_images':len(catalog),'eligible_images':len(old_ids),
        'fallback_excluded':len(catalog)-len(old_ids),'eligible_fraction':len(old_ids)/len(catalog),
        'multiple_horses_among_eligible':sum(str(audit[i].get('multiple_horses','')).lower()=='true' for i in old_ids)})
    for n,idx in enumerate(old_ids,1):
        record=cache/f'{idx}.json';bg_path=cache/f'{idx}_background.png';mask_path=cache/f'{idx}_mask.png'
        if record.exists() and bg_path.exists() and mask_path.exists():
            saved=json.loads(record.read_text(encoding='utf-8'))
            if ev.file_hash(bg_path)!=saved['background_sha256'] or ev.file_hash(mask_path)!=saved['mask_sha256']:
                raise ValueError(f'Cache modifié : {idx}')
        else:
            p=args.base.joinpath(*catalog[idx]['relative_path'].split('/'))
            with Image.open(p) as opened:image=ImageOps.exif_transpose(opened).convert('RGB')
            result=model.predict(source=image,classes=horse,imgsz=640,conf=.25,retina_masks=True,
                device='0' if args.resolved_device=='cuda' else 'cpu',verbose=False,save=False)[0]
            if result.masks is None:raise ValueError(f'Masque devenu indisponible pour {idx}; comparaison arrêtée')
            masks=result.masks.data.detach().cpu().numpy()>.5
            conf=result.boxes.conf.detach().cpu().numpy()
            if tuple(masks.shape[1:])!=(image.height,image.width):raise ValueError('Masques non natifs')
            area=masks.mean(axis=(1,2));choices=[i for i,a in enumerate(area) if .01<=a<=.98]
            if not choices:raise ValueError(f'Aucun masque admissible pour {idx}')
            chosen=max(choices,key=lambda i:(area[i],conf[i],-i));mask=masks[chosen]
            if not np.isclose(float(area[chosen]),float(audit[idx]['mask_area_fraction']),atol=1e-7,rtol=0):
                raise ValueError(f'Masque différent du précédent audit pour {idx}')
            foreground,background=two_views(image,mask)
            check=cache/f'{idx}_foreground_check.png';foreground.save(check)
            old_png=args.audit.parent/'segmentation_cache'/f'{idx}.png'
            if old_png.exists():
                with Image.open(old_png) as old:
                    identical=np.array_equal(np.asarray(foreground),np.asarray(old.convert('RGB')))
            else:
                identical=ev.file_hash(check)==audit[idx]['processed_sha256']
            if not identical:
                raise ValueError(f'Foreground reconstruit différent du cache précédent pour {idx}')
            check.unlink()
            background.save(bg_path);Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
            atomic_json(record,{'image_index':idx,'mask_sha256':ev.file_hash(mask_path),
                'background_sha256':ev.file_hash(bg_path),'status':'segmented',
                'mask_area_fraction':float(area[chosen]),'relative_path':catalog[idx]['relative_path']})
        if n%50==0 or n==len(old_ids):print(f'Masques et fonds seuls : {n}/{len(old_ids)}',flush=True)
    del model
    output=folder/'embeddings_dinov2_background.npz'
    if output.exists():
        with np.load(output,allow_pickle=False) as z:
            if str(z['signature'])!=signature or z['ids'].tolist()!=old_ids:
                raise ValueError('Cache de features background incompatible')
        return output,set(old_ids)
    model=torch.hub.load('facebookresearch/dinov2:main','dinov2_vits14',trust_repo=True).eval().to(args.resolved_device)
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if h.hexdigest()!=original_meta['model_state_sha256']:
        raise ValueError('Poids DINOv2 différents du cache original')
    vector_folder=cache/'vectors';vector_folder.mkdir(exist_ok=True)
    missing=[idx for idx in old_ids if not (vector_folder/f'{idx}.npy').exists()]
    with torch.inference_mode():
        for start in range(0,len(missing),args.batch_size):
            group=missing[start:start+args.batch_size];batch=[]
            for idx in group:
                with Image.open(cache/f'{idx}_background.png') as im:
                    arr=np.asarray(im.convert('RGB'),dtype=np.float32)/255
                arr=(arr-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
                batch.append(torch.from_numpy(arr.transpose(2,0,1).copy()))
            x=model(torch.stack(batch).to(args.resolved_device)).cpu().numpy().astype(np.float32)
            x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
            for idx,vector in zip(group,x):
                temp=vector_folder/f'{idx}.tmp.npy';np.save(temp,vector);temp.replace(vector_folder/f'{idx}.npy')
            print(f'DINOv2 fonds seuls : {min(start+args.batch_size,len(missing))}/{len(missing)} nouveaux vecteurs',flush=True)
    features=np.stack([np.load(vector_folder/f'{idx}.npy',allow_pickle=False) for idx in old_ids])
    meta={'model':'dinov2_background','model_state_sha256':h.hexdigest(),'segmentation_signature':signature,
          'preprocess':'gray127_inside_horse_mask_then_square224_imagenet',
          'note':'Background AND silhouette/position of the gray mask remain; not a pure context-only causal test.'}
    temp=output.with_suffix('.tmp.npz')
    np.savez_compressed(temp,ids=np.array(old_ids),features=features,fingerprint=fingerprint(catalog),
                        metadata=json.dumps(meta),signature=signature)
    temp.replace(output)
    atomic_json(folder/'coverage.json',{'catalog_images':len(catalog),'eligible_images':len(old_ids),
        'fallback_excluded':len(catalog)-len(old_ids),'eligible_fraction':len(old_ids)/len(catalog),
        'multiple_horses_among_eligible':sum(str(audit[i].get('multiple_horses','')).lower()=='true' for i in old_ids)})
    return output,set(old_ids)
