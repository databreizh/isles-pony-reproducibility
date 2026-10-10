"""Ablation exploratoire du fond : YOLO11n-seg automatique + DINOv2-S figé."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import evaluation

HERE = Path(__file__).resolve().parent
SETTINGS = {'segmenter': 'yolo11n-seg.pt', 'imgsz': 640, 'confidence': .25,
            'minimum_area_fraction': .01, 'maximum_area_fraction': .98,
            'selection': 'largest horse mask; confidence then index for ties',
            'background_rgb': [127, 127, 127], 'crop': False,
            'fallback': 'original image, explicitly recorded',
            'dinov2': 'dinov2_vits14', 'preprocess': evaluation.PREPROCESS}


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def square224(image):
    side = max(image.size)
    canvas = Image.new('RGB', (side, side), (127, 127, 127))
    canvas.paste(image, ((side-image.width)//2, (side-image.height)//2))
    return canvas.resize((224, 224), Image.Resampling.BICUBIC)


def select_mask(masks, confidences):
    """Returns mask/index/area/status. Geometric checks do not certify correctness."""
    if len(masks) == 0:
        return None, -1, 0., 'fallback_no_horse'
    areas = np.array([np.mean(mask) for mask in masks])
    eligible = [i for i, area in enumerate(areas) if .01 <= area <= .98]
    if not eligible:
        return None, -1, float(areas.max()), 'fallback_area_rejected'
    index = max(eligible, key=lambda i: (areas[i], confidences[i], -i))
    return masks[index], index, float(areas[index]), 'segmented'


def remove_background(image, mask):
    if mask is None:
        return image.copy()
    array = np.asarray(image).copy()
    if mask.shape != array.shape[:2]:
        raise ValueError('Masque et image de tailles différentes')
    array[~mask] = 127
    return Image.fromarray(array)


def segment(args, catalog, device):
    from ultralytics import YOLO
    model = YOLO(SETTINGS['segmenter'])
    horse = [int(key) for key, name in model.names.items() if name == 'horse']
    if len(horse) != 1:
        raise ValueError('Le modèle ne contient pas une classe horse unique')
    weight_path = Path(model.ckpt_path)
    info = {'settings': SETTINGS, 'weights_sha256': evaluation.file_hash(weight_path),
            'device': device, 'python': platform.python_version(), 'packages': {}}
    for package in ['ultralytics', 'torch', 'torchvision', 'numpy', 'Pillow']:
        info['packages'][package] = importlib.metadata.version(package)
    info['source_catalog_sha256'] = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    folder = args.output/'segmentation_cache'
    folder.mkdir(parents=True, exist_ok=True)
    signature = folder/'metadata.json'
    if signature.exists() and json.loads(signature.read_text(encoding='utf-8')) != info:
        raise ValueError('Cache de segmentation incompatible : utiliser un autre --output.')
    atomic_json(signature, info)
    rows = []
    for number, idx in enumerate(sorted(catalog, key=int), 1):
        image_path, record_path = folder/f'{idx}.png', folder/f'{idx}.json'
        if image_path.exists() and record_path.exists():
            row = json.loads(record_path.read_text(encoding='utf-8'))
            if evaluation.file_hash(image_path) != row['processed_sha256']:
                raise ValueError(f'Image de cache modifiée : {image_path}')
        else:
            path = args.base.joinpath(*catalog[idx]['relative_path'].split('/'))
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert('RGB')
            # RGB PIL is handled by Ultralytics; native masks align to original dimensions.
            result = model.predict(source=image, classes=horse, imgsz=640, conf=.25,
                                   retina_masks=True, device='0' if device=='cuda' else 'cpu',
                                   verbose=False, save=False)[0]
            if result.masks is None:
                masks, conf = [], []
            else:
                masks = result.masks.data.detach().cpu().numpy() > .5
                conf = result.boxes.conf.detach().cpu().numpy().tolist()
                if tuple(masks.shape[1:]) != (image.height, image.width):
                    raise ValueError('retina_masks ne renvoie pas la taille originale attendue')
            mask, index, fraction, status = select_mask(masks, conf)
            output = square224(remove_background(image, mask))
            temp = image_path.with_suffix('.tmp')
            output.save(temp, format='PNG')
            temp.replace(image_path)
            row = {'image_index': idx, 'identity': catalog[idx]['identity'],
                   'relative_path': catalog[idx]['relative_path'], 'status': status,
                   'horse_detections': len(masks), 'multiple_horses': len(masks)>1,
                   'mask_area_fraction': fraction, 'selected_confidence': float(conf[index]) if index>=0 else 0.,
                   'processed_sha256': evaluation.file_hash(image_path)}
            atomic_json(record_path, row)
        rows.append(row)
        if number%50==0 or number==len(catalog):
            print(f'Segmentation : {number}/{len(catalog)}', flush=True)
    with (args.output/'segmentation_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter=';');writer.writeheader();writer.writerows(rows)
    summary = {'images': len(rows), 'segmented': sum(r['status']=='segmented' for r in rows),
               'fallback': sum(r['status']!='segmented' for r in rows),
               'multiple_horses': sum(r['multiple_horses'] for r in rows),
               'per_identity': {label: {'images':sum(r['identity']==label for r in rows),
                                        'segmented':sum(r['identity']==label and r['status']=='segmented' for r in rows)}
                                for label in sorted({r['identity'] for r in rows})}}
    atomic_json(args.output/'segmentation_summary.json',summary)
    print(f"Masques retenus : {summary['segmented']}/{len(rows)} ; originaux conservés : {summary['fallback']}",flush=True)
    # Reproducible diagnostic sample; no manual edits or selection by identification scores.
    rng=np.random.default_rng(42)
    selected=set()
    for group,count in [([r for r in rows if r['status']=='segmented'],24),
                        ([r for r in rows if r['status']!='segmented'],12),
                        ([r for r in rows if r['multiple_horses']],12)]:
        if group:
            selected.update(group[int(i)]['image_index'] for i in rng.choice(len(group),min(count,len(group)),replace=False))
    previews=args.output/'apercus';previews.mkdir(exist_ok=True)
    for idx in sorted(selected,key=int):
        row=next(r for r in rows if r['image_index']==idx)
        original=evaluation.load_image(args.base.joinpath(*row['relative_path'].split('/')))
        with Image.open(folder/f'{idx}.png') as processed:
            panel=Image.new('RGB',(448,252),'white');panel.paste(original,(0,28));panel.paste(processed,(224,28))
        ImageDraw.Draw(panel).text((4,5),f"{idx} | {row['status']} | chevaux={row['horse_detections']}",fill='black')
        panel.save(previews/f'{idx}.jpg',quality=95)
    return info,summary,folder


def foreground_embeddings(args, catalog, ids, original_meta, folder, segmentation_info, device):
    path=args.output/'embeddings_dinov2_foreground.npz'
    signature=hashlib.sha256(json.dumps({'segmentation':segmentation_info,'original':original_meta},sort_keys=True).encode()).hexdigest()
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if str(z['signature'])!=signature or z['ids'].tolist()!=ids:
                raise ValueError('Cache de vecteurs segmentés incompatible')
            return z['features'].copy(),json.loads(str(z['metadata']))
    import torch
    from torchvision import transforms
    model=torch.hub.load('facebookresearch/dinov2:main','dinov2_vits14',trust_repo=True).eval().to(device)
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if h.hexdigest()!=original_meta['model_state_sha256']:
        raise ValueError('Poids DINOv2 différents de la baseline : comparaison arrêtée')
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(ids),args.batch_size):
            tensors=[]
            for idx in ids[start:start+args.batch_size]:
                with Image.open(folder/f'{idx}.png') as im:tensors.append(transform(im.convert('RGB')))
            vectors.append(model(torch.stack(tensors).to(device)).cpu().numpy().astype(np.float32))
            print(f'DINOv2 fond traité : {min(start+args.batch_size,len(ids))}/{len(ids)}',flush=True)
    features=np.concatenate(vectors);features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    metadata={'model':'dinov2_foreground_with_fallback','model_state_sha256':h.hexdigest(),
              'model_training_on_corpus':False,'segmentation':segmentation_info,
              'preprocess':'horse_mask_gray127_same_canvas_then_'+evaluation.PREPROCESS,
              'number_images':len(ids),'dimension':int(features.shape[1])}
    np.savez_compressed(path,ids=np.array(ids),features=features,signature=signature,
                        fingerprint=segmentation_info['source_catalog_sha256'],metadata=json.dumps(metadata))
    return features,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=evaluation.DEFAULT_BASE)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--original-embeddings',type=Path,help='Ancien embeddings_dinov2.npz facultatif')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    if args.threads<1 or args.batch_size<1:parser.error('threads et batch-size doivent être positifs')
    import torch
    torch.manual_seed(42);torch.set_num_threads(args.threads)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA indisponible')
    if device=='cuda':torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    args.output.mkdir(parents=True,exist_ok=True)
    manifests=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((HERE/'splits').glob('seed_*.json'))]
    if len(manifests)!=5:raise ValueError('Cinq partitions attendues')
    catalog=manifests[0]['catalog'];ids=sorted(catalog,key=int)
    fingerprint=hashlib.sha256(json.dumps(catalog,sort_keys=True).encode()).hexdigest()
    for row in catalog.values():
        p=args.base.joinpath(*row['relative_path'].split('/'))
        if not p.is_file() or evaluation.file_hash(p)!=row['sha256']:raise ValueError(f'Photo absente ou modifiée : {p}')
    baseline=args.original_embeddings or HERE/'embeddings_dinov2.npz'
    if baseline.exists():
        with np.load(baseline,allow_pickle=False) as z:
            if str(z['fingerprint'])!=fingerprint or z['ids'].tolist()!=ids or str(z['preprocess'])!=evaluation.PREPROCESS:
                raise ValueError('Vecteurs originaux incompatibles')
            original=z['features'].copy();original_meta=json.loads(str(z['metadata']))
        if original_meta.get('model')!='dinov2':raise ValueError('Vecteurs DINOv2 attendus')
    elif args.original_embeddings:
        raise ValueError(f'Fichier de vecteurs absent : {baseline}')
    else:
        base_args=SimpleNamespace(**vars(args),model='dinov2',recompute=False)
        _,original,original_meta=evaluation.extract(base_args,catalog)
    info,summary,folder=segment(args,catalog,device)
    masked,masked_meta=foreground_embeddings(args,catalog,ids,original_meta,folder,info,device)
    for f in [original,masked]:
        if f.shape!=(len(ids),384) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5):
            raise ValueError('Vecteurs invalides')
    original_extract=evaluation.extract
    for name,f,meta in [('dinov2_original',original,original_meta),('dinov2_foreground',masked,masked_meta)]:
        evaluation.extract=lambda unused_args,unused_catalog,f=f,meta=meta:(ids,f,dict(meta))
        evaluation.run(SimpleNamespace(output=args.output,model=name,validation_false_acceptance=.05))
    evaluation.extract=original_extract
    results=[]
    for name in ['dinov2_original','dinov2_foreground']:
        data=json.loads((args.output/f'results_{name}.json').read_text(encoding='utf-8'))
        for row in data['results']:results.append({'method':name,**row})
    atomic_json(args.output/'results_background.json',{'metadata':{'segmentation':info,'coverage':summary,
                'original':original_meta,'foreground':masked_meta,'notes':[
                'Automated masks are not verified ground truth; original retained on detection/mask failure.',
                'All photos and fixed episodes retained; report masking coverage alongside scores.',
                'No crop, retraining, threshold selection on test or manual mask corrections.',
                'Exploratory background suppression robustness test; does not prove causal background leakage or session generalization.']},'results':results})
    summary_rows=[]
    for name in ['dinov2_original','dinov2_foreground']:
        with (args.output/f'summary_{name}.csv').open(encoding='utf-8-sig') as f:summary_rows.extend(csv.DictReader(f,delimiter=';'))
    with (args.output/'summary_background.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary_rows[0]),delimiter=';');writer.writeheader();writer.writerows(summary_rows)
    print(f'Terminé : {args.output.resolve()}',flush=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit('Arrêt. Relancer la même commande pour reprendre le cache de segmentation.')
    except Exception as exc:raise SystemExit(f'Erreur : {type(exc).__name__}: {exc}') from exc
