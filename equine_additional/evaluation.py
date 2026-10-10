"""Extraction des représentations et évaluation des protocoles fixés dans splits/."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = Path('.')
PREPROCESS = 'exif_rgb_square_gray127_resize224_bicubic_v1'


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load_image(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        side = max(im.size)
        square = Image.new('RGB', (side, side), (127, 127, 127))
        square.paste(im, ((side-im.width)//2, (side-im.height)//2))
        return square.resize((224, 224), Image.Resampling.BICUBIC)


def extract(args, catalog):
    ids = sorted(catalog, key=int)
    fingerprint = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    cache = args.output / f'embeddings_{args.model}.npz'
    # Vérifier les sources même en présence d'un cache.
    paths = []
    for idx in ids:
        path = args.base.joinpath(*catalog[idx]['relative_path'].split('/'))
        if not path.is_file():
            raise ValueError(f'Photo absente : {path}')
        if file_hash(path) != catalog[idx]['sha256']:
            raise ValueError(f'Photo modifiée depuis audit : {path}. Refaire audit/splits.')
        paths.append(path)
    if cache.exists() and not args.recompute:
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved['fingerprint']) != fingerprint or str(saved['preprocess']) != PREPROCESS:
                raise ValueError('Cache incompatible. Utiliser --recompute.')
            if saved['ids'].tolist() != ids:
                raise ValueError('Ordre du cache incompatible')
            print(f'Cache réutilisé : {cache}', flush=True)
            return ids, saved['features'].copy(), json.loads(str(saved['metadata']))
    started = time.perf_counter()
    metadata = {'model': args.model, 'preprocess': PREPROCESS, 'python': platform.python_version(),
                'packages': {}, 'source_catalog_sha256': fingerprint,
                'model_training_on_corpus': False}
    for package in ['numpy', 'Pillow', 'scikit-image', 'torch', 'torchvision']:
        try:
            metadata['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if args.model == 'hog':
        from skimage.feature import hog
        vectors = []
        for n, path in enumerate(paths, 1):
            arr = np.asarray(load_image(path).convert('L'), dtype=np.float32) / 255.0
            vectors.append(hog(arr, orientations=9, pixels_per_cell=(16,16),
                               cells_per_block=(2,2), block_norm='L2-Hys').astype(np.float32))
            if n % 100 == 0:
                print(f'HOG : {n}/{len(paths)}', flush=True)
        features = np.stack(vectors)
        metadata['weights'] = 'none'
        metadata['device'] = 'cpu'
    else:
        import torch
        from torchvision import transforms
        torch.manual_seed(42)
        torch.set_num_threads(args.threads)
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA indisponible. Utiliser --device cpu ou auto.')
        if args.model == 'resnet50':
            from torchvision.models import ResNet50_Weights, resnet50
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = torch.nn.Identity()
            metadata['weights'] = 'ResNet50_Weights.IMAGENET1K_V2'
        else:
            # Premier lancement : téléchargement du dépôt officiel et des poids.
            model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14', trust_repo=True)
            metadata['weights'] = 'official dinov2_vits14 pretrained'
            # Empreinte du modèle chargé pour tracer les téléchargements futurs.
        model.eval().to(device)
        model_hash = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            model_hash.update(name.encode())
            model_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        metadata['model_state_sha256'] = model_hash.hexdigest()
        metadata['device'] = device
        transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(paths), args.batch_size):
                batch = torch.stack([transform(load_image(p)) for p in paths[start:start+args.batch_size]])
                vectors.append(model(batch.to(device)).cpu().numpy().astype(np.float32))
                print(f'{args.model} : {min(start+args.batch_size,len(paths))}/{len(paths)}', flush=True)
        features = np.concatenate(vectors)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) < 0.99):
        raise ValueError('Vecteurs nuls ou non finis : extraction invalide')
    metadata['extraction_seconds'] = time.perf_counter() - started
    metadata['number_images'] = len(ids)
    metadata['dimension'] = int(features.shape[1])
    np.savez_compressed(cache, ids=np.array(ids), features=features, fingerprint=fingerprint,
                        preprocess=PREPROCESS, metadata=json.dumps(metadata))
    return ids, features, metadata


def rank(gallery, queries, features, lookup, catalog):
    if set(gallery) & set(queries):
        raise ValueError('Fuite galerie/requêtes')
    g = features[[lookup[str(x)] for x in gallery]]
    q = features[[lookup[str(x)] for x in queries]]
    identities = sorted({catalog[str(x)]['identity'] for x in gallery})
    glabels = np.array([catalog[str(x)]['identity'] for x in gallery])
    similarity = q @ g.T
    # Score d'une identité = maximum parmi ses photos; règle fixée avant test.
    scores = np.stack([similarity[:,glabels == i].max(axis=1) for i in identities], axis=1)
    order = np.argsort(-scores, axis=1, kind='stable')
    return identities, scores, order


def macro_ci(values, seed):
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0,len(values),size=(1000,len(values)))].mean(axis=1)
    return float(values.mean()), [float(x) for x in np.quantile(boot,[0.025,0.975])]


def evaluate_closed(episode, k, features, lookup, catalog, seed, predictions, protocol):
    identities, scores, order = rank(episode['gallery'][str(k)], episode['queries'], features, lookup, catalog)
    buckets = defaultdict(lambda: defaultdict(list))
    for n, idx in enumerate(episode['queries']):
        true = catalog[str(idx)]['identity']
        ranked = [identities[x] for x in order[n]]
        for top in [1,3,5]:
            buckets[true][top].append(float(true in ranked[:top]))
        predictions.append({'seed': seed,'protocol': protocol,'gallery_size': k,'image_index': idx,
                            'true_identity': true,'predicted_identity': ranked[0],
                            'score': float(scores[n,order[n,0]]),'accepted': '',
                            'is_unknown': False, 'top5': '|'.join(ranked[:5])})
    result = {'protocol': protocol,'seed':seed,'gallery_size':k,
              'identities':len(buckets),'queries':len(episode['queries'])}
    for top in [1,3,5]:
        vals = [np.mean(v[top]) for v in buckets.values()]
        result[f'top{top}_macro'], result[f'top{top}_macro_ci95'] = macro_ci(vals,seed)
        result[f'top{top}_micro'] = float(np.mean([x for v in buckets.values() for x in v[top]]))
    return result


def unknown_scores(episode, k, unknown, features, lookup, catalog):
    _, scores, _ = rank(episode['gallery'][str(k)], unknown, features, lookup, catalog)
    # Éviter l'arrondi implicite du seuil vers float32 dans le comptage
    # vectorisé; les décisions individuelles utilisent des floats Python.
    return scores.max(axis=1).astype(np.float64)


def calibrate_threshold(scores, alpha):
    # Empirique au niveau photo; alpha n'est pas une garantie populationnelle.
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    allowed = math.floor(alpha * len(ordered))
    return float(np.nextafter(float(ordered[len(ordered)-allowed-1]), np.inf))


def evaluate_open(episode, k, unknown, threshold, features, lookup, catalog, seed, predictions, protocol):
    ids, scores, order = rank(episode['gallery'][str(k)], episode['queries'], features, lookup, catalog)
    by_known, by_rejected, by_unknown = defaultdict(list),defaultdict(list),defaultdict(list)
    for n, idx in enumerate(episode['queries']):
        true = catalog[str(idx)]['identity']; pred = ids[order[n,0]]
        score = float(scores[n,order[n,0]]); accepted = score >= threshold
        by_known[true].append(float(accepted and true == pred))
        by_rejected[true].append(float(not accepted))
        predictions.append({'seed':seed,'protocol':protocol,'gallery_size':k,'image_index':idx,
                            'true_identity':true,'predicted_identity':pred,'score':score,
                            'accepted':accepted,'is_unknown':False,'top5':''})
    ids, scores, order = rank(episode['gallery'][str(k)], unknown, features, lookup, catalog)
    for n, idx in enumerate(unknown):
        true = catalog[str(idx)]['identity']; score = float(scores[n,order[n,0]])
        accepted = score >= threshold
        by_unknown[true].append(float(accepted))
        predictions.append({'seed':seed,'protocol':protocol,'gallery_size':k,'image_index':idx,
                            'true_identity':true,'predicted_identity':ids[order[n,0]],'score':score,
                            'accepted':accepted,'is_unknown':True,'top5':''})
    result = {'protocol':protocol,'seed':seed,'gallery_size':k,'threshold':threshold,
              'known_identities':len(by_known),'unknown_identities':len(by_unknown),
              'known_queries':len(episode['queries']),'unknown_queries':len(unknown)}
    for name, bucket in [('correct_identification_and_acceptance',by_known),
                         ('known_rejection',by_rejected),('unknown_false_acceptance',by_unknown)]:
        result[name+'_macro'], result[name+'_macro_ci95'] = macro_ci([np.mean(v) for v in bucket.values()],seed)
        result[name+'_micro'] = float(np.mean([x for v in bucket.values() for x in v]))
    return result

