"""MiewID-msv3 gelé, corpus audité et cinq partitions existantes."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import evaluation

HERE = Path(__file__).resolve().parent
MODEL_NAME = 'conservationxlabs/miewid-msv3'
# Révision publiée, fixée avant l'expérience. HF résout ce préfixe en SHA complet.
REVISION = '4f1d7f2'
PREPROCESS = 'exif_rgb_resize440x440_bilinear_imagenet_l2_v1'


def validate_features(features, count):
    if features.ndim != 2 or features.shape != (count, 2152):
        raise ValueError(f'Vecteurs MiewID attendus : ({count}, 2152), reçu {features.shape}')
    if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5):
        raise ValueError('Vecteurs invalides, non normalisés ou nuls')


def extract_miewid(args, catalog):
    ids = sorted(catalog, key=int)
    fingerprint = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    cache = args.output / 'embeddings_miewid.npz'
    paths = []
    print(f'Vérification des {len(ids)} photos...', flush=True)
    for idx in ids:
        path = args.base.joinpath(*catalog[idx]['relative_path'].split('/'))
        if not path.is_file():
            raise ValueError(f'Photo absente : {path}')
        if evaluation.file_hash(path) != catalog[idx]['sha256']:
            raise ValueError(f'Photo modifiée depuis audit : {path}')
        paths.append(path)
    if cache.exists() and not args.recompute:
        with np.load(cache, allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata']))
            if (str(z['fingerprint']) != fingerprint or z['ids'].tolist() != ids
                    or str(z['preprocess']) != PREPROCESS
                    or metadata.get('hub_model') != MODEL_NAME
                    or metadata.get('requested_revision') != REVISION):
                raise ValueError('Cache incompatible. Utiliser --recompute.')
            features = z['features'].copy()
        validate_features(features, len(ids))
        print('Vecteurs MiewID réutilisés.', flush=True)
        return ids, features, metadata
    try:
        import torch
        from PIL import Image, ImageOps
        from torchvision import transforms
        from transformers import AutoModel
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ValueError('Installer : python -m pip install -r requirements_miewid.txt') from exc

    started = time.perf_counter()
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA indisponible : utiliser --device cpu')
    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    resolved_revision = HfApi().model_info(MODEL_NAME, revision=REVISION).sha
    print(f'Chargement MiewID-msv3 ({resolved_revision}), appareil : {device}', flush=True)
    # API officielle : le modèle expose directement les embeddings, pas les logits.
    model = AutoModel.from_pretrained(MODEL_NAME, revision=resolved_revision,
                                      code_revision=resolved_revision,
                                      trust_remote_code=True).eval().to(device)
    transform = transforms.Compose([
        transforms.Resize((440, 440), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
    ])

    def input_tensor(path):
        with Image.open(path) as image:
            return transform(ImageOps.exif_transpose(image).convert('RGB'))

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    metadata = {
        'model': 'miewid', 'hub_model': MODEL_NAME, 'requested_revision': REVISION,
        'resolved_revision': resolved_revision, 'model_state_sha256': digest.hexdigest(),
        'model_training_on_corpus': False, 'preprocess': PREPROCESS,
        'device': device, 'python': platform.python_version(), 'packages': {},
        'source_catalog_sha256': fingerprint,
        'protocol_note': 'Same audited images and partitions; model-card resize 440x440, no crops or square padding. Resolution differs from 224px baselines.',
        'pretraining_note': 'Official model card lists equine data (THoDBRL2015); held-out corpus identities do not imply species-unseen pretraining.',
    }
    for package in ['numpy', 'Pillow', 'torch', 'torchvision', 'timm', 'transformers', 'huggingface-hub', 'safetensors']:
        try:
            metadata['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            batch = torch.stack([input_tensor(p) for p in paths[start:start + args.batch_size]])
            result = model(batch.to(device))
            if not torch.is_tensor(result) or result.ndim != 2 or result.shape != (len(batch), 2152):
                raise ValueError(f'Sortie inattendue : {getattr(result, "shape", type(result))}')
            vectors.append(result.detach().cpu().numpy().astype(np.float32))
            print(f'MiewID : {min(start + args.batch_size, len(paths))}/{len(paths)}', flush=True)
    features = np.concatenate(vectors)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    validate_features(features, len(ids))
    metadata.update(extraction_seconds=time.perf_counter() - started,
                    number_images=len(ids), dimension=int(features.shape[1]))
    np.savez_compressed(cache, ids=np.array(ids), features=features, fingerprint=fingerprint,
                        preprocess=PREPROCESS, metadata=json.dumps(metadata))
    print('Extraction terminée ; évaluation des cinq partitions...', flush=True)
    return ids, features, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=evaluation.DEFAULT_BASE)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size et threads doivent être positifs')
    args.model = 'miewid'
    args.validation_false_acceptance = .05
    evaluation.extract = extract_miewid
    try:
        evaluation.run(args)
    except Exception as exc:
        raise SystemExit(f'Erreur : {type(exc).__name__}: {exc}') from exc


if __name__ == '__main__':
    main()
