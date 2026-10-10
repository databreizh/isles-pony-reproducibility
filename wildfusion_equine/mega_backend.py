"""MegaDescriptor-S-224 figé; mêmes images et protocoles que HOG/ResNet/DINO."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import evaluation

HERE=Path(__file__).resolve().parent
MODEL_NAME='hf-hub:BVRA/MegaDescriptor-S-224'


def extract_mega(args,catalog):
    ids=sorted(catalog,key=int)
    fingerprint=hashlib.sha256(json.dumps(catalog,sort_keys=True).encode()).hexdigest()
    cache=args.output/'embeddings_megadescriptor.npz'
    paths=[]
    print('Vérification des 1 546 photos...',flush=True)
    for idx in ids:
        path=args.base.joinpath(*catalog[idx]['relative_path'].split('/'))
        if not path.is_file():raise ValueError(f'Photo absente : {path}')
        if evaluation.file_hash(path)!=catalog[idx]['sha256']:
            raise ValueError(f'Photo modifiée depuis audit : {path}')
        paths.append(path)
    if cache.exists() and not args.recompute:
        with np.load(cache,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if str(z['fingerprint'])!=fingerprint or z['ids'].tolist()!=ids:
                raise ValueError('Cache incompatible. Utiliser --recompute.')
            if str(z['preprocess'])!=evaluation.PREPROCESS or meta.get('hub_model')!=MODEL_NAME:
                raise ValueError('Modèle/prétraitement du cache incompatible.')
            print('Vecteurs MegaDescriptor réutilisés.',flush=True)
            return ids,z['features'].copy(),meta
    try:
        import torch
        import timm
        from torchvision import transforms
    except ImportError as exc:
        raise ValueError('Installer les dépendances : python -m pip install -r requirements_megadescriptor.txt') from exc
    started=time.perf_counter();torch.manual_seed(42);torch.set_num_threads(args.threads)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA indisponible; utiliser --device cpu')
    print('Chargement du modèle officiel MegaDescriptor-S-224 (premier lancement : téléchargement).',flush=True)
    model=timm.create_model(MODEL_NAME,pretrained=True,num_classes=0).eval().to(device)
    transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))])
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    metadata={'model':'megadescriptor','hub_model':MODEL_NAME,'weights':'official pretrained MegaDescriptor-S-224',
              'model_state_sha256':digest.hexdigest(),'model_training_on_corpus':False,
              'preprocess':evaluation.PREPROCESS,'device':device,'python':platform.python_version(),
              'source_catalog_sha256':fingerprint,'packages':{},
              'protocol_note':'Same square-padded full-image input as baselines; adapted preprocessing, no manual crops.'}
    for package in ['numpy','Pillow','torch','torchvision','timm','huggingface-hub','safetensors']:
        try:metadata['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(paths),args.batch_size):
            batch=torch.stack([transform(evaluation.load_image(p)) for p in paths[start:start+args.batch_size]])
            result=model(batch.to(device))
            if result.ndim!=2 or result.shape[0]!=batch.shape[0]:
                raise ValueError(f'Forme de sortie inattendue : {tuple(result.shape)}')
            vectors.append(result.detach().cpu().numpy().astype(np.float32))
            print(f'MegaDescriptor : {min(start+args.batch_size,len(paths))}/{len(paths)}',flush=True)
    features=np.concatenate(vectors)
    features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5):
        raise ValueError('Vecteurs invalides ou nuls')
    metadata.update(extraction_seconds=time.perf_counter()-started,number_images=len(ids),dimension=int(features.shape[1]))
    np.savez_compressed(cache,ids=np.array(ids),features=features,fingerprint=fingerprint,
                        preprocess=evaluation.PREPROCESS,metadata=json.dumps(metadata))
    print('Extraction terminée; évaluation des cinq répétitions...',flush=True)
    return ids,features,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=evaluation.DEFAULT_BASE)
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--recompute',action='store_true')
    args=p.parse_args()
    if args.batch_size<1 or args.threads<1:p.error('batch-size et threads doivent être positifs')
    args.model='megadescriptor';args.validation_false_acceptance=.05
    evaluation.extract=extract_mega
    try:evaluation.run(args)
    except Exception as exc:
        raise SystemExit(f'Erreur : {type(exc).__name__}: {exc}') from exc


if __name__=='__main__':main()
