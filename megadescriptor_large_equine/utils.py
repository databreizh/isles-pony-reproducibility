import json, hashlib
import numpy as np
from PIL import Image
import evaluation as ev

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
