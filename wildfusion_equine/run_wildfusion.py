"""Variante WildFusion : MegaDescriptor-S-224 + SuperPoint/LightGlue, calibration train-only."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.isotonic import IsotonicRegression
import evaluation
from mega_backend import extract_mega

HERE = Path(__file__).resolve().parent
SETTINGS = {
    'variant': 'MegaDescriptor-S-224 + SuperPoint/LightGlue, two streams, exhaustive galleries',
    'local_resize': [512, 512], 'max_keypoints': 1024,
    'match_filter_threshold': .1, 'depth_confidence': .95, 'width_confidence': .99,
    'local_score': 'number of accepted LightGlue matches; canonical lower-ID-first direction',
    'positive_calibration_pairs': 1000, 'negative_calibration_pairs': 1000,
    'calibration': 'sklearn isotonic regression, out_of_bounds=clip; balanced training pairs',
    'fusion': 'mean of calibrated image-pair scores, then max over identity references',
    'validation_photo_false_acceptance': .05,
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


def calibration_pairs(manifest):
    """Unique pairs, disjoint identities from validation/test; balanced case-control sample."""
    catalog = manifest['catalog']
    train = sorted(manifest['reserved_train_identities'])
    groups = {label: sorted([int(i) for i, row in catalog.items() if row['identity'] == label])
              for label in train}
    rng = np.random.default_rng(manifest['seed'])
    positives = set()
    while len(positives) < SETTINGS['positive_calibration_pairs']:
        label = train[int(rng.integers(len(train)))]
        a, b = sorted(rng.choice(groups[label], 2, replace=False).tolist())
        positives.add((a, b))
    negatives = set()
    while len(negatives) < SETTINGS['negative_calibration_pairs']:
        labels = rng.choice(len(train), 2, replace=False)
        a = int(rng.choice(groups[train[labels[0]]]))
        b = int(rng.choice(groups[train[labels[1]]]))
        negatives.add(tuple(sorted((a, b))))
    return [(a, b, 1) for a, b in sorted(positives)] + [(a, b, 0) for a, b in sorted(negatives)]


def planned_pairs(manifests):
    pairs = set()
    for m in manifests:
        pairs.update((a, b) for a, b, _ in calibration_pairs(m))
        for episode, unknown in [
                (m['reid_25'], []),
                (m['open_validation']['known'], m['open_validation']['unknown_queries']),
                (m['open_test']['known'], m['open_test']['unknown_queries'] + m['external_unknown_queries'])]:
            gallery = episode['gallery']['5']
            for q in episode['queries'] + unknown:
                for g in gallery:
                    assert int(q) != int(g)
                    pairs.add(tuple(sorted((int(q), int(g)))))
    return pairs


class LocalMatches:
    """On-disk keypoint cache and SQLite pair cache, committed after each pair."""
    def __init__(self, args, catalog):
        import torch
        import lightglue
        from lightglue import LightGlue, SuperPoint
        self.torch = torch
        self.args, self.catalog = args, catalog
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA indisponible ; utiliser --device cpu')
        torch.manual_seed(42)
        torch.set_num_threads(args.threads)
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.extractor = SuperPoint(max_num_keypoints=SETTINGS['max_keypoints']).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint', filter_threshold=.1,
                                 depth_confidence=.95, width_confidence=.99).eval().to(self.device)
        packages = {}
        for name in ['torch', 'torchvision', 'lightglue', 'numpy', 'Pillow', 'scikit-learn', 'timm']:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = 'unknown'
        weights = {}
        for name, model in [('superpoint', self.extractor), ('lightglue', self.matcher)]:
            digest = hashlib.sha256()
            for key, tensor in sorted(model.state_dict().items()):
                digest.update(key.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            weights[name] = digest.hexdigest()
        self.metadata = {'settings': SETTINGS, 'source_catalog_sha256': fingerprint(catalog),
                         'weights_sha256': weights, 'packages': packages,
                         'python': platform.python_version(), 'device': self.device}
        source_root = Path(lightglue.__file__).resolve().parent
        self.metadata['lightglue_source_sha256'] = {
            str(p.relative_to(source_root)): evaluation.file_hash(p)
            for p in sorted(source_root.rglob('*.py'))}
        # CPU/GPU and package changes cannot silently mix caches.
        signature = fingerprint(self.metadata)
        self.folder = args.output / 'local_cache'
        self.folder.mkdir(parents=True, exist_ok=True)
        info = self.folder / 'metadata.json'
        if info.exists():
            if json.loads(info.read_text(encoding='utf-8'))['signature'] != signature:
                raise ValueError('Cache local incompatible : choisir un nouveau dossier --output.')
        else:
            write_json(info, {'signature': signature, **self.metadata})
        self.db = sqlite3.connect(self.folder / 'pairs.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS pairs (a INTEGER, b INTEGER, matches INTEGER, PRIMARY KEY(a,b))')
        self.db.commit()

    @lru_cache(maxsize=64)
    def features(self, idx):
        from PIL import Image, ImageOps
        path = self.folder / f'image_{idx}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                return {k: self.torch.from_numpy(data[k].copy()) for k in data.files}
        original = self.args.base.joinpath(*self.catalog[str(idx)]['relative_path'].split('/'))
        with Image.open(original) as image:
            image = ImageOps.exif_transpose(image).convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy() / 255
        tensor = self.torch.from_numpy(array).permute(2, 0, 1).to(self.device)
        with self.torch.inference_mode():
            features = self.extractor.extract(tensor, resize=None)
        features = {k: v.detach().cpu() for k, v in features.items()}
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **{k: v.numpy() for k, v in features.items()})
        temp.replace(path)
        return features

    def score(self, a, b):
        from lightglue.utils import rbd
        a, b = sorted((int(a), int(b)))
        cached = self.db.execute('SELECT matches FROM pairs WHERE a=? AND b=?', (a, b)).fetchone()
        if cached is not None:
            return cached[0]
        x, y = self.features(a), self.features(b)
        if x['keypoints'].shape[1] == 0 or y['keypoints'].shape[1] == 0:
            count = 0
        else:
            with self.torch.inference_mode():
                output = self.matcher({'image0': {k: v.to(self.device) for k, v in x.items()},
                                       'image1': {k: v.to(self.device) for k, v in y.items()}})
                count = int(rbd(output)['matches'].shape[0])
        self.db.execute('INSERT INTO pairs VALUES (?,?,?)', (a, b, count))
        self.db.commit()
        return count

    def precompute(self, pairs):
        done = set(self.db.execute('SELECT a,b FROM pairs').fetchall())
        remaining = sorted(pairs - done)
        print(f'{len(pairs):,} paires nécessaires ; {len(remaining):,} restant à calculer. Appareil : {self.device}', flush=True)
        started = time.perf_counter()
        for n, (a, b) in enumerate(remaining, 1):
            self.score(a, b)
            if n % 100 == 0 or n == len(remaining):
                seconds = time.perf_counter() - started
                eta = seconds / n * (len(remaining) - n)
                print(f'LightGlue : {n}/{len(remaining)} nouvelles paires ; estimation restante {eta / 60:.1f} min', flush=True)


def make_ranker(global_scores, local_scores, lookup, calibration_global, calibration_local, method):
    def rank(gallery, queries, features, unused_lookup, catalog):
        if set(gallery) & set(queries):
            raise ValueError('Fuite galerie/requêtes')
        gi = [lookup[str(i)] for i in gallery]
        qi = [lookup[str(i)] for i in queries]
        raw = global_scores[np.ix_(qi, gi)].astype(np.float64)
        if method == 'megadescriptor_raw':
            pair_scores = raw
        else:
            pg = calibration_global.predict(raw.ravel()).reshape(raw.shape)
            if method == 'global_calibrated':
                pair_scores = pg
            else:
                local = local_scores[np.ix_(qi, gi)]
                if np.any(local < 0):
                    raise ValueError('Paires locales manquantes')
                pl = calibration_local.predict(local.ravel()).reshape(local.shape)
                pair_scores = pl if method == 'local_calibrated' else (pg + pl) / 2
        identities = sorted({catalog[str(i)]['identity'] for i in gallery})
        labels = np.array([catalog[str(i)]['identity'] for i in gallery])
        scores = np.stack([pair_scores[:, labels == label].max(axis=1) for label in identities], axis=1)
        order = np.argsort(-scores, axis=1, kind='stable')
        return identities, scores, order
    return rank


def evaluate(manifests, ids, features, local_scores, output):
    catalog = manifests[0]['catalog']
    lookup = {i: n for n, i in enumerate(ids)}
    global_scores = features @ features.T
    results, predictions, calibration_records = [], [], []
    for manifest in manifests:
        seed = manifest['seed']
        pairs = calibration_pairs(manifest)
        a = [lookup[str(x)] for x, _, _ in pairs]
        b = [lookup[str(y)] for _, y, _ in pairs]
        labels = np.array([label for _, _, label in pairs])
        local = local_scores[a, b]
        if np.any(local < 0):
            raise ValueError('Calibration locale incomplète')
        cg = IsotonicRegression(out_of_bounds='clip').fit(global_scores[a, b].astype(np.float64), labels)
        cl = IsotonicRegression(out_of_bounds='clip').fit(local, labels)
        calibration_records.append({'seed': seed, 'pairs': pairs,
                                    'global_x': cg.X_thresholds_.tolist(), 'global_y': cg.y_thresholds_.tolist(),
                                    'local_x': cl.X_thresholds_.tolist(), 'local_y': cl.y_thresholds_.tolist()})
        for method in ['megadescriptor_raw', 'global_calibrated', 'local_calibrated', 'wildfusion_two_streams']:
            evaluation.rank = make_ranker(global_scores, local_scores, lookup, cg, cl, method)
            before = len(predictions)
            for k in [1, 3, 5]:
                row = evaluation.evaluate_closed(manifest['reid_25'], k, features, lookup, catalog,
                                                  seed, predictions, 'reid_25')
                row['method'] = method
                results.append(row)
                val = manifest['open_validation']
                scores = evaluation.unknown_scores(val['known'], k, val['unknown_queries'], features, lookup, catalog)
                threshold = evaluation.calibrate_threshold(scores, .05)
                test = manifest['open_test']
                for protocol, unknown in [('open_test', test['unknown_queries']),
                                          ('external_7', manifest['external_unknown_queries'])]:
                    row = evaluation.evaluate_open(test['known'], k, unknown, threshold, features, lookup,
                                                   catalog, seed, predictions, protocol)
                    row.update(method=method, validation_target_photo_false_acceptance=.05,
                               validation_observed_photo_false_acceptance=float(np.mean(scores >= threshold)))
                    results.append(row)
            for row in predictions[before:]:
                row['method'] = method
        print(f'Partition {seed} évaluée.', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'calibrations_wildfusion.json', calibration_records)
    write_json(output / 'scores_wildfusion.json', {'results': results})
    write_csv(output / 'predictions_wildfusion.csv', predictions)
    summary = []
    for method in sorted({r['method'] for r in results}):
        for protocol in ['reid_25', 'open_test', 'external_7']:
            for k in [1, 3, 5]:
                subset = [r for r in results if (r['method'], r['protocol'], r['gallery_size']) == (method, protocol, k)]
                metrics = ['top1_macro', 'top3_macro', 'top5_macro'] if protocol == 'reid_25' else [
                    'correct_identification_and_acceptance_macro', 'known_rejection_macro', 'unknown_false_acceptance_macro']
                for metric in metrics:
                    values = [r[metric] for r in subset]
                    summary.append({'method': method, 'protocol': protocol, 'gallery_size': k, 'metric': metric,
                                    'mean': float(np.mean(values)), 'std_across_seeds': float(np.std(values, ddof=1))})
    write_csv(output / 'summary_wildfusion.csv', summary)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=evaluation.DEFAULT_BASE)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--mega-cache', type=Path, help='Ancien embeddings_megadescriptor.npz, facultatif')
    parser.add_argument('--plan', action='store_true', help='Compter les comparaisons, sans télécharger de modèles')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size et threads doivent être positifs')
    manifests = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((HERE / 'splits').glob('seed_*.json'))]
    if len(manifests) != 5:
        raise ValueError('Les cinq partitions doivent être présentes dans splits.')
    catalog = manifests[0]['catalog']
    for m in manifests:
        train, val, test = map(set, [m['reserved_train_identities'], m['validation_identities'], m['test_identities']])
        if m['catalog'] != catalog or (len(train), len(val), len(test)) != (70, 20, 25) or train & val or train & test or val & test:
            raise ValueError('Partitions incohérentes')
    pairs = planned_pairs(manifests)
    print(f'{len(catalog)} images ; {len(pairs):,} comparaisons locales uniques, sans shortlist.', flush=True)
    if args.plan:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    # Vérification des photos même lorsqu'un cache global est fourni.
    for idx, row in catalog.items():
        path = args.base.joinpath(*row['relative_path'].split('/'))
        if not path.is_file() or evaluation.file_hash(path) != row['sha256']:
            raise ValueError(f'Photo absente ou modifiée : {path}')
    if args.mega_cache:
        with np.load(args.mega_cache, allow_pickle=False) as cache:
            ids = cache['ids'].tolist()
            features = cache['features'].copy()
            global_metadata = json.loads(str(cache['metadata']))
            if (str(cache['fingerprint']) != fingerprint(catalog) or ids != sorted(catalog, key=int)
                    or str(cache['preprocess']) != evaluation.PREPROCESS
                    or global_metadata.get('hub_model') != 'hf-hub:BVRA/MegaDescriptor-S-224'):
                raise ValueError('Cache MegaDescriptor incompatible')
    else:
        mega_args = SimpleNamespace(**vars(args), recompute=False)
        ids, features, global_metadata = extract_mega(mega_args, catalog)
    if features.ndim != 2 or len(features) != len(ids) or not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5):
        raise ValueError('Vecteurs globaux invalides')
    local = LocalMatches(args, catalog)
    local.precompute(pairs)
    lookup = {idx: n for n, idx in enumerate(ids)}
    local_scores = np.full((len(ids), len(ids)), -1, dtype=np.int32)
    for a, b, score in local.db.execute('SELECT a,b,matches FROM pairs'):
        i, j = lookup[str(a)], lookup[str(b)]
        local_scores[i, j] = local_scores[j, i] = score
    local.db.close()
    results = evaluate(manifests, ids, features, local_scores, args.output)
    write_json(args.output / 'results_wildfusion.json', {
        'metadata': {'settings': SETTINGS, 'global': global_metadata, 'local': local.metadata,
                     'evaluation_version': '1.0.1', 'unique_local_pairs': len(pairs),
                     'manifest_sha256': {p.name: evaluation.file_hash(p) for p in sorted((HERE / 'splits').glob('seed_*.json'))},
                     'notes': ['Independent two-stream implementation inspired by WildFusion, not its full published ensemble.',
                               'Calibration on 70 training identities only; validation used only for rejection thresholds.',
                               'Balanced calibration gives a case-control score, not an operational identity posterior.',
                               'No shortlist, query expansion or test-label fitting. No learned-calibration score among 115 identities.',
                               'Exploratory extension; repeated partitions overlap. Sessions, views, seasons and ages uncontrolled.']},
        'results': results})
    print(f'Terminé : {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit('Arrêt demandé. Relancer la même commande pour reprendre les paires enregistrées.')
    except Exception as exc:
        raise SystemExit(f'Erreur : {type(exc).__name__}: {exc}') from exc
