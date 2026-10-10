"""Crée des manifestes déterministes à partir de l'inventaire, sans lire les photos."""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = [42, 43, 44, 45, 46]


def prepare(inventory=HERE / 'inventaire_images.csv', output=HERE / 'splits'):
    raw = list(csv.DictReader(inventory.open(encoding='utf-8-sig'), delimiter=';'))
    if len({r['index'] for r in raw}) != len(raw):
        raise ValueError('Indices non uniques')
    groups = defaultdict(list)
    for r in raw:
        if r['status'] == 'ok':
            if not r['candidate_group']:
                raise ValueError('Groupe manquant')
            groups[(r['identity'], r['candidate_group'])].append(r)
    selected, excluded = [], []
    for rows in groups.values():
        rows.sort(key=lambda r: r['relative_path'])
        selected.append(rows[0])
        excluded.extend(rows[1:])
    selected.sort(key=lambda r: int(r['index']))
    # Interdire les copies certaines entre identités plutôt que choisir une étiquette.
    for field in ['sha256', 'pixel_sha256']:
        seen = {}
        for r in selected:
            if r[field] in seen:
                raise ValueError(f'Doublon certain restant : {field}')
            seen[r[field]] = r['identity']
    by_id = defaultdict(list)
    for r in selected:
        by_id[r['identity']].append(int(r['index']))
    main = sorted({r['identity'] for r in selected if r['categorie'] == 'au_moins_10_photos'})
    extra = sorted({r['identity'] for r in selected if r['categorie'] == 'moins_de_10_photos'})
    if len(main) != 115 or len(extra) != 7:
        raise ValueError('Ce protocole est fixé pour les 115 + 7 identités auditées')
    if any(len(by_id[i]) < 6 for i in main):
        raise ValueError('Moins de six images conservées pour une identité principale')
    catalog = {str(r['index']): {'identity': r['identity'], 'poney': r['poney'],
               'relative_path': r['categorie'] + '/' + r['relative_path'].replace('\\', '/'),
               'sha256': r['sha256'], 'candidate_group': r['candidate_group']} for r in selected}
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(inventory.read_bytes()).hexdigest()
    summaries = []
    for seed in SEEDS:
        rng = random.Random(seed)
        identities = main[:]
        rng.shuffle(identities)
        train, val, test = identities[:70], identities[70:90], identities[90:]
        galleries, queries = {}, {}
        for identity in main:
            images = sorted(by_id[identity])
            rng.shuffle(images)
            galleries[identity], queries[identity] = images[:5], images[5:]
        def episode(ids):
            return {'identities': ids,
                    'gallery': {str(k): [x for i in ids for x in galleries[i][:k]] for k in [1,3,5]},
                    'queries': [x for i in ids for x in queries[i]]}
        def open_episode(ids, known_count):
            known, unknown = ids[:known_count], ids[known_count:]
            return {'known': episode(known), 'unknown_identities': unknown,
                    'unknown_queries': [x for i in unknown for x in by_id[i]]}
        manifest = {'version': '1.0', 'seed': seed, 'inventory_sha256': source_hash,
                    'catalog': catalog, 'reserved_train_identities': train,
                    'validation_identities': val, 'test_identities': test,
                    'closed_115': episode(main), 'reid_25': episode(test),
                    'open_validation': open_episode(val, 10),
                    'open_test': open_episode(test, 10),
                    'external_unknown_identities': extra,
                    'external_unknown_queries': [x for i in extra for x in by_id[i]],
                    'excluded_similarity_candidates': [int(r['index']) for r in excluded],
                    'notes': ['Images entières; modèles figés; aucun apprentissage sur le corpus.',
                              'Une image par groupe de similarité pHash; pas de séparation certifiée par séance.',
                              'Galeries emboîtées 1/3/5; mêmes requêtes pour les trois tailles.',
                              'closed_115 est descriptif et chevauche validation/test des autres protocoles.',
                              'Aucun choix de modèle ou hyperparamètre ne doit utiliser closed_115 ou les résultats test.']}
        (output / f'seed_{seed}.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        summaries.append({'seed': seed, 'images_conservees': len(selected),
                          'images_ecartees_candidates': len(excluded),
                          'closed_queries': len(manifest['closed_115']['queries']),
                          'reid_queries': len(manifest['reid_25']['queries']),
                          'open_test_known_queries': len(manifest['open_test']['known']['queries']),
                          'open_test_unknown_queries': len(manifest['open_test']['unknown_queries']),
                          'external_unknown_queries': len(manifest['external_unknown_queries'])})
    (output / 'resume_splits.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(json.dumps(summaries, indent=2))
    return summaries


if __name__ == '__main__':
    prepare()
