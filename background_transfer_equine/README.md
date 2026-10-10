# Changement de fond, projections figées

Cette expérience teste la sensibilité au contexte en réattribuant des décors indépendamment à chaque photo. Elle réutilise DINOv2 ViT-S/14, les deux projections enregistrées (apprises sur les originaux ou les foregrounds), les masques et les épisodes appariés de Robustesse_Equine_v2. Aucun entraînement n'est lancé et aucune annotation n'est requise.

## Installation et lancement Windows

Décompresser ce dossier dans `${EQUINE_PROJECT_ROOT}`, à côté de `robustness_equine`. Conserver le dossier complet `robustness_equine\results`, notamment les masques natifs et les checkpoints NPZ. Activer l'environnement virtuel habituel, puis :

```powershell
cd "${EQUINE_PROJECT_ROOT}\background_transfer_equine"
python -m pip install -r requirements.txt
python self_test.py
python run_background_transfer.py --project-root ".." --robustness-results "..\robustness_equine\results" --base "${EQUINE_DATA_ROOT}" --previews
```

Le premier téléchargement de DINOv2 nécessite Internet si le modèle n'est pas déjà dans le cache Torch. Le script vérifie que ses poids correspondent exactement à ceux des descripteurs précédents. Il utilise le GPU si disponible, sinon le CPU. Les vecteurs sont enregistrés progressivement : après une interruption, relancer exactement la même commande. Pour changer les paramètres, utiliser un nouveau dossier `--output`.

Les caches `embeddings_dinov2.npz`, `embeddings_dinov2_foreground.npz` et `segmentation_audit.csv` sont recherchés dans le projet. Si plusieurs versions différentes existent, préciser les bons fichiers via `--original-embeddings`, `--foreground-embeddings` et `--audit`.

## Protocole

- Cinq partitions existantes, graines 42–46, mêmes photos sources et mêmes épisodes que l'évaluation appariée v2 ; aucune image fallback.
- Trois conditions : originaux, foregrounds sur gris, puis changement de décor. Trois méthodes dans chaque condition : DINOv2 brut, projection apprise sur les originaux, projection apprise sur les foregrounds.
- Par défaut, les décors sont des morceaux de photos hors du masque dilaté du poney. Les photos avec plusieurs chevaux détectés sont exclues du réservoir de décors. Les patches uniformes, petits ou identiques sont exclus.
- Pour chaque partition, seuls les poneys d'entraînement fournissent les décors utilisés. Leurs identités sont réparties en trois réserves disjointes : références, requêtes de validation et requêtes de test. Chaque photo reçoit un décor tiré indépendamment, sans association fixe entre poney et décor.
- Trois tirages de décors par partition (`--repeats 3`). Les métriques sont d'abord moyennées sur les tirages d'une partition, puis sur les cinq partitions. Ces dernières se recouvrent ; les tirages ne constituent pas de nouvelles observations animales indépendantes.
- Closed-set : Top-1/3/5 et mAP, références de taille 1/3/5. Open-set : seuil calibré exclusivement sur la validation de chaque méthode, condition et tirage, cible de 5 % de fausses acceptations par photo ; ce n'est pas une garantie de 5 % sur le test. Les poids des modèles restent figés, mais les seuils sont recalibrés. Une courbe de points de fonctionnement issus de validation est aussi exportée.

Option facultative : `--background-dir "D:\decors_sans_animaux"` utilise vos propres photos de décors, au moins neuf fichiers différents après déduplication. Les fichiers sont répartis entre les trois réserves. Choisir un dossier contenant exclusivement des décors sans animaux : le script ne certifie pas automatiquement leur absence. Cette option n'est pas nécessaire pour le lancement par défaut.

## Résultats à transmettre

Dans `results`, envoyer ces petits fichiers :

1. `summary_transfer.csv` : moyennes et écarts-types entre partitions.
2. `results_transfer.json` : résultats par partition et tirage, paramètres et provenance des checkpoints.
3. `episode_counts_transfer.csv` : effectifs évalués.

`operating_points_transfer.csv` contient les courbes open-set. `donor_candidates.json`, `donor_pools_seed_*.json` et `assignments_seed_*.csv` documentent les décors et les affectations. Avec `--previews`, six petits aperçus original/composite sont créés par partition et tirage dans `results\cache\seed_*\repeat_*\previews`. Il suffit d'en examiner quelques-uns localement ; ne pas envoyer tout le cache.

## Interprétation et limites

Une baisse sur les fonds réattribués indique une sensibilité au décor et/ou aux artefacts de segmentation et de composition. Une stabilité renforce l'argument de robustesse au contexte dans cette expérience synthétique. Elle ne prouve pas la reconnaissance d'un poney réellement retrouvé sur un nouveau site : les décors par défaut proviennent de la même collection, parfois du même propriétaire ou environnement. Des réserves disjointes de photos ou d'identités donatrices ne garantissent pas des lieux différents.

Les masques automatiques peuvent garder des pixels de contexte ou enlever des parties du poney. Un patch admissible peut contenir un animal non détecté. La silhouette, le cadrage et les pixels du poney sont conservés par le compositing, avec contours binaires ; le redimensionnement et la marge grise carrée restent ceux du pipeline précédent. Les photos sources avec plusieurs chevaux restent dans la cohorte v2 : le masque ne certifie pas l'identité du cheval sélectionné.

Les tests inclus utilisent des données synthétiques et vérifient la conservation des métriques lors du changement d'indices, la séparation des réserves, le compositing et l'agrégation statistique. Ils ne sont pas des résultats sur votre corpus. L'extraction réelle doit être exécutée sur votre ordinateur.
