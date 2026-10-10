# Analyse des vues avec annotations humaines

Ce script remplace les catégories CLIP par vos annotations manuelles. Aucun modèle n'est téléchargé, aucune photo ni masque natif n'est nécessaire pour le recalcul : il réutilise les descripteurs et les projections NPZ. Pas d'entraînement.

Décompresser dans `equitrain`, puis lancer dans l'environnement virtuel habituel :

```powershell
cd "${EQUINE_PROJECT_ROOT}\human_views_equine"
python run_human_views.py --annotations "..\annotation_vues_equine\annotations\annotations_vues.csv"
```

Au besoin : `python -m pip install numpy Pillow threadpoolctl`. Si le CSV téléchargé est ailleurs, donner son chemin après `--annotations`. Les caches et checkpoints sont pris dans les mêmes dossiers que précédemment. Des options explicites existent pour `--robustness-results`, `--original-embeddings`, `--foreground-embeddings`, `--audit` et `--output`.

Le CSV est relié au corpus par le chemin relatif ET le SHA-256. Les numéros de l'outil d'annotation ne sont pas les indices expérimentaux. Les trois photos déjà exclues du protocole restent exclues. Un choix vide reste « unannotated » ; les vues partielles sont conservées séparément.

Deux cohortes : `full_original` évalue les épisodes Test-25 existants sur les originaux, y compris les photos dont la segmentation avait échoué ; `matched_segmented` reprend exactement les épisodes appariés v2 sur les originaux et les foregrounds. Les manifests v2 sont comparés à leur reconstruction à partir de l'audit. Ne pas confondre les scores des deux cohortes.

Pour chaque cohorte/condition et méthode :

- `fixed_gallery` : références 1/3/5 inchangées, scores par vue des requêtes ;
- `controlled_pairs` : une référence tête et une référence corps par identité parmi les cinq références réservées ; quatre cellules tête/tête, tête/corps, corps/tête, corps/corps sur les mêmes identités candidates. Les requêtes gardent leur séparation initiale.

Les groupes de requêtes peuvent représenter des poneys différents. Les petites cellules ne démontrent pas un effet causal de la vue. Vérifier les effectifs et le hasard lié au nombre d'identités candidates. Les cellules avec moins de deux candidats ou aucune requête n'ont pas de score. Les moyennes regroupent les partitions évaluables, dont le nombre est indiqué ; cinq partitions se recouvrant ne donnent pas cinq échantillons indépendants. Une seule annotatrice et une analyse ajoutée en cours d'étude doivent être mentionnées dans le papier.

Envoyer après le recalcul : `results\human_view_summary.csv`, `human_view_counts.csv`, `human_view_results.json` et `human_view_segmentation_coverage.csv`. Le dernier fichier décrit le succès de segmentation par vue, pour mesurer la sélection induite par l'exclusion des fallback.

`--raw-only` est un mode de diagnostic sans checkpoints : il calcule uniquement DINOv2 brut. S'ils sont absents, les épisodes appariés sont reconstruits par le protocole v2. Le calcul complet par défaut nécessite les projections déjà présentes sur votre ordinateur.

## Diagnostic déjà calculé sur votre corpus

Le dossier `diagnostic_dinov2_brut` de l'archive contient les vrais résultats recalculés ici avec les descripteurs fournis et vos annotations. Il ne contient PAS les résultats des projections, dont les checkpoints sont sur votre ordinateur. Ces derniers seront écrits dans `results` après le lancement normal.

Sur la cohorte complète Test-25, originaux, cinq références : Top-1 macro moyen 52,33 % pour les têtes, 76,96 % pour les corps entiers et 60,20 % pour les vues partielles. Les groupes de requêtes peuvent représenter des poneys différents. Ce sont des résultats descriptifs, pas une estimation causale de l'effet de la vue.
