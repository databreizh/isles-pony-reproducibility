# DINOv2 ViT-B/14 : comparaison de taille du backbone

Décompresser `dinov2_base_equine` dans `${EQUINE_PROJECT_ROOT}`, à côté de `background_equine` et `robustness_equine`. Activer le même environnement virtuel, puis :

```powershell
cd "${EQUINE_PROJECT_ROOT}\dinov2_base_equine"
python -m pip install -r requirements.txt
python run_dinov2_base.py --base "${EQUINE_DATA_ROOT}"
```

Le script utilise les 1 546 photos originales retenues et les partitions fixées graines 42–46. DINOv2 ViT-B/14 officiel, **sans registers**, reste gelé ; seul le passage de ViT-S à ViT-B change le backbone. Prétraitement identique : EXIF, RGB, marge carrée grise 127, bicubique 224, normalisation ImageNet, descripteurs normalisés L2. Les caractéristiques de ViT-B sont à 768 dimensions.

Trois variantes : brut, PCA non centrée 128D calculée exclusivement sur l'entraînement, projection supervisée 128D initialisée par cette PCA. Pour chaque partition, mêmes 70/20/25 identités entraînement/validation/test, même budget de 200 époques × 10 lots, 16 identités × 4 photos par lot, batch-hard cosine marge 0,2, Adam taux 3e-4 avec L2 couplée 1e-4. Sélection exclusivement sur validation : Top-1 macro moyen pour 1/3/5 références, tous les cinq epochs, première époque en cas d'égalité exacte. Aucun choix sur test.

Closed-set Top-1/3/5, open-set et inconnus externes sont évalués avec les mêmes fonctions que les expériences v2. Le seuil open-set vise au plus 5 % de fausses acceptations par photo sur validation ; ce n'est pas une garantie sur le test. Les partitions se recouvrent ; les écarts-types entre partitions ne sont pas des intervalles de confiance indépendants.

En fin de calcul, les trois variantes ViT-S sont réévaluées avec leurs caches et checkpoints 200 époques existants, sans réentraînement, pour créer une comparaison directe dans `results\comparison`. Les signatures de ces checkpoints sont vérifiées avec les mêmes features, partitions et paramètres. Les références restent 1/3/5 par identité, avec les mêmes requêtes.

## Temps et reprise

Le premier chargement peut télécharger les poids officiels et nécessite Internet. Le CPU est utilisé si aucun GPU n'est disponible. Les poids et versions sont tracés par empreinte. Après environ 50 premières photos, le terminal donne une estimation du temps restant pour **l'extraction seulement**, hors téléchargement et apprentissage des projections. Cette estimation évolue avec les lots et ne garantit pas un temps total.

Pour ne faire que ce chronométrage :

```powershell
python run_dinov2_base.py --base "${EQUINE_DATA_ROOT}" --benchmark-only
```

Puis relancer sans `--benchmark-only` : les vecteurs déjà enregistrés sont conservés. Les lots sont enregistrés progressivement ; les checkpoints des partitions entièrement entraînées sont réutilisés. Une partition interrompue pendant son entraînement redémarre de l'époque zéro. Ne pas lancer deux instances dans le même dossier de résultats.

En cas de mémoire insuffisante : `--batch-size 1`. Les paramètres de lots d'extraction et de threads peuvent changer en reprise. Pour changer le budget d'entraînement, choisir un autre `--output` ; le budget ViT-S doit aussi être identique pour la comparaison. `--epochs` existe pour les tests, conserver 200 pour cette expérience.

Par défaut, le cache ViT-S est celui de `..\background_equine\results\embeddings_dinov2.npz`, et ses projections sont dans `..\robustness_equine\results\projections\dinov2`. Au besoin, préciser `--small-embeddings` et `--small-projections`. L'option `--skip-small` calcule ViT-B seul ; elle ne supprime aucun résultat antérieur.

## Fichiers à envoyer

- `results\summary.csv`, `results\results.json`, `results\episode_counts.csv` ;
- si la comparaison est terminée : `results\comparison\summary.csv` et `results\comparison\results.json` ;
- les cinq `selection_seed_*.json` et `training_history_seed_*.csv` de `results\projections\dinov2_vitb14`.

`results\operating_points.csv` contient les courbes de rejet ; `extraction_timing.csv` donne les temps mesurés. Ne pas envoyer les gros vecteurs NPZ ou le cache partiel sauf demande. Les résultats racine sont ViT-B seul ; `comparison` inclut ViT-S et ViT-B.

Il s'agit d'un contrôle de taille dans la famille DINOv2. Il n'établit pas une supériorité générale sur toutes les tailles des modèles animaliers. ViT-B et ViT-S n'ont pas non plus le même nombre de paramètres de projection malgré une sortie commune 128D.

Source officielle du chargement du backbone : https://github.com/facebookresearch/dinov2 .

`python self_test.py` vérifie hors ligne le prétraitement, les sources, le cache de vecteurs, les projections et les épisodes sur données synthétiques. Il ne télécharge pas les poids ViT-B et ne produit pas de résultats sur votre corpus.
