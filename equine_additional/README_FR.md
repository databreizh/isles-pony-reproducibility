# Expériences complémentaires — identification des poneys

Trois scripts compatibles avec les cinq manifestes originaux (seeds 42–46).
Les manifestes sont inclus. Aucun chemin personnel, aucune photographie et aucun
ancien poids de projection ne sont inclus.

## Installation

Utiliser de préférence l'environnement Python qui a exécuté les expériences
existantes, avec PyTorch et torchvision compatibles. Dans un terminal ouvert
dans ce dossier :

```powershell
python -m pip install -r requirements.txt
python self_test.py
```

Pour CUDA, conserver une installation PyTorch adaptée à votre GPU. Ne pas
remplacer volontairement une installation GPU fonctionnelle par une version CPU.
Le premier lancement du fine-tuning télécharge le dépôt et les poids officiels
DINOv2. Leurs empreintes sont enregistrées. Pas de dépendance à xformers.

## 1. Exporter les prédictions des modèles du papier

Il faut le cache **DINOv2 ViT-S/14 384D**, et le dossier des cinq projections
**ViT-S** issues des runs finaux avec budget de **200 époques**.
Ne pas utiliser les projections ViT-B (768D), MegaDescriptor-L (1536D), ni les
projections entraînées sur les seuls avant-plans.

Les chemins ci-dessous sont des exemples à remplacer par vos dossiers réels.
Les variables PowerShell évitent de répéter les chemins.

```powershell
$embeddings = "D:\experiences\embeddings_dinov2.npz"
$projections = "D:\experiences\projection_vits_200epochs"
$photos = "D:\corpus\datasetDesIsles"

python export_predictions.py --embeddings "$embeddings" --projections "$projections" --output reference_results
```

Le dossier `$photos` contient directement `au_moins_10_photos` et
`moins_de_10_photos`. Le dossier `$projections` contient
`projection_seed_42.npz` à `projection_seed_46.npz`, et, si disponibles,
les `selection_seed_42.json` à `selection_seed_46.json` correspondants.
Les NPZ avec ou sans clé `selected_epoch` sont pris en charge. Si l'époque est
absente du NPZ et que le fichier de sélection existe, le script contrôle sa
cohérence par le score de validation. Sans ces métadonnées, l'époque reste
inconnue : la vérifier dans les logs originaux.

Le score projeté cinq références doit reproduire **89,75 %** à l'arrondi du papier.
Les checkpoints attendus pour ViT-S sont **75, 60, 65, 65, 55**. Si le score ou les
époques diffèrent, retrouver les bons fichiers avant d'interpréter les analyses.
Le script ne choisit jamais une projection en fonction de son score de test.
Les catalogues et identités d'entraînement doivent correspondre aux manifestes.

Sorties : `predictions_reference.csv`, `results_reference.json`.

## 2. Bootstrap et open-set, sans apprentissage

```powershell
python analyze_predictions.py --predictions reference_results/predictions_reference.csv --method-a dinov2 --method-b projection_triplet --replicates 10000 --output analysis_projection
```

Le bootstrap rééchantillonne les **identités distinctes**. Toutes les occurrences
d'un même poney dans les partitions reçoivent la même multiplicité. Les méthodes
sont appariées sur les mêmes requêtes. Dans chaque répétition, on recalcule la
différence macro de chaque partition, puis la moyenne des cinq différences :
l'estimand reste celui des tableaux du papier. Une identité apparaissant dans
plusieurs partitions n'est donc pas traitée comme plusieurs clusters indépendants.

L'intervalle percentile à 95 % est **conditionnel aux modèles ajustés et épisodes
enregistrés**. Il ne couvre ni l'incertitude due au réentraînement, ni la dépendance
entre séances, ni la généralisation à d'autres élevages. Aucun p-value confirmatoire
n'est produit pour cette étude exploratoire.

Sorties :

- `paired_bootstrap.json` : différence B − A en points et intervalle à 95 % ;
- `open_set_operating_errors.csv` : CIA, rejet des connus, mauvaise identité acceptée,
  acceptation des inconnus, pour les décisions calibrées sur validation ;
- `open_set_curves_by_seed.csv` : balayage de seuils séparé par méthode et partition ;
- `open_set_mean_envelope.csv`, `cia_ufa.pdf`, `cia_ufa.png` : enveloppes descriptives
  du compromis CIA–UFA, moyennées aux mêmes budgets UFA ;
- `interpretation.txt` : portée des analyses.

Les courbes balaient les **scores test a posteriori** : elles décrivent un compromis,
elles ne fournissent pas un seuil opérationnel. Les points affichés sont les
résultats avec le seuil originel calibré sur validation à 5 % de fausse acceptation
au niveau photo. Ne pas choisir un nouveau seuil sur test. Les métriques des
courbes sont macro par identité ; les enveloppes ne constituent pas une expérience
à seuil commun ni une validation prospective. L'écart-type entre partitions reste
descriptif. L'ensemble sparse `external_7` reste dans les exports, mais n'entre pas
dans la courbe principale, qui utilise `open_test`.

## 3. Adaptation partielle de DINOv2 ViT-S

```powershell
python finetune_last_block.py --base "$photos" --embeddings "$embeddings" --device auto --batch-size 8 --epochs 200 --steps 10 --output last_block_results
```

Seuls le dernier bloc Transformer **et la normalisation finale** sont entraînés.
La sortie reste le descripteur CLS normalisé de **384 dimensions**, sans nouvelle
tête. C'est donc une baseline d'adaptation partielle ; ce n'est pas une reproduction
du modèle projeté 128D ni un fine-tuning complet. Cette différence doit être décrite
dans le papier.

Réglages fixés à l'avance : Adam, LR `1e-5`, L2 couplé `1e-4`, triplet batch-hard
cosinus, marge 0,2, 16 identités × 4 images, dix étapes par époque, 200 époques,
aucune augmentation. LR plus faible que pour la projection parce que les paramètres
préentraînés sont adaptés ; il est fixé, sans recherche d'hyperparamètres sur test.
Les blocs gelés sont exécutés en mode évaluation, et leurs tokens sont mis en cache
en float32. Le dernier bloc reste en mode évaluation pour éviter d'ajouter un effet
de régularisation stochastique ; ce mode n'empêche pas le calcul des gradients.

Avant apprentissage, le script vérifie les hashes des photos et reproduit le cache
gelé fourni, avec une tolérance numérique `2e-5` par composante. Une incohérence de
poids ou de prétraitement bloque le run. Le cache de tokens occupe environ **610 Mo**
(hors poids et résultats). CUDA est préférable ; CPU est accepté, mais peut être lent.
La taille 8 découpe les forwards ; le graphe complet du batch de 64 images reste
nécessaire pour les triplets. Une petite taille ne garantit donc pas l'absence de
manque de mémoire GPU.

Sélection sur les seules 20 identités de validation : Top-1 macro moyen des
galeries 1/3/5, époque zéro puis toutes les cinq époques, premier checkpoint en cas
d'égalité. Les identités de test ne servent pas à la sélection. Les cinq partitions,
les images de référence et les requêtes ne sont pas modifiées.

Sorties : `predictions_last_block.csv`, `results_last_block.json`, `selection.csv`,
`training_history.csv`, `run_config.json`, plus les poids et embeddings de chaque
seed. Le temps enregistré inclut entraînement et évaluations de validation, exclut
le téléchargement, le cache initial et l'évaluation test finale : ne pas le comparer
à des logs de projection de périmètre différent. Le nombre de paramètres entraînés
est enregistré. Aucune revendication d'égalité de coût calcul entre méthodes.

Une seed terminée est réutilisée au redémarrage avec les mêmes paramètres. Une seed
interrompue recommence depuis l'initialisation. Ne pas modifier les poids téléchargés
ou les paramètres au milieu d'un dossier de résultats ; utiliser un nouveau dossier.
Ne pas effacer le cache de tokens pour simplement relancer après interruption.

## 4. Comparer adaptation partielle et projection

```powershell
python analyze_predictions.py --predictions reference_results/predictions_reference.csv last_block_results/predictions_last_block.csv --method-a projection_triplet --method-b dinov2_last_block --output analysis_last_block
```

Les courbes incluront les trois méthodes ; le bootstrap compare uniquement A et B.
Les résultats de l'adaptation partielle sont nouveaux et exploratoires. Ne pas
modifier les hyperparamètres à partir de son score de test.

## Vérifications effectuées sur les scripts

Syntaxe Python et interfaces CLI vérifiées. Export et analyses exécutés avec des
embeddings et projections existants compatibles ; ces anciennes projections sont
utilisées uniquement pour vérifier le fonctionnement des scripts, pas pour produire
les nouveaux chiffres du papier. Les tests synthétiques couvrent les identités
répétées, l'appariement et la décomposition des erreurs. Le fine-tuning GPU complet
n'a pas été exécuté dans l'environnement de préparation, qui n'a pas PyTorch.
Il reste à exécuter chez vous avec les photographies et les poids adéquats.

Implementation DINOv2 : https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
