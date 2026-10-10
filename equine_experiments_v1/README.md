# Expérimentations — corpus des Isles, version 1

## Démarrage dans PyCharm sous Windows

Python 3.10 ou plus récent recommandé. Extraire entièrement l'archive, ouvrir
le dossier `equine_experiments_v1` et son terminal. Les chemins des photos sont
déjà configurés ; aucune copie ou modification de photo n'est effectuée.

Commencer par HOG, qui ne télécharge aucun poids de modèle :

```powershell
python -m pip install -r requirements_hog.txt
python run_experiments.py --model hog
```

Puis installer les dépendances des modèles profonds et lancer chaque modèle :

```powershell
python -m pip install -r requirements_deep.txt
python run_experiments.py --model resnet50
python run_experiments.py --model dinov2
```

Le premier lancement des modèles profonds nécessite Internet pour télécharger
les poids (et le dépôt officiel pour DINOv2). Aucune photo n'est envoyée.
Les versions CPU de PyTorch suffisent ; `--device auto` utilise CUDA si disponible.
Si l'installation PyTorch ne convient pas à la carte graphique, utiliser la
commande proposée sur https://pytorch.org/get-started/locally/ pour son matériel.
Le temps de calcul dépend de la machine ; aucune durée n'est promise.

Exécuter les commandes depuis le dossier extrait. Si les images ont été déplacées :

```powershell
python run_experiments.py --model hog --base "D:\datasetDesIsles"
```

Les résultats et vecteurs sont enregistrés dans `results`. Une nouvelle exécution
réutilise les vecteurs du modèle après vérification des empreintes des photos.
Pour refaire l'extraction : `--recompute`. Pour limiter la mémoire des modèles
profonds : `--batch-size 2`. Les manifestes sont déjà fournis ; le script
`prepare_splits.py` permet de les reconstruire à l'identique à partir de l'inventaire.
Ne pas modifier l'inventaire, les paramètres ou les splits après lecture des
résultats de test pour améliorer les scores.

## Fichiers à transmettre après chaque modèle

- `results/summary_hog.csv` et `results/results_hog.json` pour HOG ; mêmes noms
  avec `resnet50` ou `dinov2` ensuite.
- `results/predictions_MODELE.csv` si une analyse des erreurs est souhaitée.
- Inutile de transmettre les fichiers `embeddings_MODELE.npz` dans un premier temps.

## Corpus réellement audité

122 poneys, 1 549 images JPEG lisibles, aucune date EXIF exploitable,
aucun doublon exact de fichier ou de pixels détecté. 115 poneys ont **au moins**
10 photos (26 en ont exactement 10), pour 1 505 images. Les 7 autres ont
44 images, entre 5 et 7 chacune. Médiane globale : 12 images ; maximum : 22.

Trois paires de proximité pHash ont été détectées chez Jump, Ladygaga et padawan.
Par précaution, un seul représentant par groupe est conservé, choisi par ordre
lexicographique du chemin relatif. Cela écarte trois fichiers de l'évaluation,
sans les supprimer sur disque : 1 502 images principales et 44 complémentaires.
Ladygaga conserve neuf images, ce qui suffit au protocole choisi.

## Protocoles préfixés

Les cinq répétitions utilisent les graines 42, 43, 44, 45 et 46. Les modèles sont
figés : **aucun entraînement sur ces photos**. Les 70 identités dites
`reserved_train_identities` sont réservées pour une éventuelle extension apprise,
mais ne servent pas à ajuster les modèles dans cette version.

| Protocole | Identités et usage |
|---|---|
| `closed_115` | Galerie des 115 poneys principaux ; 927 requêtes fixes par répétition |
| `reid_25` | 25 identités réservées au test ; aucune adaptation sur elles |
| `open_validation` | 20 identités : 10 enrôlées et 10 inconnues pour fixer le seuil |
| `open_test` | 25 identités : 10 enrôlées et 15 inconnues, disjointes de la validation |
| `external_7` | Même galerie des 10 connus du test ; les 44 photos des 7 autres comme inconnus |

Dans chaque répétition, les 115 identités sont tirées en 70 réservées / 20
validation / 25 test. Pour chaque poney principal, cinq photos sont réservées
à la galerie maximale ; toutes les autres sont des requêtes. Les galeries 1, 3
et 5 sont emboîtées et les requêtes restent les mêmes, même si certaines
références ne sont pas utilisées dans les petites galeries. Les inconnus sont
entièrement absents de la galerie ; toutes leurs photos servent de requêtes.

`closed_115` est une mesure descriptive qui chevauche les identités des autres
protocoles. Elle ne doit pas servir à choisir les méthodes, les seuils ou les
hyperparamètres. Les trois modèles et règles sont fixés avant mesure. `reid_25`
mesure un transfert de représentations préentraînées ; il ne démontre pas
la généralisation d'un modèle entraîné sur les 70 identités, puisqu'il n'y a
pas encore cet entraînement.

## Représentations et scores

Toutes les images subissent la correction d'orientation EXIF, la conversion RGB,
un remplissage gris (127) pour obtenir un carré et un redimensionnement bicubique
à 224 × 224. L'image entière est conservée : aucun détecteur ni annotation.

- HOG : niveaux de gris, 9 orientations, cellules 16 × 16, blocs 2 × 2, L2-Hys.
- ResNet50 : poids explicites `IMAGENET1K_V2`, tête retirée, vecteur de 2 048 dimensions.
- DINOv2 : modèle officiel `dinov2_vits14`, vecteur CLS de 384 dimensions.

Les modèles profonds utilisent la normalisation ImageNet. Tous les vecteurs
sont normalisés L2. Similarité cosinus ; score d'une identité = maximum parmi
ses références. Les ex aequo sont résolus par ordre lexical des identités.
Le remplissage carré est une adaptation explicitement choisie pour conserver
le corps entier ; ce n'est pas le recadrage central par défaut de ResNet.
Les poids, versions installées et empreintes des paramètres des modèles profonds
sont consignés. Le dépôt DINOv2 est téléchargé depuis sa branche principale ;
l'empreinte des poids et le cache permettent de tracer cette dépendance.

## Mesures

Identification : Top-1 / Top-3 / Top-5, moyenne par photo (`micro`) et par poney
(`macro`). Des intervalles percentiles à 95 % sont calculés pour la moyenne
macro de chaque répétition avec 1 000 rééchantillonnages de poneys, en conservant
la galerie et les prédictions fixes. Ils ne couvrent pas l'incertitude sur les
séances photographiques ni toute celle due au choix de galerie.

Rejet : acceptation si le meilleur score est supérieur ou égal au seuil.
Pour chaque modèle, répétition et taille de galerie, le seuil est choisi pour
obtenir au plus 5 % de fausses acceptations **empiriques par photo** parmi
les inconnus de validation. C'est un objectif de calibration, pas une garantie
sur le test. Les galeries de validation et de test ont chacune 10 identités
connues. On mesure ensuite :

- identification correcte ET acceptation des connus ;
- rejet des connus ;
- fausse acceptation des inconnus, par photo et par poney.

`summary_MODELE.csv` rapporte moyenne et écart-type sur les cinq répétitions.
Elles utilisent le même corpus et se chevauchent : leur écart-type n'est pas
un intervalle de confiance de cinq jeux de données indépendants. Les sept
inconnus complémentaires constituent un résultat exploratoire limité.

## Limites à déclarer dans l'article

Les groupes pHash ne sont pas des séances certifiées : des rafales ou recadrages
peuvent être manqués. Les images entières peuvent contenir des indices liés
au décor, au cavalier, au matériel ou au photographe. Sans annotations, on
ne quantifie pas les effets de robe, pose, contexte ou expression. Le corpus
ne démontre pas une fiabilité opérationnelle sur des animaux extérieurs au
site de collecte. Le chevauchement avec les données de préentraînement des
modèles publics n'est pas vérifié. Les 1 549 photos ne sont donc pas assimilées
à 1 549 observations indépendantes de terrain.

Supprimer les anciens résultats indicatifs du manuscrit ; ne remplacer les
tableaux qu'après exécution réelle. Cette version ne met pas en œuvre
EquiFACS, la segmentation des marques ou l'apprentissage par triplets.

## Validation du programme

Les partitions et calculs de classement/rejet ont été vérifiés sur les manifestes
réels et des vecteurs synthétiques. L'extraction HOG a été vérifiée sur des
images générées pour le test. Les photos originales sont disponibles seulement
sur l'environnement d'origine : aucune performance réelle n'a été mesurée lors
de la préparation. Les extractions préentraînées nécessitent encore une
vérification d'exécution sur cet ordinateur.

Sources des implémentations :
https://github.com/facebookresearch/dinov2
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
