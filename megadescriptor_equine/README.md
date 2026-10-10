# MegaDescriptor sur le corpus des Isles

## Installation et exécution Windows/PyCharm

Extraire complètement le dossier, ouvrir son terminal et exécuter :

```powershell
python -m pip install -r requirements_megadescriptor.txt
python run_megadescriptor.py
```

Le chemin du corpus est déjà configuré :
`${EQUINE_DATA_ROOT}`.
Les sous-dossiers sont `au_moins_10_photos` et `moins_de_10_photos`.
Aucune annotation ou modification d'image n'est demandée. Aucun entraînement
n'est effectué. Internet est nécessaire pour télécharger les poids du modèle
sur Hugging Face lors de la première exécution ; les images restent locales.

Les dépendances PyTorch déjà installées peuvent être réutilisées. La commande
pip résout un couple torch/torchvision compatible ; pour CUDA, consulter la
commande adaptée au matériel sur https://pytorch.org/get-started/locally/.
Le script fonctionne également sur CPU. Le modèle est plus gros que ResNet50
en nombre de paramètres, mais aucune durée de calcul n'est garantie.

Pour réduire la mémoire :

```powershell
python run_megadescriptor.py --batch-size 1
```

Pour un corpus déplacé :

```powershell
python run_megadescriptor.py --base "D:\datasetDesIsles"
```

## Fichiers à transmettre

Dans le sous-dossier `results` :

- `summary_megadescriptor.csv` : moyennes et écarts-types ;
- `results_megadescriptor.json` : scores détaillés et versions ;
- `embeddings_megadescriptor.npz` : vecteurs pour une éventuelle analyse complémentaire.

Les prédictions sont aussi exportées dans `predictions_megadescriptor.csv`.
Une nouvelle exécution réutilise les vecteurs après vérification des photos.
Pour recalculer les vecteurs : ajouter `--recompute`.

## Choix fixé avant mesure

Modèle : **MegaDescriptor-S-224**, Swin Small préentraîné pour la
réidentification animale, 49.6 millions de paramètres, entrée 224 × 224.
Le choix Small évite de lancer d'abord le modèle Large-384 plus lourd sur CPU.
Il s'agit de cette variante précise, pas d'un résultat général de toutes les
variantes MegaDescriptor.

Les cinq séparations déjà utilisées sont incluses à l'identique. Le script
conserve les 1 546 images retenues, les galeries 1/3/5, les mêmes requêtes et
les mêmes métriques. Il utilise la correction de comptage de validation v1.0.1.
La tête de classification est supprimée (`num_classes=0`), les vecteurs sont
normalisés L2 et comparés par cosinus, avec maximum par identité.

Prétraitement : orientation EXIF, RGB, remplissage carré gris, redimensionnement
bicubique 224 × 224 et normalisation ImageNet, comme les modèles précédents.
Cela conserve l'image entière même quand seule la tête est visible, sans
recadrage ni contrôle de pose. Le remplissage est une adaptation à notre
protocole ; ce n'est pas une reproduction exacte des prétraitements de tous
les benchmarks de MegaDescriptor. Les poids sont téléchargés via timm,
leurs empreintes et versions installées sont consignées.

La comparaison reste exploratoire après examen des premières méthodes.
Le corpus de préentraînement public est différent de nos photos mais son
absence de chevauchement visuel avec nos sources n'a pas été auditée.
Les photos ne sont pas séparées par séance ou âge ; la variation de robe et
la croissance ne font pas l'objet d'une mesure isolée.

Sources officielles :
https://huggingface.co/BVRA/MegaDescriptor-S-224
https://wildlifedatasets.github.io/wildlife-tools/inference/

## Vérifications réalisées

Le chargement des manifestes, les exports et le chemin d'évaluation ont été
vérifiés sur des données synthétiques. L'extraction complète avec les poids
officiels et les photos réelles doit être exécutée sur l'ordinateur contenant
le corpus. Aucun score MegaDescriptor réel n'est encore disponible.
