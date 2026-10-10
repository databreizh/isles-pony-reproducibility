# Expériences complémentaires pour le corpus de poneys

Ce dossier se place à côté de `background_equine`, `projection_equine`, `megadescriptor_equine`, etc., dans `${EQUINE_PROJECT_ROOT}`.

## Lancement conseillé : tout le lot

Avec le même environnement virtuel que pour les expériences précédentes :

```powershell
cd "${EQUINE_PROJECT_ROOT}\robustness_equine"
python -m pip install -r requirements.txt
python run_robustness.py --project-root ".." --base "${EQUINE_DATA_ROOT}"
python plot_results.py
```

Le budget par défaut est de 200 époques, avec 10 batches par époque. Le code recherche les caches `embeddings_dinov2.npz`, `embeddings_resnet50.npz`, `embeddings_megadescriptor.npz`, `embeddings_miewid.npz` et `embeddings_dinov2_foreground.npz` dans le projet. Il affiche les chemins utilisés. Un modèle original absent est signalé et ignoré, sans inventer de résultat. L'étape fond nécessite obligatoirement les deux caches DINOv2 et l'ancien `segmentation_audit.csv`.

L'installation n'utilise pas `--upgrade`. Conserver les versions des expériences précédentes autant que possible : le script vérifie les poids et la reproduction des images masquées. Le premier chargement peut nécessiter Internet si les dépôts/poids YOLO et DINOv2 ne sont pas déjà en cache.

Les calculs peuvent prendre un certain temps sur CPU, particulièrement pour les grandes dimensions et la segmentation. Les progrès sont affichés. Il n'y a pas besoin d'annotation manuelle.

## Deux étapes séparables

Projections uniquement, sans YOLO ni nouvelle extraction :

```powershell
python run_robustness.py --project-root ".." --stage projections
```

Contrôles du fond uniquement (réutilise la projection DINOv2 200 époques si elle est déjà terminée, sinon l'entraîne) :

```powershell
python run_robustness.py --project-root ".." --stage background --base "${EQUINE_DATA_ROOT}"
```

Un test de fonctionnement très court peut utiliser `--epochs 1 --output results_smoke`. Ce n'est pas une expérience scientifique à rapporter. Pour le véritable lot, utiliser 200 époques et un autre dossier de sortie.

## Si plusieurs caches sont trouvés

Les copies ayant exactement les mêmes indices et vecteurs sont acceptées. Des copies différentes provoquent un arrêt : préciser alors le fichier à utiliser, plutôt que de sélectionner silencieusement un modèle.

```powershell
python run_robustness.py --project-root ".." --embedding "dinov2=..\background_equine\embeddings_dinov2.npz" --embedding "foreground=..\background_equine\results\embeddings_dinov2_foreground.npz" --audit "..\background_equine\results\segmentation_audit.csv" --base "${EQUINE_DATA_ROOT}"
```

Adapter ces chemins aux fichiers réellement présents. `--embedding` peut être répété pour `resnet50`, `megadescriptor`, `miewid`, `dinov2`, `foreground`. `--yolo-weights CHEMIN` permet de préciser l'ancien checkpoint YOLO. `--device cpu` ou `--device cuda` choisit le matériel pour l'extraction du fond; les projections NumPy utilisent le CPU. `--threads 4` limite également les threads BLAS.

## Ce qui est calculé

### 1. Une projection par descripteur

Pour chaque descripteur original disponible, les cinq partitions originales 42–46 sont réutilisées. Le script compare le vecteur brut, PCA-128 non centrée, et la projection 128D optimisée. La PCA est calculée par SVD sur les seules images d'entraînement. Adam, la marge 0,2, les batches 16 identités × 4 images et le terme L2 sont ceux de l'expérience précédente. Le checkpoint est choisi sur validation, toutes les cinq époques et à la dernière époque, en conservant le plus ancien en cas d'égalité exacte. Aucun score test ne sélectionne le checkpoint.

Une matrice DISTINCTE est entraînée pour chaque descripteur : les poids DINOv2 ne sont pas appliqués à une autre dimension. Les checkpoints à 200 époques sont de nouvelles expériences; ils ne remplacent pas automatiquement les scores à 60 époques du papier.

### 2. Original / poney masqué / fond avec silhouette grise

Le précédent pipeline n'enregistrait pas les masques natifs. Le script repasse donc YOLO sur les seules images notées `segmented` dans l'ancien audit. Il utilise les mêmes poids et paramètres, puis vérifie que la vue foreground reconstruite est identique à l'ancien cache. Si cette vérification échoue, il s'arrête au lieu de mélanger des interventions différentes. Les nouveaux masques et vecteurs de fond sont enregistrés pour les relances.

Le poney est remplacé par du gris sur le contrôle background. **Ce contrôle conserve la silhouette et la position du masque** : ce n'est pas une image ne contenant que le contexte au sens causal. Les anciennes images sans masque sont toutes exclues; aucun fallback n'est utilisé dans cette comparaison.

Les conditions comparées sont :

- original, foreground et background avec DINOv2 brut;
- les mêmes trois vues avec la projection entraînée sur les originaux;
- foreground avec PCA-128 et projection entraînées sur les seules images effectivement masquées.

La projection originale reste entraînée sur les 70 identités et images originales; la projection foreground utilise seulement leurs images segmentées. Les groupes d'entraînement ayant moins de quatre images masquées sont exclus de cet entraînement et listés via les identités conservées. La validation et le test restent séparés des identités d'entraînement.

### Cohorte appariée sans fallback

Filtrer les cinq anciennes références laisserait souvent trop peu d'animaux. Le script crée donc de NOUVELLES galeries, communes aux conditions : pour chaque identité, il conserve l'ordre des cinq références réservées, puis l'ordre des anciennes requêtes; il enlève les images sans masque et réserve les cinq premières images admissibles comme références. Il faut au moins six images admissibles par identité, pour garder une requête.

Les galeries 1/3/5 sont emboîtées et les requêtes restent fixes. Les mêmes images, identités candidates et seuils de référence sont utilisés dans les trois conditions; les seuils numériques de rejet sont recalibrés séparément pour chaque représentation sur ses inconnus de validation. Les rôles connu/inconnu ne changent pas. Les requêtes inconnues sans masque sont retirées.

**Ces nouveaux scores ne sont pas directement comparables au benchmark initial**, dont les références et les effectifs diffèrent. `episode_counts.csv` donne les effectifs réels. Le hasard Top-1 est calculé comme `1 / nombre d'identités candidates`, et non fixé à 4 %. La cohorte masquée peut favoriser certains types de photographies; sa couverture doit accompagner les résultats.

### Open-set

Le point principal cible au plus 5 % d'acceptation des photos inconnues de validation. Onze points de fonctionnement, dont ce point principal, sont exportés entre 0 et 50 % de cible de validation. Il ne s'agit pas d'une ROC optimisée sur le test. Les courbes relient les moyennes CIA/UFA test obtenues aux mêmes cibles de validation. Les cinq partitions se recouvrent et restent exploratoires. Les taux des fichiers CSV sont entre 0 et 1; les figures les affichent en pourcentage.

## Fichiers à envoyer après le calcul

Dans `results/projections/` :

- `summary.csv`, `results.json`, `episode_counts.csv`, `operating_points.csv`;
- les fichiers `selection_seed_*.json` et `training_history_seed_*.csv` de chaque modèle.

Dans `results/background/` :

- `summary.csv`, `results.json`, `episode_counts.csv`, `operating_points.csv`, `coverage.json`;
- les sélections et historiques du sous-dossier `foreground_projection/`.

Les figures générées par `plot_results.py` peuvent également être envoyées. Inutile d'envoyer `cache/`, les photos ou les gros fichiers NPZ pour une première analyse.

## Reprise

Relancer la même commande avec le même dossier : les masques, les lots de vecteurs de fond et les projections des graines terminées sont réutilisés. Une graine interrompue pendant l'apprentissage est recommencée; le script ne prétend pas reprendre son état Adam en milieu de graine. Changer le budget ou les données impose un nouveau `--output` pour éviter de réutiliser des checkpoints incompatibles.
