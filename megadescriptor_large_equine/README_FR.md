# MegaDescriptor-L-384 — corpus des Isles

Décompresser le dossier `megadescriptor_large_equine` à côté de `megadescriptor_equine`, `robustness_equine` et des autres expériences dans `${EQUINE_PROJECT_ROOT}`.

Dans PowerShell, avec votre environnement virtuel actif :

```powershell
cd "${EQUINE_PROJECT_ROOT}\megadescriptor_large_equine"
python -m pip install -r requirements.txt
python run_megadescriptor_large.py --base "${EQUINE_DATA_ROOT}"
```

Le téléchargement initial des poids nécessite Internet. Le script choisit CUDA si disponible, sinon le CPU. Le lot par défaut est une photo, pour limiter la mémoire. Sur GPU, `--batch-size 4` est possible si la mémoire le permet. Après 50 nouvelles photos il affiche une estimation du temps d'extraction restant, qui exclut l'entraînement des projections. Le téléchargement et le chargement des poids sont chronométrés séparément.

## Expériences

- Poids officiels gelés `hf-hub:BVRA/MegaDescriptor-L-384`, représentation de 1536 dimensions normalisée L2.
- Photos originales retenues dans les cinq manifestes fournis : contrôle des chemins et SHA256, partitions fixes 42–46 (70 identités entraînement / 20 validation / 25 test).
- Représentation brute, PCA non centrée 128D ajustée sur l'entraînement uniquement, projection supervisée 128D initialisée par cette PCA.
- Même budget que les expériences à 200 époques : 10 lots par époque, 16 identités × 4 photos, batch-hard cosine, marge 0,2, Adam 3e-4, pénalité L2 1e-4. Sélection sur validation tous les cinq epochs, PCA à l'époque 0 admissible, premier maximum retenu.
- Top-1/3/5 macro en closed-set, références 1/3/5 ; open-set avec seuil fixé sur validation à UFA photo ≤ 5 %, CIA et UFA sur test et inconnus externes. Le taux test n'est pas contraint à 5 %.
- Comparaison avec MegaDescriptor-S-224 : reprend ses vecteurs et ses cinq projections existantes, vérifie leur signature et le même budget avant d'évaluer les mêmes épisodes. Les partitions se recoupent entre graines ; leurs résultats ne sont pas indépendants.

Prétraitement : correction EXIF, RGB, conservation de toute la photo avec padding carré gris 127, redimensionnement bicubique direct des originaux à 384×384. Aucun agrandissement du cache 224, aucun recadrage manuel. Normalisation moyenne/écart-type issue de `model.pretrained_cfg`, enregistrée dans les métadonnées. La configuration publiée fournit les valeurs ImageNet ; l'exemple de la model card utilise 0,5. Ce script choisit explicitement la configuration des poids, avec la politique de cadrage complet des expériences précédentes, plutôt que l'exemple de la model card ou son center-crop. Sources : https://huggingface.co/BVRA/MegaDescriptor-L-384 et https://huggingface.co/BVRA/MegaDescriptor-L-384/raw/main/config.json.

La comparaison S-224/L-384 change à la fois capacité et résolution : elle ne mesure pas le seul effet du nombre de paramètres. Elle porte sur les originaux et ne démontre pas une généralisation à de nouveaux lieux ou fonds.

## Caches et reprise

Chemins recherchés par défaut :

```
..\megadescriptor_equine\results\embeddings_megadescriptor.npz
..\robustness_equine\results\projections\megadescriptor\projection_seed_42.npz
```

Le dossier de projections doit contenir les cinq NPZ et les cinq `selection_seed_*.json`. Autres emplacements : `--small-embeddings "chemin.npz" --small-projections "dossier"`. Pour exécuter seulement L et ses projections : `--skip-small`.

Les nouvelles sorties vont dans `results` de ce dossier. Aucun ancien cache n'est remplacé. Chaque photo extraite est sauvegardée atomiquement. Après un arrêt, relancer la même commande reprend les vecteurs ; une partition d'entraînement inachevée recommence à l'époque 0, tandis qu'une partition terminée est réutilisée. Ne pas lancer deux instances vers le même dossier. Un cache incompatible produit une erreur : utiliser un nouveau `--output`, sans effacer les anciennes expériences.

Option de mesure courte : ajouter `--benchmark-only`. Elle s'arrête après environ 50 nouvelles photos, conserve les vecteurs puis permet de poursuivre avec la commande normale.

## Fichiers à transmettre

- `results\comparison\summary.csv` et `results\comparison\results.json` : comparaison S/L complète.
- `results\projections\megadescriptor_l384\selection_seed_42.json` à `selection_seed_46.json` : choix d'époque et courbes de validation.
- `results\extraction_timing.csv` : temps observés.

Avec `--skip-small`, transmettre `results\summary.csv` et `results\results.json` à la place. Les gros NPZ ne sont pas nécessaires pour lire les résultats.

## Vérification locale

```powershell
python self_test.py
```

Tests sans téléchargement, avec un modèle simulé et des données synthétiques : extraction, reprise, cache, photos modifiées, vecteurs invalides, projection, séparation des identités, budget et calibration. Ils ne remplacent pas l'exécution réelle du modèle sur votre ordinateur et ne produisent aucun résultat scientifique sur votre corpus.
