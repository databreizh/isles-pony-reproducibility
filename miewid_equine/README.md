# MiewID-msv3 sur le corpus des Isles

## Lancement Windows

Décompresser toute l'archive : conserver `run_miewid.py`, `evaluation.py` et le dossier `splits` ensemble. Ouvrir un terminal dans ce dossier, avec Python 3.10 à 3.12 (64 bits recommandé).

```powershell
python -m pip install -r requirements_miewid.txt
python run_miewid.py --base "${EQUINE_DATA_ROOT}"
```

Le premier lancement télécharge le modèle et son code depuis Hugging Face, et éventuellement les poids du backbone chargés par ce code officiel. Internet est nécessaire. CPU par défaut en l'absence de CUDA. L'extraction peut être longue sur CPU ; la progression s'affiche. Aucun entraînement ni annotation supplémentaire.

Si la mémoire GPU est insuffisante :

```powershell
python run_miewid.py --batch-size 1
```

Pour forcer le CPU : `python run_miewid.py --device cpu`.

## Fichiers à récupérer

Dans `results` :
- `results_miewid.json` : scores des cinq partitions et métadonnées ;
- `summary_miewid.csv` : moyennes et écarts-types, séparateur point-virgule ;
- `embeddings_miewid.npz` : vecteurs pour d'éventuelles fusions ;
- `predictions_miewid.csv` : prédictions détaillées.

Envoyer les trois premiers fichiers pour l'analyse. Les vecteurs sont mis en cache : après une extraction terminée, relancer réutilise ces vecteurs et recalcule l'évaluation. `--recompute` force une nouvelle extraction. Une extraction interrompue avant création du cache reprend depuis le début.

## Protocole fixé

Modèle `conservationxlabs/miewid-msv3`, révision `4f1d7f2` résolue en SHA complet et consignée dans le JSON ; empreinte des poids et versions des dépendances également enregistrées. Embeddings gelés de 2152 dimensions, normalisés L2 ; score d'identité = meilleur cosinus parmi les références.

Prétraitement de la fiche modèle : correction EXIF, RGB, redimensionnement direct 440 × 440 bilinéaire, normalisation ImageNet. Aucun recadrage, segmentation ou padding. Les partitions et photos sont identiques aux expériences précédentes, mais le prétraitement diffère de leur entrée carrée avec padding à 224 pixels. La comparaison évalue donc les pipelines préentraînés avec cette différence explicite, pas les seuls backbones à résolution constante.

1546 images retenues dans l'audit. Les autres photos restent sur disque mais ne sont pas utilisées. Le script vérifie présence et SHA-256 de chaque image utilisée et s'arrête si le corpus a changé.

Cinq partitions existantes, graines 42 à 46 ; galeries de 1, 3 et 5 images ; requêtes inchangées pour ces trois tailles. Évaluations parmi 115 identités (`closed_115`) et parmi 25 identités de test (`reid_25`). Reconnaissance ouverte avec dix connus et quinze inconnus (`open_test`), plus les sept identités peu photographiées comme inconnus externes (`external_7`). Seuil calibré sur les inconnus de validation à au plus 5 % de fausses acceptations empiriques par photo ; ce plafond n'est pas garanti sur le test. Résultats macro par poney et micro disponibles dans le JSON. Les partitions se recouvrent entre graines ; leur écart-type n'est pas un intervalle de confiance indépendant.

Pas de contrôle par séance, âge, saison ou vue. Ces expériences sont une extension exploratoire du benchmark. La fiche MiewID liste des données équines dans ses résultats : ne pas présenter le modèle comme n'ayant jamais vu l'espèce cheval au préentraînement. L'absence de recouvrement avec les images du corpus privé n'a pas été vérifiée auprès des auteurs.

Sources officielles :
- https://huggingface.co/conservationxlabs/miewid-msv3/blob/main/README.md
- https://huggingface.co/conservationxlabs/miewid-msv3/blob/main/modeling_miewid.py

## Vérifications du dossier

Syntaxe Python, interface de lancement, cohérence des cinq partitions et évaluation avec vecteurs synthétiques vérifiées. L'inférence réelle MiewID sur les photos locales Windows doit être exécutée sur votre machine ; elle n'a pas été réalisée ici.
