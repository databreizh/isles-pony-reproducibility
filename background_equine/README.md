# DINOv2 : ablation automatique du fond

## Lancer sous Windows

Décompresser tout le dossier, garder `evaluation.py` et `splits` à côté du script, ouvrir un terminal dans ce dossier :

```powershell
python -m pip install -r requirements_background.txt
python run_background.py --base "${EQUINE_DATA_ROOT}"
```

Le modèle de segmentation et DINOv2 sont téléchargés au premier lancement. Les photos restent sur votre machine. CPU si CUDA indisponible. Le script segmente les 1546 photos retenues, extrait DINOv2 et réalise les évaluations avec et sans traitement du fond.

Pour éviter de réextraire les vecteurs originaux, placer votre `embeddings_dinov2.npz` à côté du script ou préciser son chemin :

```powershell
python run_background.py --base "${EQUINE_DATA_ROOT}" --original-embeddings "C:\chemin\embeddings_dinov2.npz"
```

Si la mémoire GPU est insuffisante : `--batch-size 1`. Pour imposer le CPU : `--device cpu`.

## Ce qui est comparé

Le modèle choisi avant l'expérience est YOLO11n-seg préentraîné, figé, classe `horse`, entrée 640, confiance minimale 0,25. Ce n'est pas une variante SAM : le modèle fournit directement des masques de chevaux, sans annotation de boîte ou de point.

Pour chaque image, correction EXIF et RGB ; retenir le masque du cheval couvrant la plus grande surface, parmi les masques couvrant entre 1 % et 98 % de l'image. En cas d'égalité, priorité à la confiance puis au premier indice. Les seuils géométriques sont des filtres grossiers, pas une certification de qualité.

Tous les pixels hors masque sont remplacés par RGB (127,127,127). **Aucun recadrage ni changement d'échelle du poney avant le prétraitement commun** : même canevas original, même padding carré, même redimensionnement bicubique 224, même normalisation ImageNet. Les images traitées mises en cache sont les entrées RGB de 224 pixels, pour limiter l'espace disque.

Si aucun cheval ou aucun masque acceptable n'est trouvé, conserver l'image originale et enregistrer explicitement l'échec. Le résultat principal est donc celui du **pipeline automatique avec repli**, et ne doit pas être nommé « toutes les photos sans fond » si la couverture est incomplète. Aucune image ni aucun poney n'est retiré des épisodes.

Les portraits de tête peuvent être mal détectés. Un masque retenu peut être incomplet ou correspondre au mauvais animal lorsqu'il y en a plusieurs. Les parties occultées, le harnachement et le cavalier peuvent également affecter le masque. Des éléments de fond peuvent rester. Contrôler les aperçus avant d'interpréter les scores ; ne pas corriger des masques ou changer les seuils en fonction des résultats de test.

## Résultats à envoyer

Dans `results` :

- `results_background.json` : les deux méthodes et leurs métadonnées ;
- `summary_background.csv` : moyennes et écarts-types ;
- `segmentation_summary.json` : couverture totale et par poney ;
- `segmentation_audit.csv` : statut, surface et détections par photo ;
- `embeddings_dinov2_foreground.npz` : vecteurs à réutiliser pour tester ensuite les matrices de projection déjà apprises.

Le dossier `apercus` contient jusqu'à 48 comparaisons original/traité, choisies avec graine 42 parmi les masques retenus, les échecs et les images avec plusieurs chevaux. Les catégories peuvent se recouvrir. Le nom de fichier et l'indice du CSV permettent de retrouver l'original. Un contrôle rapide des aperçus demande une inspection visuelle, pas l'annotation du corpus entier ; il ne remplace pas un audit exhaustif.

## Reprise et traçabilité

Chaque image segmentée est enregistrée individuellement avec son statut et une empreinte. Relancer la même commande reprend les images manquantes. Après l'extraction complète, les vecteurs sont eux aussi réutilisés. Une extraction DINOv2 interrompue avant création de son NPZ reprend depuis le début. Les originaux sont seulement lus, jamais modifiés.

Le cache de segmentation est refusé si paramètres, poids, versions ou appareil changent ; utiliser un nouveau `--output` dans ce cas. Garder les caches pour reprendre. Les empreintes des poids et les versions des dépendances sont enregistrées. Le script exige des poids DINOv2 identiques à ceux des vecteurs originaux. Les installations et le code des dépôts évoluent : conserver l'environnement utilisé et les métadonnées.

## Protocole et limites

Mêmes cinq partitions, graines 42 à 46, mêmes galeries emboîtées 1/3/5 et requêtes. Scores parmi 115 et 25 identités ; reconnaissance ouverte et sept inconnus externes. Seuils de rejet recalibrés séparément pour chaque méthode sur les mêmes inconnus de validation, plafond empirique de 5 % par photo. Pas d'entraînement ni ajustement de la segmentation sur le corpus.

La comparaison sur l'ensemble conserve les échecs et doit toujours être accompagnée du taux de couverture. Comparer uniquement les photos segmentables sélectionnerait un sous-ensemble plus facile et demanderait une évaluation appariée distincte. Les deux scores sont directement appariés par partition et épisode.

Une baisse après masquage peut refléter une dépendance au contexte, des erreurs de masque ou le changement de distribution des images. Elle ne prouve pas à elle seule une fuite par le fond. Une hausse ne prouve pas une généralisation entre séances, âges ou saisons. Le protocole reste exploratoire, sans séparation par séance et avec recouvrement entre partitions.

## Vérifications

Syntaxe, commande d'aide, sélection automatique de masques sur cas synthétiques, fond gris, conservation du canevas et identité du prétraitement en cas de repli vérifiés. L'inférence réelle YOLO/DINOv2 doit être lancée sur les photos Windows et n'a pas été exécutée ici.

Sources officielles :
- https://docs.ultralytics.com/models/yolo11/
- https://docs.ultralytics.com/tasks/segment/
- https://github.com/facebookresearch/dinov2
