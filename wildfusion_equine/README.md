# Variante WildFusion à deux composantes — corpus des Isles

## Lancer sur Windows

Décompresser toute l'archive. Garder ensemble les trois scripts et le dossier `splits`. Ouvrir un terminal dans ce dossier, puis :

```powershell
python -m pip install -r requirements_wildfusion.txt
python run_wildfusion.py --base "${EQUINE_DATA_ROOT}"
```

Les modèles sont téléchargés au premier lancement. Le programme utilise CUDA si disponible, sinon le CPU. Aucune annotation supplémentaire. Les photos restent sur la machine ; seules les ressources des modèles sont téléchargées.

Le plan comporte **197 671 comparaisons locales uniques**, incluant calibration et évaluation. Sur CPU, prévoir un calcul nettement plus long que MiewID, potentiellement plusieurs heures selon le matériel. Après les 100 premières comparaisons, une estimation du temps restant s'affiche et s'actualise. Le nombre peut être vérifié sans télécharger de modèle :

```powershell
python run_wildfusion.py --plan
```

Pour réutiliser les vecteurs MegaDescriptor déjà calculés, indiquer le chemin réel du fichier :

```powershell
python run_wildfusion.py --base "${EQUINE_DATA_ROOT}" --mega-cache "C:\chemin\embeddings_megadescriptor.npz"
```

Sans cet argument, le script extrait et sauvegarde lui-même les vecteurs MegaDescriptor. Il ne nécessite ni les vecteurs MiewID ni les vecteurs DINOv2.

Si la mémoire GPU est insuffisante pour MegaDescriptor : ajouter `--batch-size 1`. Pour forcer le CPU : ajouter `--device cpu`. Utiliser le même appareil, les mêmes dépendances et le même dossier pour reprendre un calcul local.

## Reprise

Chaque photo possède un cache de points et descripteurs locaux. Chaque comparaison est enregistrée dans `results/local_cache/pairs.sqlite` immédiatement. Après Ctrl+C ou un arrêt, relancer exactement la même commande : seules les paires manquantes sont calculées. Ne pas supprimer `results/local_cache`. La première extraction MegaDescriptor doit avoir terminé pour que son propre cache soit réutilisable.

Le script refuse un cache local si les modèles, leur code, les paramètres, l'appareil ou les versions des dépendances diffèrent. Dans ce cas, restaurer l'environnement initial ou utiliser un nouveau dossier avec `--output results_nouveau`.

## Configuration fixée avant le calcul

Cette expérience est une **implémentation indépendante d'une variante à deux composantes inspirée de WildFusion**, et non une reproduction de l'ensemble complet publié. Elle combine :

1. MegaDescriptor-S-224, figé, avec le même prétraitement que notre baseline : image entière, padding carré gris 127, 224 pixels bicubiques, normalisation ImageNet, vecteur L2.
2. SuperPoint + LightGlue officiels, figés : image RGB corrigée EXIF, redimensionnement direct 512 × 512 bilinéaire sans padding, maximum 1024 points, seuil de correspondance 0,1, adaptativité LightGlue par défaut (0,95/0,99). Score local : nombre de correspondances acceptées. Pas de vérification géométrique supplémentaire. Pour un cache indépendant du sens d'appel, les paires sont toujours traitées avec l'indice le plus petit en première image.

Pour chaque graine, tirer 1000 paires positives et 1000 paires négatives uniques exclusivement dans les 70 identités d'apprentissage. Tirage reproductible, uniforme parmi les identités avant tirage des images ; absence de paires d'une image avec elle-même. Ajuster une régression isotone pour chaque score, puis moyenner leurs scores calibrés à poids égaux. Les poids, paramètres, dimensions et tailles de lots ne sont pas sélectionnés sur les résultats de test.

Le score d'identité est le maximum des scores fusionnés de ses images de référence. Toutes les paires nécessaires requête/référence sont calculées : **aucune shortlist ni approximation**, aucun query expansion. Les scores calibrés sur un échantillon équilibré positif/négatif ne sont pas des probabilités opérationnelles correspondant à la fréquence réelle des identités.

Quatre contrôles : `megadescriptor_raw`, `global_calibrated`, `local_calibrated`, `wildfusion_two_streams`. La calibration isotone est monotone mais crée des égalités de scores ; le contrôle global calibré permet de mesurer cet effet.

## Évaluation

Même corpus audité de 1546 photos, mêmes cinq partitions (42 à 46), mêmes galeries 1/3/5 et requêtes que les expériences précédentes. Présence et SHA-256 des photos contrôlés. Identités disjointes dans chaque partition : apprentissage 70, validation 20, test 25. Les sept identités peu photographiées sont réservées comme inconnues externes.

- Recherche parmi les 25 poneys de test (`reid_25`), Top-1/3/5 macro et micro.
- Reconnaissance ouverte (`open_test`), dix identités connues et quinze inconnues.
- Sept identités inconnues externes (`external_7`).
- Seuil de rejet propre à chaque méthode, partition et taille de galerie, choisi uniquement sur les inconnus de validation pour au plus 5 % de fausses acceptations empiriques par photo. Ce plafond n'est pas garanti sur le test ; métriques de test macro par poney.
- Pas de score de la fusion calibrée parmi les 115, car sa calibration aurait utilisé une partie des identités évaluées.

Les partitions se recouvrent entre graines. Les intervalles bootstrap par graine sont conditionnels aux galeries ; l'écart-type entre graines n'est pas un intervalle de confiance indépendant. Pas de séparation par séance, âge, saison ou côté. La méthode locale peut aussi apparier des éléments de décor : elle ne résout pas la question du fond. Extension exploratoire décidée après inspection des baselines.

## Fichiers à envoyer

Dans `results`, envoyer :

- `results_wildfusion.json` (scores et métadonnées) ;
- `summary_wildfusion.csv` (moyennes et écarts-types) ;
- `calibrations_wildfusion.json` (paires et fonctions de calibration).

`predictions_wildfusion.csv` permet l'analyse des erreurs. `scores_wildfusion.json` est une copie des scores seuls. Conserver le cache local pour éviter de refaire les comparaisons. Les sources LightGlue effectivement installées, les versions des dépendances, les empreintes des poids et des partitions sont enregistrées. L'installation récupère la branche officielle courante de LightGlue ; conserver l'environnement et les métadonnées pour reproduire cette exécution.

## Vérifications et sources

Syntaxe, plan de paires, absence de calibration sur les identités validation/test, contrôle global et évaluation des quatre méthodes vérifiés avec vecteurs et scores locaux synthétiques. L'inférence réelle SuperPoint/LightGlue n'a pas été exécutée ici ; elle reste à lancer sur les photos Windows.

Documentation WildFusion (calibration et fusion) : https://wildlifedatasets.github.io/wildlife-tools/wildfusion/

Implémentation et API officielles SuperPoint/LightGlue : https://github.com/cvg/LightGlue
