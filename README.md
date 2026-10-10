# Isles Pony Identification — Reproducibility

This repository contains experiment scripts, fixed evaluation manifests, human view annotations and recorded results for photographic pony identification. Photographs are distributed separately:

https://anonymous.4open.science/r/isles-pony-identification-E535/

## Contents and scope

- `equine_experiments_v1`: frozen HOG, ResNet50 and DINOv2 ViT-S descriptors.
- `megadescriptor_equine`, `miewid_equine`: frozen animal descriptors.
- `robustness_equine`: 128-dimensional projections, PCA controls, matched foreground and animal-erased interventions.
- `background_equine`: foreground extraction and segmentation audit.
- `background_transfer_equine`: reassigned background experiments.
- `dinov2_base_equine`, `megadescriptor_large_equine`: larger backbone comparisons and adaptation.
- `human_views_equine`: human-annotated view analysis; use root `annotations_vues.csv` explicitly.
- `wildfusion_equine`: global/local matching experiment.
- `equine_additional`: last Transformer block plus final normalization adaptation, paired identity bootstrap and CIA–UFA analysis.

Directories prefixed `recorded_` contain evidence from the original runs. They are deliberately separate from newly generated `results`, to avoid using historical completion markers with absent checkpoints. Historical JSON path fields use placeholders; they are provenance records, not executable configurations. Original module documentation is retained and may describe older runs; use the commands below as the entry point.

Photos, model weights, embeddings, masks and binary caches are excluded from this lightweight package. Therefore this package supports direct reanalysis of exported predictions, and provides code for recomputation from images. It is not a self-contained offline or bitwise-reproducible archive. Downloads from upstream model repositories are necessary. The recorded weights hashes and package versions must be checked when rerunning; upstream main branches are not pinned by these original scripts.

## Protocol

The audited collection contains 1,549 photographs of 122 identities. Three near-duplicate images are excluded by the fixed manifests, leaving 1,546 images: 1,502 for 115 main identities and 44 for seven sparse identities. Five overlapping identity partitions use seeds 42–46, with 70 training, 20 validation and 25 test identities. Galleries of 1/3/5 images are nested; queries stay fixed within each episode.

Keep the shipped manifests unchanged. Image hashes are checked against them. Do not regenerate partitions merely to accommodate modified images. No acquisition-session split is certified. Partition standard deviations are descriptive. The paired identity bootstrap is conditional on recorded models and episodes; it does not account for retraining or unresolved session dependence.

## Setup

Run commands from this repository's root. The corpus directory must directly contain `au_moins_10_photos` and `moins_de_10_photos`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:EQUINE_PROJECT_ROOT = (Get-Location).Path
$env:EQUINE_DATA_ROOT = "PATH_TO_DATASET"
python -m pip install -r requirements_analysis.txt
```

The path variable is optional when scripts receive `--base`; its default is `dataset` relative to the working directory. No personal machine path is required.

For image extraction and training, install the requirement file associated with each module. Install a compatible PyTorch/torchvision pair for your device. The original per-module requirement files specify ranges, not a complete lockfile. `recorded_package_versions.json` lists versions recovered from the supplied results; different experiments may have used different environments. Do not interpret their union as one tested environment.

## 1. Verify the new statistical results without images or training

```powershell
python equine_additional/self_test.py
python equine_additional/analyze_predictions.py --predictions equine_additional/recorded_reference_results/predictions_reference.csv --method-a dinov2 --method-b projection_triplet --replicates 10000 --seed 2027 --output verification/projection
python equine_additional/analyze_predictions.py --predictions equine_additional/recorded_reference_results/predictions_reference.csv equine_additional/recorded_last_block_results/predictions_last_block.csv --method-a projection_triplet --method-b dinov2_last_block --replicates 10000 --seed 2027 --output verification/last_block
```

Expected five-reference paired Top-1 differences (B minus A, percentage points):

| Contrast | Difference | Identity-bootstrap 95% interval |
|---|---:|---:|
| Linear projection minus frozen DINOv2 ViT-S | 16.3453 | [12.8042, 19.9428] |
| Last block + norm minus linear projection | 1.0856 | [-0.4983, 2.7748] |

Frozen, projected and partially adapted Top-1 means are approximately 73.40%, 89.75% and 90.83%. The second interval does not establish superiority or equivalence. The same scripts generate error decompositions and CIA–UFA figures. Curves are retrospective test envelopes; their points use the original validation-calibrated decisions. They are not prospective threshold selection.

## 2. Recompute descriptors

Install `equine_experiments_v1/requirements_deep.txt` (which includes HOG requirements), then:

```powershell
python equine_experiments_v1/run_experiments.py --model hog --base "$env:EQUINE_DATA_ROOT"
python equine_experiments_v1/run_experiments.py --model resnet50 --base "$env:EQUINE_DATA_ROOT"
python equine_experiments_v1/run_experiments.py --model dinov2 --base "$env:EQUINE_DATA_ROOT"
```

Install each corresponding module's requirement file before running:

```powershell
python megadescriptor_equine/run_megadescriptor.py --base "$env:EQUINE_DATA_ROOT"
python miewid_equine/run_miewid.py --base "$env:EQUINE_DATA_ROOT"
python background_equine/run_background.py --base "$env:EQUINE_DATA_ROOT" --original-embeddings equine_experiments_v1/results/embeddings_dinov2.npz
```

The last command regenerates segmentation and foreground embeddings. Original segmentation audit records remain in `background_equine/recorded_results`. Model and image hashes are recorded by extraction scripts. A changed upstream model can prevent later compatibility checks; do not disable those checks to force a score.

## 3. Recompute projections and context controls

Install `robustness_equine/requirements.txt`:

```powershell
python robustness_equine/run_robustness.py --project-root . --base "$env:EQUINE_DATA_ROOT" --epochs 200 --steps-per-epoch 10
```

This discovers generated caches under the project. Incompatible duplicate caches cause an error; select them explicitly with repeated `--embedding MODEL=PATH` arguments if needed. Projection training uses only training identities, and selection uses validation identities. Context controls use a matched segmentation-eligible cohort, distinct from the original benchmark cohort.

Install the requirements for each next module:

```powershell
python background_transfer_equine/run_background_transfer.py --project-root . --robustness-results robustness_equine/results --base "$env:EQUINE_DATA_ROOT"
python human_views_equine/run_human_views.py --annotations annotations_vues.csv --project-root . --robustness-results robustness_equine/results --original-embeddings equine_experiments_v1/results/embeddings_dinov2.npz
python dinov2_base_equine/run_dinov2_base.py --base "$env:EQUINE_DATA_ROOT" --small-embeddings equine_experiments_v1/results/embeddings_dinov2.npz --epochs 200
python megadescriptor_large_equine/run_megadescriptor_large.py --base "$env:EQUINE_DATA_ROOT" --epochs 200
python wildfusion_equine/run_wildfusion.py --base "$env:EQUINE_DATA_ROOT"
```

Background reassignment confounds context changes and composition artefacts. Animal-erased controls retain the mask silhouette. These limitations apply to interpretation, even when numerical reproduction succeeds.

## 4. Recompute the final DINOv2 comparisons

Install `equine_additional/requirements.txt`:

```powershell
python equine_additional/export_predictions.py --embeddings equine_experiments_v1/results/embeddings_dinov2.npz --projections robustness_equine/results/projections/dinov2 --output equine_additional/reference_results
python equine_additional/finetune_last_block.py --base "$env:EQUINE_DATA_ROOT" --embeddings equine_experiments_v1/results/embeddings_dinov2.npz --epochs 200 --steps 10 --learning-rate 1e-5 --batch-size 8 --output equine_additional/last_block_results
```

Repeat the analysis commands in step 1, replacing `recorded_reference_results` and `recorded_last_block_results` with the newly generated directory names.

Last-block adaptation updates 1,776,000 parameters, versus 49,152 for the 128D projection (36.13 times fewer parameters for the projection). This ratio is not a compute-time or memory ratio. The partial adaptation outputs a normalized 384D descriptor. Recorded selected epochs are 55/60/80/80/45, versus 75/60/65/65/55 for the projection. The original last-block run used CPU; CUDA may introduce numerical differences. Cached prefix tokens require approximately 610 MB, in addition to model and training memory.

## Validation of this package

See `VALIDATION.md`. Code parsing, available synthetic checks and reanalysis are checked locally. Full image extraction and training are not rerun in this packaging environment. Recorded scientific results are preserved separately from newly generated verification files.

## Licensing

No new code or dataset licence is assigned by this packaging step. Dataset permission and terms are documented in the separate corpus repository. Third-party model code and weights retain their respective terms.
