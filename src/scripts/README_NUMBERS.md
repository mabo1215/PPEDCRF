# Updating paper numbers from scripts

The paper (`paper/main.tex`) uses placeholders that are filled by running the following scripts with your dataset and environment.

For the current retrieval and E4 revision, use the shared environment
`D:\source\.venv\Scripts\python.exe`. The paper retrieval tables are sourced
from `src/outputs/tomm_review_proxy12/` and
`src/outputs/tomm_review_proxy50/`, both at `sigma0=8`.

## 1. NCP privacy-to-privacy nearest-neighbor distance (d0, d1)

**Placeholders:** `\texttt{<d0>}`, `\texttt{<d1>}` (Section IV, Fig. labelMOT paragraph)

**Script:** `compute_privacy_neighbor_distances.py`

```bash
# Install deps if needed: opencv-python, torch, torchvision. Config default backbone is yolov11n (needs ultralytics).
# To avoid ultralytics/cv2 in embedder, use resnet18:
python src/scripts/compute_privacy_neighbor_distances.py --backbone resnet18 --max_clips 20

# Update paper (paper/main.tex) with computed d0, d1:
python src/scripts/compute_privacy_neighbor_distances.py --backbone resnet18 --update-tex
```

Optional: `--tex paper/main.tex` (default), `--N_max`, `--N_min` for class counts.

## 2. Top-k retrieval accuracy mean ± std (R@1, R@5, R@10)

**Placeholders:** `\texttt{<R1>}`, `\texttt{<R5>}`, `\texttt{<R10>}`, `\texttt{<R1std>}`, `\texttt{<R5std>}`, `\texttt{<R10std>}` (Section IV, Reproducibility / Threat model)

**Script:** `run_attack_multiseed.py`

Runs the retrieval attack on *protected* query frames under 3 noise seeds and reports mean ± std.

```bash
# Run with default config (needs data root, checkpoint, and dataset with train/val splits):
python src/scripts/run_attack_multiseed.py

# If your config uses a backbone that requires ultralytics (e.g. yolov11n), use resnet18:
python src/scripts/run_attack_multiseed.py --backbone resnet18

# Write mean ± std into paper/main.tex:
python src/scripts/run_attack_multiseed.py --update-tex
```

Optional: `--seeds 1234 1235 1236`, `--tex paper/main.tex`, `--max_query`, `--max_gallery`.

## 3. Figure path

The architecture figure path in `paper/main.tex` is set to `figs/architecture_of_solution.png`.  
A copy of the original file was created as `paper/figs/architecture_of_solution.png`.  
If you only have `artecture of solution.png`, copy it to `architecture_of_solution.png` in the same folder.

## 4. Controlled retrieval benchmark with local COCO / Digica pools

You can run the controlled benchmark while exposing additional distractor images
from local COCO and Digica roots. The benchmark still builds paired query/gallery
locations from monitoring clips, then appends external images as optional hard
gallery distractors when larger galleries are needed.

```bash
python src/scripts/run_controlled_retrieval_benchmark.py \
	--monitoring_root F:\\work\\datasets\\monitoring\\images \
	--coco_root C:\\work\\datasets\\Coco \
	--digica_root C:\\work\\datasets\\digica\\digica_v4.3 \
	--max_gallery 240 \
	--gallery_sizes 48 96 160 240
```

Notes:
- The script requires a SensNet checkpoint (`--checkpoint`, default `src/outputs/sensnet_final.pt`).
- Dataset source counts and external distractor usage are saved into `selection.json` and `summary.md`.

## 5. E4 same-image segmentation utility

The corrected E4 path preserves VOC palette-index masks, applies the official
DeepLabV3-ResNet50 preprocessing, and restores logits to the original image
size before mIoU computation. Run the smoke test and the manifest evaluation
with the shared CUDA environment:

```powershell
& D:\source\.venv\Scripts\python.exe src\scripts\evaluate_same_image_utility.py --smoke --device cuda
& D:\source\.venv\Scripts\python.exe src\scripts\evaluate_same_image_utility.py --mode manifest --manifest src\outputs\e4_audit\utility_manifest_segmentation.jsonl --config src\config\config.yaml --checkpoint src\outputs\sensnet_final.pt --output_dir src\outputs\tomm_same_image_utility_seg_v2 --variants full global_noise no_temporal no_ncp unary_only no_dcrf --resize_h 192 --resize_w 320 --seed 1234
```

The paper-facing summary is `src/outputs/tomm_same_image_utility_seg_v2/utility_summary.json`.

## 6. Regenerate proxy retrieval figures

```powershell
& D:\source\.venv\Scripts\python.exe src\scripts\regenerate_proxy_figures.py
```

This regenerates the fixed-budget privacy--utility and proxy12 robustness
figures under `paper/figs/`.

## 7. Public-data E1/E5 validation manifests

The public-data path is deliberately manifest-driven. Images and labels are
not copied into the repository, and a Kaggle mirror is acceptable only after
matching the official metadata, checksums, and license.

```powershell
# E1: official MSLS query/database metadata with true place clusters and GPS.
& D:\source\.venv\Scripts\python.exe src\scripts\build_msls_manifest.py `
    --msls_root F:\datasets\MSLS `
    --split train_val --cities amsterdam berlin `
    --subtask summer2winter --max_queries 200 --max_gallery 1000 `
    --output src\outputs\msls_e1\manifest.jsonl

# E5: KITTI-360 pose/semantic sequence-held-out attribution manifest.
& D:\source\.venv\Scripts\python.exe src\scripts\build_kitti360_unary_manifest.py `
    --kitti360_root F:\datasets\KITTI-360 `
    --sequences 0000 0002 --camera image_00 --stride 10 `
    --output src\outputs\kitti360_e5\manifest.jsonl

# E5: weakly supervised attribution consistency; no sensitivity-GT claim.
& D:\source\.venv\Scripts\python.exe src\scripts\validate_unary_attribution.py `
    --mode manifest --manifest src\outputs\kitti360_e5\manifest.jsonl `
    --root F:\datasets\KITTI-360 --output_dir src\outputs\kitti360_e5\attribution
```

The first command is consumed by `run_geotagged_vpr_benchmark.py`. The E5
validator writes per-query CSV and summary JSON, and fails the scientific gate
when the map is constant, the manifest is incomplete, or too few valid queries
remain.
