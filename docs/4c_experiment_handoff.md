# PPEDCRF ACM TOMM Revision-Cycle Handoff to Claude Code

**Revision cycle:** TOMM review letter dated 2026-08-30
**Repository:** `mabo1215/PPEDCRF`
**Current local state:** runnable review scripts and paper text updates are prepared.
**Paper submodule commit:** `a0f69be` (`PPEDCRF_overleaf` `main`, pushed 2026-08-30).
**Parent integration commit:** will be recorded here after the parent repository push.
**Remote target:** the `4c 3090` entry in `C:\source\.env`
**Remote start status:** blocked pending a successful TCP/SSH connection and host-key confirmation.

## What is ready

The following scripts are in the pushed revision:

- `src/scripts/run_tomm_review_proxy.py` — controlled proxy ablation, effective perturbation MSE, PSNR/SSIM, per-query rank, correct-versus-hardest-negative margin, unary-only/no-DCRF variants, and optional ResNet18 attacker-aware feature suppression.
- `src/scripts/run_geotagged_vpr_benchmark.py` — manifest-driven true-place VPR evaluation with place-level Top-k and cross-condition metadata.
- `src/scripts/evaluate_same_image_utility.py` — same-image detector mAP and segmentation mIoU evaluation for a label-backed manifest.
- `src/scripts/validate_sensnet_provenance.py` — predictor architecture/checkpoint audit and non-constant-output probe.
- `src/scripts/generate_retrieval_case_study.py` — deterministic per-query case-study selection and rendering after a complete proxy run.

Local CUDA smoke tests passed for all three evaluation paths. Smoke outputs
are engineering checks only and must not be copied into `paper/`.

## Commit and checkout gate

Before launching any long run, Claude Code must record:

1. the pushed commit SHA in this file and in `docs/progress.md`;
2. the SHA-256 of `src/outputs/sensnet_final.pt` and the remote copy;
3. the Python/Torch/CUDA versions and the output root;
4. the exact manifest or monitoring selection JSON used by the run.

The checkpoint is currently ignored by Git and must be transferred or already
available on 4c. Do not substitute a different checkpoint without recording
its provenance. The local audit found `mask_root=null` in the checkpoint
metadata and near-constant monitoring sensitivity maps; this is a scientific
gate, not a cosmetic warning.

## 4c proxy launch

Run only after the free-VRAM and host-key gates pass. Use a fresh remote output
root and preserve all logs:

```bash
cd /path/to/PPEDCRF
git fetch origin
git checkout <PUSHED_COMMIT_SHA>
python -m py_compile \
  src/scripts/run_tomm_review_proxy.py \
  src/scripts/run_geotagged_vpr_benchmark.py \
  src/scripts/evaluate_same_image_utility.py \
  src/scripts/validate_sensnet_provenance.py

python src/scripts/validate_sensnet_provenance.py \
  --checkpoint src/outputs/sensnet_final.pt \
  --output_dir /data1/PPEDCRF/tomm_revision_20260830/provenance

python src/scripts/run_tomm_review_proxy.py --mode proxy \
  --checkpoint src/outputs/sensnet_final.pt \
  --monitoring_root /path/to/monitoring/images \
  --num_queries 12 --pair_pool_size 240 --max_gallery 48 \
  --gallery_sizes 12 24 48 \
  --seeds 1234 1235 1236 \
  --backbones resnet18 resnet50 vgg16 clip_vitb32 clip_vitl14 \
              cosplace mixvpr patchnetvlad \
  --output_dir /data1/PPEDCRF/tomm_revision_20260830/proxy12

python src/scripts/run_tomm_review_proxy.py --mode proxy \
  --checkpoint src/outputs/sensnet_final.pt \
  --monitoring_root /path/to/monitoring/images \
  --num_queries 50 --pair_pool_size 600 --max_gallery 100 \
  --gallery_sizes 50 75 100 \
  --seeds 1234 1235 1236 \
  --backbones resnet18 resnet50 vgg16 clip_vitb32 clip_vitl14 \
              cosplace mixvpr patchnetvlad \
  --output_dir /data1/PPEDCRF/tomm_revision_20260830/proxy50
```

The first complete proxy run is the minimum required for E2/E6/E7. A run is
complete only when `selection.json`, `run_metadata.json`, `per_query.csv`, and
`summary.csv` exist, all requested backbone/seed/gallery cells are present,
and every numeric margin/energy field is finite. If a process disappears,
write a failure status, retain the partial JSONL/CSV/log, stop the queue, and
do not retry blindly.

## Geotagged VPR launch (E1)

Prepare a JSONL manifest as specified in `src/scripts/run_geotagged_vpr_benchmark.py`.
Each query must have a `place_id` and a gallery list containing at least one
same-place image and at least one different-place negative. Query and gallery
paths must be disjoint. Then run:

```bash
python src/scripts/run_geotagged_vpr_benchmark.py --mode geotagged \
  --manifest /path/to/geotagged_vpr_manifest.jsonl \
  --checkpoint src/outputs/sensnet_final.pt \
  --backbones resnet18 cosplace mixvpr patchnetvlad \
  --seeds 1234 1235 1236 \
  --output_dir /data1/PPEDCRF/tomm_revision_20260830/geotagged
```

Do not call the existing monitoring proxy a geotagged benchmark. If no
compliant public dataset, manifest, or labels are available on 4c, mark E1
blocked and update the paper's limitation rather than inventing place labels.

## Same-image utility launch (E4)

Prepare a label-backed JSONL manifest with image paths, detection boxes/labels,
and optionally segmentation-mask paths. Use frozen pretrained detector and
segmenter weights. The resulting mAP/mIoU is utility evidence only:

```bash
python src/scripts/evaluate_same_image_utility.py --mode manifest \
  --manifest /path/to/utility_manifest.jsonl \
  --checkpoint src/outputs/sensnet_final.pt \
  --variants full global_noise no_temporal no_ncp unary_only no_dcrf \
  --output_dir /data1/PPEDCRF/tomm_revision_20260830/utility
```

Do not use legacy training curves as a substitute for same-image mAP/mIoU.

## Monitoring and paper-writeback rules

Claude Code should poll each active run every 10 minutes and append status to
the remote run directory. It should update `docs/progress.md` and this file
with completed rows, failures, timestamps, and checksums. No output may be
copied into `paper/` until the corresponding completion gate passes. In
particular:

- smoke, partial, failed, or checkpoint-degenerate outputs are not paper evidence;
- E1 supports a true-geolocation claim only after the manifest gate passes;
- E3 attacker-aware output remains an adaptive-attacker diagnostic unless
  same-utility and same-gallery comparability passes;
- E4 updates utility claims only when labels and frozen models are verified;
- MixVPR adverse transfer must be retained and reported, not filtered out.

## Current blocker

The configured 4c/3090 endpoint from `C:\source\.env` did not accept a TCP
connection during the local handoff attempt, and the older SSH alias also has
a changed host fingerprint. Confirm the current endpoint/fingerprint or repair
the SSH configuration before launching. Do not disable host-key verification.
