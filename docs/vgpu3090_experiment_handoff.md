# vGPU 3090 TOMM Revision-Cycle Handoff (2026-08-30, session interrupted for local host restart)

The user is restarting the **local** machine (the one running Claude Code), not
the remote GPU host. This file records exact state so the next session can
resume without re-deriving anything. Read this before touching vGPU 3090 or
4c again.

## Host status summary

- **4c 3090** (`ssh mabo1215@10.126.126.1`, from `C:\source\.env`): SSH works,
  but the CUDA driver stack is broken host-wide — `cuInit(0)` returns 999
  (`CUDA_ERROR_UNKNOWN`) for every process, reproducible across two different
  Python/venv environments, and `nvidia-smi` itself fails on GPU3
  (`Unable to determine the device handle: Unknown Error`). GPU0-2 report
  idle/healthy via `nvidia-smi` but no CUDA context can be created on any of
  them. This needs a driver module reload or host reboot (root required); it
  is a **shared host with 15+ other logged-in users' sessions**, so do not
  attempt `rmmod`/`modprobe nvidia*`/reboot without the user explicitly
  re-confirming in the moment and being aware of the shared-host impact. Per
  user decision on 2026-08-30, work moved to vGPU 3090 instead; 4c is not
  otherwise touched.

- **vGPU 3090** (`ssh -p 22766 root@connect.westd.seetacloud.com`, password
  in `.env` but a passwordless key already works — do not hardcode the
  password anywhere; always read `.env` fresh). This is an AutoDL-style
  rented instance (`autodl-container-5ea9tapmbb-6a673b83`), single GPU
  reporting as "RTX 3090" with **49152 MiB** (a vGPU/virtualized allocation,
  not a physical 24GB card), driver 580.82.09 / CUDA 13.0. Confirmed powered
  on and reachable.

  **Critical open issue**: the network path from this local machine to this
  specific vGPU 3090 instance is severely bandwidth-constrained right now —
  measured throughput has been in the tens of KB/s (a 2.2GB transfer was
  projected at 16-40+ hours by rsync's own ETA), and even trivial SSH
  commands sometimes take 15-20+ seconds. This looked bidirectional (both
  upload and download directions were equally slow in raw `dd`-over-`ssh`
  tests) and is NOT a port/credential problem — TCP connects fine, SSH auth
  succeeds via existing key. Cause is unconfirmed: could be local-machine/ISP
  routing congestion (the user is restarting the local host now, which may
  or may not change this), or the vGPU instance's own throttling. **Re-test
  basic throughput before resuming any large transfer** (see "How to
  re-verify" below) rather than assuming it is still bad or assuming it is
  fixed.

## What is already done on vGPU 3090

Remote path: `/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF`
(`/root/autodl-tmp` is the 1.4TB persistent data disk; `/` is a small 30GB
container overlay — **always** put venvs, caches, and data under
`/root/autodl-tmp/`, never under plain `/root/`).

1. **Repo cloned** at commit `19f0752908f52fba00c6e353fe81f38a2faf5441`
   (== `origin/main` as of 2026-08-30 ~21:20 NZST). This already includes a
   real bugfix made this session (see "Code changes already pushed" below).
2. **Third-party submodules manually cloned** — `src/third_party/{CosPlace,
   MixVPR,Patch-NetVLAD}` are gitlinks in the parent repo's tree but have
   **no `.gitmodules` entry** (informal submodules, `git submodule update
   --init` fails with "No url found"). They were cloned directly and
   checked out to the exact recorded commits:
   - CosPlace: `https://github.com/gmberton/CosPlace.git` @
     `52b56e95ea62245281281f3bafd7b9390d19a0fd`
   - MixVPR: `https://github.com/amaralibey/MixVPR.git` @
     `4043915cef24818003ece1a8112bc8a24e69abe0`
   - Patch-NetVLAD: `https://github.com/QVPR/Patch-NetVLAD.git` @
     `cddb9ae39391bc598dfb9e28ef762f1c084ff0cd`
     (only `patchnetvlad/pretrained_models/mapillary_WPCA4096.pth.tar`,
     327MB, is actually needed at runtime — that directory is gitignored
     inside the submodule itself and was never in the clone; it still needs
     to be copied in from local, see below.)
3. **Disk cleanup on `/`**: this vGPU instance had ~24GB of leftover files
   from unrelated prior projects (`Corey_Transformer`, `miniconda3`,
   `data`, `wheels_corey_h800`, `pathc_runs_system`, `flash-attention`,
   `wacv2027`, `tamperfilter_ijmlc_vgpu`, `bench3090`) that had filled the
   30GB root overlay to 92% full. **Deleted per explicit user confirmation**
   on 2026-08-30. `/` is now at ~23GB free (was 2.5GB).
4. **pip reconfigured**: `pip config set global.cache-dir
   /root/autodl-tmp/cache/pip` and `pip config set global.index-url
   https://mirrors.aliyun.com/pypi/simple/` (both written to
   `/root/.config/pip/pip.conf`, persist across SSH sessions). The very
   first venv-setup attempt filled `/root/.cache/pip` (3.7GB, on the small
   overlay) before failing with `OSError: [Errno 28] No space left on
   device` — that cache was deleted; this is now fixed at the config level
   so it will not recur.
5. **venv created** at
   `/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/.venv_ppedcrf`.
6. **Package install launched in a detached `screen` session** (survives SSH
   disconnects) named `setup_env`, logging to
   `/root/autodl-tmp/setup_env.log`:
   ```
   source .venv_ppedcrf/bin/activate
   export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
   pip install --upgrade pip
   pip install torch torchvision
   pip install opencv-python-headless numpy "scikit-image>=0.23,<0.27" tqdm pyyaml transformers
   python3 -c "import torch; print(1234, torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   echo SETUP_ENV_DONE
   ```
   **Status at interruption: unknown** — a background poller was watching
   for `SETUP_ENV_DONE` in the log but had not seen it yet when the local
   host restart was requested. First action on resume: `ssh -p 22766
   root@connect.westd.seetacloud.com "cat /root/autodl-tmp/setup_env.log"`
   to see whether it finished, is still running, or died (in which case
   check `screen -ls` for whether `setup_env` is still attached/running,
   and re-launch only the missing tail of the install if the venv itself is
   intact).

## What is NOT yet on vGPU 3090 (data transfer never completed)

None of the actual experiment data made it to the remote host — the
monitoring-image transfer was still crawling (~5.6MB of 2.2GB after a long
wait, ETA 16-40+ hours at observed throughput) when this session paused.
**Re-test throughput first** (see below); if it's still this bad, transferring
even the shrunk 2.2GB subset this way is not viable and a different approach
(smaller subset still, or a different host, or an intermediate fast relay
such as Hugging Face Hub given `network_turbo`'s HF acceleration) should be
discussed with the user rather than blindly retried.

All of the following were prepared **locally** but only exist in this
session's now-gone scratchpad (`/tmp/claude-.../scratchpad/`), so they must
be regenerated from source on resume. Exact regeneration commands:

### 1. Monitoring image subset (2.2GB, 600 clips, 4198 files)

The subset must be **the first 600 clip IDs in sorted order with ≥6 frames**
in `F:\work\datasets\monitoring\images` (i.e. `/mnt/f/work/datasets/monitoring/images`
from WSL) — this exactly matches what
`discover_paired_locations()` in `src/scripts/run_controlled_retrieval_benchmark.py`
would select anyway for `pair_pool_size<=600` (it only ever reads
`sorted(clip_ids)[:pair_pool_size]`), so a 600-clip subset reproduces the
identical candidate pool the full 3705-clip dataset would give for both the
proxy12 (`pair_pool_size=240`) and proxy50 (`pair_pool_size=600`) runs — this
is not a lossy sample, it is the exact same deterministic prefix.

A helper script to rebuild it was written ad hoc in the scratchpad and lost;
recreate it (or write inline) as: group files by the `<clip_id>_frame<N>.jpg`
pattern, keep clip_ids with `>=6` frames, sort alphabetically, take the first
600, copy all their frame files into a staging dir, then `tar -C <staging> -cf
monitoring_subset.tar .`.

### 2. VPR backbone weights (398MB)

```
mkdir -p vpr_weights/patchnetvlad_pretrained
cp src/third_party/Patch-NetVLAD/patchnetvlad/pretrained_models/mapillary_WPCA4096.pth.tar vpr_weights/patchnetvlad_pretrained/
cp -r src/models/vpr_cache vpr_weights/
```
On remote, `mapillary_WPCA4096.pth.tar` must land at
`src/third_party/Patch-NetVLAD/patchnetvlad/pretrained_models/mapillary_WPCA4096.pth.tar`
and `vpr_cache/` at `src/models/vpr_cache/` (cosplace + mixvpr weights, 86MB).

### 3. E4 same-image utility manifest + image subset (55MB, 602 files)

```
python3 src/scripts/build_same_image_utility_manifest.py \
  --coco_root /mnt/f/work/datasets/coco --voc_root /mnt/f/work/datasets/VOC \
  --num_detection 200 --num_segmentation 200 --seed 1234 \
  --output <scratch>/utility_manifest.jsonl
```
This writes `<scratch>/utility_manifest_detection.jsonl` and
`..._segmentation.jsonl` with **absolute local paths**. A packaging step (also
ad hoc, lost — rewrite similarly) must copy only the referenced files into a
staging dir under `val2017/`, `JPEGImages/`, `SegmentationClass/`
subdirectories and rewrite the manifests' `image_path`/`segmentation_path` to
be relative to that staging dir, so `--root <staging>` resolves correctly on
the remote host (the script's `resolve_path()` only applies `--root` to
non-absolute paths).

### 4. Checkpoint

```
scp src/outputs/sensnet_final.pt root@connect.westd.seetacloud.com:/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/src/outputs/sensnet_final.pt
```
(356KB — this one is small enough to transfer even over a bad link; SHA-256
`576055d5bb173e45d29aab384abf8a0ac06e02d3f21fe3c75f66511a69d710a4`, verify it
matches after transfer.)

## Code changes already pushed (in origin/main, commit 19f0752)

1. **Real bugfix** in `src/scripts/evaluate_same_image_utility.py`'s
   `load_target()`: it was reading VOC-style palette-indexed segmentation
   masks through `_read_image()` (an RGB-photo decoder, `cv2.IMREAD_COLOR` /
   `Image.convert('RGB')`), which maps each palette index through to a
   display colour and destroys the 0..20/255 class-id semantics entirely —
   plus a second bug where the result (a `(3,H,W)` torch tensor) was passed
   directly to `cv2.resize`, which requires a numpy array and crashed
   outright. Fixed by adding `_read_class_index_mask()` (reads via PIL
   without RGB conversion) and using it instead. Verified with a local
   2-image CPU smoke run: mIoU 0.746 original, dropping less under `full`
   (PPEDCRF) than under `global_noise` (0.719 vs 0.548) — directionally
   consistent with the paper's privacy-utility story.
2. **New script** `src/scripts/build_same_image_utility_manifest.py` (COCO
   detection + VOC segmentation manifest builder for E4, see above).

This repo has an **automatic commit+push mechanism** (commits show up with
generic messages like "1", "2", "u1919" under the `mabo1215` identity,
pushed to `origin/main` on its own) — do not assume manual `git add/commit
/push` is needed for `src/`/`docs/` edits; check `git log --oneline -3` and
`git rev-parse HEAD` vs `origin/main` if in doubt, but they are normally
already in sync.

## Launch plan once data is on the remote (unchanged, not yet executed)

Five parallel `screen` sessions maximize the single 24-48GB card (all are
lightweight CNN/ViT-scale backbones, not LLMs, so concurrent GPU sharing is
fine — this was explicitly requested by the user, "多开Screen并行实验"):

- `e5_provenance` — `validate_sensnet_provenance.py` (fast, CPU-light)
- `proxy12` — `run_tomm_review_proxy.py --mode proxy --num_queries 12
  --pair_pool_size 240 --max_gallery 48 --gallery_sizes 12 24 48 --seeds 1234
  1235 1236 --backbones resnet18 resnet50 vgg16 clip_vitb32 clip_vitl14
  cosplace mixvpr patchnetvlad --include_attacker_aware` (covers E2/E3/E6/E7)
- `proxy50` — same but `--num_queries 50 --pair_pool_size 600 --max_gallery
  100 --gallery_sizes 50 75 100` (larger-scale confirmation)
- `e4_detection` — `evaluate_same_image_utility.py --mode manifest --manifest
  utility_subset/utility_manifest_detection.jsonl --root utility_subset
  --variants full global_noise no_temporal no_ncp unary_only no_dcrf`
- `e4_segmentation` — same with the segmentation manifest

E1 (geotagged VPR) stays **blocked** — no compliant place/GPS-labeled
manifest is available; do not fabricate one from the monitoring proxy data
(this was already decided in the original `docs/Design.md` plan and in
`docs/progress.md`'s blocked-items list).

10-minute status polling was requested by the user; a `remote_status.sh`
snapshot script (screen list, GPU state, per-run log tails, output row
counts) was drafted in the lost scratchpad — trivial to recreate, or just
poll `screen -ls` + `tail` the five run logs directly.

## How to re-verify network health before resuming

```
time timeout 15 ssh -p 22766 -o ConnectTimeout=10 root@connect.westd.seetacloud.com "echo PING_OK; date"
# then, only if that's fast (a couple seconds):
time (dd if=/dev/zero bs=1M count=50 2>/dev/null | ssh -p 22766 root@connect.westd.seetacloud.com "cat > /dev/null")
```
If the 50MB test takes more than ~30s (i.e. under ~1.5MB/s), treat bulk
transfer as still impractical and check with the user before spending more
time on it — do not silently retry for hours.
