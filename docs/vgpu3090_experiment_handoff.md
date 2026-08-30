## UPDATE (2026-08-31): data relay completed, experiments running

The direct local↔vGPU 3090 link was re-tested after the local host restart
and is still broken (a 50MB dd-over-ssh test did not finish in 60s). The
"What is NOT yet on vGPU 3090" section and the network-bandwidth blocker
below are now historical — resolved via a Hugging Face Hub private-dataset
relay instead of fixing the direct link. Everything from "What is already
done" through "Code changes already pushed" below is still accurate
background reading.

**What actually happened:**
- Private dataset repo created: `mabo1215/ppedcrf-tomm-vgpu-relay`
  (namespace resolved via `whoami()`, token from `.env`'s
  `Huggingface_model_token`). Contains: `monitoring_subset.tar.part00..06`
  (7×350MB chunks of the 2.2GB tar — chunked because a single Bash tool call
  is capped at 10 minutes and one `upload_file`/`hf_hub_download` call is an
  atomic, non-resumable commit), `monitoring_subset.tar.sha256`,
  `vpr_weights.tar` (417MB), `utility_subset.tar` (57MB), `sensnet_final.pt`.
- Upload (local → HF): use the **default endpoint** (`huggingface.co`
  directly, no `-x`/proxy needed — it authenticates fine from this local
  machine). `hf-mirror.com` was tested and does NOT work for this: it only
  mirrors the file-resolve/download endpoints, not the authenticated write
  API (`whoami`, `create_repo`, `upload_file` all return 401 through it).
- Download (vGPU 3090 ← HF): `source /etc/network_turbo` is required first
  (huggingface.co is unreachable direct from this host — confirmed via a
  `curl` connect-timeout). **Do not use `huggingface_hub`'s default
  transfer backend for this** — `hf_hub_download`/`snapshot_download` use
  the `hf-xet` transfer backend by default (pulled in automatically by
  `huggingface_hub>=0.24`-ish), and it reproducibly stalls at 0 bytes
  through the `network_turbo` HTTP proxy (confirmed twice, on two different
  chunks, with no error — it just never sends another byte). Use plain
  `curl` against the resolve URL instead:
  `https://huggingface.co/datasets/<repo>/resolve/main/<filename>` with
  `-H "Authorization: Bearer $HF_TOKEN"`, and wrap it in a resumable retry
  loop — `curl -L --fail -H "..." --speed-limit 3000 --speed-time 20
  --connect-timeout 20 --retry-all-errors -C - -o out.tar url` inside a
  bash `until ... ; do sleep 5; done` loop (attempt cap ~40) — because a
  stalled connection needs the loop to kill/reconnect it (the
  `--speed-limit`/`--speed-time` pair makes curl itself abort a stalled
  transfer, `-C -` resumes from the partial byte offset, `--retry-all-errors`
  handles the rest). This combination downloaded all 11 files successfully.
- Remote pip index: the pre-configured `mirrors.aliyun.com` mirror started
  returning HTTP 403 for package lookups (`curl` confirmed 403 on a direct
  test) — switched to `https://pypi.tuna.tsinghua.edu.cn/simple/` via `pip
  config set global.index-url ...`, which worked immediately. If aliyun is
  broken again on a future session, try tsinghua first before debugging
  further.
- Screen-session gotcha: `screen -dmS name bash -c '...'` inherits the
  **environment of the shell that invokes it** (so a `source
  .venv/bin/activate` run in the same SSH command *before* the `screen
  -dmS` call carries into the detached session) — but if you omit that
  `source` line in a later relaunch, the screen session silently runs with
  system Python and fails fast. Always re-source the venv immediately
  before every `screen -dmS` call in the same command, don't assume it
  carries over from a previous SSH invocation.
- Installed packages beyond the original plan (discovered via two crash
  cycles, both now fixed): `matplotlib` (missing entirely from the first
  install list — `run_controlled_retrieval_benchmark.py` imports it at
  module load time), and `faiss-cpu`, `scikit-learn`, `pandas`, `scipy`
  (needed transitively by the Patch-NetVLAD/CosPlace backbone model files).
- Checksums verified after reassembly: `monitoring_subset.tar` sha256
  matches the local one exactly; `sensnet_final.pt` sha256 matches
  `576055d5bb173e45d29aab384abf8a0ac06e02d3f21fe3c75f66511a69d710a4`.
- Data extracted to: monitoring images →
  `/root/autodl-tmp/ppedcrf_tomm_20260830/monitoring_images/` (flat
  `<clip_id>_frame<N>.jpg`, 4198 files, pass as `--monitoring_root`); VPR
  weights → `src/third_party/Patch-NetVLAD/patchnetvlad/pretrained_models/mapillary_WPCA4096.pth.tar`
  and `src/models/vpr_cache/{cosplace,mixvpr}/`; E4 manifest+images →
  `<repo>/utility_subset/` (602 files, manifests already have relative
  paths, run with `--root utility_subset`); checkpoint →
  `<repo>/src/outputs/sensnet_final.pt`.

**Experiments launched** (5 screen sessions, per the "Launch plan" section
below, all under `/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF`, logs in
`/root/autodl-tmp/ppedcrf_tomm_20260830/run_logs/`):
- `e5_provenance` — **completed**, exit 0. Result:
  `mean_probability_spatial_std ≈ 2.63e-4` — matches the previously recorded
  blocked-item value (`2.6×10⁻⁴`), so this re-confirms (does not resolve)
  the E5 blocker: the current checkpoint's unary map is still essentially
  spatially constant. E5 stays blocked pending a mask-backed checkpoint.
- `proxy12`, `proxy50` — running (covers E2/E3/E6/E7). Confirmed alive via
  `ps aux` (>580% CPU each, real compute) rather than log output, because
  Python's stdout is fully buffered when redirected to a file — don't judge
  liveness from an empty log alone, check `ps aux | grep run_tomm_review`
  and/or `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
  `src/outputs/tomm_review_proxy/selection.json` already exists, confirming
  real progress.
- `e4_detection`, `e4_segmentation` — running. Each independently triggers
  a one-time `torchvision` download of `fasterrcnn_resnet50_fpn_coco`
  (~160MB from `download.pytorch.org`, slow direct — no turbo needed but
  no acceleration either, expect several minutes) the first time; this is
  normal, not a hang. Segmentation run uses `--output_dir
  src/outputs/tomm_same_image_utility_seg` to avoid colliding with
  detection's default output dir.
- E1 (geotagged VPR) — still not attempted, per the original blocked-item
  decision (no compliant place/GPS data).

**To check status in a later session**, from local WSL:
```bash
timeout 60 ssh -p 22766 -o ConnectTimeout=45 root@connect.westd.seetacloud.com "
screen -ls
tail -30 /root/autodl-tmp/ppedcrf_tomm_20260830/run_logs/proxy12.log
tail -30 /root/autodl-tmp/ppedcrf_tomm_20260830/run_logs/proxy50.log
tail -30 /root/autodl-tmp/ppedcrf_tomm_20260830/run_logs/e4_detection.log
tail -30 /root/autodl-tmp/ppedcrf_tomm_20260830/run_logs/e4_segmentation.log
find /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/src/outputs -name '*.csv' -o -name 'summary*.json'
"
```
A run finishing shows `EXIT_CODE=0` appended to its log (the launch wrapper
adds this). Once all 4 remaining runs show `EXIT_CODE=0`, pull the result
files back (`scp` — they're small CSV/JSON, no relay needed) and audit
against the manifests/seeds/checksums before writing any numbers into the
paper, per the existing plan in this doc's "Launch plan" section.

**Housekeeping note**: the HF token got printed in full into this session's
own tool output twice (once via a `.env` line-ending bug that broke an HTTP
header and dumped it in a traceback, once via a `ps aux` listing showing a
curl command line with the token inline) — never into a commit, file, or
the paper, but visible in this interactive session's transcript. Recorded
as a housekeeping item in `docs/progress.md`'s 遗留问题 section
(recommend rotating the token; not urgent, doesn't block anything).

---

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
