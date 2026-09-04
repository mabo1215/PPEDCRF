# ICME 2027 Session 3 Handoff: N1-N5 Next-Step Experiments

Written so a fresh Claude Code session (after a restart) can resume this
exact task with no lost context, and so an in-flight background transfer is
not silently abandoned. Read this fully before taking any action if you are
picking this up cold. vGPU 3090 is in **no-card mode** (powered on, no GPU
attached, no compute billing) as of this writing -- do not tell the user to
power on the GPU until the "Remaining steps before GPU-on" section below is
fully checked off.

## What this session is doing

Following up on the "N1-N5 next-step experiments" table from the previous
report to the user (MSLS city expansion, matched-PSNR at more gallery sizes,
white-box attacker on more backbones, sensitivity-map recalibration
[deprioritized, not in scope], deterministic-baseline matched-PSNR). The
user asked to prepare code and data on vGPU 3090 in no-card mode, then
report when ready for a paid GPU-on.

## Code already pushed to origin/main (done, verified on vGPU)

- `276607e` - N3 (white-box attacker works for any `--attacker_backbone`,
  no code change needed) confirmed ready; N5 code added
  (`--blur_kernel_size`/`--mosaic_block_size` overrides in
  `run_tomm_review_proxy.py`, new `src/scripts/deterministic_baseline_psnr_match.py`).
  Smoke-tested locally (kernel_size=5 correctly gives 33.77dB vs default
  kernel=21's 28.08dB).
- `d45d88b` - **real perf bug fix** in `build_msls_manifest.py`: the
  gallery-record list (identical for every query line) was being rebuilt,
  including `Path.resolve()` on every gallery item, once per query --
  800,000 redundant filesystem round-trips at 400 queries x 2000 gallery.
  Fixed to build it once. Turned an operation that was still only 13% done
  after 70+ minutes into one that completes in under 10 minutes.
- `9c44eca` - the mirror-image bug in `run_geotagged_vpr_benchmark.py`'s
  `load_manifest()` (used by both the audit script and the actual benchmark
  run) -- same fix pattern (cache raw path/place_id per `gallery_id`,
  resolve only once).

vGPU 3090 was `git pull --ff-only`'d to include all of the above as of this
session (confirm with `git log --oneline -5` on the remote; expect
`9c44eca` or later at HEAD).

## New data built and validated locally (persisted, safe across restarts)

All under `src/outputs/icme2027_manifest_expanded/` in the repo (gitignored,
but on this local machine's disk, not in any session scratchpad -- survives
a Claude Code restart):

- `manifest_all8.jsonl` (312MB) / `.gz` (41MB) -- 400 queries, 2000-gallery,
  8 cities (manila, toronto, boston, cph, zurich, london, amman, nairobi),
  subtask=all. Audit: `manifest_all8_gate.json`, `valid=True`,
  `coverage_gate_passed=True`, 277 unique place ids, illumination
  394 day / **6 night** (up from 0 in the old 2-city manifest), viewpoint
  391 Forward / 9 Sideways. season/weather are empty **dataset-wide** in
  MSLS `postprocessed.csv` (confirmed by direct inspection across all 24
  cities this session, not just Manila/Toronto -- there is no MSLS
  season/weather metadata to expand into, full stop; do not spend more time
  looking for it).
- `manifest_o2n8.jsonl` (312MB) / `.gz` (38MB) -- subtask=o2n, 5 cities
  (cph, london, manila, toronto, zurich -- boston/amman/nairobi have no o2n
  eligible rows), 160 unique place ids (up from 81). `valid=True`,
  `coverage_gate_passed=True`.
- `manifest_n2o8.jsonl` (312MB) / `.gz` (37MB) -- subtask=n2o, same 5
  cities, 119 unique place ids (up from 54). `valid=True`,
  `coverage_gate_passed=True`.

These three replace the old 2-city (Manila/Toronto only) manifests as the
basis for a wider N1 real-MSLS run; the **old** `data/msls/manifest_{all,o2n,n2o}.jsonl`
on vGPU stay in place and are not being replaced, only supplemented.

New MSLS city images extracted locally at
`/mnt/g/work/datasets/msls/extracted/train_val/{boston,cph,zurich,london,amman,nairobi}/`
directly from the already-downloaded raw zip parts
(`/mnt/g/work/datasets/msls/raw/part0{1..9}.zip`+`part10.zip`) -- **no new
download was needed**; the full official MSLS metadata for all 25
`train_val` cities was already present locally, and the raw zips turned out
to contain all cities' images (mixed across the 10 parts), not just
Manila/Toronto as previously assumed. If this handoff is picked up on a
different machine without this local G: drive, that extraction step (see
"Regeneration commands" below) must be redone from those zips first.

## In-flight when this note was written: image transfer to vGPU

Only the files actually referenced by the three new manifests need to reach
vGPU (not full city folders): 6,799 unique files total, of which 2,711 are
Manila/Toronto (**already on vGPU**, do not re-transfer) and **4,088 are
from the 6 new cities** (need transfer). These were packaged into one
tarball:

```
/tmp/claude-1000/.../scratchpad/msls_new_cities_subset.tar.gz  (159MB)
sha256: 98a1741a256737c5ffefc1b63e6c659fe4fb729c461481ef3ad9451999dff28d
```

**This tarball lives in this session's scratchpad and will NOT survive a
Claude Code restart or a new session.** If it's gone, regenerate it with the
"Regeneration commands" below (fast, seconds, once the manifests -- which
DO persist -- and the local city image extraction are in place).

The vGPU 3090 network link is currently very slow (a 20MB single-stream
throughput test did not complete in 2 minutes). Per
`.claude/rules/shell.md` precedent, this session switched to a 16-way
parallel chunked scp transfer (10MB chunks) using
`/tmp/claude-1000/.../scratchpad/parallel_transfer.sh` (also
scratchpad-only, see "Regeneration commands" to recreate it -- it's a small,
generic chunk-and-parallel-scp-and-reassemble script, reproduced in full
below so it does not need to be reverse-engineered).

Remote destination for the chunks:
`/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/data/msls_incoming/msls_new_cities_subset.tar.gz.chunks/`
(16 chunks, `part_0000`..`part_0015`). Progress checkpoints during this
session: 6/16 done (3 succeeded, 3 dropped with "Connection closed by
36.103.198.204 port 22766" -- likely an SSH concurrent-connection limit on
the vGPU side, not a data-corruption risk), then 13/16 done. Check the
live count with the command in "If you are resuming this session" below --
do not trust the number above as current, it is a snapshot.

### If you are resuming this session (transfer possibly interrupted)

1. Check what actually landed on vGPU:
   ```bash
   ssh -p 22766 root@connect.westd.seetacloud.com \
     "ls /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/data/msls_incoming/msls_new_cities_subset.tar.gz.chunks/ 2>/dev/null | wc -l"
   ```
   Expect 16 when complete. If the reassembled `msls_new_cities_subset.tar.gz`
   already exists there instead (the chunks dir is removed after
   reassembly), check its sha256 against `98a1741a...` above -- if it
   matches, the transfer is already done, skip to "Remaining steps".
2. If chunks are missing/incomplete and the local tarball no longer exists
   (scratchpad gone), regenerate it (see below), then re-run
   `parallel_transfer.sh` -- it's safe to re-run; `scp` overwrites, and a
   full 16-way re-split-and-resend of a 159MB file is a few minutes even at
   this link's poor throughput once parallelized.
3. If the local tarball still exists (same session, no restart), just re-run
   `parallel_transfer.sh` again -- scp'ing the same 16 chunks again is cheap
   and will pick up where individual chunk failures left off (each chunk
   transfer is independent and idempotent).

## Remaining steps before GPU-on (none require the GPU itself)

1. **Finish the image transfer** (see above).
2. On vGPU, extract the tarball into the existing MSLS tree and verify:
   ```bash
   ssh -p 22766 root@connect.westd.seetacloud.com "
   cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
   sha256sum data/msls_incoming/msls_new_cities_subset.tar.gz
   tar -xzf data/msls_incoming/msls_new_cities_subset.tar.gz -C data/msls/
   find data/msls/train_val/{boston,cph,zurich,london,amman,nairobi} -type f | wc -l
   "
   ```
   Expect the sha256 to match `98a1741a...` and the file count to be 4,088.
3. **Transfer the three gzipped manifests** (117MB total, much smaller than
   the 933MB uncompressed originals -- transfer the `.gz` files, not the
   `.jsonl` files):
   ```bash
   scp -P 22766 src/outputs/icme2027_manifest_expanded/manifest_{all8,o2n8,n2o8}.jsonl.gz \
     root@connect.westd.seetacloud.com:/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/data/msls/
   ```
   If the link is still slow, use `parallel_transfer.sh` for these too (one
   file at a time; each is under 42MB so a single chunked pass should be
   quick). Then on vGPU: `cd data/msls && gunzip -k manifest_{all8,o2n8,n2o8}.jsonl.gz`
   and verify each with `sha256sum` against the local `.jsonl.gz` (not the
   decompressed `.jsonl`, since gzip is deterministic but verifying the
   compressed form avoids re-hashing 300MB+ files).
4. **Audit the three new manifests on vGPU itself** (paths must resolve
   under vGPU's own filesystem, not just locally) before spending any GPU
   time:
   ```bash
   ssh -p 22766 root@connect.westd.seetacloud.com "
   cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
   source .venv_ppedcrf/bin/activate
   for M in all8 o2n8 n2o8; do
     python src/scripts/audit_geotagged_manifest.py --mode manifest \
       --manifest data/msls/manifest_\${M}.jsonl --root data/msls \
       --output data/msls/manifest_\${M}_gate_remote.json
   done
   "
   ```
   Every one must report `valid=True coverage_gate=True queries=400 gallery=2000`.
5. **Deploy and dry-check the launch script** (`launch_session3.sh`, full
   content below -- recreate it on vGPU from this doc, scp it, `chmod +x`,
   but do NOT run it until the user has explicitly said to power on the
   GPU). It launches 16 screens:
   - `n1_<backbone>` x6 (resnet18/resnet50/vgg16/cosplace/mixvpr/patchnetvlad):
     black-box VPR on the new 8-city `manifest_all8.jsonl`.
   - `n1_whitebox`: white-box sign-gradient attacker (ResNet18) on the same
     expanded manifest.
   - `n3_<backbone>` x5 (resnet50/vgg16/cosplace/mixvpr/patchnetvlad):
     white-box attacker extended to these backbones, on the **existing**
     3-manifest set (`data/msls/manifest_{all,o2n,n2o}.jsonl`, 200-query,
     2-city -- ResNet18 white-box on this set is already done, see the
     prior session's Table~tab:msls_whitebox).
   - `n2_g12` / `n2_g100`: matched-PSNR sigma sweep (12 points) at gallery
     sizes 12 and 100 (ResNet18), extending the existing gallery-size-48-only
     result.
   - `n5_blur` / `n5_mosaic`: kernel-size {5,11,21,31} / block-size
     {4,8,12,20} sweep for the deterministic baselines (ResNet18,
     gallery 48), to replace the paper's current "PSNR gap" caveat with a
     true matched comparison via `deterministic_baseline_psnr_match.py`
     (run this analysis script locally/on-vGPU as pure CPU post-processing
     after both `n5_blur` and `n5_mosaic` finish -- it needs both sweep
     dirs and is not auto-chained in the launch script).
   `OMP_NUM_THREADS`/`MKL_NUM_THREADS` are set to 6 per screen (16 screens x
   6 = 96, matching the vGPU's core count) rather than the 10 used for the
   prior 8-screen session, to avoid CPU oversubscription at this larger
   screen count.
   **Known risk**: `patchnetvlad` appears in both `n1_patchnetvlad` and
   `n3_patchnetvlad` and has previously spiked to 20+GB VRAM alone; running
   both simultaneously carries real OOM risk (precedent: a prior 12-way
   concurrent launch OOM'd 4 jobs from simultaneous startup memory spikes,
   all of which succeeded on individual retry once other jobs freed memory).
   This is an accepted, monitored risk, not a blocker -- the 10-minute
   check-in loop should catch and allow retrying any OOM'd job once GPU-on
   happens, matching how the prior session handled the same failure mode.
6. **Tell the user it's ready for GPU-on** only after steps 1-5 all pass.
   Do not ask them to power on the GPU while data staging is still in
   progress -- no-card mode costs nothing extra while this finishes.

## Regeneration commands (if scratchpad state is lost)

Rebuild the referenced-file list and tarball (seconds, once
`src/outputs/icme2027_manifest_expanded/manifest_*.jsonl` and the extracted
city images under `/mnt/g/work/datasets/msls/extracted/train_val/` both
exist -- the manifests persist in the repo tree; the extracted images
persist on the G: drive but were produced by a one-off script, reproduced
here):

```python
# Re-extract the 6 new cities' images from the raw zips (only needed if
# /mnt/g/work/datasets/msls/extracted/train_val/{boston,cph,zurich,london,amman,nairobi}
# don't already exist -- check first, this is redundant if they're already there):
import zipfile
from pathlib import Path
RAW_DIR = Path("/mnt/g/work/datasets/msls/raw")
OUT_ROOT = Path("/mnt/g/work/datasets/msls/extracted")
TARGET_CITIES = {"boston", "zurich", "london", "amman", "nairobi", "cph"}
for part in sorted(RAW_DIR.glob("part*.zip")):
    z = zipfile.ZipFile(part)
    members = [n for n in z.namelist() if not n.endswith("/") and n.startswith("train_val/")
               and n.split("/")[1] in TARGET_CITIES and "/images/" in n]
    if members:
        z.extractall(path=OUT_ROOT, members=members)
```

```python
# Rebuild the referenced-file list and tarball from the (persisted) manifests:
import json
new_cities = {'boston','cph','zurich','london','amman','nairobi'}
paths = set()
for m in ['all8', 'o2n8', 'n2o8']:
    with open(f'src/outputs/icme2027_manifest_expanded/manifest_{m}.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            for p in [rec['query_path']] + [g['path'] for g in rec['gallery']]:
                if p.split('/')[1] in new_cities:
                    paths.add(p)
with open('/tmp/new_city_files.txt', 'w') as f:
    for p in sorted(paths):
        f.write(p + '\n')
print(len(paths), 'paths (expect 4088)')
```

```bash
cd /mnt/g/work/datasets/msls/extracted
tar -czf /tmp/msls_new_cities_subset.tar.gz -T /tmp/new_city_files.txt
sha256sum /tmp/msls_new_cities_subset.tar.gz   # expect 98a1741a25...dff28d
```

The manifests themselves, if somehow lost (they should not be -- they are
under the tracked project tree, gitignored but on local disk), are rebuilt
with (each takes under 10 minutes with the perf fix in `d45d88b`):

```bash
python3 src/scripts/build_msls_manifest.py \
  --msls_root /mnt/g/work/datasets/msls/extracted --split train_val \
  --cities manila toronto boston cph zurich london amman nairobi \
  --subtask all --max_queries 400 --max_gallery 2000 \
  --output src/outputs/icme2027_manifest_expanded/manifest_all8.jsonl \
  --metadata_output src/outputs/icme2027_manifest_expanded/manifest_all8.metadata.json
# repeat with --subtask o2n / n2o for the other two (same city list)
```

## `parallel_transfer.sh` (full content, for recreation)

```bash
#!/bin/bash
set -u
SRC="$1"
REMOTE_DIR="$2"
BASENAME="$(basename "$SRC")"
CHUNK_DIR="$(dirname "$SRC")/${BASENAME}.chunks"
N_WAYS=16

rm -rf "$CHUNK_DIR"
mkdir -p "$CHUNK_DIR"
split -b 10M -d -a 4 "$SRC" "$CHUNK_DIR/part_"
CHUNKS=("$CHUNK_DIR"/part_*)
echo "Split into ${#CHUNKS[@]} chunks of ~10MB each"

ssh -p 22766 -o ConnectTimeout=20 root@connect.westd.seetacloud.com "mkdir -p ${REMOTE_DIR}/${BASENAME}.chunks"

PIDS=()
running=0
for chunk in "${CHUNKS[@]}"; do
  scp -P 22766 -o ConnectTimeout=20 "$chunk" "root@connect.westd.seetacloud.com:${REMOTE_DIR}/${BASENAME}.chunks/$(basename "$chunk")" &
  PIDS+=($!)
  running=$((running+1))
  if [ "$running" -ge "$N_WAYS" ]; then
    wait -n
    running=$((running-1))
  fi
done
wait

echo "All chunks transferred, reassembling remotely..."
ssh -p 22766 -o ConnectTimeout=20 root@connect.westd.seetacloud.com "
cd ${REMOTE_DIR}
cat ${BASENAME}.chunks/part_* > ${BASENAME}
rm -rf ${BASENAME}.chunks
sha256sum ${BASENAME}
"
echo "Local sha256:"
sha256sum "$SRC"
```

Note: with `N_WAYS` equal to the total chunk count (16 chunks, 16-way), all
chunks launch essentially simultaneously rather than being throttled -- this
is what triggered 3 "Connection closed" failures in this session (likely an
sshd concurrent-connection limit on vGPU), not a fundamental problem with the
approach. If retrying and hitting the same issue repeatedly, lower `N_WAYS`
to 8 for real throttling via the `wait -n` gate.

## `launch_session3.sh` (full content, for recreation on vGPU)

```bash
#!/bin/bash
set -u
cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
OUT_ROOT=icme2027_revision_20260904_session3
mkdir -p "$OUT_ROOT"

# --- N1: expanded 8-city MSLS manifest, 6-backbone black-box VPR ---
for BB in resnet18 resnet50 vgg16 cosplace mixvpr patchnetvlad; do
  mkdir -p "$OUT_ROOT/n1_expanded_msls/$BB"
  screen -dmS "n1_${BB}" bash -c "
    cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
    source .venv_ppedcrf/bin/activate
    export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
    python src/scripts/run_geotagged_vpr_benchmark.py \
      --mode geotagged \
      --manifest data/msls/manifest_all8.jsonl \
      --root data/msls \
      --config src/config/config.yaml \
      --checkpoint src/outputs/sensnet_final.pt \
      --backbones ${BB} \
      --variants full \
      --seeds 1234 1235 1236 \
      --output_dir ${OUT_ROOT}/n1_expanded_msls/${BB} \
      > ${OUT_ROOT}/n1_expanded_msls/${BB}/run.log 2>&1
    echo EXIT_CODE=\$? > ${OUT_ROOT}/n1_expanded_msls/${BB}/DONE_STATUS.txt
  "
done

# --- N1b: white-box attacker on the expanded manifest, ResNet18 only ---
mkdir -p "$OUT_ROOT/n1_whitebox"
screen -dmS "n1_whitebox" bash -c "
  cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
  source .venv_ppedcrf/bin/activate
  export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
  python src/scripts/run_geotagged_vpr_benchmark.py \
    --mode geotagged \
    --manifest data/msls/manifest_all8.jsonl \
    --root data/msls \
    --config src/config/config.yaml \
    --checkpoint src/outputs/sensnet_final.pt \
    --backbones resnet18 \
    --variants full \
    --seeds 1234 1235 1236 \
    --include_attacker_aware \
    --attacker_backbone resnet18 \
    --attacker_steps 20 \
    --attacker_step_size 1.0 \
    --attacker_linf 8.0 \
    --output_dir ${OUT_ROOT}/n1_whitebox \
    > ${OUT_ROOT}/n1_whitebox/run.log 2>&1
  echo EXIT_CODE=\$? > ${OUT_ROOT}/n1_whitebox/DONE_STATUS.txt
"

# --- N3: white-box attacker extended to 5 more backbones on the existing 3 official manifests ---
for BB in resnet50 vgg16 cosplace mixvpr patchnetvlad; do
  mkdir -p "$OUT_ROOT/n3_whitebox/$BB"
  screen -dmS "n3_${BB}" bash -c "
    cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
    source .venv_ppedcrf/bin/activate
    export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
    FAIL=0
    for SPLIT in all o2n n2o; do
      python src/scripts/run_geotagged_vpr_benchmark.py \
        --mode geotagged \
        --manifest data/msls/manifest_\${SPLIT}.jsonl \
        --root data/msls \
        --config src/config/config.yaml \
        --checkpoint src/outputs/sensnet_final.pt \
        --backbones ${BB} \
        --variants full \
        --seeds 1234 1235 1236 \
        --include_attacker_aware \
        --attacker_backbone ${BB} \
        --attacker_steps 20 \
        --attacker_step_size 1.0 \
        --attacker_linf 8.0 \
        --output_dir ${OUT_ROOT}/n3_whitebox/${BB}/\${SPLIT} \
        >> ${OUT_ROOT}/n3_whitebox/${BB}/run.log 2>&1
      RC=\$?
      echo \"\${SPLIT} exit=\$RC\" >> ${OUT_ROOT}/n3_whitebox/${BB}/points.log
      if [ \$RC -ne 0 ]; then FAIL=1; fi
    done
    echo EXIT_CODE=\$FAIL > ${OUT_ROOT}/n3_whitebox/${BB}/DONE_STATUS.txt
  "
done

# --- N2: matched-PSNR sigma sweep at two more gallery sizes (ResNet18) ---
SIGMAS="4 6 8 10 12 16 20 24 28 32 40 50"
for GS in 12 100; do
  mkdir -p "$OUT_ROOT/n2_gallery_sweep/g${GS}"
  screen -dmS "n2_g${GS}" bash -c "
    cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
    source .venv_ppedcrf/bin/activate
    export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
    FAIL=0
    for S in ${SIGMAS}; do
      python src/scripts/run_tomm_review_proxy.py \
        --mode proxy \
        --sigma \$S \
        --monitoring_root /root/autodl-tmp/ppedcrf_tomm_20260830/monitoring_images \
        --checkpoint src/outputs/sensnet_final.pt \
        --backbones resnet18 \
        --gallery_sizes ${GS} \
        --max_gallery ${GS} \
        --seeds 1234 1235 1236 \
        --output_dir ${OUT_ROOT}/n2_gallery_sweep/g${GS}/sigma_\$S \
        >> ${OUT_ROOT}/n2_gallery_sweep/g${GS}/run.log 2>&1
      RC=\$?
      echo \"sigma=\$S exit=\$RC\" >> ${OUT_ROOT}/n2_gallery_sweep/g${GS}/points.log
      if [ \$RC -ne 0 ]; then FAIL=1; fi
    done
    if [ \$FAIL -eq 0 ]; then
      python src/scripts/matched_psnr_from_sweep.py \
        --sweep_dir ${OUT_ROOT}/n2_gallery_sweep/g${GS} \
        --backbone resnet18 --gallery_size ${GS} --targets 30 33 36 \
        --output ${OUT_ROOT}/n2_gallery_sweep/g${GS}/matched_psnr_table.csv \
        >> ${OUT_ROOT}/n2_gallery_sweep/g${GS}/run.log 2>&1
      python src/scripts/significance_test_matched_psnr.py \
        --sweep_dir ${OUT_ROOT}/n2_gallery_sweep/g${GS} \
        --backbone resnet18 --gallery_size ${GS} --targets 30 33 36 \
        --compare_against global_noise \
        --output ${OUT_ROOT}/n2_gallery_sweep/g${GS}/matched_psnr_significance.csv \
        >> ${OUT_ROOT}/n2_gallery_sweep/g${GS}/run.log 2>&1
    fi
    echo EXIT_CODE=\$FAIL > ${OUT_ROOT}/n2_gallery_sweep/g${GS}/DONE_STATUS.txt
  "
done

# --- N5: deterministic-baseline (blur/mosaic) matched-PSNR sweep (ResNet18, gallery 48) ---
mkdir -p "$OUT_ROOT/n5_blur_sweep" "$OUT_ROOT/n5_mosaic_sweep"
screen -dmS "n5_blur" bash -c "
  cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
  source .venv_ppedcrf/bin/activate
  export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
  FAIL=0
  for K in 5 11 21 31; do
    python src/scripts/run_tomm_review_proxy.py \
      --mode proxy \
      --blur_kernel_size \$K \
      --monitoring_root /root/autodl-tmp/ppedcrf_tomm_20260830/monitoring_images \
      --checkpoint src/outputs/sensnet_final.pt \
      --backbones resnet18 \
      --gallery_sizes 48 \
      --seeds 1234 1235 1236 \
      --output_dir ${OUT_ROOT}/n5_blur_sweep/k_\$K \
      >> ${OUT_ROOT}/n5_blur_sweep/run.log 2>&1
    RC=\$?; if [ \$RC -ne 0 ]; then FAIL=1; fi
  done
  echo EXIT_CODE=\$FAIL > ${OUT_ROOT}/n5_blur_sweep/DONE_STATUS.txt
"
screen -dmS "n5_mosaic" bash -c "
  cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
  source .venv_ppedcrf/bin/activate
  export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
  FAIL=0
  for B in 4 8 12 20; do
    python src/scripts/run_tomm_review_proxy.py \
      --mode proxy \
      --mosaic_block_size \$B \
      --monitoring_root /root/autodl-tmp/ppedcrf_tomm_20260830/monitoring_images \
      --checkpoint src/outputs/sensnet_final.pt \
      --backbones resnet18 \
      --gallery_sizes 48 \
      --seeds 1234 1235 1236 \
      --output_dir ${OUT_ROOT}/n5_mosaic_sweep/b_\$B \
      >> ${OUT_ROOT}/n5_mosaic_sweep/run.log 2>&1
    RC=\$?; if [ \$RC -ne 0 ]; then FAIL=1; fi
  done
  echo EXIT_CODE=\$FAIL > ${OUT_ROOT}/n5_mosaic_sweep/DONE_STATUS.txt
"
# Note: deterministic_baseline_psnr_match.py needs BOTH n5_blur_sweep and
# n5_mosaic_sweep to finish (different screens); run it as a separate local
# CPU-only post-processing step after pulling both sweep dirs back, not here.

sleep 3
echo "--- screens ---"
screen -ls
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Note this script uses `--manifest data/msls/manifest_all8.jsonl --root
data/msls` (i.e. the new manifests are placed directly into the *existing*
`data/msls/` root alongside the old ones, not a separate
`data/msls_expanded/` root) -- this only works once step 3 above has moved
the gunzipped manifests into `data/msls/` on vGPU, matching where their
embedded relative paths (`train_val/<city>/...`) actually resolve.

## Estimated timing once GPU-on happens

Not measured yet for this specific batch. For calibration: the prior
session's 8-screen batch (3 white-box MSLS runs + 5 backbone-wide
matched-PSNR sweeps) completed in about 24 minutes wall time. This batch has
roughly double the screens (16) and includes one meaningfully bigger job (N1
black-box VPR on a 400-query/2000-gallery manifest, 2x the query count and
2x the gallery size of the existing 200/1000 manifests) -- budget for this
being a longer session than the last one, plausibly 45-90 minutes, but this
is an engineering estimate, not a measurement; record the actual wall time
when it completes.

## Paper writeback and review-file rules (unchanged from standing project rules)

No number from this batch may be written into `paper/main.tex` or
`paper/appendix.tex` until its completion gate passes (finite values,
correct row counts, EXIT_CODE=0). After writeback, re-check whether
`docs/RevisionSuggestions.tex` needs another update -- the current version
already recommends "accept, conditional on minor revisions" as of the
previous session; N1's wider-city result in particular is exactly the kind
of evidence that would let the current review's one remaining data-driven
caveat (MSLS city/condition coverage) be marked resolved rather than
disclosed-as-limitation, if the results come out reasonably consistent with
the existing 2-city finding.
