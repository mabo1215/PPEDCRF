#!/usr/bin/env bash
# Eleventh-cycle review: R1 (optimised allocation) and R5 (the two attackers
# the strongest attacks were never run against), on the vGPU 3090.
#
# Twelve jobs share one 48 GB card. Each holds four or five frozen embedders plus
# a 2,000-image gallery index; memory is not the ceiling, the CPU-side JPEG and
# denoise round trips are, so each job is pinned to nine threads; the total is
# deliberately over 96 cores, since the jobs alternate GPU and CPU phases.
# Patch-NetVLAD runs at 480 px and is the one job whose gallery
# batch has to come down.
#
# Every job here is resumable: rows are flushed per completed unit and
# completed keys are skipped on restart, so a job that is killed for the card
# can be relaunched with the same command.
set -euo pipefail

REPO=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
PY="$REPO/.venv_ppedcrf/bin/python"
MANIFEST="$REPO/data/msls/manifest_all8.jsonl"
ROOT="$REPO/data/msls"
OUT="$REPO/src/outputs/r11"
LOGS="$REPO/logs_r11"
SEEDS="1234 5678 9012"
# The four the hardening trains on, then the eight it never saw.
EVAL_SAN="none jpeg75 jpeg50 blur denoise jpeg60 jpeg30 median3 resize_half blur2 bitdepth4 random_one jpeg50_blur"
EOT_SAN="jpeg75 jpeg50 blur denoise"

mkdir -p "$OUT" "$LOGS"
cd "$REPO"

# Twelve processes racing for one card lose the race during gallery embedding,
# which is each job's memory peak: the first pass here killed the ViT sweep with
# an out-of-memory inside a conv the moment eleven neighbours had already
# claimed 44 GB. So every job waits for a real window before it starts rather
# than thrashing the allocator, and re-running this script is the recovery
# path -- a job whose screen is gone is relaunched, and it resumes from its own
# partly written CSV.
WAIT_FREE_MB="${WAIT_FREE_MB:-9000}"

launch() {  # name, command...
  local name="$1"; shift
  if screen -list 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
    echo "[skip] $name already running"
    return
  fi
  # A screen by that name is not the invariant that matters -- two processes
  # appending to one output file is. That happened here once: a job relaunched
  # by hand and a job relaunched by this script both wrote every row, and
  # because each had read the resume set at startup neither skipped anything.
  # The result was 64k rows carrying 32k keys, a quarter of which disagreed
  # with their own duplicate, since the backward pass is not deterministic.
  # Check the file, not the screen.
  if pgrep -f "$OUT/${name}.csv" >/dev/null 2>&1; then
    echo "[skip] $name: a process is already writing $OUT/${name}.csv"
    return
  fi
  screen -dmS "$name" bash -c "
    cd $REPO
    while [ \$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt $WAIT_FREE_MB ]; do sleep 120; done
    OMP_NUM_THREADS=9 MKL_NUM_THREADS=9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $* > $LOGS/${name}.log 2>&1
    echo EXIT=\$? >> $LOGS/${name}.log"
  echo "[start] $name"
}

# ---------------------------------------------------------------------------
# R1: solve for the placement map instead of prescribing it.
#
# Two noise modes per attacker. `expectation` is the arm that answers the
# review -- the map never sees a realisation, so it stays on the allocation
# axis. `realised` lets the map see the draw it will be released with, which
# turns it into sign selection and is reported as evidence about the sign
# pattern rather than about placement.
# ---------------------------------------------------------------------------
alloc() {  # name, backbone, surrogates, mode, conditions...
  local name="$1" backbone="$2" surrogates="$3" mode="$4"; shift 4
  launch "$name" "$PY src/scripts/run_optimised_allocation_study.py \
    --manifest $MANIFEST --root $ROOT \
    --eval_backbone $backbone --surrogates $surrogates \
    --noise_mode $mode --seeds $SEEDS --batch 8 \
    --steps 20 --double_steps 40 \
    --conditions $* \
    --output $OUT/${name}.csv"
}

alloc r1_r18_exp_tr resnet18 "resnet50 vgg16 cosplace" expectation uniform opt_transfer
alloc r1_r18_exp_wb resnet18 "resnet50 vgg16 cosplace" expectation uniform opt_whitebox
alloc r1_r18_real   resnet18 "resnet50 vgg16 cosplace" realised    uniform opt_transfer opt_whitebox
alloc r1_mix_exp_tr mixvpr   "resnet18 resnet50 vgg16 cosplace" expectation uniform opt_transfer
alloc r1_mix_exp_wb mixvpr   "resnet18 resnet50 vgg16 cosplace" expectation uniform opt_whitebox
alloc r1_mix_real   mixvpr   "resnet18 resnet50 vgg16 cosplace" realised    uniform opt_transfer opt_whitebox

# ---------------------------------------------------------------------------
# R5: the two attackers the purifier and the preprocessing study never faced.
# Patch-NetVLAD carries the largest directional effect in the paper and ViT the
# smallest, so between them they decide whether the deployable claim survives.
# ---------------------------------------------------------------------------
launch r5_purify_pnv "$PY src/scripts/run_purification_attack.py \
  --manifest $MANIFEST --root $ROOT \
  --eval_backbone patchnetvlad --surrogates resnet50 vgg16 cosplace \
  --gallery_batch 24 \
  --cache $OUT/purify_cache_pnv \
  --output $OUT/purify_pnv.csv"

launch r5_purify_vit "$PY src/scripts/run_purification_attack.py \
  --manifest $MANIFEST --root $ROOT \
  --eval_backbone vit_b_16 --surrogates resnet50 vgg16 cosplace \
  --gallery_batch 64 \
  --cache $OUT/purify_cache_vit \
  --output $OUT/purify_vit.csv"

pre() {  # name, backbone, gallery_batch, extra args
  local name="$1" backbone="$2" gb="$3"; shift 3
  launch "$name" "$PY src/scripts/run_direction_transfer_study.py \
    --manifest $MANIFEST --root $ROOT \
    --eval_backbone $backbone --surrogates resnet50 vgg16 cosplace \
    --objective self --seeds $SEEDS --gallery_batch $gb \
    --conditions isotropic white_box transfer_3 \
    --eval_sanitizers $EVAL_SAN \
    $* --output $OUT/${name}.csv"
}

pre r5_pre_pnv_plain patchnetvlad 24
pre r5_pre_pnv_eot   patchnetvlad 24 --eot_sanitizers $EOT_SAN
pre r5_pre_vit_plain vit_b_16 32
pre r5_pre_vit_eot   vit_b_16 32 --eot_sanitizers $EOT_SAN

sleep 8
screen -list || true
