#!/usr/bin/env bash
# Eleventh-cycle review: R1 (optimised allocation) and R5 (the two attackers
# the strongest attacks were never run against), on the vGPU 3090.
#
# Ten jobs share one 48 GB card. Each holds four or five frozen embedders plus
# a 2,000-image gallery index; memory is not the ceiling, the CPU-side JPEG and
# denoise round trips are, so each job is pinned to nine threads (10 x 9 = 90
# of 96 cores). Patch-NetVLAD runs at 480 px and is the one job whose gallery
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

launch() {  # name, command...
  local name="$1"; shift
  if screen -list 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
    echo "[skip] $name already running"
    return
  fi
  screen -dmS "$name" bash -c "cd $REPO && OMP_NUM_THREADS=9 MKL_NUM_THREADS=9 $* > $LOGS/${name}.log 2>&1; echo EXIT=\$? >> $LOGS/${name}.log"
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
pre r5_pre_vit_plain vit_b_16 64
pre r5_pre_vit_eot   vit_b_16 64 --eot_sanitizers $EOT_SAN

sleep 8
screen -list || true
