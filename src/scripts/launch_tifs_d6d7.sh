#!/usr/bin/env bash
# Launch the D6 held-out-transform study and the D7 hardened-direction utility
# audit in parallel screens on the vGPU 3090.
#
# Six jobs share one 48 GB card. Each holds four or five frozen embedders plus
# a 2,000-image gallery index, so the memory ceiling is not the constraint; the
# CPU-side sanitizers are, which is why each job is pinned to twelve threads
# (6 x 12 = 72 of 96 cores, leaving headroom for the JPEG/denoise round trips).
set -euo pipefail

REPO=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
PY="$REPO/.venv_ppedcrf/bin/python"
MANIFEST="$REPO/data/msls/manifest_all8.jsonl"
ROOT="$REPO/data/msls"
OUT="$REPO/src/outputs/tifs_d6"
OUT7="$REPO/src/outputs/tifs_d7"
LOGS="$REPO/logs_tifs"
SEEDS="1234 5678 9012"
# The four the hardening trains on, then the eight it never saw.
EVAL_SAN="none jpeg75 jpeg50 blur denoise jpeg60 jpeg30 median3 resize_half blur2 bitdepth4 random_one jpeg50_blur"
EOT_SAN="jpeg75 jpeg50 blur denoise"

mkdir -p "$OUT" "$OUT7" "$LOGS"
cd "$REPO"

launch() {  # name, command...
  local name="$1"; shift
  if screen -list | grep -q "\.${name}[[:space:]]"; then
    echo "[skip] $name already running"
    return
  fi
  screen -dmS "$name" bash -c "cd $REPO && OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 $* > $LOGS/${name}.log 2>&1; echo EXIT=\$? >> $LOGS/${name}.log"
  echo "[start] $name"
}

d6() {  # name, backbone, surrogates, deployable condition, extra args
  local name="$1" backbone="$2" surrogates="$3" deployable="$4"; shift 4
  launch "$name" "$PY src/scripts/run_direction_transfer_study.py \
    --manifest $MANIFEST --root $ROOT \
    --eval_backbone $backbone --surrogates $surrogates \
    --objective self --seeds $SEEDS \
    --conditions isotropic white_box $deployable \
    --eval_sanitizers $EVAL_SAN \
    $* --output $OUT/${name}.csv"
}

# ResNet18: three surrogates, deployable condition transfer_3.
d6 d6_r18_plain resnet18 "resnet50 vgg16 cosplace" transfer_3
d6 d6_r18_eot   resnet18 "resnet50 vgg16 cosplace" transfer_3 --eot_sanitizers $EOT_SAN

# MixVPR: four surrogates, deployable condition transfer_4.
d6 d6_mix_plain mixvpr "resnet18 resnet50 vgg16 cosplace" transfer_4
d6 d6_mix_eot   mixvpr "resnet18 resnet50 vgg16 cosplace" transfer_4 --eot_sanitizers $EOT_SAN

# D7: the hardened direction's downstream cost, both manifests, three runs each
# (the perturbation is recomputed per run and the backward pass through the
# surrogates is not deterministic, so the spread is real).
launch d7_seg "$PY src/scripts/evaluate_direction_utility.py \
  --manifest $REPO/utility_subset/utility_manifest_segmentation.jsonl \
  --surrogates resnet50 vgg16 cosplace --seeds $SEEDS \
  --eot_sanitizers $EOT_SAN --output_dir $OUT7/seg"
launch d7_det "$PY src/scripts/evaluate_direction_utility.py \
  --manifest $REPO/utility_subset/utility_manifest_detection.jsonl \
  --surrogates resnet50 vgg16 cosplace --seeds $SEEDS \
  --eot_sanitizers $EOT_SAN --output_dir $OUT7/det"

sleep 5
screen -list || true
