#!/usr/bin/env bash
# R10, the two limits the eleventh-cycle review could only state rather than
# test: an attacker class the threat model admits but the evidence never
# covered, and a gallery size that never varied.
#
# Two RTX 4080 SUPER, 32 GB each. GPU 0 takes the CLIP attacker, whose ViT-L/14
# trunk is the memory-heavy half; GPU 1 takes the sweep, which is 32 small runs
# and is bound by optimisation count rather than by memory.
set -euo pipefail

REPO=/root/autodl-tmp/PPEDCRF
PY=/root/miniconda3/bin/python
ROOT="$REPO/data/msls"
OUT="$REPO/src/outputs/r10"
LOGS="$REPO/logs_r10"
SEEDS="1234 5678 9012"
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

launch() {  # gpu, name, command...
  local gpu="$1" name="$2"; shift 2
  if screen -list 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
    echo "[skip] $name already running"; return
  fi
  # One writer per output file is the invariant that matters, not the screen
  # name -- two processes appending to one export cost a run last cycle.
  if pgrep -f "$OUT/${name}.csv" >/dev/null 2>&1; then
    echo "[skip] $name: a process is already writing its output"; return
  fi
  screen -dmS "$name" bash -c "cd $REPO && CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $* > $LOGS/${name}.log 2>&1; \
    echo EXIT=\$? >> $LOGS/${name}.log"
  echo "[start] gpu$gpu $name"
}

# --- GPU 0: CLIP ViT-L/14, a retriever no surrogate shares a trunk with -----
launch 0 r10_clip_dir "$PY src/scripts/run_direction_transfer_study.py \
  --manifest $ROOT/manifest_all8.jsonl --root $ROOT \
  --eval_backbone clip_vitl14 --surrogates resnet50 vgg16 cosplace \
  --objective self --seeds $SEEDS --gallery_batch 48 \
  --conditions isotropic white_box transfer_3 \
  --output $OUT/r10_clip_dir.csv"

launch 0 r10_clip_alloc "$PY src/scripts/run_optimised_allocation_study.py \
  --manifest $ROOT/manifest_all8.jsonl --root $ROOT \
  --eval_backbone clip_vitl14 --surrogates resnet50 vgg16 cosplace \
  --noise_mode expectation --seeds $SEEDS --batch 4 \
  --steps 20 --double_steps 40 --gallery_batch 48 \
  --conditions uniform opt_transfer opt_whitebox \
  --output $OUT/r10_clip_alloc.csv"

# --- GPU 1: the gallery-size sweep -----------------------------------------
# Identical queries at every level; the gallery grows by whole cities, which is
# the only direction that stays correctly labelled on this benchmark.
#
# Thirty-two runs on one card, but not thirty-two at once: each holds four or
# five embedders and a gallery index, and starting them together would thrash
# a 32 GB card for no throughput. They are dealt into WORKERS shells that each
# work through their share in sequence.
WORKERS="${WORKERS:-4}"
JOBS=()
for city in boston manila cph toronto; do
  for lvl in 1 2 4 8; do
    for bb in resnet18 mixvpr; do
      JOBS+=("$bb:$city:$lvl")
    done
  done
done

for w in $(seq 0 $((WORKERS - 1))); do
  name="r10_sweep_w$w"
  if screen -list 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
    echo "[skip] $name already running"; continue
  fi
  script="$LOGS/${name}.sh"
  { echo "cd $REPO"; echo "export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"; } > "$script"
  i=0
  for job in "${JOBS[@]}"; do
    if [ $((i % WORKERS)) -eq "$w" ]; then
      bb="${job%%:*}"; rest="${job#*:}"; city="${rest%%:*}"; lvl="${rest##*:}"
      sur="resnet50 vgg16 cosplace"; cond="transfer_3"
      if [ "$bb" = mixvpr ]; then sur="resnet18 resnet50 vgg16 cosplace"; cond="transfer_4"; fi
      tag="r10_sw_${bb}_${city}_L${lvl}"
      echo "$PY src/scripts/run_direction_transfer_study.py \\
        --manifest $ROOT/sweep_${city}_L${lvl}.jsonl --root $ROOT \\
        --eval_backbone $bb --surrogates $sur \\
        --objective self --seeds $SEEDS --gallery_batch 96 \\
        --conditions isotropic $cond \\
        --output $OUT/${tag}.csv >> $LOGS/${name}.log 2>&1" >> "$script"
    fi
    i=$((i + 1))
  done
  echo "echo ALLDONE >> $LOGS/${name}.log" >> "$script"
  screen -dmS "$name" bash "$script"
  echo "[start] gpu1 $name ($(grep -c run_direction "$script") runs)"
done

sleep 6
screen -list || true
