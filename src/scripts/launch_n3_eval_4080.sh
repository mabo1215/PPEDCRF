#!/usr/bin/env bash
# Evaluate the N3 checkpoints. This is the step that produces numbers.
#
# Fine-tuning wrote checkpoints and held-out query ids; neither is a result.
# The published adaptive table is built from retrieval evaluations of those
# checkpoints on the held-out queries, which is what this runs -- once per
# checkpoint, plus one unadapted baseline per split to subtract against.
#
# Each eval reads the split its own checkpoint was trained against
# (--query_id_file), so an adapted model is never scored on queries it saw.
# Getting that wrong would inflate every adapted row and would look like the
# adaptation working.
set -euo pipefail

REPO=/root/autodl-tmp/PPEDCRF
PY="$REPO/.venv_ppedcrf/bin/python"
MANIFEST="$REPO/data/msls/manifest_all8.jsonl"
ROOT="$REPO/data/msls"
OUT="$REPO/src/outputs/n2n3"
CKPT="$OUT/ckpt"
EVAL="$REPO/src/outputs/n3_eval"
LOGS="$REPO/logs_n3_eval"
WAIT_FREE_MB="${WAIT_FREE_MB:-7000}"
STAGGER="${STAGGER:-10}"

mkdir -p "$EVAL" "$LOGS"
cd "$REPO"

JOBS=()
heavy=0

# One eval per checkpoint.
for ck in "$CKPT"/n3_*.pt; do
  [ -e "$ck" ] || continue
  n=$(basename "$ck" .pt)
  ids="$OUT/${n}_testids.json"
  [ -f "$ids" ] || { echo "[warn] no test ids for $n, skipping"; continue; }
  reb=""
  case "$n" in *_rebuilt) reb="--eval_rebuild_gallery";; esac
  g=$heavy; heavy=$((1 - heavy))
  JOBS+=("ev_${n}|$g|$PY src/scripts/run_direction_transfer_study.py \
    --manifest $MANIFEST --root $ROOT --seeds 1234 \
    --conditions isotropic transfer_3 \
    --query_id_file $ids --eval_checkpoint $ck $reb \
    --output $EVAL/${n}.csv")
done

# One unadapted baseline per distinct split, so every adapted row has a
# same-split control. Splits differ by (seed), so one baseline per seed.
for ids in "$OUT"/n3_k5_isotropic_stock_testids.json \
           "$OUT"/n3_k5_s1235_isotropic_stock_testids.json; do
  [ -f "$ids" ] || continue
  tag=$(basename "$ids" _testids.json | sed 's/n3_k5_//; s/_isotropic_stock//')
  [ -z "$tag" ] && tag="s1234"
  g=$heavy; heavy=$((1 - heavy))
  JOBS+=("ev_baseline_${tag}|$g|$PY src/scripts/run_direction_transfer_study.py \
    --manifest $MANIFEST --root $ROOT --seeds 1234 \
    --conditions isotropic transfer_3 \
    --query_id_file $ids \
    --output $EVAL/baseline_${tag}.csv")
done

echo "[eval] ${#JOBS[@]} jobs"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r name gpu cmd <<< "$entry"
  screen -list 2>/dev/null | grep -q "\.${name}[[:space:]]" && { echo "[skip] $name running"; continue; }
  grep -q "^DONE ${name} exit=0" "$LOGS/${name}.log" 2>/dev/null && { echo "[skip] $name done"; continue; }
  script="$LOGS/${name}.sh"
  {
    echo "cd $REPO"
    echo "source /etc/network_turbo >/dev/null 2>&1"
    echo "export CUDA_VISIBLE_DEVICES=$gpu"
    echo "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo "while [ \$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits) -lt $WAIT_FREE_MB ]; do sleep 30; done"
    echo "echo \"[start] $name gpu=$gpu \$(date +%H:%M:%S)\" >> $LOGS/${name}.log"
    echo "$cmd >> $LOGS/${name}.log 2>&1"
    echo "echo \"DONE $name exit=\$? \$(date +%H:%M:%S)\" >> $LOGS/${name}.log"
  } > "$script"
  screen -dmS "$name" bash "$script"
  echo "[start] $name gpu $gpu"
  sleep "$STAGGER"
done
echo "[eval] $(screen -list 2>/dev/null | grep -c '\.ev_') sessions up"
