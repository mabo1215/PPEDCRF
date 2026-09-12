#!/usr/bin/env bash
# R4: does either axis reach a model that emits a coordinate instead of a
# neighbour?
#
# The gate first, because it decides whether the arm means anything. On the 400
# MSLS query frames at this paper's working resolution, GeoCLIP places 54.5% of
# them inside 25 km and 69.0% inside 200 km, at a median error of 18.4 km. That
# is a capable attacker on a movable benchmark, so a null here would be a null
# and not a floor. Street level is a different story -- 6.0% inside 1 km -- so
# 25 km is the primary threshold and 1 km is reported without being leaned on.
#
# Nine jobs: three seeds by three cost classes. The cheap class needs no
# optimisation at all, the directional class runs twenty sign-gradient steps
# over three surrogates, and the hardened class adds expectation over four
# transforms at two samples a step. Splitting them keeps a cheap job from
# waiting behind an expensive one and gives one writer per output file.
#
# GeoCLIP is never in the optimiser: the perturbations are the retrieval
# study's, built against {ResNet50, VGG16, CosPlace} and rebuilt here from the
# same seeds at the same delivered MSE. Only the reader changes, which is what
# makes this a transfer measurement rather than a white-box bound.
#
# network_turbo is sourced per job because GeoCLIP's encoder resolves the CLIP
# processor config through Hugging Face, which is otherwise unreachable here.
set -euo pipefail

REPO=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
PY="$REPO/.venv_ppedcrf/bin/python"
MANIFEST="$REPO/data/msls/manifest_all8.jsonl"
ROOT="$REPO/data/msls"
OUT="$REPO/src/outputs/r16"
LOGS="$REPO/logs_r16"

WORKERS="${WORKERS:-6}"
WAIT_FREE_MB="${WAIT_FREE_MB:-9000}"
STAGGER="${STAGGER:-40}"

mkdir -p "$OUT" "$LOGS"
cd "$REPO"

free_mb=$(df -Pm "$OUT" | awk 'NR==2 {print $4}')
free_inodes=$(df -Pi "$OUT" | awk 'NR==2 {print $4}')
echo "[space] $free_mb MB and $free_inodes inodes free under $OUT"
if [ "$free_mb" -lt 2000 ] || [ "$free_inodes" -lt 20000 ]; then
  echo "[abort] too little space or too few inodes." >&2; exit 1
fi

# name seed conditions
# Longest first: hardened carries the EOT samples, direction the bare twenty
# steps, and the cheap arm no optimiser at all.
JOBS=(
  "r16_hard_s1234 1234 hardened"
  "r16_hard_s1235 1235 hardened"
  "r16_hard_s1236 1236 hardened"
  "r16_dir_s1234  1234 direction"
  "r16_dir_s1235  1235 direction"
  "r16_dir_s1236  1236 direction"
  "r16_cheap_s1234 1234 clean|isotropic|edge|saliency"
  "r16_cheap_s1235 1235 isotropic|edge|saliency"
  "r16_cheap_s1236 1236 isotropic|edge|saliency"
)

for w in $(seq 0 $((WORKERS - 1))); do
  name="r16_worker$w"
  if screen -list 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
    echo "[skip] $name already running"; continue
  fi
  script="$LOGS/${name}.sh"
  { echo "cd $REPO"
    echo "source /etc/network_turbo >/dev/null 2>&1"
    echo "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  } > "$script"
  i=0; n=0
  for job in "${JOBS[@]}"; do
    if [ $((i % WORKERS)) -eq "$w" ]; then
      read -r jn sd conds <<< "$job"
      condlist=${conds//|/ }
      cat >> "$script" <<INNER
if pgrep -f "$OUT/${jn}.csv" >/dev/null 2>&1; then
  echo "[skip] ${jn}: already being written" >> $LOGS/${name}.log
else
  while [ \$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt $WAIT_FREE_MB ]; do sleep 60; done
  echo "[start] ${jn} \$(date +%H:%M:%S)" >> $LOGS/${name}.log
  $PY src/scripts/run_geolocator_study.py \\
    --manifest $MANIFEST --root $ROOT \\
    --seeds $sd --conditions $condlist \\
    --output $OUT/${jn}.csv >> $LOGS/${name}.log 2>&1
  echo "DONE ${jn} exit=\$? \$(date +%H:%M:%S)" >> $LOGS/${name}.log
fi
INNER
      n=$((n + 1))
    fi
    i=$((i + 1))
  done
  echo "echo ALLDONE >> $LOGS/${name}.log" >> "$script"
  screen -dmS "$name" bash "$script"
  echo "[start] $name ($n jobs)"
  [ "$w" -lt $((WORKERS - 1)) ] && sleep "$STAGGER"
done

sleep 6
screen -list || true
