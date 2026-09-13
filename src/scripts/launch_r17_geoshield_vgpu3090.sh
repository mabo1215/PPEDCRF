#!/usr/bin/env bash
# Run the public GeoShield release through this audit's protocol on vGPU 3090.
#
# This is R3's strong form, bounded by what the release actually ships. The
# geo-semantic term is degenerate in the published code -- the stub returns
# one constant caption for every frame -- so the arm below is labelled "as
# published" and measures the released artifact, not the paper's method. The
# runner asserts that constant at startup and refuses to continue if a future
# checkout changes it, so the label cannot silently go stale.
#
# A pilot runs first and the full arm waits on it. The release costs far more
# per frame than anything else in this repository: 100 FGSM steps through a
# three-model CLIP ensemble that includes LAION ViT-G/14, at 640px, one image
# at a time. Paying for 1,200 of those before knowing the per-frame cost, or
# before knowing the perturbation is non-zero, would be the expensive kind of
# mistake.
#
# The worker guard is an atomic mkdir lock rather than the pgrep test used by
# the geolocator launcher: that test lost a race when jobs were hand-packed
# into idle workers, two processes wrote one file, and the duplicate rows
# exposed a seeding bug. A lock directory cannot be won twice.
set -euo pipefail

REPO=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
PY="$REPO/.venv_ppedcrf/bin/python"
MANIFEST="$REPO/data/msls/manifest_all8.jsonl"
ROOT="$REPO/data/msls"
OUT="$REPO/src/outputs/r17_geoshield"
LOGS="$REPO/logs_r17"
LOCKS="$OUT/.locks"
THIRD="$REPO/src/third_party/Geoshield"

WORKERS="${WORKERS:-3}"
WAIT_FREE_MB="${WAIT_FREE_MB:-20000}"
STAGGER="${STAGGER:-45}"
PILOT_QUERIES="${PILOT_QUERIES:-25}"

mkdir -p "$OUT" "$LOGS" "$LOCKS"
cd "$REPO"

# --- preconditions -------------------------------------------------------
free_mb=$(df -Pm "$OUT" | awk 'NR==2 {print $4}')
free_inodes=$(df -Pi "$OUT" | awk 'NR==2 {print $4}')
echo "[space] $free_mb MB and $free_inodes inodes free under $OUT"
if [ "$free_mb" -lt 5000 ] || [ "$free_inodes" -lt 20000 ]; then
  echo "[abort] too little space or too few inodes." >&2; exit 1
fi

if [ ! -d "$THIRD" ]; then
  echo "[setup] cloning GeoShield"
  source /etc/network_turbo >/dev/null 2>&1 || true
  git clone https://github.com/thinwayliu/Geoshield.git "$THIRD"
fi
echo "[provenance] GeoShield at $(git -C "$THIRD" rev-parse HEAD)"

# The release names a requirements.txt it does not ship, so the deps its
# imports need are installed explicitly and recorded here.
source /etc/network_turbo >/dev/null 2>&1 || true
$PY -m pip install --quiet hydra-core omegaconf wandb || {
  echo "[abort] could not install the released code's dependencies" >&2; exit 1; }

# Fail before the GPU is paid for, not after: import the release, assert the
# published caption, and stop.
$PY src/scripts/run_geoshield_audit.py --smoke \
  --manifest "$MANIFEST" --root "$ROOT" --output /dev/null || {
  echo "[abort] the released code did not import, or the stub changed" >&2
  exit 1; }

# --- pilot ---------------------------------------------------------------
# One seed, a few queries, run in the foreground. If the perturbation is zero,
# or a frame costs more than the budget allows, this is where it shows.
PILOT="$OUT/pilot_s1234.csv"
if [ ! -s "$PILOT" ]; then
  echo "[pilot] $PILOT_QUERIES queries, seed 1234, foreground"
  start=$(date +%s)
  $PY src/scripts/run_geoshield_audit.py \
    --manifest "$MANIFEST" --root "$ROOT" \
    --seeds 1234 --limit "$PILOT_QUERIES" \
    --caption_source published \
    --output "$PILOT" 2>&1 | tee "$LOGS/pilot.log"
  echo "[pilot] $(( ($(date +%s) - start) / PILOT_QUERIES ))s per frame"
  echo "[pilot] review $PILOT before launching the full arm, then rerun"
  echo "[pilot] with SKIP_PILOT=1 to continue."
  [ "${SKIP_PILOT:-0}" = "1" ] || exit 0
fi

# --- full arm ------------------------------------------------------------
# name seed
JOBS=(
  "r17_gs_s1234 1234"
  "r17_gs_s1235 1235"
  "r17_gs_s1236 1236"
)

for w in $(seq 0 $((WORKERS - 1))); do
  name="r17_worker$w"
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
      read -r jn sd <<< "$job"
      cat >> "$script" <<INNER
if ! mkdir "$LOCKS/${jn}" 2>/dev/null; then
  echo "[skip] ${jn}: another worker holds its lock" >> $LOGS/${name}.log
else
  while [ \$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt $WAIT_FREE_MB ]; do sleep 60; done
  echo "[start] ${jn} \$(date +%H:%M:%S)" >> $LOGS/${name}.log
  $PY src/scripts/run_geoshield_audit.py \\
    --manifest $MANIFEST --root $ROOT \\
    --seeds $sd --caption_source published \\
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

echo "[launched] tail -f $LOGS/r17_worker0.log"
