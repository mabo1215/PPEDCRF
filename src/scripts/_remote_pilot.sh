#!/usr/bin/env bash
# Start the GeoShield pilot detached, so weight downloads and the first frames
# run without holding an SSH session open.
set -e
REPO=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
cd "$REPO"

if screen -list 2>/dev/null | grep -q "\.r17_pilot[[:space:]]"; then
  echo "[pilot] already running"
  exit 0
fi

mkdir -p logs_r17
cat > logs_r17/pilot_runner.sh <<'INNER'
cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
source /etc/network_turbo >/dev/null 2>&1 || true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/.venv_ppedcrf/bin/python
echo "[pilot] start $(date +%H:%M:%S)"
$PY src/scripts/run_geoshield_audit.py \
  --manifest data/msls/manifest_all8.jsonl \
  --root data/msls \
  --seeds 1234 --limit 25 \
  --caption_source published \
  --output src/outputs/r17_geoshield/pilot_s1234.csv
echo "[pilot] DONE exit=$? $(date +%H:%M:%S)"
INNER

screen -dmS r17_pilot bash -c "bash logs_r17/pilot_runner.sh > logs_r17/pilot.log 2>&1"
echo "[pilot] launched; log: logs_r17/pilot.log"
sleep 20
tail -5 "$REPO/logs_r17/pilot.log" 2>/dev/null || echo "[pilot] (log not yet written)"
