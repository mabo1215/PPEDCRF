cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
source /etc/network_turbo >/dev/null 2>&1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if pgrep -f "/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/src/outputs/r16/r16_hard_s1235.csv" >/dev/null 2>&1; then
  echo "[skip] r16_hard_s1235: already being written" >> /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/logs_r16/r16_worker1.log
else
  while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 9000 ]; do sleep 60; done
  echo "[start] r16_hard_s1235 $(date +%H:%M:%S)" >> /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/logs_r16/r16_worker1.log
  /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/.venv_ppedcrf/bin/python src/scripts/run_geolocator_study.py \
    --manifest /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/data/msls/manifest_all8.jsonl --root /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/data/msls \
    --seeds 1235 --conditions hardened \
    --output /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/src/outputs/r16/r16_hard_s1235.csv >> /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/logs_r16/r16_worker1.log 2>&1
  echo "DONE r16_hard_s1235 exit=$? $(date +%H:%M:%S)" >> /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/logs_r16/r16_worker1.log
fi
if pgrep -f "/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/src/outputs/r16/r16_cheap_s1235.csv" >/dev/null 2>&1; then
  echo "[skip] r16_cheap_s1235: already being written" >> /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/logs_r16/r16_worker1.log
else
  while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 9000 ]; do sleep 60; done
  echo "[start] r16_cheap_s1235 $(date +%H:%M:%S)" >> /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/logs_r16/r16_worker1.log
  /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/.venv_ppedcrf/bin/python src/scripts/run_geolocator_study.py \
    --manifest /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/data/msls/manifest_all8.jsonl --root /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/data/msls \
    --seeds 1235 --conditions isotropic edge saliency \
    --output /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/src/outputs/r16/r16_cheap_s1235.csv >> /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/logs_r16/r16_worker1.log 2>&1
  echo "DONE r16_cheap_s1235 exit=$? $(date +%H:%M:%S)" >> /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/logs_r16/r16_worker1.log
fi
echo ALLDONE >> /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/logs_r16/r16_worker1.log
