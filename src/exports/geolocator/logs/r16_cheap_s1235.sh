cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
source /etc/network_turbo >/dev/null 2>&1
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[start] r16_cheap_s1235 $(date +%H:%M:%S)" >> logs_r16/r16_cheap_s1235.log
.venv_ppedcrf/bin/python src/scripts/run_geolocator_study.py   --manifest data/msls/manifest_all8.jsonl --root data/msls   --seeds 1235 --conditions isotropic edge saliency   --output src/outputs/r16/r16_cheap_s1235.csv >> logs_r16/r16_cheap_s1235.log 2>&1
echo "DONE r16_cheap_s1235 exit=$? $(date +%H:%M:%S)" >> logs_r16/r16_cheap_s1235.log
echo ALLDONE >> logs_r16/r16_cheap_s1235.log
