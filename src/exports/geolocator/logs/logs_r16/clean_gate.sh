cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
source /etc/network_turbo >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
.venv_ppedcrf/bin/python src/scripts/run_geolocator_study.py   --manifest data/msls/manifest_all8.jsonl --root data/msls   --seeds 1234 --conditions clean   --output src/outputs/r16_gate/geo_clean.csv > logs_r16/clean_gate.log 2>&1
echo "GATE_EXIT=$?" >> logs_r16/clean_gate.log
