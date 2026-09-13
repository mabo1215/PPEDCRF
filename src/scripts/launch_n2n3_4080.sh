#!/usr/bin/env bash
# N2 and N3 on the two-card 4080 host.
#
# Eight jobs over two 32 GiB cards. They are not interchangeable: the two N2
# jobs carry a CLIP ViT-L/14 geolocator plus three direction surrogates and are
# the memory-heavy ones, while the six N3 jobs fine-tune a ResNet18 and are
# comparatively small. So each card gets exactly one N2 job and three N3 jobs
# rather than dealing them round-robin, which would put both heavy jobs on one
# card and leave the other idle.
#
# Every job waits for free memory on ITS OWN card before starting. Reading
# global free memory would let a job on card 0 start because card 1 happened to
# be empty, which is how a two-card host OOMs while looking healthy.
#
# The four direction-trained N3 jobs share one perturbation cache. Rebuilding
# twenty sign-gradient steps over three surrogates per sample, four times, is
# the single largest avoidable cost here.
set -euo pipefail

REPO=/root/autodl-tmp/PPEDCRF
PY="$REPO/.venv_ppedcrf/bin/python"
MANIFEST="$REPO/data/msls/manifest_all8.jsonl"
ROOT="$REPO/data/msls"
OUT="$REPO/src/outputs/n2n3"
FRAMES="$OUT/frames"
CKPT="$OUT/ckpt"
DCACHE="$OUT/dircache"
LOGS="$REPO/logs_n2n3"
WAIT_FREE_MB="${WAIT_FREE_MB:-7000}"
STAGGER="${STAGGER:-25}"

mkdir -p "$OUT" "$FRAMES" "$CKPT" "$DCACHE" "$LOGS"
cd "$REPO"

free_mb=$(df -Pm "$OUT" | awk 'NR==2 {print $4}')
free_in=$(df -Pi "$OUT" | awk 'NR==2 {print $4}')
echo "[space] ${free_mb} MB and ${free_in} inodes free under $OUT"
# 1200 released PNGs plus six checkpoints; inodes matter as much as bytes on a
# shared autodl volume, where a quota can be hit while df -h still looks fine.
if [ "$free_mb" -lt 6000 ] || [ "$free_in" -lt 50000 ]; then
  echo "[abort] too little space or too few inodes." >&2; exit 1
fi

# name | gpu | command
JOBS=()

# --- N2: released frames, saved as the exact bytes the geolocator reads ------
JOBS+=("n2_cheap|0|$PY src/scripts/run_geolocator_study.py \
  --manifest $MANIFEST --root $ROOT --seeds 1234 \
  --conditions clean isotropic --save_frames $FRAMES \
  --output $OUT/n2_cheap.csv")
JOBS+=("n2_dir|1|$PY src/scripts/run_geolocator_study.py \
  --manifest $MANIFEST --root $ROOT --seeds 1234 \
  --conditions direction --save_frames $FRAMES \
  --output $OUT/n2_dir.csv")

# --- N3: the adaptive attacker at two unfreeze budgets -----------------------
# k=5 is the fully unfrozen encoder the review asks for. k=1 is the published
# arm's setting, re-run here rather than quoted from the earlier run: claiming
# "unfreezing the whole encoder does not change the conclusion" requires the
# two budgets to differ in nothing but the budget, and the published k=1 numbers
# come from a different run with its own split and seed. A control that has to
# be compared across runs is not a control.
#
# The exposure name is 'hardened_direction', not 'hardened' -- the earlier
# launch failed on exactly that, because the preflight checked that flags
# existed without checking that the values were in range.
# Card assignment alternates on the HEAVY jobs only. Alternating on every job
# aliases with the two-valued mode loop and sends every 'stock' job to card 0
# and every 'rebuilt' job to card 1 -- which is what the first launch did,
# leaving card 0 at 12% while card 1 sat at 99%. 'rebuilt' re-embeds the
# gallery at every validation and costs far more than 'stock', so the rebuilt
# jobs are what has to be split evenly; the cheap ones then fill the gaps.
heavy=0; light=0
SEEDS="${SEEDS:-1234 1235}"
for k in 5 1; do
  for sd in $SEEDS; do
    for exposure in isotropic direction hardened_direction; do
      for mode in stock rebuilt; do
        flag=""; cache=""
        if [ "$mode" = "rebuilt" ]; then
          flag="--rebuild_gallery"; g=$heavy; heavy=$((1 - heavy))
        else
          g=$light; light=$((1 - light))
        fi
        [ "$exposure" != "isotropic" ] && cache="--direction_cache $DCACHE"
        n="n3_k${k}_s${sd}_${exposure}_${mode}"
        JOBS+=("$n|$g|$PY src/scripts/finetune_adaptive_attacker.py \
          --manifest $MANIFEST --root $ROOT --seed $sd \
          --train_perturbation $exposure --unfreeze_blocks $k $flag $cache \
          --output $CKPT/${n}.pt \
          --test_ids_output $OUT/${n}_testids.json")
      done
    done
  done
done

for entry in "${JOBS[@]}"; do
  IFS='|' read -r name gpu cmd <<< "$entry"
  if screen -list 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
    echo "[skip] $name already running"; continue
  fi
  # A finished job leaves no screen, so without this a re-launch silently
  # redoes completed work. Only exit=0 counts -- a failed job should re-run.
  if grep -q "^DONE ${name} exit=0" "$LOGS/${name}.log" 2>/dev/null; then
    echo "[skip] $name already finished"; continue
  fi
  script="$LOGS/${name}.sh"
  {
    echo "cd $REPO"
    echo "source /etc/network_turbo >/dev/null 2>&1"
    echo "export CUDA_VISIBLE_DEVICES=$gpu"
    echo "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    # Per-card wait. -i selects the physical card, which CUDA_VISIBLE_DEVICES
    # has already renumbered for the process but not for nvidia-smi.
    echo "while [ \$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits) -lt $WAIT_FREE_MB ]; do sleep 45; done"
    echo "echo \"[start] $name gpu=$gpu \$(date +%H:%M:%S)\" >> $LOGS/${name}.log"
    echo "$cmd >> $LOGS/${name}.log 2>&1"
    echo "echo \"DONE $name exit=\$? \$(date +%H:%M:%S)\" >> $LOGS/${name}.log"
  } > "$script"
  screen -dmS "$name" bash "$script"
  echo "[start] $name on gpu $gpu"
  sleep "$STAGGER"
done

echo "[launch] $(screen -list 2>/dev/null | grep -c 'n2_\|n3_') sessions up"
