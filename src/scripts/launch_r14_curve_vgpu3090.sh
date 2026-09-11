#!/usr/bin/env bash
# The convergence curve the thirteenth review asked for and the fourteenth
# cycle left open.
#
# The paper currently bounds the allocation axis at a twenty-step budget and
# says so, because twenty and forty are the only two points it has -- and at
# forty every one of the five arms is still moving (-0.0200, -0.0108, -0.0408,
# -0.0025, -0.0567 against -0.0058, +0.0058, -0.0242, +0.0000, -0.0508). A
# reader is entitled to ask whether the two nulls are a plateau or the foot of
# a ramp, and two points cannot answer it.
#
# Five points per attacker: 5, 10, 20, 40, 80. Twenty and forty already exist,
# so each attacker needs two more runs:
#
#   A: --steps 5  --double_steps 10   -> the 5 and 10 points, ten steps of work
#   B: --steps 20 --double_steps 80   -> the 80 point, and the 20 point again
#
# Run B re-measures twenty on purpose. The optimiser's trajectory depends only
# on the seed and on the step index -- the per-step noise field is drawn from
# (query, seed, "opt<step>") and the checkpoint list does not enter the loop --
# so the first twenty steps of an eighty-step run are the same twenty steps.
# The reproduced value is therefore a gate on the whole curve: if it does not
# land on the existing one to within the run-to-run spread the paper already
# reports (about 0.005, the backward pass through the surrogates not being
# bitwise deterministic), the five points are not on one trajectory and the
# curve should not be drawn.
#
# Ten jobs, one card. They are dealt into WORKERS shells that each work through
# their share in sequence rather than starting together: each job holds four
# embedders and a 2,000-image gallery index, and ten at once would thrash the
# allocator during gallery embedding for no throughput. Longest first, so the
# 480 px Patch-NetVLAD and the ViT-L CLIP runs are not left to the end.
#
# Every job is resumable: rows are flushed and fsynced per completed
# (query, condition, seed) and completed keys are skipped at startup.
set -euo pipefail

REPO=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
PY="$REPO/.venv_ppedcrf/bin/python"
MANIFEST="$REPO/data/msls/manifest_all8.jsonl"
ROOT="$REPO/data/msls"
OUT="$REPO/src/outputs/r14"
LOGS="$REPO/logs_r14"
SEEDS="1234 5678 9012"
# One worker per backbone, so five jobs hold the card at once and each worker
# loads its embedders once and then runs its second job. The card is 48 GB and
# the r13 pair used 10 GB between them at half these batches, so the budget
# below is deliberately generous: the constraint that matters here is the
# eighty-step Patch-NetVLAD run on the critical path, and the way to shorten it
# is to give it queries per step rather than to hold memory in reserve.
WORKERS="${WORKERS:-5}"
# A job waits for this much free memory before it starts. Five jobs seeing an
# empty card would all start at once and the estimate below would be wrong for
# all of them, so the workers are also staggered on launch.
# 7000 was too low: a job that sees seven gigabytes free starts, and by the
# time it has built its gallery index the neighbours it started alongside have
# grown into the rest. Ten concurrent jobs at the batches below peaked at 47.2
# of 47.4 GiB and the tenth died in a conv. The gate now asks for a window big
# enough to hold a whole job.
WAIT_FREE_MB="${WAIT_FREE_MB:-11000}"
STAGGER="${STAGGER:-75}"

mkdir -p "$OUT" "$LOGS"
cd "$REPO"

free_mb=$(df -Pm "$OUT" | awk 'NR==2 {print $4}')
free_inodes=$(df -Pi "$OUT" | awk 'NR==2 {print $4}')
echo "[space] $free_mb MB and $free_inodes inodes free under $OUT"
if [ "$free_mb" -lt 2000 ] || [ "$free_inodes" -lt 20000 ]; then
  echo "[abort] too little space or too few inodes; free some before running." >&2
  exit 1
fi

# name backbone query_batch gallery_batch steps double_steps
# Longest first: the 80-step runs before the 10-step ones, and within each the
# heavier trunks first.
# Batches are roughly twice what r13 used, which measured 6 GB for
# Patch-NetVLAD at batch 4 and 4 GB for the ViT at batch 8. Five concurrent
# jobs at these settings should sit near 40 GB of the 48, and the gate above
# catches the case where the estimate is wrong.
# Measured, not estimated: at the batches this array first carried, the ten
# jobs sat at 3.99, 7.34, 6.89, 9.41, 3.99, 7.34, 6.89 and 1.40 GiB with the
# card at 47.2 of 47.4. That is not headroom, it is the edge, and the job that
# arrived last died in a conv allocating 294 MiB. These are about three
# quarters of those, which leaves a job's worth of slack on a full card.
# CLIP is listed first because it is the long pole once it finally starts.
JOBS=(
  "r14_clip_s80 clip_vitl14   6  48 20 80"
  "r14_pnv_s80  patchnetvlad  6  32 20 80"
  "r14_mix_s80  mixvpr       12  64 20 80"
  "r14_vit_s80  vit_b_16     12  64 20 80"
  "r14_r18_s80  resnet18     16  96 20 80"
  "r14_clip_s10 clip_vitl14   6  48  5 10"
  "r14_pnv_s10  patchnetvlad  6  32  5 10"
  "r14_mix_s10  mixvpr       12  64  5 10"
  "r14_vit_s10  vit_b_16     12  64  5 10"
  "r14_r18_s10  resnet18     16  96  5 10"
)

# MixVPR is the one attacker whose surrogate ensemble differs: it is the
# evaluation target, so ResNet18 joins the other three. Every other arm is
# solved against the same {resnet50, vgg16, cosplace} and is therefore the same
# map, which is what lets four of the five curves be read as one object.
surrogates_for() {
  case "$1" in
    mixvpr) echo "resnet18 resnet50 vgg16 cosplace" ;;
    *)      echo "resnet50 vgg16 cosplace" ;;
  esac
}

for w in $(seq 0 $((WORKERS - 1))); do
  name="r14_worker$w"
  if screen -list 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
    echo "[skip] $name already running"; continue
  fi
  script="$LOGS/${name}.sh"
  { echo "cd $REPO"
    echo "export OMP_NUM_THREADS=9 MKL_NUM_THREADS=9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  } > "$script"
  i=0; n=0
  for job in "${JOBS[@]}"; do
    if [ $((i % WORKERS)) -eq "$w" ]; then
      read -r jn bb qb gb st ds <<< "$job"
      # One writer per output file is the invariant, not one screen per name.
      cat >> "$script" <<EOF
if pgrep -f "$OUT/${jn}.csv" >/dev/null 2>&1; then
  echo "[skip] ${jn}: a process is already writing its output" >> $LOGS/${name}.log
else
  while [ \$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt $WAIT_FREE_MB ]; do sleep 60; done
  echo "[start] ${jn} \$(date +%H:%M:%S)" >> $LOGS/${name}.log
  $PY src/scripts/run_optimised_allocation_study.py \\
    --manifest $MANIFEST --root $ROOT \\
    --eval_backbone $bb --surrogates $(surrogates_for "$bb") \\
    --noise_mode expectation --seeds $SEEDS \\
    --batch $qb --gallery_batch $gb \\
    --steps $st --double_steps $ds \\
    --conditions uniform opt_transfer \\
    --output $OUT/${jn}.csv >> $LOGS/${name}.log 2>&1
  echo "DONE ${jn} exit=\$?" >> $LOGS/${name}.log
fi
EOF
      n=$((n + 1))
    fi
    i=$((i + 1))
  done
  echo "echo ALLDONE >> $LOGS/${name}.log" >> "$script"
  screen -dmS "$name" bash "$script"
  echo "[start] $name ($n jobs)"
  # Five workers launched together would each read an empty card and size
  # themselves for it. Staggering lets each see what the previous one took.
  [ "$w" -lt $((WORKERS - 1)) ] && sleep "$STAGGER"
done

sleep 6
screen -list || true
