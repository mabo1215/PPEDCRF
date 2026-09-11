#!/usr/bin/env bash
# Thirteenth-cycle review, the one experiment it asked for.
#
# The review's first finding was that the surrogate-solved allocation map --
# the arm the abstract calls "buys nothing" -- is one object: it is optimised
# against {ResNet50, VGG16, CosPlace} and never sees the model it is evaluated
# against. Read by ResNet18 it buys -0.006; read by CLIP ViT-L/14 it buys
# -0.051 with a place-clustered interval clear of zero. Two of the paper's five
# held-out attackers had no allocation arm at all, so "one of three" was the
# honest statement and also an obviously incomplete one.
#
# This closes it. Same manifest, same three seeds, same delivered MSE, same
# energy gate, same surrogate ensemble, same twenty-and-forty step budget --
# only the attacker that reads the released frame changes. Because the
# surrogates are identical to the ResNet18 and CLIP runs', the map solved here
# is the same map: the objective trace should read 2.841 -> 2.465 again, which
# is the check that this is a re-embedding and not a new search.
#
# Surrogates stay {resnet50, vgg16, cosplace} for both attackers, matching what
# their own direction arms used, so the allocation and direction contrasts on
# each attacker are comparable. Patch-NetVLAD shares a VGG16 trunk with a
# surrogate and the ViT shares nothing with any of them; that contrast is part
# of what the run is for.
#
# opt_whitebox is deliberately absent. It would solve a *different* map per
# attacker, which is a separate claim already established on three attackers,
# and backward passes through Patch-NetVLAD at 480 px are what would make this
# an overnight job instead of an afternoon one.
#
# Both jobs are resumable: rows are flushed and fsynced per completed
# (query, condition, seed) and completed keys are skipped at startup, so a job
# killed for the card is relaunched with the same command and continues.
set -euo pipefail

REPO=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
PY="$REPO/.venv_ppedcrf/bin/python"
MANIFEST="$REPO/data/msls/manifest_all8.jsonl"
ROOT="$REPO/data/msls"
OUT="$REPO/src/outputs/r13"
LOGS="$REPO/logs_r13"
SEEDS="1234 5678 9012"

mkdir -p "$OUT" "$LOGS"
cd "$REPO"

# This volume is shared with a dozen unrelated projects and has a fixed inode
# quota, so a byte-space check alone has been misleading here before: df -h can
# look healthy while df -i is exhausted and every write fails with a
# "No space left on device" that names the wrong resource.
free_mb=$(df -Pm "$OUT" | awk 'NR==2 {print $4}')
free_inodes=$(df -Pi "$OUT" | awk 'NR==2 {print $4}')
echo "[space] $free_mb MB and $free_inodes inodes free under $OUT"
if [ "$free_mb" -lt 2000 ] || [ "$free_inodes" -lt 20000 ]; then
  echo "[abort] too little space or too few inodes; free some before running." >&2
  exit 1
fi

# The card is shared. Twelve jobs racing for it during gallery embedding is
# what killed a ViT sweep last cycle, so each job waits for a real window
# rather than thrashing the allocator.
WAIT_FREE_MB="${WAIT_FREE_MB:-9000}"

launch() {  # name, command...
  local name="$1"; shift
  if screen -list 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
    echo "[skip] $name already running"; return
  fi
  # The invariant is one writer per output file, not one screen per name: two
  # processes appending to one export both read their resume set at startup,
  # so neither skips anything and the file ends up with duplicate keys that
  # disagree with each other.
  if pgrep -f "$OUT/${name}.csv" >/dev/null 2>&1; then
    echo "[skip] $name: a process is already writing $OUT/${name}.csv"; return
  fi
  screen -dmS "$name" bash -c "
    cd $REPO
    while [ \$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt $WAIT_FREE_MB ]; do sleep 120; done
    OMP_NUM_THREADS=9 MKL_NUM_THREADS=9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $* > $LOGS/${name}.log 2>&1
    echo EXIT=\$? >> $LOGS/${name}.log"
  echo "[start] $name"
}

alloc() {  # name, backbone, query_batch, gallery_batch
  local name="$1" backbone="$2" qb="$3" gb="$4"
  launch "$name" "$PY src/scripts/run_optimised_allocation_study.py \
    --manifest $MANIFEST --root $ROOT \
    --eval_backbone $backbone --surrogates resnet50 vgg16 cosplace \
    --noise_mode expectation --seeds $SEEDS \
    --batch $qb --gallery_batch $gb \
    --steps 20 --double_steps 40 \
    --conditions uniform opt_transfer \
    --output $OUT/${name}.csv"
}

# Patch-NetVLAD runs at 480 px and is the one that has to come down on both
# batches; the ViT is a 224 px trunk and takes the same settings as its own
# preprocessing run.
alloc r13_pnv_alloc patchnetvlad 4 24
alloc r13_vit_alloc vit_b_16     8 32

sleep 8
screen -list || true
