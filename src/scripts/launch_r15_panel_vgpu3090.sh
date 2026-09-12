#!/usr/bin/env bash
# Fourteenth-cycle review, finding R1: the two axes are not read by the same
# attackers.
#
# Every prescribed placement rule on the primary place-labelled manifest is
# reported against two attackers, ResNet18 and MixVPR. The direction arm and
# the solved maps are reported against five. That would be a presentational
# asymmetry if the missing three were arbitrary, and they are not: the two
# attackers on which a *solved* map separates from uniform -- Patch-NetVLAD and
# CLIP ViT-L/14 -- are exactly the two that no prescribed rule has ever been
# run against here, and CLIP has never appeared in a placement study at all. So
# the paper's central null is established precisely where allocation turns out
# not to work, and a referee is entitled to read the axis asymmetry as an
# artifact of the panel.
#
# This closes it. Nothing about the search changes -- prescribed placements
# involve no optimisation -- so this is a re-embedding of frames that already
# exist under three more attackers, at the same manifest, gallery, seeds,
# delivered distortion and energy gate as the ResNet18 and MixVPR runs.
#
# Ten placements per job: the uniform reference, the six prescribed rules the
# manuscript tabulates, the two gradient-guided rules, and the margin-gradient
# rule with its inverse that the margin analysis nominates. That is the same
# ten rows tab_placement_msls already prints for the weak attacker, so the new
# cells drop straight into the existing table shape.
#
# Nine jobs, one card, one output file per (attacker, seed) so there is exactly
# one writer per file. Every job is resumable: rows are flushed and fsynced per
# completed (query, placement, seed) and completed keys are skipped at startup,
# so a job killed for the card is relaunched with the same command.
set -euo pipefail

REPO=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
PY="$REPO/.venv_ppedcrf/bin/python"
MANIFEST="$REPO/data/msls/manifest_all8.jsonl"
ROOT="$REPO/data/msls"
OUT="$REPO/src/outputs/r15"
LOGS="$REPO/logs_r15"

# The ten rows tab_placement_msls prints, in its order.
PLACEMENTS="uniform learned oracle_grad anti_oracle_grad saliency center random_fixed edge margin_oracle anti_margin_oracle"

# Six workers over nine jobs. The r14 measurements on this card put a CLIP job
# near 9.4 GiB, Patch-NetVLAD near 6.9 and the ViT near 4.0 at these batches;
# six of those is about 40 of 48 GiB, which is the utilisation we want and
# still leaves a job's worth of slack. The gate below is what actually keeps
# it honest -- ten concurrent jobs once peaked at 47.2 of 47.4 GiB and the
# tenth died in a conv, so a job waits for a window big enough to hold itself
# rather than for any free memory at all.
WORKERS="${WORKERS:-6}"
WAIT_FREE_MB="${WAIT_FREE_MB:-10000}"
STAGGER="${STAGGER:-45}"

mkdir -p "$OUT" "$LOGS"
cd "$REPO"

free_mb=$(df -Pm "$OUT" | awk 'NR==2 {print $4}')
free_inodes=$(df -Pi "$OUT" | awk 'NR==2 {print $4}')
echo "[space] $free_mb MB and $free_inodes inodes free under $OUT"
if [ "$free_mb" -lt 2000 ] || [ "$free_inodes" -lt 20000 ]; then
  echo "[abort] too little space or too few inodes; free some before running." >&2
  exit 1
fi

# name backbone seed unused gallery_batch
# Longest first so the 480 px Patch-NetVLAD and the ViT-L CLIP runs are not
# left to the end; batches are the ones r14 measured on this card.
JOBS=(
  "r15_clip_s1234 clip_vitl14   1234  6 48"
  "r15_clip_s1235 clip_vitl14   1235  6 48"
  "r15_clip_s1236 clip_vitl14   1236  6 48"
  "r15_pnv_s1234  patchnetvlad  1234  6 32"
  "r15_pnv_s1235  patchnetvlad  1235  6 32"
  "r15_pnv_s1236  patchnetvlad  1236  6 32"
  "r15_vit_s1234  vit_b_16      1234 12 64"
  "r15_vit_s1235  vit_b_16      1235 12 64"
  "r15_vit_s1236  vit_b_16      1236 12 64"
)

for w in $(seq 0 $((WORKERS - 1))); do
  name="r15_worker$w"
  if screen -list 2>/dev/null | grep -q "\.${name}[[:space:]]"; then
    echo "[skip] $name already running"; continue
  fi
  script="$LOGS/${name}.sh"
  { echo "cd $REPO"
    echo "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  } > "$script"
  i=0; n=0
  for job in "${JOBS[@]}"; do
    if [ $((i % WORKERS)) -eq "$w" ]; then
      read -r jn bb sd qb gb <<< "$job"
      cat >> "$script" <<INNER
if pgrep -f "$OUT/${jn}\$" >/dev/null 2>&1; then
  echo "[skip] ${jn}: a process is already writing its output" >> $LOGS/${name}.log
else
  while [ \$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt $WAIT_FREE_MB ]; do sleep 60; done
  echo "[start] ${jn} \$(date +%H:%M:%S)" >> $LOGS/${name}.log
  $PY src/scripts/run_placement_rule_study.py \\
    --manifest $MANIFEST --root $ROOT \\
    --backbones $bb --placements $PLACEMENTS \\
    --seeds $sd --gallery_batch $gb \\
    --output_dir $OUT/${jn} >> $LOGS/${name}.log 2>&1
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
