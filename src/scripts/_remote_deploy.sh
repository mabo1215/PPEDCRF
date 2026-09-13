#!/usr/bin/env bash
# Move the three byte-identical untracked scripts aside (reversible; they were
# verified identical to the tracked versions first), fast-forward to the
# pushed commit, and verify the runner imports on the host.
set -e
REPO=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
cd "$REPO"
source /etc/network_turbo >/dev/null 2>&1 || true

BACKUP="$REPO/.pre_merge_backup_$(git rev-parse --short HEAD)"
mkdir -p "$BACKUP/src/scripts"
for f in src/scripts/launch_r15_panel_vgpu3090.sh \
         src/scripts/launch_r16_geolocator_vgpu3090.sh \
         src/scripts/run_geolocator_study.py; do
  [ -f "$f" ] && mv "$f" "$BACKUP/$f"
done
echo "[deploy] moved 3 identical scripts to $BACKUP"

git merge --ff-only origin/main
echo "[deploy] HEAD is now: $(git log --oneline -1)"

ls -l src/scripts/run_geoshield_audit.py src/scripts/launch_r17_geoshield_vgpu3090.sh

PY="$REPO/.venv_ppedcrf/bin/python"
echo "[deploy] installing the released code's deps (no requirements.txt is shipped)"
"$PY" -m pip install --quiet hydra-core omegaconf wandb 2>&1 | tail -2 || true

echo "[deploy] smoke: import the release and assert the published caption"
"$PY" src/scripts/run_geoshield_audit.py --smoke \
  --manifest "$REPO/data/msls/manifest_all8.jsonl" \
  --root "$REPO/data/msls" --output /dev/null
