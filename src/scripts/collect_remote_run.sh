#!/usr/bin/env bash
# Bring a finished remote run home, verify it, commit it -- and only then
# offer to power the host down.
#
# This implements the ordering half of the "a billing host never waits on an
# unanswered question" rule in .claude/rules/src.md. The rule authorises
# shutting a host down unattended; it does not authorise shutting one down
# with unretrieved results. /root/autodl-tmp is not guaranteed to survive a
# stop/restart cycle, so a power-off before a verified local copy exists can
# destroy the run that was just paid for.
#
# Hence: pull, verify against the data (not just "the file exists"), commit,
# and treat any mismatch as a reason to LEAVE THE HOST RUNNING. A few more
# minutes of billing is far cheaper than re-running the job.
#
# Shutdown is deliberately NOT the default. Pass --shutdown to take it.
set -euo pipefail

RUN=${RUN:-arm_published_s1234}
EXPECT_QUERIES=${EXPECT_QUERIES:-400}
EXPECT_CONDS=${EXPECT_CONDS:-3}
TARGET_MSE=${TARGET_MSE:-15.68}
REPO_LOCAL=${REPO_LOCAL:-/mnt/d/source/PPEDCRF}
REPO_REMOTE=/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
PORT=${PORT:-22766}
HOST=${HOST:-root@connect.westd.seetacloud.com}
DO_SHUTDOWN=0
[ "${1:-}" = "--shutdown" ] && DO_SHUTDOWN=1

PW=$(sed -n "7p" /mnt/c/source/.env | tr -d "\r\n")
SSH=(sshpass -p "$PW" ssh -p "$PORT" -o StrictHostKeyChecking=no
     -o UserKnownHostsFile=/dev/null)
SCP=(sshpass -p "$PW" scp -P "$PORT" -o StrictHostKeyChecking=no
     -o UserKnownHostsFile=/dev/null)

OUT="$REPO_LOCAL/src/outputs/r17_geoshield"
mkdir -p "$OUT"

echo "== 1. pull =="
"${SCP[@]}" \
  "$HOST:$REPO_REMOTE/src/outputs/r17_geoshield/${RUN}.csv" \
  "$HOST:$REPO_REMOTE/src/outputs/r17_geoshield/run_metadata.json" \
  "$OUT/"
"${SCP[@]}" "$HOST:$REPO_REMOTE/logs_r17/arm1.log" "$OUT/${RUN}.log" || \
  echo "  (log not retrieved; not fatal)"

echo "== 2. verify the local copy against the data =="
python3 - "$OUT/${RUN}.csv" "$EXPECT_QUERIES" "$EXPECT_CONDS" "$TARGET_MSE" <<'PY'
import csv, sys, collections
path, want_q, want_c, target = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
queries = {r["query_id"] for r in rows}
conds = {r["condition"] for r in rows}
problems = []
if len(queries) != want_q:
    problems.append(f"{len(queries)} queries, expected {want_q}")
if len(conds) != want_c:
    problems.append(f"{len(conds)} conditions, expected {want_c}")
if len(rows) != len(queries) * len(conds):
    problems.append(f"{len(rows)} rows != {len(queries)}x{len(conds)}")
keys = [(r["query_id"], r["condition"], r["seed"]) for r in rows]
if len(keys) != len(set(keys)):
    problems.append(f"{len(keys)-len(set(keys))} duplicate keys")
# The gate, re-checked at the destination: a truncated or corrupted transfer
# shows up here as much as a bad run does.
for c in sorted(conds):
    mse = [float(r["effective_mse"]) for r in rows if r["condition"] == c]
    lo, hi = min(mse), max(mse)
    want = 0.0 if c == "clean" else target
    if abs(lo - want) > 0.05 or abs(hi - want) > 0.05:
        problems.append(f"{c}: delivered MSE [{lo:.4f},{hi:.4f}] off {want}")
    print(f"   {c:34s} n={len(mse):4d}  MSE [{lo:.4f},{hi:.4f}]")
if problems:
    print("VERIFY FAILED: " + "; ".join(problems))
    sys.exit(1)
print(f"   OK: {len(rows)} rows, {len(queries)} queries, {len(conds)} conditions, no duplicates")
PY

echo "== 3. commit =="
cd "$REPO_LOCAL"
git add -f "src/outputs/r17_geoshield/${RUN}.csv" \
           "src/outputs/r17_geoshield/run_metadata.json"
if git diff --cached --quiet; then
  echo "   nothing new to commit"
else
  git commit -q -m "Bring the ${RUN} rows home, verified at the destination

Pulled from the GPU host and re-checked against the data rather than against
the file listing: query count, condition count, duplicate keys and the
delivered-MSE gate on every condition. A truncated transfer fails these the
same way a bad run does, which is the point -- the host is powered down
after this, and /root/autodl-tmp is not guaranteed to survive that."
  git push -q origin main
  echo "   committed and pushed"
fi

if [ "$DO_SHUTDOWN" = "1" ]; then
  echo "== 4. power off =="
  "${SSH[@]}" "$HOST" "nohup sh -c 'sleep 2; shutdown -h now' >/dev/null 2>&1 &" || true
  echo "   shutdown issued (results are already local, verified and pushed)"
else
  echo "== 4. host left running (pass --shutdown to power off) =="
fi
