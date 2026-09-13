#!/usr/bin/env bash
# Ten-minute check-in on the GeoShield arm.
#
# Two earlier versions of this watcher were wrong in opposite directions: the
# first only fired on a terminal state, so a wedged job looked identical to a
# healthy one; the second computed its fields inline inside a double-quoted
# ssh argument, where the escaping mangled the process count and produced two
# false "it died" alarms against a perfectly healthy download.
#
# So the snapshot lives in a file on the host and is piped in over stdin:
# nothing has to survive a round of shell quoting. Every interval prints a
# line whether or not anything moved, a death is only declared when the
# process count is zero AND no terminal line was written, and an energy-gate
# abort is surfaced immediately rather than waiting for the stall counter --
# a gate trip means the released frames are wrong, which is worth knowing in
# ten minutes rather than thirty.
PW=$(sed -n "7p" /mnt/c/source/.env | tr -d "\r\n")
SSH=(sshpass -p "$PW" ssh -p 22766 -o StrictHostKeyChecking=no
     -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25
     root@connect.westd.seetacloud.com)
SNAP=/mnt/d/source/PPEDCRF/src/scripts/_remote_snapshot.sh
INTERVAL=${INTERVAL:-600}
RUN=${RUN:-arm_published_s1234}
LOG=${LOG:-arm1}
TOTAL=${TOTAL:-400}

prev=""
stalls=0
misses=0

while true; do
  line=$(timeout 90 "${SSH[@]}" "RUN=$RUN LOG=$LOG bash -s" < "$SNAP" 2>/dev/null \
         | tr -d '\r' | grep -E '^[0-9]+\|' | head -1)

  if [ -z "$line" ]; then
    misses=$((misses + 1))
    echo "[arm1] snapshot unavailable (${misses} in a row)"
    [ "$misses" -ge 3 ] && { echo "[arm1] host unreachable 3x -- check it"; exit 1; }
    sleep "$INTERVAL"; continue
  fi
  misses=0

  IFS='|' read -r cache rows queries done_ err gate gpu procs <<< "$line"

  if [ "$queries" = "$prev" ]; then
    stalls=$((stalls + 1)); note="NO PROGRESS ${stalls}x"
  else
    stalls=0; note="moving"
  fi
  prev="$queries"

  pct=$(( queries * 100 / TOTAL ))
  echo "[arm1] queries=${queries}/${TOTAL} (${pct}%) rows=${rows} gpu=${gpu}% procs=${procs} -- ${note}"

  if [ "${gate:-0}" -gt 0 ]; then
    echo "[arm1] ENERGY GATE TRIPPED -- released frames are off target, stop and look"; exit 1
  fi
  if [ "${done_:-0}" -gt 0 ]; then
    echo "[arm1] ARM FINISHED with ${rows} rows over ${queries} queries"; exit 0
  fi
  if [ "${err:-0}" -gt 0 ]; then
    echo "[arm1] traceback in the log -- check logs_r17/${LOG}.log"
  fi
  if [ "${procs:-0}" -eq 0 ]; then
    echo "[arm1] NOTHING RUNNING and no DONE line -- the job died"; exit 1
  fi
  if [ "$stalls" -ge 3 ]; then
    echo "[arm1] STALLED across 3 intervals -- needs a look"; exit 1
  fi
  sleep "$INTERVAL"
done
