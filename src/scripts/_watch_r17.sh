#!/usr/bin/env bash
# Ten-minute check-in on the GeoShield run.
#
# Two earlier versions of this watcher were wrong in opposite directions: the
# first only fired on a terminal state, so a wedged job looked identical to a
# healthy one; the second computed its fields inline inside a double-quoted
# ssh argument, where the escaping mangled the process count and produced two
# false "it died" alarms against a perfectly healthy download.
#
# So the snapshot now lives in a file on the host and is piped in over stdin:
# nothing has to survive a round of shell quoting. Every interval prints a
# line whether or not anything moved, and a death is only declared when the
# process count is zero AND no terminal line was written.
PW=$(sed -n "7p" /mnt/c/source/.env | tr -d "\r\n")
SSH=(sshpass -p "$PW" ssh -p 22766 -o StrictHostKeyChecking=no
     -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25
     root@connect.westd.seetacloud.com)
SNAP=/mnt/d/source/PPEDCRF/src/scripts/_remote_snapshot.sh
INTERVAL=${INTERVAL:-600}

prev=""
stalls=0
misses=0

while true; do
  line=$(timeout 90 "${SSH[@]}" "bash -s" < "$SNAP" 2>/dev/null \
         | tr -d '\r' | grep -E '^[0-9]+\|' | head -1)

  if [ -z "$line" ]; then
    misses=$((misses + 1))
    echo "[r17] snapshot unavailable (${misses} in a row)"
    [ "$misses" -ge 3 ] && { echo "[r17] host unreachable 3x -- check it"; exit 1; }
    sleep "$INTERVAL"; continue
  fi
  misses=0

  IFS='|' read -r cache rows pre predone prefail pilotdone err gpu procs <<< "$line"

  sig="$cache/$rows/$pre/$pilotdone"
  if [ "$sig" = "$prev" ]; then
    stalls=$((stalls + 1))
    note="NO PROGRESS ${stalls}x"
  else
    stalls=0
    note="moving"
  fi
  prev="$sig"

  echo "[r17] cache=${cache}MB rows=${rows}/25 prefetched=${pre}/3 gpu=${gpu}% procs=${procs} -- ${note}"

  if [ "${pilotdone:-0}" -gt 0 ]; then
    echo "[r17] PILOT FINISHED with ${rows} rows"; exit 0
  fi
  if [ "${prefail:-0}" -gt 0 ] || [ "${err:-0}" -gt 0 ]; then
    echo "[r17] errors in logs (prefetch_fail=${prefail} traces=${err})"
  fi
  if [ "${procs:-0}" -eq 0 ]; then
    echo "[r17] NOTHING RUNNING and no DONE line -- the job died"; exit 1
  fi
  if [ "$stalls" -ge 3 ]; then
    echo "[r17] STALLED across 3 intervals -- needs a look"; exit 1
  fi
  sleep "$INTERVAL"
done
