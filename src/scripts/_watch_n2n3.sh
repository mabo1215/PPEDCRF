#!/usr/bin/env bash
# Ten-minute check-in on the eight N2/N3 jobs.
#
# Reports both cards separately: on a two-card host an idle card is the most
# likely waste and it disappears entirely if you average utilisation.
#
# A death is declared only when no sessions remain AND fewer than the expected
# number of DONE lines were written, so a finished run is never mistaken for a
# crash -- and a crash is never mistaken for a finish.
PW=$(sed -n "15p" /mnt/c/source/.env | tr -d "\r\n")
SSH=(sshpass -p "$PW" ssh -p 42044 -o StrictHostKeyChecking=no
     -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25
     root@connect.weste.seetacloud.com)
SNAP=/mnt/d/source/PPEDCRF/src/scripts/_remote_snap_n2n3.sh
INTERVAL=${INTERVAL:-600}
EXPECT=${EXPECT:-8}

prev=""; stalls=0; misses=0
while true; do
  line=$(timeout 90 "${SSH[@]}" "bash -s" < "$SNAP" 2>/dev/null \
         | tr -d '\r' | grep -E '^[0-9]+\|' | head -1)
  if [ -z "$line" ]; then
    misses=$((misses + 1))
    echo "[n2n3] snapshot unavailable (${misses} in a row)"
    [ "$misses" -ge 3 ] && { echo "[n2n3] host unreachable 3x -- check it"; exit 1; }
    sleep "$INTERVAL"; continue
  fi
  misses=0
  IFS='|' read -r sessions done_ fail frames ckpt rows g0 g1 <<< "$line"

  sig="$done_/$frames/$ckpt/$rows"
  if [ "$sig" = "$prev" ]; then stalls=$((stalls+1)); note="NO PROGRESS ${stalls}x";
  else stalls=0; note="moving"; fi
  prev="$sig"

  echo "[n2n3] done=${done_}/${EXPECT} live=${sessions} frames=${frames} ckpt=${ckpt} rows=${rows} gpu0=${g0} gpu1=${g1} -- ${note}"

  if [ "${fail:-0}" -gt 0 ]; then
    echo "[n2n3] FAILURES in logs (tracebacks/OOM/nonzero exit): ${fail}"
  fi
  if [ "${done_:-0}" -ge "$EXPECT" ]; then
    echo "[n2n3] ALL ${EXPECT} JOBS FINISHED (frames=${frames} ckpt=${ckpt})"; exit 0
  fi
  if [ "${sessions:-0}" -eq 0 ]; then
    echo "[n2n3] NO SESSIONS LEFT but only ${done_}/${EXPECT} done -- jobs died"; exit 1
  fi
  if [ "$stalls" -ge 4 ]; then
    echo "[n2n3] STALLED across 4 intervals -- needs a look"; exit 1
  fi
  sleep "$INTERVAL"
done
