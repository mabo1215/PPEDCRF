#!/usr/bin/env bash
# Assemble the anonymous reviewer artifact from the local experiment exports.
#
# results/ and MANIFEST.sha256 are build products (derived from src/outputs/,
# which this repository gitignores), so they are not tracked in git. Run this
# to regenerate them before packaging the bundle for submission.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"

echo "== collecting exports (CSV/JSON/status only; no images or weights) =="
rm -rf results
for s in session3 session4; do
  src="$REPO_ROOT/src/outputs/icme2027_revision_20260904_$s"
  [ -d "$src" ] || { echo "missing $src" >&2; exit 1; }
  find "$src" -type f \( -name '*.csv' -o -name '*.json' -o -name 'DONE_STATUS.txt' -o -name '*.sha256' \) -print0 |
    while IFS= read -r -d '' f; do
      rel="${f#"$src"/}"
      mkdir -p "results/$s/$(dirname "$rel")"
      cp "$f" "results/$s/$rel"
    done
done

echo "== scrubbing absolute/host paths for double-blind anonymity =="
python3 - <<'PY'
import json, re, pathlib
n = 0
for p in pathlib.Path('results').rglob('*.json'):
    raw = p.read_text(encoding='utf-8')
    if not re.search(r'/root/|/mnt/|autodl|seetacloud', raw, re.I):
        continue
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        continue
    def scrub(v):
        if isinstance(v, str):
            v = re.sub(r'^.*?/PPEDCRF/', '', v)
            v = re.sub(r'^/root/[^ ]*?/', '', v)
            return re.sub(r'/root/\S*', '<redacted>', v)
        if isinstance(v, list): return [scrub(x) for x in v]
        if isinstance(v, dict): return {k: scrub(x) for k, x in v.items()}
        return v
    p.write_text(json.dumps(scrub(data), indent=2, ensure_ascii=False) + "\n", encoding='utf-8')
    n += 1
print(f"  scrubbed {n} metadata files")
PY

if grep -rqiE '/root/|/mnt/|seetacloud|autodl' results; then
  echo "ERROR: host paths still present after scrubbing" >&2
  exit 1
fi

echo "== generating MANIFEST.sha256 =="
find results -type f | sort | xargs sha256sum > MANIFEST.sha256
echo "  $(wc -l < MANIFEST.sha256) files"

echo "== self-test =="
bash run_verification.sh
