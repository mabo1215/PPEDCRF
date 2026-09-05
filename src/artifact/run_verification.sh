#!/usr/bin/env bash
# One-command reproducibility check for the PPEDCRF submission.
# CPU only; no GPU, no network, no dataset or checkpoint required.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/3 dependencies =="
python3 -c "import pandas" 2>/dev/null && echo "  pandas present" || {
  echo "  installing verification dependencies..."
  python3 -m pip install --quiet "pandas==2.3.3" "numpy==2.2.6"
}

echo
echo "== 2/3 integrity of released exports =="
if command -v sha256sum >/dev/null 2>&1; then
  if sha256sum -c MANIFEST.sha256 --quiet; then
    echo "  all files match MANIFEST.sha256"
  else
    echo "  INTEGRITY FAILURE: released exports do not match their checksums" >&2
    exit 1
  fi
else
  echo "  sha256sum unavailable; skipping integrity check"
fi

echo
echo "== 3/3 recomputing paper tables from raw exports =="
python3 verify_claims.py --results_root results
