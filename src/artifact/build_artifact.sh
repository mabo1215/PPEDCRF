#!/usr/bin/env bash
# Assemble the anonymous reviewer artifact from the local experiment exports.
#
# results/, extended_evidence_report.pdf and MANIFEST.sha256 are build products
# (derived from src/outputs/, src/exports/ and paper/), so they are not tracked
# in git. Run this to regenerate them before packaging the bundle.
#
# Set PYTHON=python if `python3` is not on PATH (Git Bash on Windows resolves
# python3 to the Microsoft Store stub, which is not an interpreter).
set -euo pipefail
cd "$(dirname "$0")"
# This script lives at src/artifact/, so the repository root is two levels up.
REPO_ROOT="$(cd ../.. && pwd)"
OUT="$REPO_ROOT/src/outputs"
# src/outputs/ is gitignored and therefore machine-local. Trees small enough to
# travel are committed under src/exports/, which is searched second so a fresh
# clone can build most of the bundle without hunting for the originals.
EXPORTS="$REPO_ROOT/src/exports"
PYTHON="${PYTHON:-python3}"

# Resolve a tree name against the export roots, preferring the local originals.
resolve_tree() {  # $1 = tree name; echoes the directory, or nothing
  if [ -d "$OUT/$1" ]; then echo "$OUT/$1"
  elif [ -d "$EXPORTS/$1" ]; then echo "$EXPORTS/$1"
  fi
}

# Only these extensions travel: raw per-query outcomes, summaries, run metadata
# and status files. No imagery, no weights, no logs.
copy_tree() {  # $1 = tree name, $2 = name inside results/
  local src dst="results/$2" rel
  src="$(resolve_tree "$1")"
  [ -n "$src" ] || { echo "missing export tree: $1 (looked in $OUT and $EXPORTS)" >&2; exit 1; }
  find "$src" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.jsonl' \
       -o -name '*.txt' -o -name '*.sha256' \) -print0 |
    while IFS= read -r -d '' f; do
      rel="${f#"$src"/}"
      mkdir -p "$dst/$(dirname "$rel")"
      cp "$f" "$dst/$rel"
    done
}

copy_file() {  # $1 = relative path under an export root, $2 = path inside results/
  local src="$OUT/$1" dst="results/$2"
  [ -f "$src" ] || src="$EXPORTS/$1"
  [ -f "$src" ] || { echo "missing export file: $1 (looked in $OUT and $EXPORTS)" >&2; exit 1; }
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

echo "== collecting exports (CSV/JSON/status only; no images or weights) =="
rm -rf results

# --- allocation axis -------------------------------------------------------
copy_tree icme2027_placement_study            placement_study
copy_tree icme2027_placement_study_maskbacked placement_study_maskbacked
copy_tree icme2027_placement_sigma_sweep      placement_sigma_sweep
copy_tree icme2027_placement_highsigma_50pair placement_highsigma_50pair
copy_tree icme2027_placement_msls             placement_msls
copy_tree margin_oracle                       margin_oracle
copy_tree operator_study                      operator_study

# --- real-place manifests --------------------------------------------------
copy_tree icme2027_revision_20260904_session3 session3
copy_tree icme2027_revision_20260904_session4 session4

# --- direction axis --------------------------------------------------------
copy_file direction_transfer/per_query.csv \
          direction_transfer/per_query.csv
copy_file direction_free_d1/resnet18_merged.csv \
          direction_transfer_galleryfree/resnet18.csv
copy_file direction_free_d1/mixvpr_merged.csv \
          direction_transfer_galleryfree/mixvpr.csv
copy_file direction_positive_3seed_resnet18.csv \
          direction_transfer_galleryfree/resnet18_reference_targeted.csv
copy_tree d4_3seed      sanitize_3seed
copy_tree sanfree_3seed sanitize_galleryfree

# --- downstream utility ----------------------------------------------------
copy_file direction_utility_d3/detection/per_image.jsonl \
          direction_utility/detection.jsonl
copy_file direction_utility_d3/segmentation/per_image.jsonl \
          direction_utility/segmentation.jsonl
copy_file direction_utility_d3/detection/targets.json \
          direction_utility/targets_detection.json

# --- controlled model with a known Jacobian --------------------------------
copy_tree known_jacobian           known_jacobian
copy_tree known_jacobian_operators known_jacobian_operators

# --- fifth-review experiments (A3 factorial, A4 baseline, A7 cross-time) ----
# Added in the same session that produced them: three export trees have
# already been lost by leaving this step for later.
copy_tree tifs_a3        factorial_mse15
copy_tree tifs_a3hi      factorial_mse241
copy_tree tifs_a3b       factorial_placements
copy_tree tifs_a4        maskguided_cam25
copy_tree tifs_a4sweep   maskguided_coverage
copy_tree tifs_a4mask    maskguided_source
copy_tree tifs_a7_o2n8   crosstime_old2new
copy_tree tifs_a7_n2o8   crosstime_new2old
copy_tree tifs5_analysis analysis_reports

# --- the TIFS round: the families behind Table IV, the frontier, the release
# --- boundary, the third held-out backbone and the column-norm measurement.
# Without these the bundle cannot support the manuscript's current headlines.
copy_tree tifs_d6            transfer_table4
copy_tree tifs_d7            utility_released_frames
copy_tree tifs_a5            frontier_segmentation
copy_tree tifs_a2            jacobian_columns
copy_tree tifs_a7b           transfer_patchnetvlad
copy_tree tifs_a8            release_boundary_mse15.68
copy_tree tifs_a8_hi         release_boundary_mse241.5
copy_tree tifs_a8_mse60.0    release_boundary_mse60
copy_tree tifs_a8_mse5.0     release_boundary_mse5
copy_tree tifs_a6            adaptive_attacker_training
copy_tree tifs_a6_eval       adaptive_attacker_eval

# --- EOT hardening: one file per condition, three seeds concatenated --------
# The seeds were run as separate jobs, and one of them (ResNet18 seed 5678)
# ran on a second machine whose output file also carries stray rows from an
# interrupted 9012 job. Filtering each source to the seed its directory is
# named for is what makes the concatenation exactly three seeds per condition.
echo "== assembling the EOT tables (three seeds per condition) =="
"$PYTHON" - "$OUT" <<'PY'
import csv, pathlib, sys

out = pathlib.Path(sys.argv[1])
src = out / "direction_eot_d5"
dst = pathlib.Path("results/direction_eot")
SEED_DIRS = {
    "resnet18": {"1234": "resnet18_s1234", "5678": "resnet18_local",
                 "9012": "resnet18_s9012"},
    "mixvpr":   {"1234": "mixvpr_s1234", "5678": "mixvpr_s5678",
                 "9012": "mixvpr_s9012"},
}
CONDITIONS = ("none", "jpeg75", "jpeg50", "blur", "denoise")

for backbone, seed_dirs in SEED_DIRS.items():
    for cond in CONDITIONS:
        rows, header = [], None
        for seed, d in seed_dirs.items():
            f = src / d / f"eot_{cond}.csv"
            if not f.is_file():
                sys.exit(f"missing EOT export: {f}")
            with open(f, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                header = header or reader.fieldnames
                rows += [r for r in reader if r["seed"] == seed]
        target = dst / backbone / f"{cond}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"  {backbone}/{cond}.csv: {len(rows)} rows")
PY

echo "== scrubbing absolute/host paths =="
"$PYTHON" - <<'PY'
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

# The manuscript cites the extended evidence report wherever a result lives
# only there, and says it is released with the code. Shipping it here is what
# makes those citations resolvable.
echo "== including the extended evidence report =="
REPORT="$REPO_ROOT/paper/backup/supplementary_extended.pdf"
if [ ! -f "$REPORT" ]; then
  echo "missing $REPORT -- build the paper first (paper/build.bat)" >&2
  exit 1
fi
cp "$REPORT" extended_evidence_report.pdf

echo "== generating MANIFEST.sha256 =="
find results extended_evidence_report.pdf -type f | sort | xargs sha256sum > MANIFEST.sha256
echo "  $(wc -l < MANIFEST.sha256) files"

echo "== self-test =="
bash run_verification.sh
