"""Export the E1 real-place retrieval rows the supplement's tables are read from.

Why this exists. The two E1 tables in the Supplementary Material report 36
cells of raw against released Top-1, and the earlier two-city table adds a
place-cluster interval on each difference. Those numbers were the only ones in
either document that the claim auditor did not cover, because the runs that
produced them live under src/outputs/, which is not committed: a fresh clone
has no way to recompute them. This writes the columns the auditor needs into
src/exports/, where they travel with the repository. The directory is named
e1_msls_rows rather than e1_msls because an unrelated, uncommitted run tree
already occupies src/outputs/e1_msls, and the auditor searches both roots: two
different trees under one name would let a missing export read as a present one.

What is kept. Only what a verifier needs: the query identifier (to pair a raw
query with its released counterparts), the dataset's own place-cluster label
(the unit the intervals resample), the variant and seed, and the rank of the
correct gallery item. Everything else in the source rows -- similarities,
neighbour identifiers, PSNR, SSIM, per-frame distortion -- is dropped, which is
what takes the export from about 16 MB to a size worth committing. Top-1, Top-5
and Top-10 are all recoverable from the rank, so nothing the tables report is
lost.

Provenance. The summary records, per cell, the source path, the row count and
the gallery size, so an exported cell can always be traced back to the run
directory that produced it.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]

BACKBONES = ["resnet18", "resnet50", "vgg16", "cosplace", "mixvpr",
             "patchnetvlad"]

# (exported manifest name, source directory template). The manifest names are
# the ones the supplement prints; the source names are the ones the runs used.
TWO_CITY = [
    ("primary", "icme2027_revision_20260904/geotagged_vpr/all/{bb}"),
    ("old_to_new", "icme2027_revision_20260904/geotagged_vpr/o2n/{bb}"),
    ("new_to_old", "icme2027_revision_20260904/geotagged_vpr/n2o/{bb}"),
]
WIDE8 = [
    ("primary", "icme2027_revision_20260904_session3/n1_expanded_msls/{bb}"),
    ("old_to_new",
     "icme2027_revision_20260904_session4/n6_crosstime8/o2n8/{bb}"),
    ("new_to_old",
     "icme2027_revision_20260904_session4/n6_crosstime8/n2o8/{bb}"),
]
SCALES = [("two_city", TWO_CITY), ("wide8", WIDE8)]

KEEP = ["query_id", "place_id", "variant", "seed", "correct_rank"]


def slim(source: Path) -> Tuple[List[dict], str]:
    """Read one run's per-query rows down to the columns a verifier needs."""
    rows: List[dict] = []
    gallery = ""
    with source.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            gallery = gallery or record.get("gallery_size", "")
            rows.append({k: record[k] for k in KEEP})
    return rows, gallery


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default=str(REPO / "src" / "outputs"),
                    help="Where the uncommitted run directories live.")
    ap.add_argument("--out-dir", dest="out_dir",
                    default=str(REPO / "src" / "exports" / "e1_msls_rows"))
    args = ap.parse_args()

    outputs, out_dir = Path(args.outputs), Path(args.out_dir)
    summary: Dict[str, dict] = {}
    written = missing = 0

    for scale, manifests in SCALES:
        for manifest, template in manifests:
            for backbone in BACKBONES:
                source = (outputs / template.format(bb=backbone)
                          / "geotagged_vpr_per_query.csv")
                key = f"{scale}/{manifest}/{backbone}"
                if not source.is_file():
                    print(f"[skip] {key}: {source} absent", flush=True)
                    missing += 1
                    continue
                rows, gallery = slim(source)
                target = out_dir / scale / manifest / f"{backbone}.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=KEEP)
                    writer.writeheader()
                    writer.writerows(rows)
                queries = len({r["query_id"] for r in rows})
                summary[key] = {
                    "source": str(source.relative_to(REPO)),
                    "rows": len(rows),
                    "queries": queries,
                    "places": len({r["place_id"] for r in rows}),
                    "gallery_size": gallery,
                    "seeds": sorted({r["seed"] for r in rows if r["seed"] != "raw"}),
                }
                written += 1
                print(f"[ok  ] {key}: {len(rows)} rows, {queries} queries",
                      flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e1_export_summary.json").write_text(
        json.dumps({"cells": summary, "columns": KEEP,
                    "note": ("Top-1/5/10 are recovered from correct_rank; the "
                             "place label is the dataset's own cluster id and "
                             "is the unit the supplement's intervals "
                             "resample.")},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n[done] {written} cells written, {missing} absent -> {out_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
