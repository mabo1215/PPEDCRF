"""Read the GeoShield arm against its own controls.

The question this arm answers is not "what Top-1 does GeoShield reach" but
"does it beat an isotropic perturbation of the same delivered energy" -- the
same matched-distortion contrast every other arm in the protocol is read
under. So the headline number here is the paired difference against the
isotropic control, not the absolute score.

Reads the per-query rows the runner writes and prints, per condition: Top-1,
the paired difference against the isotropic control with a place-clustered
bootstrap interval, the signed-rank p with its discordant-pair count, and the
delivered-MSE range so a reader can see the energy gate held.
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "scripts"))
from _pvalue import fmt_p  # noqa: E402
from make_msls_placement_table import boot  # noqa: E402


def per_query(rows: Sequence[dict], condition: str) -> Dict[str, float]:
    """Top-1 per query, averaging seeds within a query before pairing."""
    acc = collections.defaultdict(list)
    for r in rows:
        if r["condition"] == condition:
            acc[r["query_id"]].append(1.0 if int(r["correct_rank"]) == 1 else 0.0)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--places",
                    default=str(REPO / "src/exports/tifs_d6/d6_r18_plain.csv"),
                    help="query_id -> place, for the clustered interval")
    ap.add_argument("--reference", default="isotropic",
                    help="the control the contrast is taken against")
    args = ap.parse_args()

    rows: List[dict] = []
    for path in args.rows:
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        raise SystemExit("no rows")

    place_of = {}
    if Path(args.places).is_file():
        with open(args.places, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                place_of[r["query_id"]] = r["correct_place"]

    conditions = sorted({r["condition"] for r in rows})
    print(f"{len(rows)} rows, {len(conditions)} conditions, "
          f"{len({r['query_id'] for r in rows})} queries\n")

    # The gate, restated from the data rather than trusted.
    print("delivered MSE by condition (the energy gate, re-derived):")
    for c in conditions:
        mse = [float(r["effective_mse"]) for r in rows if r["condition"] == c]
        print(f"  {c:34s} min={min(mse):9.4f}  max={max(mse):9.4f}")
    print()

    ref = per_query(rows, args.reference)
    if not ref:
        raise SystemExit(f"no rows for reference condition {args.reference!r}")

    print(f"Top-1, and the paired contrast against '{args.reference}':")
    for c in conditions:
        cur = per_query(rows, c)
        top1 = float(np.mean(list(cur.values())))
        if c == args.reference:
            print(f"  {c:34s} Top-1 {top1:.4f}   (reference)")
            continue
        qs = sorted(set(cur) & set(ref))
        d = np.array([cur[q] - ref[q] for q in qs])
        nz = d[d != 0]
        p = float(wilcoxon(nz).pvalue) if nz.size else 1.0
        ci = boot(d, [place_of.get(q, q) for q in qs])
        verdict = ("separates" if ci[0] * ci[1] > 0 else "none det.")
        print(f"  {c:34s} Top-1 {top1:.4f}   "
              f"delta {d.mean():+.4f}  [{ci[0]:+.3f},{ci[1]:+.3f}]  "
              f"p={fmt_p(p)} ({nz.size} disc.)  {verdict}")

    print("\nNegative delta means better privacy (the attacker ranks the "
          "correct place first less often).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
