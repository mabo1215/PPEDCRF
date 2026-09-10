"""Read the optimised-allocation study on the manuscript's own terms.

Everything here uses the unit of inference the protocol declares: the three
seeds are averaged within a query, contrasts are paired against the uniform
control on identical queries, the interval is a 10,000-resample bootstrap over
queries and again over the dataset's place clusters, and the test is a Wilcoxon
signed-rank over the per-query differences.

The columns that decide what the run means are not the retrieval ones:

  surrogate_sim_start -> surrogate_sim_end
      whether the optimiser moved its own objective at all. A null in Top-1
      beside an unmoved objective says nothing; a null beside an objective that
      fell sharply says allocation cannot buy retrieval at this budget.

  <condition> vs <condition>_crossdraw
      whether the map is a spatial preference or a selection of signs. A
      genuine placement transfers to a noise draw the optimiser never saw; sign
      selection does not.

  weight_top10pct_share
      how concentrated the solved map is, on the same scale the manuscript
      reports for every prescribed rule (0.100 uniform, 0.839 edge).
"""
from __future__ import annotations

import argparse
import csv
import collections
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon


def load(paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def per_query(rows: Sequence[dict], condition: str,
              field: str = "top1") -> Dict[str, float]:
    """Query -> metric, seeds averaged within the query."""
    acc: Dict[str, List[float]] = collections.defaultdict(list)
    for r in rows:
        if r["condition"] != condition:
            continue
        if field == "top1":
            v = 1.0 if int(r["correct_rank"]) == 1 else 0.0
        elif field == "top5":
            v = 1.0 if int(r["correct_rank"]) <= 5 else 0.0
        elif field == "top10":
            v = 1.0 if int(r["correct_rank"]) <= 10 else 0.0
        else:
            v = float(r[field])
        acc[r["query_id"]].append(v)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def bootstrap(diffs: np.ndarray, groups: Sequence[str], n: int = 10000,
              seed: int = 1234) -> tuple:
    rng = np.random.default_rng(seed)
    keys = sorted(set(groups))
    index = collections.defaultdict(list)
    for i, g in enumerate(groups):
        index[g].append(i)
    means = np.empty(n)
    k = len(keys)
    for b in range(n):
        pick = rng.integers(0, k, k)
        sel = np.concatenate([index[keys[j]] for j in pick])
        means[b] = diffs[sel].mean()
    return tuple(np.percentile(means, [2.5, 97.5]))


def contrast(rows, condition: str, reference: str, place_of: Dict[str, str]):
    cur = per_query(rows, condition)
    ref = per_query(rows, reference)
    qs = sorted(set(cur) & set(ref))
    if not qs:
        return None
    d = np.array([cur[q] - ref[q] for q in qs])
    try:
        p = float(wilcoxon(d, zero_method="wilcox", method="approx").pvalue) \
            if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return {
        "n": len(qs),
        "top1": float(np.mean([cur[q] for q in qs])),
        "delta": float(d.mean()),
        "ci_query": bootstrap(d, qs),
        "ci_place": bootstrap(d, [place_of.get(q, q) for q in qs]),
        "p": p,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--places", default="src/exports/tifs_d6/d6_r18_plain.csv",
                    help="any export carrying query_id and correct_place; the "
                         "place labels are a property of the manifest, not of "
                         "the run that happens to supply them")
    ap.add_argument("--reference", default="uniform")
    args = ap.parse_args()

    place_of: Dict[str, str] = {}
    with open(args.places, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            place_of[r["query_id"]] = r["correct_place"]

    rows = load([Path(p) for p in args.rows])
    conds = sorted(set(r["condition"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))
    print(f"rows={len(rows)}  conditions={conds}  seeds={seeds}")
    ref_q = per_query(rows, args.reference)
    print(f"reference '{args.reference}': Top-1 "
          f"{np.mean(list(ref_q.values())):.4f} over {len(ref_q)} queries\n")

    hdr = (f"{'condition':<26}{'Top-1':>8}{'delta':>9}"
           f"{'query 95% CI':>22}{'place 95% CI':>22}{'p':>9}"
           f"{'top10%':>8}{'sim start->end':>18}")
    print(hdr)
    print("-" * len(hdr))
    for c in conds:
        if c == args.reference:
            continue
        st = contrast(rows, c, args.reference, place_of)
        if st is None:
            continue
        sub = [r for r in rows if r["condition"] == c]
        share = float(np.mean([float(r["weight_top10pct_share"]) for r in sub]))
        s0 = float(np.mean([float(r["surrogate_sim_start"]) for r in sub]))
        s1 = float(np.mean([float(r["surrogate_sim_end"]) for r in sub]))
        print(f"{c:<26}{st['top1']:>8.4f}{st['delta']:>+9.4f}"
              f"  [{st['ci_query'][0]:+.4f},{st['ci_query'][1]:+.4f}]"
              f"  [{st['ci_place'][0]:+.4f},{st['ci_place'][1]:+.4f}]"
              f"{st['p']:>9.3g}{share:>8.3f}"
              f"{s0:>9.4f}->{s1:.4f}")

    # The energy gate and the delivered-distortion gate, both of which must
    # hold for any of the above to be a comparison between placements.
    ms = [float(r["weight_mean_square"]) for r in rows]
    mse = [float(r["effective_mse"]) for r in rows]
    print(f"\ngates: mean(w^2) in [{min(ms):.6f},{max(ms):.6f}]; "
          f"delivered MSE in [{min(mse):.4f},{max(mse):.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
