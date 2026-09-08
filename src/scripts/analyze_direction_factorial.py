"""Read the factorial and mask-baseline exports and say what they show.

Both experiments answer questions about differences between conditions on the
same queries, so every comparison here is paired and clustered on the query:
seeds are averaged within a query first, then the interval is a query-cluster
bootstrap and the test is a Wilcoxon signed-rank over the per-query
differences. That is the unit the fifth review asks for, and it is the unit
the manuscript's transfer table already uses.

Three things are printed, in the order a reader needs them:

  1. the reproduction check --- whether the conditions that also exist in the
     published table land on the published values, which is the run's gate;
  2. the factor decomposition --- how much of the direction's effect survives
     when its signs are shuffled or its magnitudes flattened;
  3. the delivered-versus-realised energy per placement, which is the
     clamp-loss measurement the manuscript has been arguing about.

Usage:
    python analyze_direction_factorial.py --glob "src/outputs/tifs_a3/*.csv"
    python analyze_direction_factorial.py --mode mask --glob "...a4/*.csv"
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import statistics
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon

# What the manuscript prints for the conditions this run reproduces.
PUBLISHED = {
    ("resnet18", "isotropic"): 0.1967,
    ("resnet18", "direction"): 0.0317,
    ("mixvpr", "isotropic"): 0.7800,
    ("mixvpr", "direction"): 0.7317,
}


def load(paths: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def per_query(rows: Sequence[dict], key_fields: Sequence[str],
              metric: str) -> Dict[tuple, Dict[str, float]]:
    """(condition key) -> query -> metric averaged over seeds."""
    acc: Dict[tuple, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        key = tuple(r[f] for f in key_fields)
        if metric == "top1":
            value = 1.0 if int(r["correct_rank"]) == 1 else 0.0
        elif metric == "top5":
            value = float(r.get("top5_hit", int(int(r["correct_rank"]) <= 5)))
        elif metric == "top10":
            value = float(r.get("top10_hit", int(int(r["correct_rank"]) <= 10)))
        else:
            value = float(r[metric])
        acc[key][r["query_id"]].append(value)
    return {k: {q: statistics.fmean(v) for q, v in d.items()}
            for k, d in acc.items()}


def compare(cond: Dict[str, float], ref: Dict[str, float],
            n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired difference with a query-cluster bootstrap and a Wilcoxon test."""
    queries = sorted(set(cond) & set(ref))
    if not queries:
        return {}
    diff = np.array([cond[q] - ref[q] for q in queries])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(queries), size=(n_boot, len(queries)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Zero handling is delegated to scipy ("wilcox" discards the ties at
    # zero) and the normal approximation is requested explicitly. Dropping the
    # zeros here instead would leave scipy a short, apparently untied sample
    # and it would switch to the exact test -- which assumes continuous data.
    # Per-query differences of a Top-1 indicator averaged over three seeds take
    # five values, so they are heavily tied and the exact test does not apply;
    # the approximation carries the tie correction. The two paths differ only
    # where zeros dominate (p = 0.33 against 0.35 for the sign-shuffle cell),
    # never near the decision boundary, but the published tables report the
    # approximation and this is what regenerates them.
    p = (float(wilcoxon(diff, zero_method="wilcox", method="approx").pvalue)
         if np.any(diff != 0) else 1.0)
    return {"n": len(queries), "delta": float(diff.mean()),
            "lo": float(lo), "hi": float(hi), "p": p}


def report_factorial(rows: Sequence[dict]) -> None:
    # The runner does not write the attacker into the row, so pool one
    # backbone at a time: the glob selects it.
    hits = per_query(rows, ("condition", "placement"), "top1")
    conditions = sorted({k[0] for k in hits})
    placements = sorted({k[1] for k in hits})

    print(f"conditions: {conditions}")
    print(f"placements: {placements}")
    print(f"queries: {len({r['query_id'] for r in rows})}, "
          f"seeds: {sorted({r['seed'] for r in rows})}, "
          f"rows: {len(rows)}")

    print("\n== Top-1 by condition and placement ==")
    print(f"{'condition':20s} {'placement':10s} {'Top-1':>8s} {'Top-5':>8s} "
          f"{'Top-10':>8s}")
    t5 = per_query(rows, ("condition", "placement"), "top5")
    t10 = per_query(rows, ("condition", "placement"), "top10")
    for c in conditions:
        for p in placements:
            if (c, p) not in hits:
                continue
            print(f"{c:20s} {p:10s} "
                  f"{statistics.fmean(hits[(c, p)].values()):8.4f} "
                  f"{statistics.fmean(t5[(c, p)].values()):8.4f} "
                  f"{statistics.fmean(t10[(c, p)].values()):8.4f}")

    ref_key = ("isotropic", "uniform")
    if ref_key in hits:
        print("\n== Paired against the isotropic uniform control ==")
        print(f"{'condition':20s} {'placement':10s} {'delta':>9s} "
              f"{'95% CI':>22s} {'p':>10s}")
        for c in conditions:
            for p in placements:
                if (c, p) == ref_key or (c, p) not in hits:
                    continue
                st = compare(hits[(c, p)], hits[ref_key])
                if not st:
                    continue
                print(f"{c:20s} {p:10s} {st['delta']:+9.4f} "
                      f"[{st['lo']:+8.4f},{st['hi']:+8.4f}] {st['p']:10.2e}")

    if ("direction", "uniform") in hits:
        print("\n== What survives when the direction is broken ==")
        base = hits[("direction", "uniform")]
        for c in ("sign_shuffle", "magnitude_uniform"):
            if (c, "uniform") not in hits:
                continue
            st = compare(hits[(c, "uniform")], base)
            print(f"{c:20s} vs direction {st['delta']:+9.4f} "
                  f"[{st['lo']:+8.4f},{st['hi']:+8.4f}] p={st['p']:.2e}")

    print("\n== Delivered versus realised energy (the clamp's share) ==")
    print(f"{'placement':10s} {'pre-clip MSE':>13s} {'delivered':>10s} "
          f"{'lost %':>8s} {'clipped':>9s} {'max |d|':>9s}")
    for p in placements:
        sel = [r for r in rows if r["placement"] == p]
        pre = statistics.fmean(float(r["pre_clip_mse"]) for r in sel)
        eff = statistics.fmean(float(r["effective_mse"]) for r in sel)
        clip = statistics.fmean(float(r["clipped_fraction"]) for r in sel)
        mx = statistics.fmean(float(r["max_abs_delta"]) for r in sel)
        print(f"{p:10s} {pre:13.4f} {eff:10.4f} "
              f"{100 * (pre - eff) / max(pre, 1e-9):8.2f} {clip:9.4f} "
              f"{mx:9.3f}")

    print("\n== Reproduction check against the published table ==")
    for (backbone, cond), expected in PUBLISHED.items():
        key = (cond, "uniform")
        if key not in hits:
            continue
        got = statistics.fmean(hits[key].values())
        print(f"  {backbone}/{cond:10s} published={expected:.4f} "
              f"this run={got:.4f} diff={got - expected:+.4f}")
    print("  (only the row matching this run's attacker is meaningful)")


def report_mask(rows: Sequence[dict]) -> None:
    hits = per_query(rows, ("arm",), "top1")
    t5 = per_query(rows, ("arm",), "top5")
    arms = sorted({k[0] for k in hits})
    coverage = sorted({r["mask_coverage"] for r in rows if r["mask_coverage"]})
    print(f"arms: {arms}  mask coverage: {coverage}")
    print(f"queries: {len({r['query_id'] for r in rows})}, "
          f"seeds: {sorted({r['seed'] for r in rows})}, rows: {len(rows)}")

    print("\n== Top-1 by arm ==")
    for a in arms:
        print(f"  {a:16s} Top-1={statistics.fmean(hits[(a,)].values()):.4f} "
              f"Top-5={statistics.fmean(t5[(a,)].values()):.4f}")

    print("\n== Paired comparisons ==")
    for a, b in (("maskguided_pgd", "isotropic"),
                 ("fullframe_pgd", "isotropic"),
                 ("maskguided_pgd", "fullframe_pgd")):
        if (a,) not in hits or (b,) not in hits:
            continue
        st = compare(hits[(a,)], hits[(b,)])
        print(f"  {a:16s} minus {b:16s} {st['delta']:+8.4f} "
              f"[{st['lo']:+8.4f},{st['hi']:+8.4f}] p={st['p']:.2e} "
              f"(n={st['n']})")

    print("\n== Delivered distortion by arm ==")
    for a in arms:
        sel = [r for r in rows if r["arm"] == a]
        print(f"  {a:16s} delivered={statistics.fmean(float(r['effective_mse']) for r in sel):8.4f} "
              f"pre-clip={statistics.fmean(float(r['pre_clip_mse']) for r in sel):8.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True,
                    help="pattern matching the export CSVs to pool")
    ap.add_argument("--mode", default="factorial",
                    choices=("factorial", "mask"))
    args = ap.parse_args()

    paths = sorted(globmod.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files match {args.glob!r}")
    print(f"pooling {len(paths)} file(s)")
    rows = load(paths)
    if not rows:
        raise SystemExit("no rows")
    if args.mode == "factorial":
        report_factorial(rows)
    else:
        report_mask(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
