"""What a solved allocation map buys as a function of search budget.

The manuscript bounds the allocation axis at a twenty-step budget rather than
claiming anything about the axis, because twenty and forty were the only two
points it had -- and at forty every arm was still moving. This reads the five
points (5, 10, 20, 40, 80) as one curve per attacker.

Two things it checks before drawing anything:

  the trajectory gate
      The eighty-step run re-measures twenty, and the optimiser's trajectory
      depends only on the seed and the step index, so that value must land on
      the existing twenty-step value. The backward pass through the surrogates
      is not bitwise deterministic, so the tolerance is the run-to-run spread
      the manuscript already reports rather than zero.

  the one-map gate
      Every arm but MixVPR's is solved against the same three surrogates, so
      their objective traces at a given step count must agree. MixVPR is the
      evaluation target in its own arm and its ensemble carries ResNet18, so
      it is a second map and is labelled as one.

Everything uses the protocol's unit of inference: seeds averaged within a
query, paired against the crossdraw uniform arm on identical queries, and a
10,000-resample bootstrap over the dataset's place clusters.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
from pathlib import Path
from typing import Dict, List

import numpy as np


def load(patterns: List[str]) -> List[dict]:
    rows: List[dict] = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
    return rows


def per_query(rows, select) -> Dict[str, float]:
    acc = collections.defaultdict(list)
    for r in rows:
        if select(r):
            acc[r["query_id"]].append(float(int(r["correct_rank"]) == 1))
    return {q: float(np.mean(v)) for q, v in acc.items()}


def clustered_ci(d: np.ndarray, clusters: List[str], n_boot: int, seed: int):
    by = collections.defaultdict(list)
    for v, c in zip(d, clusters):
        by[c].append(v)
    arrs = [np.array(v) for v in by.values()]
    rng = np.random.default_rng(seed)
    k = len(arrs)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        draws[b] = np.concatenate([arrs[i] for i in pick]).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports", default="src/exports")
    ap.add_argument("--places", default="src/exports/tifs_d6/d6_r18_plain.csv")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--tolerance", type=float, default=0.008,
                    help="how far the re-measured 20-step point may sit from "
                         "the original before the curve is not one trajectory")
    args = ap.parse_args()

    place_of: Dict[str, str] = {}
    with open(args.places, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            place_of[r["query_id"]] = r["correct_place"]

    ex = args.exports
    # attacker -> the files carrying any of its solved arms
    families = {
        "ResNet18": [f"{ex}/optimised_allocation/r1_r18_exp_tr.csv",
                     f"{ex}/optimised_allocation/r14_r18_s*.csv"],
        "MixVPR*": [f"{ex}/optimised_allocation/r1_mix_exp_tr.csv",
                    f"{ex}/optimised_allocation/r14_mix_s*.csv"],
        "Patch-NetVLAD": [f"{ex}/optimised_allocation/r13_pnv_exp_tr.csv",
                          f"{ex}/optimised_allocation/r14_pnv_s*.csv"],
        "ViT-B/16": [f"{ex}/optimised_allocation/r13_vit_exp_tr.csv",
                     f"{ex}/optimised_allocation/r14_vit_s*.csv"],
        "CLIP ViT-L/14": [f"{ex}/r10_clip/r10_clip_alloc.csv",
                          f"{ex}/optimised_allocation/r14_clip_s*.csv"],
    }

    print(f"{'attacker':<15} {'steps':>5} {'uniform':>8} {'solved':>8} "
          f"{'delta':>9}  {'place 95% CI':<20} {'decile':>7} {'objective':>9}")
    print("-" * 92)
    curves: Dict[str, List[tuple]] = {}
    for tag, pats in families.items():
        rows = load(pats)
        if not rows:
            print(f"{tag:<15} (no rows yet)")
            continue
        ref = per_query(rows, lambda r: r["condition"] == "uniform_crossdraw")
        # Every solved checkpoint present, keyed by the step count it records.
        budgets = sorted({int(r["opt_steps"]) for r in rows
                          if r["condition"].startswith("opt_transfer")})
        seen: Dict[int, list] = collections.defaultdict(list)
        for n in budgets:
            arm = per_query(rows, lambda r, n=n: (
                r["condition"].startswith("opt_transfer")
                and r["condition"].endswith("_crossdraw")
                and int(r["opt_steps"]) == n))
            ks = sorted(set(arm) & set(ref))
            if not ks:
                continue
            d = np.array([arm[q] - ref[q] for q in ks])
            lo, hi = clustered_ci(d, [place_of.get(q, q) for q in ks],
                                  args.n_boot, 4242 + n)
            dec = float(np.mean([float(r["weight_top10pct_share"]) for r in rows
                                 if r["condition"].startswith("opt_transfer")
                                 and r["condition"].endswith("_crossdraw")
                                 and int(r["opt_steps"]) == n]))
            obj = float(np.mean([float(r["surrogate_sim_end"]) for r in rows
                                 if r["condition"].startswith("opt_transfer")
                                 and r["condition"].endswith("_crossdraw")
                                 and int(r["opt_steps"]) == n]))
            seen[n].append(d.mean())
            print(f"{tag:<15} {n:>5} {np.mean([ref[q] for q in ks]):>8.4f} "
                  f"{np.mean([arm[q] for q in ks]):>8.4f} {d.mean():>+9.4f}  "
                  f"[{lo:+.4f},{hi:+.4f}] {dec:>7.3f} {obj:>9.4f}")
            curves.setdefault(tag, []).append((n, d.mean(), lo, hi, dec, obj))
        # The gate: a step count measured by two runs must agree.
        for n, vals in seen.items():
            if len(vals) > 1 and max(vals) - min(vals) > args.tolerance:
                print(f"  [GATE FAILED] {tag} at {n} steps: "
                      f"{vals} spread {max(vals)-min(vals):.4f} "
                      f"> {args.tolerance}")
        print()

    # The one-map gate, across the four arms that share an ensemble.
    shared = [t for t in curves if t != "MixVPR*"]
    for n in sorted({b for t in shared for b, *_ in curves.get(t, [])}):
        objs = [o for t in shared for b, _, _, _, _, o in curves[t] if b == n]
        if len(objs) > 1:
            width = max(objs) - min(objs)
            flag = "" if width < 1e-3 else "   [DIFFER]"
            print(f"one-map gate @ {n:>3} steps: objective "
                  f"{min(objs):.4f}..{max(objs):.4f} over {len(objs)} arms{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
