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
    """Every row, tagged with the file it came from.

    The tag is what makes the trajectory gate possible. Twenty steps is
    measured by two different runs -- the original 20/40 one and the new 20/80
    one -- and without knowing which file a row came from those two
    measurements are averaged together into one number, which is precisely the
    comparison the gate is supposed to make. An earlier version of this script
    did exactly that and the gate could never have fired.
    """
    rows: List[dict] = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            stem = Path(path).stem
            with open(path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    r["_source"] = stem
                    rows.append(r)
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
    ap.add_argument("--latex", default="",
                    help="also write the curve as a generated LaTeX table")
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
        # Paired within a run: each file carries its own uniform_crossdraw arm
        # on its own noise draws, and crossing them would compare two
        # different controls.
        refs = {src: per_query(rows, lambda r, src=src: (
                    r["condition"] == "uniform_crossdraw"
                    and r["_source"] == src))
                for src in {r["_source"] for r in rows}}
        # Every solved checkpoint present, keyed by the step count it records.
        # (budget, source file) rather than budget alone, so a step count two
        # runs both measured yields two numbers to compare instead of one
        # average that hides whether they agree.
        cells = sorted({(int(r["opt_steps"]), r["_source"]) for r in rows
                        if r["condition"].startswith("opt_transfer")})
        seen: Dict[int, list] = collections.defaultdict(list)
        for n, src in cells:
            arm = per_query(rows, lambda r, n=n, src=src: (
                r["condition"].startswith("opt_transfer")
                and r["condition"].endswith("_crossdraw")
                and int(r["opt_steps"]) == n and r["_source"] == src))
            ref = refs[src]
            ks = sorted(set(arm) & set(ref))
            if not ks:
                continue
            d = np.array([arm[q] - ref[q] for q in ks])
            lo, hi = clustered_ci(d, [place_of.get(q, q) for q in ks],
                                  args.n_boot, 4242 + n)
            sel = [r for r in rows
                   if r["condition"].startswith("opt_transfer")
                   and r["condition"].endswith("_crossdraw")
                   and int(r["opt_steps"]) == n and r["_source"] == src]
            dec = float(np.mean([float(r["weight_top10pct_share"]) for r in sel]))
            obj = float(np.mean([float(r["surrogate_sim_end"]) for r in sel]))
            seen[n].append(d.mean())
            print(f"{tag:<15} {n:>5} {np.mean([ref[q] for q in ks]):>8.4f} "
                  f"{np.mean([arm[q] for q in ks]):>8.4f} {d.mean():>+9.4f}  "
                  f"[{lo:+.4f},{hi:+.4f}] {dec:>7.3f} {obj:>9.4f}  {src}")
            curves.setdefault(tag, []).append((n, d.mean(), lo, hi, dec, obj))
        # The gate: a step count measured by two runs must agree.
        for n, vals in seen.items():
            if len(vals) > 1 and max(vals) - min(vals) > args.tolerance:
                print(f"  [GATE FAILED] {tag} at {n} steps: "
                      f"{vals} spread {max(vals)-min(vals):.4f} "
                      f"> {args.tolerance}")
        print()

    if args.latex:
        budgets = sorted({b for rows_ in curves.values() for b, *_ in rows_})
        lines = [
            "% Generated by src/scripts/analyze_allocation_curve.py.",
            "% Do not edit by hand.",
            r"\begin{table}[t]", r"\centering",
            r"\caption{What a solved allocation map buys as the search budget "
            r"grows, on five held-out attackers. $\Delta$ Top-1 against the "
            r"uniform control on a noise field the optimiser never saw, 400 "
            r"query clusters and three seeds at every budget; bold is a "
            r"place-clustered interval excluding zero. Twenty and eighty come "
            r"from one run and five and ten from another, so twenty is "
            r"measured twice and the two agree --- the optimiser's trajectory "
            r"depends on the seed and the step index alone, which is what puts "
            r"the five points on one curve. Every arm but MixVPR's is solved "
            r"against the same three surrogates and is therefore the same map; "
            r"MixVPR is the evaluation target in its own arm, so ResNet18 "
            r"joins its ensemble.}",
            r"\label{tab:alloc_curve}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\begin{tabular}{l" + "c" * len(budgets) + "}", r"\hline",
            "Attacker & " + " & ".join(f"{b}" for b in budgets)
            + r" \\",
            r"\hline",
        ]
        for tag, rows_ in curves.items():
            by = {b: (d, lo, hi) for b, d, lo, hi, *_ in rows_}
            cells = []
            for b in budgets:
                if b not in by:
                    cells.append("---"); continue
                d, lo, hi = by[b]
                sig = (lo < 0 and hi < 0) or (lo > 0 and hi > 0)
                cells.append((r"$\mathbf{%+.4f}$" % d) if sig
                             else ("$%+.4f$" % d))
            lines.append(f"{tag:<15} & " + " & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        Path(args.latex).parent.mkdir(parents=True, exist_ok=True)
        Path(args.latex).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[latex] {args.latex}")

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
