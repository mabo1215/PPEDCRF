"""Pool the EOT-hardened direction sweep across hosts and compare it to the
un-hardened control at the same delivered distortion.

The three seeds were produced on three different machines, so this merges by
(query_id, condition, seed) rather than concatenating: a sanitizer that was
interrupted and re-run elsewhere contributes its finished rows once, and a
seed that was only partially written before being reassigned is dropped
rather than diluting the pool. Both situations occurred while these runs were
being rebalanced across hosts, and either one would silently bias a pooled
Top-1 if merged naively.
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from math import comb
from pathlib import Path


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n)


def load_pool(paths, expected_conditions: int):
    """Merge rows, keeping only seeds that are complete for every query."""
    rows = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows[(r["query_id"], r["condition"], r["seed"])] = r
    per_seed = defaultdict(lambda: defaultdict(set))
    for (q, cond, seed) in rows:
        per_seed[seed][q].add(cond)
    keep = []
    dropped = []
    for seed, queries in per_seed.items():
        complete = sum(1 for q in queries if len(queries[q]) >= expected_conditions)
        if complete == len(queries) and len(queries) >= 400:
            keep.append(seed)
        else:
            dropped.append((seed, complete, len(queries)))
    pooled = {k: v for k, v in rows.items() if k[2] in keep}
    return pooled, sorted(keep), dropped


def summarise(pooled, control="isotropic"):
    by = defaultdict(dict)
    for (q, cond, seed), r in pooled.items():
        by[(q, seed)][cond] = int(r["correct_rank"]) == 1
    mses = [float(r["effective_mse"]) for r in pooled.values()]
    n = len(by)
    base = sum(1 for k in by if by[k].get(control)) / n
    out = {"n": n, "control": base,
           "mse": (min(mses), max(mses))}
    conds = sorted({c for k in by for c in by[k]})
    for c in conds:
        if c == control:
            continue
        top1 = sum(1 for k in by if by[k].get(c)) / n
        b = sum(1 for k in by if by[k].get(control) and not by[k].get(c))
        d = sum(1 for k in by if not by[k].get(control) and by[k].get(c))
        out[c] = (top1, top1 - base, exact_mcnemar(b, d), b, d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eot_globs", nargs="+", required=True,
                    help="glob(s) matching the EOT exports, one set per host")
    ap.add_argument("--control_dir", default="",
                    help="directory of the matching un-hardened exports, "
                         "named <sanitizer>.csv")
    ap.add_argument("--control_none", default="",
                    help="export holding the un-hardened, un-preprocessed "
                         "control; the 'none' row has no <sanitizer>.csv "
                         "counterpart in control_dir.")
    ap.add_argument("--conditions", type=int, default=5)
    ap.add_argument("--transfer", default="transfer_3")
    args = ap.parse_args()

    files = defaultdict(list)
    for g in args.eot_globs:
        for f in glob.glob(g):
            files[Path(f).stem.replace("eot_", "")].append(f)

    print(f"{'sanitizer':10s} {'seeds':>7} {'n':>6} {'iso':>7} {'w.box':>7} "
          f"{'transfer':>9} {'delta':>8} {'p':>10}   {'no-EOT transfer':>15}")
    for san in ["none", "jpeg75", "jpeg50", "blur", "denoise"]:
        if san not in files:
            continue
        pooled, seeds, dropped = load_pool(files[san], args.conditions)
        if not pooled:
            print(f"{san:10s}  no complete seed")
            continue
        s = summarise(pooled)
        t, delta, p, _, _ = s[args.transfer]
        wb = s["white_box"][0]
        ctrl = ""
        if args.control_dir:
            cf = Path(args.control_dir) / f"{san}.csv"
            if san == "none":
                cf = Path(args.control_none) if args.control_none else cf
            if cf.is_file():
                cp, cs, _ = load_pool([str(cf)], args.conditions)
                if cp:
                    c = summarise(cp)
                    ctrl = f"{c[args.transfer][0]:.4f} (d={c[args.transfer][1]:+.4f})"
        print(f"{san:10s} {len(seeds):>7} {s['n']:>6} {s['control']:7.4f} "
              f"{wb:7.4f} {t:9.4f} {delta:+8.4f} {p:10.3g}   {ctrl:>15}")
        if dropped:
            print(f"           dropped incomplete seeds: "
                  f"{[(d[0], f'{d[1]}/{d[2]} queries complete') for d in dropped]}")
        lo, hi = s["mse"]
        if hi - lo > 1e-3:
            print(f"           WARNING delivered MSE not matched: [{lo}, {hi}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
