"""Re-check the placement study with the query as the unit of inference.

The manuscript's placement table already claims cluster-robust significance,
unlike its direction tables, so this is a verification rather than a
correction: it recomputes every placement against the uniform control by
collapsing each query to its seed- and backbone-averaged hit rate, then takes
a query-cluster bootstrap interval and a Wilcoxon signed-rank test over the
per-query differences.

Pooling across backbones follows the manuscript's own table, which reports one
delta per placement over six attackers; the clustering is on the query, which
is the unit the seeds and backbones repeat over.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

CONTROL = "uniform"


def load(paths, root):
    """(placement, query, backbone, seed) -> hit.

    The query id is namespaced by the benchmark directory: proxy12 and proxy50
    both number their queries from loc_000, so the bare id would merge twelve
    of proxy50's queries with all of proxy12's.
    """
    hits = {}
    for path in paths:
        bench = os.path.relpath(path, root).replace(os.sep, "/").split("/")[0]
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["placement"], bench + "/" + r["query_id"],
                       r["backbone"], r["seed"])
                hits[key] = int(r["correct_rank"]) == 1
    return hits


def compare(hits, placement, n_boot=10000, seed=0):
    per_query = defaultdict(lambda: [[], []])
    pairs = []
    for (p, q, b, sd), hit in hits.items():
        if p == CONTROL:
            per_query[q][0].append(hit)
        elif p == placement:
            per_query[q][1].append(hit)
    for (p, q, b, sd), hit in hits.items():
        if p == placement:
            ctrl = hits.get((CONTROL, q, b, sd))
            if ctrl is not None:
                pairs.append((ctrl, hit))
    qs = [q for q, (a, b) in per_query.items() if a and b]
    if not qs:
        return None
    ctrl = np.array([np.mean(per_query[q][0]) for q in qs])
    cond = np.array([np.mean(per_query[q][1]) for q in qs])
    diff = cond - ctrl
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qs), size=(n_boot, len(qs)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nz = diff[diff != 0]
    wp = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    b01 = sum(1 for a, b in pairs if a and not b)
    b10 = sum(1 for a, b in pairs if b and not a)
    mp = float(binomtest(b10, b01 + b10, 0.5).pvalue) if (b01 + b10) else 1.0
    return {"n_queries": len(qs), "n_pairs": len(pairs), "delta": diff.mean(),
            "ci_low": lo, "ci_high": hi, "wilcoxon_p": wp, "mcnemar_p": mp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="src/outputs/icme2027_placement_study")
    ap.add_argument("--label", default="released checkpoint")
    args = ap.parse_args()

    paths = glob.glob(os.path.join(args.root, "**", "per_query.csv"),
                      recursive=True)
    hits = load(paths, args.root)
    placements = sorted({p for (p, _q, _b, _s) in hits} - {CONTROL})
    print("%s: %d file(s), %d rows, %d placements"
          % (args.label, len(paths), len(hits), len(placements)))
    print("%-22s %6s %7s %8s %18s %10s %10s"
          % ("placement", "nq", "npair", "delta", "95% CI (query)",
             "wilcox p", "McNemar p"))
    for p in placements:
        st = compare(hits, p)
        if st is None:
            continue
        flag = "  *" if st["wilcoxon_p"] < 0.05 else ""
        print("%-22s %6d %7d %+8.4f  [%+7.4f,%+7.4f] %10.2e %10.2e%s"
              % (p, st["n_queries"], st["n_pairs"], st["delta"], st["ci_low"],
                 st["ci_high"], st["wilcoxon_p"], st["mcnemar_p"], flag))
    print("positive delta means worse privacy; * marks query-level significance")


if __name__ == "__main__":
    main()
