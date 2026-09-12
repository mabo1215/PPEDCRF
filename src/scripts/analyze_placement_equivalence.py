"""Equivalence and multiplicity analysis for the placement family.

The manuscript's central negative claim is a claim about the placement axis,
yet the protocol declared only the four direction contrasts confirmatory and
left the placement family uncorrected, and the equivalence verdicts rested on
an interval rule this paper defines itself rather than on a named procedure.
Both are answerable from the released per-query rows with no new experiment.

This computes, for every placement contrast on the primary place-labelled
manifest and on its strong-attacker repeat:

  * two one-sided tests (TOST) at the declared margin, run as a
    place-clustered bootstrap so the equivalence procedure uses the same unit
    of inference the protocol prescribes elsewhere;
  * Holm correction across the placement family, treating it as confirmatory;
  * a place-clustered permutation test, which makes no distributional
    assumption about a three-seed-averaged binary outcome and so answers the
    objection that a signed-rank test is being applied outside its setting;
  * the number of query clusters that would be needed to certify the declared
    margin at the observed paired variance.

Outputs a JSON summary consumed by the manuscript's text and by
make_msls_placement_table.py.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]

LABEL = {
    "learned": "learned support",
    "oracle_grad": "score-gradient",
    "anti_oracle_grad": "anti-score-grad.",
    "saliency": "saliency",
    "center": "centre bias",
    "random_fixed": "fixed random",
    "edge": "edge magnitude",
    "margin_oracle": "margin gradient",
    "anti_margin_oracle": "anti-margin grad.",
}
ORDER = list(LABEL)


def per_query(rows: Sequence[dict], placement: str) -> Dict[str, float]:
    acc = collections.defaultdict(list)
    for r in rows:
        if r["placement"] == placement:
            acc[r["query_id"]].append(1.0 if int(r["correct_rank"]) == 1 else 0.0)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def _cluster_index(groups: Sequence[str]):
    idx = collections.defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    keys = sorted(idx)
    return keys, idx


def cluster_bootstrap(d: np.ndarray, groups: Sequence[str], n: int, seed: int) -> np.ndarray:
    """Resample whole clusters, carrying every observation of a sampled one."""
    rng = np.random.default_rng(seed)
    keys, idx = _cluster_index(groups)
    k = len(keys)
    out = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, k, k)
        out[b] = d[np.concatenate([idx[keys[j]] for j in pick])].mean()
    return out


def tost(d: np.ndarray, groups: Sequence[str], margin: float, n: int, seed: int) -> float:
    """Place-clustered bootstrap TOST.

    The equivalence p-value is the larger of the two one-sided bootstrap
    tail probabilities: the share of resampled means at or above +margin and
    the share at or below -margin. Equivalence is claimed at level alpha when
    this is below alpha, which is the usual TOST reading.
    """
    boots = cluster_bootstrap(d, groups, n, seed)
    upper = float(np.mean(boots >= margin))
    lower = float(np.mean(boots <= -margin))
    return max(upper, lower)


def permutation_p(d: np.ndarray, groups: Sequence[str], n: int, seed: int) -> float:
    """Place-clustered sign-flip permutation test of the paired difference.

    Flipping the sign of every difference within a whole place respects the
    clustering, assumes only exchangeability of the two arms within a place,
    and makes no assumption about the shape of a three-seed-averaged binary
    outcome.
    """
    rng = np.random.default_rng(seed)
    keys, idx = _cluster_index(groups)
    obs = abs(float(d.mean()))
    k = len(keys)
    blocks = [np.asarray(idx[g]) for g in keys]
    hits = 0
    for _ in range(n):
        signs = rng.integers(0, 2, k) * 2 - 1
        tot = 0.0
        for s, b in zip(signs, blocks):
            tot += s * d[b].sum()
        if abs(tot / d.size) >= obs - 1e-15:
            hits += 1
    return (hits + 1) / (n + 1)


def holm(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


def required_clusters(d: np.ndarray, groups: Sequence[str], margin: float) -> int:
    """Clusters needed for a 95% interval half-width of `margin`.

    Uses the cluster-mean variance, which is what the place-clustered
    bootstrap is estimating, so the answer is on the same footing as the
    intervals the manuscript prints.
    """
    keys, idx = _cluster_index(groups)
    means = np.array([d[idx[g]].mean() for g in keys])
    k = len(keys)
    var = float(means.var(ddof=1))
    if var <= 0:
        return 0
    return int(math.ceil(1.96 ** 2 * var / margin ** 2))


def analyse(row_paths: Sequence[str], places_path: str, margin: float,
            boots: int, perms: int, seed: int) -> dict:
    place_of = {}
    with open(places_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            place_of[r["query_id"]] = r["correct_place"]

    cells: List[dict] = []
    seen = set()
    for path in row_paths:
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        ref = per_query(rows, "uniform")
        if not ref:
            continue
        for name in ORDER:
            if name in seen:
                continue
            cur = per_query(rows, name)
            if not cur:
                continue
            seen.add(name)
            qs = sorted(set(cur) & set(ref))
            d = np.array([cur[q] - ref[q] for q in qs])
            g = [place_of.get(q, q) for q in qs]
            nz = d[d != 0]
            cells.append({
                "placement": name,
                "label": LABEL[name],
                "n_queries": len(qs),
                "n_places": len(set(g)),
                "delta": float(d.mean()),
                "n_discordant": int(nz.size),
                "p_wilcoxon": float(wilcoxon(nz).pvalue) if nz.size else 1.0,
                "p_permutation": permutation_p(d, g, perms, seed),
                "p_tost": tost(d, g, margin, boots, seed),
                "required_clusters": required_clusters(d, g, margin),
            })

    for key, adj in (("p_wilcoxon", "p_wilcoxon_holm"),
                     ("p_permutation", "p_permutation_holm")):
        for c, v in zip(cells, holm([c[key] for c in cells])):
            c[adj] = v

    return {
        "margin": margin,
        "n_contrasts": len(cells),
        "bootstrap_resamples": boots,
        "permutations": perms,
        "cells": cells,
        "n_equivalent_tost_05": sum(1 for c in cells if c["p_tost"] < 0.05),
        "min_p_wilcoxon_holm": min(c["p_wilcoxon_holm"] for c in cells) if cells else None,
        "min_p_permutation_holm": min(c["p_permutation_holm"] for c in cells) if cells else None,
        "max_required_clusters": max(c["required_clusters"] for c in cells) if cells else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="+", default=[
        str(REPO / "src/exports/icme2027_placement_msls/final/per_query.csv"),
        str(REPO / "src/exports/margin_oracle/per_query.csv")])
    ap.add_argument("--places",
                    default=str(REPO / "src/exports/tifs_d6/d6_r18_plain.csv"))
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=str(
        REPO / "src/exports/placement_equivalence/msls_resnet18.json"))
    args = ap.parse_args()

    summary = analyse(args.rows, args.places, args.margin,
                      args.bootstrap, args.permutations, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[done] wrote {out}")
    print(f"margin +/-{summary['margin']}, {summary['n_contrasts']} contrasts")
    hdr = (f"{'placement':18s} {'delta':>9s} {'disc':>5s} {'wilcox':>8s} "
           f"{'holm':>7s} {'perm':>7s} {'permHolm':>9s} {'TOST':>7s} {'need n':>7s}")
    print(hdr)
    for c in summary["cells"]:
        print(f"{c['label']:18s} {c['delta']:+9.4f} {c['n_discordant']:5d} "
              f"{c['p_wilcoxon']:8.3f} {c['p_wilcoxon_holm']:7.3f} "
              f"{c['p_permutation']:7.3f} {c['p_permutation_holm']:9.3f} "
              f"{c['p_tost']:7.3f} {c['required_clusters']:7d}")
    print(f"\nTOST equivalent at 0.05: {summary['n_equivalent_tost_05']}"
          f"/{summary['n_contrasts']}")
    print(f"smallest Holm-corrected signed-rank p: {summary['min_p_wilcoxon_holm']:.3f}")
    print(f"smallest Holm-corrected permutation p: {summary['min_p_permutation_holm']:.3f}")
    print(f"clusters needed to certify the margin: {summary['max_required_clusters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
