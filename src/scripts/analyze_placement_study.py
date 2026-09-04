"""Paired analysis of the placement-rule study.

Tests, for every placement rule, whether it differs from the uniform
same-energy control on identical queries and seeds. Uses the exact McNemar
test on paired Top-1 outcomes -- the same procedure used elsewhere in this
work -- plus a query-cluster bootstrap, because repeated seeds of the same
query are not independent draws.

Also summarises the attacker-sensitivity statistics that explain the result.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

REFERENCE = "uniform"


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def cluster_bootstrap_ci(
    per_query: Dict[str, List[int]], resamples: int = 2000, seed: int = 1234
) -> tuple:
    """Bootstrap the mean paired difference, resampling *queries* not rows."""
    keys = sorted(per_query)
    if not keys:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for i in range(resamples):
        pick = rng.integers(0, len(keys), len(keys))
        vals = [v for j in pick for v in per_query[keys[j]]]
        means[i] = float(np.mean(vals)) if vals else 0.0
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyse(df: pd.DataFrame) -> List[Dict[str, object]]:
    df = df.copy()
    df["hit"] = (df["correct_rank"] == 1).astype(int)
    out: List[Dict[str, object]] = []
    for (run_id, backbone), dfb in df.groupby(["run_id", "backbone"]):
        ref = dfb[dfb.placement == REFERENCE].set_index(["query_id", "seed"])["hit"]
        if ref.index.has_duplicates:
            raise SystemExit(f"duplicate (query,seed) within run {run_id}/{backbone}")
        for placement, dfp in dfb.groupby("placement"):
            if placement == REFERENCE:
                continue
            cur = dfp.set_index(["query_id", "seed"])["hit"]
            common = ref.index.intersection(cur.index)
            if len(common) == 0:
                continue
            a = ref.loc[common]
            b_ = cur.loc[common]
            # b: placement wins (ref miss, placement hit); c: the reverse
            b = int(((a == 0) & (b_ == 1)).sum())
            c = int(((a == 1) & (b_ == 0)).sum())
            diffs: Dict[str, List[int]] = {}
            for (q, _s), dv in (b_ - a).items():
                diffs.setdefault(str(q), []).append(int(dv))
            lo, hi = cluster_bootstrap_ci(diffs)
            out.append({
                "run_id": run_id,
                "backbone": backbone,
                "placement": placement,
                "n_pairs": int(len(common)),
                "top1_reference": float(a.mean()),
                "top1_placement": float(b_.mean()),
                "top1_diff": float(b_.mean() - a.mean()),
                "n_discordant": b + c,
                "mcnemar_b_minus_c": b - c,
                "mcnemar_exact_p": exact_mcnemar_p(b, c),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                # Reported separately on purpose. The exact test treats the
                # 3 seeds of a query as independent; the cluster bootstrap does
                # not. When they disagree, the cluster-robust verdict governs,
                # because repeated seeds of one query are not independent draws.
                "mcnemar_significant": bool(exact_mcnemar_p(b, c) < 0.05),
                "cluster_robust_significant": bool(
                    exact_mcnemar_p(b, c) < 0.05 and not (lo <= 0.0 <= hi)),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study_root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.study_root)

    frames = []
    for path in sorted(root.rglob("per_query.csv")):
        sub = pd.read_csv(path)
        # query ids are synthetic pair identifiers (loc_000, ...) and are reused
        # across benchmarks, so rows must be tagged by their source run or the
        # 12-pair and 50-pair experiments silently merge.
        sub["run_id"] = str(path.parent.relative_to(root))
        frames.append(sub)
    if not frames:
        raise SystemExit(f"no per_query.csv under {root}")
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(df)} rows, "
          f"{df.placement.nunique()} placements, {df.backbone.nunique()} backbones")

    # energy gate: every placement must carry the learned map's energy
    gate = df.groupby(["run_id", "backbone", "query_id", "seed"])["effective_weight_energy"]
    rel = gate.transform(lambda s: (s - s.mean()).abs() / max(abs(s.mean()), 1e-12))
    worst = float(rel.max())
    print(f"energy-conservation gate: max relative error = {worst:.2e}")
    if worst > 1e-3:
        raise SystemExit("ENERGY GATE FAILED - placements are not energy-matched")

    rows = analyse(df)
    res = pd.DataFrame(rows).sort_values(["run_id", "backbone", "placement"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False)

    print("\n=== Top-1 by placement (pooled over backbones/seeds) ===")
    pooled = df.assign(hit=(df.correct_rank == 1).astype(int)) \
                .groupby("placement")["hit"].mean().sort_values()
    for k, v in pooled.items():
        print(f"  {k:20s} {v:.4f}")

    n_mc = int(res["mcnemar_significant"].sum())
    n_sig = int(res["cluster_robust_significant"].sum())
    print(f"\n=== paired vs '{REFERENCE}' ===")
    print(f"  comparisons: {len(res)}")
    print(f"  zero-discordant: {int((res.n_discordant == 0).sum())}")
    print(f"  exact-McNemar significant:   {n_mc}")
    print(f"  cluster-robust significant:  {n_sig}")
    print(f"\nwrote {args.output}")

    sens = sorted(root.rglob("sensitivity_stats.jsonl"))
    if sens:
        recs = [json.loads(l) for p in sens for l in p.open(encoding="utf-8") if l.strip()]
        if recs:
            sdf = pd.DataFrame(recs)
            print("\n=== attacker-sensitivity statistics ===")
            for col in ("grad_cv", "grad_energy_top10pct_oracle",
                        "grad_energy_top10pct_learned", "spearman_learned_vs_grad"):
                if col in sdf:
                    print(f"  {col:32s} mean={sdf[col].mean():.4f} "
                          f"[{sdf[col].min():.4f}, {sdf[col].max():.4f}]")


if __name__ == "__main__":
    main()
