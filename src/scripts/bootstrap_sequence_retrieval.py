"""Cluster-aware significance test for the M4 sequence-length/pooling table.

M7 (docs/RevisionSuggestions.tex) requires location/query cluster, not
query-seed rows, as the independent statistical unit, with seeds treated as
within-cluster repeated measurements. The M3 real-place table already has a
place-cluster bootstrap (paper/appendix.tex, Table tab:e1_bootstrap); this
script closes the matching gap for M4's clip-length/pooling table, whose
cluster unit is the mined paired-scene query_id (proxy pair identity, not an
officially labeled place id -- the difference is stated in the output and
must be repeated wherever this script's numbers are cited).

Reuses the exact McNemar + query-cluster bootstrap methodology already used
for the F1 matched-PSNR significance test (src/scripts/significance_test_matched_psnr.py),
so the two statistical procedures in this paper stay consistent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def min_significant_asymmetry(n_discordant_upper_bound: int, alpha: float) -> int:
    for n in range(1, n_discordant_upper_bound + 1):
        if mcnemar_exact_p(n, 0) < alpha:
            return n
    return -1


def cluster_bootstrap_ci(
    hit_a: np.ndarray,
    hit_b: np.ndarray,
    query_ids: np.ndarray,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap mean(hit_a) - mean(hit_b) by resampling unique query_ids
    with replacement; seeds are repeated measurements within each query."""
    unique_queries = np.unique(query_ids)
    observed_diff = float(hit_a.mean() - hit_b.mean())
    diffs = np.empty(n_boot, dtype=np.float64)
    row_index_by_query = {q: np.flatnonzero(query_ids == q) for q in unique_queries}
    for i in range(n_boot):
        sampled = rng.choice(unique_queries, size=len(unique_queries), replace=True)
        idx = np.concatenate([row_index_by_query[q] for q in sampled])
        diffs[i] = hit_a[idx].mean() - hit_b[idx].mean()
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return observed_diff, float(lo), float(hi)


def compare_cell(
    per_query_df: pd.DataFrame,
    backbone: str,
    clip_len: int,
    pooling: str,
    variant_a: str,
    variant_b: str,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict:
    cell = per_query_df[
        (per_query_df.backbone == backbone)
        & (per_query_df.clip_len == clip_len)
        & (per_query_df.pooling == pooling)
    ]
    rows_a = cell[cell.variant == variant_a][["query_id", "seed", "top1_hit"]]
    rows_b = cell[cell.variant == variant_b][["query_id", "seed", "top1_hit"]]
    # raw rows carry seed="raw" (one row per query, no seed axis); align on
    # query_id only in that case rather than (query_id, seed).
    join_keys = ["query_id"] if "raw" in (variant_a, variant_b) else ["query_id", "seed"]
    merged = pd.merge(rows_a, rows_b, on=join_keys, suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError(f"No aligned rows for {backbone}/{clip_len}/{pooling}: {variant_a} vs {variant_b}")

    hit_a = merged["top1_hit_a"].to_numpy(dtype=int)
    hit_b = merged["top1_hit_b"].to_numpy(dtype=int)
    b = int(np.sum((hit_a == 1) & (hit_b == 0)))
    c = int(np.sum((hit_a == 0) & (hit_b == 1)))
    n_pairs = len(merged)
    n_clusters = merged["query_id"].nunique()

    p_value = mcnemar_exact_p(b, c)
    diff, ci_lo, ci_hi = cluster_bootstrap_ci(hit_a, hit_b, merged["query_id"].to_numpy(), n_boot, alpha, rng)
    min_asym = min_significant_asymmetry(n_pairs, alpha)

    return {
        "backbone": backbone,
        "clip_len": int(clip_len),
        "pooling": pooling,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "n_pairs": n_pairs,
        "n_query_clusters": int(n_clusters),
        "discordant_b": b,
        "discordant_c": c,
        "mcnemar_p_exact": p_value,
        "top1_diff_a_minus_b": diff,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "min_significant_discordant_pairs": min_asym,
        "cluster_unit": "mined paired-scene query_id (proxy pair identity, not an official place id)",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster-aware significance test for M4 sequence-retrieval.")
    parser.add_argument("--per_query_csv", nargs="+", required=True, help="One or more per_query.csv paths, one per backbone.")
    parser.add_argument("--compare", nargs=2, action="append", metavar=("VARIANT_A", "VARIANT_B"), default=None,
                         help="Variant pair to compare (repeatable). Default: full vs raw, and full vs global_noise.")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    comparisons = args.compare or [("full", "raw"), ("full", "global_noise")]

    frames = []
    for path in args.per_query_csv:
        df = pd.read_csv(path)
        df["top1_hit"] = (df["correct_rank"].astype(float) == 1).astype(int)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    rows = []
    for backbone in sorted(all_df.backbone.unique()):
        sub = all_df[all_df.backbone == backbone]
        for clip_len in sorted(sub.clip_len.unique()):
            for pooling in sorted(sub.pooling.unique()):
                for variant_a, variant_b in comparisons:
                    try:
                        rows.append(
                            compare_cell(sub, backbone, clip_len, pooling, variant_a, variant_b, args.n_boot, args.alpha, rng)
                        )
                    except ValueError as exc:
                        rows.append({"backbone": backbone, "clip_len": int(clip_len), "pooling": pooling,
                                     "variant_a": variant_a, "variant_b": variant_b, "error": str(exc)})

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    n_sig = int((out_df.get("bootstrap_ci_lo", pd.Series(dtype=float)) > 0).sum() +
                (out_df.get("bootstrap_ci_hi", pd.Series(dtype=float)) < 0).sum()) if "bootstrap_ci_lo" in out_df else 0
    n_total = len(out_df)
    print(f"[bootstrap] wrote {n_total} cells to {args.output}; {n_sig} individually significant at alpha={args.alpha}")


if __name__ == "__main__":
    main()
