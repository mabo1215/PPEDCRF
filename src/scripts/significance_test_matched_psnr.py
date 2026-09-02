"""Formal paired significance test for the matched-PSNR comparison.

`matched_psnr_from_sweep.py` reports, for each sigma-tunable variant at a
target PSNR, the seed-averaged Top-1 accuracy read off `summary.csv`. The
main paper and appendix describe several variants as "statistically
indistinguishable" at matched PSNR purely because their rounded aggregate
Top-1 values are identical to three decimal places. With only 12 paired
locations x 3 seeds (n=36 paired binary outcomes) per cell, this is a claim
about statistical indistinguishability made without a statistical test, and
the paper says nothing about how much power that sample size actually has.

This script fixes that gap: for a chosen target PSNR and a pair of variants
(each evaluated at its own PSNR-matched sigma, as selected by
`matched_psnr_from_sweep.py`'s nearest-neighbour rule), it aligns the two
variants' per-query Top-1 hit/miss outcomes by (query_id, seed) and reports:

  * McNemar's exact test on the discordant pairs (the correct test for
    paired binary outcomes; a plain diff-of-means test is not appropriate
    here because the same 12 queries x 3 seeds appear on both sides).
  * A query-level cluster bootstrap 95% CI on the Top-1 difference (queries,
    not (query, seed) rows, are resampled, since seeds are repeated
    measurements of the same 12 underlying scenes).
  * The minimum discordant-pair asymmetry that would have reached
    significance at alpha=0.05 given the realized total pair count, so a
    reader can judge how much power the comparison actually had.

Pure post-processing over existing `per_query.csv` sigma-sweep exports; no
GPU or new inference is required to run it.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

SIGMA_DEPENDENT_VARIANTS = [
    "full",
    "no_temporal",
    "no_ncp",
    "unary_only",
    "no_dcrf",
    "global_noise",
]

VARIANT_LABELS = {
    "full": "PPEDCRF",
    "no_temporal": "w/o temporal consistency",
    "no_ncp": "w/o NCP (fixed strength)",
    "unary_only": "unary-only + NCP",
    "no_dcrf": "no-DCRF + fixed strength",
    "global_noise": "global Gaussian noise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired significance test for the matched-PSNR comparison."
    )
    parser.add_argument(
        "--sweep_dir",
        required=True,
        help="Directory containing one sigma_<S>/ subdir per sweep point "
        "(same layout consumed by matched_psnr_from_sweep.py).",
    )
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--gallery_size", type=int, default=48)
    parser.add_argument("--targets", type=float, nargs="+", default=[30.0, 33.0, 36.0])
    parser.add_argument(
        "--compare_against",
        nargs="+",
        default=["global_noise"],
        help="Variant(s) every other sigma-tunable variant is compared against.",
    )
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_sweep(sweep_dir: str, backbone: str, gallery_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, per_query_rows = [], []
    for sigma_dir in sorted(Path(sweep_dir).glob("sigma_*")):
        sigma = float(sigma_dir.name.replace("sigma_", ""))
        summary_path = sigma_dir / "summary.csv"
        per_query_path = sigma_dir / "per_query.csv"
        if not summary_path.exists() or not per_query_path.exists():
            continue
        summary_df = pd.read_csv(summary_path)
        summary_sub = summary_df[
            (summary_df.backbone == backbone) & (summary_df.gallery_size == gallery_size)
        ].copy()
        summary_sub["sigma"] = sigma
        summary_rows.append(summary_sub)

        per_query_df = pd.read_csv(per_query_path)
        per_query_sub = per_query_df[
            (per_query_df.backbone == backbone) & (per_query_df.gallery_size == gallery_size)
        ].copy()
        per_query_sub["sigma"] = sigma
        per_query_rows.append(per_query_sub)

    if not summary_rows:
        raise ValueError(f"No sigma_*/summary.csv or per_query.csv found under {sweep_dir}")
    summary_all = pd.concat(summary_rows, ignore_index=True)
    per_query_all = pd.concat(per_query_rows, ignore_index=True)
    per_query_all["top1_hit"] = (per_query_all["correct_rank"].astype(float) == 1).astype(int)
    return summary_all, per_query_all


def nearest_sigma_for_variant(summary_df: pd.DataFrame, variant: str, target_psnr: float) -> float:
    sub = summary_df[summary_df.variant == variant].copy()
    if sub.empty:
        raise ValueError(f"No summary rows for variant {variant!r}")
    sub["gap"] = (sub["psnr_mean_mean"] - target_psnr).abs()
    return float(sub.sort_values("gap").iloc[0]["sigma"])


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def min_significant_asymmetry(n_discordant_upper_bound: int, alpha: float) -> int:
    """Smallest |b-c| (at maximal asymmetry, i.e. c=0) that reaches significance
    for a given total discordant-pair count, scanned up to n_discordant_upper_bound.
    Used to report how much power the realized sample size actually had."""
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
    """Bootstrap the mean(hit_a) - mean(hit_b) difference by resampling unique
    query_ids with replacement (queries are the independent sampling unit;
    the three seeds are repeated measurements of the same underlying scene)."""
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


def compare_pair(
    per_query_df: pd.DataFrame,
    variant_a: str,
    sigma_a: float,
    variant_b: str,
    sigma_b: float,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict:
    rows_a = per_query_df[(per_query_df.variant == variant_a) & (per_query_df.sigma == sigma_a)]
    rows_b = per_query_df[(per_query_df.variant == variant_b) & (per_query_df.sigma == sigma_b)]
    merged = pd.merge(
        rows_a[["query_id", "seed", "top1_hit"]],
        rows_b[["query_id", "seed", "top1_hit"]],
        on=["query_id", "seed"],
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        raise ValueError(f"No aligned (query_id, seed) pairs between {variant_a} and {variant_b}")

    hit_a = merged["top1_hit_a"].to_numpy(dtype=int)
    hit_b = merged["top1_hit_b"].to_numpy(dtype=int)
    b = int(np.sum((hit_a == 1) & (hit_b == 0)))  # a hits, b misses
    c = int(np.sum((hit_a == 0) & (hit_b == 1)))  # a misses, b hits
    n_pairs = len(merged)

    p_value = mcnemar_exact_p(b, c)
    diff, ci_lo, ci_hi = cluster_bootstrap_ci(
        hit_a, hit_b, merged["query_id"].to_numpy(), n_boot, alpha, rng
    )
    min_asym = min_significant_asymmetry(n_pairs, alpha)

    return {
        "variant_a": variant_a,
        "matched_sigma_a": sigma_a,
        "top1_a": float(hit_a.mean()),
        "variant_b": variant_b,
        "matched_sigma_b": sigma_b,
        "top1_b": float(hit_b.mean()),
        "n_pairs": n_pairs,
        "n_discordant": b + c,
        "mcnemar_b_minus_c": b - c,
        "mcnemar_exact_p": p_value,
        "top1_diff": diff,
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "significant_at_alpha": bool(p_value < alpha),
        "min_significant_discordant_asymmetry_at_this_n": min_asym,
    }


def main() -> None:
    args = parse_args()
    summary_df, per_query_df = load_sweep(args.sweep_dir, args.backbone, args.gallery_size)
    rng = np.random.default_rng(args.seed)

    out_rows = []
    for target in args.targets:
        for variant_b in args.compare_against:
            for variant_a in SIGMA_DEPENDENT_VARIANTS:
                if variant_a == variant_b:
                    continue
                sigma_a = nearest_sigma_for_variant(summary_df, variant_a, target)
                sigma_b = nearest_sigma_for_variant(summary_df, variant_b, target)
                result = compare_pair(
                    per_query_df, variant_a, sigma_a, variant_b, sigma_b,
                    args.n_boot, args.alpha, rng,
                )
                result["target_psnr"] = target
                result["label_a"] = VARIANT_LABELS.get(variant_a, variant_a)
                result["label_b"] = VARIANT_LABELS.get(variant_b, variant_b)
                out_rows.append(result)

    out_df = pd.DataFrame(out_rows)
    column_order = [
        "target_psnr", "variant_a", "label_a", "matched_sigma_a", "top1_a",
        "variant_b", "label_b", "matched_sigma_b", "top1_b",
        "n_pairs", "n_discordant", "mcnemar_b_minus_c", "mcnemar_exact_p",
        "top1_diff", "bootstrap_ci_low", "bootstrap_ci_high",
        "significant_at_alpha", "min_significant_discordant_asymmetry_at_this_n",
    ]
    out_df = out_df[column_order]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[significance] wrote {len(out_df)} rows to {output_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
