"""Aggregate retrieval-margin subgroups from an existing per-query CSV.

Addresses reviewer request R3-2/R3-5/R3-7 (E6): report correct-versus-
hardest-negative similarities and margins before and after sanitization,
split into small-margin (harder) and large-margin (easier) query groups.

Bucket membership is fixed by each query's *raw* (pre-sanitization) retrieval
margin at a given backbone/gallery_size, using a median split. This keeps the
grouping an intrinsic property of the query rather than an artifact of any
one sanitization variant, so the same two groups can be compared before and
after sanitization for every variant.

This is a pure post-processing step over an existing ``per_query.csv`` from
``run_tomm_review_proxy.py --mode proxy``; it requires no GPU and no new
protection runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small/large-margin subgroup aggregation.")
    parser.add_argument("--per_query_csv", nargs="+", required=True,
                         help="One or more per_query.csv files from run_tomm_review_proxy.py.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def assign_margin_buckets(df: pd.DataFrame) -> pd.DataFrame:
    raw = df[df["variant"] == "raw"][["query_id", "backbone", "gallery_size", "retrieval_margin"]]
    raw = raw.rename(columns={"retrieval_margin": "raw_margin"})

    thresholds = (
        raw.groupby(["backbone", "gallery_size"])["raw_margin"]
        .median()
        .rename("raw_margin_median")
        .reset_index()
    )
    raw = raw.merge(thresholds, on=["backbone", "gallery_size"], how="left")
    raw["margin_bucket"] = np.where(
        raw["raw_margin"] <= raw["raw_margin_median"], "small_margin", "large_margin"
    )

    merged = df.merge(
        raw[["query_id", "backbone", "gallery_size", "margin_bucket", "raw_margin"]],
        on=["query_id", "backbone", "gallery_size"],
        how="inner",
    )
    return merged


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["variant", "backbone", "gallery_size", "margin_bucket"]
    agg = df.groupby(group_cols).agg(
        num_queries=("query_id", "count"),
        top1=("correct_rank", lambda s: float(np.mean(s.astype(int) == 1))),
        correct_similarity_mean=("correct_similarity", "mean"),
        correct_similarity_std=("correct_similarity", "std"),
        hardest_negative_similarity_mean=("hardest_negative_similarity", "mean"),
        hardest_negative_similarity_std=("hardest_negative_similarity", "std"),
        retrieval_margin_mean=("retrieval_margin", "mean"),
        retrieval_margin_std=("retrieval_margin", "std"),
        raw_margin_mean=("raw_margin", "mean"),
    ).reset_index()
    agg["retrieval_margin_std"] = agg["retrieval_margin_std"].fillna(0.0)
    agg["correct_similarity_std"] = agg["correct_similarity_std"].fillna(0.0)
    agg["hardest_negative_similarity_std"] = agg["hardest_negative_similarity_std"].fillna(0.0)
    return agg.sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    df = load_rows(args.per_query_csv)
    bucketed = assign_margin_buckets(df)
    summary = aggregate(bucketed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"[margin] wrote {len(summary)} subgroup rows to {output_path}")


if __name__ == "__main__":
    main()
