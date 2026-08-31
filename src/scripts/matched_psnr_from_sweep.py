"""Build a matched-PSNR/effective-MSE comparison table from a sigma sweep.

Addresses reviewer request R2-2/R3-3/R3-4 (E2): compare PPEDCRF and its
ablations against the deterministic baselines at matched image quality
rather than only at one fixed noise budget.

Input is a directory of ``run_tomm_review_proxy.py --mode proxy --sigma S``
outputs, one subdirectory per sigma value, each containing ``summary.csv``.
For every (target PSNR, variant) pair we pick the sigma whose achieved PSNR
is closest to the target and report that operating point. Deterministic
baselines (mask-guided blur/mosaic) have a fixed PSNR regardless of sigma, so
their row is taken directly from any single sweep point.

This is a pure post-processing step; it requires no GPU.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

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
FIXED_VARIANTS = ["masked_blur", "masked_mosaic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched-PSNR table from a sigma sweep.")
    parser.add_argument("--sweep_dir", required=True,
                         help="Directory containing one sigma_<S>/ subdir per sweep point.")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--gallery_size", type=int, default=48)
    parser.add_argument("--targets", type=float, nargs="+", default=[30.0, 33.0, 36.0])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_sweep(sweep_dir: str, backbone: str, gallery_size: int) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(glob.glob(str(Path(sweep_dir) / "sigma_*" / "summary.csv"))):
        sigma = float(Path(summary_path).parent.name.replace("sigma_", ""))
        df = pd.read_csv(summary_path)
        sub = df[(df.backbone == backbone) & (df.gallery_size == gallery_size)].copy()
        sub["sigma"] = sigma
        rows.append(sub)
    if not rows:
        raise ValueError(f"No summary.csv files found under {sweep_dir}")
    return pd.concat(rows, ignore_index=True)


def nearest_sigma_row(df: pd.DataFrame, variant: str, target_psnr: float) -> pd.Series:
    sub = df[df.variant == variant].copy()
    if sub.empty:
        raise ValueError(f"No rows for variant {variant}")
    sub["psnr_gap"] = (sub["psnr_mean_mean"] - target_psnr).abs()
    return sub.sort_values("psnr_gap").iloc[0]


def main() -> None:
    args = parse_args()
    df = load_sweep(args.sweep_dir, args.backbone, args.gallery_size)

    out_rows = []
    for target in args.targets:
        for variant in SIGMA_DEPENDENT_VARIANTS:
            row = nearest_sigma_row(df, variant, target)
            out_rows.append({
                "target_psnr": target,
                "variant": variant,
                "label": row["label"],
                "matched_sigma": row["sigma"],
                "actual_psnr": row["psnr_mean_mean"],
                "psnr_gap": row["psnr_gap"],
                "actual_mse": float((255.0 ** 2) / (10.0 ** (row["psnr_mean_mean"] / 10.0))),
                "top1": row["top1"],
                "retrieval_margin_mean": row.get("retrieval_margin_mean", np.nan),
            })
        for variant in FIXED_VARIANTS:
            sub = df[df.variant == variant]
            if sub.empty:
                continue
            row = sub.iloc[0]
            out_rows.append({
                "target_psnr": target,
                "variant": variant,
                "label": row["label"],
                "matched_sigma": None,
                "actual_psnr": row["psnr_mean_mean"],
                "psnr_gap": abs(row["psnr_mean_mean"] - target),
                "actual_mse": float((255.0 ** 2) / (10.0 ** (row["psnr_mean_mean"] / 10.0))),
                "top1": row["top1"],
                "retrieval_margin_mean": row.get("retrieval_margin_mean", np.nan),
            })

    out_df = pd.DataFrame(out_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[matched-psnr] wrote {len(out_df)} rows to {output_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
