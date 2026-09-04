"""Nearest-PSNR matching for the deterministic mask-guided baselines (N5).

The existing matched-PSNR table (paper/appendix.tex, Section "Matched-PSNR/
Effective-MSE Comparison") explicitly excludes mask-guided blur and mosaic
from the matched comparison because they have no sigma parameter in that
benchmark and are reported only at a single native operating point, with the
resulting PSNR gap stated instead of a true match. This script closes that
gap: it sweeps the blur kernel size and mosaic block size (added to
run_tomm_review_proxy.py's protect_review_clip as --blur_kernel_size /
--mosaic_block_size overrides) and finds, for each requested target PSNR,
the swept point whose achieved PSNR is closest -- exactly the same
nearest-neighbour procedure matched_psnr_from_sweep.py already uses for the
sigma-tunable variants.

Expected sweep directory layout: one subdirectory per swept value, each
containing summary.csv from run_tomm_review_proxy.py --mode proxy (e.g.
<blur_sweep_dir>/k_5/summary.csv, .../k_11/summary.csv, ...).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_sweep(sweep_dir: str, prefix: str, variant: str, backbone: str, gallery_size: int) -> pd.DataFrame:
    rows = []
    for point_dir in sorted(Path(sweep_dir).glob(f"{prefix}_*")):
        summary_path = point_dir / "summary.csv"
        if not summary_path.is_file():
            continue
        value = int(point_dir.name.replace(f"{prefix}_", ""))
        df = pd.read_csv(summary_path)
        sub = df[(df.variant == variant) & (df.backbone == backbone) & (df.gallery_size == gallery_size)].copy()
        if sub.empty:
            continue
        sub["param_value"] = value
        rows.append(sub)
    if not rows:
        raise ValueError(f"No {prefix}_*/summary.csv rows found for variant={variant} under {sweep_dir}")
    return pd.concat(rows, ignore_index=True)


def nearest_match(df: pd.DataFrame, target_psnr: float) -> pd.Series:
    gap = (df["psnr_mean_mean"] - target_psnr).abs()
    return df.loc[gap.idxmin()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched-PSNR table for mask-guided blur/mosaic baselines.")
    parser.add_argument("--blur_sweep_dir", required=True)
    parser.add_argument("--mosaic_sweep_dir", required=True)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--gallery_size", type=int, default=48)
    parser.add_argument("--targets", type=float, nargs="+", default=[30.0, 33.0, 36.0])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blur_df = load_sweep(args.blur_sweep_dir, "k", "masked_blur", args.backbone, args.gallery_size)
    mosaic_df = load_sweep(args.mosaic_sweep_dir, "b", "masked_mosaic", args.backbone, args.gallery_size)

    rows = []
    for target in args.targets:
        for label, df, param_name in (("mask-guided blur", blur_df, "kernel_size"), ("mask-guided mosaic", mosaic_df, "block_size")):
            best = nearest_match(df, target)
            rows.append(
                {
                    "target_psnr": target,
                    "variant": label,
                    param_name: int(best["param_value"]),
                    "actual_psnr": float(best["psnr_mean_mean"]),
                    "psnr_gap": float(abs(best["psnr_mean_mean"] - target)),
                    "top1": float(best["top1"]),
                }
            )
    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[deterministic_baseline_psnr_match] wrote {len(out_df)} rows to {args.output}")
    print(out_df.to_string())


if __name__ == "__main__":
    main()
