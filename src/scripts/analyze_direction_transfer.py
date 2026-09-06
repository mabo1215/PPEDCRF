"""Paired analysis of a direction-transfer export.

Each condition is compared against the ``isotropic`` control on the same
(query, seed) units with an exact McNemar test, so the comparison is paired
rather than a difference of two independent means.  The delivered-MSE spread
is reported as an energy gate: conditions are only comparable if every one of
them was released at the same measured distortion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

CONTROL = "isotropic"
UNIT = ["query_id", "seed"]


def exact_mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    """Exact McNemar on paired binary outcomes; returns (b01, b10, p)."""
    b01 = int(((a == 1) & (b == 0)).sum())   # control hit, condition miss
    b10 = int(((a == 0) & (b == 1)).sum())   # control miss, condition hit
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    return b01, b10, float(binomtest(b10, n, 0.5).pvalue)


def analyse(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hit"] = (df["correct_rank"] == 1).astype(int)

    spread = float(df["effective_mse"].max() - df["effective_mse"].min())
    print(f"[gate] delivered MSE in "
          f"[{df['effective_mse'].min():.4f}, {df['effective_mse'].max():.4f}] "
          f"spread={spread:.2e} "
          f"{'OK' if spread <= 1e-3 else 'FAIL -- conditions not comparable'}")

    wide = df.pivot_table(index=UNIT, columns="condition", values="hit")
    if CONTROL not in wide.columns:
        raise SystemExit(f"no '{CONTROL}' rows in {path}")
    ctrl = wide[CONTROL]

    rows = []
    for cond in wide.columns:
        if cond == CONTROL:
            paired = ctrl.dropna().to_frame(CONTROL)
            b01, b10, p = 0, 0, 1.0
        else:
            paired = wide[[CONTROL, cond]].dropna()
            b01, b10, p = exact_mcnemar(paired[CONTROL], paired[cond])
        rows.append({
            "condition": cond,
            "n": int(len(paired)),
            "top1": float(wide[cond].mean()),
            "delta": float(wide[cond].mean() - ctrl.mean()),
            "discordant_control_only": b01,
            "discordant_condition_only": b10,
            "exact_p": p,
        })
    return pd.DataFrame(rows).sort_values("top1", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("per_query_csv", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional path to write the summary table as CSV")
    args = ap.parse_args()

    table = analyse(args.per_query_csv)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False,
                              float_format=lambda v: f"{v:.4f}"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
