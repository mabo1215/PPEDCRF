"""Paired analysis of a direction-transfer export.

Each condition is compared against the ``isotropic`` control on paired units.
Two units are available:

``--unit pair`` (the original behaviour) treats every (query, seed) row as an
observation and runs an exact McNemar test over them. Seeds of one query share
the frame, the gallery and the query's intrinsic difficulty, so this overstates
the evidence by up to the number of seeds.

``--unit query`` collapses each query to its seed-averaged hit rate under every
condition and reports a query-cluster bootstrap 95% CI for delta (queries
resampled with replacement, all of a query's seeds travelling together), a
Wilcoxon signed-rank test on the per-query differences, and the number of
queries carried. This is the unit the paper's E1 analysis already uses.

The delivered-MSE spread is reported as an energy gate: conditions are only
comparable if every one of them was released at the same measured distortion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

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


def query_level(wide_pair: pd.DataFrame, n_boot: int = 10000,
                seed: int = 0) -> pd.DataFrame:
    """Per-condition delta against the control with the query as the unit."""
    per_query = wide_pair.groupby(level="query_id").mean()
    ctrl = per_query[CONTROL]
    rng = np.random.default_rng(seed)
    n = len(per_query)
    idx = rng.integers(0, n, size=(n_boot, n))
    rows = []
    for cond in per_query.columns:
        diff = (per_query[cond] - ctrl).to_numpy()
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        nonzero = diff[diff != 0]
        p = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
        rows.append({
            "condition": cond,
            "n_queries": int(n),
            "seeds_per_query": float(wide_pair[cond].groupby(level="query_id")
                                     .count().mean()),
            "top1": float(per_query[cond].mean()),
            "delta": float(diff.mean()),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "queries_worse": int((diff < 0).sum()),
            "queries_better": int((diff > 0).sum()),
            "wilcoxon_p": p,
        })
    return pd.DataFrame(rows).sort_values("top1", ascending=False)


def analyse(path: Path, unit: str = "pair") -> pd.DataFrame:
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
    if unit == "query":
        return query_level(wide)
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
    ap.add_argument("--unit", choices=("pair", "query"), default="pair",
                    help="unit of inference: 'pair' = every (query, seed) row "
                         "with exact McNemar (original); 'query' = seed-"
                         "averaged per query, cluster-bootstrap CI and "
                         "Wilcoxon over queries")
    args = ap.parse_args()

    table = analyse(args.per_query_csv, unit=args.unit)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False,
                              float_format=lambda v: f"{v:.4f}"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
