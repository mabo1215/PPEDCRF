"""Check that two repeated runs of the same benchmark command are numerically
identical given fixed seeds.

Motivation: the current proxy50 write-up (Appendix, "MixVPR Margin Diagnosis
and Mitigation Status") reports that the present run no longer reproduces an
earlier adverse-transfer result for MixVPR, and states plainly that no
preserved copy of the earlier run exists to identify a root cause. That is
an honest thing to say about a run that already happened, but it leaves open
a more basic question the paper does not answer: is the *current* pipeline
even deterministic given fixed seeds, or could re-running it today produce
yet another different number?

This script answers that question directly: run the same benchmark command
twice (same seeds, same checkpoint, same code revision) into two output
directories, then diff their `per_query.csv` or `summary.csv` exports. If
every numeric column matches within tolerance, the current numbers are
reproducible and the earlier MixVPR discrepancy is attributable to a prior
code/checkpoint change rather than run-to-run nondeterminism. If it does not
match, that is itself an important, previously-undocumented finding.

Pure post-processing; no GPU is required to run this script itself (the two
input runs it compares must have already been produced separately).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLUMN_CANDIDATES = ("query_id", "variant", "backbone", "gallery_size", "seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diff two repeated-run CSV exports for numerical determinism."
    )
    parser.add_argument("run_a", help="Path to the first run's CSV (per_query.csv or summary.csv).")
    parser.add_argument("run_b", help="Path to the second run's CSV, same schema as run_a.")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_a = pd.read_csv(args.run_a)
    df_b = pd.read_csv(args.run_b)

    if list(df_a.columns) != list(df_b.columns):
        print(f"[determinism] COLUMN MISMATCH\n  a: {list(df_a.columns)}\n  b: {list(df_b.columns)}",
              file=sys.stderr)
        sys.exit(1)
    if len(df_a) != len(df_b):
        print(f"[determinism] ROW COUNT MISMATCH: {len(df_a)} (a) vs {len(df_b)} (b)", file=sys.stderr)
        sys.exit(1)

    key_columns = [c for c in KEY_COLUMN_CANDIDATES if c in df_a.columns]
    if key_columns:
        df_a = df_a.sort_values(key_columns).reset_index(drop=True)
        df_b = df_b.sort_values(key_columns).reset_index(drop=True)
    else:
        df_a = df_a.reset_index(drop=True)
        df_b = df_b.reset_index(drop=True)

    mismatches = []
    for column in df_a.columns:
        col_a, col_b = df_a[column], df_b[column]
        if pd.api.types.is_numeric_dtype(col_a) and pd.api.types.is_numeric_dtype(col_b):
            close = np.isclose(
                col_a.to_numpy(dtype=float), col_b.to_numpy(dtype=float),
                rtol=args.rtol, atol=args.atol, equal_nan=True,
            )
            if not bool(close.all()):
                mismatches.append((column, int((~close).sum())))
        else:
            differ = col_a.astype(str) != col_b.astype(str)
            if bool(differ.any()):
                mismatches.append((column, int(differ.sum())))

    if mismatches:
        print(f"[determinism] MISMATCH across {len(mismatches)} column(s) "
              f"comparing {args.run_a} vs {args.run_b}:")
        for column, n_rows in mismatches:
            print(f"  {column}: {n_rows} differing row(s)")
        sys.exit(1)

    print(f"[determinism] OK: {len(df_a)} rows x {len(df_a.columns)} columns identical "
          f"within rtol={args.rtol}, atol={args.atol} between {args.run_a} and {args.run_b}.")


if __name__ == "__main__":
    main()
