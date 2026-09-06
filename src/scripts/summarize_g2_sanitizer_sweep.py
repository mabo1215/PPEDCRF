"""Combine several direction-transfer exports (one per attacker-side
sanitizer) into a single G2 tier-1 adaptive-adversary table.

For each (attacker, sanitizer) export this reuses the paired exact-McNemar
analysis from analyze_direction_transfer.py, then adds a "% of white-box
benefit recovered" column: how much of the isotropic-to-white_box Top-1 drop
the best transfer condition still achieves once the attacker sanitizes the
frame before embedding it. A value near the un-sanitized baseline means the
directional advantage survives; a value near zero means sanitization
neutralizes it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_direction_transfer import analyse


def best_transfer_condition(table: pd.DataFrame) -> str:
    transfer_rows = [c for c in table["condition"] if str(c).startswith("transfer_")]
    if not transfer_rows:
        raise SystemExit("no transfer_* condition found in export")
    return max(transfer_rows, key=lambda c: int(c.split("_")[1]))


def summarize_one(label: str, path: Path) -> dict:
    table = analyse(path)
    row = {r["condition"]: r for r in table.to_dict(orient="records")}
    iso = row["isotropic"]["top1"]
    wb = row["white_box"]["top1"]
    best = best_transfer_condition(table)
    tr = row[best]["top1"]
    denom = iso - wb
    recovered = float("nan") if abs(denom) < 1e-12 else 100.0 * (iso - tr) / denom
    return {
        "label": label,
        "n": row["isotropic"]["n"],
        "isotropic_top1": iso,
        "white_box_top1": wb,
        "best_transfer_condition": best,
        "best_transfer_top1": tr,
        "best_transfer_delta": row[best]["delta"],
        "best_transfer_p": row[best]["exact_p"],
        "pct_whitebox_benefit_recovered": recovered,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    metavar="LABEL=PATH",
                    help="repeatable; e.g. --input resnet18/none=path.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for item in args.input:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"malformed --input {item!r}, expected LABEL=PATH")
        rows.append(summarize_one(label, Path(path)))

    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
