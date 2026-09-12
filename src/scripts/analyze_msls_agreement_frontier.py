"""Utility on the frames the attack actually sees.

The manuscript's frontier decides admissibility from segmentation measured on
VOC while retrieval is measured on MSLS, so no image contributes to both and
the verdict that decides the paper's practical conclusion is cross-corpus. The
released joint export already carries the segmenter's agreement with its own
clean prediction on the 400 MSLS query frames, at every budget and condition
the frontier reports. This turns that into the same drop-and-admissibility
reading, so the two corpora can be compared cell by cell.

Agreement is not accuracy: MSLS query frames carry no masks, so the reference
is the frozen segmenter's prediction on the clean frame and a drop of d means
the release moved d of the agreement the clean frame would have had with
itself. That makes the column comparable in units and in ordering with the VOC
column, not identical in meaning, and the manuscript says so.
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = ["isotropic", "direction", "hardened_direction"]
BUDGETS = [5.0, 15.68, 60.0, 241.5]


def per_image_miou(raw) -> float:
    """Mean over the classes present in this image of intersection / union."""
    iu = raw if isinstance(raw, dict) else ast.literal_eval(raw)
    vals = [i / u for i, u in (tuple(v) for v in iu.values()) if u > 0]
    return float(np.mean(vals)) if vals else float("nan")


def load(export_root: Path) -> Dict[tuple, Dict[str, List[float]]]:
    acc: Dict[tuple, Dict[str, List[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for path in sorted(glob.glob(str(export_root / "*" / "per_image.jsonl"))):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            key = (float(d["target_mse"]), d["condition"])
            acc[key][d["image_id"]].append(per_image_miou(d["seg_iu"]))
    return acc


def paired_bootstrap(d: np.ndarray, n: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    k = d.size
    draws = np.array([d[rng.integers(0, k, k)].mean() for _ in range(n)])
    return tuple(float(x) for x in np.percentile(draws, [2.5, 97.5]))


def verdict(lo: float, hi: float, tol: float) -> str:
    if hi <= tol:
        return "admissible"
    if lo >= tol:
        return "beyond"
    return "straddles"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-root", dest="root",
                    default=str(REPO / "src/exports/tifs6_joint"))
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tolerances", type=float, nargs="+", default=[0.05, 0.016])
    ap.add_argument("--out", default=str(
        REPO / "src/exports/msls_agreement_frontier/summary.json"))
    args = ap.parse_args()

    acc = load(Path(args.root))
    out = {"n_frames": None, "tolerances": args.tolerances, "cells": []}

    for budget in BUDGETS:
        clean = acc.get((budget, "clean"), {})
        for cond in CONDITIONS:
            cur = acc.get((budget, cond), {})
            ids = sorted(set(cur) & set(clean))
            if not ids:
                continue
            # seeds averaged within an image, exactly as the retrieval side
            # averages seeds within a query before pairing
            agree = np.array([float(np.mean(cur[i])) for i in ids])
            base = np.array([float(np.mean(clean[i])) for i in ids])
            drop = base - agree
            lo, hi = paired_bootstrap(drop, args.bootstrap, args.seed)
            out["n_frames"] = len(ids)
            out["cells"].append({
                "target_mse": budget,
                "condition": cond,
                "agreement": float(agree.mean()),
                "drop": float(drop.mean()),
                "ci": [lo, hi],
                "verdicts": {str(t): verdict(lo, hi, t) for t in args.tolerances},
            })

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {dest}  ({out['n_frames']} MSLS query frames)")
    print(f"{'MSE':>7s} {'condition':20s} {'agree':>7s} {'drop':>8s} "
          f"{'95% CI':>20s} {'0.05':>11s} {'0.016':>11s}")
    for c in out["cells"]:
        print(f"{c['target_mse']:7.2f} {c['condition']:20s} {c['agreement']:7.4f} "
              f"{c['drop']:+8.4f} [{c['ci'][0]:+7.4f},{c['ci'][1]:+7.4f}] "
              f"{c['verdicts']['0.05']:>11s} {c['verdicts']['0.016']:>11s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
