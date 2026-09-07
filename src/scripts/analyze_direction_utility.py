"""Paired analysis of the direction-axis utility export.

The aggregate mAP and mIoU written by evaluate_direction_utility.py are global
metrics, so a difference between conditions carries no error bar on its own.
This script adds the paired view: every image is measured under every
condition, so isotropic and direction can be compared image by image with a
Wilcoxon signed-rank test rather than by eyeballing two aggregates.

Per-image AP is the class-averaged AP@50 of that image alone, and per-image
IoU is the mean over the classes present in its target mask -- both computed
from the cached rows, so no model is re-run. Values are averaged over the
repeated runs first, since the direction perturbation is recomputed each run
and is not bit-identical across them (CUDA backward is nondeterministic), and
that run-to-run spread is reported alongside.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.evaluate_same_image_utility import (  # noqa: E402
    average_precision_detections,
    load_manifest as load_utility_manifest,
    load_target,
)


def per_image_scores(rows, targets_by_id):
    """Mean per-image score for each (image, condition), averaged over runs."""
    acc = defaultdict(list)
    for row in rows:
        iid, cond = row["image_id"], row["condition"]
        if "det" in row:
            target = targets_by_id[iid]
            score = average_precision_detections([row["det"]], [target])
        elif "seg_iu" in row:
            ious = [i / u for i, u in row["seg_iu"].values() if u]
            score = st.mean(ious) if ious else 0.0
        else:
            continue
        acc[(iid, cond)].append(score)
    return {k: st.mean(v) for k, v in acc.items()}


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank p, via scipy when available."""
    nz = [d for d in diffs if d != 0.0]
    if not nz:
        return 1.0, 0
    try:
        from scipy.stats import wilcoxon as _w
        return float(_w(nz).pvalue), len(nz)
    except Exception:
        return float("nan"), len(nz)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="per_image.jsonl from evaluate_direction_utility.py")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default="")
    ap.add_argument("--resize_h", type=int, default=192)
    ap.add_argument("--resize_w", type=int, default=320)
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            open(args.cache, encoding="utf-8") if line.strip()]
    records = load_utility_manifest(args.manifest, args.root)
    resize_hw = (args.resize_h, args.resize_w)
    targets_by_id = {r["image_id"]: load_target(r, args.root, resize_hw)[0]
                     for r in records}
    metric = "AP@50" if "det" in rows[0] else "IoU"

    scores = per_image_scores(rows, targets_by_id)
    ids = [r["image_id"] for r in records]
    conditions = ["clean", "isotropic", "direction"]
    print(f"per-image {metric}, {len(ids)} images, "
          f"{len({r['seed'] for r in rows})} runs")
    for cond in conditions:
        vals = [scores[(i, cond)] for i in ids if (i, cond) in scores]
        print(f"  {cond:10s} mean={st.mean(vals):.4f}")

    print("paired comparisons (Wilcoxon signed-rank, two-sided):")
    for a, b in (("isotropic", "clean"), ("direction", "clean"),
                 ("direction", "isotropic")):
        pairs = [(scores[(i, a)], scores[(i, b)]) for i in ids
                 if (i, a) in scores and (i, b) in scores]
        diffs = [x - y for x, y in pairs]
        p, n_nz = wilcoxon(diffs)
        worse = sum(1 for d in diffs if d < 0)
        better = sum(1 for d in diffs if d > 0)
        print(f"  {a:10s} vs {b:10s} mean_delta={st.mean(diffs):+.4f} "
              f"n={len(diffs)} (worse {worse} / better {better} / "
              f"tied {len(diffs) - n_nz}) p={p:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
