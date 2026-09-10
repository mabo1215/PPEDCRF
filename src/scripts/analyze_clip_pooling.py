"""What a clip-pooling attacker recovers from the directional release.

The single-frame numbers this paper leads with assume the attacker embeds one
released frame. The scenario it is motivated by uploads video, and an attacker
holding the clip can pool several released frames before ranking. The
directional release is the arm at risk: it is re-derived per frame from that
frame's own clean embedding, so its displacement points a different way in each
frame while the place they depict is common to all of them.

This script reads the clip run and answers three questions in the order a
referee would ask them.

1. Does clip length 1 with first-frame pooling reproduce the single-frame
   study? If it does not, nothing else here can be trusted.
2. What does each pooling rule do to each condition as the clip grows?
3. Does the direction still separate from its isotropic control at the longest
   clip, on the unit of inference the protocol prescribes -- the place, not the
   query?

Intervals resample places and carry every query of a sampled place; the
query-level interval is printed beside them because the manuscript reports both
wherever the two can disagree.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon

CONTROL = "isotropic"


def place_map(d6_dir: str) -> Dict[str, str]:
    """query id -> place id, from the transfer exports that carry the label."""
    places: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(d6_dir, "*.csv"))):
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if "correct_place" not in (reader.fieldnames or []):
                continue
            for row in reader:
                places.setdefault(row["query_id"], row["correct_place"])
    return places


def load(pattern: str):
    """(condition, clip_len, pooling) -> query -> mean Top-1 over seeds."""
    cells = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(pattern))
    for path in files:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["condition"], int(r["clip_len"]), r["pooling"])
                cells[key][r["query_id"]].append(
                    float(int(r["correct_rank"]) == 1))
    return ({k: {q: float(np.mean(v)) for q, v in d.items()}
             for k, d in cells.items()}, len(files))


def boot(diff: np.ndarray, ids: np.ndarray, n: int, rng) -> tuple:
    uniq, inv = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    stats = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        stats[b] = diff[np.concatenate([groups[i] for i in pick])].mean()
    return tuple(np.percentile(stats, [2.5, 97.5]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="src/exports/clip_pooling/clip_shard*.csv")
    ap.add_argument("--d6_dir", default="src/exports/tifs_d6")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--single_frame", nargs=2, type=float,
                    default=[0.1967, 0.0317],
                    help="Published single-frame Top-1 for the isotropic "
                         "control and the three-surrogate direction.")
    args = ap.parse_args()

    cells, n_files = load(args.rows)
    if not cells:
        print(f"[clip] no rows matched {args.rows}")
        return 1
    places = place_map(args.d6_dir)
    conditions = sorted({k[0] for k in cells})
    lengths = sorted({k[1] for k in cells})
    poolings = sorted({k[2] for k in cells}, key=lambda p:
                      ["first", "mean", "max", "best_frame"].index(p)
                      if p in ("first", "mean", "max", "best_frame") else 9)
    # A query that cannot reach the longest clip contributes to the short
    # lengths only, so Top-1 would improve with k partly because the sample
    # changed. Every cell is restricted to the queries present at every length.
    common = set.intersection(*[set(cells[(c, k, p)])
                                for c in conditions for k in lengths
                                for p in poolings
                                if (c, k, p) in cells])
    cells = {key: {q: v for q, v in d.items() if q in common}
             for key, d in cells.items()}
    n_q = len(common)
    print(f"[clip] {n_files} shard files, {n_q} queries reaching every clip "
          f"length, conditions {conditions}, clip lengths {lengths}")

    print("\n== 1. Does clip length 1 reproduce the single-frame study? ==")
    for cond, published in zip((CONTROL, "direction"), args.single_frame):
        got = cells.get((cond, 1, "first"))
        if got is None:
            continue
        val = float(np.mean(list(got.values())))
        print(f"  {cond:10s} clip 1, first frame: {val:.4f} "
              f"against a published {published:.4f}  "
              f"(difference {val - published:+.4f})")

    print("\n== 2. Top-1 by clip length and pooling ==")
    header = "  " + " ".join(f"{('k=' + str(k)):>8s}" for k in lengths)
    for pooling in poolings:
        print(f"\n  pooling: {pooling}")
        print(f"  {'condition':14s}{header}")
        for cond in conditions:
            row = []
            for k in lengths:
                arm = cells.get((cond, k, pooling))
                row.append(f"{np.mean(list(arm.values())):8.4f}" if arm
                           else "     ---")
            print(f"  {cond:14s}  " + " ".join(row))

    print("\n== 3. Direction against its control, place-clustered ==")
    rng = np.random.default_rng(0)
    print(f"  {'condition':10s}{'pooling':11s}{'k':>3s}{'Top-1':>9s}"
          f"{'delta':>9s}{'query 95% CI':>21s}{'place 95% CI':>21s}{'p':>10s}")
    for cond in [c for c in conditions if c != CONTROL]:
        for pooling in poolings:
            for k in lengths:
                arm = cells.get((cond, k, pooling))
                ref = cells.get((CONTROL, k, pooling))
                if not arm or not ref:
                    continue
                qs = sorted(set(arm) & set(ref) & set(places))
                d = np.array([arm[q] - ref[q] for q in qs])
                pid = np.array([places[q] for q in qs])
                qlo, qhi = boot(d, np.array(qs), args.n_boot, rng)
                plo, phi = boot(d, pid, args.n_boot, rng)
                try:
                    pv = wilcoxon(d).pvalue
                except ValueError:
                    pv = 1.0
                print(f"  {cond:10s}{pooling:11s}{k:3d}"
                      f"{np.mean([arm[q] for q in qs]):9.4f}{d.mean():+9.4f}"
                      f"  [{qlo:+.4f},{qhi:+.4f}]  [{plo:+.4f},{phi:+.4f}]"
                      f"{pv:10.2g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
