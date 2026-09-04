"""Measure how spatially concentrated each placement rule actually is.

The placement study answers "does it change retrieval accuracy". This script
answers the prior question the mechanistic account needs: how much spatial
selectivity does each rule even express? A rule that allocates energy almost
uniformly cannot be expected to behave differently from the uniform control,
and on street imagery some rules that sound highly selective turn out to be
nearly uniform in practice.

Two statistics per rule, both computed on the energy-matched weights actually
used for perturbation:

  top-decile energy share -- the fraction of total squared weight carried by
    the 10% of pixels with the largest weights. Uniform placement gives 0.10
    by construction; larger values mean more concentration.
  coefficient of variation -- std/mean of the weight map, a scale-free
    dispersion measure comparable across rules.

CPU-only and retrieval-free: it builds weight maps and measures them, so it
can run while a GPU benchmark occupies the card.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import statistics as st

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datasets.driving_clip_dataset import _read_image, _resize_if_needed  # noqa: E402
from scripts.run_placement_rule_study import (  # noqa: E402
    center_map,
    edge_map,
    random_fixed_map,
    renormalise_to_energy,
    saliency_map,
    segmentation_map,
    segmentation_map_ade,
    segmentation_map_fcn,
)

# Rules that depend only on the frame, so they can be measured without a
# retrieval context or an attacker gradient.
FRAME_RULES = {
    "uniform": None,
    "saliency": lambda f, k: saliency_map(f),
    "edge": lambda f, k: edge_map(f),
    "center": lambda f, k: center_map(f),
    "random_fixed": lambda f, k: random_fixed_map(f, 1234),
    "segmentation": segmentation_map,
    "segmentation_fcn": segmentation_map_fcn,
    "segmentation_ade": segmentation_map_ade,
}


def concentration(weights: torch.Tensor) -> tuple[float, float]:
    """Top-decile energy share and coefficient of variation of a weight map."""
    flat = weights.flatten().double()
    energy = flat.square()
    total = float(energy.sum())
    k = max(1, int(round(0.10 * flat.numel())))
    top = float(energy.topk(k).values.sum())
    share = top / total if total > 0 else float("nan")
    mean = float(flat.mean())
    cv = float(flat.std(unbiased=False) / mean) if mean > 0 else float("nan")
    return share, cv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--rules",
        nargs="+",
        default=list(FRAME_RULES),
        help="subset of frame-only placement rules to measure",
    )
    args = ap.parse_args()

    root = pathlib.Path(args.root).expanduser()
    frames: list[str] = []
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            if len(frames) >= args.limit:
                break
            rec = json.loads(line)
            frames.append(rec["query_path"])
    print(f"[concentration] {len(frames)} query frames", flush=True)

    stats: dict[str, dict[str, list[float]]] = {
        r: {"share": [], "cv": []} for r in args.rules
    }
    for i, rel in enumerate(frames, 1):
        path = root / rel
        if not path.is_file():
            continue
        frame = _resize_if_needed(_read_image(str(path)), (args.height, args.width))
        frame = frame.unsqueeze(0)
        target = torch.tensor(float(frame[0, 0].numel()), dtype=torch.float32)
        for rule in args.rules:
            fn = FRAME_RULES[rule]
            raw = torch.ones_like(frame[:, :1]) if fn is None else fn(frame, f"c{i}")
            w = renormalise_to_energy(raw, target)
            share, cv = concentration(w)
            stats[rule]["share"].append(share)
            stats[rule]["cv"].append(cv)
        if i % 50 == 0:
            print(f"  {i}/{len(frames)}", flush=True)

    rows = []
    for rule in args.rules:
        s, c = stats[rule]["share"], stats[rule]["cv"]
        if not s:
            continue
        rows.append(
            {
                "placement": rule,
                "n_frames": len(s),
                "top_decile_energy_share_mean": st.mean(s),
                "top_decile_energy_share_min": min(s),
                "top_decile_energy_share_max": max(s),
                "cv_mean": st.mean(c),
                "cv_min": min(c),
                "cv_max": max(c),
            }
        )
    rows.sort(key=lambda r: -r["top_decile_energy_share_mean"])

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print("\n placement            top-decile share (uniform=0.100)      CV")
    for r in rows:
        print(
            f"  {r['placement']:>18}: {r['top_decile_energy_share_mean']:.3f}"
            f" [{r['top_decile_energy_share_min']:.3f},"
            f" {r['top_decile_energy_share_max']:.3f}]"
            f"   {r['cv_mean']:.3f}"
            f" [{r['cv_min']:.3f}, {r['cv_max']:.3f}]"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
