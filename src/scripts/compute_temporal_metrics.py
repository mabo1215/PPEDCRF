"""Compute temporal consistency metrics (flicker score, perturbation stability) for Table 6.

This script processes sequences from the synthetic monitoring dataset through each
PPEDCRF variant using a randomly-initialized SensitiveRegionNet with a fixed seed.
All metrics are deterministic given the fixed seed.

Usage:
    python src/scripts/compute_temporal_metrics.py [--monitoring_root PATH] [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)

from datasets.monitoring_clip_dataset import MonitoringClipDataset, iter_clip_ids
from eval.metrics import flicker_score, perturbation_stability
from models.dynamic_crf import DynamicCRF, DynamicCRFConfig
from privacy.NCP import NCPAllocator, NCPConfig
from privacy.noise_injector import NoiseConfig, NoiseInjector
from utils.config import load_yaml


VARIANT_LABELS = {
    "ppedcrf": "PPEDCRF",
    "no_temporal": "w/o temporal",
    "no_ncp": "w/o NCP",
    "random_mask": "Random mask",
    "full_frame": "Global Gaussian noise",
}

# These mask IoU values come from the ablation_summary.csv (already computed)
MASK_IOU = {
    "ppedcrf":    0.998,
    "no_temporal": 1.000,
    "no_ncp":     0.998,
    "random_mask": 0.334,
    "full_frame":  1.000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute temporal metrics for Table 6.")
    parser.add_argument(
        "--monitoring_root",
        type=str,
        default=r"c:\source\PPEDCRF\synthetic_monitoring\images",
    )
    parser.add_argument("--config", type=str, default="src/config/config.yaml")
    parser.add_argument("--clip_len", type=int, default=6,
                        help="Number of frames per clip for temporal evaluation.")
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--num_clips", type=int, default=20,
                        help="Number of query clips to evaluate temporal metrics on.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--output", type=str, default="src/outputs/temporal_metrics.json")
    return parser.parse_args()


def build_sensnet_fixed(seed: int, device: torch.device):
    """Build a SensitiveRegionNet with fixed seed for reproducible masks."""
    from run_train import SensitiveRegionNet
    torch.manual_seed(seed)
    model = SensitiveRegionNet()
    model.eval()
    model.to(device)
    return model


def make_random_mask_like(prob: torch.Tensor, seed: int, t_index: int) -> torch.Tensor:
    coverage = float(prob.mean().item())
    coverage = max(0.05, min(0.95, coverage))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 9973 * (t_index + 1))
    rand = torch.rand(prob.shape, generator=generator, device="cpu").to(prob.device)
    return (rand < coverage).float()


@torch.no_grad()
def process_clip_variant(
    frames: torch.Tensor,
    sensnet,
    cfg: dict,
    device: torch.device,
    variant: str,
    seed: int,
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process a clip through a given PPEDCRF variant.

    Returns:
        protected_frames: (T, 3, H, W) sanitized sequence
        orig_frames: (T, 3, H, W) original sequence (on CPU)
    """
    dcfg = cfg["ppedcrf"]["dynamic_crf"]
    ncfg = cfg["ppedcrf"]["ncp"]

    temporal_weight = float(dcfg["temporal_weight"]) if variant != "no_temporal" else 0.0
    crf = DynamicCRF(DynamicCRFConfig(
        n_iters=int(dcfg["n_iters"]),
        spatial_weight=float(dcfg["spatial_weight"]),
        temporal_weight=temporal_weight,
        smooth_kernel=int(dcfg["smooth_kernel"]),
    ))
    ncp = NCPAllocator(
        NCPConfig(alpha=float(ncfg.get("alpha", 1.0))),
        class_sensitivity=ncfg.get("class_sensitivity", {}),
    )
    injector = NoiseInjector(NoiseConfig(
        mode="wiener",
        sigma=sigma,
        clamp_min=0.0,
        clamp_max=255.0,
        seed=seed,
    ))

    prev_prob = None
    protected_frames = []

    T = frames.size(0)
    for t in range(T):
        frame = frames[t:t+1].to(device)
        unary = sensnet(frame)

        if variant == "no_temporal":
            refined_prob, _ = crf.refine(unary, prev_prob=None, flow=None)
            prev_prob = None
        else:
            refined_prob, prev_prob = crf.refine(unary, prev_prob=prev_prob, flow=None)

        if variant == "random_mask":
            mask = make_random_mask_like(refined_prob, seed=seed, t_index=t)
            strength = torch.ones_like(refined_prob)
        elif variant == "full_frame":
            mask = torch.ones_like(refined_prob)
            strength = torch.ones_like(refined_prob)
        elif variant == "no_ncp":
            mask = refined_prob
            strength = torch.ones_like(refined_prob)
        else:
            # ppedcrf or no_temporal with NCP
            mask = refined_prob
            strength = ncp.allocate(refined_prob)

        protected = injector.apply(frame, mask, strength, foreground_mask=None, t_index=t)
        protected_frames.append(protected.squeeze(0).cpu())

    return torch.stack(protected_frames, dim=0), frames.cpu()


def compute_metrics_for_variant(
    all_clips: List[torch.Tensor],
    sensnet,
    cfg: dict,
    device: torch.device,
    variant: str,
    seed: int,
    sigma: float,
) -> Dict[str, float]:
    flicker_scores = []
    stability_scores = []

    for frames in all_clips:
        protected, orig = process_clip_variant(
            frames=frames,
            sensnet=sensnet,
            cfg=cfg,
            device=device,
            variant=variant,
            seed=seed,
            sigma=sigma,
        )
        fs = flicker_score(protected)
        ps = perturbation_stability(orig, protected)
        flicker_scores.append(fs)
        stability_scores.append(ps)

    return {
        "flicker_mean": float(np.mean(flicker_scores)),
        "flicker_std": float(np.std(flicker_scores, ddof=1)) if len(flicker_scores) > 1 else 0.0,
        "stability_mean": float(np.mean(stability_scores)),
        "stability_std": float(np.std(stability_scores, ddof=1)) if len(stability_scores) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device("cpu")

    resize_hw = (args.resize_h, args.resize_w)

    # Load query clip sequences
    print(f"Loading monitoring sequences from: {args.monitoring_root}")
    clip_ids = list(iter_clip_ids(root=args.monitoring_root, min_frames=args.clip_len))
    clip_ids = clip_ids[:args.num_clips]
    print(f"Using {len(clip_ids)} clips for temporal evaluation.")

    dataset = MonitoringClipDataset(
        root=args.monitoring_root,
        clip_ids=clip_ids,
        view="full",
        clip_len=args.clip_len,
        resize_hw=resize_hw,
        min_frames=args.clip_len,
    )

    all_clips: List[torch.Tensor] = []
    for sample in dataset:
        all_clips.append(sample.frames.float())

    print(f"Loaded {len(all_clips)} clips, each with shape {all_clips[0].shape}")

    # Build the model with fixed seed
    print(f"Initializing SensitiveRegionNet with seed={args.seed}")
    sensnet = build_sensnet_fixed(seed=args.seed, device=device)

    # Compute metrics per variant
    results = {}
    for variant in VARIANT_LABELS:
        print(f"  Processing variant: {variant}")
        metrics = compute_metrics_for_variant(
            all_clips=all_clips,
            sensnet=sensnet,
            cfg=cfg,
            device=device,
            variant=variant,
            seed=args.seed,
            sigma=args.sigma,
        )
        metrics["mask_iou"] = MASK_IOU[variant]
        results[variant] = metrics

    # Print summary table
    print("\n=== Temporal Metrics Summary ===")
    print(f"{'Method':<28} {'Flicker':>10} {'Pert.Stab.':>12} {'Mask IoU':>10}")
    print("-" * 62)
    for variant, label in VARIANT_LABELS.items():
        m = results[variant]
        print(
            f"{label:<28} "
            f"{m['flicker_mean']:>8.3f}±{m['flicker_std']:.3f}  "
            f"{m['stability_mean']:>9.3f}±{m['stability_std']:.3f}  "
            f"{m['mask_iou']:>8.3f}"
        )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Print LaTeX-formatted table rows
    print("\n=== LaTeX table rows for Table 6 (tab:temporal) ===")
    for variant, label in VARIANT_LABELS.items():
        m = results[variant]
        label_str = VARIANT_LABELS[variant]
        print(
            f"{label_str} & "
            f"${m['flicker_mean']:.3f}\\pm{m['flicker_std']:.3f}$ & "
            f"${m['stability_mean']:.3f}\\pm{m['stability_std']:.3f}$ & "
            f"{m['mask_iou']:.3f} \\\\"
        )


if __name__ == "__main__":
    main()
