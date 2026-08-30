"""Audit the unary sensitivity predictor and checkpoint provenance.

The audit is intentionally conservative. It reports the architecture and
checkpoint metadata and measures whether the predictor emits non-constant maps
on held-out monitoring frames. It does not call a map non-constantness test an
accuracy evaluation because the monitoring pool has no pixel-level sensitivity
ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from datasets.monitoring_clip_dataset import MonitoringClipDataset  # noqa: E402
from main import load_sensnet_checkpoint  # noqa: E402
from run_train import SensitiveRegionNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PPEDCRF unary predictor provenance.")
    parser.add_argument("--checkpoint", default="src/outputs/sensnet_final.pt")
    parser.add_argument("--monitoring_root", default=r"F:\work\datasets\monitoring\images")
    parser.add_argument("--output_dir", default="src/outputs/tomm_review_provenance")
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--max_clips", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_sensnet_checkpoint(str(checkpoint_path), device)
    model.eval()

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    train_cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, dict) else {}
    records = []
    dataset = MonitoringClipDataset(
        root=args.monitoring_root,
        view="query",
        clip_len=1,
        resize_hw=(int(args.resize_h), int(args.resize_w)),
        min_frames=6,
    )
    with torch.no_grad():
        for index in range(min(int(args.max_clips), len(dataset))):
            sample = dataset[index]
            logits = model(sample.frames.to(device)).squeeze(1)
            probs = torch.sigmoid(logits)
            records.append(
                {
                    "clip_id": sample.clip_id,
                    "logit_mean": float(logits.mean().item()),
                    "logit_std": float(logits.std().item()),
                    "prob_mean": float(probs.mean().item()),
                    "prob_std": float(probs.std().item()),
                    "prob_min": float(probs.min().item()),
                    "prob_max": float(probs.max().item()),
                }
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variation = float(np.mean([row["prob_std"] for row in records])) if records else 0.0
    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "architecture": {
            "class": "SensitiveRegionNet",
            "parameter_count": parameter_count,
            "encoder": [
                "Conv2d(3,32,3,pad=1)+ReLU",
                "Conv2d(32,32,3,pad=1)+ReLU",
                "MaxPool2d(2)",
                "Conv2d(32,64,3,pad=1)+ReLU",
                "Conv2d(64,64,3,pad=1)+ReLU",
                "MaxPool2d(2)",
            ],
            "decoder": [
                "ConvTranspose2d(64,32,2,stride=2)+ReLU",
                "Conv2d(32,32,3,pad=1)+ReLU",
                "ConvTranspose2d(32,16,2,stride=2)+ReLU",
                "Conv2d(16,16,3,pad=1)+ReLU",
                "Conv2d(16,1,1)",
            ],
        },
        "training_metadata": train_cfg,
        "held_out_monitoring_probe": records,
        "mean_probability_spatial_std": variation,
        "nonconstant_probe_passed": bool(variation > 1e-3),
        "ground_truth_map_accuracy_available": False,
        "scientific_evidence": False,
        "interpretation": (
            "The probe only tests whether outputs vary spatially on monitoring frames; "
            "it is not an independent accuracy evaluation without sensitivity-map labels."
        ),
    }
    (output_dir / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"device": str(device), "parameter_count": parameter_count, "mean_probability_spatial_std": variation}, indent=2))


if __name__ == "__main__":
    main()
