"""Train a mask-backed SensitiveRegionNet on KITTI-360 structural support masks.

The target is a semantic structural-support mask, not ground-truth location
sensitivity.  It provides a non-constant, independently sourced checkpoint for
the weak VPR attribution-consistency diagnostic.  Training and evaluation
sequences must be disjoint when the resulting checkpoint is used for E5.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from datasets.driving_clip_dataset import _read_image, _resize_if_needed  # noqa: E402
from run_train import SensitiveRegionNet  # noqa: E402


# Persistent scene classes in the official KITTI-360 semantic label space.
# These are used only as a structural-support training target.
STRUCTURAL_SUPPORT_IDS = (
    7,   # road
    8,   # sidewalk
    9,   # parking
    10,  # rail track
    11,  # building
    12,  # wall
    13,  # fence
    14,  # guard rail
    15,  # bridge
    16,  # tunnel
    17,  # pole
    18,  # pole group
    19,  # traffic light
    20,  # traffic sign
    21,  # vegetation
    22,  # terrain
    23,  # sky
    34,  # garage
    35,  # gate
    36,  # stop
    37,  # small pole
    38,  # lamp
    39,  # trash bin
    40,  # vending machine
    41,  # box
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SensitiveRegionNet on KITTI-360 semantic structural-support masks."
    )
    parser.add_argument("--kitti360_root", required=True)
    parser.add_argument("--sequence", default="0000", help="Training sequence, e.g. 0000.")
    parser.add_argument("--camera", default="image_00", choices=("image_00", "image_01"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=1200)
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sequence_name(sequence: str) -> str:
    return f"2013_05_28_drive_{int(sequence):04d}_sync"


def frame_pairs(root: Path, sequence: str, camera: str, stride: int, max_frames: int) -> list[tuple[Path, Path]]:
    seq_name = sequence_name(sequence)
    image_dir = root / "data_2d_raw" / seq_name / camera / "data_rect"
    semantic_dir = root / "data_2d_semantics" / "train" / seq_name / camera / "semantic"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing KITTI-360 image directory: {image_dir}")
    if not semantic_dir.is_dir():
        raise FileNotFoundError(f"Missing KITTI-360 semantic directory: {semantic_dir}")
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.glob("*.png"))[:: max(1, int(stride))]:
        semantic_path = semantic_dir / f"{int(image_path.stem):010d}.png"
        if semantic_path.is_file():
            pairs.append((image_path, semantic_path))
            if len(pairs) >= int(max_frames):
                break
    if not pairs:
        raise ValueError(f"No image/semantic pairs found for sequence {sequence}.")
    return pairs


class SemanticSupportDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], resize_hw: tuple[int, int]):
        self.pairs = pairs
        self.resize_hw = resize_hw

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, semantic_path = self.pairs[index]
        image = _resize_if_needed(_read_image(str(image_path)), self.resize_hw)
        with Image.open(semantic_path) as semantic_image:
            semantic = np.asarray(semantic_image, dtype=np.int64)
        support = np.isin(semantic, np.asarray(STRUCTURAL_SUPPORT_IDS, dtype=np.int64)).astype(np.float32)
        target = torch.from_numpy(support).unsqueeze(0)
        target = F.interpolate(target.unsqueeze(0), size=self.resize_hw, mode="nearest").squeeze(0)
        return image, target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    root = Path(args.kitti360_root).expanduser().resolve()
    resize_hw = (int(args.resize_h), int(args.resize_w))
    pairs = frame_pairs(root, args.sequence, args.camera, int(args.stride), int(args.max_frames))
    dataset = SemanticSupportDataset(pairs, resize_hw)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = SensitiveRegionNet().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    with torch.no_grad():
        sample_targets = torch.stack([dataset[index][1] for index in range(min(len(dataset), 64))])
        positive_fraction = float(sample_targets.mean().item())
    positive_weight = max(0.25, min(4.0, 1.0 - positive_fraction))
    negative_weight = max(0.25, min(4.0, positive_fraction))

    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        losses: list[float] = []
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            weights = torch.where(targets > 0.5, positive_weight, negative_weight)
            loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": float(epoch + 1), "loss": epoch_loss})
        print(f"[train] epoch={epoch + 1}/{int(args.epochs)} loss={epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        probe_images = torch.stack([dataset[index][0] for index in range(min(len(dataset), 16))]).to(device)
        probe_prob = torch.sigmoid(model(probe_images))
        probe_spatial_std = float(probe_prob.flatten(1).std(dim=1).mean().item())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "data_root": str(root),
        "camera": args.camera,
        "train_sequence": f"{int(args.sequence):04d}",
        "stride": int(args.stride),
        "max_frames": int(args.max_frames),
        "resize_hw": resize_hw,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "mask_root": str(root / "data_2d_semantics" / "train"),
        "mask_target": "KITTI-360 structural scene support; not sensitivity ground truth",
        "structural_support_ids": list(STRUCTURAL_SUPPORT_IDS),
        "seed": int(args.seed),
        "device": str(device),
    }
    torch.save({"model": model.state_dict(), "cfg": cfg}, output_path)
    summary = {
        "checkpoint": str(output_path.resolve()),
        "training_sequence": f"{int(args.sequence):04d}",
        "camera": args.camera,
        "pair_count": len(pairs),
        "positive_fraction_sample": positive_fraction,
        "probe_spatial_std": probe_spatial_std,
        "history": history,
        "scientific_evidence": False,
        "interpretation": (
            "This checkpoint is mask-backed by an independently sourced KITTI-360 "
            "structural-support target. It is not a ground-truth sensitivity model; "
            "E5 still requires the held-out VPR attribution-consistency gate."
        ),
    }
    output_path.with_suffix(".training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(output_path), "pair_count": len(pairs), "probe_spatial_std": probe_spatial_std}, indent=2))


if __name__ == "__main__":
    main()
