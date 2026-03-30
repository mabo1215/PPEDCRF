from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# Ensure src directory is importable when running from repo root.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from main import (
    load_sensnet_checkpoint,
    protect_clip,
    compute_refined_sensitivity,
    apply_mask_guided_blur,
    overlay_detection_and_sensitive_region_boxes,
)
from utils.config import load_yaml


def load_image_to_tensor(path: str, resize_hw: Optional[tuple[int, int]] = None) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if resize_hw is not None:
            img = img.resize((resize_hw[1], resize_hw[0]), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8).copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor


def save_tensor_as_image(tensor: torch.Tensor, path: str) -> None:
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(0, 255).to(torch.uint8)
    arr = tensor.permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Protect a single frame with PPEDCRF and save the sanitized output.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input RGB image.")
    parser.add_argument("--checkpoint", type=str, default="src/outputs/sensnet_final.pt", help="Path to the PPEDCRF sensitivity network checkpoint.")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--output-path", type=str, default=None, help="Output path for the protected image or visualization.")
    parser.add_argument("--vis-mode", type=str, default="noise", choices=["noise", "blur", "overlay"], help="Visualization mode: standard noise, blur on sensitive regions, or overlay sensitive region boxes.")
    parser.add_argument("--mask-threshold", type=float, default=0.85, help="Detection confidence threshold for drawing segmentation/detection boxes in overlay mode.")
    parser.add_argument("--blur-kernel", type=int, default=15, help="Kernel size for mask-guided blur.")
    parser.add_argument("--resize-h", type=int, default=384, help="Resize height for processing.")
    parser.add_argument("--resize-w", type=int, default=640, help="Resize width for processing.")
    parser.add_argument("--noise-sigma", type=float, default=None, help="Override noise sigma for the protection step.")
    parser.add_argument("--noise-mode", type=str, default=None, choices=["gaussian", "wiener"], help="Override noise mode for the protection step.")
    parser.add_argument("--device", type=str, default=None, help="Torch device to use (cpu or cuda).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.device is not None:
        cfg["project"]["device"] = args.device
    if args.noise_sigma is not None:
        cfg.setdefault("ppedcrf", {}).setdefault("noise", {})["sigma"] = args.noise_sigma
    if args.noise_mode is not None:
        cfg.setdefault("ppedcrf", {}).setdefault("noise", {})["mode"] = args.noise_mode

    device_str = str(cfg["project"].get("device", "cuda"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[protect_single_frame] device={device}")

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    frame = load_image_to_tensor(args.image_path, resize_hw=(args.resize_h, args.resize_w))
    frame = frame.unsqueeze(0)  # (1,3,H,W)
    frames = frame

    sensnet = load_sensnet_checkpoint(args.checkpoint, device)
    sensnet.eval()

    if args.output_path is None:
        args.output_path = f"paper/figs/mot_org_ppedcrf_{args.vis_mode}.png"

    if args.vis_mode == "noise":
        protected = protect_clip(frames, sensnet, cfg, device)
        if protected.size(0) == 0:
            raise RuntimeError("Protected output is empty")
        output_tensor = protected[0]

    elif args.vis_mode == "blur":
        refined_probs, _ = compute_refined_sensitivity(frames, sensnet, cfg, device)
        output_tensor = apply_mask_guided_blur(frames[0], refined_probs[0], kernel_size=args.blur_kernel)[0]

    elif args.vis_mode == "overlay":
        refined_probs, _ = compute_refined_sensitivity(frames, sensnet, cfg, device)
        output_tensor = overlay_detection_and_sensitive_region_boxes(
            frames[0],
            refined_probs[0],
            device=device,
            det_threshold=args.mask_threshold,
            mask_threshold=0.9,
            line_width=3,
            alpha=0.0,
            min_area=60,
        )[0]

    else:
        raise ValueError(f"Unsupported vis mode: {args.vis_mode}")

    out_dir = Path(args.output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tensor_as_image(output_tensor, args.output_path)
    print(f"Saved PPEDCRF visualization to {args.output_path}")


if __name__ == "__main__":
    main()
