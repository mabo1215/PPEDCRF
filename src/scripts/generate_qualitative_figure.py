"""Generate a journal-quality qualitative figure for the PPEDCRF paper.

Produces a 2x3 multi-panel figure with:
  (a) Original frame
  (b) DCRF sensitivity heatmap overlay
  (c) PPEDCRF noise-protected frame
  (d) Pixel-wise absolute difference map
  (e) Mask-guided blur result
  (f) Zoomed crop comparison of a privacy-relevant region
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from main import (
    load_sensnet_checkpoint,
    protect_clip,
    compute_refined_sensitivity,
    apply_mask_guided_blur,
)
from utils.config import load_yaml


def load_image(path: str, resize_hw: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if resize_hw is not None:
            img = img.resize((resize_hw[1], resize_hw[0]), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float()


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    return t.detach().cpu().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="src/outputs/sensnet_final.pt")
    parser.add_argument("--config", type=str, default="src/config/config.yaml")
    parser.add_argument("--output", type=str, default="paper/figs/qualitative_figure.pdf")
    parser.add_argument("--noise-sigma", type=float, default=24.0,
                        help="Use a stronger sigma for visual clarity in the figure")
    parser.add_argument("--resize-h", type=int, default=384)
    parser.add_argument("--resize-w", type=int, default=640)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("ppedcrf", {}).setdefault("noise", {})["sigma"] = args.noise_sigma
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = load_image(args.image_path, (args.resize_h, args.resize_w)).unsqueeze(0)
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)
    sensnet.eval()

    # Compute sensitivity and protection
    refined_probs, _ = compute_refined_sensitivity(frame, sensnet, cfg, device)
    protected = protect_clip(frame, sensnet, cfg, device)
    blurred = apply_mask_guided_blur(frame[0], refined_probs[0], kernel_size=21)

    orig_np = tensor_to_numpy(frame)
    prot_np = tensor_to_numpy(protected)
    blur_np = tensor_to_numpy(blurred)
    mask_np = refined_probs[0].squeeze().cpu().numpy()

    # Difference map
    diff = np.abs(orig_np.astype(np.float32) - prot_np.astype(np.float32))
    diff_gray = diff.mean(axis=2)

    # Create custom colormap for sensitivity
    sens_cmap = LinearSegmentedColormap.from_list(
        "sensitivity", [(0, 0, 0, 0), (1, 0.2, 0, 0.7)], N=256
    )

    H, W = orig_np.shape[:2]
    # Auto-detect a high-sensitivity crop region
    # Find the densest region of sensitivity
    block = 96
    best_score, best_y, best_x = -1, 0, 0
    for y in range(0, H - block, block // 4):
        for x in range(0, W - block, block // 4):
            score = mask_np[y:y+block, x:x+block].mean()
            if score > best_score:
                best_score, best_y, best_x = score, y, x

    cy, cx = best_y, best_x
    crop_h, crop_w = block, block

    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), dpi=200)
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    # (a) Original
    axes[0, 0].imshow(orig_np)
    axes[0, 0].set_title("(a) Original frame", fontsize=10, weight="bold")
    # Draw crop rectangle
    rect = plt.Rectangle((cx, cy), crop_w, crop_h, linewidth=2,
                         edgecolor="cyan", facecolor="none", linestyle="--")
    axes[0, 0].add_patch(rect)

    # (b) Sensitivity heatmap overlay
    axes[0, 1].imshow(orig_np)
    axes[0, 1].imshow(mask_np, cmap=sens_cmap, vmin=0, vmax=1, alpha=0.7)
    axes[0, 1].set_title("(b) DCRF sensitivity map", fontsize=10, weight="bold")

    # (c) PPEDCRF noise-protected
    axes[0, 2].imshow(prot_np)
    axes[0, 2].set_title(f"(c) PPEDCRF ($\\sigma_0$={int(args.noise_sigma)})", fontsize=10, weight="bold")

    # (d) Difference map
    im_diff = axes[1, 0].imshow(diff_gray, cmap="hot", vmin=0, vmax=diff_gray.max() * 0.8)
    axes[1, 0].set_title("(d) Absolute difference", fontsize=10, weight="bold")
    cbar = fig.colorbar(im_diff, ax=axes[1, 0], fraction=0.046, pad=0.04, label="pixel diff")

    # (e) Mask-guided blur
    axes[1, 1].imshow(blur_np)
    axes[1, 1].set_title("(e) Mask-guided blur (k=21)", fontsize=10, weight="bold")

    # (f) Zoomed crops comparison
    crop_orig = orig_np[cy:cy+crop_h, cx:cx+crop_w]
    crop_prot = prot_np[cy:cy+crop_h, cx:cx+crop_w]
    crop_combined = np.concatenate([crop_orig, crop_prot], axis=1)
    axes[1, 2].imshow(crop_combined)
    axes[1, 2].axvline(x=crop_w, color="white", linewidth=1.5, linestyle="--")
    axes[1, 2].text(crop_w * 0.5, crop_h * 0.07, "Original", ha="center",
                    fontsize=8, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
    axes[1, 2].text(crop_w * 1.5, crop_h * 0.07, "Protected", ha="center",
                    fontsize=8, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
    axes[1, 2].set_title("(f) Zoomed crop (cyan box)", fontsize=10, weight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved qualitative figure to {out}")


if __name__ == "__main__":
    main()
