"""
Compute PSNR, SSIM, MSE between original and PPEDCRF-protected frames for Table (tab:quality).
Also computes baselines: Global noise (Gaussian on full frame) and White-noise mask (Gaussian on sensitive mask only).
Usage:
  python scripts/compute_quality_table.py
  python scripts/compute_quality_table.py --data_root C:/work/dataset/driving
  python scripts/compute_quality_table.py --update-tex   # update doc/main.tex with all three method rows
Output: mean PSNR, SSIM, MSE for each method.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from eval.metrics import psnr_torch, ssim_grayscale_np
from utils.config import load_yaml
from main import load_sensnet_checkpoint, protect_clip
from models.dynamic_crf import DynamicCRF, DynamicCRFConfig
from privacy.noise_injector import NoiseInjector, NoiseConfig


def mse_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """MSE between two tensors (same shape), in [0,255] scale."""
    return torch.mean((x.float() - y.float()) ** 2).item()


@torch.no_grad()
def protect_global_noise(frame: torch.Tensor, sigma: float = 8.0, seed: int = 1234) -> torch.Tensor:
    """Baseline: add Gaussian noise to the entire frame. frame: (3,H,W)."""
    g = torch.Generator(device=frame.device).manual_seed(seed)
    noise = torch.randn_like(frame, generator=g) * sigma
    return (frame + noise).clamp(0.0, 255.0)


@torch.no_grad()
def protect_white_noise_mask(
    frames: torch.Tensor, sensnet, cfg: dict, device: torch.device, sigma: float = 8.0
) -> torch.Tensor:
    """Baseline: same sensitive mask as PPEDCRF (sensnet+CRF) but add Gaussian noise with strength=1 (no NCP)."""
    from privacy.NCP import NCPAllocator, NCPConfig
    dcfg = cfg["ppedcrf"]["dynamic_crf"]
    crf = DynamicCRF(DynamicCRFConfig(
        n_iters=int(dcfg["n_iters"]),
        spatial_weight=float(dcfg["spatial_weight"]),
        temporal_weight=float(dcfg["temporal_weight"]),
        smooth_kernel=int(dcfg["smooth_kernel"]),
    ))
    injector = NoiseInjector(NoiseConfig(mode="gaussian", sigma=sigma, clamp_min=0.0, clamp_max=255.0, seed=1234))
    prev_prob = None
    out = []
    for t in range(frames.size(0)):
        fr = frames[t:t + 1].to(device)
        unary = sensnet(fr)
        refined_prob, prev_prob = crf.refine(unary, prev_prob=prev_prob, flow=None)
        strength = torch.ones_like(refined_prob)
        protected = injector.apply(fr, refined_prob, strength, foreground_mask=None, t_index=t)
        out.append(protected.squeeze(0).cpu())
    return torch.stack(out, dim=0)


def main():
    p = argparse.ArgumentParser(description="Compute PSNR/SSIM/MSE for quality table")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--data_root", type=str, default=None, help="e.g. C:/source/REAEDP/data or C:/work/dataset/driving")
    p.add_argument("--checkpoint", type=str, default="outputs/sensnet_final.pt")
    p.add_argument("--split", type=str, default="test", help="Dataset split: test, val, or train")
    p.add_argument("--max_clips", type=int, default=20)
    p.add_argument("--update-tex", action="store_true", help="Update doc/main.tex Table tab:quality PPEDCRF row with computed values")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    data_root = args.data_root or cfg["data"]["root"]
    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        print("Run with --data_root <path> (e.g. C:/source/REAEDP/data)")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)

    from datasets.driving_clip_dataset import DrivingClipDataset
    resize_hw = tuple(cfg["data"]["resize_hw"])
    clip_len = int(cfg["data"]["clip_len"])
    ds = DrivingClipDataset(
        data_root, split=args.split, clip_len=clip_len, sample_mode="uniform",
        resize_hw=(int(resize_hw[0]), int(resize_hw[1])),
        max_clips=args.max_clips,
    )
    if len(ds) == 0:
        print(f"No clips in {data_root} split={args.split}")
        sys.exit(1)

    sigma = float(cfg["ppedcrf"]["noise"].get("sigma", 8.0))
    seed = cfg["ppedcrf"]["noise"].get("seed", 1234) or 1234

    ppedcrf_psnr, ppedcrf_ssim, ppedcrf_mse = [], [], []
    global_psnr, global_ssim, global_mse = [], [], []
    wn_psnr, wn_ssim, wn_mse = [], [], []

    for idx, s in enumerate(ds):
        frames = s.frames
        orig = frames[0].cpu().byte().numpy().transpose(1, 2, 0)  # H,W,3

        # PPEDCRF
        protected = protect_clip(frames, sensnet, cfg, device)
        prot = protected[0].cpu().clamp(0, 255).byte().numpy().transpose(1, 2, 0)
        ppedcrf_psnr.append(psnr_torch(frames[0], protected[0]))
        ppedcrf_ssim.append(ssim_grayscale_np(orig, prot))
        ppedcrf_mse.append(mse_torch(frames[0], protected[0]))

        # Global noise (same sigma, seed per clip for reproducibility)
        g_frame = protect_global_noise(frames[0], sigma=sigma, seed=int(seed) + idx)
        g_np = g_frame.cpu().clamp(0, 255).byte().numpy().transpose(1, 2, 0)
        global_psnr.append(psnr_torch(frames[0], g_frame))
        global_ssim.append(ssim_grayscale_np(orig, g_np))
        global_mse.append(mse_torch(frames[0], g_frame))

        # White-noise mask
        wn_protected = protect_white_noise_mask(frames, sensnet, cfg, device, sigma=sigma)
        wn_np = wn_protected[0].cpu().clamp(0, 255).byte().numpy().transpose(1, 2, 0)
        wn_psnr.append(psnr_torch(frames[0], wn_protected[0]))
        wn_ssim.append(ssim_grayscale_np(orig, wn_np))
        wn_mse.append(mse_torch(frames[0], wn_protected[0]))

    n = len(ppedcrf_psnr)
    mean_psnr = sum(ppedcrf_psnr) / n
    mean_ssim = sum(ppedcrf_ssim) / n
    mean_mse = sum(ppedcrf_mse) / n
    g_psnr = sum(global_psnr) / n
    g_ssim = sum(global_ssim) / n
    g_mse = sum(global_mse) / n
    w_psnr = sum(wn_psnr) / n
    w_ssim = sum(wn_ssim) / n
    w_mse = sum(wn_mse) / n

    print("PSNR (dB), SSIM, MSE for Table (tab:quality):")
    print(f"  PPEDCRF (sigma_0=8):     PSNR = {mean_psnr:.2f}, SSIM = {mean_ssim:.4f}, MSE = {mean_mse:.2f}")
    print(f"  Global noise:            PSNR = {g_psnr:.2f}, SSIM = {g_ssim:.4f}, MSE = {g_mse:.2f}")
    print(f"  White-noise mask:        PSNR = {w_psnr:.2f}, SSIM = {w_ssim:.4f}, MSE = {w_mse:.2f}")
    print("LaTeX rows:")
    print(f"  PPEDCRF ($\\\\sigma_0{{=}}8$) & {mean_psnr:.2f} & {mean_ssim:.4f} & {mean_mse:.2f} \\\\\\\\")
    print(f"  Global noise & {g_psnr:.2f} & {g_ssim:.4f} & {g_mse:.2f} \\\\\\\\")
    print(f"  White-noise mask & {w_psnr:.2f} & {w_ssim:.4f} & {w_mse:.2f} \\\\\\\\")

    if args.update_tex:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_tex = os.path.join(root, "doc", "main.tex")
        if not os.path.isfile(main_tex):
            print(f"Warning: {main_tex} not found, skipping --update-tex")
        else:
            with open(main_tex, "r", encoding="utf-8") as f:
                content = f.read()
            # In re.sub replacement, backslash is special; use \\ so output has single \ for LaTeX
            ppedcrf_row = "PPEDCRF ($\\\\sigma_0{=}8$) & " + f"{mean_psnr:.2f} & {mean_ssim:.4f} & {mean_mse:.2f}" + " \\\\"
            content_new = re.sub(
                r"PPEDCRF \(\\\\sigma_0\{=\}8\)\) & [\d.]+ & [\d.]+ & [\d.]+ \\\\",
                ppedcrf_row,
                content,
                count=1,
            )
            # For replace(): string ending with \\ (two chars) is correct for file.
            # For re.sub(): replacement string is interpreted; \\ in repl means one \ in output, so use \\\\\\\\ to get \\ in file.
            global_row = f"Global noise & {g_psnr:.2f} & {g_ssim:.4f} & {g_mse:.2f} \\\\"
            wn_row = f"White-noise mask & {w_psnr:.2f} & {w_ssim:.4f} & {w_mse:.2f} \\\\"
            global_row_repl = f"Global noise & {g_psnr:.2f} & {g_ssim:.4f} & {g_mse:.2f} \\\\\\\\"
            wn_row_repl = f"White-noise mask & {w_psnr:.2f} & {w_ssim:.4f} & {w_mse:.2f} \\\\\\\\"
            content_new = content_new.replace(
                "Global noise & -- & -- & -- \\\\",
                global_row,
                1,
            )
            content_new = content_new.replace(
                "White-noise mask & -- & -- & -- \\\\",
                wn_row,
                1,
            )
            content_new = re.sub(
                r"Global noise & [\d.]+ & [\d.]+ & [\d.]+ \\\\",
                global_row_repl,
                content_new,
                count=1,
            )
            content_new = re.sub(
                r"White-noise mask & [\d.]+ & [\d.]+ & [\d.]+ \\\\",
                wn_row_repl,
                content_new,
                count=1,
            )
            if content_new != content:
                with open(main_tex, "w", encoding="utf-8") as f:
                    f.write(content_new)
                print(f"Updated {main_tex} with all three method rows.")
            else:
                print("Warning: no replacement made in main.tex")
    return mean_psnr, mean_ssim, mean_mse


if __name__ == "__main__":
    main()
