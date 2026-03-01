from __future__ import annotations

import math
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def psnr_torch(x: torch.Tensor, y: torch.Tensor, data_range: float = 255.0) -> float:
    """
    PSNR between two tensors (any shape), treating them as flattened vectors.
    Typical use: x,y are images in [0,255].

    Returns:
        PSNR in dB (float).
    """
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0.0:
        return 100.0
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def ssim_grayscale_np(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    SSIM computed on grayscale conversion.

    Args:
        img1,img2: uint8 arrays with same shape (H,W) or (H,W,3).

    Returns:
        SSIM score (float).
    """
    assert img1.shape == img2.shape, "Inputs must have the same shape."

    if img1.ndim == 3:
        # Convert to grayscale (works for RGB or BGR as long as consistent)
        img1g = (0.114 * img1[..., 0] + 0.587 * img1[..., 1] + 0.299 * img1[..., 2]).astype(np.uint8)
        img2g = (0.114 * img2[..., 0] + 0.587 * img2[..., 1] + 0.299 * img2[..., 2]).astype(np.uint8)
    else:
        img1g, img2g = img1, img2

    score = ssim(img1g, img2g, data_range=255)
    return float(score)