from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DynamicCRFConfig:
    """
    Lightweight dynamic-CRF / mean-field style refinement.

    n_iters: mean-field iterations
    spatial_weight: strength of spatial smoothing term
    temporal_weight: strength of temporal consistency term
    smooth_kernel: average pooling kernel size for spatial message passing
    eps: numerical stability constant
    """
    n_iters: int = 5
    spatial_weight: float = 2.0
    temporal_weight: float = 2.0
    smooth_kernel: int = 3
    eps: float = 1e-6


def _avg_pool_smooth(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)


class DynamicCRF:
    """
    Minimal dynamic-CRF refinement:

      prob <- sigmoid( unary_logit
                       + spatial_weight * (smooth(prob) - prob)
                       + temporal_weight * (warp(prev_prob) - prob) )

    This is intentionally lightweight and dependency-free.
    You can later replace it with DenseCRF / full mean-field if needed.
    """

    def __init__(self, cfg: DynamicCRFConfig):
        self.cfg = cfg

    @torch.no_grad()
    def refine(
        self,
        unary_logit: torch.Tensor,                  # (B,1,H,W) or (B,H,W)
        prev_prob: Optional[torch.Tensor] = None,   # (B,1,H,W) or (B,H,W)
        flow: Optional[torch.Tensor] = None,        # (B,2,H,W) optical flow (dx,dy) in pixels
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if unary_logit.dim() == 3:
            unary_logit = unary_logit.unsqueeze(1)

        prob = torch.sigmoid(unary_logit)

        warped_prev = None
        if prev_prob is not None:
            if prev_prob.dim() == 3:
                prev_prob = prev_prob.unsqueeze(1)
            warped_prev = prev_prob if flow is None else self.warp(prev_prob, flow)

        for _ in range(self.cfg.n_iters):
            spatial = _avg_pool_smooth(prob, self.cfg.smooth_kernel) - prob
            spatial_term = self.cfg.spatial_weight * spatial

            if warped_prev is None:
                temporal_term = 0.0
            else:
                temporal_term = self.cfg.temporal_weight * (warped_prev - prob)

            logit = unary_logit + spatial_term + temporal_term
            prob = torch.sigmoid(logit)

        return prob, prob  # refined prob, and next-frame prior

    def warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp tensor x using optical flow.

        Args:
            x: (B,1,H,W)
            flow: (B,2,H,W) where flow[:,0]=dx, flow[:,1]=dy in pixels

        Returns:
            warped x: (B,1,H,W)
        """
        B, _, H, W = x.shape

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,H,W,2)

        fx = flow[:, 0] / ((W - 1) / 2.0 + self.cfg.eps)
        fy = flow[:, 1] / ((H - 1) / 2.0 + self.cfg.eps)
        grid = grid + torch.stack([fx, fy], dim=-1)

        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)