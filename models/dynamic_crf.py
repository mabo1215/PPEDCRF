# ppedcrf/models/dynamic_crf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class DynamicCRFConfig:
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
    轻量 Dynamic-CRF / mean-field 迭代：
      p <- sigmoid( logit_unary + spatial_term + temporal_term )
    """
    def __init__(self, cfg: DynamicCRFConfig):
        self.cfg = cfg

    @torch.no_grad()
    def refine(
        self,
        unary_logit: torch.Tensor,              # (B,1,H,W)
        prev_prob: Optional[torch.Tensor] = None, # (B,1,H,W) 上一帧 refined prob（可选）
        flow: Optional[torch.Tensor] = None,      # (B,2,H,W) 光流（可选，用于 warp）
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if unary_logit.dim() == 3:
            unary_logit = unary_logit.unsqueeze(1)

        prob = torch.sigmoid(unary_logit)

        # temporal prior: warp(prev_prob)
        warped_prev = None
        if prev_prob is not None:
            if prev_prob.dim() == 3:
                prev_prob = prev_prob.unsqueeze(1)
            warped_prev = prev_prob
            if flow is not None:
                warped_prev = self.warp(prev_prob, flow)

        for _ in range(self.cfg.n_iters):
            spatial = _avg_pool_smooth(prob, self.cfg.smooth_kernel) - prob
            spatial_term = self.cfg.spatial_weight * spatial

            temporal_term = 0.0
            if warped_prev is not None:
                temporal_term = self.cfg.temporal_weight * (warped_prev - prob)

            logit = unary_logit + spatial_term + temporal_term
            prob = torch.sigmoid(logit)

        # 返回 refined prob + 作为下一帧 prior 的 prob
        return prob, prob

    def warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        x: (B,1,H,W)
        flow: (B,2,H,W)  (dx,dy) in pixels
        """
        B, _, H, W = x.shape
        # grid in [-1,1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,H,W,2)

        # flow pixels -> normalized
        fx = flow[:, 0] / ((W - 1) / 2.0 + self.cfg.eps)
        fy = flow[:, 1] / ((H - 1) / 2.0 + self.cfg.eps)
        grid = grid + torch.stack([fx, fy], dim=-1)

        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)