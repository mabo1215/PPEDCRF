from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class NoiseConfig:
    """
    Noise injection configuration.

    mode:
      - "gaussian": independent Gaussian noise per frame
      - "wiener":  reproducible pseudo Wiener-style per-frame noise via seed offset
                 (upgradeable to a true cumulative Wiener process if desired)

    sigma: noise std in pixel domain (0~255 scale)
    clamp_min/max: output clamp range
    seed: if set, ensures reproducibility
    """
    mode: str = "gaussian"
    sigma: float = 8.0
    clamp_min: float = 0.0
    clamp_max: float = 255.0
    seed: Optional[int] = 1234


class NoiseInjector:
    """
    Apply noise only on sensitive background regions using:
      - sens_mask (0..1)
      - strength map from NCP (can be >1)
      - optional foreground_mask to preserve detection utility
    """

    def __init__(self, cfg: NoiseConfig):
        self.cfg = cfg

    def _rng(self) -> Optional[torch.Generator]:
        if self.cfg.seed is None:
            return None
        g = torch.Generator(device="cpu")
        g.manual_seed(self.cfg.seed)
        return g

    def _seeded_randn_like(self, ref: torch.Tensor, g: Optional[torch.Generator]) -> torch.Tensor:
        noise_cpu = torch.randn(ref.shape, generator=g, dtype=ref.dtype, device="cpu")
        return noise_cpu.to(ref.device)

    @torch.no_grad()
    def apply(
        self,
        frame: torch.Tensor,                 # (B,3,H,W) in [0,255]
        sens_mask: torch.Tensor,             # (B,1,H,W) in [0,1]
        strength: torch.Tensor,              # (B,1,H,W), NCP weights
        foreground_mask: Optional[torch.Tensor] = None,  # (B,1,H,W), 1=foreground (protect)
        t_index: int = 0,
    ) -> torch.Tensor:
        assert frame.dim() == 4 and frame.size(1) == 3

        if sens_mask.dim() == 3:
            sens_mask = sens_mask.unsqueeze(1)
        if strength.dim() == 3:
            strength = strength.unsqueeze(1)

        g = self._rng()

        # Target mask: sensitive background regions only
        mask = sens_mask.clamp(0.0, 1.0) * strength

        if foreground_mask is not None:
            if foreground_mask.dim() == 3:
                foreground_mask = foreground_mask.unsqueeze(1)
            mask = mask * (1.0 - foreground_mask.clamp(0.0, 1.0))

        if self.cfg.mode == "gaussian":
            noise = self._seeded_randn_like(frame, g) * float(self.cfg.sigma)

        elif self.cfg.mode == "wiener":
            if g is not None:
                g.manual_seed(int(self.cfg.seed) + int(t_index))
            noise = self._seeded_randn_like(frame, g) * float(self.cfg.sigma)

        else:
            raise ValueError(f"Unknown noise mode: {self.cfg.mode}")

        out = frame + noise * mask  # broadcast mask to 3 channels
        out = out.clamp(self.cfg.clamp_min, self.cfg.clamp_max)
        return out