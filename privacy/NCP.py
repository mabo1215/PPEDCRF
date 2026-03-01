from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import torch


@dataclass
class NCPConfig:
    """
    NCP (Normalized Control Penalty) configuration.
    alpha: global control knob for perturbation strength.
    eps: numerical stability constant.
    """
    alpha: float = 1.0
    eps: float = 1e-6


class NCPAllocator:
    """
    Allocate per-pixel perturbation weights based on:
      - a continuous sensitivity map (sens_map)
      - optional per-pixel class id map (class_id_map) for hierarchical sensitivity
    """

    def __init__(
        self,
        cfg: NCPConfig,
        class_sensitivity: Optional[Dict[int, float]] = None,
    ):
        self.cfg = cfg
        self.class_sensitivity = class_sensitivity or {}  # class_id -> sensitivity scalar

    def allocate(
        self,
        sens_map: torch.Tensor,                 # (B,1,H,W) or (B,H,W)
        class_id_map: Optional[torch.Tensor] = None,  # (B,1,H,W) or (B,H,W)
    ) -> torch.Tensor:
        """
        Returns:
            w: (B,1,H,W) per-pixel perturbation weight.
        """
        if sens_map.dim() == 3:
            sens_map = sens_map.unsqueeze(1)

        w = sens_map.clamp(min=0.0)

        if class_id_map is not None:
            if class_id_map.dim() == 3:
                class_id_map = class_id_map.unsqueeze(1)

            class_weight = torch.ones_like(w)
            for cid, s in self.class_sensitivity.items():
                class_weight = torch.where(
                    class_id_map == cid,
                    torch.tensor(float(s), device=w.device),
                    class_weight,
                )
            w = w * class_weight

        # Normalize to [0,1] then scale by alpha (control knob)
        w = w / (w.amax(dim=(-2, -1), keepdim=True) + self.cfg.eps)
        w = w * float(self.cfg.alpha)
        return w