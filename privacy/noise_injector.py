# ppedcrf/privacy/noise_injector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import torch


@dataclass
class NoiseConfig:
    mode: str = "gaussian"   # "gaussian" | "wiener"
    sigma: float = 8.0       # 噪声强度（像素域 0~255）
    clamp_min: float = 0.0
    clamp_max: float = 255.0
    seed: Optional[int] = 1234

    # wiener config
    T: float = 1.0
    n_steps: int = 1000       # 用于构造“时间一致”的噪声过程（可按 clip_len 调整）


class NoiseInjector:
    def __init__(self, cfg: NoiseConfig):
        self.cfg = cfg

    def _rng(self, device: torch.device) -> Optional[torch.Generator]:
        if self.cfg.seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(self.cfg.seed)
        return g

    @torch.no_grad()
    def apply(
        self,
        frame: torch.Tensor,        # (B,3,H,W) 取值 0~255
        sens_mask: torch.Tensor,    # (B,1,H,W) 0~1，动态CRF refined 结果
        strength: torch.Tensor,     # (B,1,H,W) NCP 输出，可 >1
        foreground_mask: Optional[torch.Tensor] = None, # (B,1,H,W) 1=foreground (不扰动)
        t_index: int = 0,           # 第几帧（用于 wiener 可复现）
    ) -> torch.Tensor:
        """
        return: perturbed frame
        """
        assert frame.dim() == 4 and frame.size(1) == 3
        if sens_mask.dim() == 3:
            sens_mask = sens_mask.unsqueeze(1)
        if strength.dim() == 3:
            strength = strength.unsqueeze(1)

        device = frame.device
        g = self._rng(device)

        # mask：只对敏感背景注入
        mask = sens_mask.clamp(0, 1) * strength
        if foreground_mask is not None:
            if foreground_mask.dim() == 3:
                foreground_mask = foreground_mask.unsqueeze(1)
            mask = mask * (1.0 - foreground_mask.clamp(0, 1))

        if self.cfg.mode == "gaussian":
            noise = torch.randn_like(frame, generator=g) * float(self.cfg.sigma)
        elif self.cfg.mode == "wiener":
            # 用 t_index 控制每帧噪声的“相关性/可复现”
            # 最小实现：不同帧用不同 seed offset，等价于离散增量；你也可以升级为真正的累积 W_t
            if g is not None:
                g.manual_seed(int(self.cfg.seed) + int(t_index))
            noise = torch.randn_like(frame, generator=g) * float(self.cfg.sigma)
        else:
            raise ValueError(f"Unknown noise mode: {self.cfg.mode}")

        out = frame + noise * mask  # 广播到 3 通道
        out = out.clamp(self.cfg.clamp_min, self.cfg.clamp_max)
        return out