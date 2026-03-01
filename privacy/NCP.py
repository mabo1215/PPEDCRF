# ppedcrf/privacy/ncp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch


@dataclass
class NCPConfig:
    # alpha 控制整体扰动强度（论文里的“control knob”）
    alpha: float = 1.0
    # epsilon 防止除零
    eps: float = 1e-6


class NCPAllocator:
    """
    把“层级敏感度/类别敏感度” -> 噪声强度权重 (0~1 或更大)
    """
    def __init__(self, cfg: NCPConfig, class_sensitivity: Optional[Dict[int, float]] = None):
        self.cfg = cfg
        self.class_sensitivity = class_sensitivity or {}  # class_id -> sensitivity scalar

    def allocate(self, sens_map: torch.Tensor, class_id_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        sens_map: (B,1,H,W) 或 (B,H,W)  连续敏感度（来自你的敏感区域网络/显著性）
        class_id_map: (B,1,H,W) 可选，像素级类别 id，用于“层级敏感度树”
        return: w (B,1,H,W) 噪声幅度权重
        """
        if sens_map.dim() == 3:
            sens_map = sens_map.unsqueeze(1)

        w = sens_map.clamp(min=0.0)

        if class_id_map is not None:
            if class_id_map.dim() == 3:
                class_id_map = class_id_map.unsqueeze(1)
            # 逐像素按类别敏感度加权（你可以替换成“树结构 NCP”）
            # 这里给一个最小可用版本：w *= s_class
            s_class = torch.ones_like(w)
            for cid, s in self.class_sensitivity.items():
                s_class = torch.where(class_id_map == cid, torch.tensor(float(s), device=w.device), s_class)
            w = w * s_class

        # Normalize 到 [0,1]，再乘 alpha
        w = w / (w.amax(dim=(-2, -1), keepdim=True) + self.cfg.eps)
        w = w * float(self.cfg.alpha)
        return w