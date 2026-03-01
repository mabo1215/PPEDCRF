from __future__ import annotations

from typing import List, Optional
import torch

from models.dynamic_crf import DynamicCRF, DynamicCRFConfig
from privacy.NCP import NCPAllocator, NCPConfig
from privacy.noise_injector import NoiseInjector, NoiseConfig


class DummySensitiveRegionNet(torch.nn.Module):
    """
    Placeholder sensitive-region network.
    Replace this with your actual sensitive background predictor.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) in [0,255]
        # return unary logits: (B,1,H,W)
        return torch.randn((x.size(0), 1, x.size(2), x.size(3)), device=x.device)


@torch.no_grad()
def process_clip(
    frames: List[torch.Tensor],  # list of (B,3,H,W) in [0,255]
    foreground_masks: Optional[List[torch.Tensor]] = None,  # list of (B,1,H,W), 1=foreground
) -> List[torch.Tensor]:
    device = frames[0].device

    sens_net = DummySensitiveRegionNet().to(device).eval()
    crf = DynamicCRF(DynamicCRFConfig(n_iters=5, spatial_weight=2.0, temporal_weight=2.0))
    ncp = NCPAllocator(NCPConfig(alpha=1.0), class_sensitivity=None)
    injector = NoiseInjector(NoiseConfig(mode="wiener", sigma=8.0, seed=1234))

    prev_prob = None
    protected_frames: List[torch.Tensor] = []

    for t, frame in enumerate(frames):
        unary_logit = sens_net(frame)  # (B,1,H,W)
        refined_prob, prev_prob = crf.refine(unary_logit, prev_prob=prev_prob, flow=None)

        # Use refined_prob as sensitivity map (you can swap with a richer sensitivity definition)
        strength = ncp.allocate(refined_prob)

        fg = None if foreground_masks is None else foreground_masks[t]
        protected = injector.apply(
            frame=frame,
            sens_mask=refined_prob,
            strength=strength,
            foreground_mask=fg,
            t_index=t,
        )
        protected_frames.append(protected)

    return protected_frames