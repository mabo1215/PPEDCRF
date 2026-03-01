# run_eval.py (核心逻辑示例)
import torch

from ppedcrf.models.dynamic_crf import DynamicCRF, DynamicCRFConfig
from ppedcrf.privacy.ncp import NCPAllocator, NCPConfig
from ppedcrf.privacy.noise_injector import NoiseInjector, NoiseConfig


class DummySensitiveNet(torch.nn.Module):
    """示例：输出 (B,1,H,W) 的敏感度 logit。你实际可用分割/显著性/自定义网络。"""
    def forward(self, x):
        # x: (B,3,H,W) 0~255
        # 这里仅示意：随机输出
        return torch.randn((x.size(0), 1, x.size(2), x.size(3)), device=x.device)


@torch.no_grad()
def process_clip(frames, foreground_masks=None):
    """
    frames: list[Tensor], each (B,3,H,W) in 0~255
    foreground_masks: list[Tensor] or None, each (B,1,H,W) 1=foreground
    """
    device = frames[0].device

    sens_net = DummySensitiveNet().to(device).eval()

    crf = DynamicCRF(DynamicCRFConfig(n_iters=5, spatial_weight=2.0, temporal_weight=2.0))
    ncp = NCPAllocator(NCPConfig(alpha=1.0), class_sensitivity=None)
    injector = NoiseInjector(NoiseConfig(mode="wiener", sigma=8.0, seed=1234))

    prev_prob = None
    out_frames = []

    for t, frame in enumerate(frames):
        unary_logit = sens_net(frame)                 # (B,1,H,W)
        refined_prob, prev_prob = crf.refine(unary_logit, prev_prob=prev_prob, flow=None)

        # NCP：把 refined_prob 当作 sens_map（你也可以用“树结构/层级敏感度”加强）
        strength = ncp.allocate(refined_prob)

        fg = None if foreground_masks is None else foreground_masks[t]
        protected = injector.apply(frame, refined_prob, strength, foreground_mask=fg, t_index=t)
        out_frames.append(protected)

    return out_frames