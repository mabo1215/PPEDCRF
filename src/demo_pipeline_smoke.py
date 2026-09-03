"""NOT a scientific evaluation entry point.

This script wires the DCRF/NCP/noise-injection pipeline together on
synthetic random frames using an UNTRAINED, RANDOM unary network
(``RandomUntrainedUnaryNet`` below returns ``torch.randn`` for every input,
regardless of image content). It exists only to smoke-test that the module
plumbing runs end to end (shapes, device placement, the refine/allocate/apply
call sequence).

None of this paper's reported numbers come from this script. Every real
evaluation (proxy12/50 retrieval, MSLS, the M2/M3/M4 GPU experiments) loads
the trained checkpoint via ``main.load_sensnet_checkpoint`` and runs one of
the dedicated ``scripts/run_*.py`` runners instead.
"""

from __future__ import annotations

from typing import List, Optional
import torch

from models.dynamic_crf import DynamicCRF, DynamicCRFConfig
from privacy.NCP import NCPAllocator, NCPConfig
from privacy.noise_injector import NoiseInjector, NoiseConfig


class RandomUntrainedUnaryNet(torch.nn.Module):
    """Returns random logits regardless of input; plumbing smoke-test only.

    Not connected to any trained checkpoint or paper result -- see the
    module docstring above.
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
    print(f"[process_clip] num_frames={len(frames)} device={device}")

    sens_net = RandomUntrainedUnaryNet().to(device).eval()
    crf = DynamicCRF(DynamicCRFConfig(n_iters=5, spatial_weight=2.0, temporal_weight=2.0))
    ncp = NCPAllocator(NCPConfig(alpha=1.0), class_sensitivity=None)
    injector = NoiseInjector(NoiseConfig(mode="indexed_gaussian", sigma=8.0, seed=1234))

    prev_prob = None
    protected_frames: List[torch.Tensor] = []

    for t, frame in enumerate(frames):
        print(f"[process_clip] t={t}, frame_shape={tuple(frame.shape)}")
        unary_logit = sens_net(frame)  # (B,1,H,W)
        refined_prob, prev_prob = crf.refine(unary_logit, prev_prob=prev_prob, flow=None)
        print(
            "[process_clip] "
            f"t={t}, prob_min={refined_prob.min().item():.4f}, "
            f"prob_max={refined_prob.max().item():.4f}"
        )

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

    print(f"[process_clip] done, protected {len(protected_frames)} frames")
    return protected_frames


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, B, H, W = 4, 1, 384, 640
    frames = [torch.rand(B, 3, H, W, device=device) * 255.0 for _ in range(T)]
    print(
        "[demo_pipeline_smoke] NOT a scientific run: random frames, "
        "untrained random unary network, no paper result depends on this."
    )
    out = process_clip(frames)
    print(f"[demo_pipeline_smoke] got {len(out)} protected frames, first frame shape={tuple(out[0].shape)}")
