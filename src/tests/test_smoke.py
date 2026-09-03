from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from models.dynamic_crf import DynamicCRF, DynamicCRFConfig  # noqa: E402
from privacy.NCP import NCPAllocator, NCPConfig  # noqa: E402
from privacy.noise_injector import NoiseInjector, NoiseConfig  # noqa: E402


def test_smoke() -> None:
    """Basic smoke test so pytest collects at least one test."""
    assert True


def test_dcrf_preserves_shape_and_range() -> None:
    """Refined probability map must keep the input's spatial shape and stay in [0,1]."""
    crf = DynamicCRF(DynamicCRFConfig(n_iters=5, spatial_weight=2.0, temporal_weight=2.0))
    unary = torch.randn(2, 1, 16, 24)
    refined, next_prior = crf.refine(unary, prev_prob=None, flow=None)
    assert refined.shape == (2, 1, 16, 24)
    assert next_prior.shape == refined.shape
    assert torch.all(refined >= 0.0) and torch.all(refined <= 1.0)


def test_dcrf_temporal_state_reset() -> None:
    """A fresh call with prev_prob=None must not depend on any prior call's state."""
    crf = DynamicCRF(DynamicCRFConfig(n_iters=5, spatial_weight=2.0, temporal_weight=2.0))
    torch.manual_seed(0)
    unary = torch.randn(1, 1, 8, 8)

    # Warm up internal state with an unrelated sequence.
    warm_unary = torch.randn(1, 1, 8, 8)
    warm_refined, _ = crf.refine(warm_unary, prev_prob=None, flow=None)
    _, _ = crf.refine(unary, prev_prob=warm_refined, flow=None)

    # A fresh, independent call with prev_prob=None must reproduce the
    # first-frame result exactly, regardless of what ran before.
    first_call, _ = crf.refine(unary, prev_prob=None, flow=None)
    reset_call, _ = crf.refine(unary, prev_prob=None, flow=None)
    assert torch.equal(first_call, reset_call)


def test_ncp_allocate_shape() -> None:
    ncp = NCPAllocator(NCPConfig(alpha=1.0), class_sensitivity=None)
    sens_map = torch.rand(2, 1, 16, 24)
    strength = ncp.allocate(sens_map)
    assert strength.shape == sens_map.shape
    assert torch.all(strength >= 0.0)


def test_noise_injector_output_is_clamped() -> None:
    injector = NoiseInjector(NoiseConfig(mode="indexed_gaussian", sigma=64.0, seed=1234))
    frame = torch.full((1, 3, 8, 8), 128.0)
    sens_mask = torch.ones(1, 1, 8, 8)
    strength = torch.ones(1, 1, 8, 8)
    out = injector.apply(frame, sens_mask, strength, t_index=0)
    assert out.shape == frame.shape
    assert torch.all(out >= 0.0) and torch.all(out <= 255.0)


def test_noise_injector_deterministic_seed() -> None:
    """Same seed and t_index must reproduce byte-identical output; a different
    seed must not."""
    frame = torch.full((1, 3, 8, 8), 128.0)
    sens_mask = torch.ones(1, 1, 8, 8)
    strength = torch.ones(1, 1, 8, 8)

    out_a = NoiseInjector(NoiseConfig(mode="indexed_gaussian", sigma=8.0, seed=1234)).apply(
        frame, sens_mask, strength, t_index=0
    )
    out_b = NoiseInjector(NoiseConfig(mode="indexed_gaussian", sigma=8.0, seed=1234)).apply(
        frame, sens_mask, strength, t_index=0
    )
    out_c = NoiseInjector(NoiseConfig(mode="indexed_gaussian", sigma=8.0, seed=5678)).apply(
        frame, sens_mask, strength, t_index=0
    )
    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, out_c)


def test_noise_injector_zero_mask_leaves_frame_unchanged() -> None:
    """Zero sensitivity mask means zero perturbation everywhere."""
    injector = NoiseInjector(NoiseConfig(mode="indexed_gaussian", sigma=8.0, seed=1234))
    frame = torch.full((1, 3, 8, 8), 128.0)
    sens_mask = torch.zeros(1, 1, 8, 8)
    strength = torch.ones(1, 1, 8, 8)
    out = injector.apply(frame, sens_mask, strength, t_index=0)
    assert torch.equal(out, frame)
