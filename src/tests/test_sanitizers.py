"""Smoke tests for the attacker-side sanitizers and the direction optimiser.

These run on CPU with a toy embedder so they exercise the code path a D6
held-out-transform run takes -- optimise once, release at a target MSE, embed
under several transforms -- without any dataset or checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from eval.sanitizers import (  # noqa: E402
    HELD_OUT, RANDOM_ONE_POOL, SANITIZERS, TRAINED)
from scripts.run_direction_transfer_study import (  # noqa: E402
    directional_delta, release_at_mse)


def _frame(seed: int = 0, h: int = 48, w: int = 80) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand((1, 3, h, w), generator=g) * 255.0)


def test_every_sanitizer_preserves_shape_range_and_dtype() -> None:
    frame = _frame()
    for name, op in SANITIZERS.items():
        out = op(frame)
        assert out.shape == frame.shape, name
        assert out.dtype == frame.dtype, name
        assert float(out.min()) >= 0.0 and float(out.max()) <= 255.0, name


def test_trained_and_held_out_sets_are_registered_and_disjoint() -> None:
    assert set(TRAINED) <= set(SANITIZERS)
    assert set(HELD_OUT) <= set(SANITIZERS)
    assert not set(TRAINED) & set(HELD_OUT)
    assert "none" not in TRAINED and "none" not in HELD_OUT


def test_held_out_operators_actually_change_the_frame() -> None:
    frame = _frame(seed=3)
    for name in HELD_OUT:
        out = SANITIZERS[name](frame)
        assert float((out - frame).abs().mean()) > 0.0, name


def test_bit_depth_reduction_uses_sixteen_levels() -> None:
    out = SANITIZERS["bitdepth4"](_frame(seed=5))
    levels = torch.unique(out.round())
    assert len(levels) <= 16


def test_random_one_is_deterministic_in_the_frame() -> None:
    frame = _frame(seed=7)
    a = SANITIZERS["random_one"](frame)
    b = SANITIZERS["random_one"](frame.clone())
    assert torch.equal(a, b)


def test_random_one_pool_excludes_the_composites() -> None:
    """The pool must not contain random_one, or a frame can recurse forever.

    Checking membership rather than calling it: which slot a frame selects is
    a function of its content, so a call-based test passes or fails by luck.
    """
    assert "random_one" not in RANDOM_ONE_POOL
    assert "jpeg50_blur" not in RANDOM_ONE_POOL
    assert set(RANDOM_ONE_POOL) <= set(SANITIZERS)


def test_random_one_reaches_every_operator_in_its_pool() -> None:
    hit = {RANDOM_ONE_POOL[
        __import__("zlib").crc32(
            SANITIZERS["none"](_frame(seed=s)).detach().to("cpu").clamp(0, 255)
            .round().byte().numpy().tobytes()) & 0x7FFFFFFF
        % len(RANDOM_ONE_POOL)] for s in range(200)}
    assert len(hit) >= len(RANDOM_ONE_POOL) - 2, sorted(hit)


class _ToyEmbedder(torch.nn.Module):
    """A fixed random linear map so the optimiser has a real gradient."""

    def __init__(self, h: int, w: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(11)
        self.proj = torch.nn.Parameter(
            torch.randn((3 * h * w, 16), generator=g) / 100.0,
            requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1) @ self.proj


def test_optimise_once_release_once_then_evaluate_under_many_transforms(
        monkeypatch) -> None:
    """The D6 driver path: one perturbation, one release, N attacker-side views."""
    import scripts.run_direction_transfer_study as mod

    h, w = 48, 80
    emb = _ToyEmbedder(h, w).eval()
    monkeypatch.setattr(mod, "normalised_embedding",
                        lambda e, f, isz: torch.nn.functional.normalize(
                            e(f / 255.0), dim=-1))
    frame = _frame(seed=1, h=h, w=w)
    with torch.no_grad():
        target = mod.normalised_embedding(emb, frame, None).detach()
    delta = directional_delta(frame, [target], [emb], [None], steps=3,
                              step_size=1.0, linf=16.0, random_start=1.0,
                              generator=torch.Generator().manual_seed(2),
                              eot_ops=[SANITIZERS["jpeg50"]], eot_samples=1)
    assert float(delta.abs().max()) > 0.0
    released = release_at_mse(frame, delta, target_mse=15.68)
    mse = float((released - frame).square().mean())
    assert abs(mse - 15.68) < 0.05
    ranks = {}
    for name in ("none",) + TRAINED + HELD_OUT:
        view = SANITIZERS[name](released)
        assert view.shape == frame.shape
        with torch.no_grad():
            ranks[name] = mod.normalised_embedding(emb, view, None)
    assert len(ranks) == 1 + len(TRAINED) + len(HELD_OUT)
