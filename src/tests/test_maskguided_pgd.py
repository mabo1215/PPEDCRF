"""Tests for the mask-guided PGD baseline (experiment A4).

The comparison against the closest prior method is only fair if the masked
arm differs from the unmasked one in exactly one respect: where the budget is
allowed to go. These tests check the mask is a real restriction, that it has
the requested coverage, and that removing it recovers the unmasked optimiser.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

import scripts.run_maskguided_pgd_baseline as mod  # noqa: E402
from scripts.run_maskguided_pgd_baseline import (  # noqa: E402
    ARMS, MASK_SOURCES, coverage_mask, masked_directional_delta)


class _ToyEmbedder(torch.nn.Module):
    """A fixed random linear map, so the optimiser has a real gradient."""

    def __init__(self, h: int, w: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(13)
        self.proj = torch.nn.Parameter(
            torch.randn((3 * h * w, 16), generator=g) / 100.0,
            requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1) @ self.proj


def _frame(seed: int = 0, h: int = 32, w: int = 48) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, h, w), generator=g) * 255.0


def _patch_embedding(monkeypatch) -> None:
    monkeypatch.setattr(mod, "normalised_embedding",
                        lambda e, f, isz: torch.nn.functional.normalize(
                            e(f / 255.0), dim=-1))


def test_coverage_mask_admits_the_requested_fraction() -> None:
    raw = torch.rand((1, 1, 32, 48))
    for coverage in (0.1, 0.25, 0.5):
        mask = coverage_mask(raw, coverage)
        share = float(mask.mean())
        assert abs(share - coverage) < 0.02, (coverage, share)
        assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_coverage_mask_selects_the_largest_values() -> None:
    raw = torch.arange(24, dtype=torch.float32).view(1, 1, 4, 6)
    mask = coverage_mask(raw, 0.25)
    assert float(raw[mask > 0].min()) > float(raw[mask == 0].max())


def test_masked_optimiser_leaves_masked_out_pixels_untouched(monkeypatch):
    _patch_embedding(monkeypatch)
    h, w = 32, 48
    emb = _ToyEmbedder(h, w).eval()
    frame = _frame(seed=1, h=h, w=w)
    with torch.no_grad():
        target = mod.normalised_embedding(emb, frame, None).detach()
    mask = torch.zeros((1, 1, h, w))
    mask[:, :, :, : w // 4] = 1.0
    delta = masked_directional_delta(
        frame, [target], [emb], [None], mask, steps=3, step_size=1.0,
        linf=16.0, random_start=1.0,
        generator=torch.Generator().manual_seed(4))
    assert float(delta[:, :, :, w // 4:].abs().max()) == 0.0
    assert float(delta[:, :, :, : w // 4].abs().max()) > 0.0


def test_no_mask_is_the_unmasked_optimiser(monkeypatch):
    """A full mask and no mask must give the same perturbation.

    If they diverge, the masked arm is not the unmasked one plus a
    restriction, and any difference the experiment reports is confounded.
    """
    _patch_embedding(monkeypatch)
    h, w = 32, 48
    emb = _ToyEmbedder(h, w).eval()
    frame = _frame(seed=2, h=h, w=w)
    with torch.no_grad():
        target = mod.normalised_embedding(emb, frame, None).detach()
    kwargs = dict(steps=3, step_size=1.0, linf=16.0, random_start=1.0)
    a = masked_directional_delta(
        frame, [target], [emb], [None], None,
        generator=torch.Generator().manual_seed(6), **kwargs)
    b = masked_directional_delta(
        frame, [target], [emb], [None], torch.ones((1, 1, h, w)),
        generator=torch.Generator().manual_seed(6), **kwargs)
    assert torch.allclose(a, b)


def test_gradient_saliency_is_one_channel_and_nonnegative(monkeypatch):
    _patch_embedding(monkeypatch)
    h, w = 32, 48
    emb = _ToyEmbedder(h, w).eval()
    frame = _frame(seed=3, h=h, w=w)
    with torch.no_grad():
        target = mod.normalised_embedding(emb, frame, None).detach()
    raw = mod.gradient_saliency(frame, [emb], [target], [None])
    assert raw.shape == (1, 1, h, w)
    assert float(raw.min()) >= 0.0
    assert float(raw.max()) > 0.0


def test_arm_and_mask_source_names_are_stable() -> None:
    assert ARMS == ("maskguided_pgd", "fullframe_pgd", "isotropic")
    assert MASK_SOURCES == ("cam", "edge", "saliency")
