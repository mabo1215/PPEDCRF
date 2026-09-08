"""Tests for the magnitude/sign/placement decomposition (experiment A3).

The controls only mean what the paper will say they mean if each one keeps
exactly the property it claims to keep and destroys exactly the other. These
tests check that on tensors, so a factorial run that reports "sign carries the
effect" is not resting on an untested derivation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from scripts.run_direction_factorial import (  # noqa: E402
    CONDITIONS, PLACEMENTS, derive_delta, placement_weight,
    release_with_stats)


def _frame(seed: int = 0, h: int = 32, w: int = 48) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, h, w), generator=g) * 255.0


def _delta(seed: int = 1, h: int = 32, w: int = 48) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand((1, 3, h, w), generator=g) - 0.5) * 8.0


def test_sign_shuffle_keeps_magnitudes_and_changes_signs() -> None:
    base = _delta()
    out = derive_delta("sign_shuffle", base,
                       torch.Generator().manual_seed(5))
    assert torch.allclose(out.abs(), base.abs())
    agree = (torch.sign(out) == torch.sign(base)).float().mean()
    assert 0.4 < float(agree) < 0.6, float(agree)


def test_magnitude_uniform_keeps_signs_and_flattens_magnitudes() -> None:
    base = _delta()
    out = derive_delta("magnitude_uniform", base,
                       torch.Generator().manual_seed(5))
    assert torch.equal(torch.sign(out), torch.sign(base))
    nonzero = out[out != 0].abs()
    assert torch.allclose(nonzero, torch.ones_like(nonzero))


def test_direction_is_returned_unchanged_but_not_aliased() -> None:
    base = _delta()
    out = derive_delta("direction", base, torch.Generator().manual_seed(5))
    assert torch.equal(out, base)
    assert out.data_ptr() != base.data_ptr()


def test_every_placement_map_is_normalised_to_unit_mean_square() -> None:
    frame = _frame(seed=2)
    for name in PLACEMENTS:
        w = placement_weight(name, frame)
        assert w.shape == (1, 1, frame.shape[2], frame.shape[3]), name
        assert float(w.min()) >= 0.0, name
        assert abs(float(w.square().mean()) - 1.0) < 1e-5, name


def test_uniform_placement_leaves_a_perturbation_untouched() -> None:
    frame, base = _frame(seed=3), _delta(seed=3)
    assert torch.allclose(base * placement_weight("uniform", frame), base)


def test_edge_placement_concentrates_more_than_uniform() -> None:
    frame = _frame(seed=4)
    edge = placement_weight("edge", frame).square().flatten()
    unit = placement_weight("uniform", frame).square().flatten()
    k = max(1, edge.numel() // 10)
    top_edge = edge.sort(descending=True).values[:k].sum() / edge.sum()
    top_unit = unit.sort(descending=True).values[:k].sum() / unit.sum()
    assert float(top_edge) > float(top_unit)


def test_release_hits_the_target_mse_and_reports_the_clamp() -> None:
    frame, base = _frame(seed=6), _delta(seed=6)
    stats = release_with_stats(frame, base, target_mse=15.68)
    assert abs(stats["effective_mse"] - 15.68) < 0.05
    assert stats["pre_clip_mse"] >= stats["effective_mse"] - 1e-6
    assert 0.0 <= stats["clipped_fraction"] <= 1.0
    assert float(stats["frame"].min()) >= 0.0
    assert float(stats["frame"].max()) <= 255.0


def test_saturated_frame_shows_the_clamp_taking_energy() -> None:
    """A frame at the top of the range cannot absorb a positive perturbation.

    This is the case the manuscript argues about without measuring: the
    delivered MSE can be matched while the clamp removes a large share of what
    was applied, and the two numbers are exported separately for exactly this
    reason.
    """
    frame = torch.full((1, 3, 32, 48), 255.0)
    delta = torch.ones_like(frame)
    stats = release_with_stats(frame, delta, target_mse=15.68)
    assert stats["effective_mse"] == 0.0
    assert stats["clipped_fraction"] == 0.0
    half = torch.cat([torch.full((1, 3, 32, 24), 255.0),
                      torch.full((1, 3, 32, 24), 10.0)], dim=3)
    stats = release_with_stats(half, torch.ones_like(half), target_mse=15.68)
    assert stats["pre_clip_mse"] > stats["effective_mse"]
    assert stats["clipped_fraction"] > 0.4


def test_condition_names_are_the_ones_the_analysis_expects() -> None:
    assert CONDITIONS == ("direction", "sign_shuffle", "magnitude_uniform",
                          "isotropic")


def test_paired_test_uses_the_tie_corrected_approximation() -> None:
    """Pin the zero handling that decides the sign-shuffle p-values.

    Per-query Top-1 differences averaged over three seeds take five values, so
    a sign-shuffle contrast is mostly exact zeros with a handful of +/-1/3 and
    +/-2/3 ties. Discarding the zeros before calling scipy leaves it a short,
    apparently untied sample and it switches to the exact test, which assumes
    continuous data and does not apply here. The published tables report the
    tie-corrected normal approximation; this test fails if the analysis drifts
    back to the exact path, which is how the script and Table VI came to
    disagree (p = 0.35 against the published 0.33).
    """
    from scipy.stats import wilcoxon  # noqa: E402

    from scripts.analyze_direction_factorial import compare  # noqa: E402

    third = 1.0 / 3.0
    diff = [0.0] * 372 + [third] * 8 + [-third] * 14 + [-2 * third] * 6
    cond = {f"q{i}": d for i, d in enumerate(diff)}
    ref = {f"q{i}": 0.0 for i in range(len(diff))}

    got = compare(cond, ref)
    expected = wilcoxon(diff, zero_method="wilcox", method="approx").pvalue
    assert got["p"] == pytest.approx(expected)

    nonzero_only = [d for d in diff if d != 0.0]
    exact = wilcoxon(nonzero_only).pvalue
    assert got["p"] != pytest.approx(exact), (
        "compare() fell back to the exact test on tied data")


def test_paired_test_reports_unity_when_nothing_differs() -> None:
    """Two identical arms have no non-zero differences left to rank."""
    from scripts.analyze_direction_factorial import compare  # noqa: E402

    cond = {f"q{i}": 0.5 for i in range(20)}
    assert compare(cond, dict(cond))["p"] == 1.0
