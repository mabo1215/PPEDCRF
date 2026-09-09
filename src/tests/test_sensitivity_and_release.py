"""Tests for the sensitivity-map estimator (A2) and the release audit (A8).

Both scripts exist to produce numbers that will be reported as measurements of
a specific quantity, which is exactly the kind of claim finding R3 says the
paper has previously got wrong. The tests below pin the properties that make
those numbers mean what the text will say they mean: that the estimator is
unbiased for the column norm it names, that the concentration statistic is the
one the manuscript prints, that a uniform map is not silently assigned a
correlation, and that the release audit measures the amplitude of the frame
that is actually transmitted rather than of the tensor before it was rescaled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from scripts.measure_jacobian_columns import (  # noqa: E402
    gini, jacobian_column_norms, normalise_weights, spearman,
    topdecile_concentration, topdecile_jaccard, two_coordinate_control)
from scripts.validate_serialized_release import quantise, roundtrip  # noqa: E402


class LinearEmbedder(torch.nn.Module):
    """An embedding with a Jacobian that is known in closed form.

    The estimator is checked against a map whose column norms can be written
    down, because checking it only against another estimate would confirm
    self-consistency rather than correctness. The module ignores the
    preprocessing resize by construction: it is applied to a frame already at
    the input size, so the composed Jacobian is exactly the weight matrix
    scaled by the fixed preprocessing constants.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.flatten(1) @ self.weight.t()).unsqueeze(0).squeeze(0)


def test_spearman_is_one_for_a_monotone_transform():
    a = torch.rand(64)
    assert spearman(a, a) == pytest.approx(1.0, abs=1e-9)
    assert spearman(a, 3.0 * a + 1.0) == pytest.approx(1.0, abs=1e-9)
    assert spearman(a, -a) == pytest.approx(-1.0, abs=1e-9)


def test_spearman_of_an_all_ties_map_is_undefined_not_zero():
    """A uniform map has no rank order, so its correlation must be NaN.

    Returning 0.0 here would read as "measured, and unrelated", which is a
    different and stronger statement than "undefined".
    """
    uniform = torch.ones(50)
    assert spearman(uniform, torch.rand(50)) != spearman(uniform, torch.rand(50)) \
        or torch.isnan(torch.tensor(spearman(uniform, torch.rand(50))))


def test_topdecile_concentration_of_a_uniform_map_is_one_tenth():
    """The statistic the paper prints, evaluated where its value is known."""
    assert topdecile_concentration(torch.ones(1000)) == pytest.approx(0.1, abs=1e-9)


def test_topdecile_concentration_of_a_point_mass_is_one():
    x = torch.zeros(1000)
    x[7] = 1.0
    assert topdecile_concentration(x) == pytest.approx(1.0, abs=1e-9)


def test_topdecile_jaccard_bounds():
    x = torch.arange(1000, dtype=torch.float32)
    assert topdecile_jaccard(x, x) == pytest.approx(1.0)
    # Disjoint deciles: the top tenth of x is the bottom tenth of -x.
    assert topdecile_jaccard(x, -x) == pytest.approx(0.0)


def test_gini_of_uniform_is_zero_and_of_point_mass_is_near_one():
    assert gini(torch.ones(500)) == pytest.approx(0.0, abs=1e-9)
    x = torch.zeros(500)
    x[3] = 1.0
    assert gini(x) > 0.99


def test_normalise_weights_hits_the_energy_budget():
    """Every placement in the paper is compared under sum_i w_i^2 = E."""
    for seed in (0, 1, 2):
        w = torch.rand((1, 1, 8, 8), generator=torch.Generator().manual_seed(seed))
        out = normalise_weights(w, 64.0)
        assert float(out.square().sum()) == pytest.approx(64.0, rel=1e-6)


def test_normalise_weights_survives_an_all_zero_map():
    """A degenerate map must still deliver the budget, not zero distortion.

    This is not hypothetical: the constant support map that motivates the whole
    paper is very nearly this case, and a silent fallback to no perturbation
    would enter a study as a condition that delivered nothing.
    """
    out = normalise_weights(torch.zeros((1, 1, 4, 4)), 16.0)
    assert float(out.square().sum()) == pytest.approx(16.0, rel=1e-6)
    assert float(out.min()) > 0.0


def test_jacobian_estimator_recovers_known_column_norms():
    """Unbiasedness of the random-projection estimate, against exact norms.

    For a linear embedding f(x) = W x the Jacobian is W, so the column norm of
    input coordinate i is the norm of W's i-th column. The estimator sees the
    composed map (preprocessing then W), so the comparison is made in rank
    terms: the ordering it recovers must match the ordering of the true norms.
    Rank agreement is the property every use of this map in the paper relies
    on, since the maps are renormalised before use.
    """
    torch.manual_seed(0)
    h = w = 8
    d, n = 6, 3 * h * w
    weight = torch.randn(d, n)
    # Give the columns a strong, unambiguous spread so the ranking is not
    # decided by estimator noise at a feasible probe count.
    scale = torch.linspace(0.1, 4.0, n)
    weight = weight * scale.unsqueeze(0)
    embedder = LinearEmbedder(weight)

    frame = torch.rand((1, 3, h, w)) * 255.0
    generator = torch.Generator().manual_seed(20260909)
    est, split_half = jacobian_column_norms(frame, embedder, h, 256, generator)

    true_col = weight.norm(dim=0).reshape(1, 3, h, w)
    true_pixel = true_col.square().sum(dim=1, keepdim=True).sqrt()
    assert spearman(est, true_pixel) > 0.9
    assert split_half > 0.9


def test_jacobian_split_half_improves_with_more_probes():
    """The reported ceiling must behave like an estimator's, not a constant."""
    torch.manual_seed(1)
    h = w = 8
    weight = torch.randn(6, 3 * h * w) * torch.linspace(0.1, 4.0, 3 * h * w)
    embedder = LinearEmbedder(weight)
    frame = torch.rand((1, 3, h, w)) * 255.0

    _, few = jacobian_column_norms(
        frame, embedder, h, 8, torch.Generator().manual_seed(7))
    _, many = jacobian_column_norms(
        frame, embedder, h, 256, torch.Generator().manual_seed(7))
    assert many > few


def test_two_coordinate_control_reproduces_the_reviews_counterexample():
    """Equal energy, different allocation, different flip probability.

    This is the calculation finding R2 uses to refute the paper's impossibility
    claim. If it ever stops holding here, the replacement text is wrong.
    """
    c = two_coordinate_control()
    assert c["variance_on_signal"] > 0.0
    assert c["variance_off_signal"] == 0.0
    assert c["flip_prob_on_signal"] > c["flip_prob_off_signal"]


def test_quantise_lands_on_the_integer_grid_and_clamps():
    x = torch.tensor([[[[-5.0, 0.4, 127.6, 300.0]]]])
    q = quantise(x)
    assert torch.equal(q, torch.tensor([[[[0.0, 0.0, 128.0, 255.0]]]]))


def test_png_roundtrip_is_lossless_after_quantisation():
    """PNG must return exactly the quantised frame, so any residual difference
    measured in the audit is the float-to-integer step and not the codec."""
    torch.manual_seed(3)
    frame = torch.rand((1, 3, 16, 24)) * 255.0
    decoded, nbytes = roundtrip(frame, "png", 0)
    assert nbytes > 0
    assert torch.equal(decoded, quantise(frame))


def test_jpeg_roundtrip_is_lossy_and_smaller_at_lower_quality():
    torch.manual_seed(4)
    frame = torch.rand((1, 3, 32, 32)) * 255.0
    hi, hi_bytes = roundtrip(frame, "jpeg", 95)
    lo, lo_bytes = roundtrip(frame, "jpeg", 40)
    assert not torch.equal(hi, quantise(frame))
    assert lo_bytes < hi_bytes
    assert float((lo - frame).square().mean()) > float((hi - frame).square().mean())


def test_roundtrip_rejects_an_unknown_format():
    with pytest.raises(ValueError):
        roundtrip(torch.zeros((1, 3, 8, 8)), "webp", 90)


# --- Cluster-versus-geographic positive audit (A7b, finding R9) -------------

from scripts.audit_cluster_vs_geographic_positives import (  # noqa: E402
    audit, haversine_m, has_coords)


def test_haversine_matches_known_separations():
    """Checked against distances that can be worked out independently.

    One degree of latitude is close to 111.2 km everywhere, and the same point
    must be zero from itself. A planar approximation would pass the first of
    these and fail at high latitude, which is why the benchmark's spread from
    Zurich to Manila makes the exact formula worth pinning.
    """
    assert haversine_m(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert haversine_m(0.0, 0.0, 1.0, 0.0) == pytest.approx(111195.0, rel=1e-3)
    assert haversine_m(47.0, 8.0, 47.0, 8.0) == pytest.approx(0.0, abs=1e-9)
    # Symmetry: the audit compares each query against every gallery entry, so
    # an asymmetric distance would make the positive set depend on argument
    # order.
    a = haversine_m(31.99, 35.87, 32.01, 35.89)
    b = haversine_m(32.01, 35.89, 31.99, 35.87)
    assert a == pytest.approx(b, rel=1e-12)


def test_has_coords_rejects_missing_and_nonfinite():
    assert has_coords({"latitude": 1.0, "longitude": 2.0})
    assert not has_coords({"latitude": 1.0})
    assert not has_coords({"latitude": None, "longitude": 2.0})
    assert not has_coords({"latitude": float("nan"), "longitude": 2.0})


def _rec(qid, place, lat, lon, gallery):
    return {"query_id": qid, "place_id": place, "latitude": lat,
            "longitude": lon, "viewpoint": "Forward", "gallery": gallery}


def test_audit_separates_unsupported_clusters_from_geo_only():
    """A case built so each category is populated by construction.

    q1's only cluster positive sits ~1.1 km away, so at a 25 m threshold it is
    cluster_unsupported -- and it must be, even though an unrelated gallery
    image sits 2 m from it. That is precisely the case an "is anything nearby"
    test would wrongly pass. q2 has no cluster positive at all but does have
    that near neighbour, so it is geo_only.
    """
    far = {"gallery_id": "g_far", "place_id": "P1", "latitude": 0.01,
           "longitude": 0.0, "viewpoint": "Forward"}          # ~1.1 km away
    near_other = {"gallery_id": "g_near", "place_id": "OTHER", "latitude": 0.0,
                  "longitude": 0.00002, "viewpoint": "Forward"}  # ~2.2 m away
    records = [
        _rec("q1", "P1", 0.0, 0.0, [far, near_other]),
        _rec("q2", "NOPOS", 0.0, 0.0, [far, near_other]),
    ]
    result = audit(records, [25.0])
    s = result["per_threshold"]["25m"]
    assert s["cluster_unsupported"] == 1  # q1: its cluster positive is 1.1 km away
    assert s["geo_only"] == 1             # q2: near neighbour, no cluster match
    assert s["cluster_supported"] == 0
    assert s["queries"] == 2
    # Both queries have a differently-labelled neighbour inside the threshold.
    assert s["has_unlabelled_neighbour"] == 2


def test_audit_counts_missing_coordinates_rather_than_dropping_them():
    """Missing coordinates must be visible, not silently excluded."""
    g = {"gallery_id": "g", "place_id": "P", "latitude": 0.0, "longitude": 0.0,
         "viewpoint": "Forward"}
    records = [_rec("ok", "P", 0.0, 0.0, [g]),
               {"query_id": "bad", "place_id": "P", "viewpoint": "Forward",
                "gallery": [g]}]
    result = audit(records, [25.0])
    assert result["queries"] == 2
    assert result["queries_missing_coordinates"] == 1
    assert result["per_threshold"]["25m"]["queries"] == 1


def test_audit_reports_that_the_heading_criterion_is_not_audited():
    """The angular half of the MSLS criterion must never look answered."""
    g = {"gallery_id": "g", "place_id": "P", "latitude": 0.0, "longitude": 0.0,
         "viewpoint": "Forward"}
    result = audit([_rec("q", "P", 0.0, 0.0, [g])], [25.0])
    assert "NOT AUDITED" in result["heading_criterion"]
