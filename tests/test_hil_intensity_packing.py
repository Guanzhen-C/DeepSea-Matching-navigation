"""Synthetic equivalence only; not a completed navigation/HIL experiment."""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from oceansim.slam.intensity_map import AcousticIntensityMap
from standalone.hil_intensity_packing import build_packed_query, install_layout


@pytest.mark.parametrize("heading", [None, 0., .4])
@pytest.mark.parametrize("cutoff", [None, 6, 10])
def test_layout_candidate_preserves_every_sample_bit_for_bit(heading, cutoff):
    rng = np.random.default_rng(31)
    model = AcousticIntensityMap(voxel_size=2., min_gradient=1e-6)
    for observation, anchor, yaw in ((5, 3, heading), (9, 3, heading), (7, 8, heading)):
        x, y = np.meshgrid(np.arange(-7., 7., .5), np.arange(-6., 6., .5))
        points = np.c_[x.ravel(), y.ravel(), (-8. + .01*x).ravel()]
        points += rng.normal(0., .02, points.shape)
        values = .4 + .01*points[:, 0] + .02*points[:, 1] + rng.normal(0., .01, len(points))
        model.add_points(points, values, heading_rad=yaw, observation_id=observation, anchor_id=anchor)
    queries = rng.uniform([-12., -12., -8.], [12., 12., -8.], (300, 3))
    queries[0] = np.nan
    queries[1] = np.inf
    original = AcousticIntensityMap.query_fixed_anchor_batch
    candidate = build_packed_query(original)
    for anchor in (3, 8, 12):
        kwargs = dict(heading_rad=heading, maximum_observation_id=cutoff, anchor_id=anchor)
        expected = original(model, queries, **kwargs)
        actual = candidate(model, queries, **kwargs)
        for a, b in zip(expected, actual):
            assert (a is None) == (b is None)
            if a is not None:
                for field in fields(a):
                    np.testing.assert_array_equal(getattr(a, field.name), getattr(b, field.name))


def test_layout_candidate_empty_and_all_nonfinite():
    model = AcousticIntensityMap()
    candidate = build_packed_query(AcousticIntensityMap.query_fixed_anchor_batch)
    kwargs = dict(heading_rad=None, maximum_observation_id=None, anchor_id=0)
    assert candidate(model, np.empty((0, 3)), **kwargs) == []
    assert candidate(model, np.full((3, 3), np.nan), **kwargs) == [None]*3


def test_layout_candidate_refuses_unknown_algorithm_source():
    def different_method(self, points):
        return points
    with pytest.raises(ValueError, match="unrecognized"):
        build_packed_query(different_method)


def test_original_layout_never_changes_algorithm_methods():
    original = AcousticIntensityMap.query_fixed_anchor_batch
    audit = {}
    install_layout(audit, "original")
    assert AcousticIntensityMap.query_fixed_anchor_batch is original
    assert audit == {}
    with pytest.raises(ValueError, match="unknown"):
        install_layout(audit, "silent-best")
    assert AcousticIntensityMap.query_fixed_anchor_batch is original


def test_explicit_combined_layout_is_audited_and_validates_both_sources(monkeypatch):
    from oceansim.slam.feature_sonar_odometry import FeatureSonarOdometry
    original_query = AcousticIntensityMap.query_fixed_anchor_batch
    original_frame = FeatureSonarOdometry._canonical_frame
    # Register restoration even though the installer, not monkeypatch, assigns.
    monkeypatch.setattr(AcousticIntensityMap, "query_fixed_anchor_batch", original_query)
    monkeypatch.setattr(FeatureSonarOdometry, "_canonical_frame", original_frame)
    audit = {}
    install_layout(audit, "packed-and-feature-batch")
    assert AcousticIntensityMap.query_fixed_anchor_batch is not original_query
    assert FeatureSonarOdometry._canonical_frame is not original_frame
    assert audit["deployment_candidate"]["layout"] == "packed-and-feature-batch"
    assert not audit["deployment_candidate"]["altered_plane_fit_or_thresholds"]
