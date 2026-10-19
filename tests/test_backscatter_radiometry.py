from __future__ import annotations

import numpy as np
import pytest

from standalone.prepare_public_terrain_datasets import _relative_power_registered_layer
from oceansim.slam.acoustic_radiometry import log_backscatter
from oceansim.slam import (
    AcousticTerrainParticleLikelihood, BackscatterMapMatcher,
    GlobalTerrainRelocalizer, GlobalTerrainRelocalizerConfig,
    InertialState, MeasuredBathymetryGrid,
)


def test_documented_db_preserves_power_ratios_and_invalid_coverage() -> None:
    values = np.array([[-50.0, -40.0, -30.0, np.nan]])
    valid = np.array([[True, True, True, False]])
    power, metadata = _relative_power_registered_layer(values, valid)
    np.testing.assert_allclose(power[valid], [0.1, 1.0, 10.0], rtol=1e-6)
    assert np.isnan(power[0, 3])
    assert not metadata["percentile_clipping"]


def test_constant_survey_gain_does_not_change_relative_power() -> None:
    values = np.array([[-58.1, -51.3, -50.0, -43.1, -39.0]])
    valid = np.ones_like(values, dtype=bool)
    first, _ = _relative_power_registered_layer(values, valid)
    second, _ = _relative_power_registered_layer(values + 170.0, valid)
    np.testing.assert_allclose(first, second, rtol=1e-6)


def test_weak_db_variation_survives_inside_a_wide_dynamic_range() -> None:
    values = np.array([[-90.0, -50.0, -49.0, -48.0, -10.0]])
    power, _ = _relative_power_registered_layer(values, np.ones_like(values, dtype=bool))
    assert power[0, 2] / power[0, 1] == pytest.approx(10 ** 0.1)
    assert power[0, 3] / power[0, 1] == pytest.approx(10 ** 0.2)


def test_unsupported_db_dynamic_range_is_not_silently_clipped() -> None:
    with pytest.raises(ValueError, match="dynamic range"):
        _relative_power_registered_layer(np.array([[-1000.0, 1000.0]]), np.ones((1, 2), bool))


def _gain_case(map_gain=1.0, observation_gain=1.0):
    x, y = np.meshgrid(np.arange(-100, 101), np.arange(-100, 101))
    texture = np.exp(0.5 * np.sin(x / 5.7) + 0.4 * np.cos(y / 7.3)
                     + 0.3 * np.sin((x + y) / 3.9))
    world = MeasuredBathymetryGrid(
        np.full_like(texture, -60.0), resolution_m=1,
        backscatter=texture * map_gain,
        metadata={"radiometry": {"encoding": "relative_linear_power_from_documented_db"}},
    )
    a, b = np.meshgrid(np.linspace(-20, 20, 15), np.linspace(-50, 50, 25))
    points = np.c_[a.ravel(), b.ravel(), np.full(a.size, -30)]
    state = InertialState(position=np.array([4.0, -3.0, -30.0]))
    values = world.measured_reflectivity(a.ravel() + 4, b.ravel() - 3)
    values *= observation_gain / map_gain
    return world, state, points, values


def test_relative_log_floor_is_invariant_to_unknown_gain() -> None:
    values = np.array([0.2, 0.5, 1.0, 1.8, 0.001])
    first = log_backscatter(values, 0.01, relative_power=True)
    second = log_backscatter(values * 1e-8, 0.01, relative_power=True)
    np.testing.assert_allclose(second - np.log(1e-8), first, atol=1e-12)
    np.testing.assert_array_equal(log_backscatter(values, .01), np.log(np.clip(values, .01, None)))


def test_matcher_and_particle_scores_survive_dark_relative_power() -> None:
    scores, correlations = [], []
    for gain in (1.0, 1e-7):
        world, state, points, values = _gain_case(gain, gain * 0.2)
        matcher = BackscatterMapMatcher(world)
        diagnostics = matcher._score(
            np.zeros(3), state, points, values, matcher._radiometric_design(points),
            (), matcher._context_group_ids(points, 40),
        )
        correlations.append(diagnostics[0])
        likelihood = AcousticTerrainParticleLikelihood(
            world, state, np.empty((0, 3)),
            backscatter_points_body=points, backscatter_values=values,
        )
        scores.append(likelihood.evaluate(np.array([[4, -3, 0], [15, -3, 0], [4, 9, 0]])))
    assert min(correlations) > .99
    np.testing.assert_allclose(scores[0], scores[1], atol=1e-10)
    assert np.argmax(scores[1]) == 0


def test_global_reference_preserves_db_contrast_below_legacy_floor() -> None:
    first, _, _, _ = _gain_case(1)
    dark, _, _, _ = _gain_case(1e-7)
    config = GlobalTerrainRelocalizerConfig(pyramid_stride=2)
    reference = GlobalTerrainRelocalizer(first, config)._backscatter
    scaled = GlobalTerrainRelocalizer(dark, config)._backscatter
    np.testing.assert_allclose(scaled - np.log(1e-7), reference, atol=3e-6)
    assert np.std(scaled) > .3


def test_particle_radiometric_projection_uses_common_valid_support() -> None:
    world, state, points, values = _gain_case()
    # Unmeasured samples must not change the fit of nuisance gains on the
    # measured half of the mosaic. This models a strip crossing coverage gaps.
    world.backscatter_valid_mask = np.indices(world.elevation_m.shape)[1] < 105
    predicted = world.measured_reflectivity(points[:, 0] + 4, points[:, 1] - 3)
    outside = ~np.isfinite(predicted)
    values[outside] *= np.exp(2.0 + 0.04 * points[outside, 1])
    likelihood = AcousticTerrainParticleLikelihood(
        world, state, np.empty((0, 3)),
        backscatter_points_body=points, backscatter_values=values,
    )
    likelihood.evaluate(np.array([[4, -3, 0]]))
    assert likelihood.last_diagnostics.best_backscatter_correlation > .998


def test_relative_power_does_not_invent_evidence_on_uniform_seabed() -> None:
    world, state, points, _ = _gain_case(1e-7)
    world.backscatter[:] = 1e-7
    values = 1e-8 * np.exp(.2 * np.sign(points[:, 1]) - .002 * np.abs(points[:, 1]))
    likelihood = AcousticTerrainParticleLikelihood(
        world, state, np.empty((0, 3)),
        backscatter_points_body=points, backscatter_values=values,
    )
    np.testing.assert_allclose(likelihood.evaluate(np.array([[4, -3, 0], [15, 5, .1]])), 0, atol=1e-12)
    result = BackscatterMapMatcher(world).match(points, values, state)
    assert not result.accepted
    assert result.rejection_reason == "insufficient_backscatter_texture"
    assert result.valid_fraction == 1.0
    assert result.correspondences == len(points)
