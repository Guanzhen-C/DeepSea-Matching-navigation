from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from oceansim.slam import (
    AcousticESIKF,
    BackscatterMapMatcher,
    BackscatterMatcherConfig,
    BathymetricTerrainMatcher,
    InertialState,
    MeasuredBathymetryGrid,
    TerrainRecoveryController,
    TerrainMatcherConfig,
    apply_terrain_match,
    covariance_intersection_pose2,
    fixed_lag_fused_deskew_states,
    information_with_covariance_floor,
    make_terrain_recovery_config,
    motion_compensate_to_body,
    terrain_prior_prefit_bypass_allowed,
    uniform_full_swath_scan_indices,
    voxel_average_backscatter_patch,
    voxel_average_terrain_patch,
)
from oceansim.slam.math3d import rotation_from_rpy


def _textured_world(size: int = 241) -> MeasuredBathymetryGrid:
    axis = np.arange(size, dtype=np.float64) - 0.5 * (size - 1)
    x, y = np.meshgrid(axis, axis)
    elevation = (
        -60.0
        + 0.025 * x
        - 0.018 * y
        + 2.3 * np.sin(x / 11.0)
        + 1.7 * np.cos(y / 13.0)
        + 0.8 * np.sin((x + 1.3 * y) / 7.0)
        + 0.4 * np.cos((2.0 * x - y) / 9.0)
    )
    return MeasuredBathymetryGrid(elevation, resolution_m=1.0)


def _periodic_world(size: int = 401) -> MeasuredBathymetryGrid:
    """Fully measured 2-D terrain with an exact 20 m x ambiguity."""

    axis = np.arange(size, dtype=np.float64) - 0.5 * (size - 1)
    x, y = np.meshgrid(axis, axis)
    elevation = (
        -60.0
        + 2.0 * np.sin(2.0 * np.pi * x / 20.0)
        + 1.4 * np.cos(2.0 * np.pi * y / 37.0)
        + 0.7 * np.sin(2.0 * np.pi * (x / 20.0 + y / 29.0))
    )
    return MeasuredBathymetryGrid(elevation, resolution_m=1.0)


def _backscatter_world(size: int = 321) -> MeasuredBathymetryGrid:
    axis = np.arange(size, dtype=np.float64) - 0.5 * (size - 1)
    x, y = np.meshgrid(axis, axis)
    elevation = np.full_like(x, -60.0)
    texture = np.clip(
        0.45
        + 0.18 * np.sin(x / 5.3)
        + 0.16 * np.cos(y / 7.1)
        + 0.12 * np.sin((x + 1.7 * y) / 3.9)
        + 0.08 * np.cos((2.3 * x - y) / 6.7),
        0.02,
        0.98,
    )
    valid = np.ones_like(texture, dtype=bool)
    return MeasuredBathymetryGrid(
        elevation,
        resolution_m=1.0,
        backscatter=texture,
        backscatter_valid_mask=valid,
    )


def _bottom_patch(
    world: MeasuredBathymetryGrid,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    noisy: bool = False,
) -> np.ndarray:
    x_offset, y_offset = np.meshgrid(
        np.linspace(-35.0, 35.0, 29),
        np.linspace(-25.0, 25.0, 23),
    )
    x = position[0] + x_offset.reshape(-1)
    y = position[1] + y_offset.reshape(-1)
    z = world.seabed_height(x, y)
    if noisy:
        generator = np.random.default_rng(2)
        z = z + generator.normal(0.0, 0.03, z.size)
        outlier = np.arange(0, z.size, 19)
        z[outlier] += generator.choice((-2.5, 2.5), outlier.size)
    points_world = np.column_stack((x, y, z))
    return (points_world - position) @ rotation


def _matcher_config(**overrides) -> TerrainMatcherConfig:
    values = dict(
        measurement_std_m=0.15,
        huber_delta_m=0.25,
        inlier_threshold_m=0.50,
        minimum_points=50,
        maximum_points=700,
        effective_independent_points=80,
        maximum_translation_m=8.0,
        maximum_yaw_deg=4.0,
        coarse_translation_step_m=2.0,
        coarse_yaw_step_deg=1.0,
        minimum_information_eigen_ratio=1.0e-4,
        maximum_reduced_chi_square=3.0,
        maximum_innovation_nis=100.0,
    )
    values.update(overrides)
    return TerrainMatcherConfig(**values)


def test_automatic_covariance_intersection_uses_complementary_information() -> None:
    bathymetry_information = np.diag([100.0, 3.0, 0.2])
    backscatter_information = np.diag([2.0, 80.0, 30.0])
    fused_pose, fused_information, weight = covariance_intersection_pose2(
        np.array([0.0, 0.0, np.deg2rad(179.0)]),
        bathymetry_information,
        np.array([1.0, -1.0, np.deg2rad(-179.0)]),
        backscatter_information,
    )

    assert 0.0 < weight < 1.0
    assert np.linalg.slogdet(fused_information)[1] > np.linalg.slogdet(
        bathymetry_information
    )[1]
    assert np.linalg.slogdet(fused_information)[1] > np.linalg.slogdet(
        backscatter_information
    )[1]
    assert np.isfinite(fused_pose).all()
    assert abs(np.rad2deg(fused_pose[2])) > 170.0


def test_covariance_intersection_preserves_a_shared_unobservable_direction() -> None:
    first = np.diag([20.0, 10.0, 0.0])
    second = np.diag([5.0, 30.0, 0.0])

    _, information, _ = covariance_intersection_pose2(
        np.zeros(3), first, np.ones(3), second
    )

    assert information[2, 2] == 0.0


def test_covariance_floor_preserves_an_unobservable_yaw() -> None:
    information = np.diag([4.0, 9.0, 0.0])
    floor = np.diag([1.0, 0.25, np.deg2rad(3.0) ** 2])

    floored = information_with_covariance_floor(information, floor)

    np.testing.assert_allclose(
        floored,
        np.diag([1.0 / 1.25, 1.0 / (1.0 / 9.0 + 0.25), 0.0]),
        atol=1.0e-12,
    )


def test_covariance_floor_matches_full_rank_covariance_addition() -> None:
    information = np.array(
        [[5.0, 1.0, 0.2], [1.0, 3.0, -0.1], [0.2, -0.1, 8.0]]
    )
    floor = np.diag([0.5, 0.5, 0.01])

    expected = np.linalg.inv(np.linalg.inv(information) + floor)
    actual = information_with_covariance_floor(information, floor)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-11, atol=1.0e-12)


def test_full_swath_budget_preserves_cross_track_fans_across_ping_rates() -> None:
    high_rate = uniform_full_swath_scan_indices([96] * 200, 3072)
    low_rate = uniform_full_swath_scan_indices([96] * 30, 3072)

    assert high_rate.size == 32
    assert high_rate[0] == 0
    assert high_rate[-1] == 199
    assert np.unique(high_rate).size == high_rate.size
    assert int(np.max(np.diff(high_rate))) <= 7
    np.testing.assert_array_equal(low_rate, np.arange(30))


def test_fixed_lag_deskew_keeps_acoustic_endpoints_and_smooths_interior() -> None:
    timestamps = (0.0, 1.0, 2.0)
    inertial = [
        InertialState(
            timestamp=t,
            position=np.array([t, 0.0, -10.0]),
            rotation=rotation_from_rpy(0.0, 0.0, 0.1 * t),
        )
        for t in timestamps
    ]
    acoustic = [state.copy() for state in inertial]
    acoustic[0].position += np.array([10.0, 0.0, 0.0])
    acoustic[1].position += np.array([50.0, 7.0, 0.0])
    acoustic[2].position += np.array([12.0, 0.0, 0.0])
    acoustic[0].rotation = rotation_from_rpy(0.0, 0.0, 0.3)
    acoustic[1].rotation = rotation_from_rpy(0.0, 0.0, -1.2)
    acoustic[2].rotation = rotation_from_rpy(0.0, 0.0, 0.7)

    fused = fixed_lag_fused_deskew_states(acoustic, inertial)

    np.testing.assert_allclose(fused[0].position, acoustic[0].position)
    np.testing.assert_allclose(fused[-1].position, acoustic[-1].position)
    np.testing.assert_allclose(fused[0].rotation, acoustic[0].rotation)
    np.testing.assert_allclose(fused[-1].rotation, acoustic[-1].rotation)
    np.testing.assert_allclose(fused[1].position, [12.0, 0.0, -10.0])
    expected_middle_rotation = rotation_from_rpy(0.0, 0.0, 0.5)
    np.testing.assert_allclose(fused[1].rotation, expected_middle_rotation, atol=1e-12)
    assert not np.allclose(fused[1].position, acoustic[1].position)


def test_fixed_lag_deskew_validates_state_pairing() -> None:
    state = InertialState(timestamp=0.0)
    with pytest.raises(ValueError, match="equal length"):
        fixed_lag_fused_deskew_states([state], [])
    with pytest.raises(ValueError, match="timestamps"):
        fixed_lag_fused_deskew_states(
            [InertialState(timestamp=0.0)],
            [InertialState(timestamp=1.0)],
        )


def test_full_swath_budget_rejects_invalid_counts_and_keeps_one_oversize_fan() -> None:
    np.testing.assert_array_equal(
        uniform_full_swath_scan_indices([0, 120, 0], 96),
        [1],
    )
    with pytest.raises(ValueError, match="non-negative integers"):
        uniform_full_swath_scan_indices([96, -1], 3072)
    with pytest.raises(ValueError, match="positive integer"):
        uniform_full_swath_scan_indices([96], 0)


def test_voxel_terrain_reduction_is_bounded_deterministic_and_spatial() -> None:
    x, y = np.meshgrid(
        np.linspace(-15.0, 15.0, 121),
        np.linspace(-80.0, 80.0, 193),
    )
    points = np.column_stack(
        (
            x.reshape(-1),
            y.reshape(-1),
            (0.03 * x + np.sin(y / 9.0)).reshape(-1),
        )
    )

    first, first_size = voxel_average_terrain_patch(points, 3072)
    second, second_size = voxel_average_terrain_patch(points, 3072)

    assert 0 < first.shape[0] <= 3072
    assert first_size > 0.0
    assert second_size == pytest.approx(first_size)
    np.testing.assert_array_equal(first, second)
    # Centroid averaging retains both dimensions of the original footprint.
    assert np.ptp(first[:, 0]) > 28.0
    assert np.ptp(first[:, 1]) > 155.0


def test_voxel_terrain_reduction_keeps_small_input_and_validates_budget() -> None:
    points = np.array(
        [[-1.0, 2.0, -5.0], [3.0, 4.0, -6.0], [np.nan, 0.0, 0.0]]
    )
    reduced, voxel_size = voxel_average_terrain_patch(points, 2)

    np.testing.assert_array_equal(reduced, points[:2])
    assert voxel_size == 0.0
    with pytest.raises(ValueError, match="positive integer"):
        voxel_average_terrain_patch(points, 0)


def test_backscatter_map_match_removes_range_gain_and_recovers_pose() -> None:
    world = _backscatter_world()
    x, y = np.meshgrid(
        np.linspace(-22.0, 22.0, 23),
        np.concatenate((np.linspace(-65.0, -8.0, 25), np.linspace(8.0, 65.0, 25))),
    )
    points_body = np.column_stack(
        (x.reshape(-1), y.reshape(-1), np.full(x.size, -35.0))
    )
    truth_position = np.array([13.0, -11.0, -25.0])
    truth_yaw = np.deg2rad(17.0)
    truth_rotation = rotation_from_rpy(0.0, 0.0, truth_yaw)
    points_world = points_body @ truth_rotation.T + truth_position
    reflectivity = world.measured_reflectivity(
        points_world[:, 0], points_world[:, 1]
    )
    range_gain = (
        1.25
        - 0.004 * np.abs(points_body[:, 1])
        + 0.08 * np.sign(points_body[:, 1])
    )
    intensities = np.clip(reflectivity * range_gain, 0.001, 1.0)
    seed = InertialState(
        timestamp=2.0,
        position=truth_position + np.array([-4.0, 3.0, 0.0]),
        rotation=rotation_from_rpy(0.0, 0.0, truth_yaw - np.deg2rad(1.2)),
    )
    config = BackscatterMatcherConfig(
        maximum_points=2000,
        minimum_points=300,
        minimum_correlation=0.25,
        minimum_basin_correlation_gap=0.01,
        maximum_translation_m=8.0,
        maximum_yaw_deg=2.0,
    )

    result = BackscatterMapMatcher(world, config).match(
        points_body, intensities, seed
    )

    assert result.accepted, result.rejection_reason
    np.testing.assert_allclose(result.matched_pose2[:2], truth_position[:2], atol=0.8)
    assert abs(result.matched_pose2[2] - truth_yaw) < np.deg2rad(0.25)
    assert result.correlation_after > result.correlation_before
    assert result.basin_correlation_gap >= config.minimum_basin_correlation_gap


def test_backscatter_global_match_uses_scale_space_and_context_consistency() -> None:
    world = _backscatter_world()
    x, y = np.meshgrid(
        np.linspace(-45.0, 45.0, 37),
        np.concatenate(
            (np.linspace(-80.0, -8.0, 33), np.linspace(8.0, 80.0, 33))
        ),
    )
    points_body = np.column_stack(
        (x.reshape(-1), y.reshape(-1), np.full(x.size, -35.0))
    )
    truth_position = np.array([13.0, -11.0, -25.0])
    truth_yaw = np.deg2rad(17.0)
    truth_rotation = rotation_from_rpy(0.0, 0.0, truth_yaw)
    points_world = points_body @ truth_rotation.T + truth_position
    reflectivity = world.measured_reflectivity(
        points_world[:, 0], points_world[:, 1]
    )
    range_gain = (
        1.25
        - 0.004 * np.abs(points_body[:, 1])
        + 0.08 * np.sign(points_body[:, 1])
    )
    intensities = np.clip(reflectivity * range_gain, 0.001, 1.0)
    seed = InertialState(
        timestamp=2.0,
        position=truth_position + np.array([-4.0, 3.0, 0.0]),
        rotation=rotation_from_rpy(0.0, 0.0, truth_yaw - np.deg2rad(1.2)),
    )
    config = BackscatterMatcherConfig(
        maximum_points=4096,
        minimum_points=300,
        minimum_correlation=0.20,
        minimum_basin_correlation_gap=0.01,
        maximum_translation_m=8.0,
        maximum_yaw_deg=2.0,
        scale_space_cell_sizes_m=(5.0, 15.0, 30.0),
        scale_space_weight=0.35,
        context_consistency_weight=0.20,
        context_length_m=20.0,
        minimum_scale_cells=8,
        minimum_context_points=20,
    )

    result = BackscatterMapMatcher(world, config).match(
        points_body, intensities, seed
    )

    assert result.accepted, result.rejection_reason
    np.testing.assert_allclose(result.matched_pose2[:2], truth_position[:2], atol=0.8)
    assert abs(result.matched_pose2[2] - truth_yaw) < np.deg2rad(0.25)
    assert np.isfinite(result.fine_correlation_after)
    assert np.isfinite(result.scale_space_correlation_after)
    assert np.isfinite(result.context_consistency_after)
    assert result.valid_contexts_after >= 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"scale_space_cell_sizes_m": (10.0, 5.0)},
        {"scale_space_cell_sizes_m": (5.0, 5.0)},
        {"scale_space_weight": 0.2},
        {
            "scale_space_cell_sizes_m": (5.0,),
            "scale_space_weight": 0.6,
            "context_consistency_weight": 0.4,
        },
    ],
)
def test_backscatter_scale_space_configuration_is_validated(overrides) -> None:
    with pytest.raises(ValueError):
        BackscatterMatcherConfig(**overrides)


def test_backscatter_voxel_average_preserves_registered_mean() -> None:
    points = np.array(
        [
            [0.1, 0.1, -2.0],
            [0.2, 0.2, -4.0],
            [5.1, 0.1, -3.0],
            [np.nan, 0.0, 0.0],
        ]
    )
    values = np.array([0.2, 0.6, 0.9, 0.4])

    reduced_points, reduced_values, voxel_size = (
        voxel_average_backscatter_patch(
            points, values, 8, minimum_voxel_size_m=5.0
        )
    )

    assert voxel_size == pytest.approx(5.0)
    assert reduced_points.shape == (2, 3)
    np.testing.assert_allclose(reduced_points[0], [0.15, 0.15, -3.0])
    np.testing.assert_allclose(reduced_values, [0.4, 0.9])


def test_robust_terrain_match_recovers_absolute_planar_pose() -> None:
    world = _textured_world()
    true_rotation = rotation_from_rpy(0.05, -0.04, 0.35)
    true_position = np.array([12.0, -9.0, -20.0])
    points_sensor = _bottom_patch(
        world, true_position, true_rotation, noisy=True
    )
    expected = np.array([3.2, -2.4, np.deg2rad(1.7)])
    seed_position = true_position.copy()
    seed_position[:2] -= expected[:2]
    seed = InertialState(
        timestamp=5.0,
        rotation=rotation_from_rpy(0.05, -0.04, 0.35 - expected[2]),
        position=seed_position,
        covariance=np.eye(15) * 25.0,
    )

    result = BathymetricTerrainMatcher(
        world, _matcher_config()
    ).match(points_sensor, seed)

    assert result.accepted, result.rejection_reason
    np.testing.assert_allclose(result.correction[:2], expected[:2], atol=0.06)
    np.testing.assert_allclose(result.correction[2], expected[2], atol=0.002)
    np.testing.assert_allclose(result.matched_pose2, [12.0, -9.0, 0.35], atol=0.06)
    assert result.observable_rank == 3
    assert result.translation_observable_rank == 2
    assert result.yaw_observable
    assert result.inliers < result.correspondences
    assert result.rmse_after < result.rmse_before
    assert result.information.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(result.information) > 0.0)
    assert result.distinct_basin_count >= 1
    assert np.isfinite(result.best_basin_cost)
    assert (
        result.distinct_basin_count == 1
        or result.basin_log_likelihood_ratio
        >= result.minimum_basin_log_likelihood_ratio
    )


def test_periodic_equivalent_terrain_basins_are_rejected_as_ambiguous() -> None:
    world = _periodic_world()
    true_position = np.array([0.0, 0.0, -20.0])
    points = _bottom_patch(world, true_position, np.eye(3))
    # The correct +40 m displacement is deliberately outside the search.
    # Because terrain repeats every 20 m, corrections 0 m and +20 m both
    # produce essentially zero residual and must not become an absolute fix.
    seed = InertialState(
        position=np.array([-40.0, 0.0, -20.0]),
        covariance=np.eye(15) * 100.0,
    )
    config = _matcher_config(
        maximum_translation_m=30.0,
        maximum_yaw_deg=4.0,
        coarse_translation_step_m=5.0,
        coarse_yaw_step_deg=1.0,
        enable_innovation_nis_gate=False,
    )

    result = BathymetricTerrainMatcher(world, config).match(points, seed)

    assert not result.accepted
    assert result.rejection_reason == "ambiguous_match"
    assert result.distinct_basin_count >= 2
    assert result.coarse_candidates >= 2
    assert np.isfinite(result.best_basin_cost)
    assert np.isfinite(result.second_best_basin_cost)
    assert result.second_best_basin_cost >= result.best_basin_cost
    assert result.basin_cost_gap >= 0.0
    assert (
        result.basin_log_likelihood_ratio
        < result.minimum_basin_log_likelihood_ratio
    )
    assert result.rmse_after < 1.0e-10


def test_recovery_rejects_equivalent_basins_inside_coarse_seed_radius() -> None:
    """Wide-search seed spacing must not define refined-basin uniqueness."""

    size = 301
    axis = np.arange(size, dtype=np.float64) - 0.5 * (size - 1)
    x, y = np.meshgrid(axis, axis)
    elevation = -48.0 + 0.001 * x + 0.002 * y

    def motif(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (
            -60.0
            + 0.17 * u
            - 0.11 * v
            + 1.8 * np.sin((u + 1.3 * v) / 2.7)
            + 0.9 * np.cos((1.7 * u - v) / 3.1)
        )

    # These compact, fully three-dimensionally observable motifs are exact
    # copies, but their 19 m separation is below recovery's 20 m coarse-seed
    # NMS radius.  They must nevertheless remain separate refined hypotheses.
    for center_x in (0.0, 19.0):
        mask = (np.abs(x - center_x) <= 9.0) & (np.abs(y) <= 9.0)
        elevation[mask] = motif(x[mask] - center_x, y[mask])
    world = MeasuredBathymetryGrid(elevation, resolution_m=1.0)

    offset_x, offset_y = np.meshgrid(
        np.linspace(-4.0, 4.0, 11),
        np.linspace(-4.0, 4.0, 9),
    )
    patch_x = offset_x.reshape(-1)
    patch_y = offset_y.reshape(-1)
    patch_z = world.seabed_height(patch_x, patch_y)
    np.testing.assert_allclose(
        world.seabed_height(patch_x + 19.0, patch_y),
        patch_z,
        atol=1.0e-12,
    )

    seed_position = np.array([0.0, 0.0, -20.0])
    points_body = (
        np.column_stack((patch_x, patch_y, patch_z)) - seed_position
    )
    seed = InertialState(
        position=seed_position,
        covariance=np.eye(15) * 100.0,
    )
    recovery_config = make_terrain_recovery_config(
        _matcher_config(
            maximum_translation_m=30.0,
            require_yaw_observable=True,
            enable_innovation_nis_gate=False,
        ),
        maximum_translation_m=100.0,
    )

    result = BathymetricTerrainMatcher(world, recovery_config).match(
        points_body,
        seed,
        points_are_body_frame=True,
    )

    assert not result.accepted
    assert result.rejection_reason == "ambiguous_match"
    assert result.observable_rank == 3
    assert result.distinct_basin_count >= 2
    assert result.best_basin_cost < 1.0e-12
    assert result.second_best_basin_cost < 1.0e-12
    assert (
        result.basin_log_likelihood_ratio
        < result.minimum_basin_log_likelihood_ratio
    )


def test_match_returns_rank_two_translation_when_yaw_is_unobservable() -> None:
    axis = np.arange(241, dtype=np.float64) - 120.0
    x, y = np.meshgrid(axis, axis)
    radius_squared = x**2 + y**2
    world = MeasuredBathymetryGrid(
        -50.0 + 0.01 * radius_squared + 0.00002 * radius_squared**1.5,
        resolution_m=1.0,
    )
    true_position = np.array([0.0, 0.0, -20.0])
    points = _bottom_patch(world, true_position, np.eye(3))
    estimator = AcousticESIKF()
    estimator.initialize(
        1.0,
        rotation=rotation_from_rpy(0.0, 0.0, 0.08),
        position=np.array([-3.0, 2.0, -20.0]),
        covariance=np.eye(15) * 100.0,
    )

    result = BathymetricTerrainMatcher(
        world,
        _matcher_config(
            maximum_yaw_deg=6.0,
            minimum_information_eigen_ratio=1.0e-3,
        ),
    ).match(points, estimator.state)

    assert result.accepted, result.rejection_reason
    np.testing.assert_allclose(result.correction, [3.0, -2.0, 0.0], atol=0.03)
    assert result.observable_rank == 2
    assert result.translation_observable_rank == 2
    assert not result.yaw_observable
    assert result.distinct_basin_count >= 1
    assert (
        result.distinct_basin_count == 1
        or result.basin_log_likelihood_ratio
        >= result.minimum_basin_log_likelihood_ratio
    )
    np.testing.assert_allclose(result.information[2], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.information[:, 2], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        result.observable_projector_scaled, np.diag([1.0, 1.0, 0.0]), atol=1.0e-10
    )
    yaw_before = np.arctan2(
        estimator.state.rotation[1, 0], estimator.state.rotation[0, 0]
    )
    assert apply_terrain_match(estimator, result)
    yaw_after = np.arctan2(
        estimator.state.rotation[1, 0], estimator.state.rotation[0, 0]
    )
    np.testing.assert_allclose(yaw_after, yaw_before, atol=1.0e-12)
    np.testing.assert_allclose(estimator.state.position[:2], 0.0, atol=0.03)


def test_required_yaw_observability_rejects_rank_two_with_diagnostics() -> None:
    axis = np.arange(241, dtype=np.float64) - 120.0
    x, y = np.meshgrid(axis, axis)
    radius_squared = x**2 + y**2
    world = MeasuredBathymetryGrid(
        -50.0 + 0.01 * radius_squared + 0.00002 * radius_squared**1.5,
        resolution_m=1.0,
    )
    points = _bottom_patch(
        world, np.array([0.0, 0.0, -20.0]), np.eye(3)
    )
    seed = InertialState(
        rotation=rotation_from_rpy(0.0, 0.0, 0.08),
        position=np.array([-3.0, 2.0, -20.0]),
        covariance=np.eye(15) * 100.0,
    )

    result = BathymetricTerrainMatcher(
        world,
        _matcher_config(
            maximum_yaw_deg=6.0,
            minimum_information_eigen_ratio=1.0e-3,
            require_yaw_observable=True,
        ),
    ).match(points, seed)

    assert not result.accepted
    assert result.rejection_reason == "yaw_unobservable"
    assert result.observable_rank == 2
    assert result.translation_observable_rank == 2
    assert not result.yaw_observable
    np.testing.assert_allclose(result.correction, [3.0, -2.0, 0.0], atol=0.03)
    np.testing.assert_allclose(result.information[2], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.information[:, 2], 0.0, atol=1.0e-12)
    assert np.isfinite(result.rmse_before)
    assert np.isfinite(result.rmse_after)
    assert np.isfinite(result.reduced_chi_square)
    assert np.isfinite(result.innovation_nis)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_require_yaw_observable_configuration_requires_boolean(value) -> None:
    with pytest.raises(ValueError, match="require_yaw_observable"):
        _matcher_config(require_yaw_observable=value)


def test_planar_slope_is_rejected_as_horizontally_unobservable() -> None:
    axis = np.arange(121, dtype=np.float64) - 60.0
    x, y = np.meshgrid(axis, axis)
    world = MeasuredBathymetryGrid(
        -50.0 + 0.1 * x + 0.2 * y, resolution_m=1.0
    )
    true_position = np.array([0.0, 0.0, -20.0])
    points = _bottom_patch(world, true_position, np.eye(3))
    seed = InertialState(
        position=np.array([-2.0, 1.0, -20.0]),
        covariance=np.eye(15) * 100.0,
    )

    result = BathymetricTerrainMatcher(
        world,
        _matcher_config(
            maximum_translation_m=5.0,
            maximum_yaw_deg=3.0,
            coarse_translation_step_m=1.0,
            enable_innovation_nis_gate=False,
        ),
    ).match(points, seed)

    assert not result.accepted
    assert result.rejection_reason == "unobservable_terrain"
    assert result.translation_observable_rank == 1


def test_nis_gate_is_strict_by_default_but_can_be_disabled_for_global_seed() -> None:
    world = _textured_world(161)
    true_position = np.array([0.0, 0.0, -20.0])
    points = _bottom_patch(world, true_position, np.eye(3))
    tight_seed = InertialState(
        position=np.array([-3.0, 2.0, -20.0]),
        covariance=np.eye(15) * 1.0e-6,
    )
    strict_config = _matcher_config(
        maximum_translation_m=6.0,
        coarse_translation_step_m=1.0,
        maximum_innovation_nis=10.0,
    )

    rejected = BathymetricTerrainMatcher(
        world, strict_config
    ).match(points, tight_seed)
    accepted = BathymetricTerrainMatcher(
        world,
        _matcher_config(
            maximum_translation_m=6.0,
            coarse_translation_step_m=1.0,
            maximum_innovation_nis=10.0,
            enable_innovation_nis_gate=False,
        ),
    ).match(points, tight_seed)

    assert not rejected.accepted
    assert rejected.rejection_reason == "innovation_nis"
    assert rejected.innovation_nis > 10.0
    assert accepted.accepted
    np.testing.assert_allclose(accepted.correction[:2], [3.0, -2.0], atol=0.03)


def test_apply_match_is_covariance_aware_and_rejects_stale_result() -> None:
    world = _textured_world(181)
    true_rotation = rotation_from_rpy(0.02, -0.03, 0.20)
    true_position = np.array([4.0, -3.0, -20.0])
    points = _bottom_patch(world, true_position, true_rotation)
    estimator = AcousticESIKF()
    covariance = np.eye(15)
    covariance[0:3, 0:3] *= 0.2**2
    covariance[3:6, 3:6] *= 5.0**2
    estimator.initialize(
        7.0,
        rotation=rotation_from_rpy(0.02, -0.03, 0.18),
        position=np.array([2.0, -1.5, -20.0]),
        covariance=covariance,
    )
    result = BathymetricTerrainMatcher(
        world,
        _matcher_config(
            maximum_translation_m=5.0,
            maximum_yaw_deg=3.0,
            coarse_translation_step_m=1.0,
        ),
    ).match(points, estimator.state)

    before_error = np.linalg.norm(estimator.state.position[:2] - true_position[:2])
    assert result.accepted
    assert apply_terrain_match(estimator, result)
    after_error = np.linalg.norm(estimator.state.position[:2] - true_position[:2])
    assert after_error < 0.01 * before_error
    np.testing.assert_allclose(estimator.state.position[2], -20.0, atol=1.0e-12)
    assert not apply_terrain_match(estimator, result)


def test_motion_compensation_uses_only_estimated_poses_and_extrinsics() -> None:
    acquisition = InertialState(
        timestamp=2.0,
        rotation=rotation_from_rpy(0.08, -0.04, 0.35),
        position=np.array([12.0, -7.0, -18.0]),
    )
    reference = InertialState(
        timestamp=5.0,
        rotation=rotation_from_rpy(-0.03, 0.06, 0.52),
        position=np.array([16.0, -4.0, -19.0]),
    )
    sensor_rotation = rotation_from_rpy(0.01, -0.02, 0.03)
    sensor_translation = np.array([0.4, -0.2, 0.15])
    points_sensor = np.array(
        [[1.0, 2.0, -20.0], [-3.0, 4.0, -25.0], [2.5, -1.0, -17.0]]
    )

    compensated = motion_compensate_to_body(
        points_sensor,
        acquisition,
        reference,
        sensor_rotation_body=sensor_rotation,
        sensor_translation_body=sensor_translation,
    )
    reconstructed_world = compensated @ reference.rotation.T + reference.position
    expected_world = (
        (points_sensor @ sensor_rotation.T + sensor_translation)
        @ acquisition.rotation.T
        + acquisition.position
    )
    np.testing.assert_allclose(reconstructed_world, expected_world, atol=1.0e-12)


@pytest.mark.parametrize(
    "override",
    [
        {"coarse_top_k": 1},
        {"minimum_basin_translation_separation_m": 0.0},
        {"minimum_basin_translation_separation_m": np.nan},
        {
            "coarse_seed_translation_separation_m": 1.0,
            "coarse_translation_step_m": 2.0,
        },
        {"coarse_seed_translation_separation_m": 0.0},
        {"coarse_seed_translation_separation_m": np.nan},
        {
            "coarse_seed_translation_separation_m": 17.0,
            "maximum_translation_m": 8.0,
        },
        {
            "minimum_basin_translation_separation_m": 17.0,
            "maximum_translation_m": 8.0,
        },
        {"minimum_basin_yaw_separation_deg": 0.0},
        {"minimum_basin_yaw_separation_deg": np.inf},
        {
            "coarse_seed_yaw_separation_deg": 0.5,
            "coarse_yaw_step_deg": 1.0,
        },
        {"coarse_seed_yaw_separation_deg": 0.0},
        {"coarse_seed_yaw_separation_deg": np.inf},
        {
            "coarse_seed_yaw_separation_deg": 9.0,
            "maximum_yaw_deg": 4.0,
        },
        {
            "minimum_basin_yaw_separation_deg": 9.0,
            "maximum_yaw_deg": 4.0,
        },
        {"minimum_basin_log_likelihood_ratio": 0.0},
        {"minimum_basin_log_likelihood_ratio": np.nan},
    ],
)
def test_basin_uniqueness_configuration_is_strict(override: dict) -> None:
    with pytest.raises(ValueError):
        _matcher_config(**override)


def test_recovery_controller_uses_prior_insertion_and_resets_episode() -> None:
    controller = TerrainRecoveryController(enabled=True, after_rejections=3)

    assert not controller.should_attempt_recovery(normal_prior_added=False)
    controller.complete_observation(prior_added=False, recovery_invoked=False)
    assert not controller.should_attempt_recovery(normal_prior_added=False)
    controller.complete_observation(prior_added=False, recovery_invoked=False)

    # The current failed normal factor is the third failure, so the same patch
    # gets its first recovery invocation now.  A matcher acceptance whose graph
    # prefit failed is still passed as prior_added=False and cannot reset it.
    assert controller.should_attempt_recovery(normal_prior_added=False)
    controller.complete_observation(prior_added=False, recovery_invoked=True)
    assert controller.consecutive_failed_observations == 3
    assert controller.maximum_consecutive_failed_observations == 3
    assert controller.recovery_events == 1
    assert controller.recovery_active

    assert controller.should_attempt_recovery(normal_prior_added=False)
    controller.complete_observation(prior_added=True, recovery_invoked=True)
    assert controller.consecutive_failed_observations == 0
    assert controller.maximum_consecutive_failed_observations == 3
    assert controller.recovery_events == 1
    assert not controller.recovery_active


def test_recovery_config_is_wide_strict_and_preserves_statistical_gates() -> None:
    normal = _matcher_config(
        maximum_translation_m=30.0,
        require_yaw_observable=True,
        minimum_basin_log_likelihood_ratio=np.log(100.0),
    )
    recovery = make_terrain_recovery_config(
        normal,
        maximum_translation_m=100.0,
    )

    expected_changes = {
        "maximum_translation_m": 100.0,
        "coarse_translation_step_m": 10.0,
        "coarse_top_k": 16,
        "coarse_seed_translation_separation_m": 20.0,
        "minimum_basin_translation_separation_m": 5.0,
        "boundary_translation_margin_m": 5.0,
    }
    for config_field in fields(TerrainMatcherConfig):
        name = config_field.name
        expected = expected_changes.get(name, getattr(normal, name))
        assert getattr(recovery, name) == expected
    assert recovery.require_yaw_observable
    assert recovery.minimum_basin_log_likelihood_ratio == pytest.approx(
        np.log(100.0)
    )

    with pytest.raises(ValueError, match="strict rank-three"):
        make_terrain_recovery_config(
            _matcher_config(require_yaw_observable=False),
            maximum_translation_m=100.0,
        )


@pytest.mark.parametrize("mode", ["normal", "recovery"])
def test_local_terrain_modes_cannot_bypass_graph_prefit(mode: str) -> None:
    assert not terrain_prior_prefit_bypass_allowed(
        mode=mode, initializes_global_gauge=False
    )


@pytest.mark.parametrize("mode", ["emergency", "global"])
def test_confirmed_relocalization_modes_can_bypass_graph_prefit(
    mode: str,
) -> None:
    assert terrain_prior_prefit_bypass_allowed(
        mode=mode, initializes_global_gauge=False
    )


def test_initial_gauge_confirmation_can_bypass_graph_prefit() -> None:
    assert terrain_prior_prefit_bypass_allowed(
        mode="normal", initializes_global_gauge=True
    )


def test_unknown_terrain_factor_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported terrain factor mode"):
        terrain_prior_prefit_bypass_allowed(
            mode="typo", initializes_global_gauge=False
        )
