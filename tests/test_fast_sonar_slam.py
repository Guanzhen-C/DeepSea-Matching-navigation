from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oceansim.slam import (
    AcousticESIKF,
    AcousticIntensityMap,
    AcousticPoseGraph,
    ESIKFConfig,
    FastSonarSLAM,
    FastSonarSLAMConfig,
    IntensityUpdateResult,
    PlanarPoseGraph,
    PoseGraphConfig,
    MeasuredBathymetryGrid,
    PoseGraphEdge,
    SonarUpdateResult,
    SurfelVoxelMap,
    TerrainFollowingSurvey,
    gauge_anchored_initial_covariance,
    initialize_from_imu_dvl_depth,
    sidescan_descriptor,
)
from oceansim.slam.large_scale_sim import (
    FigureEightSurvey,
    LargeScaleSensorSimulator,
    LargeUnderwaterWorld,
)
from oceansim.slam.math3d import skew, so3_exp, so3_log, transform_points


def test_so3_round_trip() -> None:
    tangent = np.array([0.12, -0.08, 0.21])
    np.testing.assert_allclose(so3_log(so3_exp(tangent)), tangent, atol=1.0e-10)


def test_intensity_reference_delay_must_be_strictly_causal() -> None:
    with pytest.raises(ValueError, match="must be at least one"):
        FastSonarSLAM(
            FastSonarSLAMConfig(intensity_min_separation_pings=0)
        )


def test_sensor_only_navigation_initialization() -> None:
    rotation = so3_exp([0.08, -0.04, 0.7])
    # Use RPY yaw because yaw is explicitly a local-frame gauge input.
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    world_up = np.array([0.0, 0.0, 9.80665])
    world_velocity = np.array([2.0, -0.4, 0.1])
    initialized = initialize_from_imu_dvl_depth(
        np.tile(rotation.T @ world_up, (16, 1)),
        np.tile(rotation.T @ world_velocity, (16, 1)),
        np.full(16, 12.5),
        yaw_gauge_rad=yaw,
        horizontal_origin_m=(4.0, -3.0),
    )
    np.testing.assert_allclose(initialized.rotation, rotation, atol=2.0e-3)
    np.testing.assert_allclose(initialized.velocity, world_velocity, atol=3.0e-3)
    np.testing.assert_allclose(initialized.position, [4.0, -3.0, -12.5])
    covariance = gauge_anchored_initial_covariance(
        gyro_bias_std_rad_s=2.0e-6
    )
    assert covariance.shape == (15, 15)
    assert covariance[2, 2] < covariance[0, 0]
    np.testing.assert_allclose(covariance[9, 9], (2.0e-6) ** 2)


def test_robust_dvl_update_recovers_after_acoustic_attitude_change() -> None:
    estimator = AcousticESIKF()
    estimator.initialize(
        0.0,
        rotation=so3_exp([0.0, 0.0, np.deg2rad(3.0)]),
        velocity=np.array([4.0, 0.0, 0.0]),
    )
    before = np.linalg.norm(
        np.array([4.0, 0.0, 0.0])
        - estimator.state.rotation.T @ estimator.state.velocity
    )
    assert estimator.update_dvl(np.array([4.0, 0.0, 0.0]))
    after = np.linalg.norm(
        np.array([4.0, 0.0, 0.0])
        - estimator.state.rotation.T @ estimator.state.velocity
    )
    assert after < before


def test_dvl_observability_projector_preserves_pose_gauge() -> None:
    estimator = AcousticESIKF()
    estimator.initialize(
        0.0,
        rotation=so3_exp(np.deg2rad([8.0, -6.0, 35.0])),
        position=np.array([120.0, -45.0, -30.0]),
        velocity=np.array([2.0, 0.4, -0.1]),
    )
    projector = estimator._dvl_correction_projector()
    world_yaw_axis_body = estimator.state.rotation.T @ np.array(
        [0.0, 0.0, 1.0]
    )
    observable_tilt = np.cross(world_yaw_axis_body, [1.0, 0.0, 0.0])
    observable_tilt /= np.linalg.norm(observable_tilt)

    np.testing.assert_allclose(projector[3:6], 0.0, atol=0.0)
    np.testing.assert_allclose(
        world_yaw_axis_body @ projector[0:3, 0:3],
        0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        projector[0:3, 0:3] @ observable_tilt,
        observable_tilt,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(projector[6:, 6:], np.eye(9), atol=0.0)


def test_dvl_uses_velocity_without_retracting_local_map_pose() -> None:
    estimator = AcousticESIKF()
    rotation = so3_exp(np.deg2rad([7.0, -5.0, 28.0]))
    velocity = np.array([2.2, 0.35, -0.08])
    # Deliberately create the strong pose/velocity cross-covariance that made
    # an ordinary 0.1 m/s DVL residual jump the long-run local pose by metres.
    coupled_direction = np.zeros(15)
    coupled_direction[0:3] = [0.08, -0.05, 0.04]
    coupled_direction[3:6] = [3.0, -2.0, 0.7]
    coupled_direction[6:9] = [0.22, -0.14, 0.06]
    covariance = np.diag(np.full(15, 1.0e-5)) + np.outer(
        coupled_direction, coupled_direction
    )
    estimator.initialize(
        0.0,
        rotation=rotation,
        position=np.array([1300.0, -720.0, -55.0]),
        velocity=velocity,
        covariance=covariance,
    )
    measurement = rotation.T @ velocity + np.array([0.12, -0.07, 0.03])
    position_before = estimator.state.position.copy()
    rotation_before = estimator.state.rotation.copy()
    residual_before = np.linalg.norm(
        measurement - rotation_before.T @ estimator.state.velocity
    )

    assert estimator.update_dvl(measurement)

    residual_after = np.linalg.norm(
        measurement
        - estimator.state.rotation.T @ estimator.state.velocity
    )
    attitude_correction_body = so3_log(
        rotation_before.T @ estimator.state.rotation
    )
    world_yaw_axis_body = rotation_before.T @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(estimator.state.position, position_before, atol=0.0)
    assert residual_after < residual_before
    assert np.linalg.norm(estimator.state.velocity - velocity) > 1.0e-5
    assert np.linalg.norm(attitude_correction_body) > 1.0e-7
    assert abs(attitude_correction_body @ world_yaw_axis_body) < 1.0e-12
    assert np.isfinite(estimator.state.covariance).all()
    assert np.linalg.eigvalsh(estimator.state.covariance)[0] > 0.0


def test_kalman_update_resets_the_joseph_posterior_covariance() -> None:
    estimator = AcousticESIKF()
    generator = np.random.default_rng(73)
    square_root = generator.normal(size=(15, 15)) * 0.015
    prior = square_root @ square_root.T + np.eye(15) * 2.0e-4
    estimator.initialize(
        0.0,
        rotation=so3_exp([0.2, -0.1, 0.3]),
        position=np.array([4.0, -2.0, -8.0]),
        velocity=np.array([1.5, -0.4, 0.2]),
        covariance=prior,
    )
    jacobian = generator.normal(size=(3, 15))
    residual = np.array([0.32, -0.18, 0.11])
    measurement_covariance = np.diag([0.03, 0.04, 0.05]) ** 2
    innovation = jacobian @ prior @ jacobian.T + measurement_covariance
    gain = np.linalg.solve(innovation, jacobian @ prior).T
    error = gain @ residual
    identity_minus = np.eye(15) - gain @ jacobian
    joseph = (
        identity_minus @ prior @ identity_minus.T
        + gain @ measurement_covariance @ gain.T
    )
    reset = estimator._error_reset_jacobian(error)
    expected = estimator._regularize_covariance(reset @ joseph @ reset.T)

    assert estimator._kalman_update(
        residual, jacobian, measurement_covariance
    )

    np.testing.assert_allclose(
        estimator.state.covariance, expected, rtol=2.0e-11, atol=2.0e-12
    )
    # This guard specifically detects the former bug, where the reset result
    # was immediately overwritten by the untransformed Joseph covariance.
    assert np.linalg.norm(expected - joseph) > 1.0e-6


def test_dvl_extrinsic_lever_arm_does_not_create_false_velocity() -> None:
    estimator = AcousticESIKF(
        ESIKFConfig(dvl_velocity_std=0.01),
        dvl_translation_body=np.array([1.2, 0.0, 0.0]),
    )
    estimator.initialize(0.0)
    estimator.propagate(
        0.02,
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.0, 9.80665]),
    )
    state_before = estimator.state.copy()
    expected_dvl_velocity = np.cross(
        np.array([0.0, 0.0, 0.5]), np.array([1.2, 0.0, 0.0])
    )

    assert estimator.update_dvl(expected_dvl_velocity)
    np.testing.assert_allclose(
        estimator.state.velocity, state_before.velocity, atol=1.0e-12
    )
    np.testing.assert_allclose(
        estimator.state.gyro_bias, state_before.gyro_bias, atol=1.0e-12
    )


def test_point_to_surfel_update_corrects_vertical_pose_error() -> None:
    surfels = SurfelVoxelMap(
        voxel_size=1.5,
        min_points=4,
        max_planarity=0.2,
        max_plane_distance=1.0,
    )
    grid_x, grid_y = np.meshgrid(np.linspace(-8.0, 8.0, 40), np.linspace(-8.0, 8.0, 40))
    map_points = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, -10.0))
    )
    surfels.add_points(map_points)

    config = ESIKFConfig(
        sonar_noise_floor=0.05,
        sonar_max_translation_step=0.5,
        sonar_max_rotation_step_deg=2.0,
        degeneracy_eigenvalue=1.0,
    )
    estimator = AcousticESIKF(config)
    estimator.initialize(0.0, position=np.array([0.0, 0.0, 0.2]))
    points_sensor = map_points[::8]
    result = estimator.update_sonar(points_sensor, surfels)

    assert result.accepted
    assert result.rmse_after < result.rmse_before
    assert abs(estimator.state.position[2]) < 0.08


def test_sonar_information_scale_tempers_overlapping_window_likelihood() -> None:
    surfels = SurfelVoxelMap(
        voxel_size=1.5,
        min_points=4,
        max_planarity=0.2,
        max_plane_distance=1.0,
    )
    grid_x, grid_y = np.meshgrid(
        np.linspace(-8.0, 8.0, 40), np.linspace(-8.0, 8.0, 40)
    )
    map_points = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, -10.0))
    )
    surfels.add_points(map_points)
    config = ESIKFConfig(
        sonar_noise_floor=0.05,
        sonar_max_translation_step=0.5,
        sonar_max_rotation_step_deg=2.0,
        degeneracy_eigenvalue=1.0,
    )
    full = AcousticESIKF(config)
    tempered = AcousticESIKF(config)
    for estimator in (full, tempered):
        estimator.initialize(0.0, position=np.array([0.0, 0.0, 0.2]))

    full_result = full.update_sonar(
        map_points[::8], surfels, information_scale=1.0
    )
    tempered_result = tempered.update_sonar(
        map_points[::8], surfels, information_scale=0.25
    )

    assert full_result.accepted and tempered_result.accepted
    assert tempered_result.information_scale == 0.25
    assert abs(full.state.position[2]) < abs(tempered.state.position[2]) < 0.2
    assert tempered.state.covariance[5, 5] > full.state.covariance[5, 5]


def test_surfel_map_slides_without_leaving_stale_point_counts() -> None:
    surfels = SurfelVoxelMap(voxel_size=1.0, min_points=3)
    local = np.array(
        [[0.1, 0.1, -5.0], [0.2, 0.1, -5.0], [0.1, 0.2, -5.0]]
    )
    remote = local + np.array([100.0, 0.0, 0.0])
    surfels.add_points(np.vstack((local, remote)))
    revision = surfels.revision

    removed_voxels, removed_points = surfels.remove_outside_box(
        np.array([0.0, 0.0, -5.0]), 10.0
    )

    assert removed_voxels == 1
    assert removed_points == 3
    assert surfels.point_count == 3
    assert len(surfels) == 1
    assert surfels.revision == revision + 1


def test_surfel_association_box_preserves_revisitable_history() -> None:
    surfels = SurfelVoxelMap(
        voxel_size=10.0, min_points=4, min_tangent_ratio=0.01
    )
    dx, dy = np.meshgrid(np.linspace(-0.9, 0.9, 5), np.linspace(-0.9, 0.9, 5))
    first = np.column_stack(
        (5.0 + dx.ravel(), 5.0 + dy.ravel(), np.full(dx.size, -5.0))
    )
    second = first + np.array([100.0, 0.0, 0.0])
    surfels.add_points(np.vstack((first, second)))
    total_points = surfels.point_count

    surfels.set_association_box(np.array([5.0, 5.0, -5.0]), 20.0)
    first_snapshot = surfels.association_snapshot()
    surfels.set_association_box(np.array([105.0, 5.0, -5.0]), 20.0)
    second_snapshot = surfels.association_snapshot()

    assert first_snapshot.centroids.shape[0] == 1
    assert second_snapshot.centroids.shape[0] == 1
    assert first_snapshot.centroids[0, 0] < 20.0
    assert second_snapshot.centroids[0, 0] > 90.0
    assert surfels.point_count == total_points
    assert len(surfels) == 2


def test_batched_surfel_insert_matches_pointwise_sufficient_statistics() -> None:
    rng = np.random.default_rng(91)
    points = np.column_stack(
        (
            rng.uniform(-4.0, 4.0, 600),
            rng.uniform(-3.0, 3.0, 600),
            -9.0 + rng.normal(0.0, 0.03, 600),
        )
    )
    intensities = rng.uniform(0.05, 0.95, points.shape[0])
    batched = SurfelVoxelMap(voxel_size=1.5, min_points=4)
    pointwise = SurfelVoxelMap(voxel_size=1.5, min_points=4)

    batched.add_points(points, intensities)
    for point, intensity in zip(points, intensities):
        pointwise.add_points(point[None, :], np.array([intensity]))

    assert batched.point_count == pointwise.point_count
    assert len(batched) == len(pointwise)
    assert batched.mature_voxel_count == pointwise.mature_voxel_count
    for key, batched_stats in batched._voxels.items():
        pointwise_stats = pointwise._voxels[key]
        assert batched_stats.count == pointwise_stats.count
        np.testing.assert_allclose(
            batched_stats.mean, pointwise_stats.mean, rtol=1.0e-13, atol=1.0e-13
        )
        np.testing.assert_allclose(
            batched_stats.scatter,
            pointwise_stats.scatter,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        assert batched_stats.intensity_sum == pytest.approx(
            pointwise_stats.intensity_sum, rel=1.0e-13, abs=1.0e-13
        )


def test_observable_subspace_update_keeps_only_supported_plane_directions() -> None:
    surfels = SurfelVoxelMap(
        voxel_size=1.5,
        min_points=4,
        max_planarity=0.2,
        max_plane_distance=1.0,
    )
    grid_x, grid_y = np.meshgrid(
        np.linspace(-8.0, 8.0, 40), np.linspace(-8.0, 8.0, 40)
    )
    map_points = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, -10.0))
    )
    surfels.add_points(map_points)
    estimator = AcousticESIKF(
        ESIKFConfig(
            sonar_noise_floor=0.05,
            sonar_max_translation_step=0.5,
            sonar_max_rotation_step_deg=2.0,
            degeneracy_eigenvalue=1.0,
            degeneracy_rotation_eigenvalue=1.0,
            sonar_observable_subspace_projection=True,
        )
    )
    estimator.initialize(0.0, position=np.array([0.0, 0.0, 0.2]))

    result = estimator.update_sonar(map_points[::8], surfels)

    assert result.accepted
    assert result.degenerate
    assert result.translation_observable_rank == 1
    assert result.rotation_observable_rank == 2
    assert abs(estimator.state.position[2]) < 0.01


def test_continuous_observability_shaping_preserves_strong_and_removes_null() -> None:
    jacobian = np.diag([100.0, 10.0, 0.0])
    weights = np.ones(3)

    transform, eigenvalues, response = (
        AcousticESIKF._continuous_observability_transform(
            jacobian,
            weights,
            absolute_knee=100.0,
            relative_knee=0.0,
        )
    )
    shaped = jacobian @ transform
    shaped_information = shaped.T @ shaped

    np.testing.assert_allclose(eigenvalues, [0.0, 100.0, 10_000.0])
    np.testing.assert_allclose(response, [0.0, 0.5, 100.0 / 101.0])
    np.testing.assert_allclose(
        np.linalg.eigvalsh(shaped_information),
        eigenvalues * response,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.linalg.norm(transform[:, 0]) > 0.99
    assert 0.70 < np.linalg.norm(transform[:, 1]) < 0.71
    assert np.linalg.norm(transform[:, 2]) == pytest.approx(0.0)


def test_vectorized_sonar_linearization_matches_pointwise_model() -> None:
    surfels = SurfelVoxelMap(
        voxel_size=2.0,
        min_points=4,
        max_planarity=0.2,
        max_plane_distance=1.5,
    )
    x, y = np.meshgrid(np.linspace(-7.0, 7.0, 29), np.linspace(-7.0, 7.0, 29))
    map_points = np.column_stack(
        (x.ravel(), y.ravel(), -10.0 + 0.08 * x.ravel() - 0.04 * y.ravel())
    )
    surfels.add_points(map_points)
    estimator = AcousticESIKF(
        ESIKFConfig(sonar_range_std=0.04, sonar_angle_std_deg=0.3),
        sensor_translation_body=np.array([0.2, -0.1, 0.05]),
    )
    estimator.initialize(
        0.0,
        rotation=so3_exp(np.deg2rad([2.0, -1.0, 3.0])),
        position=np.array([0.1, -0.2, 0.15]),
    )
    points_sensor = map_points[::11]
    residual, jacobian, variance = estimator._sonar_linearization(
        estimator.state, points_sensor, surfels
    )

    points_body = transform_points(
        estimator.sensor_rotation_body,
        estimator.sensor_translation_body,
        points_sensor,
    )
    points_world = transform_points(
        estimator.state.rotation, estimator.state.position, points_body
    )
    associations = surfels.associate_points(points_world)
    expected_residual = []
    expected_jacobian = []
    expected_variance = []
    rotation_ws = estimator.state.rotation @ estimator.sensor_rotation_body
    angle_std = np.deg2rad(estimator.config.sonar_angle_std_deg)
    for index, (point_sensor, point_body, point_world) in enumerate(
        zip(points_sensor, points_body, points_world)
    ):
        if not associations.valid_mask[index]:
            continue
        normal = associations.normals[index]
        sonar_range = float(np.linalg.norm(point_sensor))
        if sonar_range < 1.0e-6:
            continue
        direction = point_sensor / sonar_range
        radial = np.outer(direction, direction)
        covariance_sensor = (
            estimator.config.sonar_range_std**2 * radial
            + (angle_std * sonar_range) ** 2 * (np.eye(3) - radial)
        )
        covariance_world = rotation_ws @ covariance_sensor @ rotation_ws.T
        row = np.zeros(15)
        row[0:3] = normal @ (-estimator.state.rotation @ skew(point_body))
        row[3:6] = normal
        expected_residual.append(
            normal @ (point_world - associations.centroids[index])
        )
        expected_jacobian.append(row)
        expected_variance.append(
            normal @ covariance_world @ normal
            + max(0.0, associations.normal_variances[index])
            + estimator.config.sonar_noise_floor**2
        )

    np.testing.assert_allclose(residual, expected_residual, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(variance, expected_variance, rtol=1.0e-12, atol=1.0e-12)


def test_direct_sidescan_intensity_update_reduces_yaw_error() -> None:
    intensity_map = AcousticIntensityMap(
        voxel_size=1.5,
        min_cell_points=2,
        min_neighbours=6,
        min_gradient=1.0e-5,
    )
    x, y = np.meshgrid(
        np.linspace(-8.0, 8.0, 65),
        np.linspace(-28.0, 28.0, 113),
    )

    def texture(x_coordinate, y_coordinate):
        return (
            0.52
            + 0.16 * np.sin(0.31 * x_coordinate + 0.08 * y_coordinate)
            + 0.12 * np.cos(0.17 * y_coordinate)
            + 0.08 * np.sin(0.14 * x_coordinate - 0.23 * y_coordinate)
        )

    map_points = np.column_stack(
        (x.ravel(), y.ravel(), np.full(x.size, -10.0))
    )
    intensity_map.add_points(
        map_points,
        texture(x.ravel(), y.ravel()),
        heading_rad=0.0,
        observation_id=0,
        anchor_id=0,
    )

    cross_track = np.linspace(-25.0, 25.0, 101)
    points_sensor = np.column_stack(
        (
            np.zeros_like(cross_track),
            cross_track,
            np.full_like(cross_track, -10.0),
        )
    )
    measured = texture(points_sensor[:, 0], points_sensor[:, 1])
    config = ESIKFConfig(
        intensity_noise_std=0.025,
        intensity_min_correspondences=30,
        intensity_max_points=101,
        intensity_max_iterations=5,
        intensity_max_rotation_step_deg=0.5,
        intensity_min_yaw_information=0.05,
    )
    estimator = AcousticESIKF(config)
    yaw_error = np.deg2rad(0.8)
    covariance = np.eye(15) * 1.0e-8
    covariance[2, 2] = np.deg2rad(3.0) ** 2
    covariance[3:6, 3:6] = np.eye(3) * 1.0e-6
    estimator.initialize(
        0.0,
        rotation=so3_exp([0.0, 0.0, yaw_error]),
        covariance=covariance,
    )

    result = estimator.update_intensity(
        points_sensor, measured, intensity_map
    )

    assert result.accepted
    assert result.overlap_ratio > 0.9
    assert result.normalized_correlation > 0.9
    assert result.planar_observable_rank >= 2
    assert result.anchor_keyframe_index == 0
    assert np.isfinite(result.photometric_pose2).all()
    assert np.all(np.linalg.eigvalsh(result.information_pose2) >= -1.0e-9)
    corrected_yaw = so3_log(estimator.state.rotation)[2]
    assert abs(corrected_yaw) < abs(yaw_error)


def test_coregistered_intensity_continuously_downweights_wrong_surface() -> None:
    intensity_map = AcousticIntensityMap(
        voxel_size=1.0,
        min_cell_points=1,
        min_neighbours=5,
        min_gradient=1.0e-6,
    )
    x, y = np.meshgrid(
        np.arange(-3.0, 4.0) + 0.25,
        np.arange(-3.0, 4.0) + 0.25,
    )
    z = -10.0 + 0.08 * x - 0.04 * y
    values = 0.45 + 0.03 * x - 0.02 * y
    intensity_map.add_points(
        np.column_stack((x.ravel(), y.ravel(), z.ravel())),
        values.ravel(),
        heading_rad=0.0,
        observation_id=1,
        anchor_id=0,
    )
    matched = np.column_stack(
        (
            x.ravel(),
            y.ravel(),
            z.ravel(),
        )
    )
    displaced = matched.copy()
    displaced[:, 2] += 3.0
    measured = values.ravel()
    estimator = AcousticESIKF(
        ESIKFConfig(intensity_surface_consistency_std=0.5)
    )
    estimator.initialize(0.0)

    matched_variance = estimator._intensity_linearization(
        estimator.state,
        matched,
        measured,
        intensity_map,
        anchor_id=0,
    )[4]
    displaced_variance = estimator._intensity_linearization(
        estimator.state,
        displaced,
        measured,
        intensity_map,
        anchor_id=0,
    )[4]

    assert matched_variance.size > 10
    assert displaced_variance.size == matched_variance.size
    assert np.median(displaced_variance) > 20.0 * np.median(matched_variance)


def test_acoustic_sample_deskew_includes_translation_rotation_and_lever_arm() -> None:
    estimator = AcousticESIKF(
        sensor_translation_body=np.array([0.4, -0.1, 0.0])
    )
    estimator.initialize(
        0.0,
        rotation=np.eye(3),
        velocity=np.array([2.0, 0.0, 0.0]),
    )
    estimator._last_gyro = np.array([0.0, 0.0, 0.5])
    points = np.array([[1.0, 0.0, -3.0], [1.0, 0.0, -3.0]])
    offsets = np.array([0.0, 0.2])

    deskewed = estimator.deskew_sensor_points(points, offsets)

    np.testing.assert_allclose(deskewed[0], points[0], atol=1.0e-15)
    point_body = points[1] + np.array([0.4, -0.1, 0.0])
    expected_body = so3_exp([0.0, 0.0, 0.1]) @ point_body
    expected_body += np.array([0.4, 0.0, 0.0])
    np.testing.assert_allclose(
        deskewed[1], expected_body - np.array([0.4, -0.1, 0.0])
    )


def test_direct_intensity_update_rejects_anti_correlated_strip() -> None:
    intensity_map = AcousticIntensityMap(
        voxel_size=1.5,
        min_cell_points=2,
        min_neighbours=6,
        min_gradient=1.0e-5,
    )
    x, y = np.meshgrid(
        np.linspace(-8.0, 8.0, 65), np.linspace(-28.0, 28.0, 113)
    )
    texture = (
        0.5
        + 0.15 * np.sin(0.31 * x + 0.08 * y)
        + 0.1 * np.cos(0.17 * y)
    )
    intensity_map.add_points(
        np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -10.0))),
        texture.ravel(),
        heading_rad=0.0,
        observation_id=0,
    )
    cross_track = np.linspace(-25.0, 25.0, 101)
    points = np.column_stack(
        (
            np.zeros_like(cross_track),
            cross_track,
            np.full_like(cross_track, -10.0),
        )
    )
    true_line = (
        0.5
        + 0.15 * np.sin(0.08 * cross_track)
        + 0.1 * np.cos(0.17 * cross_track)
    )
    estimator = AcousticESIKF(
        ESIKFConfig(
            intensity_noise_std=0.025,
            intensity_min_correspondences=30,
            intensity_max_points=101,
            intensity_min_yaw_information=0.05,
            intensity_min_normalized_correlation=0.5,
        )
    )
    estimator.initialize(0.0, covariance=np.eye(15) * 1.0e-10)

    result = estimator.update_intensity(
        points, 1.0 - true_line, intensity_map
    )

    assert not result.accepted
    assert result.rejection_reason == "insufficient_correlation"
    assert result.normalized_correlation < 0.0


def test_intensity_map_rejects_recent_or_wrong_view_support() -> None:
    intensity_map = AcousticIntensityMap(
        voxel_size=1.0,
        min_cell_points=1,
        min_neighbours=5,
        min_gradient=1.0e-5,
    )
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 17), np.linspace(-2.0, 2.0, 17))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -8.0)))
    values = 0.5 + 0.04 * x.ravel() - 0.03 * y.ravel()
    intensity_map.add_points(
        points,
        values,
        heading_rad=0.0,
        observation_id=100,
    )

    query = np.array([0.1, -0.1, -8.0])
    assert (
        intensity_map.query(
            query,
            heading_rad=0.0,
            maximum_observation_id=99,
        )
        is None
    )
    assert intensity_map.query(query, heading_rad=np.pi / 2.0) is None
    sample = intensity_map.query(
        query,
        heading_rad=0.0,
        maximum_observation_id=100,
    )
    assert sample is not None
    np.testing.assert_allclose(sample.gradient_xy, [0.04, -0.03], atol=0.01)


def test_planar_pose_graph_closes_a_drifted_square() -> None:
    config = PoseGraphConfig(optimizer_iterations=20)
    graph = PlanarPoseGraph(config)
    raw = (
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, np.pi / 2.0],
        [0.0, 10.0, np.pi],
        [1.5, -0.8, -np.pi / 2.0],
    )
    for pose in raw:
        graph.add_pose(pose)
    information = np.diag([25.0, 25.0, 400.0])
    for index in range(4):
        measurement = np.array([10.0, 0.0, np.pi / 2.0])
        graph.add_edge(index, index + 1, measurement, information, kind="odometry")
    graph.add_edge(0, 4, np.zeros(3), information * 4.0, kind="loop")
    before = np.linalg.norm(graph.poses[-1, :2])
    optimized = graph.optimize()
    assert np.linalg.norm(optimized[-1, :2]) < before
    assert graph.final_cost < graph.initial_cost


def _feature_cloud(offset: float) -> np.ndarray:
    rng = np.random.default_rng(9)
    x = rng.uniform(-3.0, 3.0, 140)
    y = rng.uniform(-5.0, 5.0, 140)
    z = -8.0 + 0.15 * x + 0.08 * y + offset * np.sin(y)
    return np.column_stack((x, y, z))


def test_acoustic_pose_graph_requires_and_accepts_a_loop_sequence() -> None:
    config = PoseGraphConfig(
        keyframe_distance_m=0.0,
        keyframe_max_interval_s=0.0,
        loop_min_separation_s=10.0,
        loop_search_radius_m=20.0,
        loop_confirmation_count=2,
        icp_min_inliers=30,
        ordered_registration_max_rmse_m=0.1,
    )
    backend = AcousticPoseGraph(config)
    cloud_a = _feature_cloud(0.35)
    cloud_b = _feature_cloud(0.75)
    appearance_a = sidescan_descriptor(np.linspace(0, 1, 80), np.linspace(1, 0, 80))
    appearance_b = sidescan_descriptor(np.sin(np.linspace(0, np.pi, 80)), np.zeros(80))
    identity = np.eye(3)

    backend.add_if_needed(0.0, identity, np.array([0.0, 0.0, -5.0]), cloud_a, None, appearance_a)
    backend.add_if_needed(2.0, identity, np.array([10.0, 0.0, -5.0]), cloud_b, None, appearance_b)
    backend.add_if_needed(20.0, identity, np.array([2.0, 0.0, -5.0]), cloud_a, None, appearance_a)
    assert backend.diagnostics().loop_edges == 0
    backend.add_if_needed(22.0, identity, np.array([12.0, 0.0, -5.0]), cloud_b, None, appearance_b)

    assert backend.diagnostics().loop_edges == 2
    _, optimized = backend.optimized_keyframe_positions()
    assert abs(optimized[2, 0]) < 2.0
    assert abs(optimized[3, 0] - 10.0) < 2.0

    # A keyframe added after the optimizer ran must inherit the global
    # correction instead of snapping the reported global pose back to raw
    # odometry until the next loop closure.
    previous_optimized_x = optimized[3, 0]
    backend.add_if_needed(
        24.0,
        identity,
        np.array([20.0, 0.0, -5.0]),
        _feature_cloud(4.0),
        None,
        np.zeros_like(appearance_a),
    )
    _, propagated = backend.optimized_keyframe_positions()
    assert abs(propagated[4, 0] - (previous_optimized_x + 8.0)) < 0.2


def test_acoustic_pose_graph_can_relocalize_outside_odometric_radius() -> None:
    config = PoseGraphConfig(
        keyframe_distance_m=0.0,
        keyframe_max_interval_s=0.0,
        loop_min_separation_s=10.0,
        loop_search_radius_m=2.0,
        loop_global_descriptor_threshold=0.02,
        loop_global_max_candidates=2,
        loop_confirmation_count=2,
        icp_min_inliers=30,
        ordered_registration_max_rmse_m=0.1,
    )
    backend = AcousticPoseGraph(config)
    cloud_a = _feature_cloud(0.35)
    cloud_b = _feature_cloud(0.75)
    appearance_a = sidescan_descriptor(
        np.linspace(0, 1, 80), np.linspace(1, 0, 80)
    )
    appearance_b = sidescan_descriptor(
        np.sin(np.linspace(0, np.pi, 80)), np.zeros(80)
    )
    identity = np.eye(3)

    backend.add_if_needed(
        0.0, identity, np.array([0.0, 0.0, -5.0]), cloud_a, None, appearance_a
    )
    backend.add_if_needed(
        2.0, identity, np.array([10.0, 0.0, -5.0]), cloud_b, None, appearance_b
    )
    # Raw odometry is wrong by 45 m, far outside the 2 m spatial gate.
    backend.add_if_needed(
        20.0, identity, np.array([45.0, 0.0, -5.0]), cloud_a, None, appearance_a
    )
    backend.add_if_needed(
        22.0, identity, np.array([55.0, 0.0, -5.0]), cloud_b, None, appearance_b
    )

    assert backend.diagnostics().loop_edges == 2
    _, optimized = backend.optimized_keyframe_positions()
    assert optimized[2, 0] < 20.0


def test_large_area_sensor_and_slam_smoke() -> None:
    world = LargeUnderwaterWorld()
    survey = FigureEightSurvey()
    sensors = LargeScaleSensorSimulator(world, survey, num_beams=48, seed=5)
    truth = survey.state(0.0)
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(mbes_group_pings=4, bootstrap_pings=8)
    )
    slam.initialize(
        0.0,
        rotation=truth.rotation,
        position=truth.position + np.array([0.1, -0.1, 0.05]),
        velocity=truth.velocity,
    )
    for index in range(1, 401):
        timestamp = index * 0.02
        truth = survey.state(timestamp)
        gyro, acceleration = sensors.imu(truth)
        slam.process_imu(timestamp, gyro, acceleration)
        if index % 10 == 0:
            slam.process_dvl(sensors.dvl(truth))
            ping = sensors.bathymetric_sidescan_ping(truth)
            imaging_points, imaging_values, _ = (
                ping.valid_imaging_samples(frame="sensor")
            )
            slam.process_bathymetric_sidescan(
                ping.bathymetry.points_sensor,
                ping.bathymetry.intensities,
                ping.side_scan.port,
                ping.side_scan.starboard,
                timestamp=timestamp,
                imaging_points_sensor=imaging_points,
                imaging_backscatter=imaging_values,
            )
        if index % 5 == 0:
            slam.process_depth(sensors.depth(truth))

    diagnostics = slam.diagnostics()
    assert (world.width, world.length) == (300.0, 220.0)
    assert diagnostics.map_points > 1_000
    assert diagnostics.sidescan_lines == 40
    assert diagnostics.integrated_sonar_pings == 40
    assert diagnostics.coregistered_intensity_samples > 20_000
    assert diagnostics.legacy_range_attached_intensity_samples == 0
    assert (
        diagnostics.intensity_map_points
        + diagnostics.pending_intensity_points
        > 500
    )
    assert diagnostics.pending_intensity_batches >= 3
    assert diagnostics.formed_intensity_strips >= 3
    assert diagnostics.pose_graph.keyframes >= 3
    assert np.isfinite(slam.state.position).all()
    assert np.linalg.norm(sensors.gyro_bias) <= np.deg2rad(1.0) / 3600.0


def test_quality_gate_keeps_rejected_scan_out_of_reference_map() -> None:
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            mbes_group_pings=1,
            bootstrap_pings=0,
            quality_gate_map_updates=True,
        )
    )
    slam.initialize(0.0)
    grid_x, grid_y = np.meshgrid(
        np.linspace(-4.0, 4.0, 16), np.linspace(-4.0, 4.0, 16)
    )
    seed_points = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, -10.0))
    )
    slam.map.add_points(seed_points)
    stable_points_before = slam.map.point_count
    far_points = np.column_stack(
        (
            np.full(40, 30.0),
            np.linspace(-6.0, 6.0, 40),
            np.full(40, -10.0),
        )
    )

    result = slam.process_mbes(far_points)

    assert result is not None and not result.accepted
    assert result.rejection_reason == "insufficient_correspondences"
    assert slam.map.point_count == stable_points_before
    assert slam.quarantine_map.point_count == far_points.shape[0]
    diagnostics = slam.diagnostics()
    assert diagnostics.quarantined_geometry_groups == 1
    assert diagnostics.sonar_rejection_reasons == {
        "insufficient_correspondences": 1
    }


def test_sidescan_reference_delay_and_motion_compensated_strip() -> None:
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            mbes_group_pings=8,
            intensity_bootstrap_pings=100,
            intensity_min_separation_pings=2,
            intensity_group_pings=3,
            intensity_min_group_span_m=0.5,
        )
    )
    slam.initialize(0.0)
    cross_track = np.linspace(-25.0, 25.0, 48)
    points = np.column_stack(
        (
            np.zeros_like(cross_track),
            cross_track,
            np.full_like(cross_track, -10.0),
        )
    )
    side = np.linspace(0.1, 0.9, 160)

    for ping in range(1, 7):
        slam.filter.state.position[0] = float(ping - 1)
        slam.filter.state.timestamp = float(ping)
        slam.process_sidescan(side, side[::-1], timestamp=float(ping))
        slam.process_mbes(points)

    diagnostics = slam.diagnostics()
    assert diagnostics.committed_intensity_batches == 1
    assert diagnostics.pending_intensity_batches == 1
    assert diagnostics.formed_intensity_strips == 2
    assert diagnostics.skipped_intensity_strips == 0
    assert diagnostics.last_intensity_strip_span_m == 2.0
    assert diagnostics.intensity_map_points == 3 * points.shape[0]
    assert diagnostics.pending_intensity_points == 3 * points.shape[0]


def test_causal_sliding_mbes_window_updates_each_ping_without_self_insertion(
    monkeypatch,
) -> None:
    point_count = 96
    cross_track = np.linspace(-24.0, 24.0, point_count)
    points = np.column_stack(
        (
            0.15 * np.sin(0.2 * cross_track),
            cross_track,
            -10.0 - 0.03 * cross_track,
        )
    )
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            filter=ESIKFConfig(sonar_max_points=64),
            bootstrap_pings=3,
            mbes_group_pings=3,
            mbes_sliding_window_enabled=True,
            map_insert_stride=1,
            intensity_updates_enabled=False,
        )
    )
    slam.initialize(0.0)
    update_sizes: list[int] = []

    def accepted_update(grouped_points, surfel_map, *, information_scale=1.0):
        assert surfel_map.point_count == 3 * point_count + len(update_sizes) * point_count
        assert information_scale == pytest.approx(1.0 / 3.0)
        update_sizes.append(int(grouped_points.shape[0]))
        return SonarUpdateResult(
            accepted=True,
            correspondences=grouped_points.shape[0],
            inliers=grouped_points.shape[0],
            iterations=1,
            rmse_before=0.2,
            rmse_after=0.1,
            minimum_translation_eigenvalue=1.0,
            information_condition=2.0,
            adaptive_weight=1.0,
            rotation_observable_rank=3,
            translation_observable_rank=3,
            rotation_correction_deg=0.0,
            translation_correction=0.0,
        )

    monkeypatch.setattr(slam.filter, "update_sonar", accepted_update)
    for ping_index in range(1, 8):
        slam.filter.state.timestamp = float(ping_index)
        slam.filter.state.position[0] = float(ping_index - 1)
        slam.process_mbes(points)

    diagnostics = slam.diagnostics()
    # Three seed pings form the immutable reference. Pings four and five are
    # withheld; ping six aligns the 4/5/6 window and only then commits ping 4.
    assert update_sizes == [64, 64]
    assert diagnostics.accepted_sonar_updates == 2
    assert diagnostics.committed_geometry_groups == 5
    assert diagnostics.map_points == 5 * point_count
    assert len(slam._mbes_group) == 2


def test_single_v2_ping_uses_its_simultaneous_row_aperture() -> None:
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            mbes_group_pings=1,
            intensity_bootstrap_pings=100,
            intensity_min_separation_pings=1,
            intensity_group_pings=1,
            intensity_min_group_span_m=1.0,
        )
    )
    slam.initialize(0.0)
    row_centres = np.linspace(-0.525, 0.525, 8)
    cross_track = np.concatenate(
        (np.linspace(-20.0, -1.0, 24), np.linspace(1.0, 20.0, 24))
    )
    points = np.concatenate(
        [
            np.column_stack(
                (
                    np.full(cross_track.size, row),
                    cross_track,
                    np.full(cross_track.size, -10.0),
                )
            )
            for row in row_centres
        ]
    )
    values = np.linspace(0.1, 0.9, points.shape[0])
    side = np.linspace(0.1, 0.9, 160)
    slam.filter.state.timestamp = 1.0

    slam.process_bathymetric_sidescan(
        points,
        values,
        side,
        side[::-1],
        timestamp=1.0,
        along_track_coverage_m=1.2,
    )

    diagnostics = slam.diagnostics()
    assert diagnostics.formed_intensity_strips == 1
    assert diagnostics.skipped_intensity_strips == 0
    assert diagnostics.last_intensity_strip_span_m == pytest.approx(1.2)
    assert diagnostics.pending_intensity_batches == 1


def test_process_integrated_ping_uses_one_decimated_xyzi_array() -> None:
    world = LargeUnderwaterWorld()
    survey = FigureEightSurvey()
    sensors = LargeScaleSensorSimulator(
        world,
        survey,
        num_beams=12,
        imaging_rays_per_side=48,
        sidescan_range_bins=80,
        seed=47,
    )
    truth = survey.state(0.0)
    ping = sensors.bathymetric_sidescan_ping(truth)
    ping.along_track_coverage_m = 1.2
    acoustic_points, _, _ = ping.valid_acoustic_samples()
    sample_time_offsets = ping.valid_acoustic_sample_time_offsets()
    assert sample_time_offsets.shape == (acoustic_points.shape[0],)
    np.testing.assert_array_equal(sample_time_offsets, 0.0)
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            mbes_group_pings=1,
            intensity_group_pings=1,
            intensity_min_group_span_m=1.0,
            intensity_bootstrap_pings=100,
        )
    )
    slam.initialize(
        0.0,
        rotation=truth.rotation,
        position=truth.position,
        velocity=truth.velocity,
    )

    slam.process_integrated_ping(ping, sample_stride=2)

    expected_samples = acoustic_points[::2].shape[0]
    diagnostics = slam.diagnostics()
    assert diagnostics.integrated_sonar_pings == 1
    assert diagnostics.coregistered_intensity_samples == expected_samples
    assert diagnostics.total_geometry_points_committed == expected_samples
    assert diagnostics.pending_intensity_points == expected_samples


def test_reference_delay_is_measured_from_oldest_strip_line(monkeypatch) -> None:
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            mbes_group_pings=8,
            intensity_bootstrap_pings=0,
            intensity_min_separation_pings=2,
            intensity_group_pings=3,
            intensity_min_group_span_m=0.5,
        )
    )
    slam.initialize(0.0)
    cross_track = np.linspace(-25.0, 25.0, 48)
    points = np.column_stack(
        (
            np.zeros_like(cross_track),
            cross_track,
            np.full_like(cross_track, -10.0),
        )
    )
    side = np.linspace(0.1, 0.9, 160)
    # Only map non-emptiness matters because the direct matcher is replaced
    # below; this isolates the temporal cutoff from association behaviour.
    slam.intensity_map.add_points(
        np.array([[0.0, 0.0, -10.0]]),
        np.array([0.5]),
        heading_rad=0.0,
        observation_id=-100,
        anchor_id=-1,
    )
    observed_cutoffs: list[tuple[int, int]] = []

    def rejected_update(*args, **kwargs) -> IntensityUpdateResult:
        del args
        observed_cutoffs.append(
            (slam._ping_count, int(kwargs["maximum_observation_id"]))
        )
        return IntensityUpdateResult(
            accepted=False,
            correspondences=0,
            inliers=0,
            iterations=0,
            rmse_before=np.inf,
            rmse_after=np.inf,
            yaw_information=0.0,
            overlap_ratio=0.0,
            normalized_correlation=0.0,
            planar_observable_rank=0,
            gain=1.0,
            offset=0.0,
            rotation_correction_deg=0.0,
            translation_correction=0.0,
            rejection_reason="test_rejection",
        )

    monkeypatch.setattr(slam.filter, "update_intensity", rejected_update)
    for ping in range(1, 7):
        slam.filter.state.position[0] = float(ping - 1)
        slam.filter.state.timestamp = float(ping)
        slam.process_sidescan(side, side[::-1], timestamp=float(ping))
        slam.process_mbes(points)

    # The strips contain ids 1,2,3 and 4,5,6. A two-ping delay must therefore
    # cut off at -1 and 2 respectively, not at current-id minus delay (1,4).
    assert observed_cutoffs == [(3, -1), (6, 2)]


def _stub_intensity_result(
    *, accepted: bool, anchor: int | None, reason: str | None = None
) -> IntensityUpdateResult:
    return IntensityUpdateResult(
        accepted=accepted,
        correspondences=48 if accepted else 0,
        inliers=48 if accepted else 0,
        iterations=1,
        rmse_before=0.1,
        rmse_after=0.05 if accepted else np.inf,
        yaw_information=20.0 if accepted else 0.0,
        overlap_ratio=0.8 if accepted else 0.0,
        normalized_correlation=0.9 if accepted else 0.0,
        planar_observable_rank=3 if accepted else 0,
        gain=1.0,
        offset=0.0,
        rotation_correction_deg=0.01 if accepted else 0.0,
        translation_correction=0.01 if accepted else 0.0,
        rejection_reason=reason,
        anchor_keyframe_index=anchor,
        photometric_pose2=np.zeros(3),
        information_pose2=np.eye(3),
    )


def _run_stubbed_intensity_strip(slam: FastSonarSLAM) -> None:
    cross_track = np.concatenate(
        (np.linspace(-20.0, -1.0, 24), np.linspace(1.0, 20.0, 24))
    )
    points = np.column_stack(
        (
            np.zeros_like(cross_track),
            cross_track,
            np.full_like(cross_track, -10.0),
        )
    )
    values = np.linspace(0.1, 0.9, points.shape[0])
    for observation_id, x_position in ((2, 0.0), (3, 1.0)):
        slam._ping_count = observation_id
        slam.filter.state.timestamp = float(observation_id)
        slam.filter.state.position[0] = x_position
        slam._process_sidescan_intensity(points, values)


def _make_stubbed_intensity_slam() -> FastSonarSLAM:
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            intensity_bootstrap_pings=0,
            intensity_min_separation_pings=1,
            intensity_group_pings=2,
            intensity_min_group_span_m=0.5,
            quality_gate_map_updates=True,
        )
    )
    slam.initialize(0.0)
    slam.intensity_map.add_points(
        np.array([[0.0, 0.0, -10.0]]),
        np.array([0.5]),
        heading_rad=0.0,
        observation_id=-100,
        anchor_id=0,
    )
    return slam


def test_graph_seeded_match_counts_only_after_factor_acceptance(monkeypatch) -> None:
    slam = _make_stubbed_intensity_slam()
    slam.pose_graph.loop_edges.append(
        PoseGraphEdge(0, 1, np.zeros(3), np.eye(3), kind="loop")
    )
    local_rejected = _stub_intensity_result(
        accepted=False, anchor=None, reason="insufficient_correspondences"
    )
    candidate_one = _stub_intensity_result(accepted=True, anchor=0)
    candidate_two = _stub_intensity_result(accepted=True, anchor=1)
    results = iter((local_rejected, candidate_one, candidate_two))
    monkeypatch.setattr(
        slam.filter, "update_intensity", lambda *args, **kwargs: next(results)
    )
    monkeypatch.setattr(
        slam.pose_graph,
        "intensity_seed_candidates",
        lambda **kwargs: [
            (0, np.array([0.0, 0.0, 0.0])),
            (1, np.array([0.1, 0.0, 0.0])),
        ],
    )
    factor_results = iter((False, True))
    factor_calls: list[int] = []

    def add_factor(**kwargs) -> bool:
        factor_calls.append(int(kwargs["anchor_keyframe_index"]))
        return next(factor_results)

    monkeypatch.setattr(
        slam.pose_graph, "add_intensity_observation", add_factor
    )
    _run_stubbed_intensity_strip(slam)

    diagnostics = slam.diagnostics()
    assert factor_calls == [0, 1]
    assert diagnostics.accepted_intensity_updates == 1
    assert diagnostics.accepted_graph_seeded_intensity_updates == 1
    assert diagnostics.accepted_local_intensity_updates == 0
    assert diagnostics.rejected_intensity_updates == 0
    assert slam._last_intensity_update is candidate_two


def test_rejected_graph_seed_does_not_count_or_trust_intensity(monkeypatch) -> None:
    slam = _make_stubbed_intensity_slam()
    slam._geometry_pose_trusted = False
    slam.pose_graph.loop_edges.append(
        PoseGraphEdge(0, 1, np.zeros(3), np.eye(3), kind="loop")
    )
    local_rejected = _stub_intensity_result(
        accepted=False, anchor=None, reason="insufficient_correspondences"
    )
    candidate = _stub_intensity_result(accepted=True, anchor=0)
    results = iter((local_rejected, candidate))
    monkeypatch.setattr(
        slam.filter, "update_intensity", lambda *args, **kwargs: next(results)
    )
    monkeypatch.setattr(
        slam.pose_graph,
        "intensity_seed_candidates",
        lambda **kwargs: [(0, np.zeros(3))],
    )
    monkeypatch.setattr(
        slam.pose_graph,
        "add_intensity_observation",
        lambda **kwargs: False,
    )
    _run_stubbed_intensity_strip(slam)

    diagnostics = slam.diagnostics()
    assert diagnostics.accepted_intensity_updates == 0
    assert diagnostics.accepted_graph_seeded_intensity_updates == 0
    assert diagnostics.rejected_intensity_updates == 1
    assert diagnostics.intensity_rejection_reasons == {
        "graph_factor_rejected": 1
    }
    assert diagnostics.committed_intensity_batches == 0
    assert diagnostics.quarantined_intensity_batches == 1
    assert slam._last_intensity_update is not None
    assert not slam._last_intensity_update.accepted


def test_local_match_without_graph_factor_is_counted_and_absorbed_once(
    monkeypatch,
) -> None:
    slam = _make_stubbed_intensity_slam()
    local_result = _stub_intensity_result(accepted=True, anchor=0)
    monkeypatch.setattr(
        slam.filter,
        "update_intensity",
        lambda *args, **kwargs: local_result,
    )
    monkeypatch.setattr(
        slam.pose_graph,
        "add_intensity_observation",
        lambda **kwargs: False,
    )
    absorbed: list[int] = []

    def absorb(**kwargs) -> bool:
        absorbed.append(int(kwargs["observation_id"]))
        return True

    monkeypatch.setattr(
        slam.pose_graph, "absorb_latest_local_posterior", absorb
    )
    _run_stubbed_intensity_strip(slam)

    diagnostics = slam.diagnostics()
    assert absorbed == [3]
    assert diagnostics.accepted_intensity_updates == 1
    assert diagnostics.accepted_local_intensity_updates == 1
    assert diagnostics.accepted_local_intensity_updates_with_graph_factor == 0
    assert diagnostics.accepted_local_intensity_updates_without_graph_factor == 1
    assert diagnostics.accepted_graph_seeded_intensity_updates == 0


def test_integrated_sonar_uses_point_registered_backscatter_not_image_lookup() -> None:
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            mbes_group_pings=2,
            intensity_bootstrap_pings=100,
            intensity_min_separation_pings=10,
            intensity_group_pings=2,
            intensity_min_group_span_m=0.5,
        )
    )
    slam.initialize(0.0)
    cross_track = np.concatenate(
        (np.linspace(-20.0, -1.0, 24), np.linspace(1.0, 20.0, 24))
    )
    points = np.column_stack(
        (
            np.zeros_like(cross_track),
            cross_track,
            np.full_like(cross_track, -12.0),
        )
    )
    backscatter = np.linspace(0.1, 0.9, points.shape[0])
    # Deliberately blank image rows prove that direct fusion is using the
    # point-wise co-registered values, not reattaching pixels by slant range.
    blank = np.zeros(160)
    for ping in range(1, 3):
        slam.filter.state.position[0] = float(ping - 1)
        slam.filter.state.timestamp = float(ping)
        slam.process_bathymetric_sidescan(
            points,
            backscatter,
            blank,
            blank,
            timestamp=float(ping),
        )

    diagnostics = slam.diagnostics()
    assert diagnostics.integrated_sonar_pings == 2
    assert diagnostics.sidescan_lines == 2
    assert diagnostics.coregistered_intensity_samples == 2 * points.shape[0]
    assert diagnostics.legacy_range_attached_intensity_samples == 0
    assert diagnostics.formed_intensity_strips == 1
    assert diagnostics.pending_intensity_points == 2 * points.shape[0]

    with pytest.raises(ValueError, match="must match the propagated state"):
        slam.process_bathymetric_sidescan(
            points,
            backscatter,
            blank,
            blank,
            timestamp=slam.state.timestamp + 0.1,
        )


def test_measured_bathymetry_and_terrain_following_survey() -> None:
    root = Path(__file__).resolve().parents[1]
    world = MeasuredBathymetryGrid.from_npz(
        root / "data/monterey_canyon_800x600_5m.npz"
    )
    assert (world.width, world.length) == (800.0, 600.0)
    assert world.elevation_m.shape == (121, 161)
    assert 72.0 < np.ptp(world.elevation_m) < 74.0

    mesh_points, mesh_indices = world.triangle_mesh()
    assert mesh_points.shape == (121 * 161, 3)
    assert mesh_indices.shape == (120 * 160 * 6,)
    assert mesh_points.dtype == np.float32
    assert mesh_indices.dtype == np.int32
    first_triangle = mesh_points[mesh_indices[:3]]
    first_normal = np.cross(
        first_triangle[1] - first_triangle[0],
        first_triangle[2] - first_triangle[0],
    )
    assert first_normal[2] > 0.0

    survey = TerrainFollowingSurvey(world)
    initial = survey.state(0.0)
    repeated = survey.state(survey.period)
    np.testing.assert_allclose(repeated.position, initial.position, atol=1.0e-8)
    np.testing.assert_allclose(repeated.rotation, initial.rotation, atol=1.0e-8)
    altitude = initial.position[2] - world.seabed_height(*initial.position[:2])
    assert 20.0 < altitude < 35.0

    directions = np.array(
        [[0.0, 0.0, -1.0], [0.0, 0.5, -0.866], [0.0, -0.5, -0.866]]
    )
    hit, distances, points, incidence = world.intersect_rays(
        initial.position,
        directions @ initial.rotation.T,
        min_range=1.0,
        max_range=140.0,
    )
    assert np.all(hit)
    assert np.isfinite(distances).all()
    assert np.isfinite(points).all()
    assert np.all(incidence > 0.0)

    constant_speed = TerrainFollowingSurvey(world, speed_mps=1.5)
    sample_times = np.linspace(
        0.0, constant_speed.period, 101, endpoint=False
    )
    speeds = np.asarray(
        [
            np.linalg.norm(constant_speed.kinematics(float(timestamp))[1])
            for timestamp in sample_times
        ]
    )
    np.testing.assert_allclose(speeds, 1.5, atol=2.0e-3)
    np.testing.assert_allclose(
        constant_speed.state(constant_speed.period).position,
        constant_speed.state(0.0).position,
        atol=1.0e-8,
    )
