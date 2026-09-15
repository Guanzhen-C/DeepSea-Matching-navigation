from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from oceansim.slam import AcousticESIKF, FeatureSonarOdometryConfig
from oceansim.slam.feature_sonar_odometry import (
    _FeatureFrame,
    _PairEstimate,
    _estimate_planar_rigid_ransac,
    _estimate_prediction_centered_ransac,
    _gravity_yaw_correction,
    _relative_covariance,
    FeatureSonarOdometry,
)
from oceansim.slam.math3d import rotation_from_rpy, rpy_from_rotation


def _apply_planar(points: np.ndarray, pose: np.ndarray, z: float) -> np.ndarray:
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    output = np.asarray(points, dtype=np.float64).copy()
    output[:, :2] = output[:, :2] @ rotation.T + pose[:2]
    output[:, 2] += z
    return output


def _synthetic_integrated_ping(vehicle_x: float, timestamp: float):
    """Eight-row, co-registered XYZ/intensity ping with 75% overlap."""

    rows = 8
    rays = 256
    ground_range = np.linspace(1.0, 245.0, rays)
    row_origins = (np.arange(rows) - 3.5) * 0.15
    records = []
    for side in (1.0, -1.0):
        for row, origin_x in enumerate(row_origins):
            y = side * ground_range
            ranges = np.hypot(ground_range, 40.0)
            hits = ranges <= 250.0
            points = np.column_stack(
                (
                    np.full(rays, origin_x),
                    y,
                    np.full(rays, -40.0),
                )
            )
            directions = points - np.array([origin_x, 0.0, 0.0])
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            world_x = vehicle_x + origin_x
            texture = (
                0.42
                + 0.15 * np.sin(0.7 * world_x + 0.035 * y)
                + 0.13 * np.cos(0.31 * world_x - 0.11 * y)
                + 0.12 * np.sin(0.018 * y * y + 0.9 * world_x)
                + 0.08 * np.cos(0.23 * y)
            )
            records.append(
                (row, ranges, np.clip(texture, 0.01, 1.0), hits, points, directions)
            )
    imaging = SimpleNamespace(
        along_track_row_indices=np.concatenate(
            [np.full(rays, record[0]) for record in records]
        ),
        intensities=np.concatenate([record[2] for record in records]),
        points_sensor=np.concatenate([record[4] for record in records]),
        beam_directions=np.concatenate([record[5] for record in records]),
        hit_mask=np.concatenate([record[3] for record in records]),
    )
    return SimpleNamespace(
        imaging=imaging,
        side_scan=SimpleNamespace(
            range_bin_centres=(np.arange(12_500) + 0.5) * 0.02
        ),
        timestamp=timestamp,
    )


def test_feature_xyz_ransac_recovers_planar_relative_pose_with_outliers() -> None:
    rng = np.random.default_rng(17)
    current = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, 80),
            rng.uniform(-80.0, 80.0, 80),
            rng.normal(-40.0, 2.0, 80),
        )
    )
    truth = np.array([0.31, -0.08, np.deg2rad(1.1)])
    previous = _apply_planar(current, truth, 0.04)
    previous += rng.normal(0.0, 0.025, previous.shape)
    previous[:18] = rng.uniform([-3.0, -90.0, -47.0], [3.0, 90.0, -33.0], (18, 3))

    transform, rmse, inliers = _estimate_planar_rigid_ransac(
        current,
        previous,
        iterations=400,
        residual_limit_m=0.18,
        minimum_baseline_m=1.0,
        random_seed=9,
    )

    assert transform is not None
    recovered = np.array(
        [
            transform[0, 3],
            transform[1, 3],
            math.atan2(transform[1, 0], transform[0, 0]),
        ]
    )
    np.testing.assert_allclose(recovered[:2], truth[:2], atol=0.02)
    assert abs(recovered[2] - truth[2]) < 3.0e-4
    assert np.count_nonzero(inliers) >= 60
    assert rmse < 0.08

    covariance, eigenvalues = _relative_covariance(
        current[inliers],
        transform,
        rmse,
        position_floor_m=0.2,
        yaw_floor_rad=np.deg2rad(0.3),
    )
    assert covariance is not None
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    assert eigenvalues[0] > 0.0


def test_prediction_centered_fit_preserves_roll_pitch_and_corrects_gravity_yaw(
) -> None:
    rng = np.random.default_rng(71)
    current = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, 90),
            rng.uniform(-110.0, 110.0, 90),
            rng.normal(-40.0, 3.0, 90),
        )
    )
    previous_world_sensor_rotation = rotation_from_rpy(0.11, -0.08, 0.6)
    gravity_axis = previous_world_sensor_rotation.T @ np.array([0.0, 0.0, 1.0])
    predicted = np.eye(4)
    predicted[:3, :3] = rotation_from_rpy(0.018, -0.014, 0.006)
    predicted[:3, 3] = [0.29, 0.015, -0.012]
    correction = np.array([0.035, -0.025, np.deg2rad(0.22)])
    expected = _gravity_yaw_correction(correction, gravity_axis) @ predicted
    previous = current @ expected[:3, :3].T + expected[:3, 3]
    previous += rng.normal(0.0, 0.004, previous.shape)
    previous[:15] = rng.uniform(
        [-2.0, -120.0, -48.0], [2.0, 120.0, -32.0], (15, 3)
    )

    transform, rmse, inliers = _estimate_prediction_centered_ransac(
        current,
        previous,
        predicted,
        gravity_axis,
        iterations=400,
        residual_limit_m=0.12,
        minimum_baseline_m=1.0,
        random_seed=13,
        along_track_weight=10.0,
        vertical_weight=2.0,
    )

    assert transform is not None
    np.testing.assert_allclose(transform[:3, 3], expected[:3, 3], atol=0.012)
    rotation_error = transform[:3, :3] @ expected[:3, :3].T
    angle_error = math.acos(
        float(np.clip((np.trace(rotation_error) - 1.0) * 0.5, -1.0, 1.0))
    )
    assert angle_error < np.deg2rad(0.02)
    assert np.count_nonzero(inliers) >= 70
    assert rmse < 0.06


def test_cuda_prediction_consensus_matches_cpu_result() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(107)
    current = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, 180),
            rng.uniform(-115.0, 115.0, 180),
            rng.normal(-40.0, 3.0, 180),
        )
    )
    rotation_world_sensor = rotation_from_rpy(0.09, -0.07, 0.5)
    gravity_axis = rotation_world_sensor.T @ np.array([0.0, 0.0, 1.0])
    predicted = np.eye(4)
    predicted[:3, :3] = rotation_from_rpy(0.015, -0.011, 0.004)
    predicted[:3, 3] = [0.30, 0.012, -0.009]
    expected = (
        _gravity_yaw_correction(
            np.array([0.032, -0.021, np.deg2rad(0.18)]), gravity_axis
        )
        @ predicted
    )
    previous = current @ expected[:3, :3].T + expected[:3, 3]
    previous += rng.normal(0.0, 0.004, previous.shape)
    previous[:35] = rng.uniform(
        [-2.0, -120.0, -48.0], [2.0, 120.0, -32.0], (35, 3)
    )
    arguments = dict(
        iterations=256,
        residual_limit_m=0.12,
        minimum_baseline_m=1.0,
        random_seed=19,
        along_track_weight=10.0,
        vertical_weight=2.0,
    )

    cpu_transform, cpu_rmse, cpu_inliers = (
        _estimate_prediction_centered_ransac(
            current,
            previous,
            predicted,
            gravity_axis,
            compute_backend="cpu",
            **arguments,
        )
    )
    cuda_transform, cuda_rmse, cuda_inliers = (
        _estimate_prediction_centered_ransac(
            current,
            previous,
            predicted,
            gravity_axis,
            compute_backend="cuda",
            torch_module=torch,
            **arguments,
        )
    )

    np.testing.assert_array_equal(cuda_inliers, cpu_inliers)
    np.testing.assert_allclose(cuda_transform, cpu_transform, atol=1.0e-12)
    assert cuda_rmse == pytest.approx(cpu_rmse, abs=1.0e-14)


def test_feature_compute_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="compute_backend"):
        FeatureSonarOdometry(
            FeatureSonarOdometryConfig(compute_backend="unknown")
        )


def test_feature_matcher_uses_mutual_descriptors_then_xyz_geometry() -> None:
    rng = np.random.default_rng(5)
    count = 40
    current_points = np.column_stack(
        (
            rng.uniform(-0.6, 0.6, count),
            rng.uniform(-70.0, 70.0, count),
            rng.normal(-35.0, 1.5, count),
        )
    )
    pose = np.array([0.3, 0.02, np.deg2rad(0.4)])
    previous_points = _apply_planar(current_points, pose, 0.0)
    descriptors = rng.normal(size=(count, 128)).astype(np.float32)
    previous = _FeatureFrame(
        0.0,
        np.zeros((32, 128), dtype=np.uint8),
        np.zeros((count, 2)),
        previous_points,
        descriptors,
    )
    current = _FeatureFrame(
        0.2,
        np.zeros((32, 128), dtype=np.uint8),
        np.zeros((count, 2)),
        current_points,
        descriptors + rng.normal(0.0, 1.0e-4, descriptors.shape),
    )
    estimator = FeatureSonarOdometry(
        FeatureSonarOdometryConfig(
            minimum_matches=8,
            ransac_residual_m=0.1,
        )
    )

    result = estimator._estimate_pair(previous, current)

    assert result.accepted
    assert result.matches == count
    assert result.inliers == count
    recovered = result.transform_previous_sensor_current_sensor
    np.testing.assert_allclose(recovered[:2, 3], pose[:2], atol=1.0e-8)
    assert math.isclose(
        math.atan2(recovered[1, 0], recovered[0, 0]),
        pose[2],
        abs_tol=1.0e-10,
    )


def test_eight_row_canonical_images_recover_adjacent_ping_advance() -> None:
    estimator = FeatureSonarOdometry(FeatureSonarOdometryConfig())
    previous = estimator._canonical_frame(
        _synthetic_integrated_ping(0.0, 0.0)
    )
    current = estimator._canonical_frame(
        _synthetic_integrated_ping(0.3, 0.2)
    )

    result = estimator._estimate_pair(previous, current)

    assert result.accepted
    assert result.matches >= 40
    assert result.inliers >= 30
    transform = result.transform_previous_sensor_current_sensor
    np.testing.assert_allclose(transform[:2, 3], [0.3, 0.0], atol=0.03)
    assert abs(math.atan2(transform[1, 0], transform[0, 0])) < np.deg2rad(0.05)


def test_klt_with_motion_prior_recovers_eight_row_advance() -> None:
    estimator = FeatureSonarOdometry(FeatureSonarOdometryConfig())
    previous = estimator._canonical_frame(
        _synthetic_integrated_ping(0.0, 0.0)
    )
    current = estimator._canonical_frame(
        _synthetic_integrated_ping(0.3, 0.2)
    )
    previous.rotation_world_sensor = np.eye(3)
    current.rotation_world_sensor = np.eye(3)
    predicted = np.eye(4)
    predicted[0, 3] = 0.29

    result = estimator._estimate_pair(previous, current, predicted)

    assert result.accepted
    assert result.matches >= 24
    assert result.inliers >= 24
    transform = result.transform_previous_sensor_current_sensor
    assert abs(transform[0, 3] - 0.3) < 0.04
    # The one-dimensional update policy retains predicted nuisance motion.
    assert transform[1, 3] == 0.0
    assert math.atan2(transform[1, 0], transform[0, 0]) == 0.0


def test_multi_ping_chain_emits_one_relative_factor_without_reusing_updates(
    monkeypatch,
) -> None:
    estimator = FeatureSonarOdometry(
        FeatureSonarOdometryConfig(subframe_pings=4)
    )
    frames = [
        _FeatureFrame(
            0.2 * index,
            np.zeros((32, 64), dtype=np.uint8),
            np.zeros((12, 2)),
            np.zeros((12, 3)),
            np.zeros((12, 128), dtype=np.float32),
        )
        for index in range(4)
    ]
    frame_iterator = iter(frames)
    monkeypatch.setattr(estimator, "_canonical_frame", lambda ping: next(frame_iterator))
    pair_transform = np.eye(4)
    pair_transform[0, 3] = 0.3
    pair_covariance = np.diag([0.04, 0.04, np.deg2rad(0.3) ** 2])
    monkeypatch.setattr(
        estimator,
        "_estimate_pair",
        lambda previous, current, predicted: _PairEstimate(
            True,
            None,
            24,
            21,
            0.12,
            pair_transform.copy(),
            pair_covariance.copy(),
        ),
    )
    rotation = np.eye(3)
    covariance = np.eye(15) * 1.0e-4
    results = []
    for index in range(4):
        result = estimator.process(
            object(),
            rotation_world_body=rotation,
            position_world_body=np.array([0.3 * index, 0.0, -40.0]),
            state_covariance=covariance,
        )
        results.append(result)
        estimator.finalize(
            rotation_world_body=rotation,
            position_world_body=np.array([0.3 * index, 0.0, -40.0]),
            state_covariance=covariance,
        )

    assert sum(result.subframe_ready for result in results) == 1
    np.testing.assert_allclose(results[-1].measured_position_xy_world, [0.9, 0.0])
    diagnostics = estimator.diagnostics()
    assert diagnostics.frames == 4
    assert diagnostics.pair_attempts == 3
    assert diagnostics.accepted_pairs == 3
    assert diagnostics.completed_subframes == 1


def test_planar_pose_update_reduces_xy_yaw_without_changing_depth() -> None:
    estimator = AcousticESIKF()
    estimator.initialize(
        0.0,
        rotation=rotation_from_rpy(0.08, -0.05, 0.12),
        position=np.array([2.0, -1.5, -42.0]),
        covariance=np.eye(15) * 0.5,
    )
    depth_before = float(estimator.state.position[2])
    error_before = np.linalg.norm(estimator.state.position[:2])
    yaw_before = abs(float(rpy_from_rotation(estimator.state.rotation)[2]))

    assert estimator.update_planar_pose(
        np.zeros(2),
        0.0,
        np.diag([0.04, 0.04, np.deg2rad(0.5) ** 2]),
    )

    assert np.linalg.norm(estimator.state.position[:2]) < error_before
    assert abs(float(rpy_from_rotation(estimator.state.rotation)[2])) < yaw_before
    assert estimator.state.position[2] == depth_before


def test_along_axis_update_cannot_change_cross_track_depth_or_attitude() -> None:
    estimator = AcousticESIKF()
    estimator.initialize(
        0.0,
        rotation=rotation_from_rpy(0.08, -0.05, 0.12),
        position=np.array([2.0, -1.5, -42.0]),
        velocity=np.array([0.4, -0.2, 0.1]),
        covariance=np.eye(15) * 0.5,
    )
    rotation_before = estimator.state.rotation.copy()
    y_before = float(estimator.state.position[1])
    depth_before = float(estimator.state.position[2])

    assert estimator.update_horizontal_position_along_axis(
        0.0, np.array([1.0, 0.0]), 0.04
    )

    assert abs(estimator.state.position[0]) < 2.0
    assert estimator.state.position[1] == y_before
    assert estimator.state.position[2] == depth_before
    np.testing.assert_allclose(
        estimator.state.rotation, rotation_before, atol=5.0e-15, rtol=0.0
    )
