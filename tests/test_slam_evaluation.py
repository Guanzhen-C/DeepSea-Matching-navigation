import numpy as np
import pytest

from oceansim.slam import (
    map_to_odom_change_metrics,
    se2_relative_pose_error_by_distance,
    se3_relative_pose_error_by_distance,
)
from oceansim.slam.math3d import rotation_from_rpy


def test_map_to_odom_audit_is_zero_for_identical_frames() -> None:
    times = np.array([0.0, 0.1, 0.2])
    positions = np.array([[0.0, 0.0, -5.0], [0.1, 0.0, -5.0], [0.2, 0.0, -5.0]])
    rotations = np.repeat(np.eye(3)[None, :, :], 3, axis=0)

    metrics = map_to_odom_change_metrics(
        times, positions, rotations, positions, rotations
    )

    assert metrics["nonfinite_samples"] == 0
    assert metrics["maximum_translation_change_m"] == pytest.approx(0.0)
    assert metrics["maximum_yaw_change_deg"] == pytest.approx(0.0)
    assert metrics["maximum_global_position_step_m"] == pytest.approx(0.1)


def test_map_to_odom_audit_recovers_translation_and_yaw_changes() -> None:
    times = np.array([0.0, 1.0, 2.0])
    local = np.array([[0.0, 0.0, -5.0], [1.0, 0.0, -5.0], [2.0, 0.0, -5.0]])
    local_rotations = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    correction_yaws = np.deg2rad([0.0, 2.0, 2.5])
    translations = np.array([[0.0, 0.0], [6.0, 8.0], [6.3, 8.4]])
    global_rotations = np.array(
        [rotation_from_rpy(0.0, 0.0, yaw) for yaw in correction_yaws]
    )
    global_positions = local.copy()
    for index, yaw in enumerate(correction_yaws):
        rotation = global_rotations[index, :2, :2]
        global_positions[index, :2] = (
            rotation @ local[index, :2] + translations[index]
        )

    metrics = map_to_odom_change_metrics(
        times,
        local,
        local_rotations,
        global_positions,
        global_rotations,
    )

    assert metrics["maximum_translation_change_m"] == pytest.approx(10.0)
    assert metrics["maximum_yaw_change_deg"] == pytest.approx(2.0)
    assert metrics["translation_changes_over_5m"] == 1
    assert metrics["yaw_changes_over_1deg"] == 1
    assert metrics["maximum_change_timestamp_s"] == pytest.approx(1.0)


def test_map_to_odom_audit_reports_nonfinite_samples_without_masking_them() -> None:
    times = np.array([0.0, 1.0])
    local = np.zeros((2, 3))
    global_positions = local.copy()
    global_positions[1, 0] = np.nan
    rotations = np.repeat(np.eye(3)[None, :, :], 2, axis=0)

    metrics = map_to_odom_change_metrics(
        times, local, rotations, global_positions, rotations
    )

    assert metrics["nonfinite_samples"] == 1
    assert metrics["maximum_translation_change_m"] is None


def test_map_to_odom_audit_rejects_mismatched_track_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        map_to_odom_change_metrics(
            np.array([0.0, 1.0]),
            np.zeros((1, 3)),
            np.eye(3)[None, :, :],
            np.zeros((1, 3)),
            np.eye(3)[None, :, :],
        )


def test_se3_rpe_removes_one_constant_global_gauge() -> None:
    truth_positions = np.array(
        [[0.0, 0.0, -5.0], [50.0, 0.0, -5.0], [100.0, 0.0, -5.0]]
    )
    truth_rotations = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    gauge = rotation_from_rpy(0.0, 0.0, np.deg2rad(50.0))
    estimate_positions = truth_positions @ gauge.T + np.array(
        [400.0, -300.0, 20.0]
    )
    estimate_rotations = np.einsum(
        "ij,njk->nik", gauge, truth_rotations
    )

    metrics = se3_relative_pose_error_by_distance(
        estimate_positions,
        estimate_rotations,
        truth_positions,
        truth_rotations,
        np.array([0.0, 50.0, 100.0]),
        100.0,
    )

    assert metrics["available"]
    assert metrics["samples"] == 1
    assert metrics["rmse_translation_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["rmse_rotation_deg"] == pytest.approx(0.0, abs=1.0e-12)


def test_se2_rpe_removes_one_constant_global_gauge() -> None:
    truth = np.array(
        [[0.0, 0.0, 0.0], [40.0, 0.0, 0.1], [99.0, 10.0, 0.2]]
    )
    gauge_yaw = np.deg2rad(50.0)
    gauge_rotation = np.array(
        [
            [np.cos(gauge_yaw), -np.sin(gauge_yaw)],
            [np.sin(gauge_yaw), np.cos(gauge_yaw)],
        ]
    )
    estimate = truth.copy()
    estimate[:, :2] = truth[:, :2] @ gauge_rotation.T + np.array(
        [400.0, -300.0]
    )
    estimate[:, 2] = truth[:, 2] + gauge_yaw

    metrics = se2_relative_pose_error_by_distance(
        estimate, truth, np.array([0.0, 40.0, 100.0]), 100.0
    )

    assert metrics["samples"] == 1
    assert metrics["rmse_translation_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["rmse_yaw_deg"] == pytest.approx(0.0, abs=1.0e-12)


def test_se2_rpe_reports_relative_translation_and_yaw_error() -> None:
    truth = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    estimate = np.array([[0.0, 0.0, 0.0], [110.0, 0.0, np.deg2rad(3.0)]])

    metrics = se2_relative_pose_error_by_distance(
        estimate, truth, np.array([0.0, 100.0]), 100.0
    )

    assert metrics["rmse_translation_m"] == pytest.approx(10.0)
    assert metrics["rmse_yaw_deg"] == pytest.approx(3.0)


def test_se3_rpe_reports_relative_scale_and_yaw_error() -> None:
    truth_positions = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    estimate_positions = np.array([[0.0, 0.0, 0.0], [110.0, 0.0, 0.0]])
    truth_rotations = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    estimate_rotations = np.array(
        [
            np.eye(3),
            rotation_from_rpy(0.0, 0.0, np.deg2rad(3.0)),
        ]
    )

    metrics = se3_relative_pose_error_by_distance(
        estimate_positions,
        estimate_rotations,
        truth_positions,
        truth_rotations,
        np.array([0.0, 100.0]),
        100.0,
    )

    assert metrics["rmse_translation_m"] == pytest.approx(10.0)
    assert metrics["rmse_rotation_deg"] == pytest.approx(3.0)


def test_se3_rpe_reports_an_unavailable_too_long_interval() -> None:
    positions = np.zeros((2, 3))
    rotations = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    metrics = se3_relative_pose_error_by_distance(
        positions,
        rotations,
        positions,
        rotations,
        np.array([0.0, 1.0]),
        10.0,
    )

    assert not metrics["available"]
    assert metrics["samples"] == 0
    assert metrics["rmse_translation_m"] is None
