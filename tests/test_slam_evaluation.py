import numpy as np
import pytest

from oceansim.slam import map_to_odom_change_metrics
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
