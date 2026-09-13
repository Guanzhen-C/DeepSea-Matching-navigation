from __future__ import annotations

import numpy as np
import pytest

from oceansim.slam import (
    FastSonarSLAM,
    FastSonarSLAMConfig,
    MapOdomSoftCorrectionConfig,
    MapOdomSoftCorrector,
    apply_map_to_odom_pose,
    apply_map_to_odom_pose2,
    map_to_odom_from_poses,
)
from oceansim.slam.math3d import rotation_from_rpy, rpy_from_rotation


def test_map_to_odom_pose2_round_trip_at_large_map_coordinates() -> None:
    odom_pose = np.array([12_400.0, -7_800.0, np.deg2rad(170.0)])
    expected_transform = np.array([42.0, -17.0, np.deg2rad(8.0)])

    map_pose = apply_map_to_odom_pose2(expected_transform, odom_pose)
    recovered_transform = map_to_odom_from_poses(map_pose, odom_pose)

    np.testing.assert_allclose(recovered_transform, expected_transform)
    np.testing.assert_allclose(
        apply_map_to_odom_pose2(recovered_transform, odom_pose), map_pose
    )


@pytest.mark.parametrize(
    "field",
    (
        "translation_time_constant_s",
        "yaw_time_constant_s",
        "maximum_translation_rate_mps",
        "maximum_yaw_rate_deg_s",
    ),
)
def test_soft_correction_config_rejects_nonpositive_dynamics(field: str) -> None:
    values = {field: 0.0}
    with pytest.raises(ValueError, match=field):
        MapOdomSoftCorrectionConfig(**values)


def test_soft_correction_is_rate_limited_at_vehicle_not_map_origin() -> None:
    corrector = MapOdomSoftCorrector(
        MapOdomSoftCorrectionConfig(
            translation_time_constant_s=0.1,
            yaw_time_constant_s=0.1,
            maximum_translation_rate_mps=2.0,
            maximum_yaw_rate_deg_s=10.0,
        )
    )
    corrector.reset(timestamp=0.0)
    odom_pose = np.array([20_000.0, -13_000.0, np.deg2rad(170.0)])
    initial_map_pose = odom_pose.copy()
    target_map_pose = initial_map_pose.copy()
    target_map_pose[:2] += np.array([100.0, -50.0])
    target_map_pose[2] += np.deg2rad(30.0)

    corrected = corrector.update(1.0, odom_pose, target_map_pose)

    translation_step = corrected[:2] - initial_map_pose[:2]
    assert np.linalg.norm(translation_step) == pytest.approx(2.0)
    yaw_step = np.arctan2(
        np.sin(corrected[2] - initial_map_pose[2]),
        np.cos(corrected[2] - initial_map_pose[2]),
    )
    assert np.rad2deg(yaw_step) == pytest.approx(10.0)
    diagnostics = corrector.diagnostics()
    assert diagnostics["translation_rate_limit_activations"] == 1
    assert diagnostics["yaw_rate_limit_activations"] == 1


def test_graph_update_at_same_timestamp_cannot_jump_online_pose() -> None:
    corrector = MapOdomSoftCorrector()
    corrector.reset(timestamp=12.0)
    odom_pose = np.array([80.0, -20.0, 0.3])
    first = corrector.update(12.0, odom_pose, odom_pose)
    target = odom_pose + np.array([60.0, -30.0, 0.4])

    same_epoch = corrector.update(12.0, odom_pose, target)

    np.testing.assert_allclose(same_epoch, first)
    assert corrector.diagnostics()["remaining_translation_m"] > 60.0


def test_first_order_update_uses_pose_residual_without_origin_artifact() -> None:
    config = MapOdomSoftCorrectionConfig(
        translation_time_constant_s=10.0,
        yaw_time_constant_s=10.0,
        maximum_translation_rate_mps=1_000.0,
        maximum_yaw_rate_deg_s=1_000.0,
    )
    corrector = MapOdomSoftCorrector(config)
    corrector.reset(timestamp=0.0)
    odom_pose = np.array([25_000.0, 18_000.0, -2.5])
    target = odom_pose + np.array([10.0, -4.0, np.deg2rad(5.0)])

    corrected = corrector.update(0.1, odom_pose, target)

    alpha = 1.0 - np.exp(-0.1 / 10.0)
    np.testing.assert_allclose(
        corrected[:2], odom_pose[:2] + alpha * np.array([10.0, -4.0])
    )
    yaw_step = np.arctan2(
        np.sin(corrected[2] - odom_pose[2]),
        np.cos(corrected[2] - odom_pose[2]),
    )
    assert yaw_step == pytest.approx(alpha * np.deg2rad(5.0))


def test_full_pose_correction_retains_frontend_depth_roll_and_pitch() -> None:
    odom_rotation = rotation_from_rpy(0.12, -0.08, 0.7)
    odom_position = np.array([400.0, -250.0, -37.5])
    transform = np.array([14.0, -9.0, np.deg2rad(11.0)])

    corrected_rotation, corrected_position = apply_map_to_odom_pose(
        transform, odom_rotation, odom_position
    )

    odom_rpy = rpy_from_rotation(odom_rotation)
    corrected_rpy = rpy_from_rotation(corrected_rotation)
    np.testing.assert_allclose(corrected_rpy[:2], odom_rpy[:2], atol=1.0e-12)
    assert corrected_position[2] == odom_position[2]
    assert corrected_rpy[2] == pytest.approx(odom_rpy[2] + transform[2])


def test_fast_sonar_slam_softens_output_but_not_raw_graph_or_esikf() -> None:
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            map_to_odom_soft_correction_enabled=True,
            map_to_odom_translation_time_constant_s=0.1,
            map_to_odom_yaw_time_constant_s=0.1,
            map_to_odom_maximum_translation_rate_mps=2.0,
            map_to_odom_maximum_yaw_rate_deg_s=10.0,
        )
    )
    initial_rotation = rotation_from_rpy(0.02, -0.03, 0.4)
    initial_position = np.array([100.0, -50.0, -25.0])
    slam.initialize(
        0.0,
        rotation=initial_rotation,
        position=initial_position,
    )
    target_yaw_correction = np.deg2rad(30.0)
    target_rotation = (
        rotation_from_rpy(0.0, 0.0, target_yaw_correction)
        @ initial_rotation
    )
    target_position = initial_position + np.array([100.0, 0.0, 0.0])
    slam.pose_graph.corrected_pose = lambda _rotation, _position: (
        target_rotation.copy(),
        target_position.copy(),
    )

    state_rotation_before = slam.state.rotation.copy()
    state_position_before = slam.state.position.copy()
    raw_rotation, raw_position = slam.raw_global_pose()
    online_rotation, online_position = slam.global_pose()

    np.testing.assert_allclose(raw_rotation, target_rotation)
    np.testing.assert_allclose(raw_position, target_position)
    np.testing.assert_allclose(online_rotation, initial_rotation)
    np.testing.assert_allclose(online_position, initial_position)
    np.testing.assert_allclose(slam.state.rotation, state_rotation_before)
    np.testing.assert_allclose(slam.state.position, state_position_before)

    # Advance estimator time with zero world acceleration.  The next online
    # read can move only by the configured map-output rate limits.
    assert slam.process_imu(
        0.2,
        np.zeros(3),
        initial_rotation.T @ np.array([0.0, 0.0, 9.80665]),
    )
    state_rotation_before = slam.state.rotation.copy()
    state_position_before = slam.state.position.copy()
    online_rotation, online_position = slam.global_pose()

    assert np.linalg.norm(
        online_position[:2] - state_position_before[:2]
    ) == pytest.approx(0.4)
    yaw_delta = rpy_from_rotation(online_rotation)[2] - rpy_from_rotation(
        state_rotation_before
    )[2]
    yaw_delta = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    assert np.rad2deg(yaw_delta) == pytest.approx(2.0)
    np.testing.assert_allclose(slam.state.rotation, state_rotation_before)
    np.testing.assert_allclose(slam.state.position, state_position_before)
