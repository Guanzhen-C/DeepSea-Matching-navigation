from __future__ import annotations

import numpy as np

from oceansim.slam import (
    GlobalTerrainRelocalizer,
    GlobalTerrainRelocalizerConfig,
    InertialState,
    MeasuredBathymetryGrid,
)
from oceansim.slam.math3d import rotation_from_rpy, wrap_angle


def _joint_world(size: int = 501) -> MeasuredBathymetryGrid:
    axis = np.arange(size, dtype=np.float64) - 0.5 * (size - 1)
    x, y = np.meshgrid(axis, axis)
    elevation = (
        -72.0
        + 0.014 * x
        - 0.009 * y
        + 3.1 * np.sin(x / 31.0)
        + 2.4 * np.cos(y / 43.0)
        + 1.2 * np.sin((1.3 * x + y) / 17.0)
        + 0.7 * np.cos((x - 1.8 * y) / 11.0)
    )
    backscatter = np.clip(
        0.48
        + 0.17 * np.sin(x / 8.3)
        + 0.13 * np.cos(y / 12.7)
        + 0.11 * np.sin((x + 1.6 * y) / 6.1)
        + 0.07 * np.cos((2.2 * x - y) / 9.7),
        0.02,
        0.98,
    )
    valid = np.ones_like(elevation, dtype=bool)
    return MeasuredBathymetryGrid(
        elevation,
        resolution_m=1.0,
        backscatter=backscatter,
        backscatter_valid_mask=valid,
    )


def _joint_patch(
    world: MeasuredBathymetryGrid,
    pose2: np.ndarray,
    *,
    depth_m: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    along, cross = np.meshgrid(
        np.linspace(-24.0, 24.0, 17),
        np.concatenate(
            (np.linspace(-105.0, -8.0, 28), np.linspace(8.0, 105.0, 28))
        ),
    )
    body_xy = np.column_stack((along.reshape(-1), cross.reshape(-1)))
    yaw = float(pose2[2])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    world_xy = pose2[:2] + np.column_stack(
        (
            cosine * body_xy[:, 0] - sine * body_xy[:, 1],
            sine * body_xy[:, 0] + cosine * body_xy[:, 1],
        )
    )
    elevation = world.seabed_height(world_xy[:, 0], world_xy[:, 1])
    points_body = np.column_stack((body_xy, elevation + depth_m))
    reflectivity = world.measured_reflectivity(world_xy[:, 0], world_xy[:, 1])
    range_gain = (
        1.30
        - 0.0035 * np.abs(body_xy[:, 1])
        + 0.09 * np.sign(body_xy[:, 1])
    )
    intensity = np.clip(reflectivity * range_gain, 0.005, 1.0)
    return points_body, intensity


def _config(**overrides) -> GlobalTerrainRelocalizerConfig:
    values = dict(
        pyramid_stride=2,
        yaw_half_window_deg=20.0,
        yaw_step_deg=5.0,
        maximum_candidates=5,
        candidates_per_yaw=20,
        basin_translation_separation_m=60.0,
        basin_yaw_separation_deg=8.0,
        minimum_bathymetry_bins=40,
        minimum_backscatter_bins=40,
        minimum_joint_correlation=0.25,
        minimum_basin_score_gap=0.04,
    )
    values.update(overrides)
    return GlobalTerrainRelocalizerConfig(**values)


def test_joint_global_search_recovers_pose_outside_local_window() -> None:
    world = _joint_world()
    truth = np.array([72.0, -58.0, np.deg2rad(27.0)])
    points, intensity = _joint_patch(world, truth)
    # The x/y seed is deliberately hundreds of metres away.  Only pressure
    # depth and an IMU yaw centre within the configured uncertainty are used.
    state = InertialState(
        timestamp=12.0,
        position=np.array([-135.0, 118.0, -35.0]),
        rotation=rotation_from_rpy(0.0, 0.0, truth[2] + np.deg2rad(10.0)),
    )

    result = GlobalTerrainRelocalizer(world, _config()).search(
        points,
        state,
        backscatter_points_body=points,
        backscatter_values=intensity,
        yaw_center_rad=truth[2] + np.deg2rad(10.0),
    )

    assert result.accepted, result.rejection_reason
    assert result.candidates
    best = result.candidates[0]
    assert np.linalg.norm(best.pose2[:2] - truth[:2]) <= 3.0
    assert abs(float(wrap_angle(best.pose2[2] - truth[2]))) <= np.deg2rad(2.6)
    assert best.joint_correlation >= _config().minimum_joint_correlation
    assert result.basin_score_gap >= _config().minimum_basin_score_gap


def test_global_search_covers_recorded_full_route_yaw_drift() -> None:
    world = _joint_world()
    truth = np.array([72.0, -58.0, np.deg2rad(27.0)])
    points, intensity = _joint_patch(world, truth)
    # The controlled 72 km run recorded 78.6 deg maximum local yaw error.
    # Exercise a slightly larger 80 deg error inside the configured 90 deg
    # recovery envelope.
    yaw_seed = truth[2] + np.deg2rad(80.0)
    state = InertialState(
        timestamp=12.0,
        position=np.array([-135.0, 118.0, -35.0]),
        rotation=rotation_from_rpy(0.0, 0.0, yaw_seed),
    )

    result = GlobalTerrainRelocalizer(
        world,
        _config(yaw_half_window_deg=90.0, yaw_step_deg=5.0),
    ).search(
        points,
        state,
        backscatter_points_body=points,
        backscatter_values=intensity,
        yaw_center_rad=yaw_seed,
    )

    assert result.accepted, result.rejection_reason
    best = result.candidates[0]
    assert np.linalg.norm(best.pose2[:2] - truth[:2]) <= 3.0
    assert abs(float(wrap_angle(best.pose2[2] - truth[2]))) <= np.deg2rad(2.6)


def test_global_search_fails_closed_without_required_backscatter() -> None:
    world = _joint_world()
    truth = np.array([20.0, 15.0, np.deg2rad(-12.0)])
    points, _ = _joint_patch(world, truth)
    state = InertialState(
        timestamp=4.0,
        position=np.array([0.0, 0.0, -35.0]),
        rotation=rotation_from_rpy(0.0, 0.0, truth[2]),
    )

    result = GlobalTerrainRelocalizer(world, _config()).search(points, state)

    assert not result.accepted
    assert result.rejection_reason == "insufficient_backscatter_points"
    assert not result.candidates


def test_global_search_rejects_flat_modalities() -> None:
    size = 241
    flat = np.full((size, size), -60.0)
    texture = np.full((size, size), 0.4)
    world = MeasuredBathymetryGrid(
        flat,
        resolution_m=1.0,
        backscatter=texture,
        backscatter_valid_mask=np.ones_like(flat, dtype=bool),
    )
    x, y = np.meshgrid(np.linspace(-15.0, 15.0, 15), np.linspace(-70.0, 70.0, 31))
    points = np.column_stack((x.reshape(-1), y.reshape(-1), np.full(x.size, -25.0)))
    values = np.full(points.shape[0], 0.4)
    state = InertialState(position=np.array([0.0, 0.0, -35.0]))

    result = GlobalTerrainRelocalizer(world, _config()).search(
        points,
        state,
        backscatter_points_body=points,
        backscatter_values=values,
    )

    assert not result.accepted
    assert result.rejection_reason == "no_finite_global_candidates"


def test_joint_texture_disambiguates_periodic_bathymetry() -> None:
    size = 501
    axis = np.arange(size, dtype=np.float64) - 0.5 * (size - 1)
    x, y = np.meshgrid(axis, axis)
    # Bathymetry alone repeats exactly every 80 m in x.  The side-scan texture
    # deliberately does not share that period.
    elevation = (
        -66.0
        + 2.5 * np.sin(2.0 * np.pi * x / 80.0)
        + 1.7 * np.cos(y / 37.0)
        + 0.8 * np.sin(2.0 * np.pi * x / 80.0 + y / 19.0)
    )
    texture = np.clip(
        0.50
        + 0.19 * np.sin(x / 9.1)
        + 0.14 * np.cos(y / 13.7)
        + 0.10 * np.sin((x + 1.9 * y) / 5.7),
        0.02,
        0.98,
    )
    world = MeasuredBathymetryGrid(
        elevation,
        resolution_m=1.0,
        backscatter=texture,
        backscatter_valid_mask=np.ones_like(elevation, dtype=bool),
    )
    truth = np.array([91.0, -43.0, np.deg2rad(15.0)])
    points, intensity = _joint_patch(world, truth)
    state = InertialState(
        timestamp=8.0,
        position=np.array([-170.0, 120.0, -35.0]),
        rotation=rotation_from_rpy(0.0, 0.0, truth[2]),
    )

    result = GlobalTerrainRelocalizer(world, _config()).search(
        points,
        state,
        backscatter_points_body=points,
        backscatter_values=intensity,
    )

    assert result.accepted, result.rejection_reason
    assert np.linalg.norm(result.candidates[0].pose2[:2] - truth[:2]) <= 3.0
    assert result.candidates[0].backscatter_correlation > 0.5


def test_repeated_joint_texture_is_rejected_as_globally_ambiguous() -> None:
    size = 401
    axis = np.arange(size, dtype=np.float64) - 0.5 * (size - 1)
    x, y = np.meshgrid(axis, axis)
    elevation = (
        -64.0
        + 2.1 * np.sin(2.0 * np.pi * x / 80.0)
        + 1.5 * np.cos(2.0 * np.pi * y / 100.0)
        + 0.9 * np.sin(2.0 * np.pi * (x / 80.0 + y / 100.0))
    )
    texture = np.clip(
        0.50
        + 0.18 * np.sin(2.0 * np.pi * x / 80.0)
        + 0.14 * np.cos(2.0 * np.pi * y / 100.0)
        + 0.10 * np.sin(2.0 * np.pi * (x / 80.0 + y / 100.0)),
        0.02,
        0.98,
    )
    world = MeasuredBathymetryGrid(
        elevation,
        resolution_m=1.0,
        backscatter=texture,
        backscatter_valid_mask=np.ones_like(elevation, dtype=bool),
    )
    truth = np.array([0.0, 0.0, 0.0])
    points, intensity = _joint_patch(world, truth)
    state = InertialState(position=np.array([130.0, -120.0, -35.0]))
    config = _config(
        yaw_half_window_deg=5.0,
        yaw_step_deg=5.0,
        minimum_basin_score_gap=0.08,
    )

    result = GlobalTerrainRelocalizer(world, config).search(
        points,
        state,
        backscatter_points_body=points,
        backscatter_values=intensity,
        yaw_center_rad=0.0,
    )

    assert not result.accepted
    assert result.rejection_reason == "ambiguous_global_match"
    assert len(result.candidates) >= 2
    assert result.basin_score_gap < config.minimum_basin_score_gap


def test_absolute_depth_resolves_a_zero_mean_relief_alias() -> None:
    size = 501
    axis = np.arange(size, dtype=np.float64) - 0.5 * (size - 1)
    x, y = np.meshgrid(axis, axis)
    # Texture and zero-mean relief repeat every 80 m in x.  The regional depth
    # trend shares the pressure sensor/map vertical datum and is the only
    # evidence that distinguishes those otherwise identical x basins.
    elevation = (
        -72.0
        + 0.10 * x
        + 2.4 * np.sin(2.0 * np.pi * x / 80.0)
        + 1.6 * np.cos(y / 31.0)
        + 0.7 * np.sin(y / 12.0)
    )
    texture = np.clip(
        0.50
        + 0.18 * np.sin(2.0 * np.pi * x / 80.0)
        + 0.14 * np.cos(y / 13.0)
        + 0.08 * np.sin(y / 5.7),
        0.02,
        0.98,
    )
    world = MeasuredBathymetryGrid(
        elevation,
        resolution_m=1.0,
        backscatter=texture,
        backscatter_valid_mask=np.ones_like(elevation, dtype=bool),
    )
    truth = np.array([80.0, 0.0, 0.0])
    points, intensity = _joint_patch(world, truth)
    state = InertialState(
        position=np.array([-150.0, 130.0, -35.0]),
        rotation=rotation_from_rpy(0.0, 0.0, 0.0),
    )
    config = _config(
        yaw_half_window_deg=5.0,
        yaw_step_deg=5.0,
        maximum_bathymetry_rmse_m=5.0,
        bathymetry_absolute_weight=0.5,
    )

    result = GlobalTerrainRelocalizer(world, config).search(
        points,
        state,
        backscatter_points_body=points,
        backscatter_values=intensity,
        yaw_center_rad=0.0,
    )

    assert result.accepted, result.rejection_reason
    best = result.candidates[0]
    assert np.linalg.norm(best.pose2[:2] - truth[:2]) <= 3.0
    assert best.bathymetry_rmse_m < 1.0
    assert result.basin_score_gap >= config.minimum_basin_score_gap
