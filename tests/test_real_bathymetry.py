from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oceansim.slam.real_bathymetry import (
    MeasuredBathymetryGrid,
    TerrainFollowingSurvey,
    WaypointTerrainFollowingSurvey,
)


def _write_complete_map(directory) -> tuple[np.ndarray, np.ndarray]:
    directory.mkdir()
    rows, columns = np.indices((5, 5), dtype=np.float32)
    elevation = -40.0 + 0.5 * columns + 0.25 * rows
    valid = np.ones(elevation.shape, dtype=bool)
    valid[1, 1] = False
    elevation[~valid] = np.nan
    navigation = elevation.copy()

    backscatter = np.full(elevation.shape, np.nan, dtype=np.float32)
    backscatter[3, 3] = 0.73
    backscatter_valid = np.isfinite(backscatter)
    source_mask = np.zeros(elevation.shape, dtype=np.uint8)
    source_mask[3, 3] = 4

    np.save(directory / "elevation_m.npy", elevation)
    np.save(directory / "navigation_elevation_m.npy", navigation)
    np.save(directory / "bathymetry_valid_mask.npy", valid)
    np.save(directory / "backscatter.npy", backscatter)
    np.save(directory / "backscatter_valid_mask.npy", backscatter_valid)
    np.save(directory / "backscatter_source_mask.npy", source_mask)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "title": "small measured fixture",
                "resolution_m": 2.0,
                "source_shape_pixels": [5, 5],
            }
        ),
        encoding="utf-8",
    )
    return elevation, valid


def test_complete_directory_is_memory_mapped_and_preserves_provenance(
    tmp_path,
) -> None:
    source = tmp_path / "complete_map"
    elevation, valid = _write_complete_map(source)

    world = MeasuredBathymetryGrid.from_path(source)

    assert isinstance(world.elevation_m, np.memmap)
    assert isinstance(world.navigation_elevation_m, np.memmap)
    assert isinstance(world.valid_mask, np.memmap)
    assert isinstance(world.backscatter, np.memmap)
    assert isinstance(world.backscatter_source_mask, np.memmap)
    assert world.source_path == source
    assert world.metadata["title"] == "small measured fixture"
    np.testing.assert_array_equal(world.valid_mask, valid)
    assert np.isnan(world.elevation_m[1, 1])
    assert world.has_measured_backscatter
    assert world.backscatter_coverage_fraction == pytest.approx(1.0 / 24.0)

    # Exact grid vertices do not need an unused, zero-weight neighbouring
    # corner, but an interpolation genuinely spanning NoData is rejected.
    assert world.seabed_height(0.0, -2.0) == pytest.approx(elevation[1, 2])
    assert np.isnan(world.seabed_height(-1.0, -1.0))
    assert np.isnan(world.seabed_height(5.0, 0.0))

    assert world.measured_reflectivity(2.0, 2.0) == pytest.approx(0.73)
    assert world.reflectivity(2.0, 2.0) == pytest.approx(0.73)
    assert int(world.reflectivity_source(2.0, 2.0)) == 4
    assert np.isnan(world.measured_reflectivity(-2.0, 2.0))
    assert np.isfinite(world.reflectivity(-2.0, 2.0))
    assert int(world.reflectivity_source(-2.0, 2.0)) == 0


def test_lod_mesh_is_compact_and_never_caps_a_native_nodata_hole() -> None:
    rows, columns = np.indices((5, 5), dtype=np.float32)
    elevation = -20.0 + 0.2 * columns + 0.1 * rows
    valid = np.ones(elevation.shape, dtype=bool)
    valid[1, 1] = False
    elevation[~valid] = np.nan
    world = MeasuredBathymetryGrid(
        elevation,
        resolution_m=2.0,
        valid_mask=valid,
    )

    grid_x, grid_y, grid_z = world.mesh_grid(stride=2)
    assert grid_x.shape == grid_y.shape == grid_z.shape == (3, 3)
    np.testing.assert_allclose(grid_x[0], [-4.0, 0.0, 4.0])
    np.testing.assert_allclose(grid_y[:, 0], [-4.0, 0.0, 4.0])

    points, indices = world.triangle_mesh(stride=2, compact=True)
    # Of the four 4 m LOD quads, the south-west one contains the native
    # NoData cell and therefore contributes no triangles at all.
    assert points.shape == (8, 3)
    assert indices.shape == (3 * 2 * 3,)
    assert indices.dtype == np.int32
    assert np.isfinite(points).all()
    triangles = points[indices.reshape(-1, 3)]
    assert not np.any(
        np.all(triangles[..., 0] <= 0.0, axis=1)
        & np.all(triangles[..., 1] <= 0.0, axis=1)
    )
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    assert np.all(normals[:, 2] > 0.0)

    alias_points, alias_indices = world.mesh(stride=2)
    np.testing.assert_array_equal(alias_points, points)
    np.testing.assert_array_equal(alias_indices, indices)

    native_points, native_indices = world.triangle_mesh(stride=1)
    assert np.isfinite(native_points).all()
    assert native_indices.max() < native_points.shape[0]
    with pytest.raises(ValueError, match="positive integer"):
        world.triangle_mesh(stride=0)


def test_from_path_keeps_legacy_npz_compatibility(tmp_path) -> None:
    elevation = np.arange(12, dtype=np.float32).reshape(3, 4)
    source = tmp_path / "legacy.npz"
    np.savez(source, elevation_m=elevation, resolution_m=np.float64(2.5))
    source.with_suffix(".json").write_text(
        json.dumps({"title": "legacy fixture"}), encoding="utf-8"
    )

    world = MeasuredBathymetryGrid.from_path(source)

    assert world.elevation_m.shape == (3, 4)
    assert (world.width, world.length) == (7.5, 5.0)
    assert world.metadata["title"] == "legacy fixture"
    assert world.backscatter is None
    points, indices = world.triangle_mesh()
    assert points.shape == (12, 3)
    assert indices.shape == (2 * 3 * 6,)


@pytest.mark.parametrize(
    "invalid_parameter",
    (
        {"period": np.nan},
        {"speed_mps": np.inf},
        {"x_amplitude": 0.0},
        {"y_amplitude": -1.0},
        {"altitude_m": np.nan},
        {"center_x_m": np.inf},
        {"center_y_m": np.nan},
        {"pattern_yaw_deg": -np.inf},
    ),
)
def test_terrain_following_survey_rejects_nonphysical_parameters(
    invalid_parameter,
) -> None:
    elevation = np.full((101, 101), -100.0)
    world = MeasuredBathymetryGrid(elevation, resolution_m=10.0)

    with pytest.raises(ValueError):
        TerrainFollowingSurvey(world, **invalid_parameter)


def test_terrain_following_survey_checks_rotated_bounds_and_nodata() -> None:
    elevation = np.full((201, 201), -100.0)
    world = MeasuredBathymetryGrid(elevation, resolution_m=5.0)

    with pytest.raises(ValueError, match="grid x bounds"):
        TerrainFollowingSurvey(
            world,
            x_amplitude=400.0,
            y_amplitude=100.0,
            center_x_m=200.0,
            pattern_yaw_deg=45.0,
        )

    valid = np.ones(elevation.shape, dtype=bool)
    valid[100, 100] = False
    world_with_gap = MeasuredBathymetryGrid(
        elevation,
        resolution_m=5.0,
        valid_mask=valid,
    )
    with pytest.raises(ValueError, match="bathymetry NoData"):
        TerrainFollowingSurvey(world_with_gap)


def test_recommended_large_route_has_safe_measured_coverage() -> None:
    root = Path(__file__).resolve().parents[1]
    world = MeasuredBathymetryGrid.from_path(
        root / "data/monterey_canyon_full_5m"
    )
    survey = TerrainFollowingSurvey(
        world,
        speed_mps=1.5,
        x_amplitude=1_200.0,
        y_amplitude=700.0,
        altitude_m=28.0,
        center_x_m=-3_500.0,
        center_y_m=-1_500.0,
        pattern_yaw_deg=165.0,
    )

    np.testing.assert_allclose(
        survey.state(0.0).position[:2], [-3_500.0, -1_500.0]
    )
    report = survey.validate_route(
        sample_count=8_192,
        minimum_clearance_m=250.0,
        require_measured_backscatter=True,
    )

    assert report.bathymetry_fully_valid
    assert report.backscatter_fully_valid
    assert report.path_length_m == pytest.approx(7_938.14, abs=0.1)
    assert report.period_s == pytest.approx(report.path_length_m / 1.5, abs=0.1)
    np.testing.assert_allclose(
        report.x_range_m, [-4_710.27, -2_289.73], atol=0.1
    )
    np.testing.assert_allclose(
        report.y_range_m, [-2_404.01, -595.99], atol=0.1
    )
    np.testing.assert_allclose(
        report.auv_depth_range_m, [68.96, 138.41], atol=0.1
    )
    assert report.altitude_range_m[0] > 26.0
    assert report.altitude_range_m[1] < 30.1
    assert report.pitch_range_deg[0] > -11.0
    assert report.pitch_range_deg[1] < 11.0
    assert report.bathymetry_min_clearance_m > 1_300.0
    assert report.backscatter_min_clearance_m is not None
    assert report.backscatter_min_clearance_m > 600.0

    with pytest.raises(ValueError, match="measured-backscatter clearance"):
        survey.validate_route(
            sample_count=8_192,
            minimum_clearance_m=700.0,
            require_measured_backscatter=True,
        )


def _sloped_waypoint_world(
    *, valid_mask: np.ndarray | None = None,
    backscatter_valid_mask: np.ndarray | None = None,
) -> MeasuredBathymetryGrid:
    rows, columns = np.indices((241, 241), dtype=np.float64)
    x = (columns - 120.0) * 2.0
    y = (rows - 120.0) * 2.0
    elevation = -100.0 + 0.03 * x - 0.02 * y
    valid = (
        np.ones(elevation.shape, dtype=bool)
        if valid_mask is None
        else valid_mask
    )
    backscatter_valid = (
        valid.copy()
        if backscatter_valid_mask is None
        else backscatter_valid_mask
    )
    backscatter = np.full(elevation.shape, 0.6, dtype=np.float64)
    return MeasuredBathymetryGrid(
        elevation,
        resolution_m=2.0,
        navigation_elevation_m=elevation,
        valid_mask=valid,
        backscatter=backscatter,
        backscatter_valid_mask=backscatter_valid,
    )


def test_waypoint_survey_is_closed_smooth_and_constant_3d_speed() -> None:
    world = _sloped_waypoint_world()
    # A repeated first point is a conventional closed-polyline encoding and
    # is deliberately accepted without creating a zero-length closing span.
    waypoints = np.array(
        [
            [-120.0, -80.0],
            [100.0, -90.0],
            [130.0, 40.0],
            [20.0, 120.0],
            [-130.0, 70.0],
            [-120.0, -80.0],
        ]
    )
    survey = WaypointTerrainFollowingSurvey(
        world,
        waypoints,
        speed_mps=2.0,
        altitude_m=25.0,
        smoothing_iterations=4,
    )

    assert survey.waypoints_xy_m.shape == (5, 2)
    assert survey.smoothed_waypoints_xy_m.shape == (5 * 2**4, 2)
    assert not survey.waypoints_xy_m.flags.writeable
    assert survey.period == pytest.approx(survey.path_length_m / 2.0)

    state0 = survey.state(0.0)
    state_period = survey.state(survey.period)
    np.testing.assert_allclose(state_period.position, state0.position, atol=1e-10)
    np.testing.assert_allclose(state_period.velocity, state0.velocity, atol=1e-9)
    np.testing.assert_allclose(state_period.rotation, state0.rotation, atol=1e-9)

    sample_times = np.linspace(0.0, survey.period, 80, endpoint=False)
    states = [survey.state(timestamp) for timestamp in sample_times]
    positions = np.asarray([state.position for state in states])
    speeds = np.asarray([np.linalg.norm(state.velocity) for state in states])
    expected_height = world.navigation_height(
        positions[:, 0], positions[:, 1]
    ) + 25.0
    np.testing.assert_allclose(positions[:, 2], expected_height, atol=1e-10)
    np.testing.assert_allclose(speeds, 2.0, rtol=8e-3, atol=8e-3)

    report = survey.validate_route(
        sample_count=512,
        minimum_clearance_m=50.0,
        require_measured_backscatter=True,
    )
    assert report.sample_count == 512
    assert report.waypoint_count == 5
    assert report.validation_sample_count is not None
    assert report.validation_sample_count > report.waypoint_count
    assert report.validation_max_spacing_m == pytest.approx(0.5)
    assert report.maximum_horizontal_curvature_per_m is not None
    assert np.isfinite(report.maximum_horizontal_curvature_per_m)
    assert report.path_length_m == pytest.approx(survey.path_length_m)
    assert report.backscatter_fully_valid


@pytest.mark.parametrize(
    ("waypoints", "keyword", "value", "message"),
    (
        ([[0.0, 0.0], [1.0, 1.0]], None, None, "at least three"),
        (
            [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            None,
            None,
            "consecutive",
        ),
        (
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            "smoothing_iterations",
            True,
            "smoothing_iterations",
        ),
        (
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            "smoothing_iterations",
            13,
            "smoothing_iterations",
        ),
    ),
)
def test_waypoint_survey_rejects_invalid_route_inputs(
    waypoints, keyword, value, message
) -> None:
    world = _sloped_waypoint_world()
    options = {} if keyword is None else {keyword: value}
    with pytest.raises(ValueError, match=message):
        WaypointTerrainFollowingSurvey(world, waypoints, **options)


def test_waypoint_survey_cannot_step_over_native_nodata() -> None:
    waypoints = np.array(
        [
            [-100.0, -70.0],
            [100.0, -70.0],
            [100.0, 90.0],
            [-100.0, 90.0],
        ]
    )
    gap = np.ones((241, 241), dtype=bool)
    gap[:, 120] = False
    bathymetry_gap_world = _sloped_waypoint_world(valid_mask=gap)

    with pytest.raises(ValueError, match="bathymetry NoData"):
        WaypointTerrainFollowingSurvey(
            bathymetry_gap_world,
            waypoints,
            smoothing_iterations=0,
        )

    backscatter_gap_world = _sloped_waypoint_world(
        backscatter_valid_mask=gap
    )
    survey = WaypointTerrainFollowingSurvey(
        backscatter_gap_world,
        waypoints,
        smoothing_iterations=0,
    )
    with pytest.raises(ValueError, match="measured backscatter"):
        survey.validate_route(
            sample_count=32,
            require_measured_backscatter=True,
        )


def test_waypoint_survey_enforces_continuous_clearance() -> None:
    world = _sloped_waypoint_world()
    waypoints = np.array(
        [
            [-80.0, -60.0],
            [80.0, -60.0],
            [80.0, 60.0],
            [-80.0, 60.0],
        ]
    )
    survey = WaypointTerrainFollowingSurvey(
        world,
        waypoints,
        smoothing_iterations=2,
    )
    report = survey.validate_route(
        sample_count=64,
        require_measured_backscatter=True,
    )
    assert report.bathymetry_min_clearance_m > 0.0
    assert report.backscatter_min_clearance_m is not None

    with pytest.raises(ValueError, match="bathymetry clearance"):
        survey.validate_route(
            sample_count=64,
            minimum_clearance_m=report.bathymetry_min_clearance_m + 0.01,
        )
    backscatter_valid = np.ones((241, 241), dtype=bool)
    backscatter_valid[:, :20] = False
    backscatter_world = _sloped_waypoint_world(
        backscatter_valid_mask=backscatter_valid
    )
    backscatter_survey = WaypointTerrainFollowingSurvey(
        backscatter_world,
        waypoints,
        smoothing_iterations=2,
    )
    backscatter_report = backscatter_survey.validate_route(
        sample_count=64,
        require_measured_backscatter=True,
    )
    assert backscatter_report.backscatter_min_clearance_m is not None
    assert (
        backscatter_report.backscatter_min_clearance_m
        < backscatter_report.bathymetry_min_clearance_m
    )
    threshold = 0.5 * (
        backscatter_report.backscatter_min_clearance_m
        + backscatter_report.bathymetry_min_clearance_m
    )
    with pytest.raises(ValueError, match="measured-backscatter clearance"):
        backscatter_survey.validate_route(
            sample_count=64,
            minimum_clearance_m=threshold,
            require_measured_backscatter=True,
        )


@pytest.mark.parametrize(
    "maximum_pitch_deg", (True, -0.01, 90.0, np.inf, np.nan)
)
def test_waypoint_survey_rejects_invalid_pitch_limit(
    maximum_pitch_deg,
) -> None:
    world = _sloped_waypoint_world()
    with pytest.raises(ValueError, match="maximum_pitch_deg"):
        WaypointTerrainFollowingSurvey(
            world,
            [[-20.0, -20.0], [20.0, -20.0], [0.0, 20.0]],
            maximum_pitch_deg=maximum_pitch_deg,
        )


def test_waypoint_pitch_envelope_is_periodic_safe_and_speed_constant() -> None:
    rows, columns = np.indices((301, 301), dtype=np.float64)
    x = (columns - 150.0) * 2.0
    y = (rows - 150.0) * 2.0
    navigation = -120.0 + 0.55 * x + 0.04 * y
    # The raw seabed deliberately differs from the navigation surface.  The
    # pitch controller must majorize the navigation target; reported altitude
    # remains relative to this raw measured surface.
    elevation = navigation + 3.0 * np.cos(y / 35.0)
    world = MeasuredBathymetryGrid(
        elevation,
        resolution_m=2.0,
        navigation_elevation_m=navigation,
        backscatter=np.full(elevation.shape, 0.7),
    )
    survey = WaypointTerrainFollowingSurvey(
        world,
        [
            [-130.0, -80.0],
            [130.0, -80.0],
            [145.0, 20.0],
            [70.0, 115.0],
            [-100.0, 120.0],
            [-145.0, 10.0],
        ],
        speed_mps=2.2,
        altitude_m=24.0,
        smoothing_iterations=4,
        maximum_pitch_deg=12.0,
    )

    assert survey.maximum_pitch_deg == pytest.approx(12.0)
    assert survey.pitch_envelope_guard_m > 0.0
    state0 = survey.state(0.0)
    state_period = survey.state(survey.period)
    np.testing.assert_allclose(state_period.position, state0.position, atol=1e-10)
    np.testing.assert_allclose(state_period.velocity, state0.velocity, atol=1e-9)

    timestamps = np.linspace(0.0, survey.period, 1_024, endpoint=False)
    states = [survey.state(timestamp) for timestamp in timestamps]
    positions = np.asarray([state.position for state in states])
    velocities = np.asarray([state.velocity for state in states])
    target_height = world.navigation_height(
        positions[:, 0], positions[:, 1]
    ) + survey.altitude_m
    assert np.all(positions[:, 2] >= target_height - 1.0e-9)

    horizontal_speed = np.linalg.norm(velocities[:, :2], axis=1)
    pitch_deg = np.rad2deg(
        np.abs(np.arctan2(velocities[:, 2], horizontal_speed))
    )
    assert np.max(pitch_deg) <= 12.0 + 0.15
    np.testing.assert_allclose(
        np.linalg.norm(velocities, axis=1),
        2.2,
        rtol=1.0e-2,
        atol=1.0e-2,
    )

    report = survey.validate_route(
        sample_count=1_024,
        require_measured_backscatter=True,
    )
    assert report.pitch_range_deg[0] >= -12.0 - 0.15
    assert report.pitch_range_deg[1] <= 12.0 + 0.15
    assert report.altitude_range_m[0] < survey.altitude_m
