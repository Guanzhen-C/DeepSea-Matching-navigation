from __future__ import annotations

import numpy as np

from oceansim.slam.large_scale_sim import (
    FigureEightSurvey,
    LargeScaleSensorSimulator,
    LargeUnderwaterWorld,
)


def test_large_scale_integrated_ping_uses_one_dense_union_intersection() -> None:
    world = LargeUnderwaterWorld()
    survey = FigureEightSurvey()
    simulator = LargeScaleSensorSimulator(
        world,
        survey,
        num_beams=12,
        imaging_rays_per_side=48,
        sidescan_range_bins=80,
        seed=19,
    )
    original_intersect = world.intersect_rays
    launches: list[np.ndarray] = []

    def counted_intersect(origin_world, directions_world, **kwargs):
        launches.append(np.asarray(directions_world).copy())
        return original_intersect(origin_world, directions_world, **kwargs)

    world.intersect_rays = counted_intersect
    truth = survey.state(0.0)
    ping = simulator.bathymetric_sidescan_ping(truth)

    assert len(launches) == 1
    assert launches[0].shape == (ping.shared_ray_count, 3)
    assert ping.bathymetry.points_sensor.shape == (12, 3)
    assert ping.imaging.points_sensor.shape == (96, 3)
    assert ping.side_scan.port.shape == (80,)
    assert ping.side_scan.starboard.shape == (80,)
    assert ping.timestamp == truth.timestamp
    assert ping.frame_id == "integrated_sonar"
    assert ping.products_share_raw_returns
    assert ping.shared_ray_count == ping.imaging.ranges.size
    np.testing.assert_array_equal(
        ping.imaging.source_ray_indices, np.arange(ping.shared_ray_count)
    )
    np.testing.assert_array_equal(ping.origin_world, truth.position)
    np.testing.assert_allclose(
        ping.shared_ray_directions[ping.bathymetry_source_ray_indices],
        ping.bathymetry.beam_directions,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        ping.shared_ray_directions[ping.imaging.source_ray_indices],
        ping.imaging.beam_directions,
        atol=1.0e-14,
    )
    selected = ping.bathymetry_source_ray_indices
    np.testing.assert_array_equal(
        ping.bathymetry.ranges, ping.imaging.ranges[selected]
    )
    np.testing.assert_array_equal(
        ping.bathymetry.intensities, ping.imaging.intensities[selected]
    )
    np.testing.assert_array_equal(
        ping.bathymetry.points_sensor, ping.imaging.points_sensor[selected]
    )
    points, values, source_rays = ping.valid_imaging_samples()
    assert points.shape[0] > ping.bathymetry.points_sensor.shape[0]
    assert points.shape[0] == values.size == source_rays.size
    assert np.all(source_rays < ping.shared_ray_count)


def test_large_scale_legacy_tuple_remains_sparse_and_compatible() -> None:
    world = LargeUnderwaterWorld()
    survey = FigureEightSurvey()
    simulator = LargeScaleSensorSimulator(
        world,
        survey,
        num_beams=14,
        imaging_rays_per_side=24,
        sidescan_range_bins=60,
        seed=23,
    )

    points, intensities, hit, port, starboard = (
        simulator.bathymetric_sidescan(survey.state(0.0))
    )

    assert points.shape == (14, 3)
    assert intensities.shape == hit.shape == (14,)
    assert port.shape == starboard.shape == (60,)
    assert simulator.last_integrated_ping is not None
    assert simulator.last_integrated_ping.imaging.ranges.shape == (48,)


def test_deterministic_sensor_supports_eight_parallel_rows_per_side() -> None:
    world = LargeUnderwaterWorld(terrain_profile="continental_margin")
    survey = FigureEightSurvey(nominal_depth=8.5)
    simulator = LargeScaleSensorSimulator(
        world,
        survey,
        num_beams=96,
        imaging_rays_per_side=64,
        sidescan_range_bins=160,
        along_track_rows_per_side=8,
        along_track_sample_spacing_m=0.15,
        seed=31,
    )

    ping = simulator.bathymetric_sidescan_ping(survey.state(0.0))

    assert ping.shared_ray_count == 2 * 8 * 64
    assert ping.imaging.points_sensor.shape == (2 * 8 * 64, 3)
    np.testing.assert_array_equal(
        np.unique(ping.imaging.along_track_row_indices), np.arange(8)
    )
    assert np.unique(ping.imaging.ray_origins_sensor[:, 0]).size == 8
    assert ping.along_track_coverage_m == 1.2
    selected = ping.bathymetry_source_ray_indices
    np.testing.assert_array_equal(
        ping.bathymetry.points_sensor, ping.imaging.points_sensor[selected]
    )


def test_subgrid_reflectivity_is_opt_in_world_fixed_and_zero_mean() -> None:
    world = LargeUnderwaterWorld(terrain_profile="continental_margin")
    survey = FigureEightSurvey()
    baseline = LargeScaleSensorSimulator(
        world, survey, subgrid_reflectivity_amplitude=0.0
    )
    textured = LargeScaleSensorSimulator(
        world, survey, subgrid_reflectivity_amplitude=0.18
    )
    x = np.linspace(-20.0, 20.0, 801)
    y = 0.37 * x + 4.0

    measured = world.reflectivity(x, y)
    np.testing.assert_array_equal(
        baseline._acoustic_reflectivity(x, y), measured
    )
    first = textured._acoustic_reflectivity(x, y)
    second = textured._acoustic_reflectivity(x, y)
    np.testing.assert_array_equal(first, second)
    assert np.std(first - measured) > 0.01
    assert abs(float(np.mean(first - measured))) < 0.03
