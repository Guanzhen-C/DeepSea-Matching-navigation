from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "isaacsim"
    / "oceansim"
    / "utils"
    / "acoustic_visualization.py"
)
SPEC = importlib.util.spec_from_file_location(
    "oceansim_acoustic_visualization", MODULE_PATH
)
visualization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(visualization)


def _sample_ping():
    directions = np.array(
        [
            [0.0, 0.7071, -0.7071],
            [0.0, 0.0, -1.0],
            [0.0, -0.7071, -0.7071],
        ]
    )
    ranges = np.array([10.0, 7.0, 10.0])
    return SimpleNamespace(
        beam_directions=directions,
        ranges=ranges,
        intensities=np.array([0.25, 1.0, 0.5]),
        hit_mask=np.ones(3, dtype=bool),
        points_sensor=directions * ranges[:, None],
    )


def test_multibeam_renderer_returns_visible_rgba_swath():
    image = visualization.render_multibeam_swath(
        _sample_ping(), max_range=15.0, width=320, height=180
    )

    assert image.shape == (180, 320, 4)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert np.all(image[..., 3] == 255)
    assert np.unique(image.reshape(-1, 4), axis=0).shape[0] > 5


def test_sidescan_colorizer_marks_nadir_and_preserves_contrast():
    waterfall = np.tile(np.arange(64, dtype=np.uint8), (20, 2))
    image = visualization.colorize_sidescan(waterfall)

    assert image.shape == (20, 128, 4)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(
        image[:, 64, :3],
        np.tile([21, 171, 196], (image.shape[0], 1)),
    )
    assert image[..., :3].max() > image[..., :3].min()


def test_ground_range_resampling_removes_water_column_sampling_holes():
    altitude = 40.0
    half_swath = np.deg2rad(85.0)
    angles = half_swath * (
        np.arange(256, dtype=np.float64) + 0.5
    ) / 256.0
    directions = np.column_stack(
        (
            np.zeros_like(angles),
            np.sin(angles),
            -np.cos(angles),
        )
    )
    ranges = altitude / np.cos(angles)
    hits = ranges <= 250.0
    attenuation = 0.003
    # Simulate the same two-way spreading and absorption terms used by the
    # geometric backend.  TVG should recover an approximately uniform bottom.
    intensities = (
        np.square(altitude / ranges)
        * np.exp(-2.0 * attenuation * (ranges - altitude))
    )

    power, coverage, measured_altitude, ground_extent = (
        visualization.resample_sidescan_ground_range(
            ranges,
            intensities,
            hits,
            directions,
            max_range=250.0,
            output_bins=640,
            attenuation=attenuation,
        )
    )

    np.testing.assert_allclose(measured_altitude, altitude, atol=1.0e-10)
    np.testing.assert_allclose(
        ground_extent,
        np.sqrt(250.0**2 - altitude**2),
        rtol=2.0e-3,
    )
    assert np.mean(coverage) > 0.98
    np.testing.assert_allclose(power[coverage], 1.0, rtol=2.0e-3)


def test_sidescan_tone_mapping_is_seeded_and_grayscale():
    ground_power = np.tile(np.linspace(0.08, 1.0, 128), (4, 2, 1))
    coverage = np.ones_like(ground_power, dtype=bool)
    first = visualization.tone_map_sidescan_power(
        ground_power,
        coverage,
        rng=np.random.default_rng(19),
        nadir_mute_bins=2,
    )
    second = visualization.tone_map_sidescan_power(
        ground_power,
        coverage,
        rng=np.random.default_rng(19),
        nadir_mute_bins=2,
    )

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8
    assert np.count_nonzero(first) > 0.9 * first.size
    rgba = visualization.colorize_sidescan(
        first[0], gamma=1.0, mark_nadir=False, palette="gray"
    )
    np.testing.assert_array_equal(rgba[..., 0], rgba[..., 1])
    np.testing.assert_array_equal(rgba[..., 1], rgba[..., 2])


def test_amber_sidescan_palette_is_monotonic_and_warm():
    waterfall = np.arange(256, dtype=np.uint8).reshape(1, -1)
    rgba = visualization.colorize_sidescan(
        waterfall,
        gamma=1.0,
        mark_nadir=False,
        palette="amber",
    )

    luminance = (
        0.2126 * rgba[0, :, 0]
        + 0.7152 * rgba[0, :, 1]
        + 0.0722 * rgba[0, :, 2]
    )
    assert np.all(np.diff(luminance) >= 0.0)
    np.testing.assert_array_equal(rgba[0, 0], [0, 0, 0, 255])
    assert rgba[0, 180, 0] > rgba[0, 180, 1] > rgba[0, 180, 2]
    assert rgba[0, -1, 0] > 250
    assert rgba[0, -1, 1] > 235


def test_bathymetry_renderer_draws_map_and_vehicle_track():
    x = np.linspace(-5.0, 5.0, 30)
    y = np.linspace(-3.0, 3.0, 20)
    xx, yy = np.meshgrid(x, y)
    zz = -8.0 + 0.4 * np.sin(xx) * np.cos(yy)
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)))
    track = np.column_stack((x, np.zeros_like(x), np.full_like(x, -2.0)))

    image = visualization.render_bathymetry_map(
        points, track, width=320, height=180
    )

    assert image.shape == (180, 320, 4)
    assert image.dtype == np.uint8
    assert np.unique(image.reshape(-1, 4), axis=0).shape[0] > 8
