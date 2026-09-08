from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "isaacsim"
    / "oceansim"
    / "utils"
    / "acoustic_sonar.py"
)
SPEC = importlib.util.spec_from_file_location("oceansim_acoustic_sonar", MODULE_PATH)
acoustic_sonar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acoustic_sonar
SPEC.loader.exec_module(acoustic_sonar)


def test_multibeam_fan_uses_downlooking_ros_frame():
    angles, directions = acoustic_sonar.multibeam_directions(3, 120.0)

    np.testing.assert_allclose(np.rad2deg(angles), [-60.0, 0.0, 60.0])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert directions[0, 1] > 0.0  # port is +Y
    np.testing.assert_allclose(directions[1], [0.0, 0.0, -1.0], atol=1e-12)
    assert directions[-1, 1] < 0.0  # starboard is -Y


def test_integrated_fan_selects_bathymetry_from_one_raw_echo_grid():
    (
        shared,
        bathymetry_source,
        imaging,
        imaging_source,
        imaging_angles,
    ) = acoustic_sonar.integrated_bathymetric_sidescan_fan(7, 120.0, 32)
    assert imaging.shape == (64, 3)
    assert shared.shape == imaging.shape
    np.testing.assert_array_equal(imaging_source, np.arange(64))
    assert bathymetry_source.shape == (7,)
    assert np.unique(bathymetry_source).size == 7
    assert np.all(np.isin(bathymetry_source, imaging_source))
    np.testing.assert_allclose(shared[imaging_source], imaging, atol=1.0e-14)
    assert np.all(imaging_angles[:32] < 0.0)
    assert np.all(imaging_angles[32:] > 0.0)
    assert np.all(imaging[:32, 1] > 0.0)
    assert np.all(imaging[32:, 1] < 0.0)
    assert np.max(np.abs(imaging_angles)) < np.deg2rad(60.0)


def test_integrated_fan_supports_eight_beam_per_side_3d_imaging():
    rays_per_side = 5
    beams_per_side = 8
    (
        shared,
        bathymetry_source,
        imaging,
        imaging_source,
        imaging_angles,
    ) = acoustic_sonar.integrated_bathymetric_sidescan_fan(
        12,
        120.0,
        rays_per_side,
        along_track_beams_per_side=beams_per_side,
        along_track_beam_spacing_deg=0.14,
    )

    samples_per_side = beams_per_side * rays_per_side
    assert imaging.shape == (2 * samples_per_side, 3)
    assert imaging_angles.shape == (2 * samples_per_side,)
    np.testing.assert_allclose(np.linalg.norm(imaging, axis=1), 1.0)
    np.testing.assert_allclose(
        shared[imaging_source], imaging, atol=1.0e-14
    )
    assert bathymetry_source.shape == (12,)
    assert np.unique(bathymetry_source).size == 12
    assert np.all(np.isin(bathymetry_source, imaging_source))
    np.testing.assert_array_equal(imaging_source, np.arange(80))

    port = imaging[:samples_per_side].reshape(
        beams_per_side, rays_per_side, 3
    )
    starboard = imaging[samples_per_side:].reshape(
        beams_per_side, rays_per_side, 3
    )
    assert np.all(port[:, :, 1] > 0.0)
    assert np.all(starboard[:, :, 1] < 0.0)
    assert np.all(imaging[:, 2] < 0.0)
    # Even beam counts have no zero-angle centre: every imaging ray therefore
    # has a genuine along-track component, symmetric fore/aft by beam index.
    assert np.all(np.abs(imaging[:, 0]) > 0.0)
    np.testing.assert_allclose(port[:, 0, 0], -port[::-1, 0, 0])
    np.testing.assert_allclose(
        starboard[:, 0, 0], -starboard[::-1, 0, 0]
    )
    # Beam-major ordering repeats one side's cross-track fan for every beam.
    for beam_index in range(1, beams_per_side):
        np.testing.assert_allclose(
            imaging_angles[
                beam_index * rays_per_side : (beam_index + 1) * rays_per_side
            ],
            imaging_angles[:rays_per_side],
        )


def test_integrated_multiline_fan_requires_positive_along_track_spacing():
    with np.testing.assert_raises_regex(
        ValueError, "spacing_deg must be > 0"
    ):
        acoustic_sonar.integrated_bathymetric_sidescan_fan(
            12,
            120.0,
            5,
            along_track_beams_per_side=8,
        )


@pytest.mark.parametrize(
    ("spacing_m", "coverage_m"), ((0.15, 1.2), (0.075, 0.6))
)
def test_parallel_multiline_rays_use_spatial_phase_centres(
    spacing_m: float, coverage_m: float
):
    rays_per_side = 5
    rows_per_side = 8
    (
        shared_origins,
        shared_directions,
        bathymetry_source,
        imaging_origins,
        imaging_directions,
        imaging_source,
        imaging_angles,
    ) = acoustic_sonar.parallel_bathymetric_sidescan_rays(
        12,
        120.0,
        rays_per_side,
        rows_per_side,
        spacing_m,
    )

    samples_per_side = rows_per_side * rays_per_side
    assert shared_origins.shape == shared_directions.shape == (
        2 * samples_per_side,
        3,
    )
    assert bathymetry_source.shape == (12,)
    assert np.unique(bathymetry_source).size == 12
    np.testing.assert_array_equal(imaging_source, np.arange(2 * samples_per_side))
    assert np.all(np.isin(bathymetry_source, imaging_source))
    np.testing.assert_allclose(shared_origins[imaging_source], imaging_origins)
    np.testing.assert_allclose(
        shared_directions[imaging_source], imaging_directions
    )

    port_origins = imaging_origins[:samples_per_side].reshape(
        rows_per_side, rays_per_side, 3
    )
    starboard_origins = imaging_origins[samples_per_side:].reshape(
        rows_per_side, rays_per_side, 3
    )
    row_centres = port_origins[:, 0, 0]
    np.testing.assert_allclose(np.diff(row_centres), spacing_m)
    assert row_centres[0] - 0.5 * spacing_m == pytest.approx(
        -0.5 * coverage_m
    )
    assert row_centres[-1] + 0.5 * spacing_m == pytest.approx(
        0.5 * coverage_m
    )
    np.testing.assert_allclose(starboard_origins, port_origins)
    np.testing.assert_array_equal(port_origins[:, :, 1:], 0.0)

    port_directions = imaging_directions[:samples_per_side].reshape(
        rows_per_side, rays_per_side, 3
    )
    starboard_directions = imaging_directions[samples_per_side:].reshape(
        rows_per_side, rays_per_side, 3
    )
    for row in range(1, rows_per_side):
        np.testing.assert_allclose(port_directions[row], port_directions[0])
        np.testing.assert_allclose(
            starboard_directions[row], starboard_directions[0]
        )
    np.testing.assert_array_equal(imaging_directions[:, 0], 0.0)
    assert imaging_angles.shape == (2 * samples_per_side,)


def test_broadcast_ray_origins_preserves_legacy_and_per_ray_forms():
    shared = acoustic_sonar.broadcast_ray_origins([1.0, 2.0, 3.0], 2)
    explicit = acoustic_sonar.broadcast_ray_origins(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 2
    )

    np.testing.assert_array_equal(shared, [[1.0, 2.0, 3.0]] * 2)
    np.testing.assert_array_equal(
        explicit, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    with pytest.raises(ValueError, match="origin_world must have shape"):
        acoustic_sonar.broadcast_ray_origins(np.zeros((3, 3)), 2)
    with pytest.raises(ValueError, match="finite"):
        acoustic_sonar.broadcast_ray_origins([0.0, np.nan, 0.0], 1)


def test_sidescan_fans_point_to_opposite_sides_and_down():
    port, starboard = acoustic_sonar.sidescan_directions(
        rays_per_side=5,
        tilt_deg=45.0,
        vertical_beam_width_deg=40.0,
        along_track_beam_width_deg=2.0,
        along_track_samples=3,
    )

    assert port.shape == starboard.shape == (15, 3)
    assert np.all(port[:, 1] > 0.0)
    assert np.all(starboard[:, 1] < 0.0)
    assert np.all(port[:, 2] < 0.0)
    assert np.all(starboard[:, 2] < 0.0)
    np.testing.assert_allclose(np.linalg.norm(port, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(starboard, axis=1), 1.0)


def test_backscatter_respects_range_and_incidence():
    intensity = acoustic_sonar.compute_backscatter_intensity(
        distances=[1.0, 2.0, 1.0],
        directions_world=np.array(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
        ),
        normals_world=np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        reflectivities=[1.0, 1.0, 1.0],
        hit_mask=[True, True, True],
        attenuation=0.0,
    )

    assert intensity[0] > intensity[1] > 0.0
    assert intensity[2] == 0.0


def test_sidescan_binning_and_display_order():
    centres, binned = acoustic_sonar.bin_sidescan_returns(
        ranges=[1.0, 2.1, 2.2, 5.0, np.nan],
        intensities=[0.2, 0.4, 0.8, 1.0, 1.0],
        hit_mask=[True, True, True, True, False],
        min_range=1.0,
        max_range=5.0,
        num_bins=4,
    )

    np.testing.assert_allclose(centres, [1.5, 2.5, 3.5, 4.5])
    np.testing.assert_allclose(binned, [0.2, 0.8, 0.0, 1.0])
    line = acoustic_sonar.compose_sidescan_line(
        port=[0.0, 0.5, 1.0], starboard=[0.1, 0.6, 0.9]
    )
    np.testing.assert_allclose(line, [1.0, 0.5, 0.0, 0.1, 0.6, 0.9])


def test_noise_is_seeded_and_bounded():
    kwargs = dict(
        signal=np.linspace(0.0, 1.0, 16),
        range_bin_centres=np.linspace(1.0, 20.0, 16),
        max_range=20.0,
        multiplicative_std=0.1,
        additive_std=0.05,
    )
    first = acoustic_sonar.add_sonar_noise(
        rng=np.random.default_rng(7), **kwargs
    )
    second = acoustic_sonar.add_sonar_noise(
        rng=np.random.default_rng(7), **kwargs
    )

    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_standard_payload_shapes():
    ping = acoustic_sonar.MultibeamPing(
        timestamp=1.25,
        frame_id="mbes",
        sound_speed=1500.0,
        frequency=400_000.0,
        beam_angles=np.array([-0.1, 0.1]),
        beam_directions=np.array([[0.0, 0.1, -0.99], [0.0, -0.1, -0.99]]),
        ranges=np.array([10.0, np.nan]),
        two_way_travel_times=np.array([2.0 / 150.0, np.nan]),
        intensities=np.array([1.0, 0.0]),
        hit_mask=np.array([True, False]),
        points_sensor=np.array([[0.0, 1.0, -9.9], [np.nan] * 3]),
        points_world=np.array([[0.0, 1.0, -9.9], [np.nan] * 3]),
    )
    payload = ping.as_sonar_ranges_dict()

    assert payload["ping_info"]["sound_speed"] == 1500.0
    assert payload["beam_unit_vec"].shape == (2, 3)
    np.testing.assert_array_equal(payload["flags"], [0, 1])
    detections = ping.as_sonar_detections_dict()
    np.testing.assert_allclose(detections["rx_angles"], [-0.1, 0.1])
    np.testing.assert_allclose(
        detections["two_way_travel_times"][:1], [2.0 / 150.0]
    )

    side_ping = acoustic_sonar.SideScanPing(
        timestamp=2.0,
        frame_id="sidescan",
        sound_speed=1500.0,
        frequency=900_000.0,
        range_bin_centres=np.array([1.5, 2.5, 3.5]),
        port=np.array([0.1, 0.2, 0.3]),
        starboard=np.array([0.4, 0.5, 0.6]),
        display_line=np.zeros(6, dtype=np.uint8),
        port_points_world=np.empty((0, 3)),
        starboard_points_world=np.empty((0, 3)),
    )
    raw_payload = side_ping.as_raw_sonar_image_dict()
    np.testing.assert_allclose(raw_payload["rx_angles"], [np.pi / 2, -np.pi / 2])
    assert raw_payload["image"]["beam_count"] == 2
    assert len(raw_payload["image"]["data"]) == 2 * 3 * 4


def test_raw_sonar_image_supports_eight_beams_per_side():
    sample_count = 12_500
    range_centres = (np.arange(sample_count, dtype=np.float64) + 0.5) * 0.02
    port_beams = np.arange(8 * sample_count, dtype=np.float32).reshape(
        8, sample_count
    )
    starboard_beams = port_beams + 1_000_000.0
    side_ping = acoustic_sonar.SideScanPing(
        timestamp=2.0,
        frame_id="sidescan",
        sound_speed=1500.0,
        frequency=400_000.0,
        range_bin_centres=range_centres,
        port=port_beams.mean(axis=0),
        starboard=starboard_beams.mean(axis=0),
        display_line=np.zeros(2 * sample_count, dtype=np.uint8),
        port_points_world=np.empty((0, 3)),
        starboard_points_world=np.empty((0, 3)),
        port_beams=port_beams,
        starboard_beams=starboard_beams,
    )

    payload = side_ping.as_raw_sonar_image_dict()

    assert payload["sample_rate"] == 37_500.0
    assert payload["samples_per_beam"] == sample_count
    assert payload["image"]["beam_count"] == 16
    assert len(payload["image"]["data"]) == 16 * sample_count * 4
    assert payload["tx_delays"].shape == (16,)
    assert payload["tx_angles"].shape == (16,)
    np.testing.assert_allclose(payload["rx_angles"][:8], np.pi / 2.0)
    np.testing.assert_allclose(payload["rx_angles"][8:], -np.pi / 2.0)
    decoded = np.frombuffer(payload["image"]["data"], dtype=np.float32).reshape(
        16, sample_count
    )
    np.testing.assert_array_equal(decoded[:8], port_beams)
    np.testing.assert_array_equal(decoded[8:], starboard_beams)


def test_raw_sonar_image_rejects_incomplete_or_mismatched_beam_matrices():
    kwargs = dict(
        timestamp=2.0,
        frame_id="sidescan",
        sound_speed=1500.0,
        frequency=400_000.0,
        range_bin_centres=np.array([0.01, 0.03, 0.05]),
        port=np.zeros(3),
        starboard=np.zeros(3),
        display_line=np.zeros(6, dtype=np.uint8),
        port_points_world=np.empty((0, 3)),
        starboard_points_world=np.empty((0, 3)),
    )
    incomplete = acoustic_sonar.SideScanPing(
        **kwargs, port_beams=np.zeros((8, 3))
    )
    with np.testing.assert_raises_regex(ValueError, "supplied together"):
        incomplete.as_raw_sonar_image_dict()

    mismatched = acoustic_sonar.SideScanPing(
        **kwargs,
        port_beams=np.zeros((8, 3)),
        starboard_beams=np.zeros((8, 2)),
    )
    with np.testing.assert_raises_regex(ValueError, "same shape"):
        mismatched.as_raw_sonar_image_dict()

def test_imaging_lineage_translation_keeps_exact_sparse_sources() -> None:
    from oceansim.utils.acoustic_sonar import imaging_lineage_to_bathymetry

    imaging_sources = np.array([2, 4, 7, 9, 12], dtype=np.int64)
    bathymetry_sources = np.array([1, 4, 8, 12], dtype=np.int64)
    lineage = np.array([-1, 0, 1, 2, 4, 3, -1], dtype=np.int64)

    translated = imaging_lineage_to_bathymetry(
        lineage, imaging_sources, bathymetry_sources
    )

    np.testing.assert_array_equal(translated, [-1, -1, 1, -1, 3, -1, -1])
