from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from QiandaoLocalScene import (bottom_points, sample_native_profiles,
                              grid_average, bounded_interpolation, physical_partition, reject_gps_file)


def test_synchronous_beams_have_spatial_offsets_not_time_advancement():
    angles = np.zeros((8, 10))
    delay = np.full((8, 10), .06)
    points, ranges, beam, _ = bottom_points(angles, delay, side=0,
        sound_speed=1480., installation_deg=30., beam_spacing_m=.15,
        vehicle_along_m=100., platform_depth_m=2.3)
    np.testing.assert_allclose(points[:, 0], 100.+(beam-3.5)*.15)
    np.testing.assert_allclose(points[:, 2], -ranges*.5-2.3)
    assert np.all(points[:, 1] > 0)


def test_port_and_starboard_geometry_have_opposite_y_same_range_and_z():
    args = dict(sound_speed=1480., installation_deg=30., beam_spacing_m=.15,
                vehicle_along_m=4., platform_depth_m=2.3)
    angle = np.zeros((8, 5))
    delay = np.full((8, 5), .06)
    port, *_ = bottom_points(angle, delay, side=0, **args)
    stbd, *_ = bottom_points(angle, delay, side=1, **args)
    np.testing.assert_allclose(port[:, [0, 2]], stbd[:, [0, 2]])
    np.testing.assert_allclose(port[:, 1], -stbd[:, 1])


def test_native_sample_origin_is_one_not_zero_and_does_not_use_rgb():
    a = np.tile(np.arange(100, dtype=float), (8, 1))
    b = np.array([0, 4, 7, 1])
    r = np.array([1., 2.5, 100., 101.])*1480./(2*37650.)
    out, ok = sample_native_profiles(a, b, r, sound_speed=1480., sampling_frequency=37650.)
    np.testing.assert_allclose(out[:3], [0., 1.5, 99.])
    assert not ok[-1] and np.isnan(out[-1])


def test_physical_record_holdout_never_reuses_same_id():
    ref, query = physical_partition(np.arange(128, 608))
    assert len(ref) == len(query) == 240
    assert not np.intersect1d(ref, query).size
    with pytest.raises(ValueError):
        physical_partition([1, 1, 2])


def test_grid_scatter_duplicates_average_and_preserve_nodata():
    xyz = np.array([[0., 0., -20.], [0., 0., -24.], [100., 0., -40.]])
    field, count = grid_average(xyz, {'z': xyz[:, 2]}, shape=(5, 5), resolution=1.)
    assert count[2, 2] == 2 and count.sum() == 2
    assert field['z'][2, 2] == -22.
    assert np.isnan(field['z'][0, 0])
    out, mask = bounded_interpolation(field['z'], count, resolution=1., max_distance_m=1.)
    assert mask.sum() == 5
    assert np.isnan(out[0, 0])
    assert out[2, 2] == -22.


def test_invalid_angle_delay_geometry_not_filled_or_fabricated():
    a = np.full((8, 5), np.nan)
    points, *_ = bottom_points(a, a, side=0, sound_speed=1480.,
        installation_deg=30., beam_spacing_m=.15, vehicle_along_m=0., platform_depth_m=2.3)
    assert points.shape == (0, 3)


def test_gps_file_input_denied_but_nogps_output_and_source_allowed():
    with pytest.raises(PermissionError):
        reject_gps_file('open', ('/somewhere/GPS_data.mat', 'r', 0))
    reject_gps_file('open', ('/somewhere/qiandao_nogps/scene_report.json', 'w', 0))
    reject_gps_file('open', ('/somewhere/PortNormalModeDepthTotal.mat', 'r', 0))
