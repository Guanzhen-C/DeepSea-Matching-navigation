from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from QiandaoNativeBottomProjection import native_bottom_samples


def project(angles, delays):
    magnitude = np.tile(np.arange(1, 101, dtype=float), (8, 1))
    return native_bottom_samples(angles, delays, magnitude, np.full_like(magnitude, np.log(.5)),
        side=0, sound_speed=2, sampling_frequency=1, installation_deg=30,
        beam_spacing_m=.15, vehicle_along_m=100., platform_depth_m=2.3)


def test_native_amplitude_is_not_interpolated_or_replaced_by_palette():
    # Constant30deg grazing, r20..80; physical depth gate10..60.
    angles = np.zeros((8, 4))
    delays = np.tile([20., 40., 60., 80.], (8, 1))
    xyz, value, beam, sample = project(angles, delays)
    np.testing.assert_allclose(value, (sample+1)*.5, atol=1e-13)
    np.testing.assert_allclose(xyz[:, 0], 100.+(beam-3.5)*.15, atol=1e-13)
    np.testing.assert_allclose(xyz[:, 1], (sample+1)*np.cos(np.pi/6), atol=1e-12)
    np.testing.assert_allclose(xyz[:, 2], -(sample+1)*.5-2.3, atol=1e-12)
    assert set(beam) == set(range(8))


def test_invalid_angle_gap_is_not_bridged_by_native_interpolation():
    angles = np.zeros((8, 5))
    angles[:, 2] = np.nan
    delays = np.tile([25., 35., 45., 55., 65.], (8, 1))
    xyz, _, _, sample = project(angles, delays)
    assert len(xyz) > 0
    assert not np.any(((sample+1) >= 35)&((sample+1) < 55))


def test_geometry_uses_both_real_angle_and_delay_and_is_not_flat():
    delays = np.tile([25., 35., 45., 55., 65.], (8, 1))
    angles = np.tile([0., 2., 4., 6., 8.], (8, 1))
    xyz, _, _, _ = project(angles, delays)
    assert np.ptp(xyz[:, 2]) > 10
    empty = project(np.full((8, 5), 80.), delays)
    assert empty[0].shape == (0, 3) and len(empty[1]) == 0
