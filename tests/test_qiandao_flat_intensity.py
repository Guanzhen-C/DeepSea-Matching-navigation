from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from QiandaoFlatIntensity import project_native_intensity, sample_mask, flat_fan_support


def test_only_native_amplitude_gain_and_assumed_flat_range_determine_map_values():
    # Set dr=1m, H=3m. Across=4m samples r=5m => native index4.
    profiles = np.tile(np.arange(1, 21, dtype=float), (8, 1))
    gain = np.full_like(profiles, np.log(.5))
    values, valid = project_native_intensity(profiles, gain, [0, 4, -4, 30],
        source_altitude_m=3, sound_speed=2, sampling_frequency=1)
    np.testing.assert_allclose(values[:, :3], np.tile([1.5, 2.5, 2.5], (8, 1)))
    assert valid[:, :3].all() and not valid[:, 3].any()
    assert np.isnan(values[:, 3]).all()


def test_zero_tail_remains_missing_not_filled_by_rgb_or_synthetic_texture():
    profiles = np.ones((8, 20))
    profiles[:, 8:] = 0
    values, valid = project_native_intensity(profiles, np.zeros_like(profiles), [0, 12],
        source_altitude_m=3, sound_speed=2, sampling_frequency=1)
    assert valid[:, 0].all() and not valid[:, 1].any()
    assert np.all(values[:, 1] == 0)
    with pytest.raises(ValueError):
        project_native_intensity(profiles, profiles[:4], [0], source_altitude_m=3,
                                 sound_speed=2, sampling_frequency=1)


def test_bilinear_support_rejects_holes_and_exterior():
    mask = np.ones((9, 9), bool)
    mask[4, 4] = False
    assert not sample_mask(mask, np.array([-.1]), np.array([-.1]), 1., all_four=True)[0]
    assert not sample_mask(mask, np.array([4.1]), np.array([0.]), 1., all_four=True)[0]
    assert sample_mask(mask, np.array([2.]), np.array([2.]), 1., all_four=True)[0]


def test_yaw_rotates_virtual_flat_fan_and_keeps_outside_returns_invalid():
    mask = np.ones((11, 11), bool)
    origins = np.zeros((2, 3))
    directions = np.array([[0, .6, -.8], [0, -.6, -.8]])
    hit, x, y = flat_fan_support(np.array([[0., 0.], [0., 0.], [5., 0.]]),
        np.array([[1., 0.], [0., 1.], [1., 0.]]), origins, directions,
        mask=mask, resolution=1, altitude_m=4, max_range_m=10)
    np.testing.assert_allclose(x[:2], [[0, 0], [-3, 3]], atol=1e-12)
    np.testing.assert_allclose(y[:2], [[3, -3], [0, 0]], atol=1e-12)
    assert hit[:2].all() and not hit[2].any()
