from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from QiandaoLocalRaycaster import intersect_local_grid


class FlatGrid:
    elevation_m = np.full((3, 3), -20.)
    ray_support_mask = np.ones((3, 3), bool)

    @staticmethod
    def _sample(values, x, y, *, valid_mask):
        return np.where((abs(x) <= 30) & (abs(y) <= 30), -20., np.nan)

    @staticmethod
    def normal(x, y):
        return np.tile([0., 0., 1.], (np.size(x), 1))


def test_surface_before_map_exit_found_even_when_max_range_endpoint_is_nodata():
    hit, ranges, xyz, _ = intersect_local_grid(FlatGrid(), np.zeros(3),
        np.array([[1., 0., -1.], [0., 0., -1.], [1., 0., 1.]]), min_range=0., max_range=250.)
    np.testing.assert_array_equal(hit, [True, True, False])
    np.testing.assert_allclose(ranges[:2], [20*np.sqrt(2), 20.], atol=1e-6)
    np.testing.assert_allclose(xyz[0], [20., 0., -20.], atol=1e-6)
    assert np.isnan(ranges[2]) and np.isnan(xyz[2]).all()


def test_absent_bottom_is_never_filled_or_assigned_fake_returns():
    class MissingGrid(FlatGrid):
        @staticmethod
        def _sample(values, x, y, *, valid_mask):
            return np.full(np.shape(x), np.nan)
    hit, ranges, xyz, intensity = intersect_local_grid(MissingGrid(), np.zeros(3),
        np.array([[0., 0., -1.]]), min_range=0., max_range=250.)
    assert not hit.any()
    assert np.isnan(ranges).all() and np.isnan(xyz).all()
    assert not intensity.any()
