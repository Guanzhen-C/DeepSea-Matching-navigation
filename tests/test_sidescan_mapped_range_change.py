from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanMappedRangeChange import (MappedRangeChange, conservative_range_change_sigma,
                                      profiled_range_residual_jacobians, rotation, world_origins)


def case(za=-8., zb=-8., offset=True, heights=None):
    n = 32
    a = np.array([3., 2., .08]); b = np.array([8., 30., -.13])
    xy = np.column_stack([np.linspace(-15., 23., n), np.linspace(40., 145., n)])
    z = -np.linspace(18., 65., n) if heights is None else np.asarray(heights)
    oa = np.column_stack([np.linspace(-7., 1., n), np.zeros(n)]) if offset else np.zeros((n, 2))
    ob = np.column_stack([np.linspace(-5., 2., n), np.zeros(n)]) if offset else np.zeros((n, 2))
    pa = world_origins(a, oa)[0]; pb = world_origins(b, ob)[0]
    ra = np.sqrt(np.sum((xy-pa)**2, axis=1)+(z-za)**2)
    rb = np.sqrt(np.sum((xy-pb)**2, axis=1)+(z-zb)**2)
    data = MappedRangeChange(xy, oa, ob, np.full(n, za), np.full(n, zb), ra, rb,
                            conservative_range_change_sigma(ra, rb, np.linalg.norm(b[:2]-a[:2])))
    return a, b, data


@pytest.mark.parametrize('dz', [0., .01, -.3, 3., -8.])
def test_arbitrary_unknown_nonflat_heights_remain_physically_compatible(dz):
    a, b, d = case(zb=-8.+dz)
    e, _, _ = profiled_range_residual_jacobians(a, b, d)
    np.testing.assert_allclose(e, 0, atol=2e-13)
    assert not d.payload()['DEM_or_altitude_used']


def test_equal_depth_has_real_cross_track_information_without_bottom_height():
    a, b, d = case()
    e, ja, jb = profiled_range_residual_jacobians(a, b, d)
    np.testing.assert_allclose(e, 0, atol=2e-13)
    assert np.linalg.norm(jb[:, 1]) > .1
    shifted = b+np.array([0., 10., 0.])
    assert np.linalg.norm(profiled_range_residual_jacobians(a, shifted, d)[0]) > 1


@pytest.mark.parametrize('robust', [False, True])
def test_analytic_world_jacobians_agree_with_independent_finite_difference(robust):
    a, b, d = case(zb=-8.04)
    a += [.2, -2., -.02]; b += [-.3, 7., .04]
    _, ja, jb = profiled_range_residual_jacobians(a, b, d, robust=robust)
    for which, jac in [(0, ja), (1, jb)]:
        numeric = []
        for k in range(3):
            epsilon = np.eye(3)[k]*1e-6
            aa, bb = (a+epsilon, b) if which == 0 else (a, b+epsilon)
            ac, bc = (a-epsilon, b) if which == 0 else (a, b-epsilon)
            numeric.append((profiled_range_residual_jacobians(aa, bb, d, robust=robust)[0]
                            -profiled_range_residual_jacobians(ac, bc, d, robust=robust)[0])/2e-6)
        np.testing.assert_allclose(jac, np.column_stack(numeric), rtol=2e-6, atol=2e-8)


def test_nonzero_depth_change_keeps_a_true_ambiguity_interval():
    a, b, d = case(zb=-3.)
    # A tiny perturbation is genuinely compatible with some unknown heights.
    e, ja, jb = profiled_range_residual_jacobians(a, b+np.array([0., .001, 0.]), d)
    np.testing.assert_array_equal(e, np.zeros(d.features))
    np.testing.assert_array_equal(ja, np.zeros_like(ja))
    np.testing.assert_array_equal(jb, np.zeros_like(jb))


def test_common_map_xy_translation_is_not_a_fake_absolute_range_measurement():
    a, b, d = case()
    delta = np.array([100., -50.])
    shifted = MappedRangeChange(d.landmark_xy+delta, d.origin_a, d.origin_b,
        d.sensor_z_a, d.sensor_z_b, d.range_a, d.range_b, d.standard_deviation_m)
    aa, bb = a.copy(), b.copy(); aa[:2] += delta; bb[:2] += delta
    np.testing.assert_allclose(profiled_range_residual_jacobians(a, b, d)[0],
                               profiled_range_residual_jacobians(aa, bb, shifted)[0], atol=1e-12)


def test_zero_baseline_retains_joint_translation_null():
    a, _, d = case(offset=False)
    d = MappedRangeChange(d.landmark_xy, d.origin_a, d.origin_a, d.sensor_z_a,
                         d.sensor_z_a, d.range_a, d.range_a, d.standard_deviation_m)
    e, ja, jb = profiled_range_residual_jacobians(a, a, d)
    np.testing.assert_allclose(e, 0, atol=1e-12)
    np.testing.assert_allclose((ja+jb)[:, :2], 0, atol=1e-12)


def test_many_features_are_not_counted_as_unlimited_independent_information():
    a, b, d = case()
    def duplicate(count):
        kwargs = {k: np.tile(v, (count, 1)) if v.ndim == 2 else np.tile(v, count)
                  for k, v in d.__dict__.items() if isinstance(v, np.ndarray)}
        return MappedRangeChange(**kwargs)
    _, _, j1 = profiled_range_residual_jacobians(a, b, duplicate(1))
    _, _, j4 = profiled_range_residual_jacobians(a, b, duplicate(4))
    np.testing.assert_allclose(j1.T@j1, j4.T@j4, atol=1e-12)


def test_observations_are_immutable_and_invalid_inputs_refused():
    _, _, d = case()
    with pytest.raises(ValueError): d.range_a[0] = 0
    with pytest.raises(ValueError):
        MappedRangeChange(d.landmark_xy, d.origin_a, d.origin_b, d.sensor_z_a, d.sensor_z_b,
                         d.range_a, d.range_b, np.zeros(d.features))
