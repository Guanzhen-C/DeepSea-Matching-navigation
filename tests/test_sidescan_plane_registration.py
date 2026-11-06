from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanPlaneRegistration import (plane_residual_jacobian, fit_observed_scan_planes,
                                      fit_to_world_pose, observed_basis)


def make_case(normals, parameters=(2., -3., .03), seed=57):
    rng = np.random.default_rng(seed)
    p = np.column_stack([rng.uniform(-35., -5., len(normals)), rng.uniform(-100., 100., len(normals))])
    offsets = plane_residual_jacobian(parameters, p, normals, np.zeros(len(p)))[0]
    return p, offsets


def test_plane_analytic_jacobian_matches_independent_finite_difference():
    angles = np.linspace(-.5, .5, 30); n = np.column_stack([np.cos(angles), np.sin(angles)])
    p, o = make_case(n); parameter = np.array([2., -3., .03])
    _, analytic = plane_residual_jacobian(parameter, p, n, o)
    numeric = np.column_stack([(plane_residual_jacobian(parameter+np.eye(3)[k]*1e-6, p, n, o)[0]
                              -plane_residual_jacobian(parameter-np.eye(3)[k]*1e-6, p, n, o)[0])/2e-6 for k in range(3)])
    np.testing.assert_allclose(analytic, numeric, atol=2e-8)


def test_straight_or_opposite_scan_planes_retain_exact_lateral_null():
    n = np.tile([1., 0.], (80, 1)); n[::2] *= -1
    p, o = make_case(n)
    r = fit_observed_scan_planes(p, n, o)
    assert r['accepted'] and r['observable_rank'] == 2
    np.testing.assert_allclose(r['parameters'], [2., 0., .03], atol=1e-8)
    np.testing.assert_allclose(r['information']@np.array([0., 1., 0.]), 0., atol=1e-9)
    assert not r['optimizer_seed_information_included']


def test_diverse_actual_scan_planes_recover_cross_track_without_depth():
    angles = np.linspace(-.5, .5, 80); n = np.column_stack([np.cos(angles), np.sin(angles)])
    p, o = make_case(n); r = fit_observed_scan_planes(p, n, o)
    assert r['accepted'] and r['observable_rank'] == 3
    np.testing.assert_allclose(r['parameters'], [2., -3., .03], atol=1e-8)
    assert np.linalg.eigvalsh(r['information']).min() > 0
    assert not r['slant_range_used_as_horizontal']


def test_false_points_do_not_turn_straight_inliers_into_a_third_observation():
    n = np.tile([1., 0.], (100, 1)); n[-15:] = [0., 1.]
    p, o = make_case(n); o[-15:] += np.arange(15)*20+40
    r = fit_observed_scan_planes(p, n, o)
    assert r['accepted'] and r['observable_rank'] == 2
    assert r['inliers'] == 85
    np.testing.assert_allclose(r['parameters'][1], 0., atol=1e-8)


def test_world_precision_transform_matches_numerical_mapping_and_preserves_null():
    seed = np.array([10., 20., .3]); beta = np.array([2., -3., .02]); info = np.diag([1., 0., 100.])
    pose, world_info = fit_to_world_pose(seed, beta, info)
    jac = np.column_stack([(fit_to_world_pose(seed, beta+np.eye(3)[k]*1e-6, info)[0]
                          -fit_to_world_pose(seed, beta-np.eye(3)[k]*1e-6, info)[0])/2e-6 for k in range(3)])
    np.testing.assert_allclose(jac.T@world_info@jac, info, atol=2e-7)
    assert np.linalg.matrix_rank(world_info, tol=1e-8) == 2


def test_tiny_noisy_heading_variations_do_not_qualify_as_a_lateral_measurement():
    angles = np.linspace(-1e-5, 1e-5, 80)
    n = np.column_stack([np.cos(angles), np.sin(angles)])
    p, o = make_case(n); result = fit_observed_scan_planes(p, n, o)
    assert result['accepted'] and result['observable_rank'] == 2
    assert abs(result['parameters'][1]) < .01
    assert np.linalg.matrix_rank(result['information'], tol=1e-8) == 2


def test_arbitrary_parallel_scan_direction_has_its_physical_translation_null():
    angle = .4; normal = np.array([np.cos(angle), np.sin(angle)])
    n = np.tile(normal, (80, 1)); p, o = make_case(n)
    result = fit_observed_scan_planes(p, n, o)
    assert result['accepted'] and result['observable_rank'] == 2
    np.testing.assert_allclose(result['information']@np.r_[-normal[1], normal[0], 0.], 0., atol=1e-8)
