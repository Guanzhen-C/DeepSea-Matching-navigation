from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanLoopGeometry import (rotation, range_plane_residual_jacobian, data_only_schur_information,
                                  fit_range_plane_loop, inflate_information, lift_loop_to_graph_nodes)
from SideScanFeatureMap import pose_matrix, matrix_pose


def synthetic_observations(pose, n=24):
    rng = np.random.default_rng(58)
    landmarks = np.column_stack([rng.uniform(-35, -5, n), rng.uniform(40, 100, n), rng.uniform(-40, -15, n)])
    transformed = np.column_stack([(landmarks[:, :2]-pose[:2])@rotation(pose[2]), landmarks[:, 2]])
    a_origins = np.column_stack([landmarks[:, 0], np.zeros((n, 2))])
    b_origins = np.column_stack([transformed[:, 0], np.zeros((n, 2))])
    a = dict(origins=a_origins, normals=np.tile([1., 0.], (n, 1)),
             ranges=np.linalg.norm(landmarks-a_origins, axis=1), sides=np.sign(landmarks[:, 1]))
    b = dict(origins=b_origins, normals=np.tile([1., 0.], (n, 1)),
             ranges=np.linalg.norm(transformed-b_origins, axis=1), sides=np.sign(transformed[:, 1]))
    return landmarks, a, b


def test_range_plane_analytic_jacobian_matches_independent_numerical_difference():
    pose = np.array([-30., 28., .31]); landmarks, a, b = synthetic_observations(pose, 8)
    residual, jac = range_plane_residual_jacobian(pose, landmarks, a, b, 0.)
    x = np.r_[pose, landmarks.ravel()]; numeric = np.empty_like(jac)
    for k in range(len(x)):
        plus, minus = x.copy(), x.copy(); plus[k] += 1e-6; minus[k] -= 1e-6
        rp, _ = range_plane_residual_jacobian(plus[:3], plus[3:].reshape(-1, 3), a, b, 0.)
        rm, _ = range_plane_residual_jacobian(minus[:3], minus[3:].reshape(-1, 3), a, b, 0.)
        numeric[:, k] = (rp-rm)/2e-6
    np.testing.assert_allclose(jac, numeric, atol=3e-8)
    np.testing.assert_allclose(residual, 0., atol=1e-12)


def test_parallel_tracks_cannot_manufacture_lateral_information_from_unknown_heights():
    pose = np.array([-30., 28., np.pi]); landmarks, a, b = synthetic_observations(pose)
    _, j = range_plane_residual_jacobian(pose, landmarks, a, b, 0.)
    info = data_only_schur_information(j, len(landmarks))
    assert np.linalg.matrix_rank(info, tol=1e-6) == 2
    np.testing.assert_allclose(info@np.array([0., 1., 0.]), 0., atol=1e-7)
    shrunk = inflate_information(info, np.diag([4., 4., .01]))
    np.testing.assert_allclose(shrunk@np.array([0., 1., 0.]), 0., atol=1e-7)
    with pytest.raises(ValueError): data_only_schur_information(np.vstack([j, np.zeros((3, j.shape[1]))]), len(landmarks))


def test_two_straight_views_retain_height_pose_ambiguity_even_at_different_headings():
    pose = np.array([-30., 28., .5]); landmarks, a, b = synthetic_observations(pose)
    _, j = range_plane_residual_jacobian(pose, landmarks, a, b, 0.)
    info = data_only_schur_information(j, len(landmarks))
    assert np.linalg.matrix_rank(info, tol=1e-6) == 2
    null = np.array([-pose[1], pose[1]/np.tan(pose[2]), 1.])
    np.testing.assert_allclose(info@null, 0., atol=1e-6)


def test_diverse_observed_scan_planes_can_resolve_more_than_two_straight_views():
    pose = np.array([-30., 28., .5]); landmarks, a, b = synthetic_observations(pose)
    angles = np.linspace(-.2, .2, len(landmarks))
    a['normals'] = np.column_stack([np.cos(angles), np.sin(angles)])
    b['normals'] = np.column_stack([np.cos(angles[::-1]), np.sin(angles[::-1])])
    pb = np.column_stack([(landmarks[:, :2]-pose[:2])@rotation(pose[2]), landmarks[:, 2]])
    for observation, local in [(a, landmarks), (b, pb)]:
        offset = np.einsum('ij,ij->i', observation['normals'], local[:, :2])
        observation['origins'][:, :2] = observation['normals']*offset[:, None]
        observation['ranges'] = np.linalg.norm(local-observation['origins'], axis=1)
    _, j = range_plane_residual_jacobian(pose, landmarks, a, b, 0.)
    assert np.linalg.matrix_rank(data_only_schur_information(j, len(landmarks)), tol=1e-6) == 3


def test_unknown_height_solver_uses_data_only_information_and_recovers_observable_components():
    pose = np.array([-30., 28., np.pi]); _, a, b = synthetic_observations(pose, 12)
    result = fit_range_plane_loop(a, b, pose+np.array([1., 3., .01]))
    assert result['accepted']
    assert abs(result['relative_pose2'][0]-pose[0]) < .1
    assert abs(np.arctan2(np.sin(result['relative_pose2'][2]-pose[2]), np.cos(result['relative_pose2'][2]-pose[2]))) < .002
    assert not result['landmark_depth_prior_used']
    assert not result['motion_seed_information_added_to_measurement']


def test_frame_to_keyframe_lift_preserves_relative_transform_and_information_null():
    m = np.array([-30., 28., np.pi]); info = np.diag([4., 0., 100.])
    a, b = pose_matrix([7., 1., .2]), pose_matrix([3., -1., -.1])
    measurement, lifted = lift_loop_to_graph_nodes(m, info, a, b)
    np.testing.assert_allclose(pose_matrix(measurement), a@pose_matrix(m)@np.linalg.inv(b), atol=1e-12)
    assert np.linalg.matrix_rank(lifted, tol=1e-8) == 2


def test_one_wrong_physical_side_match_is_excluded_not_used_to_reject_or_fake_entire_loop():
    pose = np.array([-30., 28., np.pi]); _, a, b = synthetic_observations(pose, 30)
    b['sides'][0] *= -1  # explicit false correspondence; no true pose used by solver
    result = fit_range_plane_loop(a, b, pose+np.array([1., 3., .01]), minimum_features=24)
    assert result['accepted'] and result['outlier_refinement_performed']
    assert 0 not in result['selected_feature_indices']
    assert result['features'] == 29
    assert abs(result['relative_pose2'][0]-pose[0]) < .1
    assert not result['motion_seed_information_added_to_measurement']


def test_refinement_never_accepts_less_than_required_physical_support():
    pose = np.array([-30., 28., np.pi]); _, a, b = synthetic_observations(pose, 24)
    b['sides'][0] *= -1
    result = fit_range_plane_loop(a, b, pose+np.array([1., 3., .01]), minimum_features=24)
    assert not result['accepted']
