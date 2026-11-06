from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from AnalyzeSideScanDirectionalError import (directional_errors, absolute_information_summary,
    evaluate_loop_information, relative_pose, require_completed_inputs)


def test_directional_mse_is_the_actual_3d_metric_and_zero_error_is_not_nan():
    p = np.zeros((5, 4, 3)); p[:, 1] = [3., 4., 0.]
    error, result = directional_errors(p)
    assert error.shape == (5, 3, 3)
    assert result['DR']['RMSE_3d_m'] == 5.
    np.testing.assert_allclose(result['DR']['axis_MSE_fractions_xyz'], [.36, .64, 0.])
    assert result['Local']['axis_MSE_fractions_xyz'] == [0., 0., 0.]
    p[2, 2, 0] = np.nan
    with pytest.raises(ValueError): directional_errors(p)


def test_information_diagnostic_retains_exact_null_and_does_not_claim_marginal_covariance():
    priors = [dict(information=np.diag([4., 0., 100.]).tolist())]
    result = absolute_information_summary(priors)
    assert result['rank_counts'] == {'2': 1}
    assert result['Y_over_X_diagonal_ratio']['median'] == 0.
    assert result['information_diagonals_are_not_marginal_covariance']
    assert absolute_information_summary([])['count'] == 0
    with pytest.raises(ValueError): absolute_information_summary([dict(information=np.diag([-1., 0., 1.]))])


def test_loop_evaluation_uses_the_graph_coordinate_residual_and_retains_unobservable_large_error():
    poses = np.array([[1., 2., .5], [7., 3., 1.]])
    expected = relative_pose(poses[0], poses[1])
    bad_lateral = expected.copy(); bad_lateral[1] += 100.
    factors = dict(edges=[dict(kind='loop', source=0, target=1, measurement=bad_lateral,
                               information=np.diag([4., 0., 100.]))])
    result = evaluate_loop_information(factors, poses, np.array([0., 200.]))
    assert result['count'] == 1
    assert result['squared_residual_min_median_P95_max'] == [0., 0., 0., 0.]
    assert abs(result['largest_weighted_residuals'][0]['error_xy_yaw_rad'][1]) == 100.
    assert result['not_a_calibrated_chi_square_test']


def test_directional_diagnostic_requires_terminal_scope_and_matching_independent_review_before_truth_io():
    original = dict(status='completed_pure_side_scan_FEATURE_route_test')
    review = dict(status='independently_verified_complete_executed_scope', source_run_SHA256='exact')
    require_completed_inputs(original, review, 'exact')
    require_completed_inputs(original, review, 'exact', dict(status='completed_lossless_navigation_input_replay'))
    with pytest.raises(ValueError, match='completed'): require_completed_inputs(dict(status='running'), review, 'exact')
    with pytest.raises(ValueError, match='completed'): require_completed_inputs(original, review, 'changed')
    with pytest.raises(ValueError, match='partial replay'): require_completed_inputs(original, review, 'exact', dict(status='running_lossless_navigation_input_replay'))
