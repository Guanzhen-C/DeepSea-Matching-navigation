from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from AnalyzeQiandaoNoGPSNavigation import relative_errors


def test_relative_error_is_invariant_to_a_constant_SE3_gauge_offset():
    n = 100
    truth = np.column_stack((np.arange(n)*.5, np.sin(np.arange(n)*.1), np.ones(n)*-8.))
    angle = .4
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    estimate = truth@r.T+[20., -7., 1.]
    positions = np.stack((truth, estimate, estimate, estimate), axis=1)
    rotations = np.repeat(np.stack((np.eye(3), r, r, r))[None], n, axis=0)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
    values = relative_errors(positions, rotations, distance, 10.)
    for v in values.values():
        assert v['samples'] > 50
        assert v['translation_RMSE_m'] < 1e-12
        assert v['rotation_RMSE_deg'] < 1e-6


def test_relative_error_detects_velocity_scale_error_not_just_absolute_offset():
    n = 100
    truth = np.column_stack((np.arange(n)*.5, np.zeros(n), np.ones(n)*-8.))
    estimate = truth.copy()
    estimate[:, 0] *= 1.1
    positions = np.stack((truth, estimate, estimate, estimate), axis=1)
    rotations = np.tile(np.eye(3), (n, 4, 1, 1))
    distance = np.arange(n)*.5
    values = relative_errors(positions, rotations, distance, 10.)
    for v in values.values():
        np.testing.assert_allclose(v['translation_RMSE_m'], 1., atol=1e-12)
