from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from AnalyzePureSideScanShort import summarize, planar_relative_metrics


def test_independent_metrics_are_synchronized_3D_norms_not_xy_or_mean_norm():
    positions = np.zeros((2, 4, 3))
    positions[:, 1] = [[3, 4, 12], [0, 0, 0]]
    positions[:, 2] = [[0, 0, 2], [0, 0, 4]]
    positions[:, 3] = [[0, 0, 1], [0, 0, 1]]
    metrics = summarize(positions)
    assert metrics['DR']['RMSE_m'] == pytest.approx(np.sqrt(169/2))
    assert metrics['Local']['RMSE_m'] == pytest.approx(np.sqrt(10))
    assert metrics['Global']['RMSE_m'] == 1.
    assert metrics['DR']['final_m'] == 0.


def test_planar_relative_error_distinguishes_fixed_world_offset_and_increment_drift():
    position = np.zeros((61, 4, 3)); position[:, :, 0] = np.arange(61)[:, None]
    position[:, 1, :2] += [100., 50.]
    position[:, 2, 0] *= 1.02
    yaw = np.zeros((61, 4)); distances = np.arange(61)
    result = planar_relative_metrics(position, yaw, distances)
    assert result['10m']['RMSE_m']['DR'] == 0.
    assert result['10m']['RMSE_m']['Local'] == pytest.approx(.2)
    assert result['30m']['RMSE_m']['Local'] == pytest.approx(.6)
    assert result['30m']['RMSE_m']['Global'] == 0.


def test_phase_relative_pairs_never_bridge_a_turn_or_a_missing_phase():
    position = np.zeros((61, 4, 3)); position[:, :, 0] = np.arange(61)[:, None]
    yaw = np.zeros((61, 4)); distances = np.arange(61)
    phases = np.r_[np.zeros(25), np.ones(10), np.full(26, 2.)]
    result = planar_relative_metrics(position, yaw, distances, segment_index=phases, selected_segment=1)
    assert result['10m']['pairs'] == 0 and result['30m']['pairs'] == 0
