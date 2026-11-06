from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from AnalyzePureSideScanShort import summarize


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
