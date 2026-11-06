from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanFeatureMap import SideScanFeatureWindow
from PureSideScanFeatures import SideScanFeaturePacket
from SideScanProjectedSubmap import build_projected_submap
from ReviewProjectedSideScanTurn import verify_received_projection


def fixture():
    received = np.arange(2*2*8*16, dtype=float).reshape(2, 2, 8, 16)+1
    received[..., 3] = 0  # explicit missing acquisition: must not bridge it
    calibration = dict(bins=16, row_centres_m=((np.arange(8)-3.5)*.15).tolist())
    w = SideScanFeatureWindow(); position = np.zeros((2, 3, 3)); yaw = np.zeros((2, 3))
    for i in range(2):
        side, row, index = np.nonzero(received[i] > 0)
        p = SideScanFeaturePacket(timestamp=float(i), ranges_m=(index+.5)*32/16, amplitudes=received[i, side, row, index],
            rows=row, sides=side, sample_sequence=index, row_centres_m=np.array(calibration['row_centres_m']), max_range_m=32.)
        pose = np.array([i*1.3, i*.7, i*.4]); position[i, 1] = [pose[0], pose[1], -8.+.1*i]; yaw[i, 1] = pose[2]
        w.add(p, pose, i+1, vertical_position_m=position[i, 1, 2])
    q = build_projected_submap(w, pose, bins=16, extract=False)
    return q, received, dict(positions=position, yaw_rad=yaw, timestamps=np.arange(2, dtype=float)), calibration


def test_independent_projection_review_checks_real_raw_source_and_reports_resampler_loss():
    query, received, estimate, calibration = fixture()
    result = verify_received_projection(query, received, estimate, calibration)
    assert result['physical_pings'] == 2
    assert result['native_bins_excluded_by_existing_shared_resampler'] > 0
    assert result['amplitude_blends_verified'] and result['every_labeled_feature_source_is_a_real_maximum_contribution']


def test_projection_review_rejects_mutated_amplitude_or_fabricated_source():
    query, received, estimate, calibration = fixture()
    query['assembled_amplitude'][query['support']] += 1.
    with pytest.raises(AssertionError): verify_received_projection(query, received, estimate, calibration)
