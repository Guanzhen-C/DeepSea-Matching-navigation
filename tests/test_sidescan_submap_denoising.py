from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanSubmapDenoising import fuse_aligned_rows
from SideScanFeatureMap import SideScanFeatureWindow
from PureSideScanFeatures import SideScanFeaturePacket


def test_aligned_fusion_suppresses_outlier_without_filling_missing_cells():
    rows = [(0, np.array([2., 5., 0.])), (0, np.array([2., 5., 0.])),
            (0, np.array([200., 5., 0.])), (1, np.array([0., 7., 0.]))]
    result = fuse_aligned_rows(rows, 3, 3)
    assert result['denoised_amplitude'][0, 0] == pytest.approx(2., abs=1e-4)
    assert result['before_denoising_amplitude'][0, 0] == pytest.approx(68.)
    np.testing.assert_allclose(result['denoised_amplitude'][1], [0., 7., 0.])
    assert not result['denoised_amplitude'][2].any()
    assert not result['denoised_amplitude'][:, 2].any()
    assert result['observation_count'][0, 0] == 3
    assert not result['denoising']['frozen_complex_IQ_model_used']
    assert not result['denoising']['reference_map_read_by_denoiser']


def test_noiseless_amplitude_is_unchanged_and_input_is_not_modified():
    original = np.array([.002, 0., 2., 7.])
    result = fuse_aligned_rows([(0, original)]*5, 2, 4)
    np.testing.assert_allclose(result['denoised_amplitude'][0], original, rtol=1e-12)
    np.testing.assert_array_equal(original, [.002, 0., 2., 7.])


def test_submap_accepts_only_causal_received_packets():
    packet = SideScanFeaturePacket(timestamp=1., ranges_m=np.array([2., 4.]),
        amplitudes=np.ones(2), rows=np.zeros(2, int), sides=np.ones(2, int),
        row_centres_m=(np.arange(8)-3.5)*.15, sample_sequence=np.arange(2))
    window = SideScanFeatureWindow()
    window.add(packet, np.zeros(3), 1)
    with pytest.raises(ValueError, match='strictly causal'):
        window.add(packet, np.zeros(3), 2)


def test_denoiser_rejects_nonfinite_or_negative_measurements():
    with pytest.raises(ValueError): fuse_aligned_rows([(0, [1., np.nan])], 2, 2)
    with pytest.raises(ValueError): fuse_aligned_rows([(0, [1., -1.])], 2, 2)
