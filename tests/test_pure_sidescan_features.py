from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from PureSideScanFeatures import (SideScanFeaturePacket, packet_from_simulated_side_scan,
                                  robust_feature_row_motion, resample_supported_echoes)


def test_feature_motion_preserves_fractional_rows_without_XYZ_or_optical_flow(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, 'calcOpticalFlowPyrLK', lambda *a, **k: (_ for _ in ()).throw(AssertionError('No flow allowed')))
    rng = np.random.default_rng(31)
    n = 100
    descriptor = rng.normal(size=(n, 128)).astype(np.float32)
    descriptor /= np.linalg.norm(descriptor, axis=1, keepdims=True)
    slant = np.r_[np.linspace(50, 180, n//2), -np.linspace(50, 180, n//2)]
    x = rng.uniform(-.4, .1, n)
    pixels = np.column_stack([np.where(slant > 0, 200, 800), 10+x])
    current = dict(descriptors=descriptor, pixels=pixels, bins=512, along_m=x,
                   signed_slant_m=slant, timestamp=.33)
    previous = dict(current, along_m=x+.264+.0005*slant)
    result = robust_feature_row_motion(current, previous, .264)
    assert result is not None and result['descriptor_matching']
    assert result['measured_along_m'] == pytest.approx(.264, abs=1e-12)
    assert abs(result['correction_m']) < 1e-12
    assert result['optical_flow_used'] is result['MBES_XYZ_used'] is result['DVL_altitude_used'] is False


def test_packet_export_never_reads_bathymetry_DOA_or_XYZ():
    class Image:
        ranges = np.array([10., 20., 30., 40.])
        intensities = np.array([.1, .2, .3, .4])
        along_track_row_indices = np.array([0, 1, 0, 1])
        hit_mask = np.ones(4, bool)
        @property
        def points_sensor(self):
            raise AssertionError('Forbidden measured XYZ')
        @property
        def beam_directions(self):
            raise AssertionError('Forbidden angular/depth geometry')
    packet = packet_from_simulated_side_scan(SimpleNamespace(timestamp=1., imaging=Image()))
    np.testing.assert_array_equal(packet.ranges_m, [10, 20, 30, 40])
    assert not packet.amplitudes.flags.writeable
    with pytest.raises(TypeError):
        SideScanFeaturePacket(timestamp=0., ranges_m=np.ones(2), amplitudes=np.ones(2),
            rows=np.zeros(2), sides=np.zeros(2), row_centres_m=np.arange(8)*.15, measured_height=50.)


def test_insufficient_feature_support_produces_no_pose_not_a_fake_fix():
    frame = dict(descriptors=np.ones((1, 128), np.float32), pixels=np.array([[2, 3]]),
                 bins=512, along_m=np.array([0.]), signed_slant_m=np.array([80.]), timestamp=0.)
    assert robust_feature_row_motion(frame, frame, .264) is None


def test_echo_resampling_preserves_missing_samples_and_water_column():
    packet = SideScanFeaturePacket(timestamp=0., ranges_m=np.array([2., 4., 8., 10.]),
        amplitudes=np.array([1., 3., 7., 9.]), rows=np.zeros(4, int), sides=np.ones(4, int),
        row_centres_m=(np.arange(8)-3.5)*.15, sample_sequence=np.array([0, 1, 3, 4]), max_range_m=12.)
    raster = resample_supported_echoes(packet, bins=12)
    np.testing.assert_allclose(raster[0, 12+2:12+4], [1.5, 2.5])
    np.testing.assert_array_equal(raster[0, 12+4:12+8], 0.)
    assert not raster[:, :12].any() and not raster[1:].any()
    assert not raster[0, 12:14].any() and not raster[0, 22:].any()
