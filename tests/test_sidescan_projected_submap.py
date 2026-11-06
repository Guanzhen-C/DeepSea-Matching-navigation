from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanFeatureMap import SideScanFeatureWindow
from PureSideScanFeatures import SideScanFeaturePacket
from SideScanProjectedSubmap import build_projected_submap, projected_feature_observations


def packet(stamp=0.):
    return SideScanFeaturePacket(timestamp=stamp, ranges_m=np.array([8., 12.]), amplitudes=np.array([2., 1.]),
        rows=np.array([0, 1]), sides=np.array([0, 1]), row_centres_m=(np.arange(8)-3.5)*.15,
        sample_sequence=np.array([0, 1]), max_range_m=32.)


def test_projection_retains_every_orthogonal_and_opposite_ping_and_real_source_geometry(monkeypatch):
    import SideScanProjectedSubmap as module
    raster = np.arange(8*32, dtype=float).reshape(8, 32)+1
    monkeypatch.setattr(module, 'resample_supported_echoes', lambda packet, bins: raster.copy())
    w = SideScanFeatureWindow()
    for i, angle in enumerate([0., np.pi/2, np.pi]):
        w.add(packet(float(i)), np.r_[0., 0., angle], i+1, vertical_position_m=-8.+i*.1)
    q = build_projected_submap(w, np.zeros(3), bins=16, extract=False)
    assert q['projected_contributions'] == q['received_contributions'] == 3*8*32
    np.testing.assert_array_equal(q['physical_ping_observation_ids'], [1, 2, 3])
    yi, xi = np.nonzero(q['support']); obs = projected_feature_observations(q, np.column_stack([xi, yi]))
    assert set(obs['packet_ids']) == {1, 2, 3}
    np.testing.assert_allclose(np.linalg.norm(obs['normals'], axis=1), 1.)
    assert set(np.round(obs['ranges'], 5)) <= set(np.arange(1., 32., 2.))
    assert set(obs['timestamps_s']) == {0., 1., 2.}
    assert not np.any(q['assembled_amplitude'][~q['support']])
    assert np.all(q['source_phase_centre'][~q['support']] == -1)


def test_image_anchor_or_grid_coordinates_cannot_change_the_stored_received_range(monkeypatch):
    import SideScanProjectedSubmap as module
    a = np.zeros((8, 32)); a[4, 3] = 7.
    monkeypatch.setattr(module, 'resample_supported_echoes', lambda packet, bins: a)
    w = SideScanFeatureWindow(); w.add(packet(), np.array([3., 4., .7]), 1, vertical_position_m=-8.)
    for anchor in [np.zeros(3), np.array([10., 20., 1.4])]:
        q = build_projected_submap(w, anchor, bins=16, extract=False)
        y, x = np.nonzero(q['support']); o = projected_feature_observations(q, np.column_stack([x, y]))
        np.testing.assert_array_equal(o['ranges'], [25.])
        np.testing.assert_array_equal(o['beam_rows'], [4])
        assert q['assembled_amplitude'][y, x] == 7.


def test_overlaps_average_received_amplitude_but_label_an_actual_dominant_observation(monkeypatch):
    import SideScanProjectedSubmap as module
    def raster(p, bins):
        a = np.zeros((8, 32)); a[4, 3] = 1.+2*p.timestamp; return a
    monkeypatch.setattr(module, 'resample_supported_echoes', raster)
    w = SideScanFeatureWindow()
    for i in range(2): w.add(packet(float(i)), np.zeros(3), i+1)
    q = build_projected_submap(w, np.zeros(3), bins=16, extract=False)
    y, x = np.nonzero(q['support']); o = projected_feature_observations(q, np.column_stack([x, y]))
    np.testing.assert_array_equal(q['assembled_amplitude'][y, x], [2.])
    np.testing.assert_array_equal(q['observation_count'][y, x], [2])
    np.testing.assert_array_equal(o['packet_ids'], [2])
    np.testing.assert_array_equal(o['ranges'], [25.])
    assert q['received_contributions'] == q['projected_contributions'] == 2


def test_unsupported_holes_cannot_be_used_as_invented_landmark_measurements(monkeypatch):
    import SideScanProjectedSubmap as module
    a = np.zeros((8, 32)); a[4, 3] = 7.
    monkeypatch.setattr(module, 'resample_supported_echoes', lambda packet, bins: a)
    w = SideScanFeatureWindow(); w.add(packet(), np.zeros(3), 1)
    q = build_projected_submap(w, np.zeros(3), bins=16, extract=False)
    with pytest.raises(ValueError, match='actually received'):
        projected_feature_observations(q, [[0., 0.]])
