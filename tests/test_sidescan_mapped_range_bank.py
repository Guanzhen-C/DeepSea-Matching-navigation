from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanMappedRangeBank import MappedRangeBank, received_feature_ranges
from SideScanFeatureMap import SideScanFeatureWindow, pose_matrix
from PureSideScanFeatures import SideScanFeaturePacket


def query_match(start, stamp, n=12):
    query = dict(received_phase_centre=np.arange(n)[:, None],
        received_range_column=(np.arange(n)[:, None]+20),
        received_phase_origins=np.zeros((n, 3)),
        received_packet_ids=np.arange(start, start+n), received_beam_rows=np.zeros(n, int),
        received_timestamps_s=np.arange(n)*.1+stamp-n*.1,
        bins=512, max_range_m=250., scale=1, timestamp=stamp,
        patch_start_id=start, patch_end_id=start+n-1, anchor_vertical_z_m=-8.)
    match = dict(accepted=True, query_pixels=np.column_stack([np.zeros(n), np.arange(n)]),
        landmark_xy=np.column_stack([np.arange(n)*3., np.full(n, 100.)]), score=1.)
    return query, match


class Graph:
    def __init__(self):
        self.poses = np.zeros((1, 3)); self.added = []

    def add_mapped_range_change(self, source, target, data, **kwargs):
        self.added.append((source, target, data))


def test_ranges_come_from_labeled_physical_ping_not_average_row_geometry():
    q, m = query_match(1, 20.)
    q['row_scan_origins'] = np.full((12, 3), 999.)
    result = received_feature_ranges(q, m['query_pixels'])
    np.testing.assert_array_equal(result['origins'], np.zeros((12, 3)))
    np.testing.assert_array_equal(result['packet_ids'], np.arange(1, 13))
    np.testing.assert_allclose(result['ranges'], (512-.5-np.arange(20, 32))*250/512)


def test_missing_source_labels_and_holes_cannot_create_observations():
    q, m = query_match(1, 20.)
    q['received_phase_centre'][0, 0] = -1
    with pytest.raises(ValueError, match='holes'):
        received_feature_ranges(q, m['query_pixels'])
    del q['received_phase_centre']
    with pytest.raises(ValueError, match='provenance'):
        received_feature_ranges(q, m['query_pixels'])


def test_causal_independent_revisit_makes_a_real_two_variable_pose_factor_once():
    bank = MappedRangeBank(); graph = Graph()
    q, m = query_match(1, 20.)
    assert bank.observe(q, m, np.zeros(3), graph, [np.eye(3)]) == 0
    graph.poses = np.array([[0., 0., 0.], [0., 28., 0.]])
    poses = [pose_matrix(p) for p in graph.poses]
    q, m = query_match(1001, 400.)
    assert bank.observe(q, m, graph.poses[-1], graph, poses) == 1
    assert len(graph.added) == 1
    source, target, data = graph.added[0]
    assert (source, target, data.features) == (0, 1, 12)
    assert bank.diagnostics()['used_received_range_tokens'] == 24
    tokens = [tuple(t) for row in bank.trace for t in row['received_tokens']]
    assert len(tokens) == len(set(tokens))
    graph.poses = np.vstack([graph.poses, [0., 56., 0.]])
    q, m = query_match(2001, 800.)
    assert bank.observe(q, m, graph.poses[-1], graph, [pose_matrix(p) for p in graph.poses]) == 0
    assert len(graph.added) == 1  # both old AND formerly-current tokens consumed


def test_nearby_overlapping_windows_are_not_independent_received_constraints():
    bank = MappedRangeBank(); graph = Graph()
    q, m = query_match(1, 20.)
    bank.observe(q, m, np.zeros(3), graph, [np.eye(3)])
    graph.poses = np.array([[0., 0., 0.], [0., 28., 0.]])
    q, m = query_match(5, 30.)
    assert bank.observe(q, m, graph.poses[-1], graph, [pose_matrix(p) for p in graph.poses]) == 0
    assert not graph.added


def test_far_map_points_are_not_forced_into_nearest_neighbor_correspondences():
    bank = MappedRangeBank(); graph = Graph()
    q, m = query_match(1, 20.)
    bank.observe(q, m, np.zeros(3), graph, [np.eye(3)])
    graph.poses = np.array([[0., 0., 0.], [0., 28., 0.]])
    q, m = query_match(1001, 400.); m['landmark_xy'][:, 1] += 10.
    assert bank.observe(q, m, graph.poses[-1], graph, [pose_matrix(p) for p in graph.poses]) == 0


def test_dominant_source_labels_do_not_change_any_assembled_intensity_or_features(monkeypatch):
    import SideScanFeatureMap as module
    raster = np.arange(8*32, dtype=float).reshape(8, 32)+1.
    monkeypatch.setattr(module, 'resample_supported_echoes', lambda packet, bins: raster*(packet.timestamp+1))
    monkeypatch.setattr(module, 'extract_features', lambda *args: (np.zeros((0, 2)), np.zeros((0, 128))))
    windows = [SideScanFeatureWindow(received_provenance=v) for v in (False, True)]
    for window in windows:
        for i in range(3):
            packet = SideScanFeaturePacket(timestamp=float(i), ranges_m=np.array([8., 12.]),
                amplitudes=np.array([2., 1.]), rows=np.array([0, 1]), sides=np.array([0, 1]),
                row_centres_m=(np.arange(8)-3.5)*.15, sample_sequence=np.array([0, 1]), max_range_m=32.)
            window.add(packet, np.zeros(3), i+1, vertical_position_m=-8.)
    a, b = [w.build(np.zeros(3), bins=16) for w in windows]
    for name in ('image', 'raw_amplitude', 'assembled_amplitude', 'observation_count',
                 'row_normals', 'row_scan_origins', 'support', 'pixels', 'descriptors'):
        np.testing.assert_array_equal(a[name], b[name])
    y, x = np.nonzero(b['support'])
    observed = received_feature_ranges(b, np.column_stack([x, y]))
    assert set(observed['packet_ids']) == {3}
    assert 'received_phase_centre' not in a


def test_unconfirmed_matches_cannot_populate_the_bank():
    q, m = query_match(1, 20.); m['accepted'] = False
    bank = MappedRangeBank()
    with pytest.raises(ValueError, match='confirmed'):
        bank.observe(q, m, np.zeros(3), Graph(), [np.eye(3)])
    assert not bank.observed_tokens
