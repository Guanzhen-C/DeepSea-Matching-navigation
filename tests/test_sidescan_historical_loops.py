from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
import SideScanLoopClosure as module
from SideScanFeatureMap import pose_matrix
from SideScanLoopClosure import SideScanHistoricalLoops, feature_observations, overlap_supported


def query(start_id=1, end_id=150, start_s=0., end_s=49.):
    h, bins = 80, 512
    return dict(image=np.full((h, 2*bins), 100, np.uint8), support=np.ones((h, 2*bins), bool),
        descriptors=np.ones((30, 128), np.float32), row_normals=np.tile([1., 0.], (h, 1)),
        row_scan_origins=np.column_stack([np.arange(h)*.5-40., np.zeros((h, 2))]),
        anchor_vertical_z_m=-20., bins=bins, scale=1, step=.5, x0=-40., x1=-.5,
        max_range_m=250., timestamp=end_s, patch_start_s=start_s,
        patch_start_id=start_id, patch_end_id=end_id)


def graph():
    g = SimpleNamespace(poses=np.zeros((3, 3)), _timestamps=[0., 60., 900.], edges=[])
    g.add_edge = lambda *args, **kw: g.edges.append((args, kw))
    return g


def frame(q, node=0):
    return dict(query=q, node=node, offset=np.eye(3), timestamp=q['timestamp'],
                descriptor=SideScanHistoricalLoops.descriptor(q))


def test_feature_pixels_are_observed_slant_ranges_not_fabricated_ground_coordinates():
    q = query(); pixels = np.array([[511., 20.], [512., 30.]])
    o = feature_observations(q, pixels)
    np.testing.assert_allclose(o['ranges'], .5*250/512)
    np.testing.assert_array_equal(o['sides'], [1., -1.])
    np.testing.assert_allclose(o['origins'][:, 0], [-30., -25.])
    np.testing.assert_allclose(o['origins'][:, 1:], 0.)


def test_retrieval_excludes_recent_windows_shared_pings_and_nonoverlapping_swaths():
    loops = SideScanHistoricalLoops(None)
    g = graph(); current = query(601, 750, 600., 649.)
    accepted = frame(query())
    loops.frames = [accepted, frame(query(151, 300, 450., 499.)),
                    frame(query(600, 700, 0., 49.))]
    far = frame(query()); far['offset'] = pose_matrix([10000., 0., 0.])
    loops.frames.append(far)
    assert loops.candidates(current, np.eye(3), g) == [accepted]
    assert overlap_supported(query(), np.eye(3), query(), pose_matrix([0., 28., np.pi]))


def test_reverse_view_appearance_transform_is_undone_before_physical_geometry(monkeypatch):
    old, new = query(), query(601, 750, 600., 649.)
    points = np.column_stack([np.linspace(20., 1000., 30), np.linspace(5., 70., 30)])
    new['image'] = np.arange(new['image'].size, dtype=np.uint8).reshape(new['image'].shape)
    def matcher(a, b, ma, mb):
        np.testing.assert_array_equal(b, new['image'][::-1, ::-1])
        return points, points, np.ones(30)
    monkeypatch.setattr(module, 'row_compatible_pairs', lambda *args: np.ones(30, bool))
    captured = {}
    def fit(a, b, seed, delta_z, **options):
        captured.update(a=a, b=b, delta_z=delta_z)
        assert options['minimum_features'] == 24
        return dict(accepted=True)
    monkeypatch.setattr(module, 'fit_range_plane_loop', fit)
    result = SideScanHistoricalLoops(matcher).match_pair(frame(old), new, [0., 28., np.pi])
    restored = np.array([1023., 79.])-points
    np.testing.assert_array_equal(result['new_pixels'], restored)
    np.testing.assert_allclose(captured['b']['ranges'], feature_observations(new, restored)['ranges'])
    assert result['appearance_reversed_for_matching']


def test_multiple_candidates_from_one_window_cannot_confirm_a_loop(monkeypatch):
    loops = SideScanHistoricalLoops(None); g = graph()
    old1, old2 = frame(query()), frame(query(151, 300, 50., 99.), 1)
    monkeypatch.setattr(loops, 'candidates', lambda *args: [old1, old2])
    monkeypatch.setattr(loops, 'match_pair', lambda old, q, seed: dict(accepted=True,
        relative_pose2=np.zeros(3), information=np.diag([1., 0., 10.]),
        standardized_rmse=.1 if old is old1 else .2, observable_rank=2, matches=40, inliers=30,
        old_pixels=np.zeros((30, 2)), new_pixels=np.zeros((30, 2))))
    q = query(301, 450, 850., 899.)
    assert loops.observe(q, np.zeros(3), g, [np.eye(3)]*3, 899.) is None
    assert loops.added_edges == 0 and len(g.edges) == 0
    assert loops.pending['old_start'] == 1  # best candidate, not the last one
    monkeypatch.setattr(loops, 'candidates', lambda *args: [old2])
    monkeypatch.setattr(module, 'lift_loop_to_graph_nodes',
        lambda *args, **kw: (np.zeros(3), np.diag([1., 0., 10.])))
    result = loops.observe(query(451, 600, 900., 949.), np.zeros(3), g, [np.eye(3)]*3, 949.,
                           shared_current_window=True)
    assert result is not None and loops.added_edges == 1
    assert g.edges[0][1]['kind'] == 'loop'
    assert np.linalg.matrix_rank(g.edges[0][0][3], tol=1e-8) == 2
    np.testing.assert_array_equal(g.edges[0][0][3], np.diag([.5, 0., 5.]))
    assert result['current_window_information_weight'] == .5
    # Exactly the same physical window is not another independent loop.
    assert loops.observe(query(451, 600, 900., 949.), np.zeros(3), g, [np.eye(3)]*3, 949.) is None


def test_no_candidate_is_logged_as_no_candidate_not_as_stale_success():
    loops = SideScanHistoricalLoops(None); g = graph()
    loops.last = dict(status='confirmed')
    assert loops.observe(query(), np.zeros(3), g, [np.eye(3)]*3, 49.) is None
    assert loops.last['status'] == 'no_time_separated_overlapping_historical_submap'
    assert loops.diagnostics()['actual_loop_edges'] == 0


def test_trace_records_actual_inserted_information_after_graph_tempering(monkeypatch):
    loops = SideScanHistoricalLoops(None); g = graph()
    g.add_edge = lambda source, target, measurement, info, **kw: g.edges.append(
        SimpleNamespace(information=np.asarray(info)*.5))
    old1, old2 = frame(query()), frame(query(151, 300, 50., 99.), 1)
    monkeypatch.setattr(loops, 'candidates', lambda *args: [old1])
    monkeypatch.setattr(loops, 'match_pair', lambda *args: dict(accepted=True,
        relative_pose2=np.zeros(3), information=np.diag([1., 0., 10.]),
        standardized_rmse=.1, observable_rank=2, matches=40, inliers=30,
        old_pixels=np.zeros((30, 2)), new_pixels=np.zeros((30, 2))))
    monkeypatch.setattr(module, 'lift_loop_to_graph_nodes',
                        lambda *a, **kw: (np.zeros(3), np.diag([1., 0., 10.])))
    loops.observe(query(301, 450, 850., 899.), np.zeros(3), g, [np.eye(3)]*3, 899.)
    monkeypatch.setattr(loops, 'candidates', lambda *args: [old2])
    selected = loops.observe(query(451, 600, 900., 949.), np.zeros(3), g, [np.eye(3)]*3, 949.,
                             shared_current_window=True)
    assert selected is not None
    np.testing.assert_array_equal(selected['information'], np.diag([.25, 0., 2.5]))
    np.testing.assert_array_equal(loops.trace[-1]['factor']['information'], g.edges[-1].information)
    assert loops.trace[-1]['factor']['information_is_actual_graph']
