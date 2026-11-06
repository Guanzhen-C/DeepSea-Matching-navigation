from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from PureSideScanFeatureNavigation import add_side_scan_feature_prior
from oceansim.slam.pose_graph import GTSAMPlanarPoseGraph, PoseGraphConfig


def test_real_GTSAM_feature_priors_sync_and_marginalize_without_fake_lateral_fix():
    graph = GTSAMPlanarPoseGraph(PoseGraphConfig(fixed_lag_seconds=180., terrain_prior_enabled=False))
    graph.add_pose(np.zeros(3), timestamp=0.)
    for k in range(1, 7):
        graph.add_pose(np.array([10.*k, 0., 0.]), timestamp=40.*k)
        graph.add_edge(k-1, k, np.array([10., 0., 0.]), np.diag([1., 1., 10.]), kind='odometry')
        add_side_scan_feature_prior(graph, k, np.array([10.*k+1., 900., 0.]),
            np.diag([4., 0., 10.]), score=.8)
        graph.optimize()
        assert np.isfinite(graph.poses).all()
        assert abs(graph.poses[-1, 1]) < 1e-6
    assert graph.backend_name == 'gtsam_isam2_incremental_fixed_lag'
    payload = graph.factor_stream_payload()
    assert len(payload['priors']) == 6
    assert {x['kind'] for x in payload['priors']} == {'side_scan_feature_prior'}
    assert graph.poses[-1, 0] > 60.1


def test_late_rank_deficient_side_scan_loop_reaches_live_window_after_marginalization():
    graph = GTSAMPlanarPoseGraph(PoseGraphConfig(fixed_lag_seconds=180., terrain_prior_enabled=False))
    graph.add_pose(np.zeros(3), timestamp=0.)
    for k in range(1, 11):
        graph.add_pose(np.array([10.*k, 0., 0.]), timestamp=40.*k)
        graph.add_edge(k-1, k, np.array([10., 0., 0.]), np.diag([1., 1., 100.]), kind='odometry')
        graph.optimize()
    assert graph._key(0) not in graph._fixed_lag_active_keys
    before = graph.poses[-1].copy()
    # Lateral measurement is intentionally absurd, with exactly ZERO lateral
    # information. It must not drag the live estimate towards y=900m.
    graph.add_edge(0, 10, np.array([95., 900., 0.]), np.diag([4., 0., 100.]), kind='loop')
    graph.optimize()
    assert graph.backend_name == 'gtsam_isam2_incremental_fixed_lag'
    assert graph.global_only_factor_count == 1
    assert graph.global_anchor_bridge_updates > 0
    assert graph.poses[-1, 0] < before[0]-1.
    np.testing.assert_allclose(graph.poses[-1, 1], 0., atol=1e-7)
    assert [e['kind'] for e in graph.factor_stream_payload()['edges']].count('loop') == 1
