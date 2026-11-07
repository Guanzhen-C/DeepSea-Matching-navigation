from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'standalone'))
pytest.importorskip('gtsam')
from oceansim.slam.pose_graph import PoseGraphConfig
from SideScanMappedRangeGraph import MappedRangePoseGraph
from SideScanFeatureMap import pose_matrix, matrix_pose
from SideScanMappedRangeChange import MappedRangeChange, conservative_range_change_sigma


def graph_and_data():
    a = np.array([0., 0., 0.]); b = np.array([4., 28., 0.])
    xy = np.column_stack([np.linspace(-12., 22., 32), np.linspace(40., 150., 32)])
    height = -np.linspace(20., 60., 32)
    ra = np.sqrt(np.sum((xy-a[:2])**2, axis=1)+(height+8)**2)
    rb = np.sqrt(np.sum((xy-b[:2])**2, axis=1)+(height+8)**2)
    data = MappedRangeChange(xy, np.zeros((32, 2)), np.zeros((32, 2)),
        np.full(32, -8.), np.full(32, -8.), ra, rb,
        conservative_range_change_sigma(ra, rb, 30.))
    graph = MappedRangePoseGraph(PoseGraphConfig(fixed_lag_seconds=180.))
    graph.add_pose(a, timestamp=0.)
    return graph, data, a, b


def test_real_gtsam_range_revisit_corrects_cross_track_with_no_height_input():
    graph, data, a, truth = graph_and_data()
    seed = truth+np.array([0., 9., 0.])
    graph.add_pose(seed, timestamp=30.)
    graph.add_edge(0, 1, seed, np.diag([1., 1/20.**2, 100.]), kind='odometry')
    graph.add_prior(1, truth, np.diag([4., 0., 1000.]), kind='terrain_prior')
    graph.add_mapped_range_change(0, 1, data)
    for _ in range(5): graph.optimize()
    assert abs(graph.poses[1, 1]-truth[1]) < .8
    assert abs(graph.poses[1, 0]-truth[0]) < .3
    payload = graph.factor_stream_payload()
    assert len(payload['mapped_range_changes']) == 1
    assert payload['edges'][-1]['kind'] == 'mapped_range_change'
    assert not payload['mapped_range_changes'][0]['observations']['DEM_or_altitude_used']


def test_custom_factor_local_tangent_jacobians_match_gtsam_retraction():
    import gtsam
    graph, data, a, b = graph_and_data()
    graph.add_pose(b+[.2, 5., .03], timestamp=30.)
    graph.add_mapped_range_change(0, 1, data)
    factor = graph._edge_factor(graph.edges[-1])
    values = gtsam.Values()
    values.insert(graph._key(0), graph._pose2(a))
    values.insert(graph._key(1), graph._pose2(graph.poses[1]))
    linear = factor.linearize(values)
    matrix = linear.getA()
    from SideScanMappedRangeChange import profiled_range_residual_jacobians
    for node in (0, 1):
        numeric = []
        for k in range(3):
            v0 = values.atPose2(graph._key(node))
            delta = np.eye(3)[k]*1e-6
            args = [a.copy(), graph.poses[1].copy()]
            args[node] = graph._pose_array(v0.retract(delta))
            plus = profiled_range_residual_jacobians(*args, data)[0]
            args[node] = graph._pose_array(v0.retract(-delta))
            minus = profiled_range_residual_jacobians(*args, data)[0]
            numeric.append((plus-minus)/2e-6)
        np.testing.assert_allclose(matrix[:, node*3:(node+1)*3], np.column_stack(numeric),
                                   rtol=2e-6, atol=2e-8)


def test_correlation_budget_leaves_odometry_unchanged_and_tempers_map_and_loop():
    graph, data, a, b = graph_and_data()
    graph.add_pose(b, timestamp=30.)
    graph.add_edge(0, 1, b, np.eye(3), kind='odometry')
    graph.add_edge(0, 1, b, np.eye(3), kind='loop')
    graph.add_prior(1, b, np.eye(3), kind='terrain_prior')
    np.testing.assert_array_equal(graph.edges[0].information, np.eye(3))
    np.testing.assert_array_equal(graph.edges[1].information, .5*np.eye(3))
    np.testing.assert_array_equal(graph.priors[0].information, .5*np.eye(3))


def test_long_revisit_uses_retained_graph_not_frozen_historical_pose():
    graph, data, a, b = graph_and_data()
    previous = a.copy()
    for i in range(1, 11):
        seed = (b+[0., 8., 0.])*i/10
        graph.add_pose(seed, timestamp=30.*i)
        relative = matrix_pose(np.linalg.inv(pose_matrix(previous))@pose_matrix(seed))
        graph.add_edge(i-1, i, relative, np.diag([1., 1/4.**2, 100.]), kind='odometry')
        graph.optimize(); previous = seed
    before = abs(graph.poses[-1, 1]-b[1])
    graph.add_mapped_range_change(0, 10, data)
    for _ in range(5): graph.optimize()
    assert abs(graph.poses[-1, 1]-b[1]) < before/2
    assert graph.global_only_factor_count > 0
    np.testing.assert_allclose(graph.poses[-1], graph.global_isam_poses[-1], atol=1e-8)


def test_optional_navigator_reports_the_actual_observation_scope_without_changing_initial_states():
    from types import SimpleNamespace
    from PureSideScanFeatureNavigation import PureSideScanFeatureNavigator
    initial = SimpleNamespace(position=np.array([1., 2., -8.]), rotation=np.eye(3), velocity=np.zeros(3))
    startup = np.tile([0., 0., 9.81], (32, 1))
    baseline = PureSideScanFeatureNavigator(None, initial, startup)
    candidate = PureSideScanFeatureNavigator(None, initial, startup, mapped_range_changes=True)
    assert 'two-reception range-change' in candidate.diagnostics()['observation_limit']
    assert 'Scan-plane geometry only' in baseline.diagnostics()['observation_limit']
    np.testing.assert_array_equal(baseline.local.state.position, candidate.local.state.position)
    np.testing.assert_array_equal(baseline.local.state.rotation, candidate.local.state.rotation)
    np.testing.assert_array_equal(baseline.dr.state.covariance, candidate.dr.state.covariance)


def test_preserve_legacy_policy_never_downweights_already_validated_information():
    graph = MappedRangePoseGraph(PoseGraphConfig(), legacy_precision_budget=1.)
    graph.add_pose(np.zeros(3), timestamp=0.)
    graph.add_pose(np.array([1., 2., .1]), timestamp=30.)
    info = np.diag([1., 0., 100.])
    graph.add_prior(1, np.array([1., 2., .1]), info, kind='terrain_prior')
    graph.add_edge(0, 1, np.array([1., 2., .1]), info, kind='loop')
    np.testing.assert_array_equal(graph.priors[0].information, info)
    np.testing.assert_array_equal(graph.edges[0].information, info)
    assert graph.factor_stream_payload()['mapped_range_change_policy']['original_map_and_loop_precision_budget'] == 1.
