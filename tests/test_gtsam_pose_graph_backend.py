"""Contract and factor-equivalence tests for the optional GTSAM backend."""

from __future__ import annotations

import json

import numpy as np
import pytest

from oceansim.slam import (
    AcousticPoseGraph,
    GTSAMPlanarPoseGraph,
    PlanarPoseGraph,
    PoseGraphConfig,
    PoseGraphEdge,
)
from oceansim.slam.pose_graph import (
    _edge_error_and_jacobians,
    _relative_pose2,
)
from oceansim.slam.math3d import rotation_from_rpy


gtsam = pytest.importorskip("gtsam")


def _gtsam_config(**overrides) -> PoseGraphConfig:
    values = {
        "optimizer_backend": "gtsam_fixed_lag_isam2",
        "fixed_lag_seconds": 2.0,
        "isam_relinearize_threshold": 1.0e-5,
        "isam_relinearize_skip": 1,
    }
    values.update(overrides)
    return PoseGraphConfig(**values)


def _add_chain(graph: GTSAMPlanarPoseGraph, count: int) -> None:
    information = np.diag([100.0, 100.0, 400.0])
    for index in range(count):
        graph.add_pose(
            np.array([float(index), 0.0, 0.0]), timestamp=float(index)
        )
        if index:
            graph.add_edge(
                index - 1,
                index,
                np.array([1.0, 0.0, 0.0]),
                information,
                kind="odometry",
            )
            graph.optimize()


def test_acoustic_graph_backend_is_explicit_and_numpy_stays_default() -> None:
    reference = AcousticPoseGraph(PoseGraphConfig())
    candidate = AcousticPoseGraph(_gtsam_config())

    assert type(reference.graph) is PlanarPoseGraph
    assert reference.graph.backend_name == "numpy_batch"
    assert isinstance(candidate.graph, GTSAMPlanarPoseGraph)
    assert candidate.graph.backend_name == "gtsam_isam2_incremental_fixed_lag"


def test_unknown_backend_is_rejected_instead_of_silently_falling_back() -> None:
    with pytest.raises(ValueError, match="optimizer_backend"):
        AcousticPoseGraph(PoseGraphConfig(optimizer_backend="unknown"))


def test_gtsam_custom_factors_reproduce_reference_residual_coordinates() -> None:
    graph = GTSAMPlanarPoseGraph(_gtsam_config(fixed_lag_seconds=100.0))
    source = np.array([4.0, -2.0, 1.1])
    target = np.array([7.5, 3.0, -0.4])
    measurement = np.array([5.1, -1.3, -1.2])
    information = np.array(
        [[8.0, 1.5, -0.2], [1.5, 5.0, 0.4], [-0.2, 0.4, 30.0]]
    )
    graph.add_pose(source, timestamp=0.0)
    graph.add_pose(target, timestamp=1.0)
    graph.add_edge(0, 1, measurement, information, kind="odometry")
    factor = graph._edge_factor(graph.edges[0])
    values = gtsam.Values()
    values.insert(graph._key(0), graph._pose2(source))
    values.insert(graph._key(1), graph._pose2(target))

    expected, _, _ = _edge_error_and_jacobians(
        source, target, measurement
    )
    np.testing.assert_allclose(factor.unwhitenedError(values), expected, atol=1e-12)

    prior_measurement = np.array([6.8, 2.9, -0.35])
    graph.add_prior(1, prior_measurement, information, robust=False)
    prior_factor = graph._prior_factor(graph.priors[0])
    expected_prior = target - prior_measurement
    expected_prior[2] = (expected_prior[2] + np.pi) % (2.0 * np.pi) - np.pi
    np.testing.assert_allclose(
        prior_factor.unwhitenedError(values), expected_prior, atol=1e-12
    )

    # CustomFactor Jacobians are with respect to Pose2's local retract
    # coordinates, whereas OceanSim's reference residual is additive in world
    # x/y/yaw. Verify the conversion at a deliberately non-zero heading.
    _, jacobian_source_world, jacobian_target_world = (
        _edge_error_and_jacobians(source, target, measurement)
    )
    expected_jacobians = (
        jacobian_source_world @ graph._local_delta_to_world(source),
        jacobian_target_world @ graph._local_delta_to_world(target),
    )
    epsilon = 1.0e-6
    for key_index, key in enumerate((graph._key(0), graph._key(1))):
        finite_difference = np.empty((3, 3), dtype=np.float64)
        for axis in range(3):
            perturbation = np.zeros(3, dtype=np.float64)
            perturbation[axis] = epsilon
            plus = gtsam.Values(values)
            minus = gtsam.Values(values)
            plus.update(key, values.atPose2(key).retract(perturbation))
            minus.update(key, values.atPose2(key).retract(-perturbation))
            finite_difference[:, axis] = (
                factor.unwhitenedError(plus)
                - factor.unwhitenedError(minus)
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            finite_difference, expected_jacobians[key_index], atol=2.0e-6
        )


def test_gtsam_and_numpy_solve_same_unmarginalized_factor_stream() -> None:
    common = {
        "optimizer_iterations": 50,
        "fixed_lag_seconds": 1000.0,
        "isam_relinearize_threshold": 1.0e-5,
        "isam_relinearize_skip": 1,
    }
    reference = PlanarPoseGraph(PoseGraphConfig(**common))
    candidate = GTSAMPlanarPoseGraph(_gtsam_config(**common))
    truth = np.array(
        [[0.0, 0.0, 0.7], [4.0, 3.0, 0.75], [7.0, 7.0, 0.9], [4.0, 10.0, 1.4]]
    )
    seeds = truth + np.array(
        [[0.0, 0.0, 0.0], [0.8, -0.4, 0.12], [1.4, -0.3, 0.2], [2.0, 0.5, 0.3]]
    )
    edge_information = np.array(
        [[25.0, 4.0, 1.0], [4.0, 16.0, -0.5], [1.0, -0.5, 120.0]]
    )
    for index, seed in enumerate(seeds):
        for graph in (reference, candidate):
            graph.add_pose(seed, timestamp=float(index))
        if index:
            measurement = _relative_pose2(truth[index - 1], truth[index])
            for graph in (reference, candidate):
                graph.add_edge(
                    index - 1,
                    index,
                    measurement,
                    edge_information,
                    kind="odometry",
                )
    prior_information = np.array(
        [[9.0, 2.0, 0.2], [2.0, 12.0, -0.3], [0.2, -0.3, 60.0]]
    )
    for graph in (reference, candidate):
        graph.add_prior(
            3,
            truth[3] + np.array([0.1, -0.1, 0.01]),
            prior_information,
            robust=False,
        )
        graph.optimize()

    np.testing.assert_allclose(candidate.poses, reference.poses, atol=7.0e-5)
    assert candidate.final_cost == pytest.approx(reference.final_cost, abs=2.0e-6)


def test_robust_loop_and_terrain_objective_matches_numpy_reference() -> None:
    common = {
        "optimizer_iterations": 80,
        "fixed_lag_seconds": 1000.0,
        "isam_relinearize_threshold": 1.0e-6,
        "robust_loop_delta": 2.5,
        "robust_terrain_prior_delta": 2.5,
    }
    reference = PlanarPoseGraph(PoseGraphConfig(**common))
    candidate = GTSAMPlanarPoseGraph(_gtsam_config(**common))
    truth = np.array(
        [[0.0, 0.0, 0.4], [3.0, 1.0, 0.5], [5.0, 4.0, 0.8], [0.0, 9.0, 2.0]]
    )
    seeds = truth + np.array(
        [[0.0, 0.0, 0.0], [0.5, -0.2, 0.05], [1.0, -0.4, 0.1], [2.0, -0.2, 0.2]]
    )
    odometry_information = np.diag([25.0, 25.0, 100.0])
    for index, seed in enumerate(seeds):
        for graph in (reference, candidate):
            graph.add_pose(seed, timestamp=float(index))
        if index:
            measurement = _relative_pose2(truth[index - 1], truth[index])
            for graph in (reference, candidate):
                graph.add_edge(
                    index - 1,
                    index,
                    measurement,
                    odometry_information,
                    kind="odometry",
                )
    loop = _relative_pose2(truth[0], truth[-1]) + np.array([1.5, -1.0, 0.2])
    for graph in (reference, candidate):
        graph.add_edge(
            0,
            3,
            loop,
            np.diag([16.0, 16.0, 64.0]),
            kind="loop",
        )
        graph.add_prior(
            3,
            truth[-1] + np.array([1.0, -1.0, 0.2]),
            np.diag([9.0, 9.0, 36.0]),
            robust=True,
        )
    for _ in range(8):
        reference.optimize()
        candidate.optimize()

    np.testing.assert_allclose(candidate.poses, reference.poses, atol=1.0e-6)
    assert candidate.final_cost == pytest.approx(reference.final_cost, abs=1.0e-8)


def test_fixed_lag_marginalizes_and_preserves_last_online_estimates() -> None:
    graph = GTSAMPlanarPoseGraph(_gtsam_config(fixed_lag_seconds=2.0))
    _add_chain(graph, 8)

    assert graph.fixed_lag_active_variables == 3
    assert graph.fixed_lag_marginalized_variables == 5
    np.testing.assert_allclose(graph.poses[:, 0], np.arange(8.0), atol=1e-8)
    np.testing.assert_allclose(
        graph.global_isam_poses[:, 0], np.arange(8.0), atol=1e-8
    )

    graph.add_prior(
        7,
        np.array([6.5, 0.2, 0.01]),
        np.diag([25.0, 25.0, 100.0]),
        robust=False,
    )
    graph.optimize()
    assert graph.poses[-1, 0] < 7.0
    assert graph.fixed_lag_active_variables == 3
    assert graph.isam_updates >= 8
    assert graph.fixed_lag_updates >= 16


def test_fixed_lag_consumes_backlog_longer_than_window_chronologically() -> None:
    graph = GTSAMPlanarPoseGraph(_gtsam_config(fixed_lag_seconds=2.0))
    information = np.diag([100.0, 100.0, 400.0])
    _add_chain(graph, 8)
    for index in range(8, 14):
        graph.add_pose(
            np.array([float(index), 0.0, 0.0]), timestamp=float(index)
        )
        graph.add_edge(
            index - 1,
            index,
            np.array([1.0, 0.0, 0.0]),
            information,
            kind="odometry",
        )
    graph.add_prior(
        13,
        np.array([12.8, 0.1, 0.01]),
        np.diag([25.0, 25.0, 100.0]),
        robust=False,
    )
    graph.optimize()

    assert graph.fixed_lag_active_variables == 3
    assert graph.fixed_lag_marginalized_variables == 11
    assert graph.poses.shape == (14, 3)
    assert np.isfinite(graph.poses).all()

    graph.add_prior(
        2,
        np.array([2.1, -0.1, -0.01]),
        np.diag([4.0, 4.0, 25.0]),
        robust=True,
    )
    graph.optimize()
    assert graph.fixed_lag_active_variables == 3
    assert graph.global_only_factor_count == 1
    assert np.isfinite(graph.poses).all()
    assert np.isfinite(graph.global_isam_poses).all()


def test_fixed_lag_accepts_duplicate_timestamps_from_pose_only_image_nodes() -> None:
    graph = GTSAMPlanarPoseGraph(_gtsam_config(fixed_lag_seconds=2.0))
    information = np.diag([100.0, 100.0, 400.0])
    timestamps = (0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0)
    for index, timestamp in enumerate(timestamps):
        graph.add_pose(np.array([float(index), 0.0, 0.0]), timestamp=timestamp)
        if index:
            graph.add_edge(
                index - 1,
                index,
                np.array([1.0, 0.0, 0.0]),
                information,
                kind="odometry",
            )
    graph.optimize()

    assert graph.fixed_lag_active_variables == 4
    assert graph.fixed_lag_marginalized_variables == 3
    assert np.isfinite(graph.poses).all()


def test_external_relocalization_seed_forces_a_safe_solver_rebuild() -> None:
    graph = GTSAMPlanarPoseGraph(_gtsam_config(fixed_lag_seconds=2.0))
    _add_chain(graph, 8)
    old_edge = graph.edges[-1]
    graph.poses[5:, 0] -= 0.4
    graph.edges[-1] = PoseGraphEdge(
        source=old_edge.source,
        target=old_edge.target,
        measurement=old_edge.measurement + np.array([-0.1, 0.05, 0.01]),
        information=old_edge.information,
        kind=old_edge.kind,
        score=old_edge.score,
    )
    graph.optimize()

    assert graph.rebuilds == 1
    assert graph.fixed_lag_active_variables == 3
    assert graph.fixed_lag_marginalized_variables == 5
    assert np.isfinite(graph.poses).all()
    assert np.isfinite(graph.global_isam_poses).all()


def test_confirmed_distant_terrain_relocalization_rebuilds_live_window() -> None:
    backend = AcousticPoseGraph(
        _gtsam_config(
            fixed_lag_seconds=2.0,
            keyframe_distance_m=0.1,
            keyframe_max_interval_s=1.0,
            terrain_prior_optimize_interval=1,
        )
    )
    identity = np.eye(3)
    empty = np.empty((0, 3), dtype=np.float64)
    for observation_id in range(8):
        assert backend.add_if_needed(
            float(observation_id),
            identity,
            np.array([float(observation_id), 0.0, -5.0]),
            empty,
            observation_id=observation_id,
            force=True,
            detect_loops=False,
        )
        backend.graph.optimize()

    matched = np.array([4_007.0, -2_500.0, 0.35])
    assert backend.add_terrain_prior(
        timestamp=7.0,
        observation_id=7,
        prior_pose2=np.array([7.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([7.0, 0.0, -5.0]),
        matched_pose2=matched,
        information_world=np.diag([10_000.0, 10_000.0, 10_000.0]),
        optimize_interval=1,
        relocalization_seed=True,
    )

    np.testing.assert_allclose(backend.graph.poses[-1], matched, atol=2.0e-4)
    diagnostics = backend.diagnostics()
    assert diagnostics.last_terrain_prefit_bypassed_for_relocalization
    assert not diagnostics.last_terrain_prefit_performed
    assert diagnostics.terrain_relocalization_seeds == 1
    assert diagnostics.optimizer_backend == "gtsam_isam2_incremental_fixed_lag"
    assert diagnostics.fixed_lag_active_variables == 3


def test_rebuild_bridges_a_new_cross_window_factor_exactly_once() -> None:
    graph = GTSAMPlanarPoseGraph(_gtsam_config(fixed_lag_seconds=2.0))
    _add_chain(graph, 8)

    # A relocalization seed edits existing values, so the next optimize must
    # rebuild both solvers.  Commit a long-baseline factor in that same update:
    # its old endpoint has already left the live two-second window.
    graph.poses[5:, 1] += 1.0
    graph.add_edge(
        0,
        7,
        np.array([6.5, 0.2, 0.01]),
        np.diag([25.0, 25.0, 100.0]),
        kind="relocalization",
    )
    graph.optimize()

    assert graph.rebuilds == 1
    assert graph.global_only_factor_count == 1
    assert graph.global_anchor_bridge_updates == 1
    np.testing.assert_allclose(
        graph.poses[-1], graph.global_isam_poses[-1], atol=1.0e-10
    )

    # A later update may reapply the persistent map anchor, but the unique
    # global-only factor count must not grow simply because the graph ran again.
    graph.optimize()
    assert graph.global_only_factor_count == 1
    assert np.isfinite(graph.poses).all()


def test_old_to_live_loop_anchors_window_without_reviving_old_key() -> None:
    graph = GTSAMPlanarPoseGraph(_gtsam_config(fixed_lag_seconds=2.0))
    _add_chain(graph, 8)
    online_before = graph.poses.copy()
    global_before = graph.global_isam_poses.copy()
    graph.add_edge(
        0,
        7,
        np.array([6.5, 0.2, 0.01]),
        np.diag([25.0, 25.0, 100.0]),
        kind="relocalization",
    )
    graph.optimize()

    assert graph.global_only_factor_count == 1
    assert graph.fixed_lag_active_variables == 3
    np.testing.assert_allclose(graph.poses[:5], online_before[:5], atol=1.0e-10)
    np.testing.assert_allclose(
        graph.poses[-1], graph.global_isam_poses[-1], atol=1.0e-10
    )
    np.testing.assert_allclose(
        _relative_pose2(graph.poses[5], graph.poses[7]),
        _relative_pose2(online_before[5], online_before[7]),
        atol=1.0e-10,
    )
    assert graph.global_anchor_bridge_updates == 1
    assert not np.allclose(graph.global_isam_poses, global_before)

    # The public live-window poses now carry the rigid map anchor while the
    # smoother retains its own marginal frame internally.  Subsequent nodes
    # must still enter continuously and preserve the odometry increment; this
    # catches a frame mismatch on the first update after relocalization.
    previous = graph.poses[-1].copy()
    information = np.diag([100.0, 100.0, 400.0])
    for index in range(8, 13):
        seed = np.array(
            [
                previous[0] + np.cos(previous[2]),
                previous[1] + np.sin(previous[2]),
                previous[2],
            ]
        )
        graph.add_pose(seed, timestamp=float(index))
        graph.add_edge(
            index - 1,
            index,
            np.array([1.0, 0.0, 0.0]),
            information,
            kind="odometry",
        )
        graph.optimize()
        np.testing.assert_allclose(
            _relative_pose2(previous, graph.poses[-1]),
            np.array([1.0, 0.0, 0.0]),
            atol=2.0e-4,
        )
        np.testing.assert_allclose(
            graph.poses[-1], graph.global_isam_poses[-1], atol=2.0e-4
        )
        previous = graph.poses[-1].copy()

    assert graph.fixed_lag_active_variables == 3
    assert graph.fixed_lag_marginalized_variables == 10
    assert graph.global_anchor_bridge_updates == 6
    assert np.isfinite(graph.poses).all()


def test_rank_deficient_acoustic_information_is_not_given_fake_rank() -> None:
    graph = GTSAMPlanarPoseGraph(_gtsam_config())
    basis, _ = np.linalg.qr(
        np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5], [0.3, -0.7, 2.0]])
    )
    information_matrices = (
        np.diag([16.0, 0.0, 0.0]),
        basis @ np.diag([25.0, 4.0, 0.0]) @ basis.T,
    )
    for expected_rank, information in enumerate(information_matrices, start=1):
        noise = graph._noise_model(information)
        recovered = np.asarray(noise.R()).T @ np.asarray(noise.R())
        np.testing.assert_allclose(recovered, information, atol=2.0e-12)
        assert np.linalg.matrix_rank(recovered, tol=1.0e-10) == expected_rank


def test_acoustic_frontend_constructs_identical_factors_for_both_solvers() -> None:
    common = {
        "keyframe_distance_m": 0.1,
        "keyframe_max_interval_s": 1.0,
        "icp_min_inliers": 3,
        "icp_max_points": 20,
        "optimizer_iterations": 50,
        "fixed_lag_seconds": 100.0,
        "isam_relinearize_threshold": 1.0e-5,
    }
    reference = AcousticPoseGraph(PoseGraphConfig(**common))
    candidate = AcousticPoseGraph(_gtsam_config(**common))
    points = np.array(
        [[-1.0, -2.0, -10.0], [0.0, 0.0, -10.2], [1.0, 2.0, -9.8]]
    )
    poses = (
        np.array([0.0, 0.0, 0.3]),
        np.array([2.0, 1.0, 0.35]),
        np.array([4.2, 2.1, 0.42]),
    )
    for index, pose in enumerate(poses):
        rotation = rotation_from_rpy(0.0, 0.0, pose[2])
        position = np.array([pose[0], pose[1], -40.0])
        for backend in (reference, candidate):
            assert backend.add_if_needed(
                float(index),
                rotation,
                position,
                points,
                observation_id=index,
                force=True,
                odometry_pose2=pose,
                detect_loops=False,
            )

    matched = poses[-1] + np.array([-0.2, 0.1, -0.01])
    information = np.array(
        [[12.0, 2.0, 0.2], [2.0, 9.0, -0.1], [0.2, -0.1, 80.0]]
    )
    for backend in (reference, candidate):
        assert backend.add_terrain_prior(
            timestamp=2.0,
            observation_id=2,
            prior_pose2=poses[-1],
            raw_rotation_world_body=rotation_from_rpy(0.0, 0.0, poses[-1][2]),
            raw_position_world=np.array([poses[-1][0], poses[-1][1], -40.0]),
            matched_pose2=matched,
            information_world=information,
            optimize_interval=1,
        )

    reference_stream = reference.graph.factor_stream_payload()
    candidate_stream = candidate.graph.factor_stream_payload()
    for key in ("pose_initializations", "pose_timestamps_s", "edges", "priors"):
        assert candidate_stream[key] == reference_stream[key]
    json.dumps(candidate_stream)
    np.testing.assert_allclose(
        candidate.graph.poses, reference.graph.poses, atol=1.0e-4
    )
