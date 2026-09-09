from __future__ import annotations

import numpy as np
import pytest

from oceansim.slam import AcousticPoseGraph, PlanarPoseGraph, PoseGraphConfig
from oceansim.slam.math3d import rotation_from_rpy, wrap_angle


def _information(xy: float = 100.0, yaw: float = 400.0) -> np.ndarray:
    return np.diag([xy, xy, yaw]).astype(np.float64)


def test_odometry_random_walk_information_is_invariant_to_keyframe_cadence() -> None:
    """Two 30 s edges must carry the same 60 s variance as one 60 s edge."""

    identity = np.eye(3)
    empty = np.empty((0, 3), dtype=np.float64)

    def build(timestamps: tuple[float, ...]) -> AcousticPoseGraph:
        backend = AcousticPoseGraph(
            PoseGraphConfig(
                odometry_reference_interval_s=60.0,
                odometry_xy_std_m=0.32,
                odometry_yaw_std_deg=0.8,
            )
        )
        for observation_id, timestamp in enumerate(timestamps):
            assert backend.add_if_needed(
                timestamp,
                identity,
                np.array([timestamp, 0.0, -5.0]),
                empty,
                observation_id=observation_id,
                force=True,
                detect_loops=False,
            )
        return backend

    one_edge = build((0.0, 60.0))
    two_edges = build((0.0, 30.0, 60.0))
    one_variance = 1.0 / np.diag(one_edge.graph.edges[0].information)
    two_variance = sum(
        (1.0 / np.diag(edge.information) for edge in two_edges.graph.edges),
        start=np.zeros(3, dtype=np.float64),
    )
    np.testing.assert_allclose(two_variance, one_variance, rtol=1.0e-12)


def test_absolute_terrain_prior_corrects_planar_odometry_drift() -> None:
    graph = PlanarPoseGraph(PoseGraphConfig(optimizer_iterations=30))
    graph.add_pose(np.array([0.0, 0.0, 0.0]))
    graph.add_pose(np.array([10.1, -0.05, 0.01]))
    graph.add_pose(np.array([20.2, -0.1, 0.02]))
    odometry_information = _information(xy=100.0, yaw=400.0)
    graph.add_edge(
        0,
        1,
        np.array([10.0, 0.0, 0.0]),
        odometry_information,
        kind="odometry",
    )
    graph.add_edge(
        1,
        2,
        np.array([10.0, 0.0, 0.0]),
        odometry_information,
        kind="odometry",
    )
    graph.add_prior(
        2,
        np.array([20.0, 0.0, 0.0]),
        _information(xy=25.0, yaw=100.0),
    )

    before = np.linalg.norm(graph.poses[2, :2] - np.array([20.0, 0.0]))
    optimized = graph.optimize()

    assert np.linalg.norm(optimized[2, :2] - [20.0, 0.0]) < 1.0e-5
    assert np.linalg.norm(optimized[2, :2] - [20.0, 0.0]) < before
    np.testing.assert_allclose(optimized[0], [0.0, 0.0, 0.0])
    assert graph.final_cost < graph.initial_cost


def test_cauchy_terrain_kernel_bounds_a_gross_alias_more_than_huber() -> None:
    """A redescending loss should suppress a map alias, not delete its factor."""

    def optimized_endpoint(kernel: str) -> float:
        graph = PlanarPoseGraph(
            PoseGraphConfig(
                optimizer_iterations=50,
                terrain_prior_robust_kernel=kernel,
                robust_terrain_prior_delta=2.5,
            )
        )
        graph.add_pose(np.array([0.0, 0.0, 0.0]))
        graph.add_pose(np.array([10.0, 0.0, 0.0]))
        graph.add_edge(
            0,
            1,
            np.array([10.0, 0.0, 0.0]),
            _information(xy=100.0, yaw=400.0),
            kind="odometry",
        )
        graph.add_prior(
            1,
            np.array([100.0, 0.0, 0.0]),
            _information(xy=100.0, yaw=400.0),
        )
        return float(graph.optimize()[1, 0])

    huber_endpoint = optimized_endpoint("huber")
    cauchy_endpoint = optimized_endpoint("cauchy")
    assert abs(cauchy_endpoint - 10.0) < abs(huber_endpoint - 10.0)
    assert abs(cauchy_endpoint - 10.0) < 0.1


def test_invalid_terrain_robust_kernel_is_rejected() -> None:
    with pytest.raises(ValueError, match="terrain_prior_robust_kernel"):
        PlanarPoseGraph(PoseGraphConfig(terrain_prior_robust_kernel="invalid"))


def test_absolute_terrain_prior_wraps_yaw_at_pi() -> None:
    graph = PlanarPoseGraph(PoseGraphConfig(optimizer_iterations=30))
    initial_yaw = -np.pi + 0.02
    matched_yaw = np.pi - 0.02
    graph.add_pose(np.zeros(3))
    graph.add_pose(np.array([1.0, 0.0, initial_yaw]))
    graph.add_edge(
        0,
        1,
        np.array([1.0, 0.0, initial_yaw]),
        _information(xy=100.0, yaw=1.0),
        kind="odometry",
    )
    graph.add_prior(
        1,
        np.array([1.0, 0.0, matched_yaw]),
        _information(xy=100.0, yaw=1_000.0),
    )

    optimized_yaw = float(graph.optimize()[1, 2])

    assert abs(float(wrap_angle(optimized_yaw - matched_yaw))) < 1.0e-3
    assert abs(optimized_yaw) > 3.0


def test_terrain_prior_robust_kernel_limits_a_large_outlier() -> None:
    graph = PlanarPoseGraph(
        PoseGraphConfig(
            optimizer_iterations=40,
            robust_terrain_prior_delta=2.5,
        )
    )
    for x in (0.0, 1.0, 2.0):
        graph.add_pose(np.array([x, 0.0, 0.0]))
    odometry_information = _information(xy=400.0, yaw=400.0)
    for source in (0, 1):
        graph.add_edge(
            source,
            source + 1,
            np.array([1.0, 0.0, 0.0]),
            odometry_information,
            kind="odometry",
        )
    graph.add_prior(
        2,
        np.array([100.0, 0.0, 0.0]),
        _information(xy=10_000.0, yaw=10_000.0),
    )

    optimized = graph.optimize()

    # Without a robust loss this very confident false match would put the
    # trajectory close to x=100. The standardized Huber influence is bounded.
    assert 2.0 < optimized[2, 0] < 5.0
    assert optimized[1, 0] < 3.0


def test_confirmed_terrain_relocalization_rebases_an_unknown_initial_gauge() -> None:
    """A verified absolute recovery must move the gauge, not bend the chain."""

    backend = AcousticPoseGraph(
        PoseGraphConfig(
            optimizer_iterations=30,
            terrain_prior_optimize_interval=1,
        )
    )
    identity = np.eye(3)
    empty = np.empty((0, 3), dtype=np.float64)
    # The complete local trajectory has a 120 m launch-coordinate offset;
    # all relative odometry is nevertheless internally consistent.
    for observation_id, x_m in enumerate((120.0, 130.0, 140.0)):
        assert backend.add_if_needed(
            float(observation_id),
            identity,
            np.array([x_m, 0.0, -5.0]),
            empty,
            observation_id=observation_id,
            force=True,
            detect_loops=False,
        )

    assert backend.add_terrain_prior(
        timestamp=2.0,
        observation_id=2,
        prior_pose2=np.array([140.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([140.0, 0.0, -5.0]),
        matched_pose2=np.array([20.0, 0.0, 0.0]),
        information_world=np.diag([0.2, 0.2, 25.0]),
        optimize_interval=1,
        relocalization_seed=True,
    )

    np.testing.assert_allclose(
        backend.graph.poses,
        np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        atol=1.0e-6,
    )
    assert not backend.graph.priors[-1].robust
    diagnostics = backend.diagnostics()
    assert diagnostics.terrain_relocalization_seeds == 1
    assert not diagnostics.last_terrain_prefit_performed
    assert diagnostics.last_terrain_prefit_bypassed_for_relocalization
    assert diagnostics.last_terrain_relocalization_translation_m == pytest.approx(
        120.0
    )
    assert diagnostics.last_terrain_relocalization_yaw_deg == pytest.approx(0.0)
    assert diagnostics.last_terrain_relocalization_anchor_index is None


def test_confirmed_relocalization_can_cross_a_distant_graph_basin() -> None:
    """Independent confirmation, not the stale graph, gates relocalization."""

    backend = AcousticPoseGraph(
        PoseGraphConfig(
            optimizer_iterations=30,
            terrain_prior_optimize_interval=1,
        )
    )
    identity = np.eye(3)
    empty = np.empty((0, 3), dtype=np.float64)
    for observation_id, x_m in enumerate((0.0, 10.0, 20.0)):
        assert backend.add_if_needed(
            float(observation_id),
            identity,
            np.array([x_m, 0.0, -5.0]),
            empty,
            observation_id=observation_id,
            force=True,
            detect_loops=False,
        )

    matched = np.array([5_020.0, -3_000.0, 0.4])
    assert backend.add_terrain_prior(
        timestamp=2.0,
        observation_id=2,
        prior_pose2=np.array([20.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([20.0, 0.0, -5.0]),
        matched_pose2=matched,
        information_world=np.diag([10_000.0, 10_000.0, 10_000.0]),
        optimize_interval=1,
        relocalization_seed=True,
    )

    np.testing.assert_allclose(backend.graph.poses[-1], matched, atol=1.0e-7)
    diagnostics = backend.diagnostics()
    assert diagnostics.last_terrain_prefit_bypassed_for_relocalization
    assert not diagnostics.last_terrain_prefit_performed
    assert diagnostics.last_terrain_prefit_chi2 == 0.0
    assert diagnostics.last_terrain_prefit_dof == 0


def test_confirmed_relocalization_starts_after_latest_absolute_anchor() -> None:
    """The recovery seed is introduced only across the outage segment."""

    backend = AcousticPoseGraph(
        PoseGraphConfig(
            optimizer_iterations=50,
            terrain_prior_optimize_interval=100,
        )
    )
    identity = np.eye(3)
    empty = np.empty((0, 3), dtype=np.float64)
    for observation_id, x_m in enumerate((0.0, 10.0, 25.0, 40.0)):
        assert backend.add_if_needed(
            float(observation_id),
            identity,
            np.array([x_m, 0.0, -5.0]),
            empty,
            observation_id=observation_id,
            force=True,
            detect_loops=False,
        )
    assert backend.add_terrain_prior(
        timestamp=1.0,
        observation_id=1,
        prior_pose2=np.array([10.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([10.0, 0.0, -5.0]),
        matched_pose2=np.array([10.0, 0.0, 0.0]),
        information_world=_information(),
        optimize_interval=100,
    )
    assert backend.add_terrain_prior(
        timestamp=3.0,
        observation_id=3,
        prior_pose2=np.array([40.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([40.0, 0.0, -5.0]),
        matched_pose2=np.array([30.0, 0.0, 0.0]),
        information_world=_information(),
        optimize_interval=1,
        relocalization_seed=True,
    )

    # Node zero retains its history and the current pose reaches the new
    # absolute basin.  The robust historical prior can move slightly when the
    # optimizer balances systematic odometry drift, but far less than the
    # previously unanchored current branch.
    np.testing.assert_allclose(backend.graph.poses[0], [0.0, 0.0, 0.0])
    assert abs(backend.graph.poses[1, 0] - 10.0) < 3.0
    assert abs(backend.graph.poses[3, 0] - 30.0) < 1.0
    diagnostics = backend.diagnostics()
    assert diagnostics.last_terrain_relocalization_anchor_index == 1


def test_acoustic_terrain_prior_reuses_nodes_and_keeps_factors_independent() -> None:
    backend = AcousticPoseGraph(
        PoseGraphConfig(
            loop_min_separation_s=1_000.0,
            terrain_prior_optimize_interval=100,
        )
    )
    identity = np.eye(3)
    empty = np.empty((0, 3), dtype=np.float64)
    assert backend.add_if_needed(
        0.0,
        identity,
        np.array([0.0, 0.0, -5.0]),
        empty,
        observation_id=0,
        force=True,
        detect_loops=False,
    )
    assert backend.add_if_needed(
        1.0,
        identity,
        np.array([10.0, 0.0, -5.0]),
        empty,
        observation_id=1,
        force=True,
        detect_loops=False,
    )

    # Observation 1 already has a node, so only the unary factor is added.
    assert backend.add_terrain_prior(
        timestamp=1.0,
        observation_id=1,
        prior_pose2=np.array([10.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([10.0, 0.0, -5.0]),
        matched_pose2=np.array([9.5, 0.2, 0.01]),
        information_world=_information(),
    )
    assert len(backend.keyframes) == 2
    assert backend.graph.priors[-1].target == 1

    # A new observation gets a pose-only node. Its odometry edge terminates
    # at the *pre-match* prior (x=21), not the absolute optimum (x=18).
    raw_rotation = rotation_from_rpy(0.0, 0.0, 0.1)
    assert backend.add_terrain_prior(
        timestamp=2.0,
        observation_id=2,
        prior_pose2=np.array([21.0, 0.0, 0.1]),
        raw_rotation_world_body=raw_rotation,
        raw_position_world=np.array([21.4, 0.0, -5.0]),
        matched_pose2=np.array([18.0, 1.0, 0.05]),
        information_world=_information(),
    )
    incoming = next(
        edge
        for edge in backend.graph.edges
        if edge.kind == "odometry" and edge.target == 2
    )
    np.testing.assert_allclose(incoming.measurement, [11.0, 0.0, 0.1])
    np.testing.assert_allclose(
        backend.graph.priors[-1].measurement,
        [18.0, 1.0, 0.05],
    )
    assert not backend.keyframes[-1].loop_eligible
    diagnostics = backend.diagnostics()
    assert diagnostics.terrain_prior_count == 2
    assert diagnostics.accepted_terrain_priors == 2
    assert diagnostics.rejected_terrain_priors == 0


def test_rejected_or_duplicate_terrain_prior_does_not_pollute_topology() -> None:
    backend = AcousticPoseGraph(
        PoseGraphConfig(terrain_prior_optimize_interval=100)
    )
    identity = np.eye(3)
    assert backend.add_if_needed(
        0.0,
        identity,
        np.array([0.0, 0.0, -5.0]),
        np.empty((0, 3)),
        observation_id=0,
        force=True,
        detect_loops=False,
    )
    assert backend.add_terrain_prior(
        timestamp=1.0,
        observation_id=1,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([1.0, 0.0, -5.0]),
        matched_pose2=np.array([0.9, 0.1, 0.0]),
        information_world=_information(),
    )

    topology = (
        len(backend.keyframes),
        backend.graph.poses.shape[0],
        len(backend.graph.edges),
        len(backend.graph.priors),
    )
    assert not backend.add_terrain_prior(
        timestamp=1.0,
        observation_id=1,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([1.0, 0.0, -5.0]),
        matched_pose2=np.array([0.8, 0.0, 0.0]),
        information_world=_information(),
    )
    assert not backend.add_terrain_prior(
        timestamp=2.0,
        observation_id=2,
        prior_pose2=np.array([2.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([2.0, 0.0, -5.0]),
        matched_pose2=np.array([2.0, 0.0, 0.0]),
        information_world=np.diag([1.0, -1.0, 1.0]),
    )
    assert topology == (
        len(backend.keyframes),
        backend.graph.poses.shape[0],
        len(backend.graph.edges),
        len(backend.graph.priors),
    )
    diagnostics = backend.diagnostics()
    assert diagnostics.accepted_terrain_priors == 1
    assert diagnostics.rejected_terrain_priors == 2
    assert diagnostics.terrain_prior_rejection_reasons == {
        "duplicate_observation": 1,
        "invalid_information": 1,
    }

    backend.reset()
    reset_diagnostics = backend.diagnostics()
    assert reset_diagnostics.terrain_prior_count == 0
    assert reset_diagnostics.accepted_terrain_priors == 0
    assert reset_diagnostics.rejected_terrain_priors == 0
    assert reset_diagnostics.terrain_prior_rejection_reasons == {}
    assert not reset_diagnostics.last_terrain_prefit_performed
    assert reset_diagnostics.last_terrain_prefit_chi2 == 0.0
    assert reset_diagnostics.last_terrain_prefit_dof == 0


def test_terrain_prefit_accepts_large_but_compatible_global_drift() -> None:
    """The broad safety gate must not defeat absolute drift recovery."""

    backend = AcousticPoseGraph(
        PoseGraphConfig(terrain_prior_optimize_interval=100)
    )
    identity = np.eye(3)
    empty = np.empty((0, 3), dtype=np.float64)
    assert backend.add_if_needed(
        0.0,
        identity,
        np.array([0.0, 0.0, -5.0]),
        empty,
        observation_id=0,
        force=True,
        detect_loops=False,
    )
    assert backend.add_if_needed(
        1.0,
        identity,
        np.array([10.0, 0.0, -5.0]),
        empty,
        observation_id=1,
        force=True,
        detect_loops=False,
    )

    # Represent a previously optimized +12 m/-3 m global correction.  The
    # next node must be predicted by propagating its incoming odometry from
    # this graph pose, not by falling back to raw local coordinates.
    backend.graph.poses[1] = np.array([22.0, -3.0, 0.2])
    backend.graph.optimization_runs = 1
    prior = np.array([20.0, 0.0, 0.1])
    predicted = np.array(
        [
            22.0 + 10.0 * np.cos(0.2),
            -3.0 + 10.0 * np.sin(0.2),
            0.3,
        ]
    )
    # A 14.4 m independent absolute correction is plausible inside the
    # terrain search basin and should pass the deliberately broad default.
    matched = predicted + np.array([12.0, -8.0, 0.04])
    information = np.diag([4.0, 4.0, 400.0])

    assert backend.add_terrain_prior(
        timestamp=2.0,
        observation_id=2,
        prior_pose2=prior,
        raw_rotation_world_body=rotation_from_rpy(0.0, 0.0, 0.1),
        raw_position_world=np.array([20.0, 0.0, -5.0]),
        matched_pose2=matched,
        information_world=information,
    )

    np.testing.assert_allclose(backend.graph.poses[2], predicted)
    diagnostics = backend.diagnostics()
    assert diagnostics.last_terrain_prefit_performed
    assert diagnostics.last_terrain_prefit_dof == 3
    assert diagnostics.last_terrain_prefit_chi2 == pytest.approx(832.64)
    assert diagnostics.last_terrain_prior_rejection_reason is None


def test_terrain_prefit_rejects_confident_jump_transactionally() -> None:
    backend = AcousticPoseGraph(
        PoseGraphConfig(terrain_prior_optimize_interval=100)
    )
    identity = np.eye(3)
    assert backend.add_if_needed(
        0.0,
        identity,
        np.array([0.0, 0.0, -5.0]),
        np.empty((0, 3)),
        observation_id=0,
        force=True,
        detect_loops=False,
    )
    poses_before = backend.graph.poses.copy()
    topology_before = (
        len(backend.keyframes),
        len(backend.graph.edges),
        len(backend.graph.priors),
    )

    # A false multi-modal optimum 20 m away with centimetric claimed
    # uncertainty is far beyond the broad catastrophic-jump threshold.
    assert not backend.add_terrain_prior(
        timestamp=1.0,
        observation_id=1,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([1.0, 0.0, -5.0]),
        matched_pose2=np.array([21.0, 0.0, 0.0]),
        information_world=np.diag([10_000.0, 10_000.0, 2_500.0]),
    )

    np.testing.assert_array_equal(backend.graph.poses, poses_before)
    assert topology_before == (
        len(backend.keyframes),
        len(backend.graph.edges),
        len(backend.graph.priors),
    )
    diagnostics = backend.diagnostics()
    assert diagnostics.last_terrain_prefit_performed
    assert diagnostics.last_terrain_prefit_dof == 3
    assert diagnostics.last_terrain_prefit_chi2 == pytest.approx(4_000_000.0)
    assert diagnostics.last_terrain_prior_rejection_reason == "prefit_inconsistent"
    assert diagnostics.terrain_prior_rejection_reasons == {
        "prefit_inconsistent": 1
    }

    # A rejected hypothesis does not consume the observation id.
    assert backend.add_terrain_prior(
        timestamp=1.0,
        observation_id=1,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([1.0, 0.0, -5.0]),
        matched_pose2=np.array([1.1, 0.0, 0.0]),
        information_world=np.diag([100.0, 100.0, 400.0]),
    )


def test_rank_two_terrain_prefit_ignores_unobservable_yaw() -> None:
    backend = AcousticPoseGraph(
        PoseGraphConfig(
            terrain_prior_optimize_interval=100,
            terrain_prior_prefit_chi2_rank2=120.0,
            # Prove the rank-two limit, rather than the rank-three limit, is
            # selected for a yaw-null measurement.
            terrain_prior_prefit_chi2_rank3=1.0,
        )
    )
    identity = np.eye(3)
    assert backend.add_if_needed(
        0.0,
        identity,
        np.array([0.0, 0.0, -5.0]),
        np.empty((0, 3)),
        observation_id=0,
        force=True,
        detect_loops=False,
    )

    assert backend.add_terrain_prior(
        timestamp=1.0,
        observation_id=1,
        prior_pose2=np.array([1.0, 0.0, 0.2]),
        raw_rotation_world_body=rotation_from_rpy(0.0, 0.0, 0.2),
        raw_position_world=np.array([1.0, 0.0, -5.0]),
        matched_pose2=np.array([3.0, -1.0, 2.5]),
        information_world=np.diag([25.0, 16.0, 0.0]),
    )
    diagnostics = backend.diagnostics()
    assert diagnostics.last_terrain_prefit_performed
    assert diagnostics.last_terrain_prefit_dof == 2
    assert diagnostics.last_terrain_prefit_chi2 == pytest.approx(116.0)


def test_terrain_prior_new_node_requires_strictly_new_timestamp() -> None:
    backend = AcousticPoseGraph(
        PoseGraphConfig(terrain_prior_optimize_interval=100)
    )
    identity = np.eye(3)
    assert backend.add_if_needed(
        1.0,
        identity,
        np.array([1.0, 0.0, -5.0]),
        np.empty((0, 3)),
        observation_id=1,
        force=True,
        detect_loops=False,
    )
    poses_before = backend.graph.poses.copy()

    assert not backend.add_terrain_prior(
        timestamp=1.0,
        observation_id=2,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        raw_rotation_world_body=identity,
        raw_position_world=np.array([1.0, 0.0, -5.0]),
        matched_pose2=np.array([1.0, 0.0, 0.0]),
        information_world=_information(),
    )
    np.testing.assert_array_equal(backend.graph.poses, poses_before)
    diagnostics = backend.diagnostics()
    assert diagnostics.last_terrain_prior_rejection_reason == (
        "nonmonotonic_timestamp"
    )
    assert not diagnostics.last_terrain_prefit_performed
