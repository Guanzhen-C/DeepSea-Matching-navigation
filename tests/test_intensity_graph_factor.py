from __future__ import annotations

import numpy as np
import pytest

from oceansim.slam import (
    AcousticESIKF,
    AcousticIntensityMap,
    AcousticPoseGraph,
    ESIKFConfig,
    PoseGraphConfig,
)
from oceansim.slam.math3d import rotation_from_rpy
from oceansim.slam.pose_graph import sidescan_descriptor


def _feature_cloud() -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-3.0, 3.0, 14), np.linspace(-5.0, 5.0, 10))
    z = -8.0 + 0.08 * x + 0.03 * y + 0.12 * np.sin(0.5 * x)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def test_intensity_query_does_not_mix_newer_reference_observations() -> None:
    intensity_map = AcousticIntensityMap(
        voxel_size=1.0,
        min_cell_points=1,
        min_neighbours=5,
        min_gradient=1.0e-5,
    )
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 17), np.linspace(-2.0, 2.0, 17))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -8.0)))
    old_values = 0.4 + 0.04 * x.ravel() - 0.03 * y.ravel()
    intensity_map.add_points(points, old_values, heading_rad=0.0, observation_id=0)
    intensity_map.add_points(
        points,
        1.0 - old_values,
        heading_rad=0.0,
        observation_id=100,
    )

    assert (
        intensity_map.query(
            np.array([0.1, -0.1, -8.0]),
            heading_rad=0.0,
            maximum_observation_id=0,
        )
        is None
    )


def test_accepted_intensity_alignment_has_independent_odom_and_image_edges() -> None:
    config = PoseGraphConfig(
        keyframe_distance_m=0.0,
        keyframe_max_interval_s=0.1,
        loop_min_separation_s=1_000.0,
        icp_min_inliers=30,
    )
    backend = AcousticPoseGraph(config)
    cloud = _feature_cloud()
    appearance = sidescan_descriptor(np.linspace(0.0, 1.0, 80), np.linspace(1.0, 0.0, 80))
    identity = np.eye(3)

    assert backend.add_if_needed(
        0.0,
        identity,
        np.array([0.0, 0.0, -5.0]),
        cloud,
        None,
        appearance,
        observation_id=0,
    )
    assert backend.add_if_needed(
        1.0,
        identity,
        np.array([10.0, 0.0, -5.0]),
        cloud,
        None,
        appearance,
        observation_id=10,
    )
    assert backend.add_intensity_observation(
        timestamp=1.5,
        observation_id=15,
        anchor_keyframe_index=0,
        # The odometry prior drifted to x=11.5. The filter posterior lies
        # between that prior and the independent image-only optimum below.
        prior_pose2=np.array([11.5, 0.0, 0.0]),
        posterior_rotation_world_body=identity,
        posterior_position_world=np.array([11.2, 0.1, -5.0]),
        photometric_pose2=np.array([11.0, 0.2, 0.0]),
        information_world=np.diag([25.0, 16.0, 900.0]),
        points_body=cloud,
        intensities=np.linspace(0.2, 0.8, cloud.shape[0]),
        appearance_descriptor=appearance,
        score=0.92,
    )

    diagnostics = backend.diagnostics()
    assert diagnostics.intensity_edges == 1
    assert diagnostics.pending_intensity_factors == 0
    assert diagnostics.optimization_runs == 1
    odometry = next(
        edge
        for edge in backend.graph.edges
        if edge.kind == "odometry" and edge.target == 2
    )
    factor = next(edge for edge in backend.graph.edges if edge.kind == "intensity")
    assert (factor.source, factor.target) == (0, 2)
    # The image optimum changes only the image measurement. It is absent from
    # the incoming odometry edge, so the same residual is not counted twice.
    np.testing.assert_allclose(odometry.measurement, [1.5, 0.0, 0.0])
    np.testing.assert_allclose(factor.measurement, [11.0, 0.2, 0.0])
    assert not np.allclose(
        factor.measurement,
        [11.2, 0.1, 0.0],
    )
    assert np.all(np.linalg.eigvalsh(factor.information) > 0.0)


def test_intensity_factor_cadence_bounds_pose_only_node_growth() -> None:
    config = PoseGraphConfig(
        keyframe_distance_m=100.0,
        keyframe_max_interval_s=1_000.0,
        loop_min_separation_s=1_000.0,
        icp_min_inliers=30,
        intensity_factor_min_interval_s=60.0,
        intensity_factor_optimize_interval=100,
    )
    backend = AcousticPoseGraph(config)
    cloud = _feature_cloud()
    identity = np.eye(3)
    assert backend.add_if_needed(
        0.0,
        identity,
        np.array([0.0, 0.0, -5.0]),
        cloud,
        observation_id=0,
    )
    common = dict(
        anchor_keyframe_index=0,
        posterior_rotation_world_body=identity,
        information_world=np.diag([25.0, 16.0, 900.0]),
        points_body=cloud,
        intensities=None,
        appearance_descriptor=None,
        score=0.9,
    )
    assert backend.add_intensity_observation(
        timestamp=1.0,
        observation_id=10,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        posterior_position_world=np.array([0.9, 0.0, -5.0]),
        photometric_pose2=np.array([0.8, 0.0, 0.0]),
        **common,
    )
    node_count = len(backend.keyframes)
    edge_count = len(backend.graph.edges)

    assert not backend.add_intensity_observation(
        timestamp=2.0,
        observation_id=20,
        prior_pose2=np.array([1.9, 0.0, 0.0]),
        posterior_position_world=np.array([1.8, 0.0, -5.0]),
        photometric_pose2=np.array([1.7, 0.0, 0.0]),
        **common,
    )
    assert len(backend.keyframes) == node_count
    assert len(backend.graph.edges) == edge_count
    diagnostics = backend.diagnostics()
    assert diagnostics.intensity_edges == 1
    assert diagnostics.throttled_intensity_factors == 1
    assert diagnostics.rejected_intensity_factors == 0


def test_rank_deficient_image_hessian_keeps_its_nullspace() -> None:
    backend = AcousticPoseGraph(PoseGraphConfig())
    result = backend._intensity_information_in_source(
        np.diag([25.0, 16.0, 0.0]), source_yaw=0.63
    )
    assert result is not None
    eigenvalues = np.linalg.eigvalsh(result)
    assert eigenvalues[0] == pytest.approx(0.0, abs=1.0e-10)
    assert np.count_nonzero(eigenvalues > 1.0e-8) == 2


def test_pose_only_intensity_node_accepts_fewer_points_than_loop_icp() -> None:
    config = PoseGraphConfig(
        keyframe_distance_m=0.0,
        loop_min_separation_s=1_000.0,
        icp_min_inliers=70,
    )
    backend = AcousticPoseGraph(config)
    cloud = _feature_cloud()
    assert backend.add_if_needed(
        0.0,
        np.eye(3),
        np.array([0.0, 0.0, -5.0]),
        cloud,
        observation_id=0,
    )

    # The ESIKF can accept a low-dimensional direct patch with fewer points
    # than geometric ICP requires. It must still reach the graph as a
    # pose-only node; loop detection remains disabled for that node.
    assert backend.add_intensity_observation(
        timestamp=1.0,
        observation_id=10,
        anchor_keyframe_index=0,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        posterior_rotation_world_body=np.eye(3),
        posterior_position_world=np.array([0.9, 0.0, -5.0]),
        photometric_pose2=np.array([0.8, 0.0, 0.0]),
        information_world=np.diag([16.0, 9.0, 400.0]),
        points_body=cloud[:24],
        intensities=np.linspace(0.2, 0.8, 24),
        appearance_descriptor=None,
        score=0.9,
    )
    assert backend.diagnostics().intensity_edges == 1


def test_equal_timestamp_corrections_use_post_image_node() -> None:
    config = PoseGraphConfig(
        keyframe_distance_m=0.0,
        loop_min_separation_s=1_000.0,
        icp_min_inliers=30,
    )
    backend = AcousticPoseGraph(config)
    cloud = _feature_cloud()
    assert backend.add_if_needed(
        0.0,
        np.eye(3),
        np.array([0.0, 0.0, -5.0]),
        cloud,
        observation_id=0,
    )
    assert backend.add_if_needed(
        1.0,
        np.eye(3),
        np.array([1.0, 0.0, -5.0]),
        cloud,
        observation_id=10,
    )
    assert backend.add_intensity_observation(
        timestamp=1.0,
        observation_id=10,
        anchor_keyframe_index=0,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        posterior_rotation_world_body=np.eye(3),
        posterior_position_world=np.array([0.8, 0.0, -5.0]),
        photometric_pose2=np.array([0.7, 0.0, 0.0]),
        information_world=np.diag([100.0, 100.0, 1_000.0]),
        points_body=cloud,
        intensities=None,
        appearance_descriptor=None,
        score=0.95,
    )
    # Geometry and radiometry from the same ping share one latent pose.
    assert len(backend.keyframes) == 2

    key_times, _, _ = backend._corrections()
    np.testing.assert_allclose(key_times, [0.0, 1.0])
    raw = np.array([[0.8, 0.0, -5.0]])
    corrected = backend.correct_trajectory(np.array([1.0]), raw)
    # Exact-time interpolation must use the later image node correction.
    np.testing.assert_allclose(
        corrected[0, :2], backend.graph.poses[-1, :2], atol=1.0e-9
    )


def test_pose_only_node_does_not_reset_geometry_keyframe_cadence() -> None:
    config = PoseGraphConfig(
        keyframe_distance_m=100.0,
        keyframe_max_interval_s=2.0,
        loop_min_separation_s=1_000.0,
        icp_min_inliers=30,
    )
    backend = AcousticPoseGraph(config)
    cloud = _feature_cloud()
    assert backend.add_if_needed(
        0.0,
        np.eye(3),
        np.array([0.0, 0.0, -5.0]),
        cloud,
        observation_id=0,
    )
    assert backend.add_intensity_observation(
        timestamp=1.5,
        observation_id=15,
        anchor_keyframe_index=0,
        prior_pose2=np.array([0.1, 0.0, 0.0]),
        posterior_rotation_world_body=np.eye(3),
        posterior_position_world=np.array([0.1, 0.0, -5.0]),
        photometric_pose2=np.array([0.1, 0.0, 0.0]),
        information_world=np.diag([16.0, 9.0, 400.0]),
        points_body=cloud[:24],
        intensities=None,
        appearance_descriptor=None,
        score=0.9,
    )
    assert not backend.keyframes[-1].loop_eligible
    # 2.1 s is measured from the previous geometry node at t=0, not from the
    # intervening image-only node at t=1.5.
    assert backend.add_if_needed(
        2.1,
        np.eye(3),
        np.array([0.2, 0.0, -5.0]),
        cloud,
        observation_id=21,
    )
    assert backend.keyframes[-1].geometry_sequence_index == 1


def test_unexported_local_image_update_is_absorbed_once_into_odometry() -> None:
    backend = AcousticPoseGraph(
        PoseGraphConfig(
            keyframe_distance_m=0.0,
            loop_min_separation_s=1_000.0,
            icp_min_inliers=30,
        )
    )
    cloud = _feature_cloud()
    assert backend.add_if_needed(
        0.0,
        np.eye(3),
        np.array([0.0, 0.0, -5.0]),
        cloud,
        observation_id=0,
    )
    assert backend.add_if_needed(
        1.0,
        np.eye(3),
        np.array([1.0, 0.0, -5.0]),
        cloud,
        observation_id=10,
    )
    assert backend.absorb_latest_local_posterior(
        timestamp=1.0,
        observation_id=10,
        posterior_rotation_world_body=np.eye(3),
        posterior_position_world=np.array([0.8, 0.1, -5.0]),
    )
    incoming = next(
        edge
        for edge in backend.graph.edges
        if edge.kind == "odometry" and edge.target == 1
    )
    np.testing.assert_allclose(incoming.measurement, [0.8, 0.1, 0.0])
    np.testing.assert_allclose(backend.keyframes[-1].raw_pose2, [0.8, 0.1, 0.0])
    np.testing.assert_allclose(backend.graph.poses[-1], [0.8, 0.1, 0.0])
    assert backend.diagnostics().intensity_edges == 0


def test_inconsistent_intensity_factor_rejects_without_mutating_target() -> None:
    backend = AcousticPoseGraph(
        PoseGraphConfig(
            keyframe_distance_m=0.0,
            loop_min_separation_s=1_000.0,
            icp_min_inliers=30,
            intensity_factor_min_xy_std_m=1.0,
            intensity_factor_prefit_chi2_rank2=4.0,
            intensity_factor_prefit_chi2_rank3=4.0,
            intensity_factor_optimize_interval=100,
        )
    )
    cloud = _feature_cloud()
    assert backend.add_if_needed(
        0.0,
        np.eye(3),
        np.array([0.0, 0.0, -5.0]),
        cloud,
        observation_id=0,
    )
    assert backend.add_if_needed(
        1.0,
        np.eye(3),
        np.array([1.0, 0.0, -5.0]),
        cloud,
        observation_id=10,
    )
    raw_before = backend.keyframes[-1].raw_pose2.copy()
    poses_before = backend.graph.poses.copy()
    edge_count_before = len(backend.graph.edges)

    assert not backend.add_intensity_observation(
        timestamp=1.0,
        observation_id=10,
        anchor_keyframe_index=0,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        posterior_rotation_world_body=np.eye(3),
        posterior_position_world=np.array([7.0, 0.4, -5.0]),
        photometric_pose2=np.array([4.0, 0.0, 0.0]),
        information_world=np.diag([100.0, 100.0, 100.0]),
        points_body=cloud,
        intensities=None,
        appearance_descriptor=None,
        score=0.99,
    )

    np.testing.assert_allclose(backend.keyframes[-1].raw_pose2, raw_before)
    np.testing.assert_allclose(backend.graph.poses, poses_before)
    assert len(backend.keyframes) == 2
    assert len(backend.graph.edges) == edge_count_before
    diagnostics = backend.diagnostics()
    assert diagnostics.intensity_edges == 0
    assert diagnostics.rejected_intensity_factors == 1
    assert diagnostics.intensity_factor_rejection_reasons == {
        "prefit_inconsistent": 1
    }
    assert diagnostics.last_intensity_factor_rejection_reason == (
        "prefit_inconsistent"
    )
    assert diagnostics.last_intensity_prefit_performed
    assert diagnostics.last_intensity_prefit_dof == 3
    assert diagnostics.last_intensity_prefit_chi2 == pytest.approx(9.0)


def test_consistency_gate_uses_globally_corrected_target_pose() -> None:
    backend = AcousticPoseGraph(
        PoseGraphConfig(
            keyframe_distance_m=0.0,
            loop_min_separation_s=1_000.0,
            icp_min_inliers=30,
            intensity_factor_min_xy_std_m=1.0,
            intensity_factor_prefit_chi2_rank2=4.0,
            intensity_factor_prefit_chi2_rank3=4.0,
            intensity_factor_optimize_interval=100,
        )
    )
    cloud = _feature_cloud()
    assert backend.add_if_needed(
        0.0,
        np.eye(3),
        np.array([0.0, 0.0, -5.0]),
        cloud,
        observation_id=0,
    )
    assert backend.add_if_needed(
        10.0,
        np.eye(3),
        np.array([10.0, 0.0, -5.0]),
        cloud,
        observation_id=100,
    )
    # A previously verified loop has moved the target close to the old
    # anchor. Comparing against raw x=10 would incorrectly reject this match.
    backend.graph.poses[1] = np.array([0.6, -0.2, 0.03])

    assert backend.add_intensity_observation(
        timestamp=10.0,
        observation_id=100,
        anchor_keyframe_index=0,
        prior_pose2=np.array([10.0, 0.0, 0.0]),
        posterior_rotation_world_body=np.eye(3),
        posterior_position_world=np.array([10.0, 0.0, -5.0]),
        photometric_pose2=np.array([0.85, -0.2, 0.03]),
        information_world=np.diag([100.0, 100.0, 100.0]),
        points_body=cloud,
        intensities=None,
        appearance_descriptor=None,
        score=0.99,
    )
    assert len(backend.keyframes) == 2
    factor = next(edge for edge in backend.graph.edges if edge.kind == "intensity")
    assert (factor.source, factor.target) == (0, 1)
    np.testing.assert_allclose(factor.measurement, [0.85, -0.2, 0.03])
    diagnostics = backend.diagnostics()
    assert diagnostics.intensity_edges == 1
    assert diagnostics.last_intensity_prefit_chi2 == pytest.approx(0.0625)


def test_rejected_factor_does_not_create_pose_only_node() -> None:
    backend = AcousticPoseGraph(
        PoseGraphConfig(
            keyframe_distance_m=0.0,
            loop_min_separation_s=1_000.0,
            icp_min_inliers=30,
            intensity_factor_min_xy_std_m=1.0,
            intensity_factor_prefit_chi2_rank2=4.0,
            intensity_factor_prefit_chi2_rank3=4.0,
            intensity_factor_optimize_interval=100,
        )
    )
    cloud = _feature_cloud()
    assert backend.add_if_needed(
        0.0,
        np.eye(3),
        np.array([0.0, 0.0, -5.0]),
        cloud,
        observation_id=0,
    )
    poses_before = backend.graph.poses.copy()
    edges_before = tuple(backend.graph.edges)

    assert not backend.add_intensity_observation(
        timestamp=1.0,
        observation_id=10,
        anchor_keyframe_index=0,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        posterior_rotation_world_body=np.eye(3),
        posterior_position_world=np.array([1.0, 0.0, -5.0]),
        photometric_pose2=np.array([4.0, 0.0, 0.0]),
        information_world=np.diag([100.0, 100.0, 100.0]),
        points_body=cloud,
        intensities=None,
        appearance_descriptor=None,
        score=0.99,
    )
    assert len(backend.keyframes) == 1
    assert len(backend.graph.edges) == len(edges_before)
    assert all(
        current is previous
        for current, previous in zip(backend.graph.edges, edges_before)
    )
    np.testing.assert_allclose(backend.graph.poses, poses_before)
    diagnostics = backend.diagnostics()
    assert diagnostics.odometry_edges == 0
    assert diagnostics.intensity_edges == 0
    assert diagnostics.optimization_runs == 0
    assert diagnostics.rejected_intensity_factors == 1


def test_indefinite_image_information_is_rejected() -> None:
    config = PoseGraphConfig(
        keyframe_distance_m=0.0,
        keyframe_max_interval_s=0.1,
        loop_min_separation_s=1_000.0,
        icp_min_inliers=30,
    )
    backend = AcousticPoseGraph(config)
    cloud = _feature_cloud()
    appearance = sidescan_descriptor(np.ones(80), np.ones(80))
    backend.add_if_needed(
        0.0,
        np.eye(3),
        np.array([0.0, 0.0, -5.0]),
        cloud,
        appearance_descriptor=appearance,
        observation_id=0,
    )

    assert not backend.add_intensity_observation(
        timestamp=1.0,
        observation_id=10,
        anchor_keyframe_index=0,
        prior_pose2=np.array([1.0, 0.0, 0.0]),
        posterior_rotation_world_body=np.eye(3),
        posterior_position_world=np.array([1.0, 0.0, -5.0]),
        photometric_pose2=np.array([0.8, 0.0, 0.0]),
        information_world=np.diag([25.0, 16.0, -1.0e6]),
        points_body=cloud,
        intensities=None,
        appearance_descriptor=appearance,
        score=0.9,
    )
    assert backend.diagnostics().intensity_edges == 0


def test_world_yaw_intensity_jacobian_with_nonzero_roll_and_pitch() -> None:
    intensity_map = AcousticIntensityMap(
        voxel_size=1.0,
        min_cell_points=1,
        min_neighbours=5,
        min_gradient=1.0e-6,
    )
    x, y = np.meshgrid(np.linspace(-20.0, 20.0, 81), np.linspace(-20.0, 20.0, 81))
    map_points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -8.0)))
    map_values = 0.5 + 0.012 * x.ravel() - 0.019 * y.ravel()
    yaw = 0.4
    intensity_map.add_points(
        map_points,
        map_values,
        heading_rad=yaw,
        observation_id=0,
        anchor_id=0,
    )
    estimator = AcousticESIKF(ESIKFConfig(intensity_planar_only=True))
    rotation = rotation_from_rpy(0.25, -0.18, yaw)
    estimator.initialize(0.0, rotation=rotation)
    points_sensor = np.array([[2.0, 4.0, -8.0]])

    linearized = estimator._intensity_linearization(
        estimator.state,
        points_sensor,
        np.array([0.5]),
        intensity_map,
        anchor_id=0,
    )
    analytic = float(linearized[3][0, 2])
    epsilon = 1.0e-6
    predictions = []
    for delta in (-epsilon, epsilon):
        perturbed = estimator.state.copy()
        perturbed.rotation = rotation_from_rpy(0.0, 0.0, delta) @ rotation
        predictions.append(
            float(
                estimator._intensity_linearization(
                    perturbed,
                    points_sensor,
                    np.array([0.5]),
                    intensity_map,
                    anchor_id=0,
                )[1][0]
            )
        )
    finite_difference = (predictions[1] - predictions[0]) / (2.0 * epsilon)
    assert analytic == pytest.approx(finite_difference, abs=2.0e-4)


def test_gain_offset_elimination_cannot_add_pose_information() -> None:
    rng = np.random.default_rng(27)
    predicted = np.linspace(0.1, 0.9, 60)
    measured = 1.17 * predicted - 0.08 + 0.02 * np.sin(np.arange(60))
    outliers = np.arange(0, measured.size, 7)
    measured[outliers] += 1.5
    state_jacobian = rng.normal(size=(60, 15))
    pose_jacobian = rng.normal(size=(60, 3))
    variance = np.linspace(0.02, 0.06, 60)
    (
        residual,
        _projected_state,
        projected_pose,
        gain,
        _offset,
        weights,
    ) = AcousticESIKF._affine_intensity_residual(
        measured,
        predicted,
        state_jacobian,
        pose_jacobian,
        variance,
    )
    raw_pose = gain * pose_jacobian
    raw_information = raw_pose.T @ (weights[:, None] * raw_pose)
    projected_information = projected_pose.T @ (
        weights[:, None] * projected_pose
    )

    assert np.isfinite(residual).all()
    assert np.min(
        np.linalg.eigvalsh(raw_information - projected_information)
    ) >= -1.0e-8
    assert np.trace(projected_information) < np.trace(raw_information)
    assert np.mean(weights[outliers]) < np.mean(
        np.delete(weights, outliers)
    )


def test_each_ping_gets_its_own_affine_radiometric_nuisance() -> None:
    predicted_line = np.linspace(0.1, 0.9, 40)
    predicted = np.tile(predicted_line, 2)
    groups = np.repeat([0, 1], predicted_line.size)
    measured = np.concatenate(
        (
            0.72 * predicted_line + 0.11,
            1.75 * predicted_line - 0.19,
        )
    )
    jacobian = np.zeros((predicted.size, 15), dtype=np.float64)
    pose_jacobian = np.column_stack(
        (predicted, predicted**2, np.sin(predicted * 3.0))
    )
    variance = np.full(predicted.size, 0.02)

    single_residual = AcousticESIKF._affine_intensity_residual(
        measured,
        predicted,
        jacobian,
        pose_jacobian,
        variance,
    )[0]
    grouped_residual = AcousticESIKF._affine_intensity_residual(
        measured,
        predicted,
        jacobian,
        pose_jacobian,
        variance,
        affine_group_ids=groups,
    )[0]

    assert np.sqrt(np.mean(grouped_residual**2)) < 1.0e-7
    assert np.sqrt(np.mean(single_residual**2)) > 0.1


def test_pose_graph_seed_is_expressed_in_reference_anchor_coordinates() -> None:
    backend = AcousticPoseGraph(
        PoseGraphConfig(
            keyframe_distance_m=0.0,
            loop_min_separation_s=1_000.0,
            icp_min_inliers=30,
            intensity_seed_search_radius_m=2.0,
        )
    )
    cloud = _feature_cloud()
    assert backend.add_if_needed(
        0.0,
        np.eye(3),
        np.array([0.0, 0.0, -5.0]),
        cloud,
        observation_id=0,
    )
    assert backend.add_if_needed(
        10.0,
        np.eye(3),
        np.array([10.0, 0.0, -5.0]),
        cloud,
        observation_id=100,
    )
    # Pretend a verified loop has globally placed the drifting latest node
    # back near anchor zero. The seed must be mapped back into anchor zero's
    # raw coordinate system, where its intensity cells are stored.
    backend.graph.poses[1] = np.array([0.6, -0.2, 0.03])
    candidates = backend.intensity_seed_candidates(
        rotation_world_body=np.eye(3),
        position_world=np.array([10.0, 0.0, -5.0]),
        anchor_ids=(0,),
        maximum_observation_id=10,
    )
    assert len(candidates) == 1
    anchor_id, seed = candidates[0]
    assert anchor_id == 0
    np.testing.assert_allclose(seed, [0.6, -0.2, 0.03], atol=1.0e-9)
