import numpy as np
import pytest

from standalone.analyze_factor_stream_relative_odometry import (
    reconstruct_local_odometry,
)
from standalone.compare_pose_graph_backend_runs import _upper_bound_check
from standalone.compare_pose_graph_factor_replays import compare_replays


def _factor_stream(edges):
    return {
        "pose_timestamps_s": [0.0, 1.0, 2.0],
        "edges": edges,
    }


def test_reconstruct_local_odometry_composes_body_frame_increments() -> None:
    stream = _factor_stream(
        [
            {
                "source": 0,
                "target": 1,
                "kind": "odometry",
                "measurement": [2.0, 0.0, np.pi / 2.0],
            },
            {
                "source": 1,
                "target": 2,
                "kind": "odometry",
                "measurement": [1.0, 0.0, 0.0],
            },
            {
                "source": 0,
                "target": 2,
                "kind": "loop",
                "measurement": [99.0, 99.0, 1.0],
            },
        ]
    )

    poses = reconstruct_local_odometry(stream)

    np.testing.assert_allclose(
        poses,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, np.pi / 2.0],
                [2.0, 1.0, np.pi / 2.0],
            ]
        ),
        atol=1.0e-12,
    )


def test_reconstruct_local_odometry_rejects_an_incomplete_chain() -> None:
    stream = _factor_stream(
        [
            {
                "source": 0,
                "target": 1,
                "kind": "odometry",
                "measurement": [1.0, 0.0, 0.0],
            }
        ]
    )

    with pytest.raises(ValueError, match="complete pose chain"):
        reconstruct_local_odometry(stream)


def test_comparator_reports_unavailable_metric_instead_of_crashing() -> None:
    result = _upper_bound_check(
        "short_run_metric",
        None,
        None,
        relative_allowance=0.25,
        absolute_allowance=1.0,
    )

    assert not result["passed"]
    assert result["failure_limit"] is None
    assert result["reason"] == "metric_unavailable"


def _replay_payload(poses, *, active: int, marginalized: int, backend: str):
    return {
        "backend": backend,
        "poses": len(poses),
        "edges": 1,
        "priors": 1,
        "optimization_runs": 2,
        "initial_cost": 10.0,
        "final_cost": 10.0,
        "finite": True,
        "maximum_absolute_delta_from_logged_online_pose": 0.0,
        "fixed_lag_active_variables": active,
        "fixed_lag_marginalized_variables": marginalized,
        "fixed_lag_global_only_factors": 0,
        "final_poses": poses,
    }


def test_factor_replay_comparator_proves_full_window_equivalence() -> None:
    numpy_replay = _replay_payload(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        active=0,
        marginalized=0,
        backend="numpy_batch",
    )
    full_replay = _replay_payload(
        [[0.0, 0.0, 0.0], [1.001, 0.0, np.deg2rad(0.001)]],
        active=2,
        marginalized=0,
        backend="gtsam_isam2_incremental_fixed_lag",
    )
    lag_replay = _replay_payload(
        [[0.0, 0.0, 0.0], [1.01, 0.0, np.deg2rad(0.01)]],
        active=1,
        marginalized=1,
        backend="gtsam_isam2_incremental_fixed_lag",
    )
    factors = {
        "final_online_poses": lag_replay["final_poses"],
        "full_history_diagnostic_poses": full_replay["final_poses"],
    }

    report = compare_replays(
        numpy_replay, full_replay, lag_replay, factors
    )

    assert report["overall_verdict"] == "pass"
    assert all(report["checks"].values())
    assert report["pose_differences"][
        "gtsam_full_window_vs_numpy_batch"
    ]["translation_max_m"] == pytest.approx(0.001)


def test_factor_replay_comparator_rejects_solver_disagreement() -> None:
    numpy_replay = _replay_payload(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        active=0,
        marginalized=0,
        backend="numpy_batch",
    )
    full_replay = _replay_payload(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        active=2,
        marginalized=0,
        backend="gtsam_isam2_incremental_fixed_lag",
    )
    lag_replay = _replay_payload(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        active=1,
        marginalized=1,
        backend="gtsam_isam2_incremental_fixed_lag",
    )

    report = compare_replays(
        numpy_replay, full_replay, lag_replay, {}
    )

    assert report["overall_verdict"] == "fail"
    assert not report["checks"]["full_window_translation_max_le_0p02m"]
