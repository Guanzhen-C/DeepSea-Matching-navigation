import numpy as np
import pytest

from standalone.analyze_factor_stream_relative_odometry import (
    reconstruct_local_odometry,
)
from standalone.compare_pose_graph_backend_runs import _upper_bound_check


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
