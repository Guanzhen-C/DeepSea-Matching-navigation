"""Small synthetic contracts, not hardware or full-route acceptance data."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from oceansim.slam.pose_graph import GTSAMPlanarPoseGraph, PoseGraphConfig
from standalone.audit_hil_graph import replay_factors

pytest.importorskip("gtsam")


def synthetic_stream():
    graph = GTSAMPlanarPoseGraph(PoseGraphConfig(fixed_lag_seconds=2.))
    information = np.diag([10., 10., 50.])
    for i in range(6):
        graph.add_pose(np.array([float(i), 0., .02]), timestamp=float(i))
        if i:
            graph.add_edge(i - 1, i, np.array([1., 0., 0.]), information, kind="odometry")
            if i == 3:
                graph.add_prior(i, np.array([3., .05, .01]), information, robust=True)
            graph.optimize()
    return graph.factor_stream_payload()


def fresh_graph(stream):
    return GTSAMPlanarPoseGraph(PoseGraphConfig(**stream["objective_configuration"]))


def test_identical_saved_factors_replay_with_fixed_lag_marginalization():
    stream = synthetic_stream()
    result = replay_factors(stream, fresh_graph(stream))
    assert result["maximum_original_final_xy_difference_m"] < 1e-10
    assert result["maximum_original_final_yaw_difference_rad"] < 1e-10
    assert result["fixed_lag_marginalized_variables"] > 0
    assert len(result["optimization_snapshots"]) == 5


@pytest.mark.parametrize("corruption", ["schema", "timestamp", "boundary", "missing_tail"])
def test_rejects_incomplete_or_invalid_factor_stream(corruption):
    stream = copy.deepcopy(synthetic_stream())
    if corruption == "schema":
        stream["schema"] = "unknown"
    elif corruption == "timestamp":
        stream["pose_timestamps_s"].pop()
    elif corruption == "boundary":
        stream["optimization_boundaries"][1]["poses"] = 1
    else:
        stream["optimization_boundaries"].pop()
    with pytest.raises(ValueError):
        replay_factors(stream, fresh_graph(stream))
