#!/usr/bin/env python3
"""Replay an identical saved factor stream in the frozen native GTSAM backend.

This isolates solver portability from frontend/terrain-observation generation.
It is a diagnostic, not a replacement for raw-sensor HIL or navigation scoring.
No truth input or estimator parameter adjustment is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


def replay_factors(stream, graph):
    if stream["schema"] != "oceansim.planar_pose_graph_factor_stream.v1":
        raise ValueError("unsupported factor stream")
    if len(stream["pose_initializations"]) != len(stream["pose_timestamps_s"]):
        raise ValueError("missing pose timestamps")
    snapshots = []
    for boundary in stream["optimization_boundaries"]:
        counts = (boundary["poses"], boundary["edges"], boundary["priors"])
        current = (len(graph.poses), len(graph.edges), len(graph.priors))
        available = (len(stream["pose_initializations"]), len(stream["edges"]), len(stream["priors"]))
        if any(not isinstance(n, int) or n < lo or n > hi for n, lo, hi in zip(counts, current, available)):
            raise ValueError("invalid/nonmonotonic optimization boundary")
        for i in range(current[0], counts[0]):
            graph.add_pose(np.asarray(stream["pose_initializations"][i], dtype=np.float64),
                           timestamp=stream["pose_timestamps_s"][i])
        for edge in stream["edges"][current[1]:counts[1]]:
            graph.add_edge(edge["source"], edge["target"], np.asarray(edge["measurement"]),
                           np.asarray(edge["information"]), kind=edge["kind"], score=edge["score"])
        for prior in stream["priors"][current[2]:counts[2]]:
            graph.add_prior(prior["target"], np.asarray(prior["measurement"]),
                            np.asarray(prior["information"]), kind=prior["kind"],
                            score=prior["score"], robust=prior["robust"],
                            robust_kernel=prior["robust_kernel"], robust_delta=prior["robust_delta"])
        snapshots.append(graph.optimize().tolist())
    if not snapshots or (len(graph.poses), len(graph.edges), len(graph.priors)) != available:
        raise ValueError("incomplete saved optimization stream")
    original = np.asarray(stream["final_online_poses"], dtype=np.float64)
    if original.shape != graph.poses.shape:
        raise ValueError("final pose count mismatch")
    delta = graph.poses - original
    delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
    if not np.isfinite(delta).all():
        raise ValueError("nonfinite replay result")
    return dict(optimization_snapshots=snapshots, final_poses=graph.poses.tolist(),
                maximum_original_final_xy_difference_m=float(np.max(np.linalg.norm(delta[:, :2], axis=1))),
                maximum_original_final_yaw_difference_rad=float(np.max(np.abs(delta[:, 2]))),
                backend=graph.backend_name, fixed_lag_updates=graph.fixed_lag_updates,
                isam_updates=graph.isam_updates, fixed_lag_active_variables=graph.fixed_lag_active_variables,
                fixed_lag_marginalized_variables=graph.fixed_lag_marginalized_variables)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--factors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("fresh audit output required")
    sys.path.insert(0, str(args.bundle / "isaacsim"))
    from oceansim.slam.pose_graph import GTSAMPlanarPoseGraph, PoseGraphConfig
    payload = json.loads(args.factors.read_text())
    stream = payload.get("graph_factors", payload)
    graph = GTSAMPlanarPoseGraph(PoseGraphConfig(**stream["objective_configuration"]))
    report = replay_factors(stream, graph)
    report.update(status="completed", kind="identical_factors_solver_portability_diagnostic",
                  numpy_version=np.__version__, architecture=platform.machine(),
                  factor_input_sha256=hashlib.sha256(args.factors.read_bytes()).hexdigest(),
                  frozen_graph_source_sha256=hashlib.sha256(
                      (args.bundle / "isaacsim/oceansim/slam/pose_graph.py").read_bytes()).hexdigest(),
                  no_truth_input=True, full_route_acceptance=False)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"optimization_snapshots", "final_poses"}}, indent=2))


if __name__ == "__main__":
    main()
