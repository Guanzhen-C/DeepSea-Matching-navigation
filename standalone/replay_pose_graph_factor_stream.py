#!/usr/bin/env python3
"""Replay one recorded OceanSim factor stream through either graph solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaacsim"))

from oceansim.slam import (  # noqa: E402
    GTSAMPlanarPoseGraph,
    PlanarPoseGraph,
    PoseGraphConfig,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("factor_stream", type=Path)
    parser.add_argument(
        "--backend",
        choices=("numpy-batch", "gtsam-full-window", "gtsam-fixed-lag"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeat-optimize",
        type=int,
        default=1,
        help="additional no-new-factor solver update calls",
    )
    return parser.parse_args()


def _config(payload: dict[str, Any], backend: str) -> PoseGraphConfig:
    recorded = payload.get("objective_configuration", {})
    timestamps = np.asarray(payload["pose_timestamps_s"], dtype=np.float64)
    recorded_lag = float(recorded.get("fixed_lag_seconds", 180.0))
    if backend == "gtsam-full-window" and timestamps.size:
        recorded_lag = max(
            recorded_lag,
            float(timestamps[-1] - timestamps[0]) + 1.0,
        )
    return PoseGraphConfig(
        optimizer_backend=(
            "numpy_batch" if backend == "numpy-batch" else "gtsam_fixed_lag_isam2"
        ),
        optimizer_iterations=int(recorded.get("optimizer_iterations", 12)),
        terrain_prior_robust_kernel=str(
            recorded.get("terrain_prior_robust_kernel", "huber")
        ),
        robust_terrain_prior_delta=float(
            recorded.get("robust_terrain_prior_delta", 2.5)
        ),
        relocalization_prior_robust_kernel=str(
            recorded.get("relocalization_prior_robust_kernel", "cauchy")
        ),
        robust_relocalization_prior_delta=float(
            recorded.get("robust_relocalization_prior_delta", 2.5)
        ),
        robust_loop_delta=float(recorded.get("robust_loop_delta", 3.0)),
        robust_intensity_delta=float(
            recorded.get("robust_intensity_delta", 2.5)
        ),
        fixed_lag_seconds=recorded_lag,
        isam_relinearize_threshold=float(
            recorded.get("isam_relinearize_threshold", 0.01)
        ),
        isam_relinearize_skip=int(recorded.get("isam_relinearize_skip", 1)),
    )


def _build(payload: dict[str, Any], backend: str):
    config = _config(payload, backend)
    graph = (
        PlanarPoseGraph(config)
        if backend == "numpy-batch"
        else GTSAMPlanarPoseGraph(config)
    )
    initial = np.asarray(payload["pose_initializations"], dtype=np.float64)
    timestamps = np.asarray(payload["pose_timestamps_s"], dtype=np.float64)
    if initial.shape != (timestamps.size, 3):
        raise ValueError("pose initializations and timestamps disagree")
    gauge = payload.get("gauge_pose")
    if gauge is not None and initial.shape[0]:
        initial[0] = np.asarray(gauge, dtype=np.float64).reshape(3)
    for pose, timestamp in zip(initial, timestamps, strict=True):
        graph.add_pose(pose, timestamp=float(timestamp))
    for item in payload["edges"]:
        graph.add_edge(
            int(item["source"]),
            int(item["target"]),
            np.asarray(item["measurement"], dtype=np.float64),
            np.asarray(item["information"], dtype=np.float64),
            kind=str(item["kind"]),
            score=float(item.get("score", 0.0)),
        )
    for item in payload["priors"]:
        graph.add_prior(
            int(item["target"]),
            np.asarray(item["measurement"], dtype=np.float64),
            np.asarray(item["information"], dtype=np.float64),
            kind=str(item.get("kind", "terrain_prior")),
            score=float(item.get("score", 0.0)),
            robust=bool(item.get("robust", True)),
            robust_kernel=item.get("robust_kernel"),
            robust_delta=item.get("robust_delta"),
        )
    return graph


def main() -> int:
    args = _arguments()
    if args.repeat_optimize < 0:
        raise ValueError("repeat-optimize must be non-negative")
    payload = json.loads(args.factor_stream.read_text(encoding="utf-8"))
    if payload.get("schema") != "oceansim.planar_pose_graph_factor_stream.v1":
        raise ValueError("unsupported factor-stream schema")
    graph = _build(payload, args.backend)
    graph.optimize()
    for _ in range(args.repeat_optimize):
        graph.optimize()
    logged = np.asarray(payload.get("final_online_poses", []), dtype=np.float64)
    maximum_logged_delta = None
    if logged.shape == graph.poses.shape and logged.size:
        difference = graph.poses - logged
        difference[:, 2] = (difference[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
        maximum_logged_delta = float(np.max(np.abs(difference)))
    result = {
        "schema": "oceansim.planar_pose_graph_replay.v1",
        "factor_stream": str(args.factor_stream.resolve()),
        "backend": graph.backend_name,
        "poses": int(graph.poses.shape[0]),
        "edges": len(graph.edges),
        "priors": len(graph.priors),
        "optimization_runs": graph.optimization_runs,
        "initial_cost": float(graph.initial_cost),
        "final_cost": float(graph.final_cost),
        "finite": bool(np.isfinite(graph.poses).all()),
        "maximum_absolute_delta_from_logged_online_pose": maximum_logged_delta,
        "fixed_lag_active_variables": graph.fixed_lag_active_variables,
        "fixed_lag_marginalized_variables": graph.fixed_lag_marginalized_variables,
        "fixed_lag_global_only_factors": int(
            getattr(graph, "global_only_factor_count", 0)
        ),
        "final_poses": graph.poses.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
