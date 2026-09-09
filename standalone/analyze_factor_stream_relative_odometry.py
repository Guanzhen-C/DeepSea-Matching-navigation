#!/usr/bin/env python3
"""Evaluate local planar odometry without penalising a global pose gauge.

The factor stream records every sequential odometry measurement before the
global graph changes it.  Composing only those factors therefore reconstructs
the local front-end trajectory independently of terrain priors, loop closures,
and the selected graph backend.
"""

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
    MeasuredBathymetryGrid,
    WaypointTerrainFollowingSurvey,
    se2_relative_pose_error_by_distance,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("factor_stream", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--interval-m",
        type=float,
        action="append",
        dest="intervals_m",
        help="truth-route interval; repeat for more than one (default: 100, 1000, 10000)",
    )
    return parser.parse_args()


def _wrap_angle(value: np.ndarray | float) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) + np.pi) % (
        2.0 * np.pi
    ) - np.pi


def _compose_pose2(first: np.ndarray, relative: np.ndarray) -> np.ndarray:
    cosine = float(np.cos(first[2]))
    sine = float(np.sin(first[2]))
    return np.array(
        [
            first[0] + cosine * relative[0] - sine * relative[1],
            first[1] + sine * relative[0] + cosine * relative[1],
            float(_wrap_angle(first[2] + relative[2])),
        ],
        dtype=np.float64,
    )


def _relative_pose2(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = target[:2] - source[:2]
    cosine = float(np.cos(source[2]))
    sine = float(np.sin(source[2]))
    return np.array(
        [
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
            float(_wrap_angle(target[2] - source[2])),
        ],
        dtype=np.float64,
    )


def reconstruct_local_odometry(payload: dict[str, Any]) -> np.ndarray:
    """Compose the unique sequential odometry chain in a zero pose gauge."""

    timestamps = np.asarray(payload["pose_timestamps_s"], dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("factor stream must contain pose timestamps")
    measurements: dict[int, np.ndarray] = {}
    for item in payload["edges"]:
        if item["kind"] != "odometry":
            continue
        source = int(item["source"])
        target = int(item["target"])
        if target != source + 1:
            raise ValueError("odometry factors must connect sequential poses")
        if target in measurements:
            raise ValueError("each pose must have one incoming odometry factor")
        measurement = np.asarray(
            item["measurement"], dtype=np.float64
        ).reshape(3)
        if not np.isfinite(measurement).all():
            raise ValueError("odometry measurements must be finite")
        measurements[target] = measurement
    expected = set(range(1, timestamps.size))
    if set(measurements) != expected:
        raise ValueError("odometry factors do not form one complete pose chain")

    poses = np.zeros((timestamps.size, 3), dtype=np.float64)
    for target in range(1, timestamps.size):
        poses[target] = _compose_pose2(
            poses[target - 1], measurements[target]
        )
    return poses


def _resolve_route_path(metrics: dict[str, Any]) -> Path:
    recorded = Path(str(metrics["survey_route_file"])).expanduser()
    if recorded.is_file():
        return recorded.resolve()
    fallback = ROOT / "data" / recorded.parent.name / recorded.name
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"survey route not found: {recorded}")


def _truth_planar_poses(
    timestamps: np.ndarray, metrics: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    if metrics.get("survey_pattern") != "full-map-route":
        raise ValueError("only a full-map-route survey is currently supported")
    route_path = _resolve_route_path(metrics)
    route = json.loads(route_path.read_text(encoding="utf-8"))
    if route.get("schema") != "oceansim.measured-terrain-route/v1":
        raise ValueError("unsupported survey-route schema")
    world = MeasuredBathymetryGrid.from_path(route_path.parent, mmap_mode="r")
    route_definition = route.get("route_definition", {})
    speed_mps = float(metrics["survey_speed_mps"])
    survey = WaypointTerrainFollowingSurvey(
        world,
        np.asarray(route["control_points_xy_m"], dtype=np.float64),
        speed_mps=speed_mps,
        altitude_m=float(metrics["survey_altitude_m"]),
        smoothing_iterations=int(
            route_definition.get("smoothing_iterations", 4)
        ),
        maximum_pitch_deg=(
            None
            if metrics.get("survey_maximum_pitch_deg") is None
            else float(metrics["survey_maximum_pitch_deg"])
        ),
    )
    poses = np.empty((timestamps.size, 3), dtype=np.float64)
    for index, timestamp in enumerate(timestamps):
        state = survey.state(float(timestamp))
        poses[index, :2] = state.position[:2]
        poses[index, 2] = np.arctan2(
            state.rotation[1, 0], state.rotation[0, 0]
        )
    distance = (timestamps - timestamps[0]) * speed_mps
    return poses, distance


def _distribution(values: np.ndarray, *, rmse: bool = False) -> dict[str, float]:
    result = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }
    if rmse:
        result["rmse"] = float(np.sqrt(np.mean(values**2)))
    return result


def _incremental_metrics(
    local_poses: np.ndarray,
    truth_poses: np.ndarray,
    cumulative_distance_m: np.ndarray,
) -> dict[str, Any]:
    local_relative = np.asarray(
        [
            _relative_pose2(local_poses[index - 1], local_poses[index])
            for index in range(1, local_poses.shape[0])
        ]
    )
    truth_relative = np.asarray(
        [
            _relative_pose2(truth_poses[index - 1], truth_poses[index])
            for index in range(1, truth_poses.shape[0])
        ]
    )
    translation_error = np.linalg.norm(
        local_relative[:, :2] - truth_relative[:, :2], axis=1
    )
    yaw_error_deg = np.abs(
        np.rad2deg(_wrap_angle(local_relative[:, 2] - truth_relative[:, 2]))
    )
    spacing = np.diff(cumulative_distance_m)
    return {
        "samples": int(translation_error.size),
        "truth_route_spacing_m": _distribution(spacing),
        "translation_error_m": _distribution(
            translation_error, rmse=True
        ),
        "absolute_yaw_error_deg": _distribution(yaw_error_deg, rmse=True),
    }


def analyze(
    factor_payload: dict[str, Any],
    metrics: dict[str, Any],
    intervals_m: list[float],
) -> dict[str, Any]:
    if factor_payload.get("schema") != (
        "oceansim.planar_pose_graph_factor_stream.v1"
    ):
        raise ValueError("unsupported factor-stream schema")
    timestamps = np.asarray(
        factor_payload["pose_timestamps_s"], dtype=np.float64
    )
    if (
        timestamps.ndim != 1
        or timestamps.size < 2
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) < 0.0)
    ):
        raise ValueError("pose timestamps must be finite, monotonic, and nonempty")
    local_poses = reconstruct_local_odometry(factor_payload)
    truth_poses, cumulative_distance_m = _truth_planar_poses(
        timestamps, metrics
    )
    return {
        "schema": "oceansim.relative_odometry_evaluation.v1",
        "definition": (
            "only sequential local-front-end odometry factors are composed; "
            "terrain priors, loops, graph optimization, and the constant global "
            "position/yaw gauge are excluded"
        ),
        "nodes": int(timestamps.size),
        "start_timestamp_s": float(timestamps[0]),
        "end_timestamp_s": float(timestamps[-1]),
        "evaluated_truth_route_distance_m": float(cumulative_distance_m[-1]),
        "incremental_keyframe_odometry": _incremental_metrics(
            local_poses, truth_poses, cumulative_distance_m
        ),
        "se2_rpe_by_truth_route_distance": {
            f"{interval_m:g}m": se2_relative_pose_error_by_distance(
                local_poses,
                truth_poses,
                cumulative_distance_m,
                interval_m,
            )
            for interval_m in intervals_m
        },
    }


def main() -> int:
    args = _arguments()
    intervals_m = args.intervals_m or [100.0, 1000.0, 10000.0]
    if any(not np.isfinite(value) or value <= 0.0 for value in intervals_m):
        raise ValueError("all RPE intervals must be positive and finite")
    factor_payload = json.loads(
        args.factor_stream.read_text(encoding="utf-8")
    )
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = analyze(factor_payload, metrics, intervals_m)
    result["factor_stream"] = str(args.factor_stream.resolve())
    result["metrics"] = str(args.metrics.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
