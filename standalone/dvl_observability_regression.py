#!/usr/bin/env python3
"""Deterministic Monterey regression for the DVL pose-gauge update.

This benchmark intentionally runs only IMU, DVL, and pressure depth.  It
feeds each simulated measurement to two otherwise identical filters so the
comparison isolates the observability-constrained DVL correction from sonar
and global terrain-matching effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ISAACSIM_PACKAGE = ROOT / "isaacsim"
if str(ISAACSIM_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ISAACSIM_PACKAGE))

from oceansim.slam.esikf import AcousticESIKF, ESIKFConfig
from oceansim.slam.initialization import gauge_anchored_initial_covariance
from oceansim.slam.large_scale_sim import LargeScaleSensorSimulator
from oceansim.slam.math3d import rpy_from_rotation, so3_log
from oceansim.slam.real_bathymetry import (
    MeasuredBathymetryGrid,
    WaypointTerrainFollowingSurvey,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-distance-m", type=float, default=30_000.0)
    parser.add_argument("--segment-distance-m", type=float, default=4_500.0)
    parser.add_argument("--speed-mps", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--report-distance-m", type=float, default=250.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "demo/large_scale_slam/dvl_observability_regression.json",
    )
    args = parser.parse_args()
    if (
        args.start_distance_m < 0.0
        or args.segment_distance_m <= 0.0
        or args.speed_mps <= 0.0
        or args.report_distance_m <= 0.0
    ):
        parser.error("distances and speed must be positive")
    return args


def _yaw_error_deg(rotation: np.ndarray, truth_rotation: np.ndarray) -> float:
    yaw = float(rpy_from_rotation(rotation)[2])
    truth_yaw = float(rpy_from_rotation(truth_rotation)[2])
    return float(
        np.rad2deg(
            np.arctan2(np.sin(yaw - truth_yaw), np.cos(yaw - truth_yaw))
        )
    )


def main() -> None:
    args = _parse_args()
    terrain = MeasuredBathymetryGrid.from_path(
        ROOT / "data/monterey_canyon_full_5m"
    )
    route_metadata = json.loads(
        (
            ROOT
            / "data/monterey_canyon_full_5m/full_map_traversal_route.json"
        ).read_text()
    )
    survey = WaypointTerrainFollowingSurvey(
        terrain,
        np.asarray(route_metadata["control_points_xy_m"], dtype=np.float64),
        speed_mps=args.speed_mps,
        altitude_m=40.0,
        smoothing_iterations=4,
        maximum_pitch_deg=12.0,
    )
    time_offset_s = args.start_distance_m / args.speed_mps

    def truth_at(timestamp_s: float):
        return survey.state(float(timestamp_s) + time_offset_s)

    sensors = LargeScaleSensorSimulator(
        terrain, survey, max_range=250.0, seed=args.seed
    )
    truth0 = truth_at(0.0)
    # Consume the same deterministic startup window as the Isaac runner.
    for _ in range(32):
        sensors.imu(truth0)
        sensors.dvl(truth0)
        sensors.depth(truth0)

    gyro_bias_limit = np.deg2rad(1.0) / 3600.0
    covariance = gauge_anchored_initial_covariance(
        gyro_bias_std_rad_s=gyro_bias_limit / 2.0
    )
    estimators = {
        "unconstrained_dvl": AcousticESIKF(
            ESIKFConfig(
                gyro_bias_max_norm=gyro_bias_limit,
                dvl_preserve_pose_gauge=False,
            )
        ),
        "observable_dvl": AcousticESIKF(
            ESIKFConfig(
                gyro_bias_max_norm=gyro_bias_limit,
                dvl_preserve_pose_gauge=True,
            )
        ),
    }
    for estimator in estimators.values():
        estimator.initialize(
            0.0,
            rotation=truth0.rotation,
            position=truth0.position,
            velocity=truth0.velocity,
            covariance=covariance,
        )

    accumulators = {
        name: {
            "squared_position_error_sum": 0.0,
            "samples": 0,
            "maximum_position_error_m": 0.0,
            "maximum_dvl_position_jump_m": 0.0,
            "maximum_dvl_attitude_jump_deg": 0.0,
            "maximum_dvl_global_yaw_component_deg": 0.0,
            "final_position_error_m": 0.0,
            "final_yaw_error_deg": 0.0,
        }
        for name in estimators
    }
    checkpoints: list[dict[str, float | dict[str, float]]] = []
    imu_dt_s, dvl_dt_s, depth_dt_s = 0.02, 0.2, 0.1
    imu_index = dvl_index = depth_index = 0
    next_imu = imu_dt_s
    next_dvl = dvl_dt_s
    next_depth = depth_dt_s
    travelled_m = 0.0
    previous_truth_position = truth0.position.copy()
    next_report_m = args.report_distance_m
    end_time_s = args.segment_distance_m / args.speed_mps
    started = time.monotonic()

    while min(next_imu, next_dvl, next_depth) <= end_time_s + 1.0e-10:
        timestamp_s = min(next_imu, next_dvl, next_depth)
        truth = truth_at(timestamp_s)
        travelled_m += float(
            np.linalg.norm(truth.position - previous_truth_position)
        )
        previous_truth_position = truth.position.copy()

        if next_imu <= timestamp_s + 1.0e-10:
            gyro, specific_force = sensors.imu(truth)
            for estimator in estimators.values():
                estimator.propagate(timestamp_s, gyro, specific_force)
            imu_index += 1
            next_imu = (imu_index + 1) * imu_dt_s

        if next_dvl <= timestamp_s + 1.0e-10:
            dvl = sensors.dvl(truth)
            for name, estimator in estimators.items():
                position_before = estimator.state.position.copy()
                rotation_before = estimator.state.rotation.copy()
                estimator.update_dvl(dvl)
                attitude_delta = so3_log(
                    rotation_before.T @ estimator.state.rotation
                )
                world_yaw_axis_body = rotation_before.T @ np.array(
                    [0.0, 0.0, 1.0]
                )
                stats = accumulators[name]
                stats["maximum_dvl_position_jump_m"] = max(
                    stats["maximum_dvl_position_jump_m"],
                    float(
                        np.linalg.norm(
                            estimator.state.position - position_before
                        )
                    ),
                )
                stats["maximum_dvl_attitude_jump_deg"] = max(
                    stats["maximum_dvl_attitude_jump_deg"],
                    float(np.rad2deg(np.linalg.norm(attitude_delta))),
                )
                stats["maximum_dvl_global_yaw_component_deg"] = max(
                    stats["maximum_dvl_global_yaw_component_deg"],
                    abs(
                        float(
                            np.rad2deg(attitude_delta @ world_yaw_axis_body)
                        )
                    ),
                )
            dvl_index += 1
            next_dvl = (dvl_index + 1) * dvl_dt_s

        if next_depth <= timestamp_s + 1.0e-10:
            depth = sensors.depth(truth)
            for estimator in estimators.values():
                estimator.update_depth(depth)
            depth_index += 1
            next_depth = (depth_index + 1) * depth_dt_s

        for name, estimator in estimators.items():
            position_error_m = float(
                np.linalg.norm(estimator.state.position - truth.position)
            )
            stats = accumulators[name]
            stats["squared_position_error_sum"] += position_error_m**2
            stats["samples"] += 1
            stats["maximum_position_error_m"] = max(
                stats["maximum_position_error_m"], position_error_m
            )
            stats["final_position_error_m"] = position_error_m
            stats["final_yaw_error_deg"] = _yaw_error_deg(
                estimator.state.rotation, truth.rotation
            )

        if travelled_m + 1.0e-8 >= next_report_m:
            checkpoint = {
                "travelled_distance_m": float(travelled_m),
                "route_distance_m": float(
                    args.start_distance_m + travelled_m
                ),
                "filters": {
                    name: {
                        "position_error_m": float(
                            np.linalg.norm(
                                estimator.state.position - truth.position
                            )
                        ),
                        "yaw_error_deg": _yaw_error_deg(
                            estimator.state.rotation, truth.rotation
                        ),
                    }
                    for name, estimator in estimators.items()
                },
            }
            checkpoints.append(checkpoint)
            print(json.dumps(checkpoint, ensure_ascii=False), flush=True)
            next_report_m += args.report_distance_m

    results: dict[str, dict[str, float | int]] = {}
    for name, stats in accumulators.items():
        sample_count = int(stats.pop("samples"))
        squared_sum = float(stats.pop("squared_position_error_sum"))
        results[name] = {
            "position_rmse_m": float(np.sqrt(squared_sum / sample_count)),
            "samples": sample_count,
            **{key: float(value) for key, value in stats.items()},
        }
    report = {
        "schema": "oceansim.dvl-observability-regression/v1",
        "scope": "navigation-only IMU+DVL+depth; sonar and global matching disabled",
        "terrain": "USGS Monterey Canyon measured 5 m bathymetry",
        "seed": args.seed,
        "start_distance_m": args.start_distance_m,
        "requested_segment_distance_m": args.segment_distance_m,
        "travelled_distance_m": travelled_m,
        "speed_mps": args.speed_mps,
        "initialization": "exact truth for controlled filter comparison",
        "measurement_schedule_hz": {"imu": 50.0, "dvl": 5.0, "depth": 10.0},
        "wall_time_s": time.monotonic() - started,
        "results": results,
        "checkpoints": checkpoints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report["results"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
