#!/usr/bin/env python3
"""Deterministic measured-map A/B for direct and feature Local frontends.

This runner intentionally has no Isaac Sim dependency.  It uses the same
Monterey bathymetry/backscatter grid, full-map route, navigation noise model,
V2 eight-row integrated acoustic geometry, and estimator classes as the live
demo.  Its purpose is fast, repeatable frontend tuning while an interactive or
long-running Isaac process is active; the final candidate must still pass the
Isaac smoke test before it becomes the formal launcher default.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = str(ROOT / "isaacsim")
if LOCAL_PACKAGE not in sys.path:
    sys.path.insert(0, LOCAL_PACKAGE)

from oceansim.slam import (  # noqa: E402
    AcousticESIKF,
    ESIKFConfig,
    FastSonarSLAM,
    FastSonarSLAMConfig,
    FeatureSonarOdometryConfig,
    MeasuredBathymetryGrid,
    WaypointTerrainFollowingSurvey,
    gauge_anchored_initial_covariance,
    sanitize_for_strict_json,
)
from oceansim.slam.large_scale_sim import LargeScaleSensorSimulator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bathymetry",
        type=Path,
        default=ROOT / "data/monterey_canyon_full_5m",
    )
    parser.add_argument(
        "--survey-route",
        type=Path,
        default=(
            ROOT
            / "data/monterey_canyon_full_5m/full_map_traversal_route.json"
        ),
    )
    parser.add_argument("--survey-start-distance-m", type=float, default=30_000.0)
    parser.add_argument("--survey-segment-distance-m", type=float, default=120.0)
    parser.add_argument("--survey-speed-mps", type=float, default=1.5)
    parser.add_argument("--survey-altitude-m", type=float, default=40.0)
    parser.add_argument("--survey-maximum-pitch-deg", type=float, default=12.0)
    parser.add_argument("--imu-hz", type=float, default=50.0)
    parser.add_argument("--ping-hz", type=float, default=5.0)
    parser.add_argument(
        "--subgrid-reflectivity-amplitude",
        type=float,
        default=0.0,
        help=(
            "optional world-fixed unresolved-scattering amplitude; zero "
            "uses only the measured 5 m backscatter mosaic"
        ),
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--feature-subframe-pings", type=int, default=4)
    parser.add_argument(
        "--feature-retain-direct-mbes-state-update",
        action="store_true",
        help=(
            "hybrid ablation: retain the direct point-to-map MBES state "
            "write in addition to feature odometry"
        ),
    )
    parser.add_argument(
        "--feature-position-noise-floor-m", type=float, default=0.15
    )
    parser.add_argument(
        "--feature-yaw-noise-floor-deg", type=float, default=5.0
    )
    parser.add_argument(
        "--feature-compute-backend",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="backend for batched feature-geometry consensus",
    )
    parser.add_argument("--checkpoint-distance-m", type=float, default=20.0)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print compact final metrics instead of the complete JSON payload",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=(
            ROOT
            / "demo/large_scale_slam/feature_frontend_measured_replay.json"
        ),
    )
    args = parser.parse_args()
    positive = (
        args.survey_segment_distance_m,
        args.survey_speed_mps,
        args.survey_altitude_m,
        args.imu_hz,
        args.ping_hz,
        args.checkpoint_distance_m,
        args.feature_position_noise_floor_m,
        args.feature_yaw_noise_floor_deg,
    )
    if args.survey_start_distance_m < 0.0 or not all(
        np.isfinite(value) and value > 0.0 for value in positive
    ):
        parser.error("distances, rates, speed, and altitude must be positive")
    if args.feature_subframe_pings < 2:
        parser.error("--feature-subframe-pings must be at least two")
    if not 0.0 <= args.subgrid_reflectivity_amplitude <= 0.75:
        parser.error("--subgrid-reflectivity-amplitude must be in [0, 0.75]")
    return args


def error_metrics(errors: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    return {
        "rmse_m": float(np.sqrt(np.mean(values * values))),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95.0)),
        "maximum_m": float(np.max(values)),
        "final_m": float(values[-1]),
    }


def make_frontend(
    mode: str,
    feature_subframe_pings: int,
    feature_position_noise_floor_m: float,
    feature_yaw_noise_floor_deg: float,
    feature_retain_direct_mbes_state_update: bool = False,
    feature_compute_backend: str = "cpu",
) -> FastSonarSLAM:
    imu_bias_limit = np.deg2rad(1.0) / 3600.0
    return FastSonarSLAM(
        FastSonarSLAMConfig(
            local_frontend_mode=mode,
            feature_retain_direct_mbes_state_update=(
                feature_retain_direct_mbes_state_update
            ),
            feature_odometry=FeatureSonarOdometryConfig(
                subframe_pings=feature_subframe_pings,
                position_noise_floor_m=feature_position_noise_floor_m,
                yaw_noise_floor_deg=feature_yaw_noise_floor_deg,
                compute_backend=feature_compute_backend,
            ),
            filter=ESIKFConfig(
                gyro_bias_max_norm=imu_bias_limit,
                sonar_max_points=384,
                sonar_noise_floor=0.30,
                sonar_continuous_observability_weighting=True,
                sonar_observable_subspace_projection=False,
                sonar_state_update_policy="position",
            ),
            voxel_size=5.0,
            intensity_voxel_size=3.0,
            map_insert_stride=2,
            map_max_range=255.0,
            association_map_half_extent_m=275.0,
            mbes_group_pings=1,
            local_map_history_duration_s=30.0,
            mbes_footprint_overlap_weighting_enabled=True,
            sidescan_min_range=0.0,
            sidescan_max_range=250.0,
            intensity_min_separation_pings=4,
            intensity_group_pings=1,
            intensity_min_group_span_m=1.0,
            intensity_updates_enabled=False,
            intensity_local_state_update_enabled=False,
            trajectory_sample_interval_s=0.2,
        )
    )


def main() -> int:
    args = parse_args()
    world = MeasuredBathymetryGrid.from_path(args.bathymetry)
    with args.survey_route.open("r", encoding="utf-8") as stream:
        route_metadata = json.load(stream)
    route_definition = route_metadata.get("route_definition", {})
    survey = WaypointTerrainFollowingSurvey(
        world,
        np.asarray(route_metadata["control_points_xy_m"], dtype=np.float64),
        speed_mps=args.survey_speed_mps,
        altitude_m=args.survey_altitude_m,
        smoothing_iterations=int(route_definition.get("smoothing_iterations", 4)),
        maximum_pitch_deg=args.survey_maximum_pitch_deg,
    )
    sensors = LargeScaleSensorSimulator(
        world,
        survey,
        num_beams=96,
        swath_width_deg=170.0,
        min_range=0.0,
        max_range=250.0,
        sidescan_range_bins=12_500,
        imaging_rays_per_side=256,
        along_track_rows_per_side=8,
        along_track_sample_spacing_m=0.15,
        subgrid_reflectivity_amplitude=(
            args.subgrid_reflectivity_amplitude
        ),
        seed=args.seed,
    )
    direct = make_frontend(
        "direct",
        args.feature_subframe_pings,
        args.feature_position_noise_floor_m,
        args.feature_yaw_noise_floor_deg,
        False,
        "cpu",
    )
    feature = make_frontend(
        "feature",
        args.feature_subframe_pings,
        args.feature_position_noise_floor_m,
        args.feature_yaw_noise_floor_deg,
        args.feature_retain_direct_mbes_state_update,
        args.feature_compute_backend,
    )
    dead_reckoning = AcousticESIKF(
        ESIKFConfig(gyro_bias_max_norm=np.deg2rad(1.0) / 3600.0)
    )

    route_time_offset = (
        args.survey_start_distance_m / args.survey_speed_mps
    )
    truth0 = survey.state(route_time_offset)
    initial_covariance = gauge_anchored_initial_covariance(
        gyro_bias_std_rad_s=np.deg2rad(1.0) / 7200.0
    )
    for estimator in (direct, feature, dead_reckoning):
        estimator.initialize(
            0.0,
            rotation=truth0.rotation,
            position=truth0.position,
            velocity=truth0.velocity,
            covariance=initial_covariance,
        )

    dt = 1.0 / args.imu_hz
    duration_s = args.survey_segment_distance_m / args.survey_speed_mps
    sample_count = int(np.ceil(duration_s / dt))
    dvl_stride = max(1, int(round(args.imu_hz / 5.0)))
    depth_stride = max(1, int(round(args.imu_hz / 10.0)))
    ping_stride = max(1, int(round(args.imu_hz / args.ping_hz)))
    errors = np.empty((sample_count, 3), dtype=np.float64)
    pose_errors = np.empty((sample_count, 3, 4), dtype=np.float64)
    sampled_rows: list[tuple[float, float, float, float]] = []
    checkpoints: list[dict[str, float]] = []
    next_checkpoint = args.checkpoint_distance_m
    started = time.monotonic()

    for index in range(1, sample_count + 1):
        timestamp = min(index * dt, duration_s)
        route_truth = survey.state(route_time_offset + timestamp)
        truth = replace(route_truth, timestamp=timestamp)
        gyro, acceleration = sensors.imu(truth)
        direct.process_imu(timestamp, gyro, acceleration)
        feature.process_imu(timestamp, gyro, acceleration)
        dead_reckoning.propagate(timestamp, gyro, acceleration)
        if index % dvl_stride == 0:
            dvl = sensors.dvl(truth)
            direct.process_dvl(dvl)
            feature.process_dvl(dvl)
            dead_reckoning.update_dvl(dvl)
        if index % depth_stride == 0:
            depth = sensors.depth(truth)
            direct.process_depth(depth)
            feature.process_depth(depth)
            dead_reckoning.update_depth(depth)
        if index % ping_stride == 0:
            ping = sensors.bathymetric_sidescan_ping(truth)
            direct.process_integrated_ping(ping, sample_stride=8)
            feature.process_integrated_ping(ping, sample_stride=8)

        current = np.array(
            [
                np.linalg.norm(dead_reckoning.state.position - truth.position),
                np.linalg.norm(direct.state.position - truth.position),
                np.linalg.norm(feature.state.position - truth.position),
            ],
            dtype=np.float64,
        )
        errors[index - 1] = current
        for estimator_index, estimator in enumerate(
            (dead_reckoning, direct, feature)
        ):
            delta = estimator.state.position - truth.position
            estimated_yaw = np.arctan2(
                estimator.state.rotation[1, 0],
                estimator.state.rotation[0, 0],
            )
            truth_yaw = np.arctan2(
                truth.rotation[1, 0], truth.rotation[0, 0]
            )
            yaw_error = np.arctan2(
                np.sin(estimated_yaw - truth_yaw),
                np.cos(estimated_yaw - truth_yaw),
            )
            pose_errors[index - 1, estimator_index] = (
                delta[0],
                delta[1],
                delta[2],
                np.rad2deg(yaw_error),
            )
        distance = args.survey_speed_mps * timestamp
        if index % dvl_stride == 0:
            sampled_rows.append((distance, *current))
        if distance + 1.0e-9 >= next_checkpoint:
            checkpoint = {
                "distance_m": float(distance),
                "dead_reckoning_error_m": float(current[0]),
                "direct_local_error_m": float(current[1]),
                "feature_local_error_m": float(current[2]),
            }
            checkpoints.append(checkpoint)
            print(
                "[feature A/B] "
                f"distance={distance:7.1f} m  "
                f"DR/direct/feature={current[0]:.3f}/"
                f"{current[1]:.3f}/{current[2]:.3f} m",
                flush=True,
            )
            next_checkpoint += args.checkpoint_distance_m

    direct_diagnostics = direct.diagnostics()
    feature_diagnostics = feature.diagnostics()
    metrics = {
        "schema": "oceansim.feature-frontend-measured-replay/v1",
        "configuration": {
            "bathymetry": str(args.bathymetry.resolve()),
            "survey_route": str(args.survey_route.resolve()),
            "survey_start_distance_m": args.survey_start_distance_m,
            "survey_segment_distance_m": args.survey_segment_distance_m,
            "survey_speed_mps": args.survey_speed_mps,
            "survey_altitude_m": args.survey_altitude_m,
            "survey_maximum_pitch_deg": args.survey_maximum_pitch_deg,
            "imu_hz": args.imu_hz,
            "ping_hz": args.ping_hz,
            "subgrid_reflectivity_amplitude": (
                args.subgrid_reflectivity_amplitude
            ),
            "reflectivity_provenance": (
                "measured USGS 5 m mosaic only"
                if args.subgrid_reflectivity_amplitude == 0.0
                else (
                    "measured USGS 5 m envelope plus explicitly synthetic "
                    "world-fixed subgrid scattering"
                )
            ),
            "seed": args.seed,
            "feature_subframe_pings": args.feature_subframe_pings,
            "feature_position_noise_floor_m": (
                args.feature_position_noise_floor_m
            ),
            "feature_yaw_noise_floor_deg": args.feature_yaw_noise_floor_deg,
            "feature_compute_backend_requested": (
                args.feature_compute_backend
            ),
            "feature_compute_backend_active": (
                feature_diagnostics.feature_odometry.active_compute_backend
            ),
            "feature_compute_device": (
                feature_diagnostics.feature_odometry.compute_device
            ),
            "feature_cross_track_noise_floor_m": (
                feature.config.feature_odometry.cross_track_position_noise_floor_m
            ),
            "feature_association_method": (
                feature.config.feature_odometry.association_method
            ),
            "feature_state_update_policy": (
                feature.config.feature_odometry.state_update_policy
            ),
            "feature_prediction_correction_scale_m": (
                feature.config.feature_odometry.prediction_correction_scale_m
            ),
            "feature_retain_direct_mbes_state_update": bool(
                feature.config.feature_retain_direct_mbes_state_update
            ),
            "initialization": "exact simulated truth",
            "raw_returns_per_ping": 2 * 8 * 256,
            "physical_ping_reuse_per_local_update": 1,
        },
        "dead_reckoning": error_metrics(errors[:, 0]),
        "direct_local": error_metrics(errors[:, 1]),
        "feature_local": error_metrics(errors[:, 2]),
        "pose_component_errors": {
            label: {
                "final_dx_dy_dz_yaw_deg": pose_errors[-1, index].tolist(),
                "rmse_dx_dy_dz_yaw_deg": np.sqrt(
                    np.mean(pose_errors[:, index] ** 2, axis=0)
                ).tolist(),
                "mean_dx_dy_dz_yaw_deg": np.mean(
                    pose_errors[:, index], axis=0
                ).tolist(),
            }
            for index, label in enumerate(
                ("dead_reckoning", "direct_local", "feature_local")
            )
        },
        "feature_odometry": asdict(feature_diagnostics.feature_odometry),
        "feature_filter_updates": feature_diagnostics.feature_filter_updates,
        "feature_filter_update_failures": (
            feature_diagnostics.feature_filter_update_failures
        ),
        "direct_geometry_updates": {
            "accepted": direct_diagnostics.accepted_sonar_updates,
            "rejected": direct_diagnostics.rejected_sonar_updates,
            "observable_rank_counts": (
                direct_diagnostics.sonar_observable_rank_counts
            ),
        },
        "checkpoints": checkpoints,
        "wall_time_s": time.monotonic() - started,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    strict_metrics, nonfinite_replacements = sanitize_for_strict_json(metrics)
    strict_metrics["json_serialization"] = {
        "strict_allow_nan": False,
        "nonfinite_replacements": int(nonfinite_replacements),
    }
    with args.metrics.open("w", encoding="utf-8") as stream:
        json.dump(
            strict_metrics,
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
    csv_path = args.metrics.with_name(args.metrics.stem + "_position_errors.csv")
    np.savetxt(
        csv_path,
        np.asarray(sampled_rows),
        delimiter=",",
        header=(
            "distance_m,dead_reckoning_error_m,direct_local_error_m,"
            "feature_local_error_m"
        ),
        comments="",
    )
    if args.summary_only:
        print(
            "[feature A/B] final RMSE DR/direct/feature="
            f"{metrics['dead_reckoning']['rmse_m']:.6f}/"
            f"{metrics['direct_local']['rmse_m']:.6f}/"
            f"{metrics['feature_local']['rmse_m']:.6f} m; final="
            f"{metrics['dead_reckoning']['final_m']:.6f}/"
            f"{metrics['direct_local']['final_m']:.6f}/"
            f"{metrics['feature_local']['final_m']:.6f} m",
            flush=True,
        )
    else:
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"[feature A/B] metrics: {args.metrics}", flush=True)
    print(f"[feature A/B] samples: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
