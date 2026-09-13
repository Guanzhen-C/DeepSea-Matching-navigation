#!/usr/bin/env python3
"""Run and evaluate the large-area OceanSim acoustic-inertial SLAM stack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = ROOT / "isaacsim"
if str(LOCAL_PACKAGE) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGE))

from oceansim.slam import (  # noqa: E402
    AcousticESIKF,
    ESIKFConfig,
    FastSonarSLAM,
    FastSonarSLAMConfig,
    gauge_anchored_initial_covariance,
    initialize_from_imu_dvl_depth,
)
from oceansim.slam.large_scale_sim import (  # noqa: E402
    FigureEightSurvey,
    LargeScaleSensorSimulator,
    LargeUnderwaterWorld,
)
from oceansim.slam.math3d import rpy_from_rotation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FAST-LIVO2-style IMU/DVL/depth/MBES/side-scan SLAM on a "
            "300 x 220 m deterministic underwater survey"
        )
    )
    parser.add_argument("--duration", type=float, default=360.0)
    parser.add_argument("--survey-period-s", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--imu-hz", type=float, default=50.0)
    parser.add_argument("--acoustic-hz", type=float, default=5.0)
    parser.add_argument("--slam-max-points", type=int, default=128)
    parser.add_argument("--mbes-group-pings", type=int, default=5)
    parser.add_argument(
        "--enable-mbes-sliding-window", action="store_true"
    )
    parser.add_argument(
        "--local-observability-mode",
        choices=("continuous", "hard", "legacy"),
        default="continuous",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "demo/large_scale_slam")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--disable-intensity-updates",
        action="store_true",
        help="run an MBES-geometry-only front-end ablation",
    )
    parser.add_argument(
        "--assert-quality",
        action="store_true",
        help="return a non-zero status if the deterministic quality gate fails",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be > 0")
    if args.survey_period_s <= 0.0:
        parser.error("--survey-period-s must be > 0")
    if args.imu_hz < 20.0:
        parser.error("--imu-hz must be >= 20")
    if args.acoustic_hz <= 0.0 or args.acoustic_hz > args.imu_hz:
        parser.error("--acoustic-hz must be in (0, --imu-hz]")
    if args.slam_max_points < 32:
        parser.error("--slam-max-points must be >= 32")
    if args.mbes_group_pings < 1:
        parser.error("--mbes-group-pings must be >= 1")
    return args


def _error_metrics(error: np.ndarray) -> dict[str, float]:
    return {
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "mean_m": float(np.mean(error)),
        "p95_m": float(np.percentile(error, 95.0)),
        "max_m": float(np.max(error)),
        "final_m": float(error[-1]),
    }


def run(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    world = LargeUnderwaterWorld(width=300.0, length=220.0)
    survey = FigureEightSurvey(period=args.survey_period_s)
    sensors = LargeScaleSensorSimulator(
        world,
        survey,
        num_beams=96,
        sidescan_range_bins=160,
        seed=args.seed,
    )
    initial_truth = survey.state(0.0)
    # Average a short quasi-static startup window.  Roll/pitch, velocity, and
    # depth come only from IMU/DVL/pressure.  With no compass or GNSS, yaw and
    # horizontal origin are gauge choices; simulation uses the truth-frame
    # gauge solely so plotted trajectories can be compared without alignment.
    startup_specific_force = []
    startup_dvl = []
    startup_depth = []
    for _ in range(32):
        _, specific_force = sensors.imu(initial_truth)
        startup_specific_force.append(specific_force)
        startup_dvl.append(sensors.dvl(initial_truth))
        startup_depth.append(sensors.depth(initial_truth))
    initial = initialize_from_imu_dvl_depth(
        np.asarray(startup_specific_force),
        np.asarray(startup_dvl),
        np.asarray(startup_depth),
        yaw_gauge_rad=float(rpy_from_rotation(initial_truth.rotation)[2]),
        horizontal_origin_m=initial_truth.position[:2],
    )

    imu_bias_limit = np.deg2rad(1.0) / 3600.0
    initial_covariance = gauge_anchored_initial_covariance(
        gyro_bias_std_rad_s=imu_bias_limit / 2.0
    )
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            filter=ESIKFConfig(
                gyro_bias_max_norm=imu_bias_limit,
                sonar_max_points=args.slam_max_points,
                sonar_continuous_observability_weighting=(
                    args.local_observability_mode == "continuous"
                ),
                sonar_observable_subspace_projection=(
                    args.local_observability_mode == "hard"
                ),
            ),
            mbes_group_pings=args.mbes_group_pings,
            mbes_sliding_window_enabled=(
                args.enable_mbes_sliding_window
            ),
            map_max_range=sensors.max_range,
            sidescan_min_range=sensors.min_range,
            sidescan_max_range=sensors.max_range,
            intensity_updates_enabled=not args.disable_intensity_updates,
        )
    )
    slam.initialize(
        0.0,
        rotation=initial.rotation,
        position=initial.position,
        velocity=initial.velocity,
        covariance=initial_covariance,
    )
    dead_reckoning = AcousticESIKF(
        ESIKFConfig(gyro_bias_max_norm=imu_bias_limit)
    )
    dead_reckoning.initialize(
        0.0,
        rotation=initial.rotation,
        position=initial.position,
        velocity=initial.velocity,
        covariance=initial_covariance,
    )

    dt = 1.0 / args.imu_hz
    sample_count = int(round(args.duration / dt))
    dvl_stride = max(1, int(round(args.imu_hz / 5.0)))
    depth_stride = max(1, int(round(args.imu_hz / 10.0)))
    acoustic_stride = max(
        1, int(round(args.imu_hz / args.acoustic_hz))
    )
    timestamps = np.arange(sample_count + 1, dtype=np.float64) * dt
    truth_positions = np.empty((sample_count + 1, 3), dtype=np.float64)
    dead_positions = np.empty_like(truth_positions)
    truth_positions[0] = initial_truth.position
    dead_positions[0] = dead_reckoning.state.position
    sidescan_history: list[np.ndarray] = []

    started = time.monotonic()
    next_report = 60.0
    for index in range(1, sample_count + 1):
        timestamp = float(timestamps[index])
        truth = survey.state(timestamp)
        truth_positions[index] = truth.position
        gyro, acceleration = sensors.imu(truth)
        slam.process_imu(timestamp, gyro, acceleration)
        dead_reckoning.propagate(timestamp, gyro, acceleration)

        if index % dvl_stride == 0:
            dvl = sensors.dvl(truth)
            slam.process_dvl(dvl)
            dead_reckoning.update_dvl(dvl)
        if index % depth_stride == 0:
            depth = sensors.depth(truth)
            slam.process_depth(depth)
            dead_reckoning.update_depth(depth)
        if index % acoustic_stride == 0:
            ping = sensors.bathymetric_sidescan_ping(truth)
            slam.process_integrated_ping(ping)
            sidescan_history.append(
                np.concatenate(
                    (ping.side_scan.port[::-1], ping.side_scan.starboard)
                )
            )

        dead_positions[index] = dead_reckoning.state.position
        if timestamp + 1.0e-9 >= next_report:
            graph = slam.pose_graph.diagnostics()
            local_error = np.linalg.norm(slam.state.position - truth.position)
            print(
                f"[OceanSim SLAM] t={timestamp:6.1f}s  "
                f"local_error={local_error:6.2f}m  "
                f"keyframes={graph.keyframes:3d}  loops={graph.loop_edges:3d}",
                flush=True,
            )
            next_report += 60.0

    trajectory_times, local_positions = slam.trajectory()
    global_times, global_positions = slam.global_trajectory()
    if not np.array_equal(trajectory_times, timestamps):
        raise RuntimeError("SLAM trajectory timestamps are not synchronized")
    if not np.array_equal(global_times, timestamps):
        raise RuntimeError("global trajectory timestamps are not synchronized")

    dead_error = np.linalg.norm(dead_positions - truth_positions, axis=1)
    local_error = np.linalg.norm(local_positions - truth_positions, axis=1)
    global_error = np.linalg.norm(global_positions - truth_positions, axis=1)
    path_length = float(
        np.sum(np.linalg.norm(np.diff(truth_positions, axis=0), axis=1))
    )
    diagnostics = slam.diagnostics()
    full_two_laps = args.duration >= 2.0 * survey.period - dt
    quality_checks = {
        "large_environment": world.width >= 300.0 and world.length >= 220.0,
        "large_trajectory": path_length >= 1_000.0 if full_two_laps else path_length > 0.0,
        "finite_solution": bool(np.isfinite(global_positions).all()),
        "map_populated": diagnostics.map_points >= 10_000 if full_two_laps else diagnostics.map_points > 0,
        "loop_closures": diagnostics.pose_graph.loop_edges >= 2 if full_two_laps else True,
        "local_rmse": _error_metrics(local_error)["rmse_m"] <= 1.0 if full_two_laps else True,
        "local_p95": _error_metrics(local_error)["p95_m"] <= 1.5 if full_two_laps else True,
        "global_rmse": _error_metrics(global_error)["rmse_m"] <= 1.0 if full_two_laps else True,
        "global_p95": _error_metrics(global_error)["p95_m"] <= 1.5 if full_two_laps else True,
    }
    metrics = {
        "algorithm": "OceanSim FastSonar-LIVO2",
        "sonar_acquisition_model": (
            "single_raw_return_grid_derived_bathymetry_and_backscatter"
        ),
        "seed": args.seed,
        "duration_s": args.duration,
        "survey_period_s": args.survey_period_s,
        "intensity_updates_enabled": not args.disable_intensity_updates,
        "acoustic_rate_hz": args.acoustic_hz,
        "slam_max_points": args.slam_max_points,
        "mbes_group_pings": args.mbes_group_pings,
        "mbes_sliding_window_enabled": (
            args.enable_mbes_sliding_window
        ),
        "local_observability_mode": args.local_observability_mode,
        "runtime_s": float(time.monotonic() - started),
        "environment_m": {"width": world.width, "length": world.length},
        "trajectory": {
            "path_length_m": path_length,
            "x_span_m": float(np.ptp(truth_positions[:, 0])),
            "y_span_m": float(np.ptp(truth_positions[:, 1])),
            "samples": int(timestamps.size),
        },
        "dead_reckoning": _error_metrics(dead_error),
        "local_sonar_odometry": _error_metrics(local_error),
        "global_slam": _error_metrics(global_error),
        "mapping": {
            "input_points": diagnostics.map_points,
            "total_committed_points": (
                diagnostics.total_geometry_points_committed
            ),
            "voxels": diagnostics.map_voxels,
            "association_map_voxels": diagnostics.association_map_voxels,
            "mature_surfels": diagnostics.mature_surfels,
            "optimized_keyframe_points": int(slam.optimized_map_points().shape[0]),
            "intensity_map_points": diagnostics.intensity_map_points,
            "intensity_map_cells": diagnostics.intensity_map_cells,
        },
        "updates": {
            "integrated_sonar_pings": diagnostics.integrated_sonar_pings,
            "bathymetric_pings": diagnostics.ping_count,
            "sidescan_lines": diagnostics.sidescan_lines,
            "coregistered_intensity_samples": (
                diagnostics.coregistered_intensity_samples
            ),
            "legacy_range_attached_intensity_samples": (
                diagnostics.legacy_range_attached_intensity_samples
            ),
            "accepted_sonar_updates": diagnostics.accepted_sonar_updates,
            "rejected_sonar_updates": diagnostics.rejected_sonar_updates,
            "accepted_intensity_updates": diagnostics.accepted_intensity_updates,
            "accepted_local_intensity_updates": (
                diagnostics.accepted_local_intensity_updates
            ),
            "accepted_local_intensity_updates_with_graph_factor": (
                diagnostics.accepted_local_intensity_updates_with_graph_factor
            ),
            "accepted_local_intensity_updates_without_graph_factor": (
                diagnostics.accepted_local_intensity_updates_without_graph_factor
            ),
            "accepted_graph_seeded_intensity_updates": (
                diagnostics.accepted_graph_seeded_intensity_updates
            ),
            "rejected_intensity_updates": diagnostics.rejected_intensity_updates,
            "committed_geometry_groups": diagnostics.committed_geometry_groups,
            "quarantined_geometry_groups": diagnostics.quarantined_geometry_groups,
            "committed_intensity_batches": diagnostics.committed_intensity_batches,
            "quarantined_intensity_batches": diagnostics.quarantined_intensity_batches,
            "sonar_rejection_reasons": diagnostics.sonar_rejection_reasons,
            "sonar_observable_rank_counts": (
                diagnostics.sonar_observable_rank_counts
            ),
            "intensity_rejection_reasons": (
                diagnostics.intensity_rejection_reasons
            ),
            "intensity_planar_rank_counts": (
                diagnostics.intensity_planar_rank_counts
            ),
            "mean_accepted_intensity_correlation": (
                diagnostics.mean_accepted_intensity_correlation
            ),
            "minimum_accepted_intensity_correlation": (
                diagnostics.minimum_accepted_intensity_correlation
            ),
            "mean_accepted_intensity_overlap": (
                diagnostics.mean_accepted_intensity_overlap
            ),
            "cumulative_intensity_translation_correction_m": (
                diagnostics.cumulative_intensity_translation_correction_m
            ),
            "cumulative_intensity_rotation_correction_deg": (
                diagnostics.cumulative_intensity_rotation_correction_deg
            ),
            "pending_intensity_batches": diagnostics.pending_intensity_batches,
            "pending_intensity_points": diagnostics.pending_intensity_points,
            "formed_intensity_strips": diagnostics.formed_intensity_strips,
            "skipped_intensity_strips": diagnostics.skipped_intensity_strips,
            "last_intensity_strip_span_m": (
                diagnostics.last_intensity_strip_span_m
            ),
            "keyframes": diagnostics.pose_graph.keyframes,
            "loop_edges": diagnostics.pose_graph.loop_edges,
            "relocalization_edges": (
                diagnostics.pose_graph.relocalization_edges
            ),
            "intensity_factor_edges": (
                diagnostics.pose_graph.intensity_edges
            ),
            "rejected_intensity_factors": (
                diagnostics.pose_graph.rejected_intensity_factors
            ),
            "intensity_factor_rejection_reasons": (
                diagnostics.pose_graph.intensity_factor_rejection_reasons
            ),
            "last_intensity_factor_rejection_reason": (
                diagnostics.pose_graph.last_intensity_factor_rejection_reason
            ),
            "last_intensity_prefit_performed": (
                diagnostics.pose_graph.last_intensity_prefit_performed
            ),
            "last_intensity_prefit_chi2": (
                diagnostics.pose_graph.last_intensity_prefit_chi2
            ),
            "last_intensity_prefit_dof": (
                diagnostics.pose_graph.last_intensity_prefit_dof
            ),
            "pose_graph_optimization_runs": (
                diagnostics.pose_graph.optimization_runs
            ),
        },
        "quality_checks": quality_checks,
        "passed": bool(all(quality_checks.values())),
    }
    arrays = {
        "timestamps": timestamps,
        "truth": truth_positions,
        "dead_reckoning": dead_positions,
        "local": local_positions,
        "global": global_positions,
        "dead_error": dead_error,
        "local_error": local_error,
        "global_error": global_error,
        "optimized_map": slam.optimized_map_points(),
        "sidescan": np.asarray(sidescan_history, dtype=np.float64),
    }
    return metrics, arrays


def save_results(output: Path, metrics: dict, arrays: dict[str, np.ndarray], no_plot: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    trajectory = np.column_stack(
        (
            arrays["timestamps"],
            arrays["truth"],
            arrays["dead_reckoning"],
            arrays["local"],
            arrays["global"],
            arrays["dead_error"],
            arrays["local_error"],
            arrays["global_error"],
        )
    )
    header = (
        "timestamp,truth_x,truth_y,truth_z,dead_x,dead_y,dead_z,"
        "local_x,local_y,local_z,global_x,global_y,global_z,"
        "dead_error,local_error,global_error"
    )
    np.savetxt(output / "trajectory.csv", trajectory, delimiter=",", header=header, comments="")
    np.savez_compressed(
        output / "map_and_trajectory.npz",
        **arrays,
    )
    if no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(arrays["truth"][:, 0], arrays["truth"][:, 1], "k-", lw=2.0, label="truth")
    ax.plot(
        arrays["dead_reckoning"][:, 0], arrays["dead_reckoning"][:, 1],
        color="#e67e22", lw=1.0, alpha=0.8, label="IMU+DVL+depth",
    )
    ax.plot(arrays["local"][:, 0], arrays["local"][:, 1], color="#3ba4d8", lw=1.2, label="local acoustic ESIKF")
    ax.plot(arrays["global"][:, 0], arrays["global"][:, 1], color="#21b573", lw=1.7, label="loop-corrected SLAM")
    ax.set_title("300 x 220 m underwater survey")
    ax.set_xlabel("East / m")
    ax.set_ylabel("North / m")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(arrays["timestamps"], arrays["dead_error"], color="#e67e22", label="dead reckoning")
    ax.plot(arrays["timestamps"], arrays["local_error"], color="#3ba4d8", label="local")
    ax.plot(arrays["timestamps"], arrays["global_error"], color="#21b573", label="global")
    ax.set_title("3-D position error")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Error / m")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    map_points = arrays["optimized_map"]
    if map_points.size:
        sample = map_points[:: max(1, map_points.shape[0] // 30_000)]
        scatter = ax.scatter(sample[:, 0], sample[:, 1], c=sample[:, 2], s=1.2, cmap="viridis")
        figure.colorbar(scatter, ax=ax, label="Seabed Z / m")
    ax.set_title("Loop-corrected MBES keyframe map")
    ax.set_xlabel("East / m")
    ax.set_ylabel("North / m")
    ax.axis("equal")

    ax = axes[1, 1]
    if arrays["sidescan"].size:
        ax.imshow(arrays["sidescan"].T, origin="lower", aspect="auto", cmap="gray")
    ax.set_title("Side-scan: direct intensity + place descriptors")
    ax.set_xlabel("Ping")
    ax.set_ylabel("Port ← slant range bins → starboard")
    figure.suptitle(
        "OceanSim FastSonar-LIVO2  |  "
        f"global RMSE {metrics['global_slam']['rmse_m']:.2f} m  |  "
        f"loops {metrics['updates']['loop_edges']}  |  PASS={metrics['passed']}",
        fontsize=14,
    )
    figure.savefig(output / "overview.png", dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    metrics, arrays = run(args)
    save_results(args.output, metrics, arrays, args.no_plot)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[OceanSim SLAM] results: {args.output.resolve()}")
    if args.assert_quality and not metrics["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
