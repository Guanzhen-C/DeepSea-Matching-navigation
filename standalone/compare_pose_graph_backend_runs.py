#!/usr/bin/env python3
"""Compare two full-route OceanSim pose-graph backend experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TRACKS = (
    "dead_reckoning_imu_dvl_depth",
    "local_acoustic_esikf",
    "online_global_slam",
    "final_optimized_global_slam",
    "gtsam_full_history_isam2_diagnostic",
)
POSITION_FIELDS = (
    "rmse_3d_m",
    "median_3d_m",
    "p95_3d_m",
    "p99_3d_m",
    "max_3d_m",
    "final_3d_m",
)
COUNT_FIELDS = (
    "accepted_geometry_updates",
    "rejected_geometry_updates",
    "accepted_intensity_updates",
    "rejected_intensity_updates",
    "intensity_factor_edges",
    "rejected_intensity_factors",
    "terrain_match_invocations",
    "accepted_terrain_match_invocations",
    "rejected_terrain_match_invocations",
    "accepted_terrain_prior_factors",
    "rejected_terrain_prior_factor_attempts",
    "loop_edges",
    "relocalization_edges",
    "normal_terrain_match_invocations",
    "recovery_terrain_match_invocations",
    "emergency_terrain_match_invocations",
    "emergency_confirmed_recoveries",
)
INVARIANT_PATHS = (
    "reproducibility.random_seed",
    "navigation_initialization.uses_simulator_truth",
    "navigation_initialization.horizontal_seed_offset_m",
    "navigation_initialization.yaw_seed_offset_deg",
    "sensors",
    "sonar_acquisition_model",
    "terrain_source",
    "terrain_resolution_m",
    "terrain_matching_reference",
    "vehicle",
    "sonar_profile",
    "survey_speed_mps",
    "survey_pattern",
    "requested_survey_laps",
    "survey_altitude_m",
    "survey_maximum_pitch_deg",
    "ping_rate_hz",
    "sonar_pings_in_water",
    "sonar_beams",
    "sidescan_imaging_rays_per_side",
    "sidescan_slam_sample_stride",
    "sonar_swath_width_deg",
    "intensity_reference_delay_pings",
    "intensity_group_pings",
    "intensity_min_group_span_m",
    "intensity_submap_anchor_interval_s",
    "intensity_effective_independent_samples",
    "local_sonar_noise_floor_m",
    "local_intensity_noise_std",
    "intensity_min_singular_ratio",
    "intensity_factor_min_interval_s",
    "intensity_factor_optimize_interval",
    "terrain_prior_surface",
    "terrain_match_window_pings",
    "terrain_patch_motion_compensation",
    "terrain_match_maximum_points",
    "terrain_patch_minimum_voxel_size_m",
    "terrain_position_only_priors",
    "terrain_prior_horizontal_std_floor_m",
    "terrain_prior_yaw_std_floor_deg",
    "terrain_gauge_confirmation_patches",
    "terrain_confirm_every_prior",
    "terrain_gauge_minimum_baseline_m",
    "terrain_gauge_maximum_position_residual_m",
    "terrain_gauge_maximum_yaw_residual_deg",
    "terrain_match_translation_window_m",
    "terrain_recovery_translation_window_m",
    "terrain_emergency_recovery_translation_window_m",
    "terrain_match_yaw_window_deg",
    "terrain_require_yaw_observable",
    "terrain_prior_optimize_interval",
    "slam_max_points",
    "slam_voxel_size_m",
    "intensity_voxel_size_m",
    "map_insert_stride",
    "trajectory_sample_interval_s",
    "mbes_group_pings",
    "association_map_half_extent_m",
    "pose_graph_odometry_noise_model",
    "backscatter_map_matching.enabled",
    "backscatter_map_matching.source_product",
    "backscatter_map_matching.resolution_m",
    "backscatter_map_matching.covariance_intersection_bathymetry_weight",
    "backscatter_map_matching.maximum_points",
    "backscatter_map_matching.maximum_yaw_deg",
    "backscatter_map_matching.boundary_yaw_retry_window_deg",
    "backscatter_map_matching.large_correction_compatibility_threshold_m",
    "backscatter_map_matching.horizontal_std_floor_m",
    "backscatter_map_matching.yaw_std_floor_deg",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def _nested(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _equivalent(first: Any, second: Any) -> bool:
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return bool(
            math.isfinite(float(first))
            and math.isfinite(float(second))
            and math.isclose(
                float(first), float(second), rel_tol=1.0e-10, abs_tol=1.0e-10
            )
        )
    return first == second


def _metric_pair(baseline: Any, candidate: Any) -> dict[str, Any]:
    if baseline is None or candidate is None:
        return {"baseline": baseline, "candidate": candidate, "delta": None}
    baseline_value = float(baseline)
    candidate_value = float(candidate)
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": candidate_value - baseline_value,
        "change_percent": (
            100.0 * (candidate_value - baseline_value) / abs(baseline_value)
            if abs(baseline_value) > 1.0e-12
            else None
        ),
    }


def _load_csv(path: Path) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",", names=True)
    if values.shape == ():
        values = values.reshape(1)
    return values


def _resolve_csv(metrics_path: Path, payload: dict[str, Any]) -> Path:
    recorded = payload.get("position_error_csv")
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    sibling = metrics_path.with_name(
        f"{metrics_path.stem}_position_errors.csv"
    )
    if not sibling.is_file():
        raise FileNotFoundError(
            f"position-error CSV is missing (recorded={recorded!r}, "
            f"sibling={sibling})"
        )
    return sibling


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _upper_bound_check(
    name: str,
    baseline: float,
    candidate: float,
    *,
    relative_allowance: float,
    absolute_allowance: float,
) -> dict[str, Any]:
    limit = float(baseline) * (1.0 + relative_allowance) + absolute_allowance
    return {
        "name": name,
        "lower_is_better": True,
        "baseline": float(baseline),
        "candidate": float(candidate),
        "relative_allowance": relative_allowance,
        "absolute_allowance": absolute_allowance,
        "failure_limit": limit,
        "passed": bool(math.isfinite(float(candidate)) and candidate <= limit),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pose-graph backend A/B comparison",
        "",
        f"- Structural verdict: `{report['structural_verdict']}`",
        f"- Baseline: `{report['baseline_path']}`",
        f"- Candidate: `{report['candidate_path']}`",
        f"- Timestamp streams identical: `{report['sampling']['timestamps_identical']}`",
        f"- Fixed-input mismatches: {len(report['invariant_mismatches'])}",
        f"- Accuracy verdict: `{report['accuracy_verdict']}`",
        "",
        "## Position error",
        "",
        "| Track | Metric | Baseline | Candidate | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for track, metrics in report["position_error"].items():
        for field, pair in metrics.items():
            lines.append(
                f"| {track} | {field} | {pair['baseline']:.6g} | "
                f"{pair['candidate']:.6g} | {pair['delta']:+.6g} |"
            )
    lines.extend(
        [
            "",
            "## Translation RPE RMSE",
            "",
            "| Track | Interval | Baseline (m) | Candidate (m) | Delta (m) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for track, intervals in report["translation_rpe_rmse_m"].items():
        for interval, pair in intervals.items():
            lines.append(
                f"| {track} | {interval} | {pair['baseline']:.6g} | "
                f"{pair['candidate']:.6g} | {pair['delta']:+.6g} |"
            )
    lines.extend(
        [
            "",
            "## Optimizer and runtime",
            "",
            "```json",
            json.dumps(report["optimizer_and_runtime"], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    if report["invariant_mismatches"]:
        lines.extend(
            [
                "",
                "## Fixed-input mismatches",
                "",
                "```json",
                json.dumps(
                    report["invariant_mismatches"], ensure_ascii=False, indent=2
                ),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _arguments()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    invariant_mismatches = []
    for path in INVARIANT_PATHS:
        first = _nested(baseline, path)
        second = _nested(candidate, path)
        if not _equivalent(first, second):
            invariant_mismatches.append(
                {"path": path, "baseline": first, "candidate": second}
            )

    baseline_csv = _load_csv(_resolve_csv(args.baseline, baseline))
    candidate_csv = _load_csv(_resolve_csv(args.candidate, candidate))
    baseline_time = np.asarray(baseline_csv[baseline_csv.dtype.names[0]])
    candidate_time = np.asarray(candidate_csv[candidate_csv.dtype.names[0]])
    timestamps_identical = bool(
        baseline_time.shape == candidate_time.shape
        and np.array_equal(baseline_time, candidate_time)
    )

    baseline_position = baseline["position_error_statistics"]
    candidate_position = candidate["position_error_statistics"]
    position_error: dict[str, Any] = {}
    for track in TRACKS:
        if track not in baseline_position or track not in candidate_position:
            continue
        position_error[track] = {
            field: _metric_pair(
                baseline_position[track].get(field),
                candidate_position[track].get(field),
            )
            for field in POSITION_FIELDS
        }

    baseline_rpe = baseline_position["translation_rpe_by_truth_distance"]
    candidate_rpe = candidate_position["translation_rpe_by_truth_distance"]
    rpe: dict[str, Any] = {}
    for track in TRACKS:
        if track not in baseline_rpe or track not in candidate_rpe:
            continue
        rpe[track] = {
            interval: _metric_pair(
                baseline_rpe[track][interval].get("rmse_translation_m"),
                candidate_rpe[track][interval].get("rmse_translation_m"),
            )
            for interval in ("100m", "1000m", "10000m")
        }

    orientation = {}
    for track in TRACKS:
        first = baseline.get("orientation_error_statistics", {}).get(track)
        second = candidate.get("orientation_error_statistics", {}).get(track)
        if first is None or second is None:
            continue
        orientation[track] = {
            axis: {
                metric: _metric_pair(first[axis][metric], second[axis][metric])
                for metric in ("rmse_deg", "p95_abs_deg", "final_abs_deg")
            }
            for axis in ("roll", "pitch", "yaw", "total_rotation")
        }

    accuracy_checks: list[dict[str, Any]] = []
    baseline_online = baseline_position["online_global_slam"]
    candidate_online = candidate_position["online_global_slam"]
    for field, relative, absolute in (
        ("rmse_3d_m", 0.25, 1.0),
        ("p95_3d_m", 0.25, 2.0),
        ("max_3d_m", 0.25, 10.0),
        ("final_3d_m", 0.50, 3.0),
    ):
        accuracy_checks.append(
            _upper_bound_check(
                f"online_global_slam.{field}",
                baseline_online[field],
                candidate_online[field],
                relative_allowance=relative,
                absolute_allowance=absolute,
            )
        )
    diagnostic = candidate_position.get(
        "gtsam_full_history_isam2_diagnostic"
    )
    if diagnostic is not None:
        baseline_final = baseline_position["final_optimized_global_slam"]
        for field, relative, absolute in (
            ("rmse_3d_m", 0.25, 1.0),
            ("p95_3d_m", 0.25, 2.0),
            ("max_3d_m", 0.25, 10.0),
            ("final_3d_m", 0.50, 3.0),
        ):
            accuracy_checks.append(
                _upper_bound_check(
                    f"gtsam_full_history_vs_numpy_final.{field}",
                    baseline_final[field],
                    diagnostic[field],
                    relative_allowance=relative,
                    absolute_allowance=absolute,
                )
            )
        for interval, absolute in (("100m", 1.0), ("1000m", 2.0), ("10000m", 5.0)):
            accuracy_checks.append(
                _upper_bound_check(
                    f"gtsam_full_history_vs_numpy_final.rpe.{interval}",
                    baseline_rpe["final_optimized_global_slam"][interval][
                        "rmse_translation_m"
                    ],
                    candidate_rpe[
                        "gtsam_full_history_isam2_diagnostic"
                    ][interval]["rmse_translation_m"],
                    relative_allowance=0.30,
                    absolute_allowance=absolute,
                )
            )

    segments = {}
    baseline_segments = baseline_position.get("five_kilometre_segments", {})
    candidate_segments = candidate_position.get("five_kilometre_segments", {})
    for track in TRACKS:
        first = baseline_segments.get(track)
        second = candidate_segments.get(track)
        if first is None or second is None:
            continue
        segments[track] = [
            {
                "start_distance_m": a["start_distance_m"],
                "stop_distance_m": a["stop_distance_m"],
                "rmse_3d_m": _metric_pair(a["rmse_3d_m"], b["rmse_3d_m"]),
                "p95_3d_m": _metric_pair(a["p95_3d_m"], b["p95_3d_m"]),
                "max_3d_m": _metric_pair(a["max_3d_m"], b["max_3d_m"]),
            }
            for a, b in zip(first, second, strict=False)
        ]

    factor_counts = {
        field: _metric_pair(baseline.get(field), candidate.get(field))
        for field in COUNT_FIELDS
    }
    optimizer_and_runtime = {
        "baseline_backend": baseline.get(
            "pose_graph_optimizer_backend", "numpy_batch_preinstrumentation"
        ),
        "candidate_backend": candidate.get("pose_graph_optimizer_backend"),
        "baseline_optimization_runs": baseline.get(
            "pose_graph_optimization_runs"
        ),
        "candidate_optimization_runs": candidate.get(
            "pose_graph_optimization_runs"
        ),
        "candidate_isam_updates": candidate.get("pose_graph_isam_updates"),
        "candidate_fixed_lag_updates": candidate.get(
            "pose_graph_fixed_lag_updates"
        ),
        "candidate_fixed_lag_active_variables": candidate.get(
            "pose_graph_fixed_lag_active_variables"
        ),
        "candidate_fixed_lag_marginalized_variables": candidate.get(
            "pose_graph_fixed_lag_marginalized_variables"
        ),
        "candidate_global_only_factors": candidate.get(
            "pose_graph_fixed_lag_global_only_factors"
        ),
        "candidate_global_anchor_bridge_updates": candidate.get(
            "pose_graph_global_anchor_bridge_updates"
        ),
        "candidate_global_anchor_bridge_max_translation_m": candidate.get(
            "pose_graph_global_anchor_bridge_max_translation_m"
        ),
        "candidate_global_anchor_bridge_max_abs_yaw_deg": candidate.get(
            "pose_graph_global_anchor_bridge_max_abs_yaw_deg"
        ),
        "wall_time_s": _metric_pair(
            baseline.get("wall_time_s"), candidate.get("wall_time_s")
        ),
        "achieved_time_scale": _metric_pair(
            baseline.get("achieved_time_scale_including_startup"),
            candidate.get("achieved_time_scale_including_startup"),
        ),
    }

    completed = all(
        item.get("run_status") == "completed"
        and item.get("completed_requested_route") is True
        for item in (baseline, candidate)
    )
    finite_candidate = _finite_tree(
        {
            "position": candidate_position,
            "orientation": candidate.get("orientation_error_statistics", {}),
            "map_to_odom": candidate.get(
                "online_map_to_odom_change_audit", {}
            ),
        }
    )
    map_to_odom_audit = candidate.get(
        "online_map_to_odom_change_audit", {}
    )
    backend_contract = bool(
        candidate.get("pose_graph_optimizer_backend")
        == "gtsam_isam2_incremental_fixed_lag"
        and int(candidate.get("pose_graph_isam_updates", 0)) > 0
        and int(candidate.get("pose_graph_fixed_lag_updates", 0)) > 0
        and int(candidate.get("pose_graph_fixed_lag_active_variables", 0)) > 0
        and int(
            candidate.get("pose_graph_fixed_lag_marginalized_variables", 0)
        )
        > 0
        and diagnostic is not None
    )
    no_nonfinite_online_states = bool(
        int(map_to_odom_audit.get("nonfinite_samples", -1)) == 0
    )
    structural_verdict = (
        "pass"
        if completed
        and not invariant_mismatches
        and timestamps_identical
        and finite_candidate
        and backend_contract
        and no_nonfinite_online_states
        else "fail"
    )
    accuracy_verdict = (
        "pass"
        if accuracy_checks and all(check["passed"] for check in accuracy_checks)
        else "fail"
    )
    overall_verdict = (
        "pass"
        if structural_verdict == "pass" and accuracy_verdict == "pass"
        else "fail"
    )
    report = {
        "schema": "oceansim.pose_graph_backend_ab.v1",
        "overall_verdict": overall_verdict,
        "structural_verdict": structural_verdict,
        "accuracy_verdict": accuracy_verdict,
        "baseline_path": str(args.baseline.resolve()),
        "candidate_path": str(args.candidate.resolve()),
        "completion": {
            "both_completed": completed,
            "baseline_status": baseline.get("run_status"),
            "candidate_status": candidate.get("run_status"),
        },
        "invariant_mismatches": invariant_mismatches,
        "sampling": {
            "baseline_samples": int(baseline_time.size),
            "candidate_samples": int(candidate_time.size),
            "timestamps_identical": timestamps_identical,
            "maximum_timestamp_difference_s": (
                float(np.max(np.abs(baseline_time - candidate_time)))
                if baseline_time.shape == candidate_time.shape
                else None
            ),
        },
        "candidate_metrics_finite": finite_candidate,
        "gtsam_backend_contract_satisfied": backend_contract,
        "candidate_online_map_to_odom_change_audit": map_to_odom_audit,
        "accuracy_acceptance_checks": accuracy_checks,
        "position_error": position_error,
        "translation_rpe_rmse_m": rpe,
        "orientation_error": orientation,
        "five_kilometre_segments": segments,
        "factor_counts": factor_counts,
        "optimizer_and_runtime": optimizer_and_runtime,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if overall_verdict == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
