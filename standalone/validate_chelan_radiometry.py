#!/usr/bin/env python3
"""Paired real-map acoustic matching validation without starting a renderer.

Uses the production acoustic intensity/noise utilities, 5 Hz, eight rows per
side, 256 rays/row/side, 0.15 m row spacing, 0.028 m range noise, 150-ping
windows and 5 m/6 m map-aware voxel means. Both variants see identical ray
geometry and noise draws. Native raster intersections use the CPU height-field
backend, not Isaac/Warp triangles. Exact deskew isolates acoustic radiometry;
this is not a long-course navigation result or an independent-survey trial.
Truth generates simulated observations and evaluates output; the searched
matcher gets a deliberately perturbed seed and no truth coordinates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaacsim"))
from oceansim.slam import (  # noqa: E402
    AcousticTerrainParticleLikelihood, BackscatterMapMatcher, BackscatterMatcherConfig,
    InertialState, MeasuredBathymetryGrid, WaypointTerrainFollowingSurvey,
    voxel_average_backscatter_patch,
)
from oceansim.slam.large_scale_sim import LargeScaleSensorSimulator, unresolved_reflectivity_multiplier  # noqa: E402
from oceansim.slam.math3d import rotation_from_rpy, rpy_from_rotation, wrap_angle  # noqa: E402
from oceansim.utils.acoustic_sonar import add_sonar_noise, compute_backscatter_intensity, normalize_intensities  # noqa: E402


def summarize_native_pair(directory: Path) -> int:
    reports = {name: json.loads((directory / filename).read_text()) for name, filename in (
        ("baseline", "native_legacy_600m.json"), ("db_power", "native_db_power_600m.json"),
    )}
    reference = reports["baseline"]
    checked = (
        "ping_rate_hz", "survey_speed_mps", "survey_altitude_m", "terrain_resolution_m",
        "terrain_match_window_pings", "terrain_match_translation_window_m",
        "terrain_match_yaw_window_deg", "feature_subframe_pings",
        "sidescan_imaging_rays_per_side_per_row", "sidescan_slam_sample_stride",
        "pose_graph_backend_configuration", "navigation_initialization",
    )
    for key in checked:
        if key not in reference or any(report.get(key) != reference[key] for report in reports.values()):
            raise ValueError(f"native paired run has differing or missing configuration: {key}")
    arrays = {}
    for name, report in reports.items():
        if report["run_status"] != "completed":
            raise ValueError(f"native run did not complete: {name}")
        if abs(report["survey_travelled_distance_m"] - 600) > .5:
            raise ValueError(f"native run did not finish the requested distance: {name}")
        filename = "native_legacy_600m_position_errors.csv" if name == "baseline" else "native_db_power_600m_position_errors.csv"
        arrays[name] = np.loadtxt(directory / filename, delimiter=",", skiprows=1)
    # IMU/DVL/depth and survey motion must be identical; image content is the
    # intentionally changed input. This also checks synchronized sample counts.
    np.testing.assert_allclose(arrays["baseline"][:, :2], arrays["db_power"][:, :2], atol=1e-10, rtol=0)
    result = {
        "scope": "600 m native Isaac/Warp CUDA frontend + cohort particle + GTSAM test; not a 47 km validation",
        "same_configuration_fields": list(checked),
        "identical_timestamps_and_dead_reckoning": True,
        "source_geometry_and_coverage_equal": True,
        "results": {},
    }
    for name, report in reports.items():
        result["results"][name] = {
            "distance_m": report["survey_travelled_distance_m"],
            "samples": report["position_error_statistics"]["online_global_slam"]["samples"],
            "global": report["position_error_statistics"]["online_global_slam"],
            "local": report["position_error_statistics"]["local_acoustic_esikf"],
            "accepted_absolute_priors": report["accepted_terrain_prior_factors"],
            "side_scan_accepted": report["backscatter_map_matching"]["accepts"],
            "side_scan_attempts": report["backscatter_map_matching"]["invocations"],
            "radiometric_contract": report["backscatter_map_matching"].get("source_radiometric_contract"),
        }
    before = result["results"]["baseline"]["global"]["rmse_3d_m"]
    after = result["results"]["db_power"]["global"]["rmse_3d_m"]
    result["global_rmse_reduction_percent"] = 100 * (1 - after / before)
    output = directory / "native_pair_comparison.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "data/underwater_terrain_datasets/lake_chelan_usgs_2023_3m"
    parser.add_argument("--baseline", type=Path, default=base / "prepared")
    parser.add_argument("--corrected", type=Path, default=base / "prepared_db_power")
    parser.add_argument("--distances-m", nargs="+", type=float, default=[100, 3000, 15000, 30000, 33150, 45000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[31])
    parser.add_argument("--output", type=Path, default=ROOT / "demo/generalization/chelan_radiometry_20261019/paired_acoustic_validation.json")
    parser.add_argument("--native-results", type=Path, help="summarize a completed 600 m native pair instead of generating observations")
    args = parser.parse_args()
    if args.native_results is not None:
        return summarize_native_pair(args.native_results)
    worlds = {"baseline": MeasuredBathymetryGrid.from_path(args.baseline),
              "db_power": MeasuredBathymetryGrid.from_path(args.corrected)}
    for key in ("elevation_m.npy", "navigation_elevation_m.npy", "bathymetry_valid_mask.npy",
                "backscatter_valid_mask.npy", "backscatter_source_mask.npy"):
        hashes = [w.metadata["output_arrays"][key]["sha256"] for w in worlds.values()]
        if hashes[0] != hashes[1]:
            raise ValueError(f"paired experiment changed geometry or coverage: {key}")
    meta = json.loads((args.baseline / "full_map_traversal_route.json").read_text())
    world = worlds["baseline"]
    survey = WaypointTerrainFollowingSurvey(
        world, np.array(meta["control_points_xy_m"]), speed_mps=1.5, altitude_m=40,
        smoothing_iterations=meta["route_definition"]["smoothing_iterations"], maximum_pitch_deg=12,
    )
    simulator = LargeScaleSensorSimulator(
        world, survey, num_beams=512, max_range=250, imaging_rays_per_side=256,
        along_track_rows_per_side=8, along_track_sample_spacing_m=.15,
    )
    records = []
    started = time.monotonic()
    for seed in args.seeds:
        for distance in args.distances_m:
            endpoint = distance / 1.5
            ref = survey.state(endpoint)
            reference = InertialState(position=ref.position, rotation=ref.rotation)
            chunks, values = [], {name: [] for name in worlds}
            range_rng = np.random.default_rng(seed)
            intensity_rngs = {name: np.random.default_rng(seed + 1) for name in worlds}
            for t in endpoint - np.arange(149, -1, -1) / 5:
                truth = survey.state(float(t))
                hit, ranges, pw, _, origins, directions = simulator._intersect_sensor_rays(
                    truth, simulator.shared_ray_origins_sensor + np.array([.1, 0, -.15]),
                    simulator.shared_ray_directions_sensor,
                )
                normals = world.normal(pw[:, 0], pw[:, 1])
                noisy_ranges = ranges.copy()
                noisy_ranges[hit] += range_rng.normal(0, .028, np.count_nonzero(hit))
                noisy_points = origins + directions * noisy_ranges[:, None]
                ids = np.flatnonzero(hit)[::8]
                chunks.append((noisy_points[ids] - ref.position) @ ref.rotation)
                for name, candidate_world in worlds.items():
                    refl = candidate_world.measured_reflectivity(pw[:, 0], pw[:, 1])
                    refl *= unresolved_reflectivity_multiplier(pw[:, 0], pw[:, 1], .18)
                    raw = compute_backscatter_intensity(ranges, directions, normals, refl, hit, attenuation=.003)
                    normalized = normalize_intensities(raw, hit)
                    noisy = add_sonar_noise(normalized, np.where(hit, noisy_ranges, 0), 250,
                                            intensity_rngs[name], multiplicative_std=.04, additive_std=.008)
                    values[name].append(noisy[ids])
            points = np.vstack(chunks)
            for name, candidate_world in worlds.items():
                body, intensity, voxel = voxel_average_backscatter_patch(
                    points, np.concatenate(values[name]), 2048,
                    minimum_voxel_size_m=candidate_world.resolution_m,
                )
                matcher = BackscatterMapMatcher(candidate_world, BackscatterMatcherConfig())
                truth_score = matcher._score(
                    np.zeros(3), reference, body @ ref.rotation.T, intensity,
                    matcher._radiometric_design(body), (), matcher._context_group_ids(body, 40),
                )
                roll, pitch, yaw = rpy_from_rotation(ref.rotation)
                estimate = InertialState(position=ref.position + np.array([-4, 3, 0]),
                                         rotation=rotation_from_rpy(roll, pitch, yaw - np.deg2rad(1.2)))
                result = matcher.match(body, intensity, estimate)
                particles = np.array([[ref.position[0] + dx, ref.position[1] + dy, yaw]
                                      for dx, dy in [(0, 0), (12, 0), (-12, 0), (0, 12), (0, -12)]])
                likelihood = AcousticTerrainParticleLikelihood(
                    candidate_world, reference, np.empty((0, 3)),
                    backscatter_points_body=body, backscatter_values=intensity,
                )
                scores = likelihood.evaluate(particles)
                row = dict(variant=name, seed=seed, distance_m=distance,
                           raw_points=len(points), voxel_m=voxel, retained_points=len(body),
                           true_pose_correlation=float(truth_score[0]) if np.isfinite(truth_score[0]) else None,
                           true_pose_texture_std=float(truth_score[4]), accepted=bool(result.accepted),
                           rejection_reason=result.rejection_reason,
                           horizontal_error_m=float(np.linalg.norm(result.matched_pose2[:2] - ref.position[:2])),
                           yaw_error_deg=float(np.rad2deg(wrap_angle(result.matched_pose2[2] - yaw))),
                           final_correlation=float(result.correlation_after) if np.isfinite(result.correlation_after) else None,
                           particle_true_pose_rank=int(np.count_nonzero(scores > scores[0])) + 1,
                           particle_truth_margin=float(scores[0] - max(scores[1:])))
                records.append(row)
                print(json.dumps(row), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            summary = {}
            for variant in worlds:
                rows = [row for row in records if row["variant"] == variant]
                finite = [row["true_pose_correlation"] for row in rows
                          if row["true_pose_correlation"] is not None]
                summary[variant] = {
                    "cases": len(rows), "accepted": sum(row["accepted"] for row in rows),
                    "finite_true_pose_scores": len(finite),
                    "median_finite_true_pose_correlation": float(np.median(finite)) if finite else None,
                    "true_particle_best": sum(row["particle_true_pose_rank"] == 1 for row in rows),
                    "accepted_horizontal_error_above_10m": sum(
                        row["accepted"] and row["horizontal_error_m"] > 10 for row in rows),
                }
            report = dict(protocol=__doc__, baseline=str(args.baseline), corrected=str(args.corrected),
                          elapsed_s=time.monotonic() - started, records=records, summary=summary,
                          ping_hz=5, pings_per_window=150, rows_per_side=8,
                          source_geometry_and_coverage_equal=True,
                          exact_deskew_is_diagnostic_only=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
