#!/usr/bin/env python3
"""Deterministic, UI-free validation of the global relocalization chain.

This is an algorithm-integration validation, not an independent field-trial
accuracy claim. It samples a sequence of disjoint synthetic long strips from the
native Monterey bathymetry/backscatter product and matches bathymetry against
the separately low-pass navigation surface. Simulator truth generates the
synthetic sonar and ideal roll/pitch/depth inputs; no map-frame horizontal
position or yaw is passed to a matcher, filter, or graph. The hidden horizontal
truth is inspected only after each estimate to calculate reported errors.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaacsim"))

from oceansim.slam import (  # noqa: E402
    AcousticTerrainParticleLikelihood,
    AcousticPoseGraph,
    BBS3DCandidateVerifier,
    BBS3DVerifierConfig,
    BackscatterMapMatcher,
    BackscatterMatcherConfig,
    BathymetricTerrainMatcher,
    GlobalPoseEvidence,
    GlobalTerrainRelocalizer,
    GlobalTerrainRelocalizerConfig,
    InertialState,
    MapOdomSoftCorrectionConfig,
    MapOdomSoftCorrector,
    MeasuredBathymetryGrid,
    ParticleLikelihoodObservation,
    PoseGraphConfig,
    TerrainNavigationFilter,
    TerrainNavigationFilterConfig,
    TerrainParticleLikelihoodConfig,
    TerrainMatcherConfig,
    WaypointTerrainFollowingSurvey,
    reporting_covariance_from_information,
    sanitize_for_strict_json,
)
from oceansim.slam.math3d import (  # noqa: E402
    rotation_from_rpy,
    rpy_from_rotation,
    wrap_angle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bathymetry",
        type=Path,
        default=ROOT / "data/monterey_canyon_full_5m",
    )
    parser.add_argument(
        "--route",
        type=Path,
        default=(
            ROOT
            / "data/monterey_canyon_full_5m/full_map_traversal_route.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/oceansim_global_relocalization_chain.json"),
    )
    parser.add_argument("--seed", type=int, default=3109)
    parser.add_argument("--yaw-step-deg", type=float, default=5.0)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--bbs-timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--confirmation-probability",
        type=float,
        default=0.95,
        help="minimum posterior probability for graph-factor authorization",
    )
    parser.add_argument(
        "--confirmation-odds",
        type=float,
        default=20.0,
        help="minimum odds of the leading mode over the runner-up or null",
    )
    parser.add_argument(
        "--window-centres-s",
        type=float,
        nargs="+",
        default=tuple(float(value) for value in range(500, 6500, 500)),
        help=(
            "ordered centres of independent sonar windows; at least three "
            "are required and evaluation stops at the first safe confirmation"
        ),
    )
    args = parser.parse_args()
    if not args.bathymetry.exists():
        parser.error(f"bathymetry path does not exist: {args.bathymetry}")
    if not args.route.is_file():
        parser.error(f"route does not exist: {args.route}")
    if (
        not math.isfinite(args.yaw_step_deg)
        or args.yaw_step_deg <= 0.0
        or args.yaw_step_deg > 30.0
        or args.candidate_count < 1
        or args.bbs_timeout_ms < 0
        or not math.isfinite(args.confirmation_probability)
        or not 0.0 < args.confirmation_probability < 1.0
        or not math.isfinite(args.confirmation_odds)
        or args.confirmation_odds <= 0.0
        or len(args.window_centres_s) < 3
        or not all(math.isfinite(value) for value in args.window_centres_s)
        or not all(
            earlier < later
            for earlier, later in zip(
                args.window_centres_s, args.window_centres_s[1:]
            )
        )
    ):
        parser.error("invalid search or ordered-window configuration")
    return args


def yaw_error_deg(estimate: float, truth: float) -> float:
    return float(math.degrees(float(wrap_angle(estimate - truth))))


def raw_pose_from_map_pose(
    map_pose2: np.ndarray, map_to_odom_pose2: np.ndarray
) -> np.ndarray:
    """Invert ``map_pose = map_to_odom * odom_pose``."""

    delta_yaw = float(map_to_odom_pose2[2])
    cosine = math.cos(delta_yaw)
    sine = math.sin(delta_yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    xy = rotation.T @ (map_pose2[:2] - map_to_odom_pose2[:2])
    return np.array(
        [xy[0], xy[1], float(wrap_angle(map_pose2[2] - delta_yaw))],
        dtype=np.float64,
    )


def make_strip_observation(
    world: MeasuredBathymetryGrid,
    truth,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a motion-compensated, co-registered MBES/SSS long strip."""

    roll, pitch, yaw = rpy_from_rotation(truth.rotation)
    del roll, pitch
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    rotation_xy = np.array([[cosine, -sine], [sine, cosine]])
    along_track, cross_track = np.meshgrid(
        np.linspace(-180.0, 180.0, 37),
        np.linspace(-180.0, 180.0, 37),
        indexing="ij",
    )
    local_xy = np.column_stack(
        (along_track.reshape(-1), cross_track.reshape(-1))
    )
    world_xy = truth.position[:2] + local_xy @ rotation_xy.T
    elevation = world.seabed_height(world_xy[:, 0], world_xy[:, 1])
    reflectivity = world.measured_reflectivity(
        world_xy[:, 0], world_xy[:, 1]
    )
    valid = np.isfinite(elevation) & np.isfinite(reflectivity)
    if np.count_nonzero(valid) < 1000:
        raise RuntimeError("validation strip lacks measured joint-map support")

    # Native measured relief is the observation.  The terrain matcher below
    # sees only the low-pass navigation surface, avoiding an identical-grid
    # bathymetry inverse crime.  Small vertical noise represents the V2 range
    # repeatability without pretending to be a full acoustic ray simulation.
    noisy_elevation = elevation[valid] + rng.normal(
        0.0, 0.028, np.count_nonzero(valid)
    )
    world_points = np.column_stack((world_xy[valid], noisy_elevation))
    points_body = (world_points - truth.position) @ truth.rotation

    # Apply deterministic port/starboard gain, range trend, multiplicative
    # and additive noise.  Both global and local side-scan matchers must remove
    # the nuisance trend rather than receiving the map raster verbatim.
    cross_range = np.abs(points_body[:, 1])
    normalized_range = cross_range / max(
        float(np.percentile(cross_range, 95.0)), 1.0
    )
    log_gain = (
        0.18 * np.sign(points_body[:, 1])
        - 0.25 * normalized_range
        + rng.normal(0.0, 0.04, points_body.shape[0])
    )
    intensity = reflectivity[valid] * np.exp(log_gain)
    intensity += rng.normal(0.0, 0.008, intensity.size)
    intensity = np.clip(intensity, 0.01, None)
    return points_body, intensity


def refinement_quality(attempt: tuple[str, np.ndarray, object]) -> tuple:
    """Truth-free ordering of alternative metric-refinement seeds."""

    _source, _seed, result = attempt
    likelihood_ratio = float(result.basin_log_likelihood_ratio)
    if not np.isfinite(likelihood_ratio):
        likelihood_ratio = 1.0e300 if likelihood_ratio > 0.0 else -1.0e300
    rmse = float(result.rmse_after)
    if not np.isfinite(rmse):
        rmse = 1.0e300
    return (
        int(result.accepted),
        int(not result.reached_search_boundary),
        int(result.observable_rank),
        likelihood_ratio,
        -rmse,
        -float(result.translation_correction_m),
    )


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    world = MeasuredBathymetryGrid.from_path(args.bathymetry)
    with args.route.open("r", encoding="utf-8") as stream:
        route = json.load(stream)
    survey = WaypointTerrainFollowingSurvey(
        world,
        np.asarray(route["control_points_xy_m"], dtype=np.float64),
        speed_mps=1.5,
        altitude_m=40.0,
        smoothing_iterations=int(
            route.get("route_definition", {}).get("smoothing_iterations", 4)
        ),
        maximum_pitch_deg=12.0,
    )
    prior = MeasuredBathymetryGrid(
        world.navigation_elevation_m,
        resolution_m=world.resolution_m,
        navigation_elevation_m=world.navigation_elevation_m,
        valid_mask=world.valid_mask,
        metadata={
            **world.metadata,
            "terrain_matching_role": "same-survey low-pass navigation prior",
        },
        source_path=world.source_path,
    )

    relocalizer = GlobalTerrainRelocalizer(
        world,
        GlobalTerrainRelocalizerConfig(
            pyramid_stride=4,
            yaw_half_window_deg=180.0,
            yaw_step_deg=args.yaw_step_deg,
            maximum_candidates=args.candidate_count,
            minimum_joint_correlation=0.30,
            minimum_basin_score_gap=0.08,
        ),
    )
    bbs3d = BBS3DCandidateVerifier(
        world,
        BBS3DVerifierConfig(
            minimum_level_resolution_m=world.resolution_m,
            maximum_level=3,
            maximum_translation_m=30.0,
            maximum_yaw_deg=6.0,
            minimum_score_fraction=0.25,
            minimum_source_points=128,
            maximum_source_points=8192,
            timeout_milliseconds=args.bbs_timeout_ms,
        ),
        require_native=True,
    )
    terrain_config = TerrainMatcherConfig(
        measurement_std_m=1.50,
        huber_delta_m=2.25,
        inlier_threshold_m=3.75,
        maximum_points=768,
        maximum_translation_m=30.0,
        maximum_yaw_deg=8.0,
        require_yaw_observable=True,
        enable_innovation_nis_gate=False,
    )
    terrain_matcher = BathymetricTerrainMatcher(prior, terrain_config)
    terrain_fallback = BathymetricTerrainMatcher(
        prior, replace(terrain_config, require_yaw_observable=False)
    )
    backscatter_matcher = BackscatterMapMatcher(
        world,
        BackscatterMatcherConfig(
            maximum_points=4096,
            maximum_translation_m=30.0,
            maximum_yaw_deg=6.0,
            coarse_translation_step_m=5.0,
            coarse_yaw_step_deg=1.0,
            basin_translation_separation_m=10.0,
            basin_yaw_separation_deg=2.0,
            scale_space_cell_sizes_m=(10.0, 25.0, 50.0),
            scale_space_weight=0.35,
            context_consistency_weight=0.20,
            context_length_m=50.0,
            minimum_scale_cells=12,
            minimum_context_points=24,
        ),
    )
    tracker = TerrainNavigationFilter(
        TerrainNavigationFilterConfig(
            minimum_total_evidence=3,
            minimum_independent_evidence=3,
            confirmation_probability=args.confirmation_probability,
            confirmation_odds_over_second=args.confirmation_odds,
        )
    )
    graph = AcousticPoseGraph(
        PoseGraphConfig(
            fixed_lag_seconds=2000.0,
            keyframe_distance_m=1.0,
            keyframe_max_interval_s=1.0,
            odometry_reference_interval_s=60.0,
            terrain_prior_robust_kernel="cauchy",
            terrain_prior_optimize_interval=1,
        )
    )

    # A fixed transform creates kilometre-class local-frame error while
    # preserving exact relative SE(2) motion.  Neither matcher receives it.
    true_map_to_odom = np.array(
        [1000.0, -800.0, math.radians(20.0)], dtype=np.float64
    )
    window_records = []
    decisions = []
    latest = None

    for window_index, timestamp_s in enumerate(args.window_centres_s):
        window_started = time.perf_counter()
        truth = survey.state(timestamp_s)
        truth_rpy = rpy_from_rotation(truth.rotation)
        truth_pose2 = np.array(
            [truth.position[0], truth.position[1], truth_rpy[2]],
            dtype=np.float64,
        )
        raw_pose2 = raw_pose_from_map_pose(truth_pose2, true_map_to_odom)
        raw_state = InertialState(
            timestamp=float(timestamp_s),
            rotation=rotation_from_rpy(
                float(truth_rpy[0]),
                float(truth_rpy[1]),
                float(raw_pose2[2]),
            ),
            position=np.array(
                [raw_pose2[0], raw_pose2[1], truth.position[2]],
                dtype=np.float64,
            ),
        )
        points_body, intensities = make_strip_observation(world, truth, rng)
        observation_id = 100_000 + window_index
        patch_duration_s = 360.0 / 1.5
        patch_start_s = timestamp_s - patch_duration_s
        patch_end_s = timestamp_s
        ping_id_end = int(round(timestamp_s * 2.5))
        ping_id_start = int(round(patch_start_s * 2.5))
        raw_likelihood = AcousticTerrainParticleLikelihood(
            prior,
            raw_state,
            points_body,
            backscatter_terrain=world,
            backscatter_points_body=points_body,
            backscatter_values=intensities,
            config=TerrainParticleLikelihoodConfig(
                maximum_bathymetry_points=768,
                maximum_backscatter_points=768,
            ),
        )
        particle_observation = ParticleLikelihoodObservation(
            evaluator=raw_likelihood,
            patch_start_timestamp_s=patch_start_s,
            patch_end_timestamp_s=patch_end_s,
            patch_start_observation_id=ping_id_start,
            patch_end_observation_id=ping_id_end,
        )
        graph.add_if_needed(
            timestamp_s,
            raw_state.rotation,
            raw_state.position,
            points_body,
            observation_id=observation_id,
            force=True,
            odometry_pose2=raw_pose2,
            detect_loops=False,
        )

        search_started = time.perf_counter()
        search = relocalizer.search(
            points_body,
            raw_state,
            backscatter_points_body=points_body,
            backscatter_values=intensities,
            yaw_center_rad=float(raw_pose2[2]),
        )
        search_elapsed_s = time.perf_counter() - search_started
        evidences = []
        candidate_records = []
        verified_by_source = {}

        for source_index, candidate in enumerate(search.candidates):
            candidate_started = time.perf_counter()
            bbs_result = bbs3d.verify(
                points_body, raw_state, candidate.pose2
            )
            candidate_record = {
                "source_index": source_index,
                "coarse_pose2": candidate.pose2.tolist(),
                "coarse_joint_correlation": candidate.joint_correlation,
                "coarse_bathymetry_correlation": (
                    candidate.bathymetry_correlation
                ),
                "coarse_backscatter_correlation": (
                    candidate.backscatter_correlation
                ),
                "bbs3d_accepted": bbs_result.accepted,
                "bbs3d_score_fraction": bbs_result.score_fraction,
                "bbs3d_rejection_reason": bbs_result.rejection_reason,
                "bbs3d_pose2": bbs_result.matched_pose2.tolist(),
                "metric_seed_attempts": [],
                "accepted": False,
            }
            if not bbs_result.accepted:
                candidate_record["elapsed_s"] = (
                    time.perf_counter() - candidate_started
                )
                candidate_records.append(candidate_record)
                continue

            metric_seeds = [("appearance", candidate.pose2.copy())]
            bbs_seed = np.asarray(bbs_result.matched_pose2)
            separation_m = float(
                np.linalg.norm(bbs_seed[:2] - candidate.pose2[:2])
            )
            separation_yaw_deg = abs(
                yaw_error_deg(float(bbs_seed[2]), float(candidate.pose2[2]))
            )
            if (
                separation_m > 0.25 * world.resolution_m
                or separation_yaw_deg > 0.10
            ):
                metric_seeds.append(("bbs3d", bbs_seed.copy()))

            attempts = []
            for seed_source, seed_pose2 in metric_seeds:
                seed_state = raw_state.copy()
                seed_state.position[:2] = seed_pose2[:2]
                seed_state.rotation = rotation_from_rpy(
                    float(truth_rpy[0]),
                    float(truth_rpy[1]),
                    float(seed_pose2[2]),
                )
                terrain_result = terrain_matcher.match(
                    points_body, seed_state, points_are_body_frame=True
                )
                if (
                    not terrain_result.accepted
                    and terrain_result.rejection_reason == "yaw_unobservable"
                ):
                    terrain_result = terrain_fallback.match(
                        points_body, seed_state, points_are_body_frame=True
                    )
                attempts.append((seed_source, seed_pose2, terrain_result))
                candidate_record["metric_seed_attempts"].append(
                    {
                        "source": seed_source,
                        "seed_pose2": seed_pose2.tolist(),
                        "accepted": terrain_result.accepted,
                        "rejection_reason": terrain_result.rejection_reason,
                        "matched_pose2": terrain_result.matched_pose2.tolist(),
                        "rmse_after_m": terrain_result.rmse_after,
                        "observable_rank": terrain_result.observable_rank,
                        "search_boundary": (
                            terrain_result.reached_search_boundary
                        ),
                    }
                )
            selected_source, _, terrain_result = max(
                attempts, key=refinement_quality
            )
            candidate_record["selected_metric_seed"] = selected_source
            candidate_record["terrain_accepted"] = terrain_result.accepted
            candidate_record["terrain_rejection_reason"] = (
                terrain_result.rejection_reason
            )
            if not terrain_result.accepted:
                candidate_record["elapsed_s"] = (
                    time.perf_counter() - candidate_started
                )
                candidate_records.append(candidate_record)
                continue

            backscatter_state = raw_state.copy()
            backscatter_state.position[:2] = terrain_result.matched_pose2[:2]
            backscatter_state.rotation = rotation_from_rpy(
                float(truth_rpy[0]),
                float(truth_rpy[1]),
                float(terrain_result.matched_pose2[2]),
            )
            backscatter_result = backscatter_matcher.match(
                points_body, intensities, backscatter_state
            )
            candidate_record["backscatter_accepted"] = (
                backscatter_result.accepted
            )
            candidate_record["backscatter_rejection_reason"] = (
                backscatter_result.rejection_reason
            )
            candidate_record["backscatter_correlation"] = (
                backscatter_result.correlation_after
            )
            candidate_record["scale_space_correlation"] = (
                backscatter_result.scale_space_correlation_after
            )
            candidate_record["context_consistency"] = (
                backscatter_result.context_consistency_after
            )
            if not backscatter_result.accepted:
                candidate_record["elapsed_s"] = (
                    time.perf_counter() - candidate_started
                )
                candidate_records.append(candidate_record)
                continue

            inlier_ratio = terrain_result.inliers / max(
                terrain_result.correspondences, 1
            )
            geometry_quality = float(
                np.clip(
                    0.5 * inlier_ratio
                    + 0.5
                    * math.exp(
                        -0.5
                        * max(0.0, terrain_result.reduced_chi_square - 1.0)
                    ),
                    0.0,
                    1.0,
                )
            )
            geometry_score = math.sqrt(
                geometry_quality * bbs_result.score_fraction
            )
            backscatter_observability = float(
                np.clip(
                    backscatter_result.observation_texture_std
                    / (5.0 * backscatter_matcher.config.minimum_texture_std),
                    0.0,
                    1.0,
                )
                * np.clip(backscatter_result.valid_fraction, 0.0, 1.0)
            )
            evidence = GlobalPoseEvidence(
                map_pose2=backscatter_result.matched_pose2,
                bathymetry_score=candidate.bathymetry_evidence_score,
                backscatter_score=backscatter_result.correlation_after,
                bathymetry_observability=(
                    search.bathymetry_observability
                    * min(1.0, terrain_result.translation_observable_rank / 2.0)
                ),
                backscatter_observability=backscatter_observability,
                geometry_score=geometry_score,
                position_std_m=2.5,
                yaw_std_deg=1.5,
                patch_start_timestamp_s=patch_start_s,
                patch_end_timestamp_s=patch_end_s,
                patch_start_observation_id=ping_id_start,
                patch_end_observation_id=ping_id_end,
                source_index=source_index,
            )
            evidences.append(evidence)
            verified_by_source[source_index] = (
                evidence,
                terrain_result,
                backscatter_result,
            )
            candidate_record.update(
                {
                    "accepted": True,
                    "matched_pose2": evidence.map_pose2.tolist(),
                    "geometry_score": geometry_score,
                    "horizontal_error_m": float(
                        np.linalg.norm(evidence.map_pose2[:2] - truth_pose2[:2])
                    ),
                    "yaw_error_deg": yaw_error_deg(
                        float(evidence.map_pose2[2]), float(truth_pose2[2])
                    ),
                    "elapsed_s": time.perf_counter() - candidate_started,
                }
            )
            candidate_records.append(candidate_record)

        decision = tracker.update(
            timestamp_s=timestamp_s,
            raw_pose2=raw_pose2,
            evidences=evidences,
            particle_observation=particle_observation,
        )
        decisions.append(decision)
        particle_diagnostics = raw_likelihood.last_diagnostics
        selected_current = None
        if decision.confirmed_cohort is not None:
            selected_current = verified_by_source.get(
                decision.confirmed_cohort.last_source_index
            )
        window_records.append(
            {
                "window_index": window_index,
                "timestamp_s": timestamp_s,
                "truth_pose2": truth_pose2.tolist(),
                "raw_odom_pose2": raw_pose2.tolist(),
                "strip_points": int(points_body.shape[0]),
                "search_accepted": search.accepted,
                "search_rejection_reason": search.rejection_reason,
                "search_best_score": search.best_score,
                "search_basin_gap": search.basin_score_gap,
                "search_elapsed_s": search_elapsed_s,
                "search_candidates": len(search.candidates),
                "verified_candidates": len(evidences),
                "candidates": candidate_records,
                "cohort_status": decision.status,
                "cohort_best_probability": decision.best_probability,
                "cohort_null_probability": decision.null_probability,
                "cohort_odds_over_second": decision.odds_over_second,
                "cohort_best_independent_evidence": (
                    decision.cohorts[0].independent_evidence
                    if decision.cohorts
                    else 0
                ),
                "cohorts": [
                    {
                        "cohort_id": cohort.cohort_id,
                        "probability": cohort.probability,
                        "independent_evidence": cohort.independent_evidence,
                        "last_source_index": cohort.last_source_index,
                        "last_update_was_independent": (
                            cohort.last_update_was_independent
                        ),
                        "last_measurement_observable_rank": (
                            cohort.last_measurement_observable_rank
                        ),
                        "last_measurement_is_graph_ready": (
                            cohort.last_measurement_is_graph_ready
                        ),
                    }
                    for cohort in decision.cohorts
                ],
                "joint_particle_likelihood": (
                    None
                    if particle_diagnostics is None
                    else {
                        "particle_count": int(
                            particle_diagnostics.particle_count
                        ),
                        "bathymetry_points": int(
                            particle_diagnostics.bathymetry_points
                        ),
                        "backscatter_points": int(
                            particle_diagnostics.backscatter_points
                        ),
                        "bathymetry_observability": float(
                            particle_diagnostics.bathymetry_observability
                        ),
                        "backscatter_observability": float(
                            particle_diagnostics.backscatter_observability
                        ),
                        "log_likelihood_ratio_span": float(
                            particle_diagnostics.log_likelihood_ratio_max
                            - particle_diagnostics.log_likelihood_ratio_min
                        ),
                    }
                ),
                "elapsed_s": time.perf_counter() - window_started,
            }
        )
        mode_summary = [
            (
                item.cohort_id,
                round(item.probability, 4),
                item.independent_evidence,
            )
            for item in decision.cohorts
        ]
        print(
            f"[OceanSim] window {window_index + 1}/"
            f"{len(args.window_centres_s)}: "
            f"search={'accepted' if search.accepted else search.rejection_reason}, "
            f"candidates={len(search.candidates)}, "
            f"verified={len(evidences)}, cohort={decision.status}, "
            f"posterior={decision.best_probability:.6f}, "
            f"null={decision.null_probability:.6f}, "
            f"odds={decision.odds_over_second:.3f}, "
            f"modes={mode_summary}",
            flush=True,
        )
        latest = {
            "timestamp_s": timestamp_s,
            "truth": truth,
            "truth_pose2": truth_pose2,
            "raw_state": raw_state,
            "raw_pose2": raw_pose2,
            "points_body": points_body,
            "observation_id": observation_id,
            "decision": decision,
            "selected_current": selected_current,
            "raw_likelihood": raw_likelihood,
        }
        if (
            decision.confirmed
            and decision.confirmed_with_new_independent_evidence
        ):
            break

    if latest is None:
        raise RuntimeError("no validation windows were evaluated")
    confirmed = latest["decision"].confirmed_cohort
    if confirmed is None or latest["selected_current"] is None:
        raise RuntimeError(
            "persistent cohort did not reach the configured posterior and "
            "odds confirmation criteria within the available windows"
        )

    _selected_evidence, _selected_terrain, _selected_backscatter = latest[
        "selected_current"
    ]
    current_measurement = latest[
        "raw_likelihood"
    ].approximate_local_measurement(confirmed.map_pose2)
    reporting_covariance = reporting_covariance_from_information(
        current_measurement.information
    )
    factor_added = graph.add_terrain_prior(
        timestamp=latest["timestamp_s"],
        observation_id=latest["observation_id"],
        prior_pose2=latest["raw_pose2"],
        raw_rotation_world_body=latest["raw_state"].rotation,
        raw_position_world=latest["raw_state"].position,
        matched_pose2=current_measurement.pose2,
        information_world=current_measurement.information,
        points_body=latest["points_body"],
        optimize_interval=1,
        relocalization_seed=True,
        relocalization_seed_pose2=confirmed.map_pose2,
        relocalization_prior_robust=True,
    )
    if not factor_added:
        reason = graph.diagnostics().last_terrain_prior_rejection_reason
        raise RuntimeError(f"confirmed terrain factor was rejected: {reason}")
    graph.optimize()
    graph_rotation, graph_position = graph.corrected_pose(
        latest["raw_state"].rotation, latest["raw_state"].position
    )
    graph_pose2 = np.array(
        [
            graph_position[0],
            graph_position[1],
            rpy_from_rotation(graph_rotation)[2],
        ]
    )
    factor_payload = graph.factor_stream_payload()
    robust_prior_retained = bool(
        factor_payload["priors"]
        and factor_payload["priors"][-1]["robust"]
        and factor_payload["priors"][-1]["kind"] == "terrain_prior"
    )

    # Verify continuity separately from graph correctness.  The correction is
    # first presented at the same timestamp, then followed at one-second output
    # epochs until converged.  The local pose remains deliberately untouched.
    corrector = MapOdomSoftCorrector(
        MapOdomSoftCorrectionConfig(
            translation_time_constant_s=5.0,
            yaw_time_constant_s=5.0,
            maximum_translation_rate_mps=5.0,
            maximum_yaw_rate_deg_s=5.0,
        )
    )
    corrector.reset(timestamp=latest["timestamp_s"])
    before_rotation, before_position = corrector.correct_pose(
        latest["timestamp_s"],
        latest["raw_state"].rotation,
        latest["raw_state"].position,
        latest["raw_state"].rotation,
        latest["raw_state"].position,
    )
    first_rotation, first_position = corrector.correct_pose(
        latest["timestamp_s"],
        latest["raw_state"].rotation,
        latest["raw_state"].position,
        graph_rotation,
        graph_position,
    )
    initial_jump_m = float(np.linalg.norm(first_position - before_position))
    output_positions = [first_position.copy()]
    output_yaws = [float(rpy_from_rotation(first_rotation)[2])]
    for step in range(1, 2001):
        output_rotation, output_position = corrector.correct_pose(
            latest["timestamp_s"] + step,
            latest["raw_state"].rotation,
            latest["raw_state"].position,
            graph_rotation,
            graph_position,
        )
        output_positions.append(output_position.copy())
        output_yaws.append(float(rpy_from_rotation(output_rotation)[2]))
        if corrector.snapshot().remaining_translation_m < 0.01 and (
            corrector.snapshot().remaining_yaw_deg < 0.01
        ):
            break
    output_positions = np.asarray(output_positions)
    output_yaws = np.unwrap(np.asarray(output_yaws))
    maximum_soft_translation_step_m = float(
        np.max(np.linalg.norm(np.diff(output_positions[:, :2], axis=0), axis=1))
    )
    maximum_soft_yaw_step_deg = float(
        np.rad2deg(np.max(np.abs(np.diff(output_yaws))))
    )

    accepted_errors = [
        (
            candidate["horizontal_error_m"],
            abs(candidate["yaw_error_deg"]),
        )
        for window in window_records
        for candidate in window["candidates"]
        if candidate["accepted"]
    ]
    best_error_per_window = [
        min(
            candidate["horizontal_error_m"]
            for candidate in window["candidates"]
            if candidate["accepted"]
        )
        for window in window_records
    ]
    checks = {
        "at_least_three_disjoint_windows_have_verified_candidate": (
            len(window_records) >= 3
            and all(
                window["verified_candidates"] >= 1
                for window in window_records
            )
        ),
        "no_confirmation_before_three_independent_windows": (
            all(not decision.confirmed for decision in decisions[:2])
            and decisions[-1].confirmed
            and len(decisions) >= 3
        ),
        "confirmation_uses_current_independent_evidence": (
            decisions[-1].confirmed_with_new_independent_evidence
        ),
        "confirmed_posterior_at_least_configured_probability": (
            decisions[-1].best_probability
            >= args.confirmation_probability
        ),
        "confirmed_odds_at_least_configured_ratio": (
            decisions[-1].odds_over_second >= args.confirmation_odds
        ),
        "all_windows_use_one_joint_raw_particle_likelihood": (
            tracker.diagnostics()["joint_particle_likelihood_updates"]
            == len(window_records)
        ),
        "current_raw_measurement_has_two_observable_directions": (
            current_measurement.observable_rank >= 2
        ),
        "rank_aware_reporting_covariance_is_positive_definite": bool(
            np.all(np.linalg.eigvalsh(reporting_covariance) > 0.0)
        ),
        "best_verified_horizontal_error_each_window_below_5m": all(
            value < 5.0 for value in best_error_per_window
        ),
        "robust_gtsam_relocalization_factor_retained": robust_prior_retained,
        "graph_horizontal_error_below_5m": (
            float(
                np.linalg.norm(
                    graph_pose2[:2] - latest["truth_pose2"][:2]
                )
            )
            < 5.0
        ),
        "graph_yaw_error_below_2deg": (
            abs(
                yaw_error_deg(
                    float(graph_pose2[2]),
                    float(latest["truth_pose2"][2]),
                )
            )
            < 2.0
        ),
        "soft_correction_has_no_same_epoch_jump": initial_jump_m < 1.0e-9,
        "soft_translation_step_respects_5mps": (
            maximum_soft_translation_step_m <= 5.0 + 1.0e-9
        ),
        "soft_yaw_step_respects_5degps": (
            maximum_soft_yaw_step_deg <= 5.0 + 1.0e-9
        ),
        "soft_correction_converges": (
            corrector.snapshot().remaining_translation_m < 0.01
            and corrector.snapshot().remaining_yaw_deg < 0.01
        ),
    }
    passed = all(checks.values())
    report = {
        "schema": "oceansim.global-relocalization-chain-validation/v2",
        "passed": passed,
        "validation_scope": (
            "deterministic algorithm-integration validation on real map data; "
            "not an independent sea trial or full acoustic-fidelity validation"
        ),
        "data": {
            "bathymetry_path": str(args.bathymetry.resolve()),
            "route_path": str(args.route.resolve()),
            "source_product": world.metadata.get("source_product"),
            "resolution_m": world.resolution_m,
            "extent_m": [world.width, world.length],
            "bathymetry_observation_surface": "native measured elevation",
            "bathymetry_prior_surface": "same-survey low-pass navigation grid",
            "side_scan_observation_and_prior": (
                "same measured backscatter mosaic with simulated radiometric "
                "nuisance/noise; algorithm wiring upper bound"
            ),
            "truth_policy": (
                "used to synthesize sonar and ideal roll/pitch/depth; "
                "map-frame horizontal position/yaw hidden from every "
                "matcher, filter, and graph; inspected after estimation for "
                "error reporting"
            ),
        },
        "algorithm": {
            "whole_map_search": "joint MBES relief/depth + side-scan appearance",
            "yaw_search_deg": [-180.0, 180.0],
            "yaw_step_deg": args.yaw_step_deg,
            "bbs3d_backend": (
                "KOKIAOKI/3d_bbs@"
                "41529a34a2fb9618b5ff560fb3c2363f1615666d"
            ),
            "bbs3d_role": "required geometric veto plus optional refinement seed",
            "metric_seed_policy": (
                "truth-free dual appearance/BBS seeds; accepted rank, boundary, "
                "basin likelihood and residual ordering"
            ),
            "side_scan_refinement": (
                "geocoded NCC with scale-space and local context adaptation; "
                "SC-Match-inspired, not an exact SC-Match implementation"
            ),
            "persistent_hypotheses": (
                "jointly normalized map-frame point-mass/cohort particle "
                "filter (256 SE(2) particles per cohort, ESS-triggered "
                "regularized systematic resampling) plus explicit null, "
                "change-point candidate mass, and non-overlapping acquisition "
                "windows observed until the configured posterior is decisive"
            ),
            "particle_measurement": (
                "current raw bathymetry and co-registered side-scan map "
                "likelihood evaluated jointly across every cohort"
            ),
            "validation_confirmation_probability": (
                args.confirmation_probability
            ),
            "validation_confirmation_odds": args.confirmation_odds,
            "production_launcher_confirmation_probability": 0.97,
            "production_launcher_confirmation_odds": 20.0,
            "backend": "GTSAM IncrementalFixedLagSmoother + robust Cauchy prior",
            "map_to_odom": "output-only bounded first-order soft correction",
        },
        "injected_local_frame_transform_pose2": true_map_to_odom.tolist(),
        "windows": window_records,
        "tracker_diagnostics": tracker.diagnostics(),
        "factor": {
            "added": factor_added,
            "robust_prior_retained": robust_prior_retained,
            "graph_pose2": graph_pose2.tolist(),
            "truth_pose2": latest["truth_pose2"].tolist(),
            "horizontal_error_m": float(
                np.linalg.norm(
                    graph_pose2[:2] - latest["truth_pose2"][:2]
                )
            ),
            "yaw_error_deg": yaw_error_deg(
                float(graph_pose2[2]), float(latest["truth_pose2"][2])
            ),
            "current_measurement_pose2": current_measurement.pose2.tolist(),
            "current_measurement_observable_rank": int(
                current_measurement.observable_rank
            ),
            "current_measurement_scaled_information_eigenvalues": (
                current_measurement.scaled_information_eigenvalues.tolist()
            ),
            "reporting_covariance_positive_definite": bool(
                np.all(np.linalg.eigvalsh(reporting_covariance) > 0.0)
            ),
            "pose_graph_backend": graph.diagnostics().optimizer_backend,
        },
        "map_to_odom_soft_correction": {
            "same_epoch_initial_jump_m": initial_jump_m,
            "maximum_one_second_translation_step_m": (
                maximum_soft_translation_step_m
            ),
            "maximum_one_second_yaw_step_deg": maximum_soft_yaw_step_deg,
            "convergence_steps": len(output_positions) - 1,
            "diagnostics": corrector.diagnostics(),
        },
        "accepted_candidate_error_summary": {
            "count": len(accepted_errors),
            "maximum_horizontal_error_m": max(
                value[0] for value in accepted_errors
            ),
            "maximum_abs_yaw_error_deg": max(
                value[1] for value in accepted_errors
            ),
            "best_horizontal_error_each_window_m": best_error_per_window,
        },
        "checks": checks,
        "elapsed_s": time.perf_counter() - started,
    }
    report, replacements = sanitize_for_strict_json(report)
    report["json_nonfinite_replacements"] = replacements
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(args.output)
    print(
        "[OceanSim] global relocalization chain: "
        f"{'PASS' if passed else 'FAIL'}; "
        f"windows={len(window_records)}, "
        f"posterior={decisions[-1].best_probability:.6f}, "
        f"graph_error={report['factor']['horizontal_error_m']:.3f} m, "
        f"elapsed={report['elapsed_s']:.2f} s; report={args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
