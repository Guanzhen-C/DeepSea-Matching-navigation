#!/usr/bin/env python3
"""Causal PF-only paired replay of recorded raw sonar/odometry windows.

Both policies receive identical recorded local odometry, sonar, and external
compact proposals. Persistent-cohort refinements are regenerated causally from
each policy's own predictions. No graph or simulator is started. Proposals and
deskew come from the recording run; this isolates PF data association and is
NOT an end-to-end navigation A/B. Truth is used only for scoring after update.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'isaacsim'))
from oceansim.slam import (
    AcousticTerrainParticleLikelihood, GlobalCohortTracker, GlobalCohortTrackerConfig,
    GlobalPoseEvidence, InertialState, MeasuredBathymetryGrid, ParticleLikelihoodObservation,
    TerrainParticleLikelihoodConfig, reporting_covariance_from_information,
    sanitize_for_strict_json,
)
from oceansim.slam.raw_mode_recovery import RawModeRecovery


def current_evidence(approximation, record, config, source_index, cohort_id=None):
    d = approximation.diagnostics
    covariance = approximation.covariance
    return GlobalPoseEvidence(
        map_pose2=approximation.pose2,
        bathymetry_score=float(np.tanh((config.bathymetry_reference_normalized_loss
            - d.best_bathymetry_normalized_loss) / config.bathymetry_fit_temperature))
            if np.isfinite(d.best_bathymetry_normalized_loss) else 0.,
        backscatter_score=float(np.clip(d.best_backscatter_correlation, -1., 1.))
            if np.isfinite(d.best_backscatter_correlation) else 0.,
        bathymetry_observability=d.bathymetry_observability,
        backscatter_observability=d.backscatter_observability,
        position_std_m=float(np.sqrt(np.linalg.eigvalsh(covariance[:2, :2])[-1])),
        yaw_std_deg=float(np.rad2deg(np.sqrt(covariance[2, 2]))),
        covariance_matrix=covariance, information_matrix=approximation.information,
        target_cohort_id=cohort_id, source_index=source_index,
        **{key: record[key] for key in ('patch_start_timestamp_s', 'patch_end_timestamp_s',
                                      'patch_start_observation_id', 'patch_end_observation_id')},
    )


def replay(directory, terrain, prior, maximum_windows=None, *, compare_raw_recovery=False):
    files = sorted(directory.glob('window_*.npz'))
    if maximum_windows is not None:
        files = files[:maximum_windows]
    if not files:
        raise ValueError('no recorded sonar windows')
    trackers = {}
    recoveries = {}
    rows = []
    reproduction_errors = []
    started = time.monotonic()
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            record = json.loads(str(data['record_json']))
            if not trackers:
                profiles = (('baseline', True, False), ('raw_recovery', True, True)) if compare_raw_recovery else (
                    ('baseline', False, False), ('consolidation', True, False))
                for name, enabled, recovery_enabled in profiles:
                    config = GlobalCohortTrackerConfig(**dict(
                        record['tracker_config'], merge_converged_cohorts=enabled))
                    trackers[name] = GlobalCohortTracker(config)
                    recoveries[name] = RawModeRecovery(enabled=recovery_enabled)
            config = TerrainParticleLikelihoodConfig(**record['likelihood_config'])
            raw = AcousticTerrainParticleLikelihood(
                prior, InertialState(position=data['reference_position'], rotation=data['reference_rotation']),
                data['bathymetry_points_body'], backscatter_terrain=terrain,
                backscatter_points_body=data['backscatter_points_body'],
                backscatter_values=data['backscatter_values'], config=config,
            )
        observation = ParticleLikelihoodObservation(evaluator=raw,
            **{key: record[key] for key in ('patch_start_timestamp_s', 'patch_end_timestamp_s',
                                          'patch_start_observation_id', 'patch_end_observation_id')})
        local = np.array(record['local_pose2'])
        for name, tracker in trackers.items():
            evidences = []
            for cohort_id, predicted in tracker.prediction_seeds(local):
                approx = raw.approximate_local_measurement(predicted)
                evidences.append(current_evidence(approx, record, config, len(evidences), cohort_id))
            for payload in record['compact_evidences']:
                # Recreate the same external proposal, then find its raw mode.
                original = GlobalPoseEvidence(**payload)
                pose = raw.refine_pose(original.map_pose2)
                delta = pose - original.map_pose2
                delta[2] = (delta[2] + np.pi) % (2 * np.pi) - np.pi
                from oceansim.slam import information_with_covariance_floor
                info = information_with_covariance_floor(original.information, np.outer(delta, delta))
                covariance = reporting_covariance_from_information(info,
                    yaw_scale_m=config.yaw_rank_scale_m,
                    unobservable_variance_scaled_m2=config.unobservable_variance_scaled_m2)
                evidences.append(replace(original, map_pose2=pose, covariance_matrix=covariance,
                    information_matrix=info, position_std_m=float(np.sqrt(np.linalg.eigvalsh(covariance[:2,:2])[-1])),
                    yaw_std_deg=float(np.rad2deg(np.sqrt(covariance[2,2]))),
                    source_index=len(evidences), target_cohort_id=None))
            evidences.extend(recoveries[name].generate(raw, tracker.prediction_seeds(local),
                observation, source_index=len(evidences)))
            decision = tracker.update(timestamp_s=record['timestamp_s'], raw_pose2=local,
                                      evidences=evidences, particle_observation=observation)
            recoveries[name].observe(decision)
            truth = record.get('truth_evaluation_only')
            best = decision.cohorts[0] if decision.cohorts else None
            if (tracker.config.merge_converged_cohorts == record['tracker_config']['merge_converged_cohorts']
                    and recoveries[name].enabled == record.get('raw_mode_recovery', {}).get('enabled', False)):
                recorded = record['decision']
                if decision.status != recorded['status']:
                    raise AssertionError('baseline PF replay does not reproduce the recording status')
                probability_error = abs(decision.best_probability - recorded['best_probability'])
                pose_error = 0. if best is None else float(np.max(np.abs(
                    best.map_pose2 - np.array(recorded['best_map_pose2']))))
                if probability_error > 1.e-8 or pose_error > 1.e-7:
                    raise AssertionError(f'baseline replay differs: probability {probability_error}, pose {pose_error}')
                reproduction_errors.append((probability_error, pose_error))
            error = None if truth is None or best is None else float(np.linalg.norm(
                best.map_pose2[:2] - np.array(truth['position'])[:2]))
            row = dict(profile=name, timestamp_s=record['timestamp_s'], status=decision.status,
                probability=decision.best_probability, null_probability=decision.null_probability,
                odds=decision.odds_over_second, cohorts=len(decision.cohorts), error_xy_m=error,
                graph_ready=bool(decision.confirmed_cohort and decision.confirmed_cohort.last_measurement_is_graph_ready),
                merged=tracker.converged_cohorts_merged,
                raw_recovery=recoveries[name].diagnostics())
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for name, tracker in trackers.items():
        subset = [row for row in rows if row['profile'] == name]
        errors = [row['error_xy_m'] for row in subset if row['error_xy_m'] is not None]
        summary[name] = dict(windows=len(subset), confirmed=sum(row['status']=='confirmed' for row in subset),
            graph_ready=sum(row['graph_ready'] for row in subset),
            rmse_xy_m=float(np.sqrt(np.mean(np.square(errors)))),
            p95_xy_m=float(np.percentile(errors, 95)), max_xy_m=max(errors),
            confirmed_above_15m=sum(row['status']=='confirmed' and row['error_xy_m'] is not None
                                    and row['error_xy_m'] > 15 for row in subset),
            diagnostics=tracker.diagnostics())
    return dict(protocol=__doc__, directory=str(directory), elapsed_s=time.monotonic()-started,
                first_timestamp_s=rows[0]['timestamp_s'], last_timestamp_s=rows[-1]['timestamp_s'],
                recording_reproduction=dict(windows=len(reproduction_errors),
                    maximum_probability_difference=max((v[0] for v in reproduction_errors), default=None),
                    maximum_pose_component_difference=max((v[1] for v in reproduction_errors), default=None)),
                summary=summary, records=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--bathymetry', type=Path, default=ROOT/'data/underwater_terrain_datasets/lake_chelan_usgs_2023_3m/prepared_db_power')
    p.add_argument('--maximum-windows', type=int)
    p.add_argument('--compare-raw-recovery', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    terrain = MeasuredBathymetryGrid.from_path(args.bathymetry)
    prior = MeasuredBathymetryGrid(
        terrain.navigation_elevation_m, resolution_m=terrain.resolution_m,
        navigation_elevation_m=terrain.navigation_elevation_m,
        valid_mask=terrain.valid_mask, metadata=terrain.metadata,
        source_path=terrain.source_path,
    )
    result = replay(args.directory, terrain, prior, args.maximum_windows,
                    compare_raw_recovery=args.compare_raw_recovery)
    safe, _ = sanitize_for_strict_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(safe['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
