#!/usr/bin/env python3
"""Compare sonar-window apertures on recorded data without changing a run.

This is a local-likelihood diagnostic, NOT a navigation A/B. Compose only
recorded local-odometry reference frames (no truth deskew). All apertures use
the same final posterior seed, final estimated roll/pitch/depth, 768-point
budgets and effective-independent-sample caps. Historical windows were already
deskewed/voxelized by the recording estimator, so this cannot replace a raw-ping
long-window replay. Truth is read only after estimation to calculate errors.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'isaacsim'))
from oceansim.slam import (
    AcousticTerrainParticleLikelihood, InertialState, MeasuredBathymetryGrid,
    TerrainParticleLikelihoodConfig, sanitize_for_strict_json,
)


def transform_between_body_frames(points, old_position, old_rotation, reference):
    """Preserve estimated world XYZ; do not substitute simulated truth."""
    return ((points @ old_rotation.T + old_position) - reference.position) @ reference.rotation


def diagnose(directory: Path, terrain, end_times, window_counts, *, refinement='basin'):
    wanted = {float(end - 30 * i) for end in end_times for i in range(max(window_counts))}
    metadata = {}
    with (directory/'decisions.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['timestamp_s'] in wanted:
                metadata[record['timestamp_s']] = record
    missing = wanted - metadata.keys()
    if missing:
        raise ValueError(f'missing complete windows at {sorted(missing)}')
    prior = MeasuredBathymetryGrid(
        terrain.navigation_elevation_m, resolution_m=terrain.resolution_m,
        navigation_elevation_m=terrain.navigation_elevation_m,
        valid_mask=terrain.valid_mask, metadata=terrain.metadata,
        source_path=terrain.source_path,
    )
    rows = []
    for end in end_times:
        last = metadata[float(end)]
        if last['decision']['best_map_pose2'] is None:
            raise ValueError(f'no posterior seed at {end}; truth cannot replace it')
        seed = np.array(last['decision']['best_map_pose2'], dtype=float)
        with np.load(directory/f"window_{last['observation_id']:08d}.npz", allow_pickle=False) as data:
            reference = InertialState(position=data['reference_position'], rotation=data['reference_rotation'])
        config = TerrainParticleLikelihoodConfig(**last['likelihood_config'])
        for count in window_counts:
            geometry, appearance, values = [], [], []
            for i in reversed(range(count)):
                record = metadata[float(end - 30 * i)]
                with np.load(directory/f"window_{record['observation_id']:08d}.npz", allow_pickle=False) as data:
                    old_p, old_r = data['reference_position'], data['reference_rotation']
                    geometry.append(transform_between_body_frames(
                        data['bathymetry_points_body'], old_p, old_r, reference))
                    appearance.append(transform_between_body_frames(
                        data['backscatter_points_body'], old_p, old_r, reference))
                    values.append(data['backscatter_values'])
            raw = AcousticTerrainParticleLikelihood(
                prior, reference, np.vstack(geometry), backscatter_terrain=terrain,
                backscatter_points_body=np.vstack(appearance),
                backscatter_values=np.concatenate(values), config=config,
            )
            search_seed = raw.refine_pose(seed) if refinement == 'proposal' else seed
            approximation = raw.approximate_local_measurement(search_seed)
            truth = last['truth_evaluation_only']
            truth_p = np.array(truth['position'])
            truth_r = np.array(truth['rotation'])
            truth_yaw = math.atan2(truth_r[1, 0], truth_r[0, 0])
            yaw_error = approximation.pose2[2] - truth_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
            row = dict(
                end_timestamp_s=end, window_count=count, nominal_aperture_s=count * 30,
                approximate_route_distance_km=end * 1.5/1000,
                seed_pose2=seed.tolist(), refined_pose2=approximation.pose2.tolist(),
                refinement=refinement, local_search_seed_pose2=np.asarray(search_seed).tolist(),
                observable_rank=approximation.observable_rank,
                information_eigenvalues=approximation.scaled_information_eigenvalues.tolist(),
                horizontal_error_m=float(np.linalg.norm(approximation.pose2[:2]-truth_p[:2])),
                yaw_error_deg=math.degrees(yaw_error),
                current_window_recorded_rank=last['decision']['best_last_measurement_observable_rank'],
                diagnostics=asdict(approximation.diagnostics),
            )
            rows.append(row)
            print(json.dumps(sanitize_for_strict_json(row)[0], ensure_ascii=False), flush=True)
    return dict(protocol=__doc__, diagnostic_only=True,
                seed_provenance='recorded post-update PF MAP; not the original pre-update prediction, so diagnostic 30 s ranks can differ from recorded ranks',
                refinement=refinement, proposed_graph_fix_authorized=False,
                no_particle_tracking_or_graph_was_run=True,
                original_run_unmodified=True, truth_used_by_estimator=False,
                directory=str(directory), records=rows)


def audit_output_sources(directory: Path):
    """Evaluate a predetermined source rule at window times, not full rate.

    Does not choose sources using truth. The counterfactual rule is always:
    use the confirmed PF position for ONE ordinary cohort, otherwise retain
    the recorded published output. Ignores soft-output transient dynamics and
    future changes of proposal seeds; it is not a closed-loop A/B.
    """
    records = {}
    with (directory/'decisions.jsonl').open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                continue
            record = json.loads(line)
            records[round(record['timestamp_s'], 6)] = record
    rows = []
    with (directory/'online_poses_5hz.csv').open(newline='') as stream:
        for sample in csv.DictReader(stream):
            if any(sample.get(key) is None for key in ('timestamp_s','global_x_m','global_y_m')):
                continue
            key = round(float(sample['timestamp_s']), 6)
            if key not in records:
                continue
            record = records[key]
            decision = record['decision']
            truth = np.array(record['truth_evaluation_only']['position'])[:2]
            csv_truth = np.array([float(sample['truth_x_m']), float(sample['truth_y_m'])])
            np.testing.assert_allclose(csv_truth, truth, atol=1.e-8, rtol=0.)
            online = np.array([float(sample['global_x_m']), float(sample['global_y_m'])])
            pf = None if decision['best_map_pose2'] is None else np.array(decision['best_map_pose2'])[:2]
            eligible = decision['status'] == 'confirmed' and decision['cohort_count'] == 1
            online_error = float(np.linalg.norm(online - truth))
            pf_error = None if pf is None else float(np.linalg.norm(pf - truth))
            rows.append(dict(timestamp_s=key, travelled_distance_m=float(sample['travelled_distance_m']),
                recorded_online_error_xy_m=online_error, confirmed_single_cohort=eligible,
                pf_error_xy_m=pf_error,
                counterfactual_error_xy_m=pf_error if eligible else online_error,
                pf_status=decision['status'], observable_rank=decision['best_last_measurement_observable_rank']))
    if not rows:
        raise ValueError('no synchronized window times in the live pose journal')
    def summary(values):
        values = np.array(values, dtype=float)
        return dict(samples=len(values), rmse_xy_m=float(np.sqrt(np.mean(values**2))),
                    p95_xy_m=float(np.percentile(values,95)), max_xy_m=float(np.max(values)))
    return dict(scope='window-time output-source diagnostic only; not full-rate or closed-loop RMSE',
        rule='confirmed ordinary PF even for one cohort; otherwise recorded output',
        truth_used_for_source_selection=False, soft_output_transients_simulated=False,
        future_proposal_feedback_simulated=False, source_run_unmodified=True,
        first_timestamp_s=rows[0]['timestamp_s'], last_timestamp_s=rows[-1]['timestamp_s'],
        eligible_windows=sum(row['confirmed_single_cohort'] for row in rows),
        recorded_online=summary([row['recorded_online_error_xy_m'] for row in rows]),
        counterfactual=summary([row['counterfactual_error_xy_m'] for row in rows]),
        records=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--bathymetry', type=Path, default=ROOT/'data/underwater_terrain_datasets/lake_chelan_usgs_2023_3m/prepared_db_power')
    parser.add_argument('--end-times-s', type=float, nargs='+', default=[26370., 29520., 29640., 29970., 30120.])
    parser.add_argument('--windows', type=int, nargs='+', default=[1,2,4])
    parser.add_argument('--refinement', choices=['basin','proposal'], default='basin')
    parser.add_argument('--audit-output-source', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.windows or any(n < 1 for n in args.windows):
        parser.error('window counts must be positive')
    if args.audit_output_source:
        result = audit_output_sources(args.directory)
    else:
        world = MeasuredBathymetryGrid.from_path(args.bathymetry)
        result = diagnose(args.directory, world, args.end_times_s, args.windows,
                          refinement=args.refinement)
    result, _ = sanitize_for_strict_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
