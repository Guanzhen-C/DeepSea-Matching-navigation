#!/usr/bin/env python3
"""Summarize causal PF replay and matched native tests, never conflate them.

Truth enters metrics only, after the two policies have run. Fresh-start segment
tests are NOT reproductions of the estimator drift accumulated over 44 km.
The new full-route closed-loop experiment is still required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def statistics(rows):
    errors = np.array([row['error_xy_m'] for row in rows if row['error_xy_m'] is not None])
    return dict(windows=len(rows), confirmed=sum(row['status'] == 'confirmed' for row in rows),
        graph_ready=sum(row['graph_ready'] for row in rows),
        rmse_xy_m=float(np.sqrt(np.mean(errors**2))),
        p95_xy_m=float(np.percentile(errors, 95)), max_xy_m=float(errors.max()),
        confirmed_above_15m=sum(row['status'] == 'confirmed'
            and row['error_xy_m'] is not None and row['error_xy_m'] > 15 for row in rows))


def validate_replay(path):
    report = json.loads(path.read_text())
    baseline = [row for row in report['records'] if row['profile'] == 'baseline']
    candidate = [row for row in report['records'] if row['profile'] == 'raw_recovery']
    assert baseline and len(baseline) == len(candidate)
    assert report['recording_reproduction']['windows'] == len(baseline)
    np.testing.assert_array_equal([r['timestamp_s'] for r in baseline],
                                  [r['timestamp_s'] for r in candidate])
    intervals = {}
    for name, start_km, end_km in (('all', 0., 100.), ('from_37km', 37., 100.),
                                  ('from_44km', 44., 100.)):
        intervals[name] = {profile: statistics([r for r in rows
            if start_km <= r['timestamp_s'] * 1.5 / 1000 < end_km])
            for profile, rows in (('baseline', baseline), ('raw_recovery', candidate))}
    return dict(file=str(path), scope='PF-only same-input causal replay, not online Global error',
        recording_reproduction=report['recording_reproduction'], intervals=intervals,
        raw_recovery=candidate[-1]['raw_recovery'], truth_used_for_estimation=False)


def validate_native_pair(baseline_dir, candidate_dir):
    reports = {name: json.loads((directory / 'metrics.json').read_text())
               for name, directory in (('baseline', baseline_dir), ('raw_recovery', candidate_dir))}
    fields = ('ping_rate_hz', 'survey_speed_mps', 'survey_altitude_m',
        'survey_start_distance_m', 'requested_survey_segment_distance_m',
        'terrain_resolution_m', 'terrain_match_window_pings',
        'terrain_match_translation_window_m', 'terrain_match_yaw_window_deg',
        'feature_subframe_pings', 'sidescan_imaging_rays_per_side_per_row',
        'sidescan_slam_sample_stride', 'pose_graph_backend_configuration',
        'navigation_initialization', 'terrain_prior_surface', 'reproducibility')
    for field in fields:
        assert reports['baseline'][field] == reports['raw_recovery'][field], field
    arrays = {}
    result = {}
    for name, directory in (('baseline', baseline_dir), ('raw_recovery', candidate_dir)):
        report = reports[name]
        assert report['run_status'] == 'completed'
        assert abs(report['survey_travelled_distance_m']
                   - report['requested_survey_segment_distance_m']) < .5
        arrays[name] = np.loadtxt(directory / 'metrics_position_errors.csv', skiprows=1, delimiter=',')
        result[name] = dict(distance_m=report['survey_travelled_distance_m'],
            online_global=report['position_error_statistics']['online_global_slam'],
            raw_mode_recovery=report['terrain_cohort_particle_backend']['raw_mode_recovery'],
            accepted_absolute_priors=report['accepted_terrain_prior_factors'])
    # All synchronized times, DR and Local errors must be unchanged by a
    # backend-only implementation. Compare full poses as an additional check.
    np.testing.assert_allclose(arrays['baseline'][:, :3], arrays['raw_recovery'][:, :3],
                               atol=1.e-9, rtol=0.)
    a = np.loadtxt(baseline_dir/'terrain_replay/online_poses_5hz.csv', delimiter=',', skiprows=1)
    b = np.loadtxt(candidate_dir/'terrain_replay/online_poses_5hz.csv', delimiter=',', skiprows=1)
    np.testing.assert_allclose(a[:, :11], b[:, :11], atol=1.e-9, rtol=0.)
    return dict(scope='matched fresh-start segment, not a resumed 44 km history',
        results=result, identical_fields=list(fields), identical_time_truth_dr_local=True,
        samples=int(len(a)), global_metric_source='online published pose, not optimized history')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path)
    p.add_argument('--native-baseline', type=Path)
    p.add_argument('--native-candidate', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = dict(protocol=__doc__)
    if args.replay:
        result['replay'] = validate_replay(args.replay)
    if args.native_baseline or args.native_candidate:
        if not (args.native_baseline and args.native_candidate):
            p.error('both native pair directories are required')
        result['native_pair'] = validate_native_pair(args.native_baseline, args.native_candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
