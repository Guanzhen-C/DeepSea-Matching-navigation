#!/usr/bin/env python3
"""Audit archived mode fragmentation and the disabled-policy regression.

Archived posterior regrouping is a diagnostic counterfactual, not a causal
navigation replay or a new RMSE result. No archived truth enters the tracker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'isaacsim'))
from oceansim.slam import GlobalCohortTracker


def frozen_default_regression():
    """Compare to pre-edit source, including stochastic particle arrays."""
    name = 'oceansim.slam._frozen_tracker_96edafc'
    frozen = types.ModuleType(name)
    sys.modules[name] = frozen
    source = subprocess.check_output([
        'git', 'show', '96edafc:isaacsim/oceansim/slam/global_tracking.py'
    ], cwd=ROOT, text=True)
    exec(compile(source, 'frozen_96edafc_global_tracking.py', 'exec'), frozen.__dict__)
    trackers = [frozen.GlobalCohortTracker(), GlobalCohortTracker()]
    modules = [frozen, sys.modules[GlobalCohortTracker.__module__]]
    for epoch in range(1, 65):
        target = np.array([epoch * 10., math.sin(epoch / 3.) * 3, .02 * epoch])

        def likelihood(particles):
            residual = particles - target
            residual[:, 2] = (residual[:, 2] + np.pi) % (2 * np.pi) - np.pi
            return 3. - .5 * np.sum(residual**2 / [4.**2, 4.**2, .15**2], axis=1)

        decisions = []
        for tracker, module in zip(trackers, modules):
            evidences = [module.GlobalPoseEvidence(
                map_pose2=target + [dx, 0, 0], bathymetry_score=.8,
                backscatter_score=.7, source_index=i,
                patch_start_timestamp_s=epoch * 30. - 29.8,
                patch_end_timestamp_s=epoch * 30.,
                patch_start_observation_id=epoch * 150 - 149,
                patch_end_observation_id=epoch * 150,
            ) for i, dx in enumerate((0., 35., 100.))] if epoch % 5 else []
            observation = module.ParticleLikelihoodObservation(
                evaluator=likelihood,
                patch_start_timestamp_s=epoch * 30. - 29.8,
                patch_end_timestamp_s=epoch * 30.,
                patch_start_observation_id=epoch * 150 - 149,
                patch_end_observation_id=epoch * 150,
            )
            decisions.append(tracker.update(
                timestamp_s=epoch * 30., raw_pose2=target,
                evidences=evidences, particle_observation=observation,
            ))
        assert decisions[0].status == decisions[1].status
        assert decisions[0].best_probability == decisions[1].best_probability
        assert decisions[0].null_probability == decisions[1].null_probability
        assert len(trackers[0]._cohorts) == len(trackers[1]._cohorts)
        for old, current in zip(trackers[0]._cohorts, trackers[1]._cohorts):
            assert old.cohort_id == current.cohort_id
            np.testing.assert_array_equal(old.particles, current.particles)
            np.testing.assert_array_equal(old.log_weights, current.log_weights)
            np.testing.assert_array_equal(old.last_measurement_information, current.last_measurement_information)
        for key, value in trackers[0].diagnostics().items():
            assert trackers[1].diagnostics()[key] == value, key
    return dict(reference_commit='96edafc', epochs=64,
                bitwise_particle_weights_information_and_decisions_equal=True,
                merge_enabled_by_default=False,
                monterey_launcher_changed=False)


def audit_archive(path):
    report = json.loads(path.read_text())
    records = report['terrain_cohort_particle_backend']['records']
    counts = dict(windows=len(records), originally_confirmed=0,
                  spatially_fragmented=0, grouped_mass_above_097=0)
    examples = []
    for record in records:
        counts['originally_confirmed'] += record['status'] == 'confirmed'
        remaining = sorted(record['cohorts'], key=lambda c: -c['probability'])
        groups = []
        def close(a, b):
            delta = np.array(a['map_pose2']) - np.array(b['map_pose2'])
            yaw = abs((delta[2] + np.pi) % (2 * np.pi) - np.pi)
            return np.linalg.norm(delta[:2]) <= 10. and yaw <= np.deg2rad(2.)
        while remaining:
            group = [remaining.pop(0)]
            others = []
            for candidate in remaining:
                if all(close(member, candidate) for member in group):
                    group.append(candidate)
                else:
                    others.append(candidate)
            groups.append(group)
            remaining = others
        groups.sort(key=lambda group: -sum(c['probability'] for c in group))
        if not groups:
            continue
        best = groups[0]
        mass = sum(c['probability'] for c in best)
        counts['spatially_fragmented'] += len(best) > 1
        if len(best) > 1 and mass >= .97 and record['status'] != 'confirmed':
            counts['grouped_mass_above_097'] += 1
            if len(examples) < 8:
                examples.append(dict(timestamp_s=record['timestamp_s'],
                    best_single_probability=record['best_probability'],
                    null_probability=record['null_probability'],
                    group_mass=mass, group_size=len(best),
                    max_pairwise_position_m=max(np.linalg.norm(
                        np.array(a['map_pose2'])[:2] - np.array(b['map_pose2'])[:2]
                    ) for a in best for b in best)))
    return dict(file=str(path), counts=counts, examples=examples,
                scope='posterior-centre audit only; lacks complete particle covariance and current-window modes; not accepted fixes or a causal replay')


def native_pair(directory):
    reports = {name: json.loads((directory / filename / 'metrics.json').read_text())
               for name, filename in (
                   ('baseline', 'native_baseline_6500_8300m'),
                   ('consolidation', 'native_consolidation_6500_8300m'))}
    fields = (
        'ping_rate_hz', 'survey_speed_mps', 'survey_altitude_m',
        'survey_start_distance_m', 'requested_survey_segment_distance_m',
        'terrain_resolution_m', 'terrain_match_window_pings',
        'terrain_match_translation_window_m', 'terrain_match_yaw_window_deg',
        'feature_subframe_pings', 'sidescan_imaging_rays_per_side_per_row',
        'sidescan_slam_sample_stride', 'pose_graph_backend_configuration',
        'navigation_initialization', 'terrain_prior_surface',
        'reproducibility',
    )
    for field in fields:
        assert field in reports['baseline'], field
        assert reports['baseline'][field] == reports['consolidation'][field], field
    arrays = {}
    result = dict(scope='1.8 km fresh-start native pair at Chelan route 6.5--8.3 km; not a 47 km result',
                  identical_configuration_fields=list(fields), results={})
    for name, filename in (('baseline', 'native_baseline_6500_8300m'),
                           ('consolidation', 'native_consolidation_6500_8300m')):
        report = reports[name]
        assert report['run_status'] == 'completed'
        assert abs(report['survey_travelled_distance_m'] - 1800) < .5
        arrays[name] = np.loadtxt(directory / filename / 'metrics_position_errors.csv',
                                  skiprows=1, delimiter=',')
        result['results'][name] = dict(
            distance_m=report['survey_travelled_distance_m'],
            online_global=report['position_error_statistics']['online_global_slam'],
            local=report['position_error_statistics']['local_acoustic_esikf'],
            accepted_absolute_priors=report['accepted_terrain_prior_factors'],
            confirmed_windows=report['terrain_cohort_particle_backend']['statistics']['confirmed_updates'],
            maximum_failed_observations=report['terrain_maximum_consecutive_failed_prior_observations'],
            diagnostics=report['terrain_cohort_particle_backend']['diagnostics'])
    # Time + DR + Local must be unchanged when editing only the backend.
    np.testing.assert_allclose(arrays['baseline'][:, :3], arrays['consolidation'][:, :3],
                               atol=1.e-9, rtol=0.)
    result['identical_timestamps_dr_and_local_at_all_5hz_samples'] = True
    a = result['results']['baseline']['online_global']['rmse_3d_m']
    b = result['results']['consolidation']['online_global']['rmse_3d_m']
    result['online_global_rmse_reduction_percent'] = 100. * (1. - b/a)
    return result


def verify_frozen_monterey():
    frozen = Path('/home/cgz/OceanSim_worktrees/monterey-72km-stable-20260918')
    expected = '8b5df53cb57f9de2506d78d90b16e33bf5e96510'
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=frozen, text=True).strip()
    assert actual == expected
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                     cwd=frozen, text=True).strip()
    assert not status, 'frozen tracked files were modified'
    route_name = 'data/monterey_canyon_full_5m/full_map_traversal_route.json'
    route_bytes = (ROOT / route_name).read_bytes()
    assert route_bytes == (frozen / route_name).read_bytes()
    checksums = subprocess.check_output(['sha256sum', '-c', 'sha256sums.txt'],
                                      cwd=ROOT/'data/monterey_canyon_full_5m', text=True)
    return dict(commit=actual, worktree=str(frozen), tracked_source_clean=True,
                original_route_unchanged=True,
                route_sha256=hashlib.sha256(route_bytes).hexdigest(),
                measured_map_checksums=checksums.strip().splitlines(),
                frozen_policy='multimodal-map', experimental_default_merge_enabled=False,
                scope='code/data isolation and disabled-policy numeric regression; no new 72 km run claimed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archives', type=Path, nargs='*', default=[])
    parser.add_argument('--native-pair', type=Path)
    parser.add_argument('--verify-frozen-monterey', action='store_true')
    args = parser.parse_args()
    result = (native_pair(args.native_pair) if args.native_pair else
              dict(disabled_policy_regression=frozen_default_regression(),
                   archive_diagnostics=[audit_archive(path) for path in args.archives]))
    if args.verify_frozen_monterey:
        result['frozen_monterey_regression'] = verify_frozen_monterey()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
