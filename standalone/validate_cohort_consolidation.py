#!/usr/bin/env python3
"""Audit archived mode fragmentation and the disabled-policy regression.

Archived posterior regrouping is a diagnostic counterfactual, not a causal
navigation replay or a new RMSE result. No archived truth enters the tracker.
"""
from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archives', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    result = dict(disabled_policy_regression=frozen_default_regression(),
                  archive_diagnostics=[audit_archive(path) for path in args.archives])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
