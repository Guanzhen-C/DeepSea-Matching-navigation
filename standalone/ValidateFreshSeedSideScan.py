"""Predeclare one fresh-noise validation, serially after an exact live review.

Uses a frozen ORIGINAL runtime to create/record a new baseline, then replays
those exact observations with a separately frozen candidate and audits both.
Never edits/restarts navigation, changes parameters, uploads data, or marks
the broader improvement goal complete. A short recording remains explicitly
supplementary to the main full16.49km validation, not its replacement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from FinalizePureSideScanExperiment import service_state, sha


def require_primary_improvement(d):
    if (d.get('status') != 'completed_independent_full_identical_input_mapped_range_comparison'
            or d.get('samples') != 62458 or not d.get('all_received_events_verified')
            or not d.get('DR_and_Local_bit_exact') or not d.get('full_original_input_file_SHA256_reverified')):
        raise ValueError('Complete independently audited full16km primary required')
    a, b = d['candidate_Global'], d['baseline_Global']
    if not (a['RMSE_m'] < b['RMSE_m'] and a['P95_m'] <= b['P95_m'] and a['max_m'] <= b['max_m']):
        raise ValueError('Primary mean or tail result needs further diagnosis; do not consume the fresh seed yet')


def save(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-unit', required=True)
    p.add_argument('--review-pid', required=True, type=int)
    for name in ['primary-comparison', 'source-runtime', 'candidate-runtime', 'analysis-runtime',
                 'experiment', 'out', 'status', 'project-root']:
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--seed', type=int, default=33)
    p.add_argument('--distance-m', type=float, default=4000.)
    a = p.parse_args()
    s = service_state(a.review_unit)
    if (a.out.exists() or a.status.exists() or a.review_pid <= 1 or a.seed < 0 or a.distance_m <= 0
            or int(s['MainPID']) != a.review_pid or s['SubState'] != 'running'):
        raise ValueError('One confirmed live review and new predeclared fresh-test outputs required')
    frozen = {}
    for directory in [a.source_runtime, a.candidate_runtime, a.analysis_runtime]:
        manifest = json.loads((directory.parent/'runtime_manifest.json').read_text())
        for name, expected in manifest['files_SHA256'].items():
            if sha(directory/name) != expected: raise ValueError('Staged source changed before declaration')
            frozen[str(directory/name)] = expected
    frozen[str(Path(__file__).resolve())] = sha(Path(__file__))
    common = ['--experiment', str(a.experiment)]
    commands = [
        [sys.executable, str(a.source_runtime/'RunPureSideScanFeatureShort.py'), *common,
         '--out', str(a.out/'source_baseline'), '--distance-m', str(a.distance_m), '--seed', str(a.seed),
         '--observations', 'complex_IQ', '--feature-matcher', 'LoFTR',
         '--submap-representation', 'adaptive_projected', '--loop-closure'],
        [sys.executable, str(a.candidate_runtime/'ReplayPureSideScanNavigation.py'), *common,
         '--run', str(a.out/'source_baseline'), '--out', str(a.out/'candidate'), '--runtime', 'current',
         '--submap-representation', 'adaptive_projected', '--loop-closure',
         '--mapped-range-changes', '--mapped-range-policy', 'preserve_legacy'],
        [sys.executable, str(a.analysis_runtime/'AnalyzeMappedRangeNavigation.py'),
         '--source-run', str(a.out/'source_baseline'), '--candidate', str(a.out/'candidate'),
         '--out', str(a.out/'review'), '--heldout-recording'],
    ]
    report = dict(status='waiting_for_exact_primary_review_before_predeclared_fresh_seed',
        review_unit=a.review_unit, review_PID=a.review_pid, seed=a.seed, requested_distance_m=a.distance_m,
        commands=commands, frozen_source_SHA256=frozen,
        parameters_frozen_before_fresh_recording=True, source_and_candidate_run_serially=True,
        same_measured_source_scene_not_independent_voyage=True,
        fresh_noise_not_new_geographic_generalization=True, navigation_restart_allowed=False,
        external_upload_allowed=False, broader_goal_completion_not_automatic=True)
    save(a.status, report)
    try:
        while True:
            s = service_state(a.review_unit)
            if int(s['MainPID']) not in (0, a.review_pid):
                raise RuntimeError('Exact primary review changed; do not follow another process')
            if int(s['MainPID']) == 0:
                if int(s['ExecMainStatus']) != 0: raise RuntimeError('Primary review failed; fresh seed not consumed')
                break
            time.sleep(5.)
        require_primary_improvement(json.loads(a.primary_comparison.read_text()))
        a.out.mkdir(parents=True)
        for step, command in enumerate(commands, 1):
            if any(sha(Path(name)) != expected for name, expected in frozen.items()):
                raise RuntimeError('Frozen sources changed; never tune on the fresh recording')
            with (a.out/f'step_{step}.log').open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=a.project_root)
                report.update(status='running_serial_fresh_seed_validation', step=step, child_PID=process.pid)
                save(a.status, report)
                if process.wait() != 0: raise RuntimeError(f'Fresh validation step{step} failed; preserve artifacts')
            report['completed_steps'] = step; save(a.status, report)
        if any(sha(Path(name)) != expected for name, expected in frozen.items()):
            raise RuntimeError('Frozen sources changed during fresh evaluation')
        report.update(status='fresh_seed_source_replay_and_independent_review_complete_pending_result_review',
                      result_directory=str((a.out/'review').resolve()), external_upload_performed=False)
        save(a.status, report)
    except BaseException as exc:
        report.update(status='failed_fresh_seed_validation_preserved_no_restart',
                      error=f'{type(exc).__name__}: {exc}')
        save(a.status, report)
        raise


if __name__ == '__main__': main()
