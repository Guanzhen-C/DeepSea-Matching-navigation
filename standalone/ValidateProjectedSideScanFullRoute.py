"""Wait for one exact live run, then audit a full, same-input 2x2 ablation.

This observer never stops/restarts the simulator and never accepts a partial
archive. Replay uses a separate hashed runtime, not mutable workspace code.
The matrix distinguishes representation gains from the effect of loops.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from FinalizePureSideScanExperiment import service_state, sha
from WriteSideScanReport import dump


VARIANTS = (
    ('projected_loops', 'adaptive_projected', True),
    ('projected_no_loops', 'adaptive_projected', False),
    ('legacy_loops_control', 'legacy_strip', True),
    ('legacy_no_loops', 'legacy_strip', False),
)
COMPARISONS = (
    ('projection_with_loops', 'legacy_loops_control', 'projected_loops', 'projection'),
    ('projection_without_loops', 'legacy_no_loops', 'projected_no_loops', 'projection'),
    ('loops_with_projection', 'projected_no_loops', 'projected_loops', 'loop'),
    ('loops_without_projection', 'legacy_no_loops', 'legacy_loops_control', 'loop'),
)


def verify_target(state, pid, *, initially_live=False):
    if pid <= 1: raise ValueError('An exact real process PID is required.')
    current = int(state.get('MainPID', -1))
    if current not in (0, pid): raise RuntimeError('Target disappeared or PID changed; never attach to another run.')
    if initially_live and (current != pid or state.get('SubState') != 'running'):
        raise RuntimeError('Confirm the exact target is live before starting an observer.')
    if current == 0 and int(state.get('ExecMainStatus', -1)) != 0:
        raise RuntimeError('Source process failed; preserving results, without a restart.')
    return current == 0


def validate_full_source(report):
    if (report.get('status') != 'completed_pure_side_scan_FEATURE_route_test'
            or not report.get('full_closed_loop_requested')
            or not report.get('full_source_snapshots_verified')
            or not report.get('recorded_navigation_inputs')
            or not report.get('historical_side_scan_loops_requested')
            or report.get('submap_representation', 'legacy_strip') != 'legacy_strip'
            or report.get('scan_plane_registration') != 'rank2_baseline'
            or report.get('secondary_denoising_used')
            or report.get('event_counts') != report.get('expected_events')
            or report.get('processed_pings') != report.get('expected_events', {}).get('sonar')
            or report.get('observation_source') != 'complex_IQ'
            or report.get('requested_feature_matcher') != 'LoFTR'):
        raise ValueError('A complete legacy-loop full route and lossless input archive are required.')
    requested = report.get('requested_distance_m', float('nan'))
    complete = report.get('complete_planned_loop_length_m', float('nan'))
    if not (math.isfinite(requested) and complete > 0 and math.isclose(requested, complete, rel_tol=0, abs_tol=1e-8)):
        raise ValueError('A route prefix cannot stand in for a complete route.')


def verify_runtime(root, expected_manifest=None):
    path = root/'runtime_manifest.json'
    digest = sha(path)
    if expected_manifest is not None and digest != expected_manifest:
        raise RuntimeError('Staged runtime manifest changed.')
    report = json.loads(path.read_text())
    for name, expected in report['files_SHA256'].items():
        if Path(name).name != name or not name.endswith('.py'):
            raise ValueError('Only explicit local Python source files belong in this runtime.')
        if sha(root/'standalone'/name) != expected:
            raise RuntimeError('Staged source changed: '+name)
    required = {'ReplayPureSideScanNavigation.py', 'AnalyzeSideScanLoopReplay.py', 'SideScanHistoricalMatching.py'}
    if not required <= report['files_SHA256'].keys(): raise ValueError('Incomplete staged replay/analysis runtime.')
    return digest


def replay_command(python, runtime, experiment, run, out, representation, loops):
    command = [str(python), str(runtime/'standalone/ReplayPureSideScanNavigation.py'),
        '--experiment', str(experiment), '--run', str(run), '--out', str(out),
        '--runtime', 'current', '--submap-representation', representation]
    return command+(['--loop-closure'] if loops else [])


def verify_legacy_control(source, replay):
    import numpy as np
    with np.load(source/'navigation_states.npz') as old, np.load(replay/'estimated_states.npz') as new:
        np.testing.assert_array_equal(new['positions'], old['positions'][:, 1:])
        np.testing.assert_array_equal(new['timestamps'], old['timestamps'])
        np.testing.assert_array_equal(new['distance_m'], old['distance_m'])
        np.testing.assert_array_equal(new['yaw_rad'], old['yaw_rad'][:, 1:])
        return dict(positions_timestamps_yaw_distance_bit_exact=True, frames=len(new['timestamps']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit', required=True)
    p.add_argument('--pid', required=True, type=int)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    # The strict report writer imports the frozen JSON sanitizer lazily.
    # Unlike replay, this observer has not otherwise loaded oceansim yet.
    sys.path.insert(0, str(a.experiment/'estimator_bundle/isaacsim'))
    if a.out.exists(): raise FileExistsError('Never overwrite an earlier matrix or resume unknown partial jobs.')
    verify_target(service_state(a.unit), a.pid, initially_live=True)
    runtime_digest = verify_runtime(a.runtime)
    observer_digest = sha(Path(__file__))
    a.out.mkdir()
    report = dict(status='waiting_for_exact_live_full_source', target_unit=a.unit, target_PID=a.pid,
        source_run=str(a.run.resolve()), runtime=str(a.runtime.resolve()), runtime_manifest_SHA256=runtime_digest,
        observer_SHA256=observer_digest, estimator_restart_allowed=False,
        secondary_denoising_allowed=False, completed_replays=[], completed_comparisons=[])
    def save(): dump(a.out/'validation_status.json', report)
    def check_runtime():
        verify_runtime(a.runtime, runtime_digest)
        if sha(Path(__file__)) != observer_digest: raise RuntimeError('Observer source changed while running.')
    def execute(command):
        check_runtime(); print(json.dumps(dict(executing=command)), flush=True)
        subprocess.run(command, check=True)
        check_runtime()
    try:
        save()
        while not verify_target(service_state(a.unit), a.pid): time.sleep(5.)
        source_report = json.loads((a.run/'run_summary.json').read_text())
        validate_full_source(source_report)
        report.update(status='replaying_complete_identical_inputs', source_run_summary_SHA256=sha(a.run/'run_summary.json'))
        save()
        for name, representation, loops in VARIANTS:
            report['current_replay'] = name; save()
            execute(replay_command(sys.executable, a.runtime, a.experiment, a.run, a.out/name, representation, loops))
            report['completed_replays'].append(name); save()
        report['legacy_control'] = verify_legacy_control(a.run, a.out/'legacy_loops_control')
        report['status'] = 'independently_auditing_full_matrix'; save()
        for name, old, new, kind in COMPARISONS:
            report['current_comparison'] = name; save()
            execute([sys.executable, str(a.runtime/'standalone/AnalyzeSideScanLoopReplay.py'),
                '--source-run', str(a.run), '--reference', str(a.out/old), '--candidate', str(a.out/new),
                '--comparison', kind, '--out', str(a.out/('review_'+name))])
            report['completed_comparisons'].append(name); save()
        report['metrics'] = {name: json.loads((a.out/name/'run_summary.json').read_text())['metrics']
                             for name, _, _ in VARIANTS}
        report['Global_RMSE_change_percent'] = {name: json.loads((a.out/('review_'+name)/'independent_comparison.json').read_text())['Global_RMSE_change_percent']
                                              for name, _, _, _ in COMPARISONS}
        report.update(status='completed_independent_full_route_2x2_ablation',
            scope='Same measured-derived scene, seed and controlled initialization; not independent field validation.',
            negative_results_must_be_retained=True)
        save()
    except BaseException as exc:
        report.update(status='failed_preserved_full_matrix_no_automatic_restart', error=f'{type(exc).__name__}: {exc}')
        save(); raise


if __name__ == '__main__': main()
