"""Wait for one exact systemd-owned pure SSS run, then audit it once.

Never restart a simulation, never accept stale partial reports, and never
stop an unrelated process. This observer has no control over the simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def service_state(unit):
    output = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'MainPID', '-p', 'SubState', '-p', 'ExecMainStatus'], text=True)
    return dict(line.split('=', 1) for line in output.splitlines() if '=' in line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit', required=True)
    p.add_argument('--pid', required=True, type=int)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--review', required=True, type=Path)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--joint-reference', type=Path)
    p.add_argument('--status', required=True, type=Path)
    a = p.parse_args()
    if a.pid <= 1 or a.status.exists() or a.review.exists():
        raise ValueError('One exact run and fresh observer/review outputs required.')
    analyzer = Path(__file__).with_name('AnalyzePureSideScanShort.py')
    expected = sha(analyzer)
    state = service_state(a.unit)
    if int(state.get('MainPID', 0)) != a.pid or state.get('SubState') != 'running':
        raise ValueError('The exact expected process must be confirmed live before waiting.')
    report = dict(status='waiting_for_exact_live_process', target_unit=a.unit, target_PID=a.pid,
                  analyzer_SHA256=expected, estimator_restart_allowed=False)
    def save(): a.status.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        while True:
            state = service_state(a.unit)
            current_pid = int(state.get('MainPID', 0))
            if current_pid not in (0, a.pid): raise RuntimeError('Target PID changed; refusing another run.')
            if current_pid == 0: break
            time.sleep(5.)
        if int(state.get('ExecMainStatus', -1)) != 0: raise RuntimeError('Target run failed; preserving outputs.')
        result = json.loads((a.run/'run_summary.json').read_text())
        if (not result['status'].startswith('completed_pure_side_scan_FEATURE_')
                or result['event_counts'] != result['expected_events']):
            raise ValueError('No terminal complete result from the target run.')
        if sha(analyzer) != expected: raise RuntimeError('Analyzer changed while waiting.')
        command = [sys.executable, str(analyzer), '--run', str(a.run), '--out', str(a.review),
                   '--experiment', str(a.experiment)]
        if a.joint_reference: command += ['--joint-reference', str(a.joint_reference)]
        subprocess.run(command, check=True)
        report.update(status='exact_full_run_and_independent_review_completed',
            run_summary_SHA256=sha(a.run/'run_summary.json'), review=str(a.review.resolve()))
        save()
    except BaseException as exc:
        report.update(status='failed_preserved_observer_no_automatic_restart', error=f'{type(exc).__name__}: {exc}')
        save(); raise


if __name__ == '__main__': main()
