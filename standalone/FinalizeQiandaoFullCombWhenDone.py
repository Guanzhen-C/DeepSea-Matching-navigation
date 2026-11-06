"""Wait for ONE verified full-route PID, then review ONCE. Never restarts it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from RunQiandaoNoGPSNavigation import dump, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit', required=True)
    p.add_argument('--pid', required=True, type=int)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if not a.unit.startswith('oceansim-qiandao-') or a.pid <= 0 or a.out.exists():
        raise ValueError('Exact Qiandao-only PID/unit and new review output required.')
    script = Path(__file__).with_name('AnalyzeQiandaoFullFrameComb.py')
    digest = sha(script)
    state = a.experiment/'full_route_finalizer_status.json'
    if state.exists():
        raise FileExistsError('No duplicate finalizers for the same full-route experiment.')
    record = dict(status='waiting_for_same_verified_PID', target_unit=a.unit, target_PID=a.pid,
        analysis_SHA256=digest, estimator_restart_allowed=False, other_processes_changed=False)
    dump(state, record)
    try:
        while True:
            response = subprocess.check_output(['systemctl', '--user', 'show', a.unit,
                '-p', 'MainPID', '-p', 'ExecMainStatus', '-p', 'SubState'], text=True)
            properties = dict(x.split('=', 1) for x in response.splitlines() if '=' in x)
            pid = int(properties.get('MainPID', '0'))
            if pid:
                if pid != a.pid:
                    raise RuntimeError('Target PID changed: do not attach to a replacement or restart it.')
                time.sleep(5)
                continue
            if properties.get('ExecMainStatus') != '0' or properties.get('SubState') != 'exited':
                raise RuntimeError('Target terminal without success; preserve failure without restart.')
            report = json.loads((a.run/'run_summary.json').read_text())
            if (report['status'] != 'completed_ALL_scheduled_events_full_frozen_joint_estimator'
                    or not report['full_closed_loop_requested']):
                raise RuntimeError('Unit terminal does not prove requested full-loop completion.')
            if sha(script) != digest:
                raise RuntimeError('Analysis code changed while waiting; no silent policy change.')
            record.update(status='verified_full_run_terminal_reviewing_once', target_final=properties)
            dump(state, record)
            subprocess.run([sys.executable, str(script), '--experiment', str(a.experiment),
                            '--run', str(a.run), '--out', str(a.out)], check=True)
            record.update(status='full_run_finished_and_independent_review_completed', output=str(a.out),
                          report_SHA256=sha(a.out/'全航程独立核验与指标.json'))
            dump(state, record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            return
    except BaseException as exc:
        record.update(status='stopped_preserving_target_or_review_failure_no_restart',
                      error=type(exc).__name__+': '+str(exc))
        dump(state, record)
        raise


if __name__ == '__main__':
    main()
