"""Wait for one exact candidate PID, then independently analyze its FULL run.

Never stops or restarts navigation. No automatic success/improvement claim:
the complete comparison, observed constraints and metrics need final review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from FinalizePureSideScanExperiment import service_state, sha


def save(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit', required=True)
    p.add_argument('--pid', type=int, required=True)
    for name in ['source-run', 'candidate', 'out', 'status', 'project-root']:
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    state = service_state(a.unit)
    if (a.pid <= 1 or int(state['MainPID']) != a.pid or state['SubState'] != 'running'
            or a.status.exists() or a.out.exists()):
        raise ValueError('One confirmed exact live process and new outputs required')
    script = Path(__file__).with_name('AnalyzeMappedRangeNavigation.py')
    script_hash = sha(script)
    report = dict(status='waiting_for_exact_live_mapped_range_replay', unit=a.unit, PID=a.pid,
                  analyzer_SHA256=script_hash, navigation_restart_allowed=False,
                  external_upload_allowed=False, improvement_not_assumed=True)
    save(a.status, report)
    try:
        while True:
            state = service_state(a.unit)
            if int(state['MainPID']) not in (0, a.pid):
                raise RuntimeError('Exact process changed; do not follow another process')
            if int(state['MainPID']) == 0:
                if int(state['ExecMainStatus']) != 0:
                    raise RuntimeError('Candidate failed; preserve artifacts and do not restart automatically')
                break
            time.sleep(5.)
        if sha(script) != script_hash: raise RuntimeError('Analysis source changed while waiting')
        report['status'] = 'analyzing_complete_same_input_candidate'
        save(a.status, report)
        subprocess.run([sys.executable, str(script), '--source-run', str(a.source_run),
                        '--candidate', str(a.candidate), '--out', str(a.out)],
                       check=True, cwd=a.project_root)
        report.update(status='completed_full_mapped_range_replay_and_independent_comparison',
                      full_metrics_and_figures_need_review=True, external_upload_performed=False)
        save(a.status, report)
    except BaseException as exc:
        report.update(status='failed_preserved_candidate_or_analysis_no_restart',
                      error=f'{type(exc).__name__}: {exc}')
        save(a.status, report)
        raise


if __name__ == '__main__': main()
