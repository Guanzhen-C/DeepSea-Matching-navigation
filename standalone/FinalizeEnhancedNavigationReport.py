"""Wait for one exact live reviewer, then build a local report once.

Never controls/restarts navigation. Report generation is source-staged and
full-result gated. Stops at pending_visual_review, not final acceptance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from FinalizePureSideScanExperiment import service_state, sha


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit', required=True)
    p.add_argument('--pid', type=int, required=True)
    for name in ('finalizer-status', 'experiment', 'run', 'review', 'original-source',
                 'baseline-replay', 'report-out', 'status', 'project-root'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.status.exists() or a.report_out.exists() or a.pid <= 1:
        raise ValueError('One exact live reviewer and new outputs required')
    initial = service_state(a.unit)
    if int(initial['MainPID']) != a.pid or initial['SubState'] != 'running':
        raise ValueError('Confirm the exact reviewer is live before attaching')
    scripts = ['BuildEnhancedNavigationComparison.py', 'BuildEnhancedNavigationReport.py',
               'BuildSeabedSimulationReportDocument.py', 'AuditEnhancedNavigationReport.py']
    directory = Path(__file__).resolve().parent
    frozen = {name: sha(directory / name) for name in scripts}
    base_report = a.project_root / 'docs/reports/seabed_matching_simulation_20261107/海底地形地貌匹配算法仿真报告.md'
    original_hash = sha(base_report)
    report = dict(status='waiting_for_exact_full_run_reviewer', reviewer_unit=a.unit,
                  reviewer_PID=a.pid, scripts_SHA256=frozen, original_report_SHA256=original_hash,
                  restart_or_stop_simulation_allowed=False, external_upload_allowed=False)
    save(a.status, report)
    try:
        while True:
            state = service_state(a.unit)
            pid = int(state['MainPID'])
            if pid not in (0, a.pid):
                raise RuntimeError('Reviewer changed; never follow another process')
            if pid == 0:
                if int(state['ExecMainStatus']) != 0:
                    raise RuntimeError('Full-run review failed; preserve data and do not restart navigation')
                break
            time.sleep(5.)
        finished = json.loads(a.finalizer_status.read_text())
        if finished['status'] != 'exact_full_run_and_independent_review_completed':
            raise ValueError('No complete actual full-route review')
        if sha(base_report) != original_hash or any(sha(directory / n) != h for n, h in frozen.items()):
            raise ValueError('Original manuscript or report-generator sources changed while waiting')
        a.report_out.parent.mkdir(parents=True, exist_ok=True)
        report['status'] = 'generating_from_complete_audited_new_run'
        save(a.status, report)
        commands = [
            [sys.executable, str(directory / scripts[0]), '--experiment', str(a.experiment),
             '--run', str(a.run), '--review', str(a.review), '--original-source', str(a.original_source),
             '--baseline-replay', str(a.baseline_replay), '--out', str(a.report_out)],
            [sys.executable, str(directory / scripts[1]), '--comparison', str(a.report_out),
             '--project-root', str(a.project_root)],
            [sys.executable, str(directory / scripts[2]), '--source',
             str(a.report_out / '海底地形地貌匹配算法仿真报告.md'), '--pdf'],
            [sys.executable, str(directory / scripts[3]), '--report', str(a.report_out)],
        ]
        for index, command in enumerate(commands):
            subprocess.run(command, check=True, cwd=a.project_root)
            report['completed_steps'] = index+1
            save(a.status, report)
        report.update(status='new_full_run_report_generated_pending_visual_review',
                      report_directory=str(a.report_out), old_report_unchanged=sha(base_report) == original_hash,
                      external_upload_performed=False, visual_review_still_required=True)
        save(a.status, report)
    except BaseException as exc:
        report.update(status='failed_preserved_report_pipeline_no_navigation_restart',
                      error=f'{type(exc).__name__}: {exc}')
        save(a.status, report)
        raise


if __name__ == '__main__':
    main()
