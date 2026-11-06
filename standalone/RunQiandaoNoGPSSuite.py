"""Serial, bounded PC experiment suite; never overlaps estimator processes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


SOURCE_NAMES = ('RunQiandaoNoGPSNavigation.py', 'QiandaoVirtualNavigation.py',
                'QiandaoLocalScene.py', 'QiandaoLocalRaycaster.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    t = path.with_suffix('.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    t.replace(path)


def completed(path, task, expected_source):
    d = json.loads((path/'run_summary.json').read_text())
    if (d['status'] != 'completed_full_frozen_estimator_GPS_free_virtual_AUV_experiment'
            or d['variant'] != task['variant'] or d['seed'] != task['seed']
            or d['requested_distance_m'] != 250 or d.get('swath_width_deg') != 160
            or d['source_code_SHA256'] != expected_source
            or d['GPS_file_opened'] or d['GPS_input_or_truth']
            or d['initial_offset_m'] != task['initial_offset_m']
            or d['initial_yaw_deg'] != task['initial_yaw_deg']
            or not d.get('small_map_raycaster')
            or d['processed_pings'] != 947):
        raise ValueError('Not a completed exact-input full-scope suite task: '+str(path))
    if hashlib.sha256((path/'trajectory.csv').read_bytes()).hexdigest() != d['trajectory_SHA256']:
        raise ValueError('Changed trajectory output.')
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--warm-unit', required=True)
    p.add_argument('--warm-pid', required=True, type=int)
    a = p.parse_args()
    base = a.experiment.resolve()
    out = base/'wide_v3_suite'
    if out.exists():
        raise FileExistsError('No automatic reruns or silent resume of previous suites.')
    out.mkdir()
    source = Path(__file__).parent
    expected = {n: sha(source/n) for n in SOURCE_NAMES}
    tasks = []
    # Fixed seeds/cases selected before observing any of their results.
    for case, offset, yaw in [('exact', 0., 0.), ('offset20m_yaw5deg', 20., 5.)]:
        for seed in (31, 32, 33):
            for variant in ('raw', 'baseline', 'edge'):
                destination = (base/'wide_edge_250m_seed31_v3' if
                    (case, seed, variant) == ('exact', 31, 'edge') else
                    out/f'{case}_seed{seed}_{variant}')
                tasks.append(dict(case=case, seed=seed, variant=variant,
                    initial_offset_m=offset, initial_yaw_deg=yaw, directory=str(destination)))
    state = dict(status='checking_existing_warm_job', code_SHA256=sha(Path(__file__)),
        expected_source=expected, tasks=tasks, completed=[], one_estimator_at_a_time=True,
        warm_unit=a.warm_unit, warm_PID=a.warm_pid, GPS_used=False, start_wall_time=time.time())

    def save():
        state['elapsed_s'] = time.time()-state['start_wall_time']
        write(out/'suite_state.json', state)

    save()
    try:
        while True:
            query = subprocess.check_output(['systemctl', '--user', 'show', a.warm_unit,
                '-p', 'MainPID', '-p', 'SubState', '-p', 'ExecMainStatus'], text=True)
            props = dict(x.split('=', 1) for x in query.splitlines() if '=' in x)
            pid = int(props.get('MainPID', '0'))
            if pid:
                if pid != a.warm_pid:
                    raise RuntimeError('Warm job PID changed; do not restart or overlap it.')
                state.update(status='verified_wait_for_same_live_warm_process', live_PID=pid)
                save()
                time.sleep(2)
                continue
            if props.get('ExecMainStatus') != '0' or props.get('SubState') != 'exited':
                raise RuntimeError('Warm job not successfully terminal; preserve failure.')
            break
        # This existing completed candidate is inspected, not run a second time.
        warm = tasks[2]
        completed(Path(warm['directory']), warm, expected)
        for task in tasks:
            destination = Path(task['directory'])
            if destination == Path(warm['directory']):
                report = completed(destination, task, expected)
                reused = True
            else:
                if destination.exists():
                    raise FileExistsError('Unplanned existing task output: '+str(destination))
                command = [sys.executable, str(source/'RunQiandaoNoGPSNavigation.py'),
                    '--experiment', str(base), '--variant', task['variant'], '--seed', str(task['seed']),
                    '--distance-m', '250', '--swath-width-deg', '160',
                    '--initial-offset-m', str(task['initial_offset_m']),
                    '--initial-yaw-deg', str(task['initial_yaw_deg']), '--out', str(destination)]
                log = destination.with_suffix('.log')
                with log.open('x') as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    state.update(status='running_one_estimator', live_PID=child.pid, current_task=task)
                    save()
                    while child.poll() is None:
                        time.sleep(2)
                        save()
                    if child.returncode:
                        raise RuntimeError(f'Task exited {child.returncode}; no automatic restart: {destination}')
                report = completed(destination, task, expected)
                reused = False
            state['completed'].append(dict(**task, reused_existing_verified_output=reused,
                metrics=report['metrics'], input_hashes=report['input_hashes'],
                terrain_accepts=report.get('last_terrain_accepts'),
                report_SHA256=sha(destination/'run_summary.json')))
            print(json.dumps(dict(completed=len(state['completed']), total=len(tasks), task=task,
                                   Global=report['metrics']['Global'])), flush=True)
            save()
        # Same seed has identical sensors across both initialisation cases;
        # appearance varies by group. Never confuse B=C inputs with evidence.
        for case in ('exact', 'offset20m_yaw5deg'):
            for seed in (31, 32, 33):
                values = [v for v in state['completed'] if v['case'] == case and v['seed'] == seed]
                for channel in ('navigation_measurements', 'sonar_geometry'):
                    if len({v['input_hashes'][channel] for v in values}) != 1:
                        raise ValueError('Unfair sensor/geometry inputs across groups.')
                if len({v['input_hashes']['sonar_appearance'] for v in values}) != 3:
                    raise ValueError('The real differences between all three processed fields were not observed.')
        state.update(status='completed_all18_paired_GPS_free_PC_experiments', live_PID=0,
                     fair_navigation_and_geometry_hashes_verified=True,
                     all_three_appearance_input_hashes_distinct_in_every_case=True)
        save()
    except BaseException as exc:
        state.update(status='failed_preserved_no_automatic_restart', error=type(exc).__name__+': '+str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
