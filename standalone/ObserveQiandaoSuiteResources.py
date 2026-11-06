"""Read-only resource observation of this ONE PC experiment unit."""
from pathlib import Path
import argparse
import json
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if not a.unit.startswith('oceansim-qiandao-') or a.out.exists():
        raise ValueError('Explicit new Qiandao-only resource log required.')
    samples, peak, minimum_available, last_events = 0, 0, None, None
    with a.out.open('x') as f:
        while True:
            text = subprocess.check_output(['systemctl', '--user', 'show', a.unit,
                '-p', 'MainPID', '-p', 'SubState', '-p', 'ControlGroup', '-p', 'ExecMainStatus'], text=True)
            properties = dict(x.split('=', 1) for x in text.splitlines() if '=' in x)
            root = Path('/sys/fs/cgroup')/properties.get('ControlGroup', '').lstrip('/')
            mem = {n: (root/n).read_text().strip() if (root/n).is_file() else None
                   for n in ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.max', 'memory.events')}
            available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                             if x.startswith('MemAvailable:'))
            minimum_available = available if minimum_available is None else min(minimum_available, available)
            if mem['memory.peak']:
                peak = max(peak, int(mem['memory.peak']))
            if mem['memory.events']:
                last_events = mem['memory.events']
            entry = dict(wall_time=time.time(), unit=properties, memory=mem, PC_available_bytes=available)
            f.write(json.dumps(entry)+'\n')
            f.flush()
            samples += 1
            if properties.get('MainPID') == '0':
                break
            time.sleep(5)
    summary = dict(status='observer_finished_target_unit_terminal', samples=samples,
        target=a.unit, observed_peak_cgroup_bytes=peak,
        minimum_PC_available_bytes=minimum_available, last_memory_events=last_events,
        target_final=properties, other_processes_changed=False)
    a.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
