#!/usr/bin/env python3
"""Start ONE bounded-memory Chelan experiment in a persistent user service.

Never stops another job. Refuses low host memory, existing OceanSim processes,
or reused output directories. A cgroup limit confines a failure to this job;
it cannot guarantee unrelated applications will not exhaust host memory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def memory_available_bytes(text: str) -> int:
    entries = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
    return int(entries['MemAvailable'].split()[0]) * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('baseline', 'consolidation'), required=True)
    parser.add_argument('--start-distance-m', type=float, default=0.)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--distance-m', type=float)
    scope.add_argument('--full-route', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.start_distance_m) or args.start_distance_m < 0:
        parser.error('start distance must be finite and nonnegative')
    if args.distance_m is not None and (not math.isfinite(args.distance_m) or args.distance_m <= 0):
        parser.error('distance must be finite and positive')
    output = args.output.resolve()
    if output.exists():
        parser.error('output directory already exists; use a fresh path')
    available = memory_available_bytes(Path('/proc/meminfo').read_text())
    if available < 12 * GIB:
        parser.error(f'only {available/GIB:.1f} GiB available; require 12 GiB before launch')
    processes = subprocess.run(['pgrep', '-af', '[p]ython.*[/]large_scale_slam_isaac_demo.py'],
                               text=True, capture_output=True)
    if processes.returncode == 0:
        parser.error('another OceanSim task is already running; it was NOT stopped:\n' + processes.stdout)
    label = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    unit = f'oceansim-chelan-{args.profile}-{label}'
    command = [
        '/usr/bin/env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        '/bin/bash', str(ROOT / 'standalone/run_v2_public_terrain.sh'), 'chelan',
        '--headless', '--duration', '0', '--survey-laps', '1.0',
        '--survey-start-distance-m', str(args.start_distance_m),
        '--survey-segment-distance-m', str(args.distance_m or 0.),
        '--seed', '31', '--progress-distance-m', '100',
        '--terrain-replay-directory', str(output / 'terrain_replay'),
        '--terrain-merge-converged-cohorts' if args.profile == 'consolidation'
        else '--no-terrain-merge-converged-cohorts',
        '--metrics', str(output / 'metrics.json'),
    ]
    properties = ['MemoryHigh=6G', 'MemoryMax=7G', 'MemorySwapMax=0',
                  'CPUQuota=200%', 'CPUWeight=20', 'IOWeight=20', 'Nice=10', 'OOMPolicy=stop']
    service = ['systemd-run', '--user', '--unit', unit, '--working-directory', str(ROOT)]
    for prop in properties:
        service += ['--property', prop]
    service += ['--property', f'StandardOutput=append:{output}/simulation.log',
                '--property', f'StandardError=append:{output}/simulation.log', *command]
    manifest = dict(unit=unit + '.service', argv=command, systemd_properties=properties,
                    memory_available_at_launch_gib=available/GIB,
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    tracked_diff=subprocess.check_output(['git', 'diff', '--stat'], cwd=ROOT, text=True),
                    profile=args.profile, scope='full_route' if args.full_route else 'fresh_start_segment',
                    start_distance_m=args.start_distance_m, distance_m=args.distance_m,
                    not_a_resumed_estimator_checkpoint=True,
                    created_at=datetime.now().astimezone().isoformat())
    manifest['source_sha256'] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / 'isaacsim/oceansim/slam').glob('*.py'))
        + [ROOT / 'standalone/large_scale_slam_isaac_demo.py',
           ROOT / 'standalone/run_v2_public_terrain.sh',
           ROOT / 'standalone/run_v2_full_map_slam_isaac.sh']
    }
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        subprocess.run(service, check=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
