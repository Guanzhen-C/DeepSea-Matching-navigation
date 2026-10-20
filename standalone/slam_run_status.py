#!/usr/bin/env python3
"""Read-only health of a guarded SLAM run, including progress freshness.

An active systemd unit or Result=success is NOT evidence of forward progress
or successful route completion. Inspect the latest complete 5 Hz pose row,
checkpoint mtime, final metrics and cgroup pressure together. This utility
never restarts a process, adjusts resources, or changes estimator settings.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import time


def latest_complete_pose(path: Path) -> dict | None:
    """Read a bounded tail; ignore partial records during a concurrent write."""
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        offset = max(0, size - 16384)
        stream.seek(offset)
        payload = stream.read()
    lines = payload.splitlines()
    if offset and lines:
        lines = lines[1:]
    if not payload.endswith(b'\n') and lines:
        lines = lines[:-1]
    for line in reversed(lines):
        try:
            fields = next(csv.reader([line.decode('utf-8')]))
            if len(fields) != 14:
                continue
            values = list(map(float, fields))
            if not all(math.isfinite(value) for value in values):
                continue
        except (ValueError, UnicodeDecodeError, csv.Error):
            continue
        truth = values[2:5]
        return dict(
            timestamp_s=values[0], travelled_distance_m=values[1],
            errors_m={name: math.sqrt(sum((values[start + axis] - truth[axis])**2
                                        for axis in range(3)))
                      for name, start in [('dr', 5), ('local', 8), ('global', 11)]},
        )
    return None


def load_json_if_present(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def systemd_properties(unit: str) -> dict[str, str]:
    result = subprocess.run([
        'systemctl', '--user', 'show', unit,
        '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result',
        '-p', 'ExecMainStatus', '-p', 'MainPID', '-p', 'ControlGroup',
        '-p', 'MemoryCurrent', '-p', 'MemoryHigh', '-p', 'MemoryMax',
        '-p', 'MemorySwapMax',
    ], check=False, capture_output=True, text=True)
    if result.returncode:
        return {'ActiveState':'unknown', 'SubState':'unknown',
                'query_error':result.stderr.strip()}
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def evaluate_health(*, active: bool, final_status: str | None,
                    freshest_age_s: float | None, stale_after_s: float,
                    startup_age_s: float | None = None) -> str:
    if final_status == 'completed':
        return 'completed_cleanup_active' if active else 'completed'
    if final_status is not None and final_status != 'running':
        return 'failed_or_interrupted'
    if not active:
        return 'incomplete_not_running'
    if freshest_age_s is None:
        return ('initializing' if startup_age_s is not None
                and startup_age_s <= stale_after_s else 'no_recent_progress')
    return 'running' if freshest_age_s <= stale_after_s else 'no_recent_progress'


def read_status(directory: Path, *, stale_after_s: float = 180.,
                now: float | None = None, service: dict | None = None) -> dict:
    if not math.isfinite(stale_after_s) or stale_after_s <= 0:
        raise ValueError('stale_after_s must be finite and positive')
    now = time.time() if now is None else now
    manifest = load_json_if_present(directory / 'run_manifest.json')
    if manifest is None or 'unit' not in manifest:
        raise ValueError('a guarded run_manifest.json with unit is required')
    service = systemd_properties(manifest['unit']) if service is None else service
    progress_path = directory / 'metrics_progress.json'
    pose_path = directory / 'terrain_replay/online_poses_5hz.csv'
    final = load_json_if_present(directory / 'metrics.json')
    progress = load_json_if_present(progress_path)
    pose = latest_complete_pose(pose_path)
    # A header-only file cannot establish that any sensor event completed.
    ages = {label: max(0., now - path.stat().st_mtime)
            for label, path, present in [('checkpoint', progress_path, progress is not None),
                                         ('pose_journal', pose_path, pose is not None)]
            if present}
    freshest = min(ages.values(), default=None)
    startup_age = None
    if manifest.get('created_at'):
        startup_age = max(0., now - datetime.fromisoformat(manifest['created_at']).timestamp())
    health = evaluate_health(
        active=service.get('ActiveState') == 'active',
        final_status=None if final is None else final.get('run_status'),
        freshest_age_s=freshest, stale_after_s=stale_after_s,
        startup_age_s=startup_age,
    )
    result = dict(
        directory=str(directory), unit=manifest['unit'], health=health,
        checked_at=datetime.fromtimestamp(now).astimezone().isoformat(),
        stale_after_s=stale_after_s, file_age_s=ages, service=service,
        last_complete_live_pose=pose,
        last_checkpoint=({key: progress.get(key) for key in (
            'status', 'travelled_distance_m', 'route_completion_fraction',
            'online_global_error_m', 'local_error_m', 'terrain_priors_accepted',
            'terrain_consecutive_failed_observations', 'terrain_navigation_output_source',
        )} if progress is not None else None),
        final_metrics=({key: final.get(key) for key in (
            'run_status', 'termination_reason', 'completed_requested_route',
            'survey_travelled_distance_m', 'position_error_statistics',
        )} if final is not None else None),
    )
    group = service.get('ControlGroup')
    if group:
        cgroup = Path('/sys/fs/cgroup') / group.lstrip('/')
        for filename, label in [('memory.events', 'memory_events'),
                                ('memory.pressure', 'memory_pressure')]:
            try:
                result[label] = (cgroup / filename).read_text().strip()
            except (FileNotFoundError, PermissionError):
                pass
    if health == 'no_recent_progress':
        result['warning'] = ('Active does not prove progress. The run may be paused, '
                             'blocked, or executing a long operation; inspect pressure '
                             'and stack before intervening. Last errors are stale.')
    if final is None and (directory / 'metrics.json').exists():
        result['final_metrics_warning'] = ('Final metrics JSON is incomplete or invalid; '
                                           'completion is not established.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--stale-after-s', type=float, default=180.)
    args = parser.parse_args()
    try:
        report = read_status(args.directory, stale_after_s=args.stale_after_s)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
