"""Read-only input/source audit while the SAME full-route PID is live.

This is deliberately labelled a POST-launch audit, not a prelaunch capture.
It checks file modification times predate the live process, source bytes
against Git, bundle hashes, disjoint complete records, and producer digests.
It neither attaches to the estimator nor changes its files or parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from QiandaoLocalScene import reject_gps_file


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def current_pid(unit):
    return int(subprocess.check_output(['systemctl', '--user', 'show', unit,
                                        '-p', 'MainPID', '--value'], text=True).strip())


def process_start_unix(pid):
    # stat's comm can contain spaces, so split only after its final ')'.
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    start_ticks = int(stat[19])  # field22, with field3 starting at index0.
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                if line.startswith('btime '))
    return boot+start_ticks/os.sysconf('SC_CLK_TCK')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--unit', required=True)
    p.add_argument('--pid', required=True, type=int)
    a = p.parse_args()
    sys.addaudithook(reject_gps_file)
    base = a.experiment
    out = base/'post_launch_input_source_audit.json'
    if out.exists() or not a.unit.startswith('oceansim-qiandao-') or current_pid(a.unit) != a.pid:
        raise ValueError('One new audit of the exact currently live experiment PID required.')
    start = process_start_unix(a.pid)
    scene = json.loads((base/'scene/scene_report.json').read_text())
    run = json.loads((base/'run_full_seed31/run_summary.json').read_text())
    manifest = json.loads((base/'estimator_bundle/bundle_manifest.json').read_text())
    if scene['source_records'] != 2020 or run['source_physical_records'] != 2020 or run['GPS_used']:
        raise ValueError('Wrong scope or forbidden GPS run.')
    source_commit = run['original_estimator_source_commit']
    frozen = []
    for name, guard in manifest['files'].items():
        path = base/'estimator_bundle'/name
        if path.stat().st_mtime >= start:
            raise ValueError('An input was modified after the live estimator started: '+name)
        if path.stat().st_size != guard['bytes'] or sha(path) != guard['sha256']:
            raise ValueError('Bundle input no longer matches its producer manifest.')
        if name.startswith('map/'):
            continue
        expected = subprocess.check_output(['git', 'rev-parse', f'{source_commit}:{name}'],
                                           cwd=ROOT, text=True).strip()
        actual = subprocess.check_output(['git', 'hash-object', str(path)], cwd=ROOT, text=True).strip()
        if expected != actual:
            raise ValueError('Actual estimator source differs from the original Git baseline.')
        frozen.append(dict(path=name, Git_blob=actual))
    query = []
    for path in sorted((base/'scene/query_baseline').iterdir()):
        if path.is_file():
            if path.stat().st_mtime >= start:
                raise ValueError('Query field modified while estimator live.')
            query.append(dict(path=path.name, bytes=path.stat().st_size,
                              mtime_ns=path.stat().st_mtime_ns, SHA256=sha(path)))
    identities = np.load(base/'scene/all_record_PingNum_PingS_PingUs.npy')
    reference = np.load(base/'scene/reference_record_ids.npy')
    query_ids = np.load(base/'scene/query_record_ids.npy')
    np.testing.assert_array_equal(np.sort(np.r_[reference, query_ids]), np.arange(2020))
    if identities.shape != (2020, 3) or np.intersect1d(reference, query_ids).size:
        raise ValueError('Physical source record separation or full coverage changed.')
    for name, digest in scene['source_code_SHA256'].items():
        if sha(ROOT/'standalone'/name) != digest:
            raise ValueError('Scene preparation code changed after preparing this input.')
    for name, digest in run['source_code_SHA256'].items():
        if sha(base/'run_full_seed31/source_snapshot'/name) != digest or sha(ROOT/'standalone'/name) != digest:
            raise ValueError('Live adapter or executed source snapshot changed.')
    for item in scene['source_files']:
        if sha(Path(item['path'])) != item['SHA256']:
            raise ValueError('Original measured depth/SysPara source changed.')
    stopped = ROOT/'output/real_sonar/qiandao_navigation_nogps_20261106'
    gain = stopped/'inputs/fusion_wide_front_supported_log_delta.npy'
    if sha(gain) != scene['baseline_gain_SHA256']:
        raise ValueError('Stopped native denoising gain file changed.')
    # Both independently built measured surfaces must remain different.
    if sha(base/'scene/query_baseline/elevation_m.npy') == sha(base/'scene/reference_baseline/elevation_m.npy'):
        raise ValueError('Query terrain was replaced by the reference terrain.')
    if current_pid(a.unit) != a.pid:
        raise RuntimeError('Estimator PID changed during read-only audit.')
    result = dict(status='passed_readonly_full_scope_input_and_actual_Git_source_audit',
        capture_timing='POST-launch audit, explicitly NOT a prelaunch hash snapshot',
        verified_live_unit=a.unit, verified_same_PID=a.pid, process_started_unix_s=start,
        audit_wall_time_unix_s=time.time(), input_modification_times_before_process_start=True,
        all2020_records_and_disjoint_source_partitions_verified=True,
        source_commit=source_commit, actual_estimator_source_files=frozen,
        query_input_hashes=query, original_depth_and_SysPara_hashes_verified=True,
        all_native_denoising_gains_producer_hash_verified=True, measured_query_reference_depth_distinct=True,
        runtime_sources_and_snapshots_unchanged=True, GPS_used=False,
        estimator_files_or_parameters_changed=False, estimator_restarted=False,
        source_code_SHA256=sha(Path(__file__)))
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], source_files=len(frozen), query_files=len(query),
                         PID=a.pid, output=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
