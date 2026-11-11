"""Verify published derived results without raw sonar or an estimator rerun.

Run: python verify_bundle.py
Requires only NumPy. Does not fit, align, smooth, discard samples or write data.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED = {'sea': 243569, 'lake': 157056, 'qiandao': 62458}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def check_close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def read_csv(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        header = next(csv.reader([stream.readline()]))
        values = np.loadtxt(stream, delimiter=',')
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Incomplete or nonfinite data; no filtering allowed.')
    return {name: values[:, i] for i, name in enumerate(header)}


def main():
    manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        path = (ROOT / entry['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Invalid or missing package file.')
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['SHA256']:
            raise ValueError('Package file differs: ' + entry['path'])
    verified = json.loads((ROOT / 'data/verified_metrics.json').read_text())
    result = {}
    for scene, count in EXPECTED.items():
        if scene == 'qiandao':
            with np.load(ROOT / 'data/qiandao_NX_navigation_states.npz', allow_pickle=False) as states:
                positions = states['positions']
                time, distance = states['timestamps'], states['distance_m']
                stored_error = states['error_m']
            if positions.shape != (count, 4, 3):
                raise ValueError('Wrong channel order or count.')
            check_close(distance, (time - time[0]) * .8)
        else:
            columns = read_csv(ROOT / f'data/{scene}_hil_trajectories_full.csv.gz')
            time, distance = columns['timestamp_s'], columns['distance_m']
            positions = np.stack([
                np.column_stack([columns[f'{channel}_{axis}_m'] for axis in 'xyz'])
                for channel in ('truth', 'dr', 'local', 'global')], axis=1)
            stored_error = np.column_stack([columns[f'{ch}_error_m'] for ch in ('dr', 'local', 'global')])
            chords = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions[:, 0], axis=0), axis=1))]
            np.testing.assert_allclose(distance, chords, rtol=0, atol=1e-5)
        if len(time) != count or not np.all(np.diff(time) > 0) or not np.isfinite(positions).all():
            raise ValueError('Incomplete/invalid trajectory.')
        errors = np.linalg.norm(positions[:, 1:] - positions[:, :1], axis=2)
        check_close(errors, stored_error)
        complete = read_csv(ROOT / f'data/{scene}_complete_error_samples.csv.gz')
        check_close(complete['timestamp_s'], time)
        check_close(complete['distance_m'], distance)
        checked = {}
        for j, channel in enumerate(('DR', 'Local', 'Global')):
            error = errors[:, j]
            check_close(complete[channel + '_error_m'], error)
            metrics = {'rmse_m': float(np.sqrt(np.mean(error ** 2))),
                       'median_m': float(np.median(error)),
                       'p95_m': float(np.percentile(error, 95)),
                       'p99_m': float(np.percentile(error, 99)),
                       'max_m': float(error.max()), 'final_m': float(error[-1])}
            for key, value in metrics.items():
                check_close(value, verified[scene]['metrics'][channel][key])
            checked[channel] = metrics
        result[scene] = {'samples': count, 'distance_m': float(distance[-1]), 'metrics': checked}
    print(json.dumps({'status': 'all_hashes_and_full_position_errors_verified',
                      'package_files_checked': len(manifest['files']), 'experiments': result,
                      'raw_sonar_required': False, 'estimator_rerun': False,
                      'SE3_rotation_RPE_recomputed': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
