from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'standalone'))
from RunFrozenSonarDenoising import extract, inspect_source, render


def make_source(root, count=4, bins=2048):
    h5py = pytest.importorskip('h5py')
    root.mkdir()
    rng = np.random.default_rng(24)
    expected = []
    for part_number in (1, 2):
        directory = root / f'Data{part_number:02d}'
        directory.mkdir()
        with h5py.File(directory / 'SysPara.mat', 'w') as handle:
            group = handle.create_group('SysPara')
            for name, value in {'c': 1480., 'BeamSigFs': 37650.}.items():
                group.create_dataset(name, data=[[value]])
        sums = np.empty((2, count, 8, bins), np.complex128)
        for s, side in enumerate(('Port', 'Stbd')):
            for kind in ('Beamforming', 'Depth'):
                file = directory / f'{side}NormalMode{kind}Total.mat'
                with h5py.File(file, 'w') as h:
                    refs = h.create_dataset(file.stem+'Data', (count, 1), dtype=h5py.ref_dtype)
                    for i in range(count):
                        rec = h.create_group(f'row{i}')
                        for k, v in {'PingNum': part_number*100+i, 'PingS': 123456,
                                     'PingUs': i*100000}.items():
                            rec.create_dataset(k, data=[[v]])
                        if kind == 'Beamforming':
                            z = (rng.normal(size=(4, bins, 8)) +
                                 1j*rng.normal(size=(4, bins, 8))) * .05
                            z[:, -13:] = 0
                            compound = np.empty(z.shape, dtype=[('real', '<f8'), ('imag', '<f8')])
                            compound['real'], compound['imag'] = z.real, z.imag
                            rec.create_dataset(side+'BeamformingData', data=compound)
                            sums[s, i] = (z[0]+z[1]).T
                        else:
                            rec.create_dataset('BottomTrackingInfo', data=[[1.]])
                            rec.create_dataset('TargetDelayInfo', data=np.full((20, 8), 4./1480.))
                        refs[i, 0] = rec.ref
        expected.append(sums)
    return np.concatenate(expected, axis=1).transpose(0, 2, 1, 3)


def test_original_mat_complex_axes_identity_and_limits(tmp_path):
    expected = make_source(tmp_path / 'input')
    meta = extract(tmp_path / 'input', tmp_path / 'output')
    assert meta['native_shape'] == [2, 8, 8, 2048]
    assert [p['count'] for p in meta['parts']] == [4, 4]
    actual = np.load(tmp_path / 'output/raw/native_complex128.npy')
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.complex128
    np.testing.assert_array_equal(np.load(tmp_path / 'output/raw/bottom_first_ranges.npy'),
                                  np.broadcast_to([1., 2.], (2, 8, 2)))


def test_original_mat_identity_mismatch_and_changed_sample_rate_rejected(tmp_path):
    h5py = pytest.importorskip('h5py')
    make_source(tmp_path / 'input')
    file = tmp_path / 'input/Data02/PortNormalModeDepthTotal.mat'
    with h5py.File(file, 'r+') as h:
        h['row1/PingNum'][...] = 9999
    with pytest.raises(ValueError, match='identities'):
        inspect_source(tmp_path / 'input')
    with h5py.File(file, 'r+') as h:
        h['row1/PingNum'][...] = 201
    with h5py.File(tmp_path / 'input/Data02/SysPara.mat', 'r+') as h:
        h['SysPara/BeamSigFs'][...] = 40000.
    with pytest.raises(ValueError, match='37650'):
        inspect_source(tmp_path / 'input')


def test_raw_mat_to_real_checkpoint_edge_and_shotwell_end_to_end(tmp_path):
    make_source(tmp_path / 'input')
    ml = Path(os.environ.get('SONAR_ML_PYTHON', str(ROOT / '.venv-sonar-denoise/bin/python')))
    if not ml.exists():
        ml = Path(sys.executable)
        pytest.importorskip('torch')
    engine = ROOT / 'third_party/shotwell_0_30_14/build_palette_adapter/shotwell-enhance-palette'
    if not engine.exists():
        pytest.skip('Build original Shotwell engine or run the documented setup first')
    env = os.environ.copy()
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    command = [sys.executable, str(ROOT / 'standalone/RunFrozenSonarDenoising.py'),
               '--source', str(tmp_path / 'input'), '--out', str(tmp_path / 'output'),
               '--ml-python', str(ml), '--device', 'cpu', '--display-python', '/usr/bin/python3']
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / 'output/pipeline_manifest.json').read_text())
    assert report['status'] == 'completed_raw_MAT_to_edge_repaired_Shotwell_enhanced_native'
    assert report['stages'] == ['extract', 'infer', 'render', 'enhance']
    enhanced = json.loads((tmp_path / 'output/enhanced/处理与原生像素校验.json').read_text())
    assert enhanced['spatial_indices_byte_exact']
    assert enhanced['black_palette_entries_stay_black']
    assert enhanced['native_dimensions'] == [4096, 36]
    assert not report['source_MAT_modified'] and not report['training']
    # Review mode must refuse to label synthetic data as the reviewed real batch.
    (tmp_path / 'output/render').rename(tmp_path / 'previous_render_preserved')
    with pytest.raises(ValueError, match='Reference batch pixels differ'):
        render(tmp_path / 'output', reference_batch=True)
