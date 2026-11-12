"""Tiny fictional data exercise every real evaluation shard, not quality."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone'))
import EvaluateNoFIRSonarTrial as evaluation
import RandomRippleEvaluation as helper
from NoFIRSonarData import save, sha


def test_all_shards_write_new_known_comparisons_and_protect_real_water(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    root = tmp_path / 'candidate'
    sonar = tmp_path / 'sonar'
    out = root / 'evaluation'
    out.mkdir(parents=True)
    recovery = sonar / 'matched_processing_recovery_20261105_v1'
    voyage = recovery / 'measured_full_voyage_20261105_v1'
    (recovery / 'native_arrays').mkdir(parents=True)
    (recovery / 'remote_full_native_validation').mkdir()
    (voyage / 'native_source').mkdir(parents=True)
    (voyage / 'metadata').mkdir()
    spec = dict(events=[dict(wave_spacing_scaled_samples=4.5)],
        complex_range_PSD=np.full(512, 1 / 512).tolist(),
        beam_covariance_real=np.eye(8).tolist(), beam_covariance_imag=np.zeros((8, 8)).tolist(),
        crest_waveform=np.broadcast_to(np.maximum(np.cos(2*np.pi*np.arange(48)/48) - .5, 0), (8, 48)).tolist())
    save(root / 'random_ripple_profile.json', dict(triggered_source_model=dict(sides={'0': spec, '1': spec})))
    save(recovery / 'aggregate_appearance_profile.json', {})
    save(recovery / 'remote_full_native_validation/run_summary.json', dict(results=dict(fusion_wide_front=dict(cases=[]))))
    original = np.full((2, 8, 4, 32), .1 + .02j, np.complex128)
    original[..., :2] = 0
    gate = np.ones((4, 32), np.float32)
    gate[:, :5] = 0
    records = []
    for side, index in enumerate((6, 7)):
        source_path = recovery / f'native_arrays/source_{index:02d}_complex128.npy'
        np.save(source_path, original[side])
        support_path = root / f'support_{index}.npy'
        np.save(support_path, gate)
        stamp = source_path.stat()
        for beam in range(8):
            records.append(dict(id=f'qiandao_s{index}_b{beam}', path=str(source_path),
                file_guard=[stamp.st_size, stamp.st_mtime_ns], native_side=side,
                noise_support_file=str(support_path), noise_support_SHA256=sha(support_path),
                global_reference_RMS=.1, beam=beam))
    save(root / 'admission_manifest.json', dict(records=records))
    np.save(voyage / 'native_source/measured_complex128.npy', original)
    np.save(voyage / 'native_source/original_amplitude_float64.npy', np.abs(original))
    np.save(voyage / 'metadata/bottom_first_ranges.npy', np.broadcast_to([1., 3.], (2, 4, 2)))
    save(voyage / 'metadata/envelope_cache_summary.json', dict(parts=[dict(parameters=dict(c=2., BeamSigFs=1.))]))
    np.save(out / 'real_observed_noise_support.npy', np.stack([gate, gate]))
    for name in ('pretrain', 'adapt', 'previous_56M'):
        np.save(out / f'{name}_supported_log_delta.npy', np.full(original.shape, np.nan, np.float32))
    for name in ('reference', 'noisy', 'pretrain', 'adapt', 'previous_56M'):
        np.save(out / f'known_{name}_amplitude.npy', np.full(original.shape, np.nan, np.float32))
    cpu_torch = SimpleNamespace(device=lambda *args: torch.device('cpu'),
        cuda=SimpleNamespace(set_device=lambda *args: None, max_memory_allocated=lambda *args: 0),
        set_num_threads=torch.set_num_threads, backends=torch.backends)
    monkeypatch.setattr(evaluation, 'torch', cpu_torch)
    monkeypatch.setattr(evaluation, 'load_models', lambda *args, **kwargs: ({'pretrain': -.01, 'adapt': -.02}, {}))
    monkeypatch.setattr(helper, 'load_previous_model', lambda *args: (-.03, {}))
    monkeypatch.setattr(evaluation, 'infer_raw', lambda model, data, device: np.where(np.abs(data) > 0, model, 0).astype(np.float32))
    for worker in range(4):
        evaluation.worker(SimpleNamespace(root=root, sonar_root=sonar, worker_index=worker, workers=4, random_ripple=True))
    for name in ('pretrain', 'adapt', 'previous_56M'):
        known = []
        seen = set()
        for worker in range(4):
            report = json.loads((out / f'worker_{worker}.json').read_text())
            assert report['status'] == 'completed_all_assigned_known_and_measured_NO_FIR_beams'
            known.extend(report['known_cases'][name])
            seen.update((r['side'], r['beam']) for r in report['measured'])
        assert len(known) == 32
        assert len(seen) == 16
        assert all(c['noise_generator_identical_to_training'] for c in known)
        result = np.load(out / f'{name}_supported_log_delta.npy')
        assert np.isfinite(result).all()
        np.testing.assert_array_equal(result[..., :5], 0)
    for name in ('reference', 'noisy', 'pretrain', 'adapt', 'previous_56M'):
        assert np.isfinite(np.load(out / f'known_{name}_amplitude.npy')).all()
    noisy = np.load(out / 'known_noisy_amplitude.npy')
    restored = np.load(out / 'known_adapt_amplitude.npy')
    np.testing.assert_allclose(restored[..., 5:], noisy[..., 5:] * np.exp(-.02), rtol=2e-6)
    np.testing.assert_array_equal(restored[..., :5], np.abs(original[..., :5]).astype(np.float32))
