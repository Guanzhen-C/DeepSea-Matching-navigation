"""Whole16-beam known-input and real-voyage NO-FIR evaluation, four GPU shards.

The old both-band synthetic-input generator is ONLY a test-input generator,
not the candidate's preprocessing. Candidate gains are infer_raw(raw IQ) only.
Original phase, source bytes and native output coordinates are never replaced.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from NoFIRSonarData import read, save, sha
from NoFIRSonarWavelet import NoFIRSonarMWCNN, PROTOCOL, infer_raw
from TrainNativeLocalSonarWavelet import metrics, aggregate, array_sha
from TrainMatchedSonarFactorial import prepare_fields
from InferFullMeasuredSonar172 import verify_metadata, verify_native
from FullMeasuredSonarProtocol import supported_total_delta


def load_models(root, device, *, random_ripple=False):
    models, selected = {}, {}
    for phase in ('pretrain', 'adapt'):
        report = read(root/phase/'run_summary.json')
        item = report['selected']
        path = root/phase/item['file']
        if sha(path) != item['SHA256']:
            raise ValueError('selected new no-FIR checkpoint changed')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if (state['model_family'] != NoFIRSonarMWCNN.family
                or state['feature_protocol'] != PROTOCOL or state['no_FIR'] is not True):
            raise ValueError('explicit actually trained no-FIR family required')
        if random_ripple and (state.get('random_ripple_augmentation') is not True
                or state.get('noise_profile_SHA256') != sha(root/'random_ripple_profile.json')):
            raise ValueError('Checkpoint must be trained with this exact random-ripple profile.')
        model = NoFIRSonarMWCNN(state['width']).to(device)
        model.load_state_dict(state['model'], strict=True)
        models[phase] = model.eval()
        selected[phase] = item
        del state
        gc.collect()
    return models, selected


def worker(args):
    root, sonar = args.root, args.sonar_root
    out = root/'evaluation'
    device = torch.device('cuda', args.worker_index)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    models, selected = load_models(root, device, random_ripple=args.random_ripple)
    if args.random_ripple:
        from RandomRippleEvaluation import load_previous_model
        models['previous_56M'], selected['previous_56M'] = load_previous_model(sonar, device)
    recovery = sonar/'matched_processing_recovery_20261105_v1'
    voyage = recovery/'measured_full_voyage_20261105_v1'
    profile = read(recovery/'aggregate_appearance_profile.json')
    admission = read(root/'admission_manifest.json') if args.random_ripple else None
    if args.random_ripple:
        profile = read(root/'random_ripple_profile.json')
    old = read(recovery/'remote_full_native_validation/run_summary.json')
    expected = {(c['source_index'], c['beam_0based'], c['test_noise_law']): c
                for c in old['results']['fusion_wide_front']['cases']}
    result = dict(status='evaluating_assigned_raw_native_beams_NO_FIR', selected=selected,
        worker=args.worker_index, known_cases={name: [] for name in models}, measured=[],
        candidate_input_filter='none', real_clean_truth=False, quality_accepted=False)
    path = out/f'worker_{args.worker_index}.json'
    if path.exists():
        raise FileExistsError('preserve prior shard evidence')
    start = time.monotonic()
    def update():
        result['elapsed_s'] = time.monotonic()-start
        save(path, result)
    update()
    try:
        references = {i: np.load(recovery/'native_arrays'/f'source_{i:02d}_complex128.npy', mmap_mode='r') for i in (6, 7)}
        for index in range(16):
            if index % args.workers != args.worker_index:
                continue
            i, beam = 6+index//8, index % 8
            original = np.asarray(references[i][beam], np.complex64)
            identity = {name: infer_raw(model, original, device) for name, model in models.items()}
            if args.random_ripple:
                from RandomRippleEvaluation import prepare_fields as ripple_fields
                item = next(r for r in admission['records'] if r['id'] == f'qiandao_s{i}_b{beam}')
                fields = ripple_fields(original, profile, device, item, i, beam)
                known_gate = np.load(item['noise_support_file'], mmap_mode='r')
                identity = {name: delta * known_gate for name, delta in identity.items()}
            else:
                fields = prepare_fields(original, profile, device, i, beam, 9189000+i*10+beam, .35)
            for field in fields:
                if not args.random_ripple:
                    prior = expected[(i, beam, field['law'])]
                    if (array_sha(field['noisy']) != prior['exact_noisy_complex64_SHA256']
                            or array_sha(original) != prior['exact_complex64_reference_SHA256']):
                        raise ValueError('exact32 held-out synthetic inputs must match previous experiment')
                for name, model in models.items():
                    # The raw received synthetic input goes directly to the NN.
                    delta = infer_raw(model, field['noisy'], device)
                    if args.random_ripple:
                        delta *= known_gate
                    case = metrics(original, field['noisy'], delta, identity[name])
                    case.update(source_index=i, beam_0based=beam, test_noise_law=field['law'],
                        candidate_input_frequency_FIR=False, candidate_output_frequency_FIR=False)
                    if args.random_ripple:
                        case.update(scenario=field['scenario'], scenario_SHA256=field['scenario_SHA256'],
                                    noise_generator_identical_to_training=True)
                    result['known_cases'][name].append(case)
                    if args.random_ripple and field['law'] == 'random_ripple_0':
                        destination = np.load(out/f'known_{name}_amplitude.npy', mmap_mode='r+')
                        destination[i - 6, beam] = np.abs(field['noisy']) * np.exp(delta)
                        destination.flush()
                if args.random_ripple and field['law'] == 'random_ripple_0':
                    for label, values in (('reference', original), ('noisy', field['noisy'])):
                        destination = np.load(out/f'known_{label}_amplitude.npy', mmap_mode='r+')
                        destination[i - 6, beam] = np.abs(values)
                        destination.flush()
            update()
            print(dict(stage='full32_known_test', worker=args.worker_index, source=i, beam=beam), flush=True)
        z = np.load(voyage/'native_source/measured_complex128.npy', mmap_mode='r')
        amplitude = np.load(voyage/'native_source/original_amplitude_float64.npy', mmap_mode='r')
        fields = {name: np.load(out/f'{name}_supported_log_delta.npy', mmap_mode='r+') for name in models}
        limits = np.load(voyage/'metadata/bottom_first_ranges.npy')
        meta = read(voyage/'metadata/envelope_cache_summary.json')
        c, fs = (meta['parts'][0]['parameters'][k] for k in ('c', 'BeamSigFs'))
        for index in range(16):
            if index % args.workers != args.worker_index:
                continue
            side, beam = divmod(index, 8)
            original = np.asarray(z[side, beam], np.complex64)
            row = dict(side=side, beam=beam, source_complex64_SHA256=array_sha(original), models={})
            for name, model in models.items():
                neural = infer_raw(model, original, device)
                # NO FIR/base-log difference is added. Only the NN's gain is applied.
                if args.random_ripple:
                    gate = np.load(out/'real_observed_noise_support.npy', mmap_mode='r')[side]
                    applied = neural * gate
                    if np.any(applied[gate == 0]):
                        raise ValueError('Protected real water column changed.')
                else:
                    applied = supported_total_delta(amplitude[side, beam], neural, limits[side],
                        np.arange(z.shape[-1]), c, fs)
                fields[name][side, beam] = applied
                fields[name].flush()
                row['models'][name] = dict(gain_SHA256=array_sha(applied), native_values=applied.size,
                    neural_RMS=float(np.sqrt(np.square(neural, dtype=np.float64).mean())))
            result['measured'].append(row)
            update()
            print(dict(stage='measured_NO_FIR', worker=args.worker_index, side=side, beam=beam), flush=True)
        result['status'] = 'completed_all_assigned_known_and_measured_NO_FIR_beams'
    except BaseException as exc:
        result.update(status='failed_preserved_NO_FIR_evaluation_shard', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['peak_CUDA_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sonar-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--random-ripple', action='store_true')
    args = parser.parse_args()
    if args.worker_index is not None:
        worker(args)
        return
    out = args.root/'evaluation'
    out.mkdir(exist_ok=False)
    recovery = args.sonar_root/'matched_processing_recovery_20261105_v1'
    voyage = recovery/'measured_full_voyage_20261105_v1'
    manifest = verify_metadata(voyage, recovery)
    verify_native(voyage, manifest)
    model_names = ('pretrain', 'adapt', 'previous_56M') if args.random_ripple else ('pretrain', 'adapt')
    if args.random_ripple:
        from RandomRippleEvaluation import prepare_real_support
        prepare_real_support(args.root, voyage)
        original_shape = np.load(recovery/'native_arrays/source_06_complex128.npy', mmap_mode='r').shape
        for name in ('reference', 'noisy', *model_names):
            field = np.lib.format.open_memmap(out/f'known_{name}_amplitude.npy', mode='w+',
                dtype=np.float32, shape=(2, *original_shape))
            del field
    for name in model_names:
        field = np.lib.format.open_memmap(out/f'{name}_supported_log_delta.npy',
            mode='w+', dtype=np.float32, shape=tuple(manifest['native_shape']))
        del field
    processes = []
    for index in range(args.workers):
        command = [sys.executable, '-u', str(Path(__file__)), '--root', str(args.root),
            '--sonar-root', str(args.sonar_root), '--workers', str(args.workers), '--worker-index', str(index)]
        if args.random_ripple:
            command.append('--random-ripple')
        processes.append(subprocess.Popen(command))
    report = dict(status='running_full_native_NO_FIR_comparison', worker_PIDs=[p.pid for p in processes],
        original_voyage_readonly_integrity_verified=True, known_cases={}, measured=[], quality_accepted=False)
    save(out/'run_summary.json', report)
    codes = [process.wait() for process in processes]
    report['worker_exit_codes'] = codes
    if any(codes):
        report['status'] = 'failed_preserved_full_NO_FIR_evaluation'
        save(out/'run_summary.json', report)
        raise RuntimeError('full raw-input evaluation shard failed; do not restart automatically')
    cases = {name: [] for name in model_names}
    seen = set()
    for index in range(args.workers):
        shard = read(out/f'worker_{index}.json')
        if shard['status'] != 'completed_all_assigned_known_and_measured_NO_FIR_beams':
            raise ValueError('incomplete actual native evaluation shard')
        for name in cases:
            cases[name].extend(shard['known_cases'][name])
        for row in shard['measured']:
            key = row['side'], row['beam']
            if key in seen:
                raise ValueError('duplicate measured beam')
            seen.add(key)
            for name in cases:
                a = np.load(out/f'{name}_supported_log_delta.npy', mmap_mode='r')[key]
                if not np.isfinite(a).all() or array_sha(a) != row['models'][name]['gain_SHA256']:
                    raise ValueError('complete saved native gain differs from inference')
            report['measured'].append(row)
    if seen != {(s, b) for s in range(2) for b in range(8)}:
        raise ValueError('every real measured beam required')
    laws = ('random_ripple_0', 'random_ripple_1') if args.random_ripple else ('wide', 'limited')
    expected = {(i, b, law) for i in (6, 7) for b in range(8) for law in laws}
    for name, items in cases.items():
        if len(items) != 32 or {(c['source_index'], c['beam_0based'], c['test_noise_law']) for c in items} != expected:
            raise ValueError('every complete known input/law required')
    report.update(status='completed_full32_known_and_all16_measured_NO_FIR_fields_pending_render',
                  known_results={name: aggregate(items) for name, items in cases.items()})
    save(out/'run_summary.json', report)
    if args.random_ripple:
        from RenderRandomRippleSonarComparison import render
    else:
        from RenderNoFIRSonarComparison import render
    report['render'] = render(args.root, args.sonar_root)
    report['status'] = 'completed_full_NO_FIR_evaluation_and_native_images_NOT_quality_acceptance'
    save(out/'run_summary.json', report)


if __name__ == '__main__':
    main()
