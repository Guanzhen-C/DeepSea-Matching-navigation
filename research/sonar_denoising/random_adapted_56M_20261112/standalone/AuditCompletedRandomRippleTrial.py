"""Audit actual completed training, tensors, ledgers and full output artifacts.

Run AFTER the original unit terminates successfully. A successful execution
audit does not certify true device-noise removal or a clean real-world target.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess
import tarfile
import zlib

import numpy as np
import torch

from AuditRandomRippleProgress import audit_prefix, phase_admission_path
from NoFIRSonarData import read, save, sha, json_text
from NoFIRSonarWavelet import NoFIRSonarMWCNN, PROTOCOL
from NativeSonarPaletteStream import decoded_index_rows
from TrainNativeLocalSonarWavelet import array_sha
from caris_orangecopper import caris_orangecopper


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit_phase(root, phase, requested):
    stage = read(root / phase / 'run_summary.json')
    check(stage['status'] == 'completed_actual_no_FIR_phase_training_NOT_quality_acceptance', 'Incomplete training phase.')
    check(stage['steps_completed'] == stage['steps_requested'] == requested, 'Requested optimizer updates missing.')
    check(stage['world_size'] == stage['global_batch_samples'] == 4, 'Different training batch/world.')
    check(stage['actual_global_examples'] == 4 * requested, 'Incorrect actual exposure count.')
    prefix = audit_prefix(root, phase)
    check(prefix['actual_updates_verified'] == requested, 'Incomplete actual input ledgers.')
    check(prefix['all_declared_phase_sources_actually_augmented'], 'A training source never received nonzero noise.')
    for rank in range(4):
        receipt_path = root / phase / f'rank_{rank}_receipt.json'
        receipt = read(receipt_path)
        check(receipt['steps'] == receipt['examples'] == requested, 'Incomplete rank optimizer execution.')
        check(receipt['ledger_SHA256'] == sha(root / phase / f'actual_rank_{rank}_inputs.jsonl'), 'Rank input ledger changed.')
        expected = next(item for item in stage['rank_ledgers'] if item['rank'] == rank)
        check(sha(receipt_path) == expected['receipt_SHA256'], 'Rank execution receipt changed.')
    selected = stage['selected']
    best = min(stage['validation_history'], key=lambda v: (v['metrics']['selection_score'], v['step']))
    check(selected['step'] == best['step'], 'Selected weight does not match the predeclared validation criterion.')
    path = root / phase / selected['file']
    check(sha(path) == selected['SHA256'], 'Selected weight bytes changed.')
    state = torch.load(path, map_location='cpu', weights_only=True)
    check(state['random_ripple_augmentation'] is True and state['noise_profile_SHA256'] == sha(root / 'random_ripple_profile.json'), 'Checkpoint has another noise protocol.')
    check(state['admission_manifest_SHA256'] == sha(phase_admission_path(root, phase)), 'Checkpoint has another admitted dataset.')
    check(state['feature_protocol'] == PROTOCOL and state['model_family'] == NoFIRSonarMWCNN.family, 'Wrong selected model family.')
    check(state['phase'] == phase and state['step'] == selected['step'] and state['width'] == 96, 'Checkpoint descriptor mismatch.')
    check(sum(v.numel() for v in state['model'].values()) == 56083513, 'Wrong actual model tensor count.')
    check(all(bool(torch.isfinite(v).all()) for v in state['model'].values()), 'Nonfinite selected model tensor.')
    check(bool(torch.any(state['model']['network.tail.weight'] != 0)), 'Selected model is still the untrained zero output head.')
    if phase == 'pretrain':
        check(selected['step'] > 0, 'Untrained initialization selected instead of a trained model.')
    result = dict(selected=selected, actual_updates=requested,
        selected_adaptation_zero_means_warm_public_model=phase == 'adapt' and selected['step'] == 0,
        tensors_finite=True, actual_parameters=56083513,
        full_input_audit=prefix, actual_checkpoint_SHA256=sha(path))
    del state
    return result


def audit_png(path, descriptor, expected_shape, palette):
    expected_sha = descriptor['SHA256']
    check(sha(path) == expected_sha, 'Native output PNG changed.')
    digest, rows = hashlib.sha256(), 0
    for row in decoded_index_rows(path, palette):
        check(len(row) == expected_shape[0], 'Native PNG columns missing.')
        digest.update(row.tobytes())
        rows += 1
    encoding = descriptor['encoding']
    check(rows == expected_shape[1], 'Native PNG rows missing.')
    check(digest.hexdigest() == encoding['index_SHA256'], 'Native PNG indices do not match numeric renderer.')
    check([encoding['native_width'], encoding['native_height']] == list(expected_shape), 'Native shape descriptor mismatch.')
    return dict(SHA256=expected_sha, native_shape=list(expected_shape), full_rows_checked=rows,
                index_SHA256=digest.hexdigest())


def audit_pdf(pdf, image):
    info = read(pdf.with_suffix('.校验.json'))
    check(sha(pdf) == info['output_SHA256'] and sha(image) == info['source_SHA256'], 'PDF or its native source changed.')
    check(info['layout'] == 'waterfall' and info['rotated'] is False and info['flipped'] is True, 'Wrong waterfall orientation.')
    check(not info['cropped'] and not info['resampled'] and not info['extra_denoising'], 'Unexpected PDF image alteration.')
    check(info['original_compressed_pixels_and_palette_exact'], 'PDF is not the original native raster.')
    source_digest = hashlib.sha256()
    with image.open('rb') as stream:
        check(stream.read(8) == b'\x89PNG\r\n\x1a\n', 'Invalid native PNG signature.')
        while True:
            header = stream.read(8)
            check(len(header) == 8, 'Truncated source PNG.')
            length, kind = struct.unpack('>I4s', header)
            check(length <= 16 * 1024**2, 'Unexpected native PNG chunk length.')
            payload, crc = stream.read(length), stream.read(4)
            check(len(payload) == length and len(crc) == 4, 'Truncated PNG chunk.')
            check(struct.unpack('>I', crc)[0] == zlib.crc32(kind + payload) & 0xffffffff, 'Source PNG chunk CRC differs.')
            if kind == b'IDAT':
                source_digest.update(payload)
            if kind == b'IEND':
                break
    check(source_digest.hexdigest() == info['source_IDAT_SHA256'], 'Actual source PNG image stream differs from PDF descriptor.')
    digest = hashlib.sha256()
    with pdf.open('rb') as stream:
        stream.seek(info['image_stream_offset'])
        remaining = info['image_stream_bytes']
        while remaining:
            block = stream.read(min(remaining, 4 * 1024**2))
            check(bool(block), 'Truncated embedded PDF image.')
            digest.update(block)
            remaining -= len(block)
    check(digest.hexdigest() == info['source_IDAT_SHA256'] == info['embedded_IDAT_SHA256'], 'Actual embedded PDF image bytes differ.')
    return dict(PDF_SHA256=sha(pdf), source_native_SHA256=sha(image),
                embedded_dimensions=info['embedded_dimensions'], actual_embedded_stream_verified=True,
                unrotated_vertical_waterfall=True)


def audit(root, *, job_root):
    root, job_root = Path(root), Path(job_root)
    overall = read(root / 'run_summary.json')
    check(overall['status'] == 'completed_random_ripple_training_full_evaluation_and_render_NOT_quality_acceptance', 'Full requested execution is not finished.')
    check(overall['actual_training_completed'] is True, 'Actual training incomplete.')
    check(overall['admission_manifest_SHA256'] == sha(phase_admission_path(root, 'pretrain')), 'Initial public admission provenance differs.')
    support_transition = audit_native_support_transition(root, job_root)
    preflight = read(job_root / 'launch_preflight.json')
    previous = job_root.parent / 'nofir_multisource_20261108_v2'
    check(sha(previous / 'code_snapshot.tar.gz') == preflight['old_snapshot_archive_SHA256'], 'Original dependency snapshot changed.')
    old_files = 0
    with tarfile.open(previous / 'code_snapshot.tar.gz') as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = Path(member.name)
            check(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe old source snapshot entry.')
            check(sha(previous / relative) == hashlib.sha256(archive.extractfile(member).read()).hexdigest(), 'Original frozen dependency changed.')
            old_files += 1
    check(old_files == preflight['old_snapshot_files_checked'], 'Original dependency file set differs.')
    code = overall['source_code_SHA256']
    with tarfile.open(job_root / 'code_snapshot.tar.gz') as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        check(len(members) == len(code), 'Source snapshot file set differs.')
        for member in members:
            relative = Path(member.name)
            check(relative.parts[0] == 'standalone' and '..' not in relative.parts, 'Unsafe source snapshot entry.')
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            check(digest == code[relative.name] == sha(job_root / relative), 'Executed source snapshot changed.')
    phases = {phase: audit_phase(root, phase, steps) for phase, steps in (('pretrain', 24000), ('adapt', 8000))}
    for phase in phases:
        check(phases[phase]['selected'] == overall[phase + '_selected'], 'Pipeline selected another checkpoint.')
    evaluation = read(root / 'evaluation/run_summary.json')
    check(evaluation['status'] == 'completed_full_NO_FIR_evaluation_and_native_images_NOT_quality_acceptance', 'Full native evaluation incomplete.')
    check(evaluation['worker_exit_codes'] == [0] * 4, 'An evaluation worker did not finish.')
    previous_selected = None
    for rank in range(4):
        shard = read(root / f'evaluation/worker_{rank}.json')
        for phase in phases:
            check(shard['selected'][phase] == phases[phase]['selected'], 'An evaluation shard loaded another selected weight.')
        baseline = shard['selected']['previous_56M']
        if previous_selected is None:
            previous_selected = baseline
        else:
            check(previous_selected == baseline, 'Evaluation workers used different previous baselines.')
    check(sha(Path(previous_selected['source_root']) / previous_selected['file']) == previous_selected['SHA256'], 'Original 56M baseline changed.')
    wanted = {(i, b, f'random_ripple_{v}') for i in (6, 7) for b in range(8) for v in range(2)}
    same_inputs = {}
    for name in ('previous_56M', 'pretrain', 'adapt'):
        result = evaluation['known_results'][name]
        cases = result['cases']
        keys = {(c['source_index'], c['beam_0based'], c['test_noise_law']) for c in cases}
        check(len(cases) == 32 and keys == wanted, 'Missing full-beam known-input test.')
        for case in cases:
            key = (case['source_index'], case['beam_0based'], case['test_noise_law'])
            pair = (case['exact_complex64_reference_SHA256'], case['exact_noisy_complex64_SHA256'], case['scenario_SHA256'])
            if key in same_inputs:
                check(same_inputs[key] == pair, 'Models were evaluated on different corrupted inputs.')
            same_inputs[key] = pair
            check(case['noise_generator_identical_to_training'] and case['missing_support_delta_exact_zero'], 'Different test generator or missing support changed.')
    real = evaluation['measured']
    keys = {(r['side'], r['beam']) for r in real}
    check(len(real) == 16 and keys == {(s, b) for s in range(2) for b in range(8)}, 'Incomplete actual voyage.')
    gates = np.load(root / 'evaluation/real_observed_noise_support.npy', mmap_mode='r')
    real_fields = {n: np.load(root / f'evaluation/{n}_supported_log_delta.npy', mmap_mode='r') for n in ('previous_56M', 'pretrain', 'adapt')}
    for row in real:
        side, beam = row['side'], row['beam']
        for name, field in real_fields.items():
            values = field[side, beam]
            check(np.isfinite(values).all(), 'Nonfinite actual restored field.')
            check(array_sha(values) == row['models'][name]['gain_SHA256'], 'Saved whole-field restoration differs from inference.')
            check(not np.any(values[gates[side] == 0]), 'Protected real water-column correction is not zero.')
    renderer = read(root / 'evaluation/render_summary.json')
    check(renderer['status'] == 'completed_native_images_and_unrotated_waterfall_PDF_NOT_quality_acceptance', 'Complete renderer output missing.')
    native_shape = [2 * gates.shape[-1], 4 * (gates.shape[-2] + 1)]
    palette = (caris_orangecopper(256) * 255).astype(np.uint8)
    outputs = {name: audit_png(root / 'evaluation' / item['file'], item, native_shape, palette)
               for name, item in renderer['images'].items()}
    known = renderer['known_added_noise_comparison']
    known_outputs = {name: audit_png(root / 'evaluation' / item['file'], item, known['native_dimensions'], palette)
                     for name, item in known['images'].items()}
    pdf = root / 'evaluation' / renderer['PDF']
    pdf_result = audit_pdf(pdf, root / 'evaluation/random_adapted_56M_native.png')
    return dict(status='passed_full_requested_execution_and_artifact_integrity_NOT_real_quality_certification',
        actual_completed_optimizer_updates=32000, actual_global_sample_exposures=128000,
        old_frozen_dependency_files_verified_unchanged=old_files,
        native_support_transition=support_transition,
        previous_56M_baseline=previous_selected,
        training=phases, full_known_cases_per_model=32, full_real_voyage_beams=16,
        identical_corrupted_inputs_across_three_models=True, native_outputs=outputs,
        known_native_outputs=known_outputs, PDF=pdf_result,
        complete_run_report_SHA256=sha(root / 'run_summary.json'),
        evaluation_report_SHA256=sha(root / 'evaluation/run_summary.json'),
        real_clean_truth=False, image_visual_review_still_required=True, real_quality_accepted=False)


def audit_native_support_transition(root, job_root):
    root, job_root = Path(root), Path(job_root)
    path = root / 'native_support_transition.json'
    if not path.exists():
        return None
    from ObservedBottomSupportArtifacts import verified_metadata
    from MeasuredStripeNoiseModel import noise_support_from_bounds
    from SonarNadirProtection import from_reference_png, mosaic_seed_mask
    transition = read(path)
    check(transition['status'] == 'activated_before_native_phase_unchanged_public_training', 'Native support transition is not complete.')
    old_path, new_path = root / 'pretrain_admission_snapshot.json', root / 'admission_manifest.json'
    check(sha(old_path) == transition['old_admission_SHA256'], 'Original public admission snapshot changed.')
    check(sha(new_path) == transition['new_admission_SHA256'], 'Future native admission changed.')
    old, new = read(old_path), read(new_path)
    check(len(old['records']) == len(new['records']) and
          {r['id'] for r in old['records']} == {r['id'] for r in new['records']}, 'Source membership changed during native support preparation.')
    old_public = [r for r in old['records'] if r['domain'] != 3]
    new_public = [r for r in new['records'] if r['domain'] != 3]
    check(old_public == new_public, 'Public data were modified during public optimization.')
    check(hashlib.sha256(json_text(old_public, sort_keys=True).encode()).hexdigest() == transition['public_records_fingerprint'], 'Public data fingerprint differs.')
    allowed = {'noise_support_file', 'noise_support_SHA256', 'noise_support_file_guard', 'global_reference_RMS'}
    before = {r['id']: r for r in old['records']}
    for record in new['records']:
        previous = before[record['id']]
        check({k: v for k, v in record.items() if k not in allowed} ==
              {k: v for k, v in previous.items() if k not in allowed}, 'Original observations or split roles changed.')
        if record['domain'] == 3:
            first_beam = next(r for r in new['records'] if r['id'] == record['id'].rsplit('_b', 1)[0] + '_b0')
            check(all(record[k] == first_beam[k] for k in
                      ('noise_support_file', 'noise_support_SHA256', 'noise_support_file_guard')),
                  'Synchronous beams do not use the same canonical source support.')
    sonar = job_root.parent
    arrays, metadata, evidence = verified_metadata(sonar / 'observed_bottom_support_20261106_v1',
        sonar / 'front_target_alignment_20261105_v1/teacher_bundle/reports/native_processed_manifest.json')
    check({tuple(p['sources']) for p in transition['pairs']} == {(0, 1), (2, 3), (4, 5), (6, 7)}, 'Missing source-pair support audit.')
    palette = (caris_orangecopper(256) * 255).astype(np.uint8)
    verified = []
    for pair in transition['pairs']:
        first, second = pair['sources']
        entries = [next(r for r in new['records'] if r['id'] == f'qiandao_s{i}_b0') for i in (first, second)]
        h, w = entries[0]['shape']
        seeds = []
        for index in (first, second):
            source = metadata['sources'][index]
            seeds.append(noise_support_from_bounds(arrays[index]['bottom_first_float64'],
                source['sound_speed_m_s'], source['sampling_frequency_Hz'], w) == 0)
        image = root / f'canonical_native_support_v1/sources_{first:02d}_{second:02d}_working_reference_native.png'
        check(sha(image) == pair['reference_PNG_SHA256'], 'Canonical source-reference waterfall changed.')
        required, component, _ = from_reference_png(image, palette, h, w,
                                                    seed_mask=mosaic_seed_mask(np.stack(seeds)))
        for side, entry in enumerate(entries):
            previous = before[entry['id']]
            old_gate = np.load(previous['noise_support_file'], mmap_mode='r')
            new_gate = np.load(entry['noise_support_file'], mmap_mode='r')
            check(sha(entry['noise_support_file']) == entry['noise_support_SHA256'], 'Canonical native support changed.')
            expected = np.array(old_gate)
            expected[required[side]] = 0
            check(np.array_equal(new_gate, expected), 'Unannounced native support modification.')
            check(not np.any(new_gate[required[side]]), 'A canonical nearblack source sample can receive noise.')
        verified.append(dict(sources=pair['sources'], canonical_nearblack_pixels=int(component.sum()),
                             missing_canonical_source_samples=0))
    return dict(status='verified_phase_specific_admission_and_full_canonical_black_protection',
        old_admission_SHA256=sha(old_path), new_admission_SHA256=sha(new_path),
        public_records_exactly_unchanged=True, raw_observations_and_splits_unchanged=True,
        current_public_optimizer_inputs_changed=False, training_code_changed=False,
        complete_native_pairs=verified, source_metadata_evidence=evidence)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--job-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--unit')
    p.add_argument('--transition-only', action='store_true', help='Audit future native support only; does NOT claim training completion.')
    a = p.parse_args()
    check(not a.output.exists(), 'Preserve previous completion audit.')
    unit_state = None
    if a.unit and not a.transition_only:
        raw = subprocess.check_output(['systemctl', '--user', 'show', a.unit, '-p', 'MainPID', '-p', 'ExecMainStatus', '-p', 'SubState'], text=True)
        unit_state = dict(line.split('=', 1) for line in raw.splitlines())
        check(unit_state.get('MainPID') == '0' and unit_state.get('ExecMainStatus') == '0', 'Original execution unit is not successfully terminal.')
    if a.transition_only:
        result = audit_native_support_transition(a.root, a.job_root)
        check(result is not None, 'No native support transition exists.')
        result['training_completed'] = False
    else:
        result = audit(a.root, job_root=a.job_root)
    result['original_unit_state'] = unit_state
    result['audit_source_SHA256'] = sha(Path(__file__))
    result['ledger_audit_source_SHA256'] = sha(Path(audit_prefix.__code__.co_filename))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    save(a.output, result)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
