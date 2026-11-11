"""Audit the actual large-model floating-point result before map projection.

No edge repair, FIR, RGB decoding, palette mapping, or Shotwell enhancement
belongs to this input branch. Spatial projection is handled by the existing
scene builder; nominal coordinates are NOT independently surveyed truth.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

NATIVE_SHAPE = (2, 8, 2020, 12189)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require_direct_model_policy(report, *, edge_patch=None, appearance_lut=None):
    if edge_patch is not None or appearance_lut is not None:
        raise ValueError('Large-model-only scene forbids edge repair and appearance enhancement.')
    if (report.get('status') != 'completed_full_2020_records_all16_beams_FP32_parity_NOT_clean_truth_quality'
            or report.get('model') != 'random_adapted_56M'
            or report.get('parameters') != 56083513
            or report.get('shape') != list(NATIVE_SHAPE)
            or report.get('physical_records') != 2020 or report.get('beam_channels') != 16
            or report.get('precision') != 'FP32' or report.get('TF32') is not False
            or report.get('input_filter') != 'none' or report.get('post_filter') != 'none'
            or report.get('native_sampling_resized') is not False
            or report.get('protected_samples_unchanged') is not True):
        raise ValueError('A complete, unfiltered native FP32 large-model result is required.')
    if (not np.isfinite(report.get('max_reference_gain_abs_error', np.inf))
            or report['max_reference_gain_abs_error'] > report.get('gain_absolute_tolerance', 0)):
        raise ValueError('Large-model inference did not pass its predeclared numerical parity check.')


def audit_gain_values(raw, gain, beams, amplitude_paths, amplitude_hashes):
    """Stream one channel/chunk at a time; SHA covers numeric gain, not RGB."""
    if raw.shape != gain.shape or raw.dtype != np.float64 or gain.dtype != np.float32:
        raise ValueError('Aligned native float64 source magnitude and float32 log correction required.')
    if len(beams) != 16 or [b['index'] for b in beams] != list(range(16)):
        raise ValueError('All sixteen unique model channels must be certified.')
    for index, record in enumerate(beams):
        side, beam = divmod(index, 8)
        if (record['side'], record['beam']) != (side, beam):
            raise ValueError('Side/beam identity changed.')
        amplitude_path = Path(amplitude_paths[index])
        if sha_file(amplitude_path) != amplitude_hashes[index]:
            raise ValueError('Original model input amplitude checksum changed.')
        original = np.load(amplitude_path, mmap_mode='r', allow_pickle=False)
        if original.shape != raw.shape[2:] or original.dtype != np.float64:
            raise ValueError('Original model amplitude shape/precision changed.')
        digest = hashlib.sha256()
        for start in range(0, raw.shape[2], 128):
            stop = start+128
            magnitude = raw[side, beam, start:stop]
            delta = gain[side, beam, start:stop]
            if (not np.array_equal(magnitude, original[start:stop])
                    or not np.isfinite(magnitude).all() or (magnitude < 0).any()
                    or not np.isfinite(delta).all()):
                raise ValueError('Model/source mismatch or non-finite amplitude/correction.')
            digest.update(np.ascontiguousarray(delta).tobytes())
        if digest.hexdigest() != record['gain_SHA256']:
            raise ValueError('Actual full gain channel differs from completed inference report.')


def load_direct_model_gain(raw, gain_path, report_path, bundle, *, edge_patch=None, appearance_lut=None):
    report = json.loads(Path(report_path).read_text())
    require_direct_model_policy(report, edge_patch=edge_patch, appearance_lut=appearance_lut)
    manifest = json.loads((Path(bundle)/'manifest.json').read_text())
    if any(report.get(key) != manifest.get(key) for key in
           ('model', 'parent_checkpoint_SHA256', 'parameter_tensor_SHA256', 'shape', 'parameters')):
        raise ValueError('Completed model and original input bundle provenance differ.')
    if raw.shape != NATIVE_SHAPE or sha_file(gain_path) != report['gain_file_SHA256']:
        raise ValueError('Not the complete, certified large-model output.')
    gain = np.load(gain_path, mmap_mode='r', allow_pickle=False)
    relative = [f'beam_{i:02d}/amplitude.npy' for i in range(16)]
    audit_gain_values(raw, gain, report['beams'], [Path(bundle)/p for p in relative],
                     [manifest['files'][p] for p in relative])
    provenance = dict(model=report['model'], parameters=report['parameters'],
        parent_checkpoint_SHA256=report['parent_checkpoint_SHA256'],
        parameter_tensor_SHA256=report['parameter_tensor_SHA256'],
        gain_file_SHA256=report['gain_file_SHA256'], model_report_SHA256=sha_file(report_path),
        original_model_bundle_manifest_SHA256=sha_file(Path(bundle)/'manifest.json'),
        all_original_native_amplitudes_equal_actual_model_inputs=True,
        all16_numeric_gain_channels_verified=True, edge_repair_used=False,
        FIR_used=False, brightness_or_contrast_enhancement_used=False,
        RGB_or_PDF_input_used=False, native_amplitude_formula='original_float64_amplitude * exp(model_float32_log_gain)',
        real_clean_truth=False)
    return gain, provenance
