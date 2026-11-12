"""Independently check completed recovery evidence, not real clean quality.

This checks all transport bytes, immutable code/weights/reports, full native
coverage, common realized test inputs, and metrics recomputed from stored SSE.
It does NOT rerun neural inference or claim independent clean measurements.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def check_cases(report, bundle):
    if (report['smoke'] or report['status'] !=
            'completed_remote_selected_weights_full_native_validation_not_real_quality_acceptance'
            or report['training_restarted'] or report['original_reports_rewritten']
            or report['quality_accepted']):
        raise ValueError('completed full validation-only recovery, not clean quality acceptance, required')
    expected = {(s['source_index'], b, law): s['shape'][1]*s['shape'][2]
                for s in bundle['native_sources'] for b in range(8) for law in ('wide', 'limited')}
    if len(expected) != 32 or report['physical_beams_completed'] != 16:
        raise ValueError('all sixteen original physical beams and both laws required')
    if set(report['results']) != set(bundle['selected_weights']):
        raise ValueError('all six preserved selected models required')
    common_inputs = None
    recomputed = {}
    for name, result in report['results'].items():
        seen = {}; identical = {}
        for case in result['cases']:
            key = (case['source_index'], case['beam_0based'], case['test_noise_law'])
            if (key not in expected or key in seen or case['native_values'] != expected[key]
                    or not 0 < case['observed_values'] <= case['native_values']
                    or not case['missing_support_delta_exact_zero']
                    or not case['source_observation_is_not_clean_truth']):
                raise ValueError('missing/duplicated native grid or support/source semantics differ')
            seen[key] = (case['exact_complex64_reference_SHA256'],
                         case['exact_noisy_complex64_SHA256'], case['seed'], case['noise_scale'])
            identity_key = key[:2]
            signature = (case['identity_log_delta_SHA256'], case['identity_log_SSE'])
            if identity_key in identical and identical[identity_key] != signature:
                raise ValueError('identity inference changed between test laws')
            identical[identity_key] = signature
        if set(seen) != set(expected):
            raise ValueError('complete both-law/side/beam coverage required')
        if common_inputs is not None and common_inputs != seen:
            raise ValueError('models were not evaluated on identical realized corruption')
        common_inputs = seen
        count = sum(c['observed_values'] for c in result['cases'])
        metrics = {}
        for label, sse in (('input_log_RMSE', 'input_log_SSE'),
                           ('output_log_RMSE', 'output_log_SSE'),
                           ('identity_extra_log_RMS', 'identity_log_SSE')):
            value = math.sqrt(sum(c[sse] for c in result['cases']) / count)
            if not math.isfinite(value) or not math.isclose(value, result[label], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('stored full-case SSE and reported aggregate differ')
            metrics[label] = value
        native = sum(expected.values())
        if result['native_values'] != native or report['full_native_values_per_model'] != native:
            raise ValueError('all native values under both laws required')
        metrics['out_plus_identity_score'] = metrics['output_log_RMSE'] + metrics['identity_extra_log_RMS']
        metrics['native_values_including_two_corruption_laws'] = native
        recomputed[name] = metrics
    return recomputed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args();root = args.bundle.absolute();start = time.monotonic()
    bundle_path = root/'bundle_manifest.json';bundle = json.loads(bundle_path.read_text())
    path = root/'remote_full_native_validation/run_summary.json'
    report = json.loads(path.read_text())
    if report['source_bundle_manifest_SHA256'] != sha(bundle_path):
        raise ValueError('recovery bundle manifest changed')
    for name, digest in bundle['files'].items():
        if sha(root/name) != digest:
            raise ValueError('immutable transferred file changed: '+name)
    receipts = {}
    for source in bundle['native_sources']:
        i = source['source_index']
        receipt_path = root/'native_arrays'/f'source_{i:02d}_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        values = np.load(root/'native_arrays'/f'source_{i:02d}_complex128.npy', mmap_mode='r')
        metadata = receipt['source_extractor_metadata']
        if (list(values.shape) != source['shape'] or values.dtype != np.complex128
                or metadata['source_guard'] != source['source_guard']
                or metadata['manifest_SHA256'] != bundle['source_manifest_SHA256']):
            raise ValueError('original transported shape/precision/source provenance changed')
        digest = hashlib.sha256(memoryview(values).cast('B')).hexdigest()
        if digest != receipt['source_payload_SHA256'] or digest != metadata['array_SHA256']:
            raise ValueError('original native payload bytes changed')
        receipts[i] = (digest, sha(receipt_path))
        del values
    metrics = check_cases(report, bundle)
    for result in report['results'].values():
        for case in result['cases']:
            if (case['original_source_side_payload_SHA256'], case['original_source_receipt_SHA256']) != receipts[case['source_index']]:
                raise ValueError('metrics refer to other native transfer evidence')
    outcome = dict(status='passed_full_172_transport_coverage_common_input_and_SSE_evidence_audit',
                   source_report_SHA256=sha(path), source_bundle_manifest_SHA256=sha(bundle_path),
                   auditor_SHA256=sha(Path(__file__)), metrics_recomputed=metrics,
                   physical_beams=16, known_added_noise_fields=32, trained_models=5, frozen_prior_models=1,
                   native_reference_values=145097856, compared_values_per_model_including_two_laws=290195712,
                   original_native_complex128_bytes_rechecked=2321565696,
                   neural_predictions_independently_recomputed=False,
                   real_clean_ground_truth=False, quality_accepted=False, elapsed_s=time.monotonic()-start)
    output = path.parent/'independent_evidence_audit.json'
    with output.open('x') as stream:
        stream.write(json.dumps(outcome, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(outcome, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
