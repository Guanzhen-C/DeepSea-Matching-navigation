"""Read the audited small metadata packet; never require raw MAT on 172."""
from pathlib import Path
import hashlib
import json

import numpy as np

from ObservedBottomSupportMetadata import PROTOCOL
from FullMeasuredSonarProtocol import supported_total_delta


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()


def verified_metadata(root, source_manifest):
    root = Path(root)
    report = json.loads((root/'run_summary.json').read_text())
    if report['status'] != 'completed_all_selected_observed_depth_metadata_and_existing_support_coverage_NOT_clean_labels':
        raise ValueError('completed all-source observed metadata export required')
    audit = json.loads((root/'independent_native_metadata_audit.json').read_text())
    manifest = json.loads(Path(source_manifest).read_text())
    expected_count = sum(s['native_records'] for s in manifest['sources'])
    if (report['protocol'] != PROTOCOL or report['sources_completed'] != 8
            or len(manifest['sources']) != 8 or [s['source_index'] for s in report['sources']] != list(range(8))
            or report['source_manifest_SHA256'] != sha(source_manifest)
            or report['selected_records_exported'] != expected_count
            or audit['status'] != 'passed_all_selected_raw_depth_metadata_and_support_replay_NOT_quality_acceptance'
            or audit['producer_report_SHA256'] != sha(root/'run_summary.json')
            or audit['source_manifest_SHA256'] != sha(source_manifest)
            or audit['original_selected_records_checked'] != expected_count
            or audit['source_files_checked'] != 8):
        raise ValueError('matching complete all-source independent raw metadata audit required')
    for n, digest in report['source_code_SHA256'].items():
        if sha(root/'source_snapshot'/n) != digest:
            raise ValueError('original executed metadata source snapshot changed: '+n)
    if (sha(root/'audit_source_snapshot/AuditObservedBottomSupport.py') != audit['auditor_SHA256']
            or sha(root/'source_snapshot/FullMeasuredSonarProtocol.py') != sha(Path(__file__).with_name('FullMeasuredSonarProtocol.py'))):
        raise ValueError('audited detection or original final application formula changed')
    arrays = {}
    for source, item in zip(manifest['sources'], report['sources']):
        i, count = item['source_index'], source['native_records']
        if (any(item[k] != source[k] for k in ('physical_part', 'side', 'split', 'native_records', 'native_range_bins'))
                or item['selected_identity_SHA256'] != source['selected_identity_SHA256']
                or item['parameter_file_SHA256'] != source['parameter_SHA256']
                or item['sampling_frequency_Hz'] != source['sampling_frequency_Hz']
                or item['original_selection']['zero_based_selected_indices_sha256'] != source['selected_zero_based_original_index_SHA256']):
            raise ValueError('original physical source/identity/record/parameter mapping changed')
        expected = dict(bottom_first_float64=([count, 2], '<f8'), valid_detection_bool=([count], '|b1'),
                        ping_identity_int64=([count, 3], '<i8'), original_cell_0based_int64=([count], '<i8'))
        if set(item['files']) != set(expected):
            raise ValueError('all native bounds/validity/identity/selection arrays required')
        values = {}
        for name, (shape, dtype) in expected.items():
            meta = item['files'][name]
            file = root/meta['file']
            value = np.load(file, allow_pickle=False)
            if (list(value.shape) != shape or value.dtype.str != dtype or file.stat().st_size != meta['bytes']
                    or sha(file) != meta['file_SHA256'] or array_sha(value) != meta['array_SHA256']):
                raise ValueError('native small metadata array changed or is incomplete')
            values[name] = value
        indices = values['original_cell_0based_int64']
        bounds = values['bottom_first_float64']
        valid = np.isfinite(bounds).all(axis=1) & (bounds[:, 0] > 0) & (bounds[:, 1] > bounds[:, 0])
        if (array_sha(indices) != source['selected_zero_based_original_index_SHA256'] or (np.diff(indices) <= 0).any()
                or not np.array_equal(valid, values['valid_detection_bool'])
                or int((~valid).sum()) != item['support_summary']['invalid_detection_records']):
            raise ValueError('original row order or explicit invalid detection handling changed')
        arrays[i] = values
    return arrays, report, dict(original_selected_records_checked=expected_count, all_eight_source_arrays_verified=True,
        original_local_raw_depth_audit_verified=True, raw_MAT_files_needed_on_172=False,
        support_is_observed_application_metadata_NOT_clean_truth=True, quality_accepted=False)


def support_for_core(arrays, report, source_index, core):
    """Exact existing support on original global core coordinates, no padding."""
    y, x, h, w = map(int, core)
    if source_index not in arrays:
        raise ValueError('original qualified source required')
    source = report['sources'][source_index]
    if (source['source_index'] != source_index or min(y, x) < 0 or min(h, w) < 1
            or y+h > source['native_records'] or x+w > source['native_range_bins']):
        raise ValueError('actual original native core coordinates required')
    values = arrays[source_index]
    if not values['valid_detection_bool'][y:y+h].all():
        raise ValueError('invalid observations must not be imputed or silently dropped')
    ones = np.ones((h, w), np.float32)
    return supported_total_delta(ones, ones, values['bottom_first_float64'][y:y+h], np.arange(x, x+w),
                                 source['sound_speed_m_s'], source['sampling_frequency_Hz'])
