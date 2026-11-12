"""Small observed detection metadata, not a clean-water/terrain segmenter."""
import numpy as np

from FullMeasuredSonarProtocol import supported_total_delta


PROTOCOL = 'original_selected_depth_bottom_first_target_same_final_support_v1'


def detected_bounds(record, sound_speed):
    """Read ONLY scalar bottom and first target row; preserve invalid records."""
    if not np.isfinite(sound_speed) or sound_speed <= 0:
        raise ValueError('positive recorded sound speed required')
    bottom, first, reasons = np.nan, np.nan, []
    if 'BottomTrackingInfo' not in record:
        reasons.append('missing_BottomTrackingInfo')
    else:
        item = record['BottomTrackingInfo']
        if len(item.shape) != 2 or min(item.shape) < 1:
            reasons.append('invalid_BottomTrackingInfo_shape')
        else:
            bottom = float(item[0, 0])
    if 'TargetDelayInfo' not in record:
        reasons.append('missing_TargetDelayInfo')
    else:
        item = record['TargetDelayInfo']
        if len(item.shape) != 2 or item.shape[0] < 1 or item.shape[1] != 8:
            reasons.append('invalid_TargetDelayInfo_shape')
        else:
            # Same median convention as the existing final application, NOT
            # nanmedian, nearest valid target, or a made-up fallback detection.
            first = float(np.median(np.asarray(item[0, :], np.float64))*sound_speed/2)
    if not np.isfinite([bottom, first]).all():
        reasons.append('nonfinite_bottom_or_first_target')
    elif bottom <= 0 or first <= bottom:
        reasons.append('nonpositive_or_unordered_bottom_first_target')
    return np.asarray([bottom, first], np.float64), not reasons, reasons


def support_summary(bounds, valid, sound_speed, sampling_frequency, range_bins, *, record_block=64):
    """All original range cells, using the EXISTING final support function.

    Counts are per side-record/range cell, not multiplied by eight identical
    support copies. Invalid records remain exported; no physical label invented.
    """
    bounds, valid = np.asarray(bounds), np.asarray(valid)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or valid.shape != (len(bounds),)
            or valid.dtype != np.bool_ or min(range_bins, record_block) < 1
            or not np.isfinite([sound_speed, sampling_frequency]).all()
            or min(sound_speed, sampling_frequency) <= 0):
        raise ValueError('aligned native bounds/validity and recorded positive units required')
    groups = []
    for left, right in ((0, 512), (512, 2048), (2048, 6144), (6144, range_bins)):
        left, right = min(left, range_bins), min(right, range_bins)
        if right > left:
            groups.append(dict(range_bins_half_open=[left, right], native_values=0,
                support_exact_zero_values=0, support_transition_values=0, support_exact_one_values=0,
                support_sum_float64=0.))
    fully_protected = 0
    selected = bounds[valid]
    for start in range(0, len(selected), record_block):
        block = selected[start:start+record_block]
        ones = np.ones((len(block), range_bins), np.float32)
        support = supported_total_delta(ones, ones, block, np.arange(range_bins), sound_speed, sampling_frequency)
        fully_protected += int(np.all(support[:, :min(2048, range_bins)] == 0, axis=1).sum())
        for g in groups:
            left, right = g['range_bins_half_open']
            part = support[:, left:right]
            g['native_values'] += part.size
            g['support_exact_zero_values'] += int((part == 0).sum())
            g['support_exact_one_values'] += int((part == 1).sum())
            g['support_transition_values'] += int(((part > 0) & (part < 1)).sum())
            g['support_sum_float64'] += float(part.sum(dtype=np.float64))
    for g in groups:
        if sum(g[k] for k in ('support_exact_zero_values', 'support_transition_values', 'support_exact_one_values')) != g['native_values']:
            raise ValueError('complete disjoint original support categories required')
        g['mean_support'] = g['support_sum_float64']/g['native_values'] if g['native_values'] else None
    if sum(g['native_values'] for g in groups) != int(valid.sum())*range_bins:
        raise ValueError('every valid selected record and original range bin required')
    return dict(selected_records=len(bounds), valid_detection_records=int(valid.sum()),
        invalid_detection_records=int((~valid).sum()), first_2048_bins_fully_protected_records=fully_protected,
        groups=groups, invalid_records_preserved_not_imputed=True,
        support_formula_is_existing_final_application=True, categories_are_NOT_clean_or_physical_ground_truth=True,
        counts_do_not_include_repeated_eight_beam_copies=True)
