from copy import deepcopy
from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from AuditMatched172Validation import check_cases


def evidence():
    names = ['old', 'a', 'b', 'c', 'd', 'e']
    bundle = dict(native_sources=[dict(source_index=i, shape=[8, 4, 17]) for i in (6, 7)],
                  selected_weights={n: {} for n in names})
    cases = [dict(source_index=i, beam_0based=b, test_noise_law=law,
                  native_values=68, observed_values=60, missing_support_delta_exact_zero=True,
                  source_observation_is_not_clean_truth=True,
                  exact_complex64_reference_SHA256=f'{i}-{b}', exact_noisy_complex64_SHA256=f'{i}-{b}-{law}',
                  identity_log_delta_SHA256=f'{i}-{b}-id', seed=i*10+b, noise_scale=.35,
                  input_log_SSE=60., output_log_SSE=15., identity_log_SSE=.6)
             for i in (6, 7) for b in range(8) for law in ('wide', 'limited')]
    result = dict(cases=cases, native_values=32*68, input_log_RMSE=1., output_log_RMSE=.5,
                  identity_extra_log_RMS=math.sqrt(.01))
    report = dict(smoke=False,
                  status='completed_remote_selected_weights_full_native_validation_not_real_quality_acceptance',
                  training_restarted=False, original_reports_rewritten=False, quality_accepted=False,
                  physical_beams_completed=16, full_native_values_per_model=32*68,
                  results={n: deepcopy(result) for n in names})
    return report, bundle


def test_all_native_counts_common_noise_and_SSE_recomputed():
    report, bundle = evidence()
    metrics = check_cases(report, bundle)
    assert len(metrics) == 6
    assert metrics['a']['output_log_RMSE'] == .5


@pytest.mark.parametrize('bad', ['missing_law', 'different_noise', 'wrong_SSE', 'smoke', 'fake_acceptance'])
def test_incomplete_or_inconsistent_evidence_rejected(bad):
    report, bundle = evidence()
    if bad == 'missing_law':report['results']['b']['cases'].pop()
    elif bad == 'different_noise':report['results']['b']['cases'][0]['exact_noisy_complex64_SHA256'] = 'different'
    elif bad == 'wrong_SSE':report['results']['b']['cases'][0]['output_log_SSE'] += 100
    elif bad == 'smoke':report['smoke'] = True
    else:report['quality_accepted'] = True
    with pytest.raises(ValueError):check_cases(report, bundle)
