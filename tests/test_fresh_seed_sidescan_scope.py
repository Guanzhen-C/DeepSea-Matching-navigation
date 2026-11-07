from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from ValidateFreshSeedSideScan import require_primary_improvement


def good():
    return dict(status='completed_independent_full_identical_input_mapped_range_comparison',
        samples=62458, all_received_events_verified=True, DR_and_Local_bit_exact=True,
        full_original_input_file_SHA256_reverified=True,
        baseline_Global=dict(RMSE_m=11.7, P95_m=21.1, max_m=35.1),
        candidate_Global=dict(RMSE_m=5.3, P95_m=11.6, max_m=23.2))


def test_complete_full_mean_and_tail_improvement_is_required_before_new_data():
    require_primary_improvement(good())


@pytest.mark.parametrize('key', ['RMSE_m', 'P95_m', 'max_m'])
def test_new_data_is_not_consumed_to_mask_a_failed_primary_metric(key):
    d = good(); d['candidate_Global'][key] = d['baseline_Global'][key]+1
    with pytest.raises(ValueError): require_primary_improvement(d)


@pytest.mark.parametrize('change', [dict(samples=7576), dict(status='running'),
    dict(all_received_events_verified=False), dict(DR_and_Local_bit_exact=False)])
def test_partial_or_unverified_primary_cannot_trigger_blind_validation(change):
    d = good(); d.update(change)
    with pytest.raises(ValueError): require_primary_improvement(d)
