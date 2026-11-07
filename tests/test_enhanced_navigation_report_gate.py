from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone'))
from BuildEnhancedNavigationComparison import require_full_execution


def complete():
    report = dict(status='completed_pure_side_scan_FEATURE_route_test',
        full_closed_loop_requested=True, full_source_snapshots_verified=True,
        event_counts={'sonar': 62458}, expected_events={'sonar': 62458},
        processed_pings=62458, submap_representation='adaptive_projected',
        observation_source='complex_IQ', requested_feature_matcher='LoFTR',
        historical_side_scan_loops_requested=True, secondary_denoising_used=False,
        MBES_XYZ_given_to_estimator=False, measured_height_likelihood_used=False,
        GPS_used=False, requested_distance_m=16488.743, complete_planned_loop_length_m=16488.743)
    review = dict(status='independently_verified_complete_executed_scope',
        complete_planned_route_verified=True, recorded_navigation_inputs_verified=True,
        paired_legacy_DR_bit_exact=True)
    return report, review


def test_full_gate_accepts_complete_evidence_without_a_favourable_error_requirement():
    report, review = complete()
    # Outcome is evaluated, not selected for a pretty curve or lower RMSE.
    report['metrics'] = {'Global': {'RMSE_m': 1000}}
    require_full_execution(report, review)


@pytest.mark.parametrize('field,value', [
    ('status', 'running'), ('processed_pings', 100),
    ('full_closed_loop_requested', False), ('secondary_denoising_used', True),
    ('MBES_XYZ_given_to_estimator', True), ('GPS_used', True),
    ('requested_feature_matcher', 'other'), ('requested_distance_m', 50.),
])
def test_incomplete_prefix_or_changed_observation_scope_is_not_a_final_report(field, value):
    report, review = complete()
    report[field] = value
    with pytest.raises(ValueError):
        require_full_execution(report, review)


def test_unverified_input_pairing_blocks_final_comparison():
    report, review = complete()
    review['paired_legacy_DR_bit_exact'] = False
    with pytest.raises(ValueError):
        require_full_execution(report, review)
