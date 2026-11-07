from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from AnalyzeMappedRangeNavigation import analyze, require_execution_scope


@pytest.mark.parametrize('change', [
    dict(status='running_lossless_navigation_input_replay'),
    dict(estimated_frames=62457),
    dict(events=dict(sonar=62457)),
    dict(mapped_range_changes=False),
    dict(evaluation_truth_read_only_after_estimation=False),
])
def test_partial_or_incomparable_replay_cannot_be_exported_as_full_improvement(tmp_path, change):
    source = tmp_path/'source'; source.mkdir()
    candidate = tmp_path/'candidate'; candidate.mkdir()
    output = tmp_path/'review'
    events = dict(sonar=62458, imu=1030546, dvl=103054, depth=206109)
    old = dict(status='completed_pure_side_scan_FEATURE_route_test',
               full_closed_loop_requested=True, processed_pings=62458, expected_events=events)
    new = dict(status='completed_lossless_navigation_input_replay', mapped_range_changes=True,
               estimated_frames=62458, evaluation_truth_read_only_after_estimation=True, events=events)
    new.update(change)
    (source/'run_summary.json').write_text(json.dumps(old))
    (candidate/'run_summary.json').write_text(json.dumps(new))
    with pytest.raises(ValueError, match='Complete identical-input'):
        analyze(source, candidate, output)
    assert not output.exists()


def test_short_completed_recording_requires_explicit_heldout_scope_and_honest_label():
    events = dict(sonar=7576, imu=125000, depth=25000, dvl=12500)
    old = dict(status='completed_pure_side_scan_FEATURE_route_test',
        full_closed_loop_requested=False, processed_pings=7576, expected_events=events, event_counts=events)
    new = dict(status='completed_lossless_navigation_input_replay', mapped_range_changes=True,
        estimated_frames=7576, evaluation_truth_read_only_after_estimation=True, events=events)
    with pytest.raises(ValueError): require_execution_scope(old, new)
    assert require_execution_scope(old, new, heldout_recording=True).endswith('NOT_full16km')


def test_default_scope_still_requires_every_event_of_the_full_planned_route():
    events = dict(sonar=62458, imu=1030546, depth=206109, dvl=103054)
    old = dict(status='completed_pure_side_scan_FEATURE_route_test', full_closed_loop_requested=True,
               processed_pings=62458, expected_events=events, event_counts=events)
    new = dict(status='completed_lossless_navigation_input_replay', mapped_range_changes=True,
               estimated_frames=62458, evaluation_truth_read_only_after_estimation=True, events=events)
    assert require_execution_scope(old, new) == 'completed_full16km_planned_loop'
    with pytest.raises(ValueError): require_execution_scope(old, new, heldout_recording=True)
    old['event_counts'] = dict(events, depth=206108)
    with pytest.raises(ValueError): require_execution_scope(old, new)
