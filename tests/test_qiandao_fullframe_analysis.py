from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from AnalyzeQiandaoFullFrameComb import main, metrics


def test_error_metrics_are_full_sample_norm_statistics_not_final_error():
    m = metrics(np.array([0., 3., 4.]))
    assert m['samples'] == 3 and m['final_m'] == 4 and m['median_m'] == 3
    assert m['RMSE_m'] == pytest.approx(np.sqrt(25/3))
    assert m['P95_m'] == pytest.approx(3.9)


@pytest.mark.parametrize('status,full', [('running', True),
    ('completed_ALL_scheduled_events_full_frozen_joint_estimator', False)])
def test_running_or_short_pilot_must_not_be_reviewed_as_finished_full_route(tmp_path, monkeypatch, status, full):
    run = tmp_path/'run'; run.mkdir()
    report = dict(status=status, full_closed_loop_requested=full, source_physical_records=2020,
        GPS_used=False, after_initialization_truth_accessible_to_estimator=False,
        reference_query_depth_shared=False, both_depth_and_side_scan_global_matching_enabled=True)
    (run/'run_summary.json').write_text(json.dumps(report))
    out = tmp_path/'review'
    monkeypatch.setattr(sys, 'argv', ['analysis', '--experiment', str(tmp_path), '--run', str(run), '--out', str(out)])
    with pytest.raises(ValueError, match='Only completed'):
        main()
    assert not out.exists()
