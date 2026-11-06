from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from WriteSideScanReport import dump


def test_real_loop_progress_failure_numpy_bool_and_arrays_are_serialized_atomically(tmp_path):
    report = dict(status='running', processed_pings=6000, diagnostics=dict(loop_closure=dict(
        last=dict(accepted=np.bool_(False), appearance_reversed_for_matching=np.bool_(True)),
        actual_loop_edges=np.int64(2), information=np.eye(3))))
    target = tmp_path/'run_summary.json'; dump(target, report)
    value = json.loads(target.read_text())
    assert value['diagnostics']['loop_closure']['last']['appearance_reversed_for_matching'] is True
    assert not list(tmp_path.glob('*.tmp'))
    # Same unsafe diagnostics in an exception path must still save FAILED,
    # not leave a stale running report after the process has exited.
    report.update(status='failed_preserved_no_automatic_restart', error='original error')
    dump(target, report)
    assert json.loads(target.read_text())['status'] == 'failed_preserved_no_automatic_restart'
    assert isinstance(report['diagnostics']['loop_closure']['last']['accepted'], np.bool_)


def test_nonfinite_diagnostics_are_explicitly_recorded_not_silent_invalid_json(tmp_path):
    target = tmp_path/'run_summary.json'; dump(target, dict(diagnostics=dict(unavailable=np.nan)))
    value = json.loads(target.read_text())
    assert value['diagnostics']['unavailable'] is None
    assert value['JSON_nonfinite_values_replaced_with_null'] == 1


def test_unsupported_object_does_not_destroy_previous_valid_report(tmp_path):
    target = tmp_path/'run_summary.json'; dump(target, dict(status='valid_previous'))
    with pytest.raises(TypeError): dump(target, dict(status=object()))
    assert json.loads(target.read_text())['status'] == 'valid_previous'
