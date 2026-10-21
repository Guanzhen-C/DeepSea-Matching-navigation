import numpy as np
import pytest

from standalone.hil_live_status import evaluate_instant


def data():
    records = [dict(timestamp=t, truth_position=np.array([i, 0., 0.])) for i, t in enumerate((0., .2, .4))]
    progress = dict(timestamp=.2, ping_count=2, dr_position=np.array([2., 0., 0.]),
                    local_position=np.array([1.5, 0., 0.]), global_position=np.array([1.1, 0., 0.]))
    return records, progress


def test_live_status_scores_board_positions_not_native_estimator_values():
    records, progress = data()
    result = evaluate_instant(records, progress)
    assert result["instantaneous_errors_3d_m"] == pytest.approx({"dr": 1., "local": .5, "global": .1})
    assert result["estimated_truth_arc_distance_m"] == 1.
    assert not result["full_route_acceptance"]


def test_live_status_never_interpolates_missing_truth_epoch():
    records, progress = data()
    progress["timestamp"] = .3
    result = evaluate_instant(records, progress)
    assert result["awaiting_exact_truth_timestamp"]
    assert result["instantaneous_errors_3d_m"] is None


def test_live_reference_pending_eof_is_not_completed_evidence():
    _, progress = data()
    def incomplete():
        yield dict(timestamp=0., truth_position=np.zeros(3))
        raise EOFError("synthetic still-open member")
    result = evaluate_instant(incomplete(), progress)
    assert result["awaiting_exact_truth_timestamp"]
    assert not result["full_route_acceptance"]


def test_live_status_rejects_nonfinite_board_pose():
    records, progress = data()
    progress["global_position"] = np.full(3, np.nan)
    with pytest.raises(ValueError, match="invalid board"):
        evaluate_instant(records, progress)
