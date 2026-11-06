from pathlib import Path
import csv
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from AnalyzeSideScanLoopReplay import read_replay, metrics, sha


def write_replay(tmp_path):
    timestamps = np.arange(8, dtype=float)*50.
    truth = dict(timestamps=timestamps, distance_m=np.arange(8, dtype=float),
        positions=np.zeros((8, 4, 3)), yaw_rad=np.zeros((8, 4)), segment_index=np.zeros(8, int))
    position = np.zeros((8, 3, 3)); position[:, :, 0] = np.array([1., 2., 3.])
    truth['positions'][:, 1:3] = position[:, :2]
    np.savez(tmp_path/'estimated_states.npz', positions=position, timestamps=timestamps,
             distance_m=truth['distance_m'], yaw_rad=np.zeros((8, 3)))
    with (tmp_path/'estimated_positions.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['timestamp_s', *[f'{n}_{c}' for n in ['DR', 'Local', 'Global'] for c in 'xyz']])
        for stamp, row in zip(timestamps, position): w.writerow([stamp, *row.ravel()])
    snapshot = tmp_path/'source_snapshot'; snapshot.mkdir()
    (snapshot/'ReplayPureSideScanNavigation.py').write_text('saved driver')
    (snapshot/'sensor.py').write_text('saved sensor')
    report = dict(status='completed_lossless_navigation_input_replay', loop_closure=False,
        source_SHA256={'sensor.py': sha(snapshot/'sensor.py')}, estimated_frames=8,
        replay_driver_SHA256=sha(snapshot/'ReplayPureSideScanNavigation.py'),
        metrics={n: metrics(position[:, k, 0]) for k, n in enumerate(['DR', 'Local', 'Global'])},
        diagnostics=dict(loop_closure=dict(actual_loop_edges=0)))
    (tmp_path/'run_summary.json').write_text(json.dumps(report))
    (tmp_path/'graph_factors.json').write_text(json.dumps(dict(edges=[], global_anchor_bridge=dict(updates=0))))
    return truth, report


def test_independent_review_recomputes_saved_numbers(tmp_path):
    truth, _ = write_replay(tmp_path)
    _, review, _, _ = read_replay(tmp_path, truth)
    assert review['metrics']['Global']['RMSE_m'] == 3.
    assert review['actual_loop_edges'] == 0


def test_review_rejects_partial_run_even_if_numbers_exist(tmp_path):
    truth, report = write_replay(tmp_path); report['status'] = 'running_lossless_navigation_input_replay'
    (tmp_path/'run_summary.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='terminal complete'): read_replay(tmp_path, truth)


def test_review_rejects_claimed_loop_without_real_factor(tmp_path):
    truth, report = write_replay(tmp_path); report['diagnostics']['loop_closure']['actual_loop_edges'] = 1
    (tmp_path/'run_summary.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='real loop factors'): read_replay(tmp_path, truth)


def test_review_rejects_saved_metric_that_differs_from_complete_states(tmp_path):
    truth, report = write_replay(tmp_path); report['metrics']['Global']['RMSE_m'] = .1
    (tmp_path/'run_summary.json').write_text(json.dumps(report))
    with pytest.raises(AssertionError): read_replay(tmp_path, truth)


def test_review_rejects_modified_runtime_snapshot(tmp_path):
    truth, _ = write_replay(tmp_path); (tmp_path/'source_snapshot/sensor.py').write_text('changed')
    with pytest.raises(ValueError, match='Snapshot changed'): read_replay(tmp_path, truth)
