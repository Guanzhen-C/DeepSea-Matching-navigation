from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('slam_run_status', ROOT/'standalone/slam_run_status.py')
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


@pytest.mark.parametrize('active,final,age,expected', [
    (True, None, 10., 'running'),
    (True, None, 3600., 'no_recent_progress'),
    (False, None, 10., 'incomplete_not_running'),
    (False, 'completed', 3600., 'completed'),
    (True, 'completed', 3600., 'completed_cleanup_active'),
    (True, 'failed', 10., 'failed_or_interrupted'),
    (True, 'interrupted', None, 'failed_or_interrupted'),
    (True, None, None, 'no_recent_progress'),
])
def test_active_and_success_never_prove_progress(active, final, age, expected):
    assert status.evaluate_health(active=active, final_status=final,
        freshest_age_s=age, stale_after_s=180.) == expected


def test_startup_grace_only_before_any_sensor_event():
    assert status.evaluate_health(active=True, final_status=None, freshest_age_s=None,
        startup_age_s=20., stale_after_s=180.) == 'initializing'


def make_run(tmp_path):
    (tmp_path/'run_manifest.json').write_text(json.dumps(dict(
        unit='test.service', created_at=datetime.fromtimestamp(0, timezone.utc).isoformat())))
    (tmp_path/'terrain_replay').mkdir()
    return tmp_path/'terrain_replay/online_poses_5hz.csv'


def test_bounded_pose_tail_ignores_partial_new_record(tmp_path):
    path = make_run(tmp_path)
    path.write_text('header\n' + '1,1,0,0,0,1,0,0,2,0,0,3,0,0\n' * 1000
                    + '2,2,0,0,0,100,0,0,200,0,0,300,0,0')
    pose = status.latest_complete_pose(path)
    assert pose['timestamp_s'] == 1
    assert pose['errors_m'] == {'dr':1., 'local':2., 'global':3.}


def test_header_only_and_nonfinite_pose_are_not_progress(tmp_path):
    path = make_run(tmp_path)
    path.write_text('timestamp_s,header\n')
    assert status.latest_complete_pose(path) is None
    path.write_text('1,1,0,0,0,1,0,0,2,0,0,3,0,nan\n')
    assert status.latest_complete_pose(path) is None


def test_stale_checkpoint_cannot_hide_fresh_live_pose(tmp_path):
    path = make_run(tmp_path)
    path.write_text('1,1,0,0,0,1,0,0,2,0,0,3,0,0\n')
    progress = tmp_path/'metrics_progress.json'
    progress.write_text(json.dumps(dict(travelled_distance_m=0)))
    os.utime(path, (995, 995))
    os.utime(progress, (10, 10))
    result = status.read_status(tmp_path, now=1000,
        service={'ActiveState':'active', 'Result':'success'})
    assert result['health'] == 'running'
    assert result['file_age_s'] == dict(checkpoint=990., pose_journal=5.)
    assert result['last_complete_live_pose']['travelled_distance_m'] == 1.


def test_stale_active_unit_is_not_reported_as_running(tmp_path):
    path = make_run(tmp_path)
    path.write_text('1,1,0,0,0,1,0,0,2,0,0,3,0,0\n')
    os.utime(path, (100, 100))
    result = status.read_status(tmp_path, now=1000,
        service={'ActiveState':'active', 'Result':'success'})
    assert result['health'] == 'no_recent_progress'
    assert 'stale' in result['warning']


def test_final_metrics_are_authoritative_even_when_checkpoint_says_running(tmp_path):
    make_run(tmp_path)
    (tmp_path/'metrics_progress.json').write_text('{"status":"running"}')
    (tmp_path/'metrics.json').write_text('{"run_status":"completed"}')
    result = status.read_status(tmp_path, now=1000,
        service={'ActiveState':'inactive','Result':'success'})
    assert result['health'] == 'completed'


def test_partial_final_file_does_not_claim_completion_or_crash(tmp_path):
    path = make_run(tmp_path)
    path.write_text('1,1,0,0,0,1,0,0,2,0,0,3,0,0\n')
    os.utime(path, (995, 995))
    (tmp_path/'metrics.json').write_text('{"run_status": "completed",')
    result = status.read_status(tmp_path, now=1000, service={'ActiveState':'active'})
    assert result['health'] == 'running'
    assert result['final_metrics'] is None
    assert 'incomplete' in result['final_metrics_warning']


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_staleness_threshold_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        status.read_status(tmp_path, stale_after_s=value)
