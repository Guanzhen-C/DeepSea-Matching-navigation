import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('full_route_plot',
    Path(__file__).resolve().parents[1]/'standalone/plot_full_route_slam_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def recorded_run(tmp_path):
    (tmp_path/'terrain_replay').mkdir()
    poses = np.zeros((3, 14))
    poses[:, 0] = [0, .2, .4]
    # Deliberately not constant-speed; plotting must use recorded distance.
    poses[:, 1] = [0, .1, .6]
    poses[:, 2] = [0, .1, .6]
    errors = np.zeros((3, 6))
    errors[:, 0] = poses[:, 0]
    metrics = dict(run_status='completed', completed_requested_route=True,
                   ping_rate_hz=5., position_error_statistics={})
    for column, start, name in ((1, 5, 'dead_reckoning_imu_dvl_depth'),
            (2, 8, 'local_acoustic_esikf'), (3, 11, 'online_global_slam')):
        errors[:, column] = np.arange(3) / column
        poses[:, start] = poses[:, 2] + errors[:, column]
        values = errors[:, column]
        metrics['position_error_statistics'][name] = dict(samples=3,
            rmse_3d_m=float(np.sqrt(np.mean(values**2))),
            p95_3d_m=float(np.percentile(values, 95)),
            max_3d_m=float(values.max()), final_3d_m=float(values[-1]))
    (tmp_path/'metrics.json').write_text(json.dumps(metrics))
    np.savetxt(tmp_path/'metrics_position_errors.csv', errors, delimiter=',', header='header', comments='')
    np.savetxt(tmp_path/'terrain_replay/online_poses_5hz.csv', poses, delimiter=',', header='header', comments='')
    return metrics, errors, poses


def test_complete_poses_and_actual_distance_are_preserved(tmp_path):
    _, _, poses = recorded_run(tmp_path)
    _, _, loaded = module.load_verified_run(tmp_path)
    np.testing.assert_array_equal(loaded, poses)


def test_incomplete_run_is_not_presented_as_full_route(tmp_path):
    metrics, _, _ = recorded_run(tmp_path)
    metrics['completed_requested_route'] = False
    (tmp_path/'metrics.json').write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match='completed full-route'):
        module.load_verified_run(tmp_path)


def test_missing_sensor_sample_is_rejected(tmp_path):
    _, errors, poses = recorded_run(tmp_path)
    errors[-1, 0] = poses[-1, 0] = .6
    np.savetxt(tmp_path/'metrics_position_errors.csv', errors, delimiter=',', header='header', comments='')
    np.savetxt(tmp_path/'terrain_replay/online_poses_5hz.csv', poses, delimiter=',', header='header', comments='')
    with pytest.raises(AssertionError):
        module.load_verified_run(tmp_path)


def test_error_values_must_match_actual_published_pose(tmp_path):
    _, errors, _ = recorded_run(tmp_path)
    errors[-1, 3] = 0.
    np.savetxt(tmp_path/'metrics_position_errors.csv', errors, delimiter=',', header='header', comments='')
    with pytest.raises(AssertionError):
        module.load_verified_run(tmp_path)
