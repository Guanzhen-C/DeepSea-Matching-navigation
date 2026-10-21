from __future__ import annotations

import numpy as np
import pytest

from standalone.analyze_hil_relative_motion import analyze, relative_errors


def yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def poses():
    t = np.arange(101) * .2
    truth = np.column_stack((1.5*t, .1*t*t, np.full(len(t), -40.)))
    positions = np.repeat(truth[:, None], 4, axis=1)
    rotations = np.repeat(np.array([yaw(.03*s) for s in t])[:, None], 4, axis=1)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
    return dict(timestamps=t, positions=positions, rotations=rotations, distance=distance)


def test_constant_world_gauge_does_not_pollute_relative_motion():
    run = poses()
    gauge = yaw(.8)
    run["positions"][:, 2] = run["positions"][:, 2] @ gauge.T + [300., -200., 40.]
    run["rotations"][:, 2] = gauge @ run["rotations"][:, 2]
    translation, rotation = relative_errors(run["positions"], run["rotations"], 10)
    assert np.max(translation) < 1e-12
    assert np.max(rotation) < 3e-6


def test_relative_motion_detects_scale_and_changing_heading():
    run = poses()
    run["positions"][:, 1] *= 1.1
    run["rotations"][:, 2] = np.array([yaw(.05*s) for s in run["timestamps"]])
    translation, rotation = relative_errors(run["positions"], run["rotations"], 5)
    expected = .1*np.linalg.norm(run["positions"][5:, 0] - run["positions"][:-5, 0], axis=1)
    np.testing.assert_allclose(translation[:, 0], expected, atol=1e-12)
    np.testing.assert_allclose(rotation[:, 1], np.rad2deg(.02), atol=1e-10)
    assert np.max(translation[:, 1]) > .1
    np.testing.assert_allclose(translation[:, 2], 0., atol=1e-12)


def test_report_uses_exact_sensor_epochs_and_no_trajectory_fit():
    run = poses()
    report = analyze(run, (1., 10.))
    assert report["windows"]["1s"]["overlapping_window_count"] == 96
    assert report["windows"]["10s"]["lag_samples"] == 50
    assert report["windows"]["1s"]["channels"]["local"]["translation_m"]["rmse_m"] == 0.
    assert not report["trajectory_fit_applied"]
    assert not report["full_route_acceptance"]
    assert report["global_motion_includes_map_corrections_not_frontend_odometry"]


@pytest.mark.parametrize("windows", [(0.,), (-1.,), (.3,), (30.,), (float("nan"),), (), (1., 1.)])
def test_invalid_relative_windows_fail_closed(windows):
    with pytest.raises(ValueError):
        analyze(poses(), windows)


def test_relative_motion_does_not_bridge_missing_or_nonfinite_data():
    run = poses()
    run["timestamps"][30:] += .2
    with pytest.raises(ValueError, match="gap"):
        analyze(run, (1.,))
    run = poses()
    run["positions"][1, 2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analyze(run, (1.,))


@pytest.mark.parametrize("lag", [0, -1, 1.5, True, 101])
def test_invalid_lag_fails_closed(lag):
    run = poses()
    with pytest.raises(ValueError, match="positive integer"):
        relative_errors(run["positions"], run["rotations"], lag)
