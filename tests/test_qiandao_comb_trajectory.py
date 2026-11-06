from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'isaacsim'))
from QiandaoCombTrajectory import CombSurveyParameters, CombSurveyTrajectory


def test_closed_lawnmower_has_twelve_actual_long_legs_and_single_return():
    route = CombSurveyTrajectory()
    assert sum(s.kind == 'survey_line' for s in route.segments) == 12
    assert sum(s.kind == 'survey_uturn' for s in route.segments) == 11
    assert sum(s.kind == 'return_line' for s in route.segments) == 1
    assert 3500 < route.length_m < 4200
    p0, t0, k0, _ = route.evaluate_distance(0.)
    p1, t1, k1, _ = route.evaluate_distance(route.length_m)
    np.testing.assert_allclose(p0, p1, atol=1e-12)
    np.testing.assert_allclose(t0, t1, atol=1e-12)
    assert abs(k0)+abs(k1) < 1e-12


def test_every_join_is_position_tangent_and_curvature_continuous():
    route = CombSurveyTrajectory()
    for previous, following in zip(route.segments, route.segments[1:]+route.segments[:1]):
        a = previous.evaluate(previous.length_m)
        b = following.evaluate(0.)
        np.testing.assert_allclose(a[0], b[0], atol=1e-11)
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)
        assert abs(a[2]-b[2]) < 1e-12


def test_arclength_kinematics_and_no_turn_in_place():
    route = CombSurveyTrajectory()
    s = np.linspace(.01, route.length_m-.01, 15000)
    xy, tangent, curvature, _ = route.evaluate_distance(s)
    h = .001
    xp, tp, _, _ = route.evaluate_distance(s+h)
    xm, tm, _, _ = route.evaluate_distance(s-h)
    np.testing.assert_allclose((xp-xm)/(2*h), tangent, atol=1e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    np.testing.assert_allclose((tp-tm)/(2*h), curvature[:, None]*normal, atol=1e-6)
    assert np.isfinite(xy).all()
    assert np.abs(xy[:, 0]).max() <= 149.00001
    assert np.abs(xy[:, 1]).max() <= 154.00001
    assert np.max(abs(curvature))*route.speed_mps < np.deg2rad(7)
    assert route.duration_s == route.length_m/route.speed_mps


def test_manifest_reconstructs_identical_single_loop():
    route = CombSurveyTrajectory()
    restored = CombSurveyTrajectory(CombSurveyParameters(**route.manifest()['parameters']))
    s = np.linspace(0, route.length_m, 1000)
    for a, b in zip(route.evaluate_distance(s), restored.evaluate_distance(s)):
        np.testing.assert_array_equal(a, b)


def test_virtual_IMU_kinematics_follow_actual_curved_motion_including_joins():
    from oceansim.slam.math3d import so3_log
    route = CombSurveyTrajectory()
    joins = route.ends_m[:-1]/route.speed_mps
    times = np.r_[np.linspace(.1, route.duration_s-.1, 70), joins]
    h = .001
    for t in times:
        state = route.state(t)
        before, after = route.state(t-h), route.state(t+h)
        np.testing.assert_allclose((after.position-before.position)/(2*h), state.velocity, atol=2e-7)
        np.testing.assert_allclose((after.velocity-before.velocity)/(2*h), state.acceleration, atol=2e-5)
        omega = so3_log(before.rotation.T@after.rotation)/(2*h)
        np.testing.assert_allclose(omega, state.angular_velocity_body, atol=2e-5)
        np.testing.assert_allclose(state.rotation.T@state.rotation, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(state.rotation@state.specific_force_body,
                                   state.acceleration+[0, 0, 9.81], atol=1e-13)


def test_invalid_parameters_and_outside_time_rejected():
    for kwargs in ({'lanes': 3}, {'line_spacing_m': 0}, {'speed_mps': float('nan')},
                   {'virtual_flat_bottom_z_m': 0}, {'return_corner_width_m': 200}):
        with pytest.raises(ValueError):
            CombSurveyParameters(**kwargs)
    route = CombSurveyTrajectory()
    with pytest.raises(ValueError):
        route.position(-1.)
    with pytest.raises(ValueError):
        route.position(route.duration_s+1.)
