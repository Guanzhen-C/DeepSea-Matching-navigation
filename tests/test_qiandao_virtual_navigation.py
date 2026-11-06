from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from QiandaoVirtualNavigation import VirtualAUVTrajectory, deterministic_schedule, measured_field_sensor_class
from standalone.hil.protocol import acoustic_payload, restore_acoustic


def test_programmed_3D_truth_has_consistent_velocity_acceleration_force_and_rotation():
    trajectory = VirtualAUVTrajectory()
    t, h = 73., 1e-3
    state = trajectory.state(t)
    velocity = (trajectory.position(t+h)-trajectory.position(t-h))/(2*h)
    acceleration = (trajectory.state(t+h).velocity-trajectory.state(t-h).velocity)/(2*h)
    np.testing.assert_allclose(state.velocity, velocity, atol=1e-9)
    np.testing.assert_allclose(state.acceleration, acceleration, atol=1e-9)
    np.testing.assert_allclose(state.rotation@state.specific_force_body,
                               state.acceleration+[0, 0, 9.81], atol=1e-12)
    np.testing.assert_allclose(state.rotation.T@state.rotation, np.eye(3), atol=1e-14)
    assert np.linalg.norm(state.angular_velocity_body) > 0


def test_schedule_does_not_drop_or_fuse_navigation_events():
    schedule = deterministic_schedule(1., .33)
    assert [k for _, k in schedule].count('imu') == 50
    assert [k for _, k in schedule].count('dvl') == 5
    assert [k for _, k in schedule].count('depth') == 10
    assert [k for _, k in schedule].count('sonar') == 4
    assert len(schedule) == len(set(schedule))
    assert [k for t, k in schedule if t == 1.] == ['imu', 'dvl', 'depth']


class MeasuredFlatWorld:
    def intersect_rays(self, origin, directions, *, min_range, max_range):
        distances = (-40.-origin[2])/directions[:, 2]
        hit = (directions[:, 2] < 0) & (distances >= min_range) & (distances <= max_range)
        points = origin+directions*distances[:, None]
        points[~hit] = np.nan
        distances[~hit] = np.nan
        return hit, distances, points, np.abs(directions[:, 2])

    def reflectivity(self, x, y, *, measured_only=False):
        assert measured_only, 'No modeled texture fallback permitted.'
        return .5+.1*np.sin(np.asarray(x)*.4+np.asarray(y)*.07)


def test_scalar_measurements_lineage_and_world_truth_firewall():
    trajectory = VirtualAUVTrajectory()
    sensor = measured_field_sensor_class()(MeasuredFlatWorld(), trajectory, num_beams=32,
        imaging_rays_per_side=32, along_track_rows_per_side=8,
        along_track_sample_spacing_m=.15, max_range=250., sidescan_range_bins=512, seed=31)
    ping = sensor.bathymetric_sidescan_ping(trajectory.state(0.))
    hit = ping.imaging.hit_mask
    world = ping.imaging.points_world[hit]
    expected = MeasuredFlatWorld().reflectivity(world[:, 0], world[:, 1], measured_only=True)
    np.testing.assert_array_equal(ping.imaging.intensities[hit], expected)
    assert set(ping.imaging.along_track_row_indices) == set(range(8))
    assert ping.products_share_raw_returns
    restored = restore_acoustic(acoustic_payload(ping))
    np.testing.assert_array_equal(restored.imaging.points_sensor, ping.imaging.points_sensor)
    with pytest.raises(RuntimeError):
        _ = restored.imaging.points_world
