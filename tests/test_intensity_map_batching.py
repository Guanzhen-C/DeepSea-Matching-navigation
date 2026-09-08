import numpy as np

from oceansim.slam.fast_sonar_slam import FastSonarSLAM, FastSonarSLAMConfig
from oceansim.slam.intensity_map import AcousticIntensityMap


def test_batched_intensity_cell_statistics_match_pointwise_updates() -> None:
    rng = np.random.default_rng(24)
    first_points = rng.uniform([-8.0, -5.0, -2.0], [8.0, 5.0, 0.0], (600, 3))
    first_values = rng.uniform(0.02, 0.95, first_points.shape[0])
    second_points = rng.uniform([-4.0, -3.0, -2.0], [10.0, 7.0, 0.0], (400, 3))
    second_values = rng.uniform(0.02, 0.95, second_points.shape[0])

    batched = AcousticIntensityMap(voxel_size=2.0)
    pointwise = AcousticIntensityMap(voxel_size=2.0)
    for points, values, observation_id in (
        (first_points, first_values, 10),
        (second_points, second_values, 12),
    ):
        assert batched.add_points(
            points,
            values,
            heading_rad=0.4,
            observation_id=observation_id,
            anchor_id=3,
        ) == points.shape[0]
        for point, value in zip(points, values):
            pointwise.add_points(
                point[None, :],
                np.array([value]),
                heading_rad=0.4,
                observation_id=observation_id,
                anchor_id=3,
            )

    assert batched.point_count == pointwise.point_count
    assert batched.anchor_ids == pointwise.anchor_ids
    assert batched._cells.keys() == pointwise._cells.keys()
    for key, actual in batched._cells.items():
        expected = pointwise._cells[key]
        assert actual.count == expected.count
        assert actual.first_observation_id == expected.first_observation_id
        assert actual.last_observation_id == expected.last_observation_id
        np.testing.assert_allclose(actual.mean_xy, expected.mean_xy, atol=1.0e-12)
        np.testing.assert_allclose(
            actual.mean_intensity, expected.mean_intensity, atol=1.0e-12
        )
        np.testing.assert_allclose(
            actual.intensity_m2, expected.intensity_m2, rtol=1.0e-12, atol=1.0e-12
        )


def test_batched_intensity_insert_retains_valid_sample_accounting() -> None:
    intensity_map = AcousticIntensityMap(voxel_size=2.0)
    points = np.array(
        [[0.1, 0.2, -1.0], [0.7, 0.8, -1.0], [np.nan, 0.0, -1.0]]
    )
    values = np.array([0.2, 0.6, 0.4])

    assert intensity_map.add_points(points, values, observation_id=5) == 2
    assert intensity_map.point_count == 2
    assert len(intensity_map) == 1
    cell = next(iter(intensity_map._cells.values()))
    assert cell.count == 2
    np.testing.assert_allclose(cell.mean_xy, [0.4, 0.5])
    assert cell.mean_intensity == 0.4
    assert cell.first_observation_id == 5
    assert cell.last_observation_id == 5


def test_long_mission_trajectory_storage_is_decimated_without_state_loss() -> None:
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(trajectory_sample_interval_s=0.2)
    )
    slam.initialize(0.0)
    for timestamp in np.arange(0.02, 1.01, 0.02):
        assert slam.process_imu(
            float(timestamp),
            np.zeros(3),
            np.array([0.0, 0.0, 9.80665]),
        )

    stored_count = len(slam._trajectory)
    timestamps, positions = slam.trajectory()
    assert stored_count == 6
    assert timestamps.shape == (6,)
    assert positions.shape == (6, 3)
    assert timestamps[-1] == 1.0
    np.testing.assert_allclose(positions[-1], slam.state.position)
