from __future__ import annotations

import numpy as np

from oceansim.slam import AcousticIntensityMap


def _support_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.arange(-2.0, 3.0, dtype=np.float64) + 0.25
    x, y = np.meshgrid(coordinates, coordinates)
    points = np.column_stack(
        (x.ravel(), y.ravel(), np.full(x.size, -8.0))
    )
    return points, x.ravel(), y.ravel()


def _map() -> AcousticIntensityMap:
    return AcousticIntensityMap(
        voxel_size=1.0,
        min_cell_points=1,
        min_neighbours=5,
        min_gradient=1.0e-6,
    )


def test_overlapping_anchors_remain_separate_and_can_be_required() -> None:
    intensity_map = _map()
    points, x, y = _support_grid()
    first = 0.35 + 0.05 * x - 0.02 * y
    second = 0.75 - 0.03 * x + 0.04 * y

    # Give anchor 7 stronger point support without changing its spatial
    # footprint.  The unrestricted query should therefore select it.
    intensity_map.add_points(
        np.repeat(points, 4, axis=0),
        np.repeat(first, 4),
        heading_rad=0.0,
        observation_id=20,
        anchor_id=7,
    )
    intensity_map.add_points(
        points,
        second,
        heading_rad=0.0,
        observation_id=40,
        anchor_id=13,
    )
    assert len(intensity_map) == 50

    query = np.array([0.25, 0.25, -8.0])
    preferred = intensity_map.query(query, heading_rad=0.0)
    first_sample = intensity_map.query(
        query, heading_rad=0.0, anchor_id=7
    )
    second_sample = intensity_map.query(
        query, heading_rad=0.0, anchor_id=13
    )

    assert preferred is not None
    assert first_sample is not None
    assert second_sample is not None
    assert preferred.anchor_id == 7
    assert first_sample.anchor_id == 7
    assert second_sample.anchor_id == 13
    assert first_sample.reference_observation_id == 20
    assert second_sample.reference_observation_id == 40
    np.testing.assert_allclose(
        first_sample.gradient_xy, [0.05, -0.02], atol=0.005
    )
    np.testing.assert_allclose(
        second_sample.gradient_xy, [-0.03, 0.04], atol=0.005
    )
    assert first_sample.intensity < 0.5
    assert second_sample.intensity > 0.6
    assert (
        intensity_map.query(
            query, heading_rad=0.0, anchor_id=99
        )
        is None
    )


def test_observation_cutoff_strictly_rejects_cell_with_newer_contributor() -> None:
    intensity_map = _map()
    points, x, y = _support_grid()
    old = 0.4 + 0.04 * x - 0.03 * y

    intensity_map.add_points(
        points,
        old,
        heading_rad=0.0,
        observation_id=4,
        anchor_id=3,
    )
    # Updating the same anchor/cells makes every aggregate causally newer.
    # A maximum ID of 10 must reject the complete cells instead of reusing
    # the older portion of their running means.
    intensity_map.add_points(
        points,
        1.0 - old,
        heading_rad=0.0,
        observation_id=12,
        anchor_id=3,
    )
    intensity_map.add_points(
        points,
        0.6 - 0.02 * x + 0.05 * y,
        heading_rad=0.0,
        observation_id=8,
        anchor_id=8,
    )

    query = np.array([0.25, 0.25, -8.0])
    assert (
        intensity_map.query(
            query,
            heading_rad=0.0,
            maximum_observation_id=10,
            anchor_id=3,
        )
        is None
    )
    sample = intensity_map.query(
        query,
        heading_rad=0.0,
        maximum_observation_id=10,
    )
    assert sample is not None
    assert sample.anchor_id == 8
    assert sample.reference_observation_id == 8


def test_legacy_unanchored_points_use_minus_one_anchor() -> None:
    intensity_map = _map()
    points, x, y = _support_grid()
    intensity_map.add_points(
        points,
        0.5 + 0.03 * x + 0.02 * y,
        heading_rad=0.0,
        observation_id=2,
    )

    sample = intensity_map.query(
        np.array([0.25, 0.25, -8.0]), heading_rad=0.0
    )

    assert sample is not None
    assert sample.anchor_id == -1
    assert sample.reference_observation_id == 2


def test_fixed_anchor_batch_matches_scalar_query() -> None:
    intensity_map = _map()
    points, x, y = _support_grid()
    intensity_map.add_points(
        np.repeat(points, 3, axis=0),
        np.repeat(0.45 + 0.06 * x - 0.025 * y, 3),
        heading_rad=0.0,
        observation_id=9,
        anchor_id=4,
    )
    queries = np.array(
        [
            [-0.35, -0.20, -8.0],
            [0.25, 0.25, -8.0],
            [0.80, -0.55, -8.0],
            [12.0, 12.0, -8.0],
            [np.nan, 0.0, -8.0],
        ]
    )

    scalar = [
        intensity_map.query(
            point,
            heading_rad=0.0,
            maximum_observation_id=10,
            anchor_id=4,
        )
        for point in queries
    ]
    batched = intensity_map.query_fixed_anchor_batch(
        queries,
        heading_rad=0.0,
        maximum_observation_id=10,
        anchor_id=4,
    )

    for expected, actual in zip(scalar, batched):
        assert (expected is None) == (actual is None)
        if expected is None:
            continue
        assert actual is not None
        assert actual.neighbours == expected.neighbours
        assert actual.reference_observation_id == expected.reference_observation_id
        assert actual.anchor_id == expected.anchor_id
        np.testing.assert_allclose(actual.intensity, expected.intensity, atol=1.0e-14)
        np.testing.assert_allclose(
            actual.gradient_xy, expected.gradient_xy, atol=1.0e-14
        )
        np.testing.assert_allclose(actual.variance, expected.variance, atol=1.0e-14)
        np.testing.assert_allclose(actual.surface_z, expected.surface_z, atol=1.0e-14)
        np.testing.assert_allclose(
            actual.surface_gradient_xy,
            expected.surface_gradient_xy,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            actual.surface_variance,
            expected.surface_variance,
            atol=1.0e-14,
        )


def test_unrestricted_batch_matches_scalar_anchor_selection() -> None:
    intensity_map = _map()
    points, x, y = _support_grid()
    intensity_map.add_points(
        np.repeat(points, 3, axis=0),
        np.repeat(0.45 + 0.06 * x - 0.025 * y, 3),
        heading_rad=0.0,
        observation_id=9,
        anchor_id=4,
    )
    intensity_map.add_points(
        points,
        0.70 - 0.04 * x + 0.035 * y,
        heading_rad=0.0,
        observation_id=8,
        anchor_id=11,
    )
    queries = np.array(
        [
            [-0.35, -0.20, -8.0],
            [0.25, 0.25, -8.0],
            [0.80, -0.55, -8.0],
            [12.0, 12.0, -8.0],
            [np.nan, 0.0, -8.0],
        ]
    )

    scalar = [
        intensity_map.query(
            point,
            heading_rad=0.0,
            maximum_observation_id=10,
        )
        for point in queries
    ]
    batched = intensity_map.query_batch(
        queries,
        heading_rad=0.0,
        maximum_observation_id=10,
    )

    for expected, actual in zip(scalar, batched):
        assert (expected is None) == (actual is None)
        if expected is None:
            continue
        assert actual is not None
        assert actual.neighbours == expected.neighbours
        assert actual.reference_observation_id == expected.reference_observation_id
        assert actual.anchor_id == expected.anchor_id
        np.testing.assert_allclose(actual.intensity, expected.intensity, atol=1.0e-14)
        np.testing.assert_allclose(
            actual.gradient_xy, expected.gradient_xy, atol=1.0e-14
        )
        np.testing.assert_allclose(actual.variance, expected.variance, atol=1.0e-14)
