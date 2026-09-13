from __future__ import annotations

import numpy as np

from oceansim.slam import (
    AcousticTerrainParticleLikelihood,
    InertialState,
    MeasuredBathymetryGrid,
    TerrainParticleLikelihoodConfig,
)
from oceansim.slam.math3d import rotation_from_rpy


def _world(*, flat: bool = False, backscatter: bool = False):
    size = 321
    axis = np.arange(size, dtype=np.float64) - 0.5 * (size - 1)
    x, y = np.meshgrid(axis, axis)
    if flat:
        elevation = np.full_like(x, -60.0)
    else:
        elevation = (
            -60.0
            + 2.2 * np.sin(x / 10.7)
            + 1.8 * np.cos(y / 13.3)
            + 0.9 * np.sin((x + 1.6 * y) / 6.1)
            + 0.5 * np.cos((2.0 * x - y) / 8.9)
        )
    texture = None
    valid = None
    if backscatter:
        texture = np.clip(
            0.50
            + 0.18 * np.sin(x / 4.7)
            + 0.15 * np.cos(y / 6.3)
            + 0.11 * np.sin((x + 1.9 * y) / 3.7)
            + 0.08 * np.cos((2.1 * x - y) / 5.9),
            0.02,
            0.98,
        )
        valid = np.ones_like(texture, dtype=bool)
    return MeasuredBathymetryGrid(
        elevation,
        resolution_m=1.0,
        backscatter=texture,
        backscatter_valid_mask=valid,
    )


def _state_and_patch(world: MeasuredBathymetryGrid):
    state = InertialState(
        rotation=rotation_from_rpy(0.04, -0.03, 0.27),
        position=np.array([21.0, -17.0, -28.0]),
    )
    along, cross = np.meshgrid(
        np.linspace(-32.0, 32.0, 17),
        np.linspace(-45.0, 45.0, 23),
    )
    x = state.position[0] + along.reshape(-1)
    y = state.position[1] + cross.reshape(-1)
    z = world.seabed_height(x, y)
    points_world = np.column_stack((x, y, z))
    points_body = (points_world - state.position) @ state.rotation
    return state, points_body, x, y


def test_flat_bathymetry_is_neutral_instead_of_a_false_precise_fix() -> None:
    world = _world(flat=True)
    state, points_body, _, _ = _state_and_patch(world)
    likelihood = AcousticTerrainParticleLikelihood(
        world, state, points_body
    )
    particles = np.array(
        [
            [21.0, -17.0, 0.27],
            [41.0, -7.0, 0.27],
            [-9.0, 13.0, 0.62],
        ]
    )

    scores = likelihood.evaluate(particles)

    np.testing.assert_allclose(scores, 0.0, atol=1.0e-12)
    diagnostics = likelihood.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.bathymetry_observability == 0.0
    assert diagnostics.log_likelihood_ratio_std == 0.0


def test_structured_bathymetry_scores_the_correct_particle_highest() -> None:
    world = _world()
    state, points_body, _, _ = _state_and_patch(world)
    likelihood = AcousticTerrainParticleLikelihood(
        world,
        state,
        points_body,
        config=TerrainParticleLikelihoodConfig(
            bathymetry_measurement_std_m=0.35,
        ),
    )
    particles = np.array(
        [
            [21.0, -17.0, 0.27],
            [29.0, -17.0, 0.27],
            [21.0, -7.0, 0.27],
            [21.0, -17.0, 0.39],
        ]
    )

    scores = likelihood.evaluate(particles)

    assert int(np.argmax(scores)) == 0
    assert scores[0] - np.max(scores[1:]) > 1.0
    diagnostics = likelihood.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.bathymetry_observability > 0.50
    assert diagnostics.best_bathymetry_normalized_loss < 1.0e-8


def test_side_scan_texture_localizes_on_flat_bathymetry() -> None:
    world = _world(flat=True, backscatter=True)
    state, points_body, x, y = _state_and_patch(world)
    intensity = world.measured_reflectivity(x, y)
    likelihood = AcousticTerrainParticleLikelihood(
        world,
        state,
        np.empty((0, 3)),
        backscatter_points_body=points_body,
        backscatter_values=intensity,
    )
    particles = np.array(
        [
            [21.0, -17.0, 0.27],
            [28.0, -17.0, 0.27],
            [21.0, -9.0, 0.27],
            [21.0, -17.0, 0.40],
        ]
    )

    scores = likelihood.evaluate(particles)

    assert int(np.argmax(scores)) == 0
    assert scores[0] - np.max(scores[1:]) > 0.5
    diagnostics = likelihood.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.backscatter_observability > 0.50
    assert diagnostics.best_backscatter_correlation > 0.99


def test_coregistered_channels_use_a_conservative_opinion_pool() -> None:
    world = _world(backscatter=True)
    state, points_body, x, y = _state_and_patch(world)
    intensity = world.measured_reflectivity(x, y)
    particles = np.array(
        [
            [21.0, -17.0, 0.27],
            [27.0, -11.0, 0.31],
            [13.0, -23.0, 0.20],
        ]
    )
    bathymetry_only = AcousticTerrainParticleLikelihood(
        world, state, points_body
    ).evaluate(particles)
    backscatter_only = AcousticTerrainParticleLikelihood(
        world,
        state,
        np.empty((0, 3)),
        backscatter_points_body=points_body,
        backscatter_values=intensity,
    ).evaluate(particles)
    combined = AcousticTerrainParticleLikelihood(
        world,
        state,
        points_body,
        backscatter_points_body=points_body,
        backscatter_values=intensity,
    ).evaluate(particles)

    # Co-registered channels share the same pose and acoustic errors.  Their
    # combined log evidence is bounded by a weighted average, not their sum.
    assert np.max(np.abs(combined)) <= (
        np.max(np.abs(bathymetry_only))
        + np.max(np.abs(backscatter_only))
        + 1.0e-12
    )


def test_bathymetry_and_backscatter_can_use_distinct_prior_products() -> None:
    measured = _world(backscatter=True)
    low_pass = MeasuredBathymetryGrid(
        measured.elevation_m,
        resolution_m=measured.resolution_m,
    )
    state, points_body, x, y = _state_and_patch(measured)
    likelihood = AcousticTerrainParticleLikelihood(
        low_pass,
        state,
        points_body,
        backscatter_terrain=measured,
        backscatter_points_body=points_body,
        backscatter_values=measured.measured_reflectivity(x, y),
    )

    likelihood.evaluate(np.array([[21.0, -17.0, 0.27]]))

    assert likelihood.last_diagnostics is not None
    assert likelihood.last_diagnostics.backscatter_points > 0
    assert likelihood.last_diagnostics.best_backscatter_correlation > 0.99


def test_raw_likelihood_refinement_recovers_the_local_map_basin() -> None:
    world = _world(backscatter=True)
    state, points_body, x, y = _state_and_patch(world)
    likelihood = AcousticTerrainParticleLikelihood(
        world,
        state,
        points_body,
        backscatter_points_body=points_body,
        backscatter_values=world.measured_reflectivity(x, y),
        config=TerrainParticleLikelihoodConfig(
            bathymetry_measurement_std_m=0.35
        ),
    )

    refined = likelihood.refine_pose(
        np.array([27.0, -22.0, 0.27 + np.deg2rad(6.0)])
    )

    assert np.linalg.norm(refined[:2] - state.position[:2]) < 1.0
    assert abs(
        (refined[2] - 0.27 + np.pi) % (2.0 * np.pi) - np.pi
    ) < np.deg2rad(0.6)


def test_raw_likelihood_refinement_keeps_seed_on_unobservable_flat_map() -> None:
    world = _world(flat=True)
    state, points_body, _, _ = _state_and_patch(world)
    likelihood = AcousticTerrainParticleLikelihood(world, state, points_body)
    seed = np.array([33.0, -8.0, -0.4])

    refined = likelihood.refine_pose(seed)

    np.testing.assert_allclose(refined, seed)


def test_cohort_basin_refinement_cannot_jump_to_a_distant_alias() -> None:
    world = _world(backscatter=True)
    state, points_body, x, y = _state_and_patch(world)
    likelihood = AcousticTerrainParticleLikelihood(
        world,
        state,
        points_body,
        backscatter_points_body=points_body,
        backscatter_values=world.measured_reflectivity(x, y),
        config=TerrainParticleLikelihoodConfig(
            bathymetry_measurement_std_m=0.35
        ),
    )
    seed = np.array([22.5, -18.0, 0.28])

    refined = likelihood.refine_pose_in_basin(seed)

    assert np.linalg.norm(refined[:2] - seed[:2]) <= np.sqrt(2.0) * 2.5
    assert abs(
        (refined[2] - seed[2] + np.pi) % (2.0 * np.pi) - np.pi
    ) <= np.deg2rad(0.75) + 1.0e-12
    assert np.linalg.norm(refined[:2] - state.position[:2]) < 0.75


def test_flat_current_window_laplace_factor_has_zero_information() -> None:
    world = _world(flat=True)
    state, points_body, _, _ = _state_and_patch(world)
    likelihood = AcousticTerrainParticleLikelihood(world, state, points_body)
    seed = np.array([33.0, -8.0, -0.4])

    approximation = likelihood.approximate_local_measurement(seed)

    np.testing.assert_allclose(approximation.pose2, seed)
    np.testing.assert_allclose(approximation.information, 0.0, atol=1.0e-12)
    assert approximation.observable_rank == 0
    assert np.min(np.linalg.eigvalsh(approximation.covariance)) > 0.0


def test_structured_current_window_laplace_factor_is_current_only_and_ranked() -> None:
    world = _world(backscatter=True)
    state, points_body, x, y = _state_and_patch(world)
    likelihood = AcousticTerrainParticleLikelihood(
        world,
        state,
        points_body,
        backscatter_points_body=points_body,
        backscatter_values=world.measured_reflectivity(x, y),
        config=TerrainParticleLikelihoodConfig(
            bathymetry_measurement_std_m=0.35
        ),
    )

    approximation = likelihood.approximate_local_measurement(
        np.array([22.0, -18.0, 0.28])
    )

    assert np.linalg.norm(
        approximation.pose2[:2] - state.position[:2]
    ) < 0.75
    assert approximation.observable_rank >= 2
    assert np.max(approximation.scaled_information_eigenvalues) > 0.0
    assert np.all(np.linalg.eigvalsh(approximation.information) >= -1.0e-10)
    assert np.all(np.linalg.eigvalsh(approximation.covariance) > 0.0)
