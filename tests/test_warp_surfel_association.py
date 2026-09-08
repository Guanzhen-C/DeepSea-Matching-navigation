from __future__ import annotations

import numpy as np

from oceansim.slam import AcousticESIKF, ESIKFConfig, SurfelVoxelMap
from oceansim.slam.warp_surfel_association import WarpSurfelAssociator


def _sloped_map() -> tuple[SurfelVoxelMap, np.ndarray]:
    grid_x, grid_y = np.meshgrid(
        np.linspace(-10.0, 10.0, 101),
        np.linspace(-8.0, 8.0, 81),
    )
    grid_z = -12.0 + 0.07 * grid_x - 0.04 * grid_y
    points = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), grid_z.ravel())
    )
    surfel_map = SurfelVoxelMap(
        voxel_size=1.5,
        min_points=4,
        max_planarity=0.2,
        max_plane_distance=1.0,
    )
    surfel_map.add_points(points)
    return surfel_map, points


def test_warp_association_matches_reference_and_resynchronizes() -> None:
    surfel_map, points = _sloped_map()
    queries = points[::37].copy()
    queries[:, 2] += np.linspace(-0.18, 0.18, queries.shape[0])
    queries = np.vstack((queries, [[50.0, 50.0, -12.0]]))
    reference = surfel_map.associate_points(queries)

    associator = WarpSurfelAssociator(
        device="cpu", minimum_gpu_surfels=0
    )
    accelerated = associator.associate(queries, surfel_map, force=True)
    assert accelerated is not None
    np.testing.assert_array_equal(
        accelerated.valid_mask, reference.valid_mask
    )
    valid = reference.valid_mask
    np.testing.assert_allclose(
        accelerated.centroids[valid], reference.centroids[valid], atol=1.0e-5
    )
    np.testing.assert_allclose(
        accelerated.normals[valid], reference.normals[valid], atol=1.0e-6
    )
    np.testing.assert_allclose(
        accelerated.normal_variances[valid],
        reference.normal_variances[valid],
        atol=1.0e-8,
    )

    previous_revision = surfel_map.revision
    surfel_map.add_points(points[:20] + np.array([0.0, 0.0, 0.01]))
    assert surfel_map.revision == previous_revision + 1
    refreshed = associator.associate(queries, surfel_map, force=True)
    assert refreshed is not None
    assert associator.surfel_count == len(surfel_map.surfels())
    associator.close()


def test_warp_association_preserves_esikf_solution() -> None:
    surfel_map, points = _sloped_map()
    points_sensor = points[::18]
    config = ESIKFConfig(
        sonar_noise_floor=0.05,
        sonar_max_translation_step=0.5,
        sonar_max_rotation_step_deg=2.0,
        degeneracy_eigenvalue=1.0,
    )
    reference = AcousticESIKF(config)
    reference.initialize(0.0, position=np.array([0.0, 0.0, 0.2]))
    reference_result = reference.update_sonar(points_sensor, surfel_map)

    associator = WarpSurfelAssociator(
        device="cpu", minimum_gpu_surfels=0
    )
    accelerated = AcousticESIKF(config, surfel_associator=associator)
    accelerated.initialize(0.0, position=np.array([0.0, 0.0, 0.2]))
    accelerated_result = accelerated.update_sonar(points_sensor, surfel_map)

    assert accelerated_result.accepted == reference_result.accepted
    assert accelerated_result.correspondences == reference_result.correspondences
    np.testing.assert_allclose(
        accelerated.state.position, reference.state.position, atol=2.0e-5
    )
    np.testing.assert_allclose(
        accelerated.state.rotation, reference.state.rotation, atol=2.0e-5
    )
    np.testing.assert_allclose(
        accelerated_result.rmse_after,
        reference_result.rmse_after,
        atol=2.0e-5,
    )
    associator.close()
