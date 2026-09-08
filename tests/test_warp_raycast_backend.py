from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


# The test validates the same Warp kernel on CPU when CI has no GPU. A
# dedicated Isaac smoke test exercises the CUDA provider on GPU hardware.
os.environ.setdefault(
    "WARP_CACHE_PATH", str(Path("/tmp") / "oceansim-warp-test-cache")
)
pytest.importorskip("warp")

from oceansim.sensors.warp_raycast_backend import WarpMeshRaycastBackend
from oceansim.utils.acoustic_sonar import compute_backscatter_intensity


def test_warp_mesh_raycast_matches_reference_backscatter() -> None:
    points = np.array(
        [
            [-20.0, -20.0, -10.0],
            [20.0, -20.0, -10.0],
            [20.0, 20.0, -10.0],
            [-20.0, 20.0, -10.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    reflectivity = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    directions = np.array(
        [[0.0, 0.0, -1.0], [0.1, 0.0, -1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    backend = WarpMeshRaycastBackend(
        points,
        indices,
        vertex_reflectivities=reflectivity,
        device="cpu",
    )
    try:
        result = backend.cast_with_intensity(
            np.zeros(3), directions, 20.0, attenuation=0.01
        )
    finally:
        backend.close()

    np.testing.assert_array_equal(result.hit_mask, [True, True, False])
    np.testing.assert_allclose(result.distances[:2], [10.0, np.sqrt(101.0)], rtol=2e-6)
    np.testing.assert_allclose(result.positions_world[:2, 2], -10.0, atol=2e-6)
    np.testing.assert_allclose(result.normals_world[:2], [[0.0, 0.0, 1.0]] * 2)
    assert np.isnan(result.distances[2])
    assert np.isnan(result.positions_world[2]).all()

    reference = compute_backscatter_intensity(
        result.distances,
        directions,
        result.normals_world,
        result.reflectivities,
        result.hit_mask,
        attenuation=0.01,
    )
    np.testing.assert_allclose(result.raw_intensities, reference, rtol=3e-6, atol=1e-10)


def test_warp_mesh_raycast_accepts_one_origin_per_ray() -> None:
    points = np.array(
        [
            [-20.0, -20.0, -10.0],
            [20.0, -20.0, -10.0],
            [20.0, 20.0, -10.0],
            [-20.0, 20.0, -10.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    origins = np.array(
        [[-0.525, 0.0, 0.0], [0.525, 0.0, -2.0]], dtype=np.float64
    )
    directions = np.array([[0.0, 0.0, -1.0]] * 2, dtype=np.float64)

    backend = WarpMeshRaycastBackend(points, indices, device="cpu")
    try:
        result = backend.cast(origins, directions, 20.0)
    finally:
        backend.close()

    np.testing.assert_array_equal(result.hit_mask, [True, True])
    np.testing.assert_allclose(result.distances, [10.0, 8.0], atol=2e-6)
    np.testing.assert_allclose(result.positions_world[:, 0], origins[:, 0])
    np.testing.assert_allclose(result.positions_world[:, 2], -10.0, atol=2e-6)
    np.testing.assert_allclose(result.ray_origins_world, origins, atol=1e-7)
