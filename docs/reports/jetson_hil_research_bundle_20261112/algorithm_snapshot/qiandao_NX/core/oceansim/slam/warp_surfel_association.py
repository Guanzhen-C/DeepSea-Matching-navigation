"""Warp acceleration for exact voxel-surfel data association.

The filter itself deliberately remains a small double-precision CPU ESIKF.
Only the embarrassingly parallel part of the acoustic front end is moved to
Warp: every transformed MBES point searches the same 27 voxel neighbours as
the reference implementation.  Mature surfels are sorted once per map
revision and each neighbour is found by lexicographic binary search, avoiding
an approximate nearest-neighbour substitution that could change the estimator
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import warp as wp

from .surfel_map import (
    SurfelAssociationBatch,
    SurfelMapSnapshot,
    SurfelVoxelMap,
)


@wp.func
def _key_compare(key: wp.vec3i, x: int, y: int, z: int):
    result = 0
    if key[0] < x:
        result = -1
    elif key[0] > x:
        result = 1
    elif key[1] < y:
        result = -1
    elif key[1] > y:
        result = 1
    elif key[2] < z:
        result = -1
    elif key[2] > z:
        result = 1
    return result


@wp.func
def _find_voxel(
    voxel_keys: wp.array(dtype=wp.vec3i),
    surfel_count: int,
    x: int,
    y: int,
    z: int,
):
    # Explicit casts mark these as mutable dynamic values for Warp's codegen.
    low = int(0)
    high = int(surfel_count)
    while low < high:
        middle = low + (high - low) // 2
        comparison = _key_compare(voxel_keys[middle], x, y, z)
        if comparison < 0:
            low = middle + 1
        else:
            high = middle
    if low < surfel_count and _key_compare(voxel_keys[low], x, y, z) == 0:
        return low
    return -1


@wp.kernel
def _associate_surfels(
    points_world: wp.array(dtype=wp.vec3),
    voxel_keys: wp.array(dtype=wp.vec3i),
    centroids: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    surfel_count: int,
    inverse_voxel_size: float,
    max_plane_distance: float,
    max_centroid_distance: float,
    matched_indices: wp.array(dtype=wp.int32),
):
    point_index = wp.tid()
    point = points_world[point_index]
    base_x = int(wp.floor(point[0] * inverse_voxel_size))
    base_y = int(wp.floor(point[1] * inverse_voxel_size))
    base_z = int(wp.floor(point[2] * inverse_voxel_size))
    best_index = int(-1)
    best_score = float(3.402823466e38)

    # Loop order deliberately matches itertools.product((-1, 0, 1), repeat=3)
    # in the reference implementation, including deterministic tie handling.
    for offset_x in range(-1, 2):
        for offset_y in range(-1, 2):
            for offset_z in range(-1, 2):
                candidate = _find_voxel(
                    voxel_keys,
                    surfel_count,
                    base_x + offset_x,
                    base_y + offset_y,
                    base_z + offset_z,
                )
                if candidate >= 0:
                    delta = point - centroids[candidate]
                    centroid_distance = wp.length(delta)
                    plane_distance = wp.abs(wp.dot(normals[candidate], delta))
                    if (
                        centroid_distance <= max_centroid_distance
                        and plane_distance <= max_plane_distance
                    ):
                        score = plane_distance + 0.04 * centroid_distance
                        if score < best_score:
                            best_score = score
                            best_index = candidate
    matched_indices[point_index] = best_index


@dataclass
class _AssociationBuffers:
    points_world: object
    matched_indices: object


class WarpSurfelAssociator:
    """Exact 27-neighbour surfel lookup on a Warp device.

    ``minimum_gpu_surfels`` can enable a hybrid policy when profiling on a
    different target calls for it.  Its default is zero because the measured
    reference workload was faster without an extra map-size pre-pass. The
    live runner instead selects the provider from the configured point-batch
    size. Returning ``None`` signals the estimator to execute its reference
    CPU path.
    """

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        minimum_gpu_surfels: int = 0,
    ) -> None:
        if minimum_gpu_surfels < 0:
            raise ValueError("minimum_gpu_surfels must be non-negative")
        self._device = wp.get_device(device)
        self.minimum_gpu_surfels = int(minimum_gpu_surfels)
        self._snapshot: SurfelMapSnapshot | None = None
        self._map_revision = -1
        self._surfel_count = 0
        self._capacity = 0
        self._voxel_keys = None
        self._centroids = None
        self._normals = None
        self._key_staging = None
        self._centroid_staging = None
        self._normal_staging = None
        self._buffers: dict[int, _AssociationBuffers] = {}
        self._gpu_calls = 0
        self._cpu_fallbacks = 0
        self._last_association_ms = 0.0
        self._mean_association_ms = 0.0
        self._last_sync_ms = 0.0
        self._matched_count = 0
        self._warmup_ms = 0.0

    @staticmethod
    def cuda_available() -> bool:
        return bool(wp.is_cuda_available())

    @property
    def is_cuda(self) -> bool:
        return bool(self._device.is_cuda)

    @property
    def device_name(self) -> str:
        return str(self._device)

    @property
    def execution_provider(self) -> str:
        return "Warp CUDA" if self.is_cuda else "Warp CPU"

    @property
    def last_association_ms(self) -> float:
        """End-to-end latency, including snapshot sync and compact copies."""

        return self._last_association_ms

    @property
    def mean_association_ms(self) -> float:
        return self._mean_association_ms

    @property
    def last_sync_ms(self) -> float:
        return self._last_sync_ms

    @property
    def gpu_calls(self) -> int:
        return self._gpu_calls

    @property
    def cpu_fallbacks(self) -> int:
        return self._cpu_fallbacks

    @property
    def surfel_count(self) -> int:
        return self._surfel_count

    @property
    def matched_count(self) -> int:
        return self._matched_count

    @property
    def warmup_ms(self) -> float:
        return self._warmup_ms

    def warm_up(self) -> float:
        """Compile the kernel before live sensing begins.

        Warp caches the generated module, but a first run may still compile it.
        Keeping that one-time cost outside the estimator timing prevents a UI
        frame pause from being mistaken for steady-state association latency.
        """

        started = time.perf_counter()
        points = wp.array(
            np.array([[0.0, 0.0, 0.1]], dtype=np.float32),
            dtype=wp.vec3,
            device=self._device,
        )
        keys = wp.array(
            np.array([[0, 0, 0]], dtype=np.int32),
            dtype=wp.vec3i,
            device=self._device,
        )
        centroids = wp.array(
            np.zeros((1, 3), dtype=np.float32),
            dtype=wp.vec3,
            device=self._device,
        )
        normals = wp.array(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            dtype=wp.vec3,
            device=self._device,
        )
        matched = wp.empty(1, dtype=wp.int32, device=self._device)
        wp.launch(
            kernel=_associate_surfels,
            dim=1,
            inputs=[points, keys, centroids, normals, 1, 1.0, 1.0, 2.6],
            outputs=[matched],
            device=self._device,
        )
        wp.synchronize_device(self._device)
        self._warmup_ms = 1.0e3 * (time.perf_counter() - started)
        return self._warmup_ms

    def _ensure_capacity(self, count: int) -> None:
        if count <= self._capacity:
            return
        capacity = max(256, 1 << (count - 1).bit_length())
        self._voxel_keys = wp.empty(
            capacity, dtype=wp.vec3i, device=self._device
        )
        self._centroids = wp.empty(
            capacity, dtype=wp.vec3, device=self._device
        )
        self._normals = wp.empty(
            capacity, dtype=wp.vec3, device=self._device
        )
        self._key_staging = np.empty((capacity, 3), dtype=np.int32)
        self._centroid_staging = np.empty((capacity, 3), dtype=np.float32)
        self._normal_staging = np.empty((capacity, 3), dtype=np.float32)
        self._capacity = capacity

    def _synchronize_map(self, surfel_map: SurfelVoxelMap) -> None:
        if surfel_map.revision == self._map_revision:
            self._last_sync_ms = 0.0
            return
        started = time.perf_counter()
        snapshot = surfel_map.association_snapshot()
        count = int(snapshot.voxel_keys.shape[0])
        self._ensure_capacity(count)
        if count > 0:
            # Reusable power-of-two storage avoids allocating device arrays at
            # every incremental map revision. Only valid prefix rows are read.
            self._key_staging[:count] = snapshot.voxel_keys
            self._centroid_staging[:count] = snapshot.centroids
            self._normal_staging[:count] = snapshot.normals
            # Capacity grows geometrically, but only the valid prefix belongs
            # to this map revision.  Passing the whole staging allocation made
            # every incremental ping copy up to almost twice as many rows to
            # the device.  Warp accepts a shorter source and retains the
            # destination allocation, so this is an exact transfer reduction.
            self._voxel_keys.assign(self._key_staging[:count])
            self._centroids.assign(self._centroid_staging[:count])
            self._normals.assign(self._normal_staging[:count])
            wp.synchronize_device(self._device)
        self._snapshot = snapshot
        self._surfel_count = count
        self._map_revision = snapshot.revision
        self._last_sync_ms = 1.0e3 * (time.perf_counter() - started)

    def _get_buffers(self, count: int) -> _AssociationBuffers:
        buffers = self._buffers.get(count)
        if buffers is None:
            buffers = _AssociationBuffers(
                points_world=wp.empty(
                    count, dtype=wp.vec3, device=self._device
                ),
                matched_indices=wp.empty(
                    count, dtype=wp.int32, device=self._device
                ),
            )
            self._buffers[count] = buffers
        return buffers

    def associate(
        self,
        points_world: np.ndarray,
        surfel_map: SurfelVoxelMap,
        *,
        force: bool = False,
    ) -> SurfelAssociationBatch | None:
        """Associate points, or request CPU fallback below the map threshold."""

        points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError("points_world must contain only finite values")
        if (
            self.minimum_gpu_surfels > 0
            and not force
            and surfel_map.mature_voxel_count < self.minimum_gpu_surfels
        ):
            self._cpu_fallbacks += 1
            return None
        self._synchronize_map(surfel_map)
        if (
            not force
            and self._surfel_count < self.minimum_gpu_surfels
        ):
            self._cpu_fallbacks += 1
            return None
        count = int(points.shape[0])
        if count == 0 or self._surfel_count == 0:
            return SurfelAssociationBatch(
                valid_mask=np.zeros(count, dtype=bool),
                centroids=np.zeros((count, 3), dtype=np.float64),
                normals=np.zeros((count, 3), dtype=np.float64),
                normal_variances=np.zeros(count, dtype=np.float64),
            )

        buffers = self._get_buffers(count)
        started = time.perf_counter()
        buffers.points_world.assign(np.ascontiguousarray(points))
        wp.launch(
            kernel=_associate_surfels,
            dim=count,
            inputs=[
                buffers.points_world,
                self._voxel_keys,
                self._centroids,
                self._normals,
                self._surfel_count,
                1.0 / surfel_map.voxel_size,
                surfel_map.max_plane_distance,
                surfel_map.max_centroid_distance,
            ],
            outputs=[buffers.matched_indices],
            device=self._device,
        )
        matched_indices = buffers.matched_indices.numpy().astype(
            np.int64, copy=False
        )
        elapsed_ms = 1.0e3 * (time.perf_counter() - started)
        self._gpu_calls += 1
        self._last_association_ms = self._last_sync_ms + elapsed_ms
        self._mean_association_ms += (
            self._last_association_ms - self._mean_association_ms
        ) / float(self._gpu_calls)

        valid_mask = matched_indices >= 0
        self._matched_count = int(np.count_nonzero(valid_mask))
        centroids = np.zeros((count, 3), dtype=np.float64)
        normals = np.zeros((count, 3), dtype=np.float64)
        normal_variances = np.zeros(count, dtype=np.float64)
        if np.any(valid_mask):
            assert self._snapshot is not None
            selected = matched_indices[valid_mask]
            centroids[valid_mask] = self._snapshot.centroids[selected]
            normals[valid_mask] = self._snapshot.normals[selected]
            normal_variances[valid_mask] = (
                self._snapshot.normal_variances[selected]
            )
        return SurfelAssociationBatch(
            valid_mask=valid_mask,
            centroids=centroids,
            normals=normals,
            normal_variances=normal_variances,
        )

    def close(self) -> None:
        self._buffers.clear()
        self._snapshot = None
        self._voxel_keys = None
        self._centroids = None
        self._normals = None
        self._key_staging = None
        self._centroid_staging = None
        self._normal_staging = None
        self._capacity = 0
        self._surfel_count = 0
