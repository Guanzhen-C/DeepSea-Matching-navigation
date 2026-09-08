"""Warp mesh ray casting for static bathymetry.

The original OceanSim imaging sonar keeps its point-cloud post-processing on
the Warp device.  This backend applies the same design to the geometric MBES
and side-scan sensors: a static seabed triangle mesh is uploaded once, a BVH
is built once, and one Warp thread evaluates each acoustic ray.  Ray/mesh
intersection, face-normal handling, reflectivity interpolation, and the
first-order backscatter model are fused in one kernel.

Only the final compact ping is copied back to NumPy because the current SLAM
front end consumes NumPy arrays.  The implementation intentionally targets a
static mesh; moving scene objects must continue to use the PhysX backend until
their transforms are explicitly synchronized with a Warp mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import warp as wp

from ..utils.acoustic_sonar import RaycastBatch, broadcast_ray_origins


@wp.kernel
def _raycast_static_mesh(
    mesh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=wp.int32),
    vertex_reflectivities: wp.array(dtype=wp.float32),
    max_range: wp.float32,
    attenuation: wp.float32,
    compute_intensity: wp.int32,
    hit_mask: wp.array(dtype=wp.uint8),
    distances: wp.array(dtype=wp.float32),
    positions: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    reflectivities: wp.array(dtype=wp.float32),
    raw_intensities: wp.array(dtype=wp.float32),
):
    ray_index = wp.tid()
    origin = origins[ray_index]
    direction = wp.normalize(directions[ray_index])
    query = wp.mesh_query_ray(mesh_id, origin, direction, max_range)

    if query.result:
        normal = query.normal
        if wp.length(normal) > 1.0e-8:
            normal = wp.normalize(normal)
        else:
            normal = -direction

        # Treat the static seabed as a two-sided acoustic surface while always
        # returning a normal facing the sensor. This prevents triangle winding
        # from changing the return power.
        if wp.dot(-direction, normal) < 0.0:
            normal = -normal

        index_offset = query.face * 3
        vertex_a = triangle_indices[index_offset]
        vertex_b = triangle_indices[index_offset + 1]
        vertex_c = triangle_indices[index_offset + 2]
        bary_c = 1.0 - query.u - query.v
        reflectivity = (
            query.u * vertex_reflectivities[vertex_a]
            + query.v * vertex_reflectivities[vertex_b]
            + bary_c * vertex_reflectivities[vertex_c]
        )
        reflectivity = wp.max(reflectivity, 0.0)

        hit_mask[ray_index] = wp.uint8(1)
        distances[ray_index] = query.t
        positions[ray_index] = origin + query.t * direction
        normals[ray_index] = normal
        reflectivities[ray_index] = reflectivity

        if compute_intensity != 0:
            safe_range = wp.max(query.t, 1.0e-3)
            incidence = wp.clamp(wp.dot(-direction, normal), 0.0, 1.0)
            spreading = 1.0 / (safe_range * safe_range)
            absorption = wp.exp(-2.0 * attenuation * safe_range)
            raw_intensities[ray_index] = (
                reflectivity * incidence * spreading * absorption
            )
        else:
            raw_intensities[ray_index] = 0.0
    else:
        hit_mask[ray_index] = wp.uint8(0)
        distances[ray_index] = 0.0
        positions[ray_index] = wp.vec3(0.0, 0.0, 0.0)
        normals[ray_index] = wp.vec3(0.0, 0.0, 0.0)
        reflectivities[ray_index] = 0.0
        raw_intensities[ray_index] = 0.0


@dataclass
class _RayBuffers:
    origins: object
    directions: object
    hit_mask: object
    distances: object
    positions: object
    normals: object
    reflectivities: object
    raw_intensities: object


class WarpMeshRaycastBackend:
    """Batch closest-hit ray casting against one static triangle mesh.

    Parameters are copied to ``float32`` because Warp's accelerated mesh query
    uses single-precision vertices.  At the Monterey Canyon scene scale this
    gives sub-millimetre coordinate quantisation, well below the centimetre
    noise configured for the simulated sonar.
    """

    def __init__(
        self,
        points: np.ndarray,
        triangle_indices: np.ndarray,
        *,
        vertex_reflectivities: np.ndarray | None = None,
        default_reflectivity: float = 1.0,
        prim_path: str = "/World/Seabed",
        device: str = "cuda:0",
        bvh_constructor: str | None = None,
    ) -> None:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
            raise ValueError("points must have shape (N, 3), N >= 3")
        if not np.isfinite(points).all():
            raise ValueError("points must contain only finite values")

        indices = np.asarray(triangle_indices, dtype=np.int32).reshape(-1)
        if indices.size < 3 or indices.size % 3:
            raise ValueError("triangle_indices must contain complete triangles")
        if np.any(indices < 0) or np.any(indices >= points.shape[0]):
            raise ValueError("triangle_indices contain an out-of-range vertex")

        default_reflectivity = float(default_reflectivity)
        if not np.isfinite(default_reflectivity) or default_reflectivity < 0.0:
            raise ValueError("default_reflectivity must be finite and non-negative")
        if vertex_reflectivities is None:
            reflectivities = np.full(
                points.shape[0], default_reflectivity, dtype=np.float32
            )
        else:
            reflectivities = np.asarray(
                vertex_reflectivities, dtype=np.float32
            ).reshape(-1)
            if reflectivities.size != points.shape[0]:
                raise ValueError("vertex_reflectivities must match points")
            if not np.isfinite(reflectivities).all() or np.any(reflectivities < 0.0):
                raise ValueError(
                    "vertex_reflectivities must be finite and non-negative"
                )

        self._device = wp.get_device(device)
        self._prim_path = str(prim_path)
        self._point_count = int(points.shape[0])
        self._triangle_count = int(indices.size // 3)
        self._buffers: dict[int, _RayBuffers] = {}
        self._cast_count = 0
        self._last_cast_ms = 0.0
        self._mean_cast_ms = 0.0

        build_started = time.perf_counter()
        self._points = wp.array(points, dtype=wp.vec3, device=self._device)
        self._triangle_indices = wp.array(
            indices, dtype=wp.int32, device=self._device
        )
        self._vertex_reflectivities = wp.array(
            reflectivities, dtype=wp.float32, device=self._device
        )
        self._mesh = wp.Mesh(
            points=self._points,
            indices=self._triangle_indices,
            bvh_constructor=bvh_constructor,
        )
        wp.synchronize_device(self._device)
        self._build_ms = 1.0e3 * (time.perf_counter() - build_started)

    @staticmethod
    def cuda_available() -> bool:
        """Return whether Warp can currently access a CUDA device."""

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
    def build_ms(self) -> float:
        return self._build_ms

    @property
    def last_cast_ms(self) -> float:
        """End-to-end time including the compact device-to-host result copy."""

        return self._last_cast_ms

    @property
    def mean_cast_ms(self) -> float:
        return self._mean_cast_ms

    @property
    def cast_count(self) -> int:
        return self._cast_count

    @property
    def triangle_count(self) -> int:
        return self._triangle_count

    def _get_buffers(self, count: int) -> _RayBuffers:
        buffers = self._buffers.get(count)
        if buffers is not None:
            return buffers
        buffers = _RayBuffers(
            origins=wp.empty(count, dtype=wp.vec3, device=self._device),
            directions=wp.empty(count, dtype=wp.vec3, device=self._device),
            hit_mask=wp.empty(count, dtype=wp.uint8, device=self._device),
            distances=wp.empty(count, dtype=wp.float32, device=self._device),
            positions=wp.empty(count, dtype=wp.vec3, device=self._device),
            normals=wp.empty(count, dtype=wp.vec3, device=self._device),
            reflectivities=wp.empty(
                count, dtype=wp.float32, device=self._device
            ),
            raw_intensities=wp.empty(
                count, dtype=wp.float32, device=self._device
            ),
        )
        self._buffers[count] = buffers
        return buffers

    def cast(
        self,
        origin_world: np.ndarray,
        directions_world: np.ndarray,
        max_range: float,
    ) -> RaycastBatch:
        """Cast rays on Warp, returning geometry without acoustic power.

        ``origin_world`` may be one shared ``(3,)`` origin or an ``(N, 3)``
        array aligned with ``directions_world``. Both forms use one Warp launch.
        """

        return self._cast_impl(
            origin_world,
            directions_world,
            max_range,
            attenuation=0.0,
            compute_intensity=False,
        )

    def cast_with_intensity(
        self,
        origin_world: np.ndarray,
        directions_world: np.ndarray,
        max_range: float,
        *,
        attenuation: float,
    ) -> RaycastBatch:
        """Fuse closest-hit geometry and first-order backscatter on Warp."""

        attenuation = float(attenuation)
        if not np.isfinite(attenuation) or attenuation < 0.0:
            raise ValueError("attenuation must be finite and non-negative")
        return self._cast_impl(
            origin_world,
            directions_world,
            max_range,
            attenuation=attenuation,
            compute_intensity=True,
        )

    def _cast_impl(
        self,
        origin_world: np.ndarray,
        directions_world: np.ndarray,
        max_range: float,
        *,
        attenuation: float,
        compute_intensity: bool,
    ) -> RaycastBatch:
        directions = np.asarray(directions_world, dtype=np.float32).reshape(-1, 3)
        max_range = float(max_range)
        if not np.isfinite(directions).all():
            raise ValueError("directions_world must contain only finite values")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("directions_world must contain non-zero vectors")
        if not np.isfinite(max_range) or max_range <= 0.0:
            raise ValueError("max_range must be finite and positive")

        count = int(directions.shape[0])
        origins = broadcast_ray_origins(origin_world, count).astype(
            np.float32, copy=False
        )
        if count == 0:
            empty = np.empty(0, dtype=np.float64)
            return RaycastBatch(
                hit_mask=np.empty(0, dtype=bool),
                distances=empty,
                positions_world=np.empty((0, 3), dtype=np.float64),
                normals_world=np.empty((0, 3), dtype=np.float64),
                reflectivities=empty,
                raw_intensities=(empty if compute_intensity else None),
                ray_origins_world=np.empty((0, 3), dtype=np.float64),
            )

        directions /= norms[:, None]
        buffers = self._get_buffers(count)
        cast_started = time.perf_counter()
        buffers.origins.assign(np.ascontiguousarray(origins))
        buffers.directions.assign(np.ascontiguousarray(directions))
        wp.launch(
            kernel=_raycast_static_mesh,
            dim=count,
            inputs=[
                self._mesh.id,
                buffers.origins,
                buffers.directions,
                self._triangle_indices,
                self._vertex_reflectivities,
                max_range,
                float(attenuation),
                int(compute_intensity),
            ],
            outputs=[
                buffers.hit_mask,
                buffers.distances,
                buffers.positions,
                buffers.normals,
                buffers.reflectivities,
                buffers.raw_intensities,
            ],
            device=self._device,
        )

        # NumPy conversion synchronizes the launch. Keep this boundary explicit:
        # downstream OceanSim sensor messages and the current SLAM API are NumPy.
        hit_mask = buffers.hit_mask.numpy().astype(bool, copy=False)
        distances = buffers.distances.numpy().astype(np.float64)
        positions = buffers.positions.numpy().astype(np.float64)
        normals = buffers.normals.numpy().astype(np.float64)
        reflectivities = buffers.reflectivities.numpy().astype(np.float64)
        raw_intensities = (
            buffers.raw_intensities.numpy().astype(np.float64)
            if compute_intensity
            else None
        )
        elapsed_ms = 1.0e3 * (time.perf_counter() - cast_started)
        self._cast_count += 1
        self._last_cast_ms = elapsed_ms
        self._mean_cast_ms += (
            elapsed_ms - self._mean_cast_ms
        ) / float(self._cast_count)

        distances[~hit_mask] = np.nan
        positions[~hit_mask] = np.nan
        normals[~hit_mask] = 0.0
        reflectivities[~hit_mask] = 0.0
        if raw_intensities is not None:
            raw_intensities[~hit_mask] = 0.0
        prim_paths = tuple(
            self._prim_path if hit else "" for hit in hit_mask
        )
        return RaycastBatch(
            hit_mask=hit_mask,
            distances=distances,
            positions_world=positions,
            normals_world=normals,
            reflectivities=reflectivities,
            prim_paths=prim_paths,
            raw_intensities=raw_intensities,
            ray_origins_world=origins.astype(np.float64),
        )

    def close(self) -> None:
        """Release references to device buffers and the static mesh."""

        self._buffers.clear()
        self._mesh = None
        self._points = None
        self._triangle_indices = None
        self._vertex_reflectivities = None
