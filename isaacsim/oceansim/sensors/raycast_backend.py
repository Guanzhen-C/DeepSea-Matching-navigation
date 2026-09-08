"""Isaac Sim raycast backend used by geometric acoustic sensors."""

from __future__ import annotations

from collections.abc import Callable
import time

import carb
import numpy as np
import omni.usd
from omni.physx import get_physx_scene_query_interface
from isaacsim.core.utils import semantics as semantics_utils

from isaacsim.oceansim.utils.acoustic_sonar import (
    RaycastBatch,
    broadcast_ray_origins,
)


def _hit_field(hit, *names, default=None):
    for name in names:
        if isinstance(hit, dict) and name in hit:
            return hit[name]
        if hasattr(hit, name):
            return getattr(hit, name)
    return default


def _vec3(value, fallback) -> np.ndarray:
    if value is None:
        return np.asarray(fallback, dtype=np.float64)
    try:
        return np.asarray([value[0], value[1], value[2]], dtype=np.float64)
    except (IndexError, TypeError):
        return np.asarray([value.x, value.y, value.z], dtype=np.float64)


def semantic_reflectivity_from_hit(hit) -> float:
    """Resolve OceanSim's ``reflectivity`` semantic on a hit prim or ancestor."""

    path = _hit_field(
        hit,
        "collision",
        "rigidBody",
        "rigid_body",
        default="",
    )
    if not path:
        return np.nan
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(str(path))
    while prim and prim.IsValid() and not prim.IsPseudoRoot():
        # Isaac Sim 5 supports both the legacy SemanticsAPI and LabelsAPI,
        # while Isaac Sim 6 removed the legacy ``get_semantics`` helper.
        # Prefer the current LabelsAPI representation:
        #     instance_name="reflectivity", labels=["0.8"]
        try:
            get_labels = getattr(semantics_utils, "get_labels", None)
            labels = get_labels(prim) if get_labels is not None else {}
        except Exception:
            labels = {}
        for semantic_type, semantic_values in labels.items():
            if semantic_type == "reflectivity" and semantic_values:
                try:
                    return float(semantic_values[0])
                except (TypeError, ValueError):
                    return np.nan

        # Retain compatibility with OceanSim scenes authored with the old
        # SemanticsAPI when running on Isaac Sim 5.
        get_semantics = getattr(semantics_utils, "get_semantics", None)
        if get_semantics is not None:
            try:
                semantics = get_semantics(prim)
            except Exception:
                semantics = {}
            for semantic_type, semantic_data in semantics.values():
                if semantic_type == "reflectivity":
                    try:
                        return float(semantic_data)
                    except (TypeError, ValueError):
                        return np.nan
        prim = prim.GetParent()
    return np.nan


class PhysXRaycastBackend:
    """Cast closest-hit rays through the active Isaac Sim PhysX scene.

    Geometry must have collider APIs enabled.  The sensor origin should be
    mounted outside the vehicle collider because ``raycast_closest`` cannot
    skip the first hit and continue through the same ray.
    """

    def __init__(
        self,
        default_reflectivity: float = 1.0,
        reflectivity_resolver: Callable | None = None,
    ) -> None:
        self._default_reflectivity = max(0.0, float(default_reflectivity))
        self._reflectivity_resolver = (
            semantic_reflectivity_from_hit
            if reflectivity_resolver is None
            else reflectivity_resolver
        )
        self._query = get_physx_scene_query_interface()
        self._cast_count = 0
        self._last_cast_ms = 0.0
        self._mean_cast_ms = 0.0

    @property
    def is_cuda(self) -> bool:
        return False

    @property
    def execution_provider(self) -> str:
        return "PhysX CPU"

    @property
    def device_name(self) -> str:
        return "host"

    @property
    def build_ms(self) -> float:
        return 0.0

    @property
    def triangle_count(self) -> int:
        return 0

    @property
    def last_cast_ms(self) -> float:
        return self._last_cast_ms

    @property
    def mean_cast_ms(self) -> float:
        return self._mean_cast_ms

    def _reflectivity(self, hit) -> float:
        if self._reflectivity_resolver is None:
            return self._default_reflectivity
        try:
            value = float(self._reflectivity_resolver(hit))
        except Exception as exc:
            carb.log_warn(
                f"[OceanSim sonar] reflectivity resolver failed; using default: {exc}"
            )
            return self._default_reflectivity
        if not np.isfinite(value) or value < 0.0:
            return self._default_reflectivity
        return value

    def cast(
        self,
        origin_world: np.ndarray,
        directions_world: np.ndarray,
        max_range: float,
    ) -> RaycastBatch:
        """Cast one closest-hit ray per direction and return a dense batch.

        ``origin_world`` accepts the legacy shared ``(3,)`` origin or one
        explicit ``(N, 3)`` origin per direction.
        """

        cast_started = time.perf_counter()
        directions_world = np.asarray(directions_world, dtype=np.float64).reshape(
            -1, 3
        )
        count = directions_world.shape[0]
        ray_origins_world = broadcast_ray_origins(origin_world, count)
        norms = np.linalg.norm(directions_world, axis=1)
        if np.any(norms <= 0.0) or not np.isfinite(norms).all():
            raise ValueError("ray directions must be finite non-zero vectors")
        directions_world = directions_world / norms[:, None]

        hit_mask = np.zeros(count, dtype=bool)
        distances = np.full(count, np.nan, dtype=np.float64)
        positions = np.full((count, 3), np.nan, dtype=np.float64)
        normals = np.zeros((count, 3), dtype=np.float64)
        reflectivities = np.zeros(count, dtype=np.float64)
        prim_paths: list[str] = [""] * count

        for index, direction in enumerate(directions_world):
            ray_origin = ray_origins_world[index]
            hit = self._query.raycast_closest(
                carb.Float3(*ray_origin.tolist()),
                carb.Float3(*direction.tolist()),
                float(max_range),
            )
            if not bool(_hit_field(hit, "hit", default=False)):
                continue

            distance = float(_hit_field(hit, "distance", default=np.nan))
            if not np.isfinite(distance) or distance < 0.0 or distance > max_range:
                continue

            hit_mask[index] = True
            distances[index] = distance
            positions[index] = _vec3(
                _hit_field(hit, "position", default=None),
                ray_origin + direction * distance,
            )
            normals[index] = _vec3(
                _hit_field(hit, "normal", default=None), -direction
            )
            normal_norm = np.linalg.norm(normals[index])
            if normal_norm > 0.0:
                normals[index] /= normal_norm
            reflectivities[index] = self._reflectivity(hit)
            path = _hit_field(
                hit,
                "collision",
                "rigidBody",
                "rigid_body",
                default="",
            )
            prim_paths[index] = str(path) if path is not None else ""

        result = RaycastBatch(
            hit_mask=hit_mask,
            distances=distances,
            positions_world=positions,
            normals_world=normals,
            reflectivities=reflectivities,
            prim_paths=tuple(prim_paths),
            ray_origins_world=ray_origins_world,
        )
        elapsed_ms = 1.0e3 * (time.perf_counter() - cast_started)
        self._cast_count += 1
        self._last_cast_ms = elapsed_ms
        self._mean_cast_ms += (
            elapsed_ms - self._mean_cast_ms
        ) / float(self._cast_count)
        return result

    def close(self) -> None:
        """Match the lifecycle API of accelerated backends."""

        return None
