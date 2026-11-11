"""Planar-constrained 3D-BBS verification for MBES relocalization.

The published 3D-BBS implementation searches a six degree-of-freedom rigid
transform against a hierarchical occupancy map.  An AUV equipped with an IMU
and pressure sensor does not need (or want) an unconstrained six-dimensional
global search: roll, pitch and vehicle depth are already observed.  This
adapter pre-rotates the MBES strip by the measured roll/pitch and candidate
yaw, embeds pressure depth directly in the source cloud, and asks the official
GPU solver to search only residual ``x``, ``y`` and yaw.

The result is a geometric verifier, not an absolute factor by itself.  It is
intended to reject a high appearance-correlation peak whose full 3-D MBES
surface does not occupy the same map voxels.  Metric refinement and persistent
multi-hypothesis confirmation remain separate stages.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .esikf import InertialState
from .math3d import rotation_from_rpy, rpy_from_rotation, wrap_angle
from .real_bathymetry import MeasuredBathymetryGrid


def _readonly(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(shape).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class BBS3DVerifierConfig:
    """Settings for the official 3D-BBS candidate-verification stage."""

    minimum_level_resolution_m: float = 5.0
    maximum_level: int = 3
    maximum_translation_m: float = 30.0
    maximum_yaw_deg: float = 6.0
    minimum_score_fraction: float = 0.25
    minimum_source_points: int = 128
    maximum_source_points: int = 8_192
    target_stride: int = 1
    target_margin_m: float = 20.0
    timeout_milliseconds: int = 5_000
    reject_search_boundary: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_level",
            "minimum_source_points",
            "maximum_source_points",
            "target_stride",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "maximum_level" else 1
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))
        timeout = self.timeout_milliseconds
        if (
            isinstance(timeout, (bool, np.bool_))
            or int(timeout) != timeout
            or int(timeout) < 0
        ):
            raise ValueError("timeout_milliseconds must be a nonnegative integer")
        object.__setattr__(self, "timeout_milliseconds", int(timeout))
        for name in (
            "minimum_level_resolution_m",
            "maximum_translation_m",
            "maximum_yaw_deg",
            "target_margin_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        fraction = float(self.minimum_score_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("minimum_score_fraction must lie in (0, 1)")
        object.__setattr__(self, "minimum_score_fraction", fraction)
        if self.maximum_source_points < self.minimum_source_points:
            raise ValueError(
                "maximum_source_points must not be below minimum_source_points"
            )
        if self.maximum_yaw_deg >= 45.0:
            raise ValueError(
                "3D-BBS is a candidate verifier; maximum_yaw_deg must be below 45"
            )
        if not isinstance(self.reject_search_boundary, (bool, np.bool_)):
            raise ValueError("reject_search_boundary must be boolean")
        object.__setattr__(
            self, "reject_search_boundary", bool(self.reject_search_boundary)
        )


@dataclass(frozen=True)
class BBS3DVerificationResult:
    """Auditable result returned by :class:`BBS3DCandidateVerifier`."""

    available: bool
    accepted: bool
    matched_pose2: np.ndarray
    correction_pose2: np.ndarray
    transformation: np.ndarray
    score: int
    score_fraction: float
    source_points: int
    target_points: int
    elapsed_milliseconds: float
    timed_out: bool
    rejection_reason: str | None = None
    backend: str = "KOKIAOKI/3d_bbs@41529a3-gpu"

    def __post_init__(self) -> None:
        if not isinstance(self.available, (bool, np.bool_)):
            raise ValueError("available must be boolean")
        if not isinstance(self.accepted, (bool, np.bool_)):
            raise ValueError("accepted must be boolean")
        if not isinstance(self.timed_out, (bool, np.bool_)):
            raise ValueError("timed_out must be boolean")
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "timed_out", bool(self.timed_out))
        object.__setattr__(self, "matched_pose2", _readonly(self.matched_pose2, (3,)))
        object.__setattr__(
            self, "correction_pose2", _readonly(self.correction_pose2, (3,))
        )
        object.__setattr__(
            self, "transformation", _readonly(self.transformation, (4, 4))
        )
        for name in ("score", "source_points", "target_points"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        fraction = float(self.score_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("score_fraction must lie in [0, 1]")
        object.__setattr__(self, "score_fraction", fraction)
        elapsed = float(self.elapsed_milliseconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_milliseconds must be finite and nonnegative")
        object.__setattr__(self, "elapsed_milliseconds", elapsed)
        if self.accepted and (not self.available or self.rejection_reason is not None):
            raise ValueError("an accepted verification must be available and unrejected")
        if not self.accepted and not isinstance(self.rejection_reason, str):
            raise ValueError("a rejected verification needs a rejection reason")


class _NativeBBS3D:
    """Small ``ctypes`` binding around the pinned upstream C++ library."""

    _FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)

    def __init__(self, library_path: str | Path | None = None) -> None:
        path = self._resolve_library(library_path)
        try:
            self._library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeError(f"failed to load 3D-BBS bridge {path}: {exc}") from exc
        function = self._library.oceansim_bbs3d_localize
        function.argtypes = [
            self._FLOAT_POINTER,
            ctypes.c_size_t,
            self._FLOAT_POINTER,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_int,
            self._FLOAT_POINTER,
            self._FLOAT_POINTER,
            self._FLOAT_POINTER,
            self._FLOAT_POINTER,
            ctypes.c_float,
            ctypes.c_int,
            self._FLOAT_POINTER,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int
        self._function = function
        self.path = path

    @staticmethod
    def _repository_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _resolve_library(cls, requested: str | Path | None) -> Path:
        candidates: list[Path] = []
        if requested is not None:
            candidates.append(Path(requested).expanduser())
        environment_path = os.environ.get("OCEANSIM_BBS3D_LIBRARY")
        if environment_path:
            candidates.append(Path(environment_path).expanduser())
        candidates.append(
            cls._repository_root()
            / "native"
            / "bbs3d_bridge"
            / "build"
            / "liboceansim_bbs3d.so"
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved
        rendered = ", ".join(str(item) for item in candidates)
        raise FileNotFoundError(
            "3D-BBS bridge was not found; run "
            "standalone/build_bbs3d_bridge.sh (searched: " + rendered + ")"
        )

    @staticmethod
    def _pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def localize(
        self,
        target: np.ndarray,
        source: np.ndarray,
        *,
        minimum_level_resolution_m: float,
        maximum_level: int,
        minimum_translation_xyz: np.ndarray,
        maximum_translation_xyz: np.ndarray,
        minimum_rpy: np.ndarray,
        maximum_rpy: np.ndarray,
        timeout_milliseconds: int,
    ) -> dict[str, object]:
        target = np.ascontiguousarray(target, dtype=np.float32).reshape(-1, 3)
        source = np.ascontiguousarray(source, dtype=np.float32).reshape(-1, 3)
        minimum_translation = np.ascontiguousarray(
            minimum_translation_xyz, dtype=np.float32
        ).reshape(3)
        maximum_translation = np.ascontiguousarray(
            maximum_translation_xyz, dtype=np.float32
        ).reshape(3)
        minimum_angles = np.ascontiguousarray(minimum_rpy, dtype=np.float32).reshape(3)
        maximum_angles = np.ascontiguousarray(maximum_rpy, dtype=np.float32).reshape(3)
        transform = np.eye(4, dtype=np.float32)
        score = ctypes.c_int(0)
        score_fraction = ctypes.c_float(0.0)
        elapsed = ctypes.c_double(0.0)
        localized = ctypes.c_int(0)
        timed_out = ctypes.c_int(0)
        error = ctypes.create_string_buffer(1024)
        status = self._function(
            self._pointer(target),
            target.shape[0],
            self._pointer(source),
            source.shape[0],
            float(minimum_level_resolution_m),
            int(maximum_level),
            self._pointer(minimum_translation),
            self._pointer(maximum_translation),
            self._pointer(minimum_angles),
            self._pointer(maximum_angles),
            0.0,
            int(timeout_milliseconds),
            self._pointer(transform.reshape(-1)),
            ctypes.byref(score),
            ctypes.byref(score_fraction),
            ctypes.byref(elapsed),
            ctypes.byref(localized),
            ctypes.byref(timed_out),
            error,
            len(error),
        )
        if status != 0:
            detail = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"3D-BBS bridge failed ({status}): {detail}")
        if not localized.value:
            transform = np.eye(4, dtype=np.float32)
        return {
            "transform": np.asarray(transform, dtype=np.float64),
            "score": int(score.value),
            "score_fraction": float(score_fraction.value),
            "elapsed_milliseconds": float(elapsed.value),
            "localized": bool(localized.value),
            "timed_out": bool(timed_out.value),
        }


class BBS3DCandidateVerifier:
    """Verify one map-frame pose proposal with full 3-D MBES geometry."""

    def __init__(
        self,
        terrain: MeasuredBathymetryGrid,
        config: BBS3DVerifierConfig | None = None,
        *,
        library_path: str | Path | None = None,
        require_native: bool = False,
        native_backend: object | None = None,
    ) -> None:
        if not isinstance(terrain, MeasuredBathymetryGrid):
            raise TypeError("terrain must be a MeasuredBathymetryGrid")
        self.terrain = terrain
        self.config = config or BBS3DVerifierConfig()
        if not isinstance(self.config, BBS3DVerifierConfig):
            raise TypeError("config must be a BBS3DVerifierConfig")
        self.native_error: str | None = None
        if native_backend is not None:
            self._native = native_backend
        else:
            try:
                self._native = _NativeBBS3D(library_path)
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                self._native = None
                self.native_error = str(exc)
                if require_native:
                    raise RuntimeError(self.native_error) from exc

    @property
    def available(self) -> bool:
        return self._native is not None

    @staticmethod
    def _pose2(values: np.ndarray | Iterable[float]) -> np.ndarray:
        try:
            pose = np.asarray(values, dtype=np.float64).reshape(3).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("candidate_pose2 must contain three finite values") from exc
        if not np.isfinite(pose).all():
            raise ValueError("candidate_pose2 must contain three finite values")
        pose[2] = wrap_angle(float(pose[2]))
        return pose

    def _reject(
        self,
        reason: str,
        pose2: np.ndarray,
        *,
        source_points: int = 0,
        target_points: int = 0,
        score: int = 0,
        score_fraction: float = 0.0,
        elapsed_milliseconds: float = 0.0,
        timed_out: bool = False,
        available: bool | None = None,
        transformation: np.ndarray | None = None,
        correction_pose2: np.ndarray | None = None,
    ) -> BBS3DVerificationResult:
        return BBS3DVerificationResult(
            available=self.available if available is None else bool(available),
            accepted=False,
            matched_pose2=pose2,
            correction_pose2=(
                np.zeros(3, dtype=np.float64)
                if correction_pose2 is None
                else correction_pose2
            ),
            transformation=(
                np.eye(4, dtype=np.float64)
                if transformation is None
                else transformation
            ),
            score=score,
            score_fraction=score_fraction,
            source_points=source_points,
            target_points=target_points,
            elapsed_milliseconds=elapsed_milliseconds,
            timed_out=timed_out,
            rejection_reason=reason,
        )

    def _target_cloud(
        self, candidate_pose2: np.ndarray, footprint_radius_m: float
    ) -> np.ndarray:
        radius = (
            footprint_radius_m
            + self.config.maximum_translation_m
            + self.config.target_margin_m
        )
        resolution = self.terrain.resolution_m
        centre_column = candidate_pose2[0] / resolution + 0.5 * (
            self.terrain.navigation_elevation_m.shape[1] - 1
        )
        centre_row = candidate_pose2[1] / resolution + 0.5 * (
            self.terrain.navigation_elevation_m.shape[0] - 1
        )
        cell_radius = int(math.ceil(radius / resolution)) + 1
        column_minimum = max(0, int(math.floor(centre_column)) - cell_radius)
        column_maximum = min(
            self.terrain.navigation_elevation_m.shape[1] - 1,
            int(math.ceil(centre_column)) + cell_radius,
        )
        row_minimum = max(0, int(math.floor(centre_row)) - cell_radius)
        row_maximum = min(
            self.terrain.navigation_elevation_m.shape[0] - 1,
            int(math.ceil(centre_row)) + cell_radius,
        )
        stride = self.config.target_stride
        rows = np.arange(row_minimum, row_maximum + 1, stride, dtype=np.int64)
        columns = np.arange(
            column_minimum, column_maximum + 1, stride, dtype=np.int64
        )
        if rows.size == 0 or columns.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        valid = np.asarray(
            self.terrain.valid_mask[np.ix_(rows, columns)], dtype=bool
        )
        elevations = np.asarray(
            self.terrain.navigation_elevation_m[np.ix_(rows, columns)],
            dtype=np.float64,
        )
        valid &= np.isfinite(elevations)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)
        grid_columns, grid_rows = np.meshgrid(columns, rows)
        x = (
            grid_columns.astype(np.float64)
            - 0.5 * (self.terrain.navigation_elevation_m.shape[1] - 1)
        ) * resolution
        y = (
            grid_rows.astype(np.float64)
            - 0.5 * (self.terrain.navigation_elevation_m.shape[0] - 1)
        ) * resolution
        # Rebase horizontal coordinates at the proposal.  Source x/y remain
        # body-centred, so the native translation is exactly the residual
        # correction instead of a kilometre-scale world coordinate.
        target = np.column_stack(
            (
                x[valid] - candidate_pose2[0],
                y[valid] - candidate_pose2[1],
                elevations[valid],
            )
        )
        return np.ascontiguousarray(target, dtype=np.float32)

    def verify(
        self,
        points_body: np.ndarray,
        state: InertialState,
        candidate_pose2: np.ndarray | Iterable[float],
    ) -> BBS3DVerificationResult:
        """Score a candidate while holding pressure depth and attitude fixed."""

        if not isinstance(state, InertialState):
            raise TypeError("state must be an InertialState")
        pose2 = self._pose2(candidate_pose2)
        if not self.available:
            return self._reject(
                "native_backend_unavailable",
                pose2,
                available=False,
            )
        points = np.asarray(points_body, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < self.config.minimum_source_points:
            return self._reject(
                "insufficient_source_points",
                pose2,
                source_points=points.shape[0],
            )
        if points.shape[0] > self.config.maximum_source_points:
            selected = np.linspace(
                0,
                points.shape[0] - 1,
                self.config.maximum_source_points,
                dtype=np.int64,
            )
            points = points[selected]

        roll, pitch, _ = rpy_from_rotation(state.rotation)
        candidate_rotation = rotation_from_rpy(
            float(roll), float(pitch), float(pose2[2])
        )
        source = points @ candidate_rotation.T
        # Embedding pressure depth in the source cloud lets the native search
        # hold dz exactly at zero, rather than exploiting a vertical offset to
        # make a false horizontal basin look geometrically plausible.
        source[:, 2] += float(state.position[2])
        footprint_radius = float(np.max(np.linalg.norm(source[:, :2], axis=1)))
        target = self._target_cloud(pose2, footprint_radius)
        if target.shape[0] < self.config.minimum_source_points:
            return self._reject(
                "insufficient_target_map",
                pose2,
                source_points=source.shape[0],
                target_points=target.shape[0],
            )

        translation = self.config.maximum_translation_m
        yaw_limit = math.radians(self.config.maximum_yaw_deg)
        try:
            native = self._native.localize(
                target,
                np.ascontiguousarray(source, dtype=np.float32),
                minimum_level_resolution_m=(
                    self.config.minimum_level_resolution_m
                ),
                maximum_level=self.config.maximum_level,
                minimum_translation_xyz=np.array(
                    [-translation, -translation, 0.0], dtype=np.float32
                ),
                maximum_translation_xyz=np.array(
                    [translation, translation, 0.0], dtype=np.float32
                ),
                minimum_rpy=np.array([0.0, 0.0, -yaw_limit], dtype=np.float32),
                maximum_rpy=np.array([0.0, 0.0, yaw_limit], dtype=np.float32),
                timeout_milliseconds=self.config.timeout_milliseconds,
            )
        except (RuntimeError, ValueError, OverflowError) as exc:
            return self._reject(
                "native_backend_error:" + str(exc),
                pose2,
                source_points=source.shape[0],
                target_points=target.shape[0],
            )

        transform = np.asarray(native["transform"], dtype=np.float64).reshape(4, 4)
        elapsed = max(0.0, float(native["elapsed_milliseconds"]))
        score = max(0, int(native["score"]))
        score_fraction = float(np.clip(native["score_fraction"], 0.0, 1.0))
        timed_out = bool(native["timed_out"])
        localized = bool(native["localized"])
        if timed_out:
            return self._reject(
                "search_timeout",
                pose2,
                source_points=source.shape[0],
                target_points=target.shape[0],
                score=score,
                score_fraction=score_fraction,
                elapsed_milliseconds=elapsed,
                timed_out=True,
            )
        if not localized or not np.isfinite(transform).all():
            return self._reject(
                "no_geometric_solution",
                pose2,
                source_points=source.shape[0],
                target_points=target.shape[0],
                score=score,
                score_fraction=score_fraction,
                elapsed_milliseconds=elapsed,
            )

        yaw_correction = math.atan2(float(transform[1, 0]), float(transform[0, 0]))
        correction = np.array(
            [transform[0, 3], transform[1, 3], yaw_correction],
            dtype=np.float64,
        )
        matched = pose2 + correction
        matched[2] = wrap_angle(float(matched[2]))
        at_boundary = bool(
            abs(correction[0])
            >= translation - 0.5 * self.config.minimum_level_resolution_m
            or abs(correction[1])
            >= translation - 0.5 * self.config.minimum_level_resolution_m
            or abs(math.degrees(correction[2]))
            >= self.config.maximum_yaw_deg - 0.05
        )
        if self.config.reject_search_boundary and at_boundary:
            return self._reject(
                "search_boundary_reached",
                matched,
                source_points=source.shape[0],
                target_points=target.shape[0],
                score=score,
                score_fraction=score_fraction,
                elapsed_milliseconds=elapsed,
                transformation=transform,
                correction_pose2=correction,
            )
        if score_fraction < self.config.minimum_score_fraction:
            return self._reject(
                "geometric_score_below_threshold",
                matched,
                source_points=source.shape[0],
                target_points=target.shape[0],
                score=score,
                score_fraction=score_fraction,
                elapsed_milliseconds=elapsed,
                transformation=transform,
                correction_pose2=correction,
            )
        return BBS3DVerificationResult(
            available=True,
            accepted=True,
            matched_pose2=matched,
            correction_pose2=correction,
            transformation=transform,
            score=score,
            score_fraction=score_fraction,
            source_points=source.shape[0],
            target_points=target.shape[0],
            elapsed_milliseconds=elapsed,
            timed_out=False,
        )
