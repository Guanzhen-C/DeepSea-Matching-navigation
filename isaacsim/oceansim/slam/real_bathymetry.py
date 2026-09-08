"""Measured bathymetry and terrain-following survey support.

Both the legacy cropped NPZ and the complete, memory-mapped Monterey Canyon
product are supported.  Missing source cells remain missing throughout
sampling and mesh construction: OceanSim must not invent a surface across a
survey gap merely to make the renderer's job easier.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .large_scale_sim import GroundTruthState
from .math3d import rotation_from_rpy, so3_log


@dataclass(frozen=True)
class TerrainFollowingSurveyReport:
    """Read-only, one-lap survey geometry and coverage summary.

    Depth is positive down from the local ``z=0`` datum.  Altitude is the
    vehicle height above the measured (not navigation-smoothed) seabed.
    Clearance is a conservative centreline distance to either a NoData cell
    or the edge of the measured raster.
    """

    sample_count: int
    period_s: float
    path_length_m: float
    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]
    auv_depth_range_m: tuple[float, float]
    seabed_depth_range_m: tuple[float, float]
    altitude_range_m: tuple[float, float]
    pitch_range_deg: tuple[float, float]
    bathymetry_fully_valid: bool
    backscatter_fully_valid: bool | None
    bathymetry_min_clearance_m: float
    backscatter_min_clearance_m: float | None
    maximum_horizontal_curvature_per_m: float | None = None
    validation_sample_count: int | None = None
    validation_max_spacing_m: float | None = None
    waypoint_count: int | None = None


class MeasuredBathymetryGrid:
    """Regular, georeferenced bathymetry grid in a local Z-up frame."""

    def __init__(
        self,
        elevation_m: np.ndarray,
        *,
        resolution_m: float,
        navigation_elevation_m: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
        backscatter: np.ndarray | None = None,
        backscatter_valid_mask: np.ndarray | None = None,
        backscatter_source_mask: np.ndarray | None = None,
        metadata: dict | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        # ``asanyarray`` deliberately preserves np.memmap.  Casting the full
        # 3534 x 5842 source grid to float64 here used to allocate another
        # ~165 MB and defeated the on-disk complete-map representation.
        elevation = np.asanyarray(elevation_m)
        if elevation.ndim != 2 or min(elevation.shape) < 2:
            raise ValueError("elevation_m must be a two-dimensional grid")
        if elevation.dtype.kind == "c":
            raise ValueError("elevation_m must be real-valued")
        if elevation.dtype.kind != "f":
            elevation = np.asarray(elevation, dtype=np.float64)
        if not np.isfinite(resolution_m) or resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive")

        navigation = (
            elevation
            if navigation_elevation_m is None
            else np.asanyarray(navigation_elevation_m)
        )
        if navigation.dtype.kind == "c":
            raise ValueError("navigation elevation must be real-valued")
        if navigation.dtype.kind != "f":
            navigation = np.asarray(navigation, dtype=np.float64)
        if navigation.shape != elevation.shape:
            raise ValueError("navigation elevation must match the measured grid")

        if valid_mask is None:
            terrain_valid = np.isfinite(elevation)
        else:
            terrain_valid = np.asanyarray(valid_mask)
            if terrain_valid.shape != elevation.shape:
                raise ValueError("valid_mask must match the measured grid")
            if terrain_valid.dtype != np.bool_:
                terrain_valid = np.asarray(terrain_valid, dtype=bool)
        elevation_range = self._validate_finite_on_mask(
            elevation, terrain_valid, "elevation_m"
        )
        if elevation_range is None:
            raise ValueError("valid_mask must contain at least one measured cell")
        self._validate_finite_on_mask(
            navigation, terrain_valid, "navigation elevation"
        )

        measured_backscatter = None
        measured_backscatter_valid = None
        if backscatter is not None:
            measured_backscatter = np.asanyarray(backscatter)
            if measured_backscatter.shape != elevation.shape:
                raise ValueError("backscatter must match the measured grid")
            if measured_backscatter.dtype.kind == "c":
                raise ValueError("backscatter must be real-valued")
            if measured_backscatter.dtype.kind != "f":
                measured_backscatter = np.asarray(
                    measured_backscatter, dtype=np.float32
                )
            if backscatter_valid_mask is None:
                measured_backscatter_valid = (
                    terrain_valid & np.isfinite(measured_backscatter)
                )
            else:
                measured_backscatter_valid = np.asanyarray(
                    backscatter_valid_mask
                )
                if measured_backscatter_valid.shape != elevation.shape:
                    raise ValueError(
                        "backscatter_valid_mask must match the measured grid"
                    )
                if measured_backscatter_valid.dtype != np.bool_:
                    measured_backscatter_valid = np.asarray(
                        measured_backscatter_valid, dtype=bool
                    )
                if self._outside_mask_any(
                    measured_backscatter_valid, terrain_valid
                ):
                    raise ValueError(
                        "backscatter_valid_mask must be inside valid bathymetry"
                    )
            self._validate_finite_on_mask(
                measured_backscatter,
                measured_backscatter_valid,
                "backscatter",
            )
        elif backscatter_valid_mask is not None:
            raise ValueError(
                "backscatter_valid_mask requires a backscatter grid"
            )

        source_mask = None
        if backscatter_source_mask is not None:
            source_mask = np.asanyarray(backscatter_source_mask)
            if source_mask.shape != elevation.shape:
                raise ValueError(
                    "backscatter_source_mask must match the measured grid"
                )
            if source_mask.dtype.kind not in "uib":
                raise ValueError("backscatter_source_mask must be an integer grid")
            if measured_backscatter is None:
                raise ValueError(
                    "backscatter_source_mask requires a backscatter grid"
                )

        self.elevation_m = elevation
        self.navigation_elevation_m = navigation
        self.valid_mask = terrain_valid
        # The longer alias mirrors the complete-map file name and makes
        # provenance-related call sites self-documenting.
        self.bathymetry_valid_mask = terrain_valid
        self.backscatter = measured_backscatter
        self.backscatter_valid_mask = measured_backscatter_valid
        self.backscatter_source_mask = source_mask
        self.resolution_m = float(resolution_m)
        self.metadata = dict(metadata or {})
        self.source_path = None if source_path is None else Path(source_path)
        self.length = float((elevation.shape[0] - 1) * self.resolution_m)
        self.width = float((elevation.shape[1] - 1) * self.resolution_m)
        self._minimum_elevation_m, self._maximum_elevation_m = elevation_range

    @staticmethod
    def _outside_mask_any(selected: np.ndarray, container: np.ndarray) -> bool:
        """Chunked logical-any without materialising a full-grid temporary."""

        rows_per_chunk = 256
        for start in range(0, selected.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, selected.shape[0])
            if np.any(selected[start:stop] & ~container[start:stop]):
                return True
        return False

    @staticmethod
    def _validate_finite_on_mask(
        values: np.ndarray, valid: np.ndarray, name: str
    ) -> tuple[float, float] | None:
        """Validate large mmap grids a few rows at a time."""

        rows_per_chunk = 256
        minimum = np.inf
        maximum = -np.inf
        for start in range(0, values.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, values.shape[0])
            selected = valid[start:stop]
            if not np.any(selected):
                continue
            measured = values[start:stop][selected]
            if not np.all(np.isfinite(measured)):
                raise ValueError(f"{name} must be finite at valid cells")
            minimum = min(minimum, float(np.min(measured)))
            maximum = max(maximum, float(np.max(measured)))
        if minimum == np.inf:
            return None
        return minimum, maximum

    @property
    def has_measured_backscatter(self) -> bool:
        return self.backscatter is not None

    @property
    def backscatter_coverage_fraction(self) -> float:
        if self.backscatter_valid_mask is None:
            return 0.0
        valid_terrain = int(np.count_nonzero(self.valid_mask))
        if valid_terrain == 0:
            return 0.0
        return float(
            np.count_nonzero(self.backscatter_valid_mask) / valid_terrain
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        mmap_mode: str | None = "r",
    ) -> "MeasuredBathymetryGrid":
        """Load either a complete-map directory or a legacy NPZ file."""

        source = Path(path)
        if source.is_dir():
            return cls.from_directory(source, mmap_mode=mmap_mode)
        return cls.from_npz(source)

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        *,
        mmap_mode: str | None = "r",
    ) -> "MeasuredBathymetryGrid":
        """Load a prepared complete map from independently mmap-able NPYs."""

        source = Path(path)
        metadata_path = source / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"complete bathymetry metadata not found: {metadata_path}"
            )
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)

        def required(name: str) -> np.ndarray:
            candidate = source / name
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"complete bathymetry array not found: {candidate}"
                )
            return np.load(candidate, mmap_mode=mmap_mode, allow_pickle=False)

        def optional(name: str) -> np.ndarray | None:
            candidate = source / name
            if not candidate.is_file():
                return None
            return np.load(candidate, mmap_mode=mmap_mode, allow_pickle=False)

        elevation = required("elevation_m.npy")
        navigation = optional("navigation_elevation_m.npy")
        terrain_valid = optional("bathymetry_valid_mask.npy")
        backscatter = optional("backscatter.npy")
        backscatter_valid = optional("backscatter_valid_mask.npy")
        backscatter_sources = optional("backscatter_source_mask.npy")
        if "resolution_m" not in metadata:
            raise ValueError(
                f"resolution_m is missing from complete-map metadata: {metadata_path}"
            )
        return cls(
            elevation,
            resolution_m=float(metadata["resolution_m"]),
            navigation_elevation_m=navigation,
            valid_mask=terrain_valid,
            backscatter=backscatter,
            backscatter_valid_mask=backscatter_valid,
            backscatter_source_mask=backscatter_sources,
            metadata=metadata,
            source_path=source,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "MeasuredBathymetryGrid":
        """Load a preprocessed measured grid and its adjacent JSON metadata."""

        source = Path(path)
        with np.load(source, allow_pickle=False) as payload:
            elevation = payload["elevation_m"]
            navigation = (
                payload["navigation_elevation_m"]
                if "navigation_elevation_m" in payload
                else elevation
            )
            resolution = float(payload["resolution_m"])
            valid = (
                payload["bathymetry_valid_mask"]
                if "bathymetry_valid_mask" in payload
                else None
            )
            backscatter = (
                payload["backscatter"] if "backscatter" in payload else None
            )
            backscatter_valid = (
                payload["backscatter_valid_mask"]
                if "backscatter_valid_mask" in payload
                else None
            )
            backscatter_sources = (
                payload["backscatter_source_mask"]
                if "backscatter_source_mask" in payload
                else None
            )
        metadata_path = source.with_suffix(".json")
        metadata = {}
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
        return cls(
            elevation,
            resolution_m=resolution,
            navigation_elevation_m=navigation,
            valid_mask=valid,
            backscatter=backscatter,
            backscatter_valid_mask=backscatter_valid,
            backscatter_source_mask=backscatter_sources,
            metadata=metadata,
            source_path=source,
        )

    @staticmethod
    def _stride_indices(size: int, stride: int) -> np.ndarray:
        if isinstance(stride, bool) or int(stride) != stride or stride < 1:
            raise ValueError("mesh stride must be a positive integer")
        indices = np.arange(0, size, int(stride), dtype=np.int64)
        if indices[-1] != size - 1:
            indices = np.append(indices, size - 1)
        return indices

    def mesh_grid(
        self,
        *,
        stride: int = 1,
        copy: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return local grid vertices, optionally at a whole-map LOD stride."""

        rows, columns = self.elevation_m.shape
        row_indices = self._stride_indices(rows, stride)
        column_indices = self._stride_indices(columns, stride)
        x = column_indices.astype(np.float64) - 0.5 * (columns - 1)
        y = row_indices.astype(np.float64) - 0.5 * (rows - 1)
        grid_x, grid_y = np.meshgrid(
            x * self.resolution_m,
            y * self.resolution_m,
        )
        grid_z = self.elevation_m[np.ix_(row_indices, column_indices)]
        if copy:
            grid_z = grid_z.copy()
        return grid_x, grid_y, grid_z

    def _coarse_cells_fully_measured(
        self,
        row_indices: np.ndarray,
        column_indices: np.ndarray,
    ) -> np.ndarray:
        """Reject LOD quads that would bridge any source-grid NoData cell."""

        row0 = row_indices[:-1]
        row1 = row_indices[1:]
        column0 = column_indices[:-1]
        column1 = column_indices[1:]
        if np.all(np.diff(row_indices) == 1) and np.all(
            np.diff(column_indices) == 1
        ):
            valid = self.valid_mask
            return (
                valid[:-1, :-1]
                & valid[:-1, 1:]
                & valid[1:, :-1]
                & valid[1:, 1:]
            )

        # An integral image lets every coarse cell check its complete source
        # footprint.  Checking only four LOD corners would cap small survey
        # holes with large, fictitious triangles.
        rows, columns = self.valid_mask.shape
        invalid_integral = np.zeros((rows + 1, columns + 1), dtype=np.uint32)
        invalid_integral[1:, 1:] = ~self.valid_mask
        np.cumsum(invalid_integral, axis=0, out=invalid_integral)
        np.cumsum(invalid_integral, axis=1, out=invalid_integral)
        invalid_count = (
            invalid_integral[np.ix_(row1 + 1, column1 + 1)].astype(np.int64)
            - invalid_integral[np.ix_(row0, column1 + 1)]
            - invalid_integral[np.ix_(row1 + 1, column0)]
            + invalid_integral[np.ix_(row0, column0)]
        )
        return invalid_count == 0

    def triangle_mesh(
        self,
        *,
        stride: int = 1,
        compact: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a measured-only triangle mesh at the requested LOD.

        The winding matches the USD collision mesh used by the Isaac Sim demo.
        ``compact=True`` removes vertices unused after NoData triangles are
        discarded.  At LOD strides above one, a triangle is emitted only when
        every native 5 m cell under its quad is measured.
        """

        source_rows, source_columns = self.elevation_m.shape
        row_indices = self._stride_indices(source_rows, stride)
        column_indices = self._stride_indices(source_columns, stride)
        rows = row_indices.size
        columns = column_indices.size
        if rows * columns > np.iinfo(np.int32).max:
            raise ValueError("mesh has too many vertices for int32 indices")
        upper_left = (
            np.arange(rows - 1, dtype=np.int32)[:, None] * columns
            + np.arange(columns - 1, dtype=np.int32)[None, :]
        )
        upper_right = upper_left + 1
        lower_left = upper_left + columns
        lower_right = lower_left + 1
        triangles = np.stack(
            (
                upper_left,
                upper_right,
                lower_right,
                upper_left,
                lower_right,
                lower_left,
            ),
            axis=-1,
        ).reshape(-1, 3)

        if int(stride) == 1:
            valid = self.valid_mask
            first_valid = (
                valid[:-1, :-1]
                & valid[:-1, 1:]
                & valid[1:, 1:]
            )
            second_valid = (
                valid[:-1, :-1]
                & valid[1:, 1:]
                & valid[1:, :-1]
            )
            triangle_valid = np.stack(
                (first_valid, second_valid), axis=-1
            ).reshape(-1)
        else:
            cell_valid = self._coarse_cells_fully_measured(
                row_indices, column_indices
            )
            triangle_valid = np.repeat(cell_valid.reshape(-1), 2)
        triangles = triangles[triangle_valid]

        if triangles.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty(0, dtype=np.int32),
            )

        if compact:
            used, inverse = np.unique(triangles, return_inverse=True)
            coarse_rows = used // columns
            coarse_columns = used % columns
            source_row = row_indices[coarse_rows]
            source_column = column_indices[coarse_columns]
            z = self.elevation_m[source_row, source_column]
            x = (
                source_column.astype(np.float64)
                - 0.5 * (source_columns - 1)
            ) * self.resolution_m
            y = (
                source_row.astype(np.float64) - 0.5 * (source_rows - 1)
            ) * self.resolution_m
            points = np.column_stack((x, y, z)).astype(np.float32)
            indices = inverse.astype(np.int32, copy=False).reshape(-1)
        else:
            grid_x, grid_y, grid_z = self.mesh_grid(
                stride=stride, copy=False
            )
            points = np.column_stack(
                (grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1))
            ).astype(np.float32)
            indices = triangles.astype(np.int32, copy=False).reshape(-1)
        return np.ascontiguousarray(points), np.ascontiguousarray(indices)

    def mesh(
        self,
        *,
        stride: int = 1,
        compact: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Concise alias for :meth:`triangle_mesh`."""

        return self.triangle_mesh(stride=stride, compact=compact)

    def _sample(
        self,
        values: np.ndarray,
        x,
        y,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )
        shape = x.shape
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        result = np.full(flat_x.shape, np.nan, dtype=np.float64)
        inside = (
            (flat_x >= -0.5 * self.width)
            & (flat_x <= 0.5 * self.width)
            & (flat_y >= -0.5 * self.length)
            & (flat_y <= 0.5 * self.length)
        )
        if not np.any(inside):
            return result.reshape(shape)

        selected = np.flatnonzero(inside)
        column = (
            flat_x[selected] + 0.5 * self.width
        ) / self.resolution_m
        row = (
            flat_y[selected] + 0.5 * self.length
        ) / self.resolution_m
        column0 = np.minimum(
            np.floor(column).astype(np.int64), values.shape[1] - 2
        )
        row0 = np.minimum(
            np.floor(row).astype(np.int64), values.shape[0] - 2
        )
        column_fraction = column - column0
        row_fraction = row - row0
        weights = (
            (1.0 - row_fraction) * (1.0 - column_fraction),
            (1.0 - row_fraction) * column_fraction,
            row_fraction * (1.0 - column_fraction),
            row_fraction * column_fraction,
        )
        corners = (
            (row0, column0),
            (row0, column0 + 1),
            (row0 + 1, column0),
            (row0 + 1, column0 + 1),
        )
        support = np.ones(selected.size, dtype=bool)
        interpolated = np.zeros(selected.size, dtype=np.float64)
        for weight, (corner_row, corner_column) in zip(weights, corners):
            corner_value = values[corner_row, corner_column]
            corner_valid = np.isfinite(corner_value)
            if valid_mask is not None:
                corner_valid &= valid_mask[corner_row, corner_column]
            required = weight > 1.0e-12
            support &= ~required | corner_valid
            # NaN * 0 is NaN, hence the explicit zero for unused corners.
            interpolated += np.where(required, corner_value * weight, 0.0)
        result[selected[support]] = interpolated[support]
        return result.reshape(shape)

    def seabed_height(self, x, y) -> np.ndarray:
        return self._sample(
            self.elevation_m, x, y, valid_mask=self.valid_mask
        )

    def navigation_height(self, x, y) -> np.ndarray:
        """Return a survey-safe low-pass version of the measured surface."""

        return self._sample(
            self.navigation_elevation_m,
            x,
            y,
            valid_mask=self.valid_mask,
        )

    def normal(self, x, y) -> np.ndarray:
        epsilon = self.resolution_m
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )
        centre = self.seabed_height(x, y)
        x_before = self.seabed_height(x - epsilon, y)
        x_after = self.seabed_height(x + epsilon, y)
        y_before = self.seabed_height(x, y - epsilon)
        y_after = self.seabed_height(x, y + epsilon)
        dz_dx = np.where(
            np.isfinite(x_before) & np.isfinite(x_after),
            (x_after - x_before) / (2.0 * epsilon),
            np.where(
                np.isfinite(centre) & np.isfinite(x_after),
                (x_after - centre) / epsilon,
                (centre - x_before) / epsilon,
            ),
        )
        dz_dy = np.where(
            np.isfinite(y_before) & np.isfinite(y_after),
            (y_after - y_before) / (2.0 * epsilon),
            np.where(
                np.isfinite(centre) & np.isfinite(y_after),
                (y_after - centre) / epsilon,
                (centre - y_before) / epsilon,
            ),
        )
        normals = np.stack((-dz_dx, -dz_dy, np.ones_like(dz_dx)), axis=-1)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normalized = normals / norm
        normalized[~np.isfinite(centre)] = np.nan
        return normalized

    def measured_reflectivity(self, x, y) -> np.ndarray:
        """Sample only measured acoustic backscatter; gaps return NaN."""

        if self.backscatter is None or self.backscatter_valid_mask is None:
            x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
            return np.full(x.shape, np.nan, dtype=np.float64)
        return self._sample(
            self.backscatter,
            x,
            y,
            valid_mask=self.backscatter_valid_mask,
        )

    def _modelled_reflectivity(self, x, y) -> np.ndarray:
        normal = self.normal(x, y)
        slope = np.sqrt(np.maximum(0.0, 1.0 - normal[..., 2] ** 2))
        texture = (
            0.48
            + 0.38 * np.clip(slope / 0.55, 0.0, 1.0)
            + 0.07 * np.sin(np.asarray(x) / 17.0 + np.asarray(y) / 29.0)
        )
        return np.clip(texture, 0.12, 1.35)

    def reflectivity(self, x, y, *, measured_only: bool = False) -> np.ndarray:
        """Return measured backscatter first, with a legacy model fallback.

        ``measured_only=True`` is available to scientific call sites that must
        distinguish true acoustic coverage from bathymetry-only coverage.  The
        default fallback preserves compatibility with simulations whose route
        crosses a measured-backscatter gap.
        """

        measured = self.measured_reflectivity(x, y)
        if measured_only:
            return measured
        measured_flat = measured.reshape(-1)
        missing = ~np.isfinite(measured_flat)
        if not np.any(missing):
            return measured
        x_array, y_array = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        result = measured_flat.copy()
        result[missing] = self._modelled_reflectivity(
            x_array.reshape(-1)[missing], y_array.reshape(-1)[missing]
        )
        return result.reshape(measured.shape)

    def reflectivity_source(self, x, y) -> np.ndarray:
        """Return the nearest measured-backscatter provenance bit mask."""

        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        dtype = (
            np.uint8
            if self.backscatter_source_mask is None
            else self.backscatter_source_mask.dtype
        )
        result = np.zeros(x.shape, dtype=dtype)
        if self.backscatter_source_mask is None:
            return result
        inside = (
            (x >= -0.5 * self.width)
            & (x <= 0.5 * self.width)
            & (y >= -0.5 * self.length)
            & (y <= 0.5 * self.length)
        )
        if not np.any(inside):
            return result
        columns = np.rint(
            (x[inside] + 0.5 * self.width) / self.resolution_m
        ).astype(np.int64)
        rows = np.rint(
            (y[inside] + 0.5 * self.length) / self.resolution_m
        ).astype(np.int64)
        sources = self.backscatter_source_mask[rows, columns]
        if self.backscatter_valid_mask is not None:
            sources = np.where(
                self.backscatter_valid_mask[rows, columns], sources, 0
            )
        result[inside] = sources
        return result

    def seabed_color(self, x, y) -> np.ndarray:
        """Create sediment/rock colours from measured depth and local slope."""

        height = self.seabed_height(x, y)
        normal = self.normal(x, y)
        slope = np.clip((1.0 - normal[..., 2]) / 0.18, 0.0, 1.0)
        minimum = self._minimum_elevation_m
        relief = self._maximum_elevation_m - minimum
        high = np.clip(
            (height - minimum) / max(relief, 1.0e-6),
            0.0,
            1.0,
        )
        sediment = np.stack(
            (
                0.17 + 0.16 * high,
                0.27 + 0.15 * high,
                0.24 + 0.08 * high,
            ),
            axis=-1,
        )
        rock = np.array([0.25, 0.29, 0.27], dtype=np.float64)
        colour = sediment * (1.0 - 0.58 * slope[..., None])
        colour += rock * (0.58 * slope[..., None])
        return np.clip(colour, 0.03, 0.65)

    def intersect_rays(
        self,
        origin_world: np.ndarray,
        directions_world: np.ndarray,
        *,
        min_range: float,
        max_range: float,
        iterations: int = 24,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the measured height field using bisection."""

        origin = np.asarray(origin_world, dtype=np.float64).reshape(3)
        directions = np.asarray(directions_world, dtype=np.float64).reshape(-1, 3)
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        low = np.full(directions.shape[0], float(min_range), dtype=np.float64)
        high = np.full(directions.shape[0], float(max_range), dtype=np.float64)

        def signed_height(distance):
            points = origin + directions * distance[:, None]
            return points[:, 2] - self.seabed_height(points[:, 0], points[:, 1])

        f_low = signed_height(low)
        f_high = signed_height(high)
        valid = (
            np.isfinite(f_low)
            & np.isfinite(f_high)
            & (f_low >= 0.0)
            & (f_high <= 0.0)
            & (directions[:, 2] < -1.0e-4)
        )
        for _ in range(int(iterations)):
            middle = 0.5 * (low + high)
            f_middle = signed_height(middle)
            valid &= np.isfinite(f_middle)
            above = f_middle > 0.0
            low = np.where(valid & above, middle, low)
            high = np.where(valid & ~above, middle, high)

        distances = 0.5 * (low + high)
        distances[~valid] = np.nan
        points = origin + directions * distances[:, None]
        points[~valid] = np.nan
        normals = np.zeros_like(points)
        normals[valid] = self.normal(points[valid, 0], points[valid, 1])
        normal_valid = np.all(np.isfinite(normals), axis=1)
        valid &= normal_valid
        distances[~valid] = np.nan
        points[~valid] = np.nan
        normals[~valid] = 0.0
        incidence = np.zeros(directions.shape[0], dtype=np.float64)
        incidence[valid] = np.clip(
            np.sum(normals[valid] * -directions[valid], axis=1), 0.0, 1.0
        )
        return valid, distances, points, incidence


def _minimum_mask_clearance_m(
    world: MeasuredBathymetryGrid,
    valid_mask: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> float:
    """Return conservative route clearance from a raster validity mask.

    Only a progressively enlarged crop around the route is transformed.  It
    gives the same minimum as a whole-map Euclidean distance transform once
    the nearest invalid cell is closer than every open edge of that crop.
    Keeping the transform local matters for the 20-million-cell source map.
    """

    mask = np.asanyarray(valid_mask)
    if mask.shape != world.elevation_m.shape:
        raise ValueError("validity mask must match the measured grid")
    if mask.dtype != np.bool_:
        mask = np.asarray(mask, dtype=bool)

    x = np.asarray(x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(y_m, dtype=np.float64).reshape(-1)
    if x.size == 0 or x.shape != y.shape:
        raise ValueError("route coordinates must be non-empty and paired")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("route coordinates must be finite")

    half_width = 0.5 * world.width
    half_length = 0.5 * world.length
    edge_clearance = float(
        np.min(
            np.minimum.reduce(
                (
                    x + half_width,
                    half_width - x,
                    y + half_length,
                    half_length - y,
                )
            )
        )
    )
    if edge_clearance <= 0.0:
        return 0.0

    resolution = world.resolution_m
    columns_f = (x + half_width) / resolution
    rows_f = (y + half_length) / resolution
    rows, columns = mask.shape
    route_column_min = int(np.floor(np.min(columns_f)))
    route_column_max = int(np.ceil(np.max(columns_f)))
    route_row_min = int(np.floor(np.min(rows_f)))
    route_row_max = int(np.ceil(np.max(rows_f)))

    # Start with at least 32 m around the route, then double until an invalid
    # cell is provably the global nearest one.  The final field is discarded
    # after sampling; this helper does not cache or mutate the source map.
    margin_cells = max(4, int(np.ceil(32.0 / resolution)))
    while True:
        column_start = max(0, route_column_min - margin_cells)
        column_stop = min(columns, route_column_max + margin_cells + 1)
        row_start = max(0, route_row_min - margin_cells)
        row_stop = min(rows, route_row_max + margin_cells + 1)
        crop = np.asarray(mask[row_start:row_stop, column_start:column_stop])

        if np.any(~crop):
            try:
                from scipy.ndimage import distance_transform_edt
            except ImportError as exc:  # pragma: no cover - Isaac ships SciPy
                raise RuntimeError(
                    "SciPy is required to calculate route-mask clearance"
                ) from exc

            distance = distance_transform_edt(crop, sampling=resolution)
            local_columns = columns_f - column_start
            local_rows = rows_f - row_start
            column0 = np.floor(local_columns).astype(np.int64)
            row0 = np.floor(local_rows).astype(np.int64)
            column0 = np.clip(column0, 0, distance.shape[1] - 2)
            row0 = np.clip(row0, 0, distance.shape[0] - 2)
            column_fraction = local_columns - column0
            row_fraction = local_rows - row0
            sampled_distance = (
                (1.0 - row_fraction)
                * (1.0 - column_fraction)
                * distance[row0, column0]
                + (1.0 - row_fraction)
                * column_fraction
                * distance[row0, column0 + 1]
                + row_fraction
                * (1.0 - column_fraction)
                * distance[row0 + 1, column0]
                + row_fraction
                * column_fraction
                * distance[row0 + 1, column0 + 1]
            )
            invalid_cell_distance = float(np.min(sampled_distance))

            open_edge_clearance_cells = []
            if column_start > 0:
                open_edge_clearance_cells.append(
                    float(np.min(columns_f - column_start))
                )
            if column_stop < columns:
                open_edge_clearance_cells.append(
                    float(np.min((column_stop - 1) - columns_f))
                )
            if row_start > 0:
                open_edge_clearance_cells.append(
                    float(np.min(rows_f - row_start))
                )
            if row_stop < rows:
                open_edge_clearance_cells.append(
                    float(np.min((row_stop - 1) - rows_f))
                )
            open_edge_clearance = (
                np.inf
                if not open_edge_clearance_cells
                else resolution * min(open_edge_clearance_cells)
            )
            if invalid_cell_distance < open_edge_clearance:
                # Bilinear terrain sampling needs neighbouring cell centres.
                # Subtracting one grid interval makes the reported distance
                # conservative with respect to that interpolation footprint.
                mask_clearance = max(0.0, invalid_cell_distance - resolution)
                return min(edge_clearance, mask_clearance)

        if (
            column_start == 0
            and column_stop == columns
            and row_start == 0
            and row_stop == rows
        ):
            # A fully valid raster is limited only by its rectangular edge.
            return edge_clearance
        margin_cells *= 2


class TerrainFollowingSurvey:
    """Smooth figure-eight survey maintaining altitude over measured terrain."""

    def __init__(
        self,
        world: MeasuredBathymetryGrid,
        *,
        period: float = 360.0,
        speed_mps: float | None = None,
        x_amplitude: float = 320.0,
        y_amplitude: float = 220.0,
        altitude_m: float = 28.0,
        center_x_m: float = 0.0,
        center_y_m: float = 0.0,
        pattern_yaw_deg: float = 0.0,
    ) -> None:
        self.world = world
        self.x_amplitude = self._positive_scalar(
            "x_amplitude", x_amplitude
        )
        self.y_amplitude = self._positive_scalar(
            "y_amplitude", y_amplitude
        )
        self.altitude_m = self._positive_scalar("altitude_m", altitude_m)
        self.center_x_m = self._finite_scalar("center_x_m", center_x_m)
        self.center_y_m = self._finite_scalar("center_y_m", center_y_m)
        self.pattern_yaw_deg = self._finite_scalar(
            "pattern_yaw_deg", pattern_yaw_deg
        )
        requested_period = self._positive_scalar("period", period)
        self.speed_mps = (
            None
            if speed_mps is None
            else self._positive_scalar("speed_mps", speed_mps)
        )
        # Reducing before conversion avoids loss of trigonometric precision
        # for large, but otherwise finite, degree values.
        yaw_degrees = (
            (self.pattern_yaw_deg + 180.0) % 360.0
        ) - 180.0
        self._pattern_yaw_rad = float(np.deg2rad(yaw_degrees))
        self._pattern_cos = float(np.cos(self._pattern_yaw_rad))
        self._pattern_sin = float(np.sin(self._pattern_yaw_rad))

        self._validate_geometric_bounds()
        self._validate_measured_route_support()

        self._arc_phases: np.ndarray | None = None
        self._arc_lengths: np.ndarray | None = None
        self.path_length_m: float | None = None
        if self.speed_mps is None:
            self.period = requested_period
        else:
            # Re-parameterize the same measured-terrain path by arc length.
            # The old sinusoidal clock changed speed by almost a factor of
            # three, reaching values unsuitable for a survey AUV.  A dense
            # deterministic lookup keeps the geometry unchanged while making
            # acoustic line spacing physically meaningful.
            phases = np.linspace(0.0, 2.0 * np.pi, 16_385)
            positions = self._positions_from_phases(phases)
            segment_lengths = np.linalg.norm(
                np.diff(positions, axis=0), axis=1
            )
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
            self.path_length_m = float(cumulative[-1])
            self.period = self.path_length_m / self.speed_mps
            self._arc_phases = phases
            self._arc_lengths = cumulative
        self.omega = 2.0 * np.pi / self.period

    @staticmethod
    def _finite_scalar(name: str, value: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _positive_scalar(cls, name: str, value: float) -> float:
        result = cls._finite_scalar(name, value)
        if result <= 0.0:
            raise ValueError(f"{name} must be positive")
        return result

    @staticmethod
    def _harmonic_range(primary: float, second: float) -> tuple[float, float]:
        """Exact range of ``primary*sin(p) + second*sin(2*p)``."""

        candidates = [0.0, np.pi]
        scale = max(abs(primary), abs(second), 1.0)
        scaled_primary = primary / scale
        scaled_second = second / scale
        if abs(scaled_second) <= np.finfo(np.float64).eps:
            candidates.extend((0.5 * np.pi, 1.5 * np.pi))
        else:
            discriminant = (
                scaled_primary * scaled_primary
                + 32.0 * scaled_second * scaled_second
            )
            for cosine in (
                (-scaled_primary + np.sqrt(discriminant))
                / (8.0 * scaled_second),
                (-scaled_primary - np.sqrt(discriminant))
                / (8.0 * scaled_second),
            ):
                if -1.0 <= cosine <= 1.0:
                    phase = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
                    candidates.extend((phase, 2.0 * np.pi - phase))
        phases = np.asarray(candidates, dtype=np.float64)
        values = scale * (
            scaled_primary * np.sin(phases)
            + scaled_second * np.sin(2.0 * phases)
        )
        return float(np.min(values)), float(np.max(values))

    def _validate_geometric_bounds(self) -> None:
        x_offset_range = self._harmonic_range(
            self._pattern_cos * self.x_amplitude,
            -self._pattern_sin * self.y_amplitude,
        )
        y_offset_range = self._harmonic_range(
            self._pattern_sin * self.x_amplitude,
            self._pattern_cos * self.y_amplitude,
        )
        x_range = (
            self.center_x_m + x_offset_range[0],
            self.center_x_m + x_offset_range[1],
        )
        y_range = (
            self.center_y_m + y_offset_range[0],
            self.center_y_m + y_offset_range[1],
        )
        if not np.all(np.isfinite((*x_range, *y_range))):
            raise ValueError("survey route bounds must be finite")
        if (
            x_range[0] <= -0.5 * self.world.width
            or x_range[1] >= 0.5 * self.world.width
        ):
            raise ValueError(
                "rotated survey route must stay strictly inside the measured "
                "grid x bounds"
            )
        if (
            y_range[0] <= -0.5 * self.world.length
            or y_range[1] >= 0.5 * self.world.length
        ):
            raise ValueError(
                "rotated survey route must stay strictly inside the measured "
                "grid y bounds"
            )

    def _horizontal_from_phases(
        self, phases: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phase = np.asarray(phases, dtype=np.float64)
        local_x = self.x_amplitude * np.sin(phase)
        local_y = self.y_amplitude * np.sin(2.0 * phase)
        x = (
            self.center_x_m
            + self._pattern_cos * local_x
            - self._pattern_sin * local_y
        )
        y = (
            self.center_y_m
            + self._pattern_sin * local_x
            + self._pattern_cos * local_y
        )
        return x, y

    def _positions_from_phases(self, phases: np.ndarray) -> np.ndarray:
        phase = np.asarray(phases, dtype=np.float64)
        x, y = self._horizontal_from_phases(phase)
        seabed = np.asarray(self.world.navigation_height(x, y))
        z = seabed + self.altitude_m + 0.45 * np.sin(3.0 * phase)
        return np.stack((x, y, z), axis=-1).astype(np.float64, copy=False)

    def _validate_measured_route_support(self) -> None:
        # At most one quarter of a source-grid interval separates validation
        # samples.  A route therefore cannot step over a one-cell NoData gap.
        maximum_xy_derivative = float(
            np.hypot(self.x_amplitude, 2.0 * self.y_amplitude)
        )
        segment_count = max(
            2_048,
            int(
                np.ceil(
                    8.0
                    * np.pi
                    * maximum_xy_derivative
                    / self.world.resolution_m
                )
            ),
        )
        phases = np.linspace(
            0.0, 2.0 * np.pi, segment_count, endpoint=False
        )
        x, y = self._horizontal_from_phases(phases)
        navigation_height = self.world.navigation_height(x, y)
        if not np.all(np.isfinite(navigation_height)):
            raise ValueError(
                "survey route crosses bathymetry NoData or lacks finite "
                "navigation-height support"
            )

    def _phase(self, timestamp: float) -> float:
        finite_timestamp = self._finite_scalar("timestamp", timestamp)
        wrapped_time = finite_timestamp % self.period
        if self._arc_lengths is None or self._arc_phases is None:
            return self.omega * wrapped_time
        distance = wrapped_time * float(self.speed_mps)
        return float(np.interp(distance, self._arc_lengths, self._arc_phases))

    def _position_from_phase(self, phase: float) -> np.ndarray:
        finite_phase = self._finite_scalar("phase", phase)
        return self._positions_from_phases(np.asarray(finite_phase))

    def _position(self, timestamp: float) -> np.ndarray:
        return self._position_from_phase(self._phase(timestamp))

    def kinematics(
        self, timestamp: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The navigation surface is already low-pass filtered. A centred
        # derivative over 0.12 s further avoids feeding individual DEM-cell
        # edges into the simulated accelerometer.
        step = 0.12
        position = self._position(timestamp)
        before = self._position(timestamp - step)
        after = self._position(timestamp + step)
        velocity = (after - before) / (2.0 * step)
        acceleration = (after - 2.0 * position + before) / (step * step)
        return position, velocity, acceleration

    def rotation(self, timestamp: float) -> np.ndarray:
        _, velocity, _ = self.kinematics(timestamp)
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        yaw = float(np.arctan2(velocity[1], velocity[0]))
        pitch = float(-np.arctan2(velocity[2], max(horizontal_speed, 1.0e-6)))
        roll = float(0.025 * np.sin(2.0 * self._phase(timestamp)))
        return rotation_from_rpy(roll, pitch, yaw)

    def state(self, timestamp: float) -> GroundTruthState:
        position, velocity, acceleration = self.kinematics(timestamp)
        rotation = self.rotation(timestamp)
        epsilon = 1.0e-3
        rotation_before = self.rotation(timestamp - epsilon)
        rotation_after = self.rotation(timestamp + epsilon)
        angular_velocity = so3_log(rotation_before.T @ rotation_after) / (
            2.0 * epsilon
        )
        gravity = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
        specific_force = rotation.T @ (acceleration - gravity)
        return GroundTruthState(
            timestamp=float(timestamp),
            rotation=rotation,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            angular_velocity_body=angular_velocity,
            specific_force_body=specific_force,
        )

    def validate_route(
        self,
        *,
        sample_count: int = 8_192,
        minimum_clearance_m: float = 0.0,
        require_measured_backscatter: bool = False,
    ) -> TerrainFollowingSurveyReport:
        """Inspect one complete lap and enforce optional coverage clearance.

        The method is read-only: it neither advances survey time nor caches a
        distance field.  ``minimum_clearance_m`` applies to bathymetry and,
        when requested, measured backscatter.  Modelled intensity fallback is
        intentionally not counted as measured acoustic coverage.
        """

        try:
            count = int(sample_count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "sample_count must be an integer of at least 32"
            ) from exc
        if (
            isinstance(sample_count, (bool, np.bool_))
            or count != sample_count
            or count < 32
        ):
            raise ValueError("sample_count must be an integer of at least 32")
        clearance_required = self._finite_scalar(
            "minimum_clearance_m", minimum_clearance_m
        )
        if clearance_required < 0.0:
            raise ValueError("minimum_clearance_m must be non-negative")
        if not isinstance(require_measured_backscatter, (bool, np.bool_)):
            raise ValueError("require_measured_backscatter must be boolean")

        phases = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        positions = self._positions_from_phases(phases)
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        seabed = np.asarray(self.world.seabed_height(x, y))
        bathymetry_valid = bool(
            np.all(np.isfinite(positions)) and np.all(np.isfinite(seabed))
        )

        next_positions = np.roll(positions, -1, axis=0)
        previous_positions = np.roll(positions, 1, axis=0)
        path_length = float(
            np.sum(np.linalg.norm(next_positions - positions, axis=1))
        )
        tangent = next_positions - previous_positions
        horizontal_tangent = np.linalg.norm(tangent[:, :2], axis=1)
        pitch_deg = np.rad2deg(
            -np.arctan2(tangent[:, 2], np.maximum(horizontal_tangent, 1.0e-12))
        )
        altitude = z - seabed

        bathymetry_clearance = _minimum_mask_clearance_m(
            self.world, self.world.valid_mask, x, y
        )
        backscatter_valid: bool | None = None
        backscatter_clearance: float | None = None
        if self.world.backscatter_valid_mask is not None:
            measured = self.world.measured_reflectivity(x, y)
            backscatter_valid = bool(np.all(np.isfinite(measured)))
            backscatter_clearance = _minimum_mask_clearance_m(
                self.world, self.world.backscatter_valid_mask, x, y
            )

        report = TerrainFollowingSurveyReport(
            sample_count=count,
            period_s=float(self.period),
            path_length_m=path_length,
            x_range_m=(float(np.min(x)), float(np.max(x))),
            y_range_m=(float(np.min(y)), float(np.max(y))),
            auv_depth_range_m=(float(np.min(-z)), float(np.max(-z))),
            seabed_depth_range_m=(
                float(np.min(-seabed)),
                float(np.max(-seabed)),
            ),
            altitude_range_m=(
                float(np.min(altitude)),
                float(np.max(altitude)),
            ),
            pitch_range_deg=(
                float(np.min(pitch_deg)),
                float(np.max(pitch_deg)),
            ),
            bathymetry_fully_valid=bathymetry_valid,
            backscatter_fully_valid=backscatter_valid,
            bathymetry_min_clearance_m=bathymetry_clearance,
            backscatter_min_clearance_m=backscatter_clearance,
        )

        if not report.bathymetry_fully_valid:
            raise ValueError("survey route is not fully covered by bathymetry")
        if report.bathymetry_min_clearance_m < clearance_required:
            raise ValueError(
                "survey bathymetry clearance is below "
                f"{clearance_required:.3f} m"
            )
        if require_measured_backscatter:
            if report.backscatter_fully_valid is None:
                raise ValueError("survey map has no measured backscatter")
            if not report.backscatter_fully_valid:
                raise ValueError(
                    "survey route is not fully covered by measured backscatter"
                )
            if (
                report.backscatter_min_clearance_m is None
                or report.backscatter_min_clearance_m < clearance_required
            ):
                raise ValueError(
                    "survey measured-backscatter clearance is below "
                    f"{clearance_required:.3f} m"
                )
        return report


class WaypointTerrainFollowingSurvey:
    """Smooth, closed terrain-following survey through planar waypoints.

    The supplied waypoints form a closed control polygon; a repeated closing
    copy of the first waypoint is accepted but not required.  Periodic
    Chaikin corner cutting is applied before a periodic Catmull--Rom curve is
    evaluated.  The latter is continuously differentiable at every refined
    waypoint, including the closing seam.  Vehicle time is parameterised by
    a dense three-dimensional arc-length table, so ``speed_mps`` describes
    speed over the terrain-following trajectory rather than merely horizontal
    speed.

    The dense table also bounds the planar distance between validation
    samples to one quarter of a source-grid interval.  This prevents a long
    waypoint segment from stepping over a native NoData cell during route
    validation.
    """

    def __init__(
        self,
        world: MeasuredBathymetryGrid,
        waypoints_xy_m: np.ndarray,
        *,
        speed_mps: float = 1.5,
        altitude_m: float = 28.0,
        smoothing_iterations: int = 4,
        maximum_pitch_deg: float | None = None,
    ) -> None:
        self.world = world
        self.speed_mps = self._positive_scalar("speed_mps", speed_mps)
        self.altitude_m = self._positive_scalar("altitude_m", altitude_m)
        self.smoothing_iterations = self._smoothing_iteration_count(
            smoothing_iterations
        )
        self.maximum_pitch_deg = self._maximum_pitch(maximum_pitch_deg)

        waypoints = self._validated_waypoints(waypoints_xy_m)
        waypoints.setflags(write=False)
        self.waypoints_xy_m = waypoints
        controls = self._chaikin_closed(
            waypoints, self.smoothing_iterations
        )
        controls.setflags(write=False)
        self.smoothed_waypoints_xy_m = controls
        self._control_points = controls

        # The derivative bound used by _dense_parameters makes this an upper
        # bound on planar arc length between adjacent samples, not just on the
        # straight chord joining them.
        self._validation_max_spacing_bound_m = (
            0.25 * self.world.resolution_m
        )
        parameters = self._dense_parameters(
            self._validation_max_spacing_bound_m
        )
        horizontal = self._horizontal_from_parameters(parameters)
        self._validate_grid_bounds(horizontal)
        target_height = self._target_height(horizontal)
        if not np.all(np.isfinite(target_height)):
            raise ValueError(
                "waypoint survey route crosses bathymetry NoData or lacks "
                "finite navigation-height support"
            )

        closing_horizontal = np.vstack((horizontal, horizontal[:1]))
        horizontal_segment_lengths = np.linalg.norm(
            np.diff(closing_horizontal, axis=0), axis=1
        )
        horizontal_cumulative = np.concatenate(
            ([0.0], np.cumsum(horizontal_segment_lengths))
        )
        horizontal_path_length = float(horizontal_cumulative[-1])
        if (
            not np.isfinite(horizontal_path_length)
            or horizontal_path_length <= 0.0
        ):
            raise ValueError(
                "waypoint survey horizontal path length must be positive"
            )
        self._horizontal_arc_parameters = np.concatenate(
            (parameters, [float(self._control_points.shape[0])])
        )
        self._horizontal_arc_lengths = horizontal_cumulative
        self.horizontal_path_length_m = horizontal_path_length

        self.pitch_envelope_guard_m = 0.0
        self.pitch_slope_safety_factor = 1.0
        self._pitch_envelope_height_m: np.ndarray | None = None
        if self.maximum_pitch_deg is not None:
            self.pitch_slope_safety_factor = (
                self._horizontal_speed_safety_factor(
                    parameters, horizontal_segment_lengths
                )
            )
            maximum_slope = float(
                np.tan(np.deg2rad(self.maximum_pitch_deg))
                * self.pitch_slope_safety_factor
            )
            navigation_gradient_bound = (
                self._navigation_gradient_bound(horizontal)
            )
            # Every point of a bounded validation interval is no farther than
            # half that interval from an endpoint.  Raising the sampled target
            # by this amount protects the continuous bilinear navigation
            # surface between samples.  It is intentionally conservative; the
            # subsequent cyclic envelope is pointwise minimal for this guarded
            # target.
            self.pitch_envelope_guard_m = float(
                0.5
                * self._validation_max_spacing_bound_m
                * navigation_gradient_bound
            )
            guarded_target = target_height + self.pitch_envelope_guard_m
            envelope = self._periodic_lipschitz_upper_envelope(
                guarded_target,
                horizontal_cumulative[:-1],
                horizontal_path_length,
                maximum_slope,
            )
            self._pitch_envelope_height_m = np.concatenate(
                (envelope, envelope[:1])
            )
            route_height = envelope
        else:
            route_height = target_height

        positions = np.concatenate(
            (horizontal, route_height[:, None]), axis=1
        )
        closing_positions = np.vstack((positions, positions[:1]))
        segment_lengths = np.linalg.norm(
            np.diff(closing_positions, axis=0), axis=1
        )
        scale = max(
            float(np.max(np.abs(positions))),
            self.world.resolution_m,
            1.0,
        )
        if np.any(segment_lengths <= np.finfo(np.float64).eps * scale):
            raise ValueError(
                "waypoint survey curve contains a zero-length arc segment"
            )
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        path_length = float(cumulative[-1])
        if not np.isfinite(path_length) or path_length <= 0.0:
            raise ValueError("waypoint survey path length must be positive")

        self._arc_parameters = np.concatenate(
            (parameters, [float(self._control_points.shape[0])])
        )
        self._arc_lengths = cumulative
        self._validation_parameters = parameters
        self._validation_positions = positions
        self.path_length_m = path_length
        self.period = path_length / self.speed_mps
        self.omega = 2.0 * np.pi / self.period

    @staticmethod
    def _finite_scalar(name: str, value: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _positive_scalar(cls, name: str, value: float) -> float:
        result = cls._finite_scalar(name, value)
        if result <= 0.0:
            raise ValueError(f"{name} must be positive")
        return result

    @staticmethod
    def _smoothing_iteration_count(value: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "smoothing_iterations must be an integer from 0 to 12"
            ) from exc
        if (
            isinstance(value, (bool, np.bool_))
            or result != value
            or not 0 <= result <= 12
        ):
            raise ValueError(
                "smoothing_iterations must be an integer from 0 to 12"
            )
        return result

    @classmethod
    def _maximum_pitch(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "maximum_pitch_deg must be in the half-open interval [0, 90)"
            )
        result = cls._finite_scalar("maximum_pitch_deg", value)
        if not 0.0 <= result < 90.0:
            raise ValueError(
                "maximum_pitch_deg must be in the half-open interval [0, 90)"
            )
        return result

    @staticmethod
    def _validated_waypoints(waypoints_xy_m: np.ndarray) -> np.ndarray:
        waypoints = np.asarray(waypoints_xy_m)
        if waypoints.dtype.kind == "c":
            raise ValueError("waypoints_xy_m must be real-valued")
        try:
            waypoints = np.asarray(waypoints, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "waypoints_xy_m must have shape (N, 2)"
            ) from exc
        if waypoints.ndim != 2 or waypoints.shape[1:] != (2,):
            raise ValueError("waypoints_xy_m must have shape (N, 2)")
        if not np.all(np.isfinite(waypoints)):
            raise ValueError("waypoints_xy_m must be finite")
        if waypoints.shape[0] < 3:
            raise ValueError(
                "a closed waypoint survey requires at least three waypoints"
            )

        result = np.array(waypoints, dtype=np.float64, copy=True)
        coordinate_scale = max(float(np.max(np.abs(result))), 1.0)
        tolerance = 32.0 * np.finfo(np.float64).eps * coordinate_scale
        if np.linalg.norm(result[-1] - result[0]) <= tolerance:
            result = result[:-1]
        if result.shape[0] < 3:
            raise ValueError(
                "a closed waypoint survey requires at least three distinct "
                "waypoints"
            )

        cyclic_steps = np.linalg.norm(
            np.roll(result, -1, axis=0) - result, axis=1
        )
        if np.any(cyclic_steps <= tolerance):
            raise ValueError(
                "consecutive closed-route waypoints must be distinct"
            )
        return result

    @staticmethod
    def _chaikin_closed(
        waypoints: np.ndarray, iterations: int
    ) -> np.ndarray:
        points = np.array(waypoints, dtype=np.float64, copy=True)
        for _ in range(iterations):
            following = np.roll(points, -1, axis=0)
            refined = np.empty((2 * points.shape[0], 2), dtype=np.float64)
            refined[0::2] = 0.75 * points + 0.25 * following
            refined[1::2] = 0.25 * points + 0.75 * following
            points = refined
        return points

    def _curve_coefficients(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = self._control_points
        count = points.shape[0]
        index = np.asarray(indices, dtype=np.int64)
        p0 = points[(index - 1) % count]
        p1 = points[index % count]
        p2 = points[(index + 1) % count]
        p3 = points[(index + 2) % count]
        linear = -p0 + p2
        quadratic = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
        cubic = -p0 + 3.0 * p1 - 3.0 * p2 + p3
        return p1, linear, quadratic, cubic

    def _horizontal_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        parameter = np.asarray(parameters, dtype=np.float64)
        count = self._control_points.shape[0]
        wrapped = np.mod(parameter, float(count))
        indices = np.floor(wrapped).astype(np.int64)
        fraction = wrapped - indices
        p1, linear, quadratic, cubic = self._curve_coefficients(indices)
        t = fraction[..., None]
        return 0.5 * (
            2.0 * p1
            + linear * t
            + quadratic * t * t
            + cubic * t * t * t
        )

    def _horizontal_derivatives(
        self, parameters: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        parameter = np.asarray(parameters, dtype=np.float64)
        count = self._control_points.shape[0]
        wrapped = np.mod(parameter, float(count))
        indices = np.floor(wrapped).astype(np.int64)
        fraction = wrapped - indices
        _, linear, quadratic, cubic = self._curve_coefficients(indices)
        t = fraction[..., None]
        first = 0.5 * (
            linear + 2.0 * quadratic * t + 3.0 * cubic * t * t
        )
        second = quadratic + 3.0 * cubic * t
        return first, second

    def _dense_parameters(self, maximum_spacing_m: float) -> np.ndarray:
        """Sample every spline span using a conservative derivative bound."""

        count = self._control_points.shape[0]
        indices = np.arange(count, dtype=np.int64)
        _, linear, quadratic, cubic = self._curve_coefficients(indices)
        # For t in [0, 1], ||p'(t)|| is no larger than this coefficient
        # triangle-inequality bound.  Dividing it into this many equal pieces
        # therefore bounds planar arc length between validation samples.
        derivative_bound = 0.5 * (
            np.linalg.norm(linear, axis=1)
            + 2.0 * np.linalg.norm(quadratic, axis=1)
            + 3.0 * np.linalg.norm(cubic, axis=1)
        )
        subdivisions = np.maximum(
            16, np.ceil(derivative_bound / maximum_spacing_m).astype(int)
        )
        pieces = [
            index
            + np.arange(parts, dtype=np.float64) / float(parts)
            for index, parts in enumerate(subdivisions)
        ]
        return np.concatenate(pieces)

    def _positions_from_parameters(
        self, parameters: np.ndarray
    ) -> np.ndarray:
        horizontal = self._horizontal_from_parameters(parameters)
        z = self._height_from_parameters(parameters, horizontal)
        return np.concatenate((horizontal, z[..., None]), axis=-1)

    def _target_height(self, horizontal: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.world.navigation_height(
                horizontal[..., 0], horizontal[..., 1]
            ),
            dtype=np.float64,
        ) + self.altitude_m

    def _height_from_parameters(
        self, parameters: np.ndarray, horizontal: np.ndarray
    ) -> np.ndarray:
        if self._pitch_envelope_height_m is None:
            return self._target_height(horizontal)
        parameter = np.asarray(parameters, dtype=np.float64)
        wrapped = np.mod(parameter, float(self._control_points.shape[0]))
        horizontal_distance = np.interp(
            wrapped,
            self._horizontal_arc_parameters,
            self._horizontal_arc_lengths,
        )
        return np.interp(
            horizontal_distance,
            self._horizontal_arc_lengths,
            self._pitch_envelope_height_m,
        )

    def _navigation_gradient_bound(self, horizontal: np.ndarray) -> float:
        """Bound the bilinear navigation-surface gradient near the route."""

        rows, columns = self.world.navigation_elevation_m.shape
        resolution = self.world.resolution_m
        column = np.floor(
            (horizontal[:, 0] + 0.5 * self.world.width) / resolution
        ).astype(np.int64)
        row = np.floor(
            (horizontal[:, 1] + 0.5 * self.world.length) / resolution
        ).astype(np.int64)
        column = np.clip(column, 0, columns - 2)
        row = np.clip(row, 0, rows - 2)

        # A sample interval is at most one quarter cell long.  Including the
        # eight neighbouring cells therefore covers every cell the continuous
        # curve can enter between two samples, even when it clips a corner.
        nearby_flat = []
        for row_offset in (-1, 0, 1):
            nearby_row = np.clip(row + row_offset, 0, rows - 2)
            for column_offset in (-1, 0, 1):
                nearby_column = np.clip(
                    column + column_offset, 0, columns - 2
                )
                nearby_flat.append(
                    nearby_row * (columns - 1) + nearby_column
                )
        cells = np.unique(np.concatenate(nearby_flat))
        cell_row, cell_column = np.divmod(cells, columns - 1)
        navigation = self.world.navigation_elevation_m
        valid = self.world.valid_mask
        z00 = np.asarray(navigation[cell_row, cell_column])
        z01 = np.asarray(navigation[cell_row, cell_column + 1])
        z10 = np.asarray(navigation[cell_row + 1, cell_column])
        z11 = np.asarray(navigation[cell_row + 1, cell_column + 1])
        cells_valid = (
            valid[cell_row, cell_column]
            & valid[cell_row, cell_column + 1]
            & valid[cell_row + 1, cell_column]
            & valid[cell_row + 1, cell_column + 1]
            & np.isfinite(z00)
            & np.isfinite(z01)
            & np.isfinite(z10)
            & np.isfinite(z11)
        )
        if not np.any(cells_valid):
            raise ValueError(
                "waypoint survey lacks finite navigation-height support"
            )
        z00 = z00[cells_valid]
        z01 = z01[cells_valid]
        z10 = z10[cells_valid]
        z11 = z11[cells_valid]
        dx0 = (z01 - z00) / resolution
        dx1 = (z11 - z10) / resolution
        dy0 = (z10 - z00) / resolution
        dy1 = (z11 - z01) / resolution
        squared_gradient_bound = np.maximum.reduce(
            (
                dx0 * dx0 + dy0 * dy0,
                dx0 * dx0 + dy1 * dy1,
                dx1 * dx1 + dy0 * dy0,
                dx1 * dx1 + dy1 * dy1,
            )
        )
        return float(np.sqrt(np.max(squared_gradient_bound, initial=0.0)))

    def _horizontal_speed_safety_factor(
        self,
        parameters: np.ndarray,
        chord_lengths_m: np.ndarray,
    ) -> float:
        """Relate piecewise chord distance to true local spline distance.

        Height is interpolated against the dense chord-length table.  A cubic
        spline's horizontal speed varies slightly inside each such interval,
        so merely limiting height change per chord can exceed the requested
        local pitch by a small amount.  The affine second derivative supplies
        a rigorous lower bound on speed in every interval.  Scaling by the
        worst ratio preserves the user-facing slope limit continuously.
        """

        count = self._control_points.shape[0]
        parameter = np.asarray(parameters, dtype=np.float64)
        following = np.concatenate((parameter[1:], [float(count)]))
        interval = following - parameter
        midpoint = 0.5 * (parameter + following)
        first_midpoint, _ = self._horizontal_derivatives(midpoint)
        _, second_start = self._horizontal_derivatives(parameter)
        # The integer endpoint belongs to the following Catmull--Rom span;
        # nextafter samples the left limit of the current span instead.
        left_endpoint = np.nextafter(following, -np.inf)
        _, second_end = self._horizontal_derivatives(left_endpoint)
        second_bound = np.maximum(
            np.linalg.norm(second_start, axis=1),
            np.linalg.norm(second_end, axis=1),
        )
        local_speed_lower_bound = (
            np.linalg.norm(first_midpoint, axis=1)
            - 0.5 * interval * second_bound
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (
                local_speed_lower_bound * interval / chord_lengths_m
            )
        if not np.all(np.isfinite(ratio)) or np.min(ratio) <= 0.0:
            raise ValueError(
                "waypoint survey has a stationary cusp incompatible with "
                "a continuous pitch-limited constant-speed trajectory"
            )
        rounding_margin = 64.0 * np.finfo(np.float64).eps
        return float(
            max(
                0.0,
                min(1.0, float(np.min(ratio))) - rounding_margin,
            )
        )

    @staticmethod
    def _periodic_lipschitz_upper_envelope(
        target_height: np.ndarray,
        horizontal_distance_m: np.ndarray,
        period_m: float,
        maximum_slope: float,
    ) -> np.ndarray:
        """Return the pointwise-minimal cyclic Lipschitz majorant.

        On a cycle with geodesic distance ``d``, the result is
        ``max_j(target[j] - maximum_slope * d(i, j))``.  Three unwrapped
        copies reduce the cyclic problem to forward and backward cumulative
        maxima, retaining linear time and memory complexity.
        """

        target = np.asarray(target_height, dtype=np.float64).reshape(-1)
        distance = np.asarray(
            horizontal_distance_m, dtype=np.float64
        ).reshape(-1)
        if target.size == 0 or target.shape != distance.shape:
            raise ValueError("pitch-envelope samples must be non-empty and paired")
        extended_distance = np.concatenate(
            (distance - period_m, distance, distance + period_m)
        )
        extended_target = np.tile(target, 3)
        forward = (
            np.maximum.accumulate(
                extended_target + maximum_slope * extended_distance
            )
            - maximum_slope * extended_distance
        )
        backward = (
            np.maximum.accumulate(
                (
                    extended_target
                    - maximum_slope * extended_distance
                )[::-1]
            )[::-1]
            + maximum_slope * extended_distance
        )
        count = target.size
        envelope = np.maximum(forward[count : 2 * count], backward[count : 2 * count])
        # Suppress sub-ulp erosion of the majorant without a material offset.
        envelope = np.maximum(envelope, target)
        return envelope

    def _validate_grid_bounds(self, horizontal: np.ndarray) -> None:
        x = horizontal[:, 0]
        y = horizontal[:, 1]
        if (
            np.min(x) <= -0.5 * self.world.width
            or np.max(x) >= 0.5 * self.world.width
        ):
            raise ValueError(
                "waypoint survey route must stay strictly inside the "
                "measured grid x bounds"
            )
        if (
            np.min(y) <= -0.5 * self.world.length
            or np.max(y) >= 0.5 * self.world.length
        ):
            raise ValueError(
                "waypoint survey route must stay strictly inside the "
                "measured grid y bounds"
            )

    def _parameter(self, timestamp: float) -> float:
        finite_timestamp = self._finite_scalar("timestamp", timestamp)
        distance = (finite_timestamp % self.period) * self.speed_mps
        return float(
            np.interp(distance, self._arc_lengths, self._arc_parameters)
        )

    def _position(self, timestamp: float) -> np.ndarray:
        return self._positions_from_parameters(
            np.asarray(self._parameter(timestamp))
        )

    def _positions_at_equal_arc_intervals(self, count: int) -> np.ndarray:
        distances = np.linspace(
            0.0, self.path_length_m, count, endpoint=False
        )
        parameters = np.interp(
            distances, self._arc_lengths, self._arc_parameters
        )
        return self._positions_from_parameters(parameters)

    def kinematics(
        self, timestamp: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # A centred periodic derivative gives identical position and velocity
        # at t and t + period, including directly across the closing seam.
        step = max(min(0.12, self.period * 1.0e-3), 1.0e-6)
        position = self._position(timestamp)
        before = self._position(timestamp - step)
        after = self._position(timestamp + step)
        velocity = (after - before) / (2.0 * step)
        acceleration = (after - 2.0 * position + before) / (step * step)
        return position, velocity, acceleration

    def rotation(self, timestamp: float) -> np.ndarray:
        _, velocity, _ = self.kinematics(timestamp)
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        yaw = float(np.arctan2(velocity[1], velocity[0]))
        pitch = float(
            -np.arctan2(velocity[2], max(horizontal_speed, 1.0e-12))
        )
        return rotation_from_rpy(0.0, pitch, yaw)

    def state(self, timestamp: float) -> GroundTruthState:
        position, velocity, acceleration = self.kinematics(timestamp)
        rotation = self.rotation(timestamp)
        epsilon = max(min(1.0e-3, self.period * 1.0e-5), 1.0e-8)
        rotation_before = self.rotation(timestamp - epsilon)
        rotation_after = self.rotation(timestamp + epsilon)
        angular_velocity = so3_log(rotation_before.T @ rotation_after) / (
            2.0 * epsilon
        )
        gravity = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
        specific_force = rotation.T @ (acceleration - gravity)
        return GroundTruthState(
            timestamp=float(timestamp),
            rotation=rotation,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            angular_velocity_body=angular_velocity,
            specific_force_body=specific_force,
        )

    @staticmethod
    def _sample_count(value: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "sample_count must be an integer of at least 32"
            ) from exc
        if (
            isinstance(value, (bool, np.bool_))
            or result != value
            or result < 32
        ):
            raise ValueError("sample_count must be an integer of at least 32")
        return result

    def _maximum_horizontal_curvature(self) -> float | None:
        first, second = self._horizontal_derivatives(
            self._validation_parameters
        )
        speed_squared = np.sum(first * first, axis=1)
        numerator = np.abs(
            first[:, 0] * second[:, 1]
            - first[:, 1] * second[:, 0]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            curvature = numerator / np.power(speed_squared, 1.5)
        finite = curvature[np.isfinite(curvature)]
        if finite.size != curvature.size:
            # A cusp (normally an exact A-B-A reversal in the input route)
            # has no finite curvature.  ``None`` remains standards-compliant
            # JSON when the frozen report is serialized with ``asdict``.
            return None
        return float(np.max(finite, initial=0.0))

    def validate_route(
        self,
        *,
        sample_count: int = 8_192,
        minimum_clearance_m: float = 0.0,
        require_measured_backscatter: bool = False,
    ) -> TerrainFollowingSurveyReport:
        """Inspect a lap and enforce continuous-route measured coverage."""

        count = self._sample_count(sample_count)
        clearance_required = self._finite_scalar(
            "minimum_clearance_m", minimum_clearance_m
        )
        if clearance_required < 0.0:
            raise ValueError("minimum_clearance_m must be non-negative")
        if not isinstance(require_measured_backscatter, (bool, np.bool_)):
            raise ValueError("require_measured_backscatter must be boolean")

        validation_positions = self._validation_positions
        validation_x = validation_positions[:, 0]
        validation_y = validation_positions[:, 1]
        validation_z = validation_positions[:, 2]
        validation_seabed = np.asarray(
            self.world.seabed_height(validation_x, validation_y)
        )
        bathymetry_valid = bool(
            np.all(np.isfinite(validation_positions))
            and np.all(np.isfinite(validation_seabed))
        )

        # Pitch is a local differential quantity.  Compute it on the dense
        # validation trajectory so a deliberately small report sample_count
        # cannot turn a curved multi-kilometre chord into a fictitious slope.
        tangent = np.roll(validation_positions, -1, axis=0) - np.roll(
            validation_positions, 1, axis=0
        )
        horizontal_tangent = np.linalg.norm(tangent[:, :2], axis=1)
        pitch_deg = np.rad2deg(
            -np.arctan2(
                tangent[:, 2], np.maximum(horizontal_tangent, 1.0e-12)
            )
        )
        altitude = validation_z - validation_seabed

        # Distance to a closed set is 1-Lipschitz.  Half one bounded sample
        # interval accounts conservatively for the unsampled curve between
        # adjacent dense validation points.
        between_sample_allowance = (
            0.5 * self._validation_max_spacing_bound_m
        )
        bathymetry_clearance = max(
            0.0,
            _minimum_mask_clearance_m(
                self.world,
                self.world.valid_mask,
                validation_x,
                validation_y,
            )
            - between_sample_allowance,
        )
        backscatter_valid: bool | None = None
        backscatter_clearance: float | None = None
        if self.world.backscatter_valid_mask is not None:
            measured = self.world.measured_reflectivity(
                validation_x, validation_y
            )
            backscatter_valid = bool(np.all(np.isfinite(measured)))
            backscatter_clearance = max(
                0.0,
                _minimum_mask_clearance_m(
                    self.world,
                    self.world.backscatter_valid_mask,
                    validation_x,
                    validation_y,
                )
                - between_sample_allowance,
            )

        report = TerrainFollowingSurveyReport(
            sample_count=count,
            period_s=float(self.period),
            path_length_m=float(self.path_length_m),
            x_range_m=(
                float(np.min(validation_x)),
                float(np.max(validation_x)),
            ),
            y_range_m=(
                float(np.min(validation_y)),
                float(np.max(validation_y)),
            ),
            auv_depth_range_m=(
                float(np.min(-validation_z)),
                float(np.max(-validation_z)),
            ),
            seabed_depth_range_m=(
                float(np.min(-validation_seabed)),
                float(np.max(-validation_seabed)),
            ),
            altitude_range_m=(
                float(np.min(altitude)),
                float(np.max(altitude)),
            ),
            pitch_range_deg=(
                float(np.min(pitch_deg)),
                float(np.max(pitch_deg)),
            ),
            bathymetry_fully_valid=bathymetry_valid,
            backscatter_fully_valid=backscatter_valid,
            bathymetry_min_clearance_m=bathymetry_clearance,
            backscatter_min_clearance_m=backscatter_clearance,
            maximum_horizontal_curvature_per_m=(
                self._maximum_horizontal_curvature()
            ),
            validation_sample_count=int(
                self._validation_parameters.size
            ),
            validation_max_spacing_m=float(
                self._validation_max_spacing_bound_m
            ),
            waypoint_count=int(self.waypoints_xy_m.shape[0]),
        )

        if not report.bathymetry_fully_valid:
            raise ValueError("survey route is not fully covered by bathymetry")
        if report.bathymetry_min_clearance_m < clearance_required:
            raise ValueError(
                "survey bathymetry clearance is below "
                f"{clearance_required:.3f} m"
            )
        if require_measured_backscatter:
            if report.backscatter_fully_valid is None:
                raise ValueError("survey map has no measured backscatter")
            if not report.backscatter_fully_valid:
                raise ValueError(
                    "survey route is not fully covered by measured backscatter"
                )
            if (
                report.backscatter_min_clearance_m is None
                or report.backscatter_min_clearance_m < clearance_required
            ):
                raise ValueError(
                    "survey measured-backscatter clearance is below "
                    f"{clearance_required:.3f} m"
                )
        return report
