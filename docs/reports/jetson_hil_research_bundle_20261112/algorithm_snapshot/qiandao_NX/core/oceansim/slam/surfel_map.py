"""Incremental voxel-surfel map for acoustic scan-to-map registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np


@dataclass(frozen=True, slots=True)
class Surfel:
    """Locally planar acoustic map element."""

    centroid: np.ndarray
    normal: np.ndarray
    covariance: np.ndarray
    normal_variance: float
    planarity: float
    count: int
    intensity: float
    voxel_key: tuple[int, int, int]


@dataclass(frozen=True)
class SurfelAssociationBatch:
    """Nearest valid surfel selected for each world-frame query point.

    Arrays retain the input point order.  Values at invalid rows are zero and
    must only be consumed where ``valid_mask`` is true.  This compact common
    representation lets the estimator use either the reference Python voxel
    lookup or an accelerated data-association backend without changing the
    point-to-plane observation model.
    """

    valid_mask: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    normal_variances: np.ndarray


@dataclass(frozen=True)
class SurfelMapSnapshot:
    """Sorted immutable view of the mature surfels at one map revision."""

    revision: int
    voxel_keys: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    normal_variances: np.ndarray


@dataclass(slots=True)
class _VoxelStats:
    reference: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    count: int = 0
    mean: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    scatter: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )
    offset_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    offset_outer_sum: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )
    intensity_sum: float = 0.0
    version: int = 0
    cached_version: int = -1
    cached_surfel: Surfel | None = None

    def add(self, point: np.ndarray, intensity: float) -> None:
        self.merge(
            1,
            np.asarray(point, dtype=np.float64),
            np.zeros((3, 3)),
            intensity,
        )

    def _refresh_centered_statistics(self) -> None:
        """Derive Welford-style statistics from local additive moments."""

        if self.count <= 0:
            self.count = 0
            self.mean.fill(0.0)
            self.scatter.fill(0.0)
            self.offset_sum.fill(0.0)
            self.offset_outer_sum.fill(0.0)
            return
        self.mean = self.reference + self.offset_sum / float(self.count)
        if self.count == 1:
            # Avoid subtracting two nearly identical second moments for the
            # common single-return boundary case.
            self.scatter.fill(0.0)
            return
        centered = self.offset_outer_sum - np.outer(
            self.offset_sum, self.offset_sum
        ) / float(self.count)
        centered = 0.5 * (centered + centered.T)
        if not np.isfinite(centered).all():
            raise ValueError("voxel additive moments are not finite")
        # Do not eigendecompose every touched voxel at the sonar rate.  The
        # surfel cache performs the PSD-safe decomposition only when a voxel
        # is queried; here the locally referenced raw moments are already the
        # authoritative statistics.
        self.scatter = centered

    def merge(
        self,
        count: int,
        mean: np.ndarray,
        scatter: np.ndarray,
        intensity_sum: float,
    ) -> None:
        """Merge one voxel batch without changing its sufficient statistics."""

        batch_count = int(count)
        if batch_count <= 0:
            return
        batch_mean = np.asarray(mean, dtype=np.float64).reshape(3)
        batch_scatter = np.asarray(scatter, dtype=np.float64).reshape(3, 3)
        offset_mean = batch_mean - self.reference
        self.offset_sum += batch_count * offset_mean
        self.offset_outer_sum += batch_scatter + batch_count * np.outer(
            offset_mean, offset_mean
        )
        self.count += batch_count
        self._refresh_centered_statistics()
        self.intensity_sum += float(intensity_sum)
        self.version += 1

    def subtract(
        self,
        count: int,
        mean: np.ndarray,
        scatter: np.ndarray,
        intensity_sum: float,
    ) -> None:
        """Remove one previously merged batch from sufficient statistics.

        A fixed-lag local map must retire old scans without rebuilding every
        surviving voxel at the sonar rate.  This is the inverse of the
        parallel Chan/Welford merge above.  The caller must provide the exact
        points (and intensities) that were inserted for the retired batch.
        """

        batch_count = int(count)
        if batch_count <= 0:
            return
        if batch_count > self.count:
            raise ValueError("cannot remove more points than a voxel contains")
        if batch_count == self.count:
            self.count = 0
            self.mean.fill(0.0)
            self.scatter.fill(0.0)
            self.offset_sum.fill(0.0)
            self.offset_outer_sum.fill(0.0)
            self.intensity_sum = 0.0
            self.version += 1
            self.cached_version = -1
            self.cached_surfel = None
            return

        batch_mean = np.asarray(mean, dtype=np.float64).reshape(3)
        batch_scatter = np.asarray(scatter, dtype=np.float64).reshape(3, 3)
        offset_mean = batch_mean - self.reference
        self.offset_sum -= batch_count * offset_mean
        self.offset_outer_sum -= batch_scatter + batch_count * np.outer(
            offset_mean, offset_mean
        )
        self.count -= batch_count
        self._refresh_centered_statistics()
        self.intensity_sum -= float(intensity_sum)
        self.version += 1
        self.cached_version = -1
        self.cached_surfel = None


class SurfelVoxelMap:
    """Maintain acoustic surface statistics in an incremental voxel hash.

    Unlike a generic LiDAR map, the query exposes the surface-normal variance
    so the filter can respect the strongly anisotropic uncertainty of sonar
    measurements.  Only compact running statistics are retained, keeping a
    long survey bounded by the number of occupied voxels rather than pings.
    """

    _NEIGHBOURS = tuple(product((-1, 0, 1), repeat=3))
    _BLOCK_SIZE_VOXELS = 16

    def __init__(
        self,
        voxel_size: float = 1.5,
        *,
        min_points: int = 6,
        max_planarity: float = 0.12,
        min_tangent_ratio: float = 0.05,
        max_plane_distance: float = 1.2,
        max_centroid_distance: float | None = None,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.min_points = int(min_points)
        self.max_planarity = float(max_planarity)
        self.min_tangent_ratio = float(min_tangent_ratio)
        self.max_plane_distance = float(max_plane_distance)
        self.max_centroid_distance = float(
            2.6 * self.voxel_size
            if max_centroid_distance is None
            else max_centroid_distance
        )
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be > 0")
        if self.min_points < 3:
            raise ValueError("min_points must be >= 3")
        if not 0.0 < self.max_planarity < 1.0:
            raise ValueError("max_planarity must be in (0, 1)")
        if not 0.0 <= self.min_tangent_ratio <= 1.0:
            raise ValueError("min_tangent_ratio must be in [0, 1]")
        self._voxels: dict[tuple[int, int, int], _VoxelStats] = {}
        self._blocks: dict[
            tuple[int, int, int], set[tuple[int, int, int]]
        ] = {}
        self._point_count = 0
        self._mature_voxel_count = 0
        self._revision = 0
        self._association_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self._voxels)

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def revision(self) -> int:
        """Monotonic revision used to invalidate accelerator snapshots."""

        return self._revision

    @property
    def mature_voxel_count(self) -> int:
        return self._mature_voxel_count

    def clear(self) -> None:
        self._voxels.clear()
        self._blocks.clear()
        self._point_count = 0
        self._mature_voxel_count = 0
        self._association_bounds = None
        self._revision += 1

    def set_association_box(
        self, center_world: np.ndarray, half_extent_m: float | None
    ) -> None:
        """Restrict accelerator snapshots without deleting historical voxels.

        Scalar hash queries already inspect only 27 neighbouring cells.  A
        packed accelerator snapshot, however, used to copy every mature
        surfel on every map revision.  This active box keeps that snapshot
        local while the full probabilistic map remains available when the AUV
        revisits an older area.
        """

        if half_extent_m is None or float(half_extent_m) <= 0.0:
            bounds = None
        else:
            center = np.asarray(center_world, dtype=np.float64).reshape(3)
            half_extent = float(half_extent_m)
            if not np.isfinite(center).all() or not np.isfinite(half_extent):
                raise ValueError("association-box parameters must be finite")
            lower = np.floor(
                (center - half_extent) / self.voxel_size
            ).astype(np.int64)
            upper = np.floor(
                (center + half_extent) / self.voxel_size
            ).astype(np.int64)
            bounds = (lower, upper)

        unchanged = self._association_bounds is None and bounds is None
        if self._association_bounds is not None and bounds is not None:
            unchanged = bool(
                np.array_equal(self._association_bounds[0], bounds[0])
                and np.array_equal(self._association_bounds[1], bounds[1])
            )
        if unchanged:
            return

        # A cached ``Surfel`` carries three additional NumPy arrays.  Keeping
        # one for every historical voxel would negate much of the compact
        # running-statistics representation on a multi-hour survey.  Release
        # caches as cells leave the active association box; their sufficient
        # statistics remain intact and the exact surfel is rebuilt on revisit.
        if bounds is not None:
            candidates = (
                self._voxels.items()
                if self._association_bounds is None
                else self._association_items()
            )
            lower, upper = bounds
            for key, stats in candidates:
                if any(
                    key[axis] < lower[axis] or key[axis] > upper[axis]
                    for axis in range(3)
                ):
                    stats.cached_version = -1
                    stats.cached_surfel = None
        self._association_bounds = bounds
        self._revision += 1

    def _inside_association_box(self, key: tuple[int, int, int]) -> bool:
        if self._association_bounds is None:
            return True
        lower, upper = self._association_bounds
        return all(
            lower[axis] <= key[axis] <= upper[axis]
            for axis in range(3)
        )

    @classmethod
    def _block_key(
        cls, key: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        size = cls._BLOCK_SIZE_VOXELS
        return key[0] // size, key[1] // size, key[2] // size

    def _association_items(self):
        if self._association_bounds is None:
            yield from self._voxels.items()
            return
        lower, upper = self._association_bounds
        size = self._BLOCK_SIZE_VOXELS
        lower_block = lower // size
        upper_block = upper // size
        for block_x in range(int(lower_block[0]), int(upper_block[0]) + 1):
            for block_y in range(
                int(lower_block[1]), int(upper_block[1]) + 1
            ):
                for block_z in range(
                    int(lower_block[2]), int(upper_block[2]) + 1
                ):
                    keys = self._blocks.get((block_x, block_y, block_z))
                    if keys is None:
                        continue
                    for key in keys:
                        if self._inside_association_box(key):
                            yield key, self._voxels[key]

    @property
    def association_voxel_count(self) -> int:
        return sum(
            stats.count >= self.min_points
            for _, stats in self._association_items()
        )

    def remove_outside_box(
        self, center_world: np.ndarray, half_extent_m: float
    ) -> tuple[int, int]:
        """Delete voxels outside an axis-aligned local-map cube.

        FAST-LIVO2 keeps a sliding local map around the platform instead of
        rebuilding associations against every point ever observed.  The
        bounds are evaluated on voxel extents, so cells intersecting the box
        are retained and there is no boundary hole.

        Returns:
            ``(removed_voxels, removed_points)``.
        """

        center = np.asarray(center_world, dtype=np.float64).reshape(3)
        half_extent = float(half_extent_m)
        if not np.isfinite(center).all():
            raise ValueError("center_world must be finite")
        if not np.isfinite(half_extent) or half_extent <= 0.0:
            raise ValueError("half_extent_m must be finite and positive")
        lower = np.floor(
            (center - half_extent) / self.voxel_size
        ).astype(np.int64)
        upper = np.floor(
            (center + half_extent) / self.voxel_size
        ).astype(np.int64)
        removed_points = 0
        removed_keys: list[tuple[int, int, int]] = []
        for key, stats in self._voxels.items():
            if any(
                key[axis] < lower[axis] or key[axis] > upper[axis]
                for axis in range(3)
            ):
                removed_keys.append(key)
                removed_points += stats.count
        for key in removed_keys:
            stats = self._voxels[key]
            if stats.count >= self.min_points:
                self._mature_voxel_count -= 1
            del self._voxels[key]
            block = self._blocks[self._block_key(key)]
            block.discard(key)
            if not block:
                del self._blocks[self._block_key(key)]
        if removed_keys:
            self._point_count = max(0, self._point_count - removed_points)
            self._revision += 1
        return len(removed_keys), int(removed_points)

    def voxel_key(self, point: np.ndarray) -> tuple[int, int, int]:
        scaled = np.floor(
            np.asarray(point, dtype=np.float64).reshape(3) / self.voxel_size
        ).astype(np.int64)
        return int(scaled[0]), int(scaled[1]), int(scaled[2])

    def add_points(
        self,
        points_world: np.ndarray,
        intensities: np.ndarray | None = None,
    ) -> int:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if intensities is None:
            intensity_values = np.ones(points.shape[0], dtype=np.float64)
        else:
            supplied = np.asarray(intensities, dtype=np.float64).reshape(-1)
            if supplied.size != finite.size:
                raise ValueError("intensities must match points")
            intensity_values = supplied[finite]
            intensity_values = np.nan_to_num(
                intensity_values, nan=0.0, posinf=0.0, neginf=0.0
            )

        if points.shape[0] == 0:
            return 0

        # Most acoustic pings contribute many neighbouring returns to the
        # same coarse seabed voxel.  Group them in NumPy and merge Chan/Welford
        # sufficient statistics once per occupied voxel instead of executing
        # one Python hash lookup and allocation per return.
        voxel_keys = np.floor(points / self.voxel_size).astype(np.int64)
        unique_keys, inverse = np.unique(
            voxel_keys, axis=0, return_inverse=True
        )
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=unique_keys.shape[0])
        starts = np.concatenate(([0], np.cumsum(counts)))
        for index, key_array in enumerate(unique_keys):
            selected = order[starts[index] : starts[index + 1]]
            batch_points = points[selected]
            batch_mean = np.mean(batch_points, axis=0)
            centered = batch_points - batch_mean
            batch_scatter = centered.T @ centered
            key = tuple(int(value) for value in key_array)
            stats = self._voxels.get(key)
            if stats is None:
                stats = _VoxelStats(
                    reference=np.asarray(key, dtype=np.float64)
                    * self.voxel_size
                )
                self._voxels[key] = stats
                self._blocks.setdefault(self._block_key(key), set()).add(key)
            previous_count = stats.count
            stats.merge(
                int(selected.size),
                batch_mean,
                batch_scatter,
                float(np.sum(intensity_values[selected])),
            )
            if previous_count < self.min_points <= stats.count:
                self._mature_voxel_count += 1
        self._point_count += int(points.shape[0])
        self._revision += 1
        return int(points.shape[0])

    def remove_points(
        self,
        points_world: np.ndarray,
        intensities: np.ndarray | None = None,
    ) -> int:
        """Remove an exact, previously inserted point batch.

        This operation supports a temporal fixed-lag surfel map.  It is not a
        nearest-neighbour deletion: points are grouped by their original voxel
        keys and their sufficient statistics are subtracted exactly.
        """

        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if intensities is None:
            intensity_values = np.ones(points.shape[0], dtype=np.float64)
        else:
            supplied = np.asarray(intensities, dtype=np.float64).reshape(-1)
            if supplied.size != finite.size:
                raise ValueError("intensities must match points")
            intensity_values = np.nan_to_num(
                supplied[finite], nan=0.0, posinf=0.0, neginf=0.0
            )
        if points.shape[0] == 0:
            return 0

        voxel_keys = np.floor(points / self.voxel_size).astype(np.int64)
        unique_keys, inverse = np.unique(
            voxel_keys, axis=0, return_inverse=True
        )
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=unique_keys.shape[0])
        starts = np.concatenate(([0], np.cumsum(counts)))
        for index, key_array in enumerate(unique_keys):
            selected = order[starts[index] : starts[index + 1]]
            key = tuple(int(value) for value in key_array)
            stats = self._voxels.get(key)
            if stats is None or stats.count < selected.size:
                raise ValueError(
                    "point batch was not previously inserted in the surfel map"
                )
            batch_points = points[selected]
            batch_mean = np.mean(batch_points, axis=0)
            centered = batch_points - batch_mean
            batch_scatter = centered.T @ centered
            was_mature = stats.count >= self.min_points
            stats.subtract(
                int(selected.size),
                batch_mean,
                batch_scatter,
                float(np.sum(intensity_values[selected])),
            )
            if stats.count == 0:
                if was_mature:
                    self._mature_voxel_count -= 1
                del self._voxels[key]
                block_key = self._block_key(key)
                block = self._blocks[block_key]
                block.discard(key)
                if not block:
                    del self._blocks[block_key]
            elif was_mature and stats.count < self.min_points:
                self._mature_voxel_count -= 1

        self._point_count -= int(points.shape[0])
        if self._point_count < 0:
            raise RuntimeError("surfel-map point count became negative")
        self._revision += 1
        return int(points.shape[0])

    def _surfel_for(
        self, key: tuple[int, int, int], stats: _VoxelStats
    ) -> Surfel | None:
        if stats.count < self.min_points:
            return None
        if stats.cached_version == stats.version:
            return stats.cached_surfel

        covariance = stats.scatter / max(1, stats.count - 1)
        covariance = 0.5 * (covariance + covariance.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            stats.cached_version = stats.version
            stats.cached_surfel = None
            return None
        eigenvalues = np.maximum(eigenvalues, 1.0e-9)
        total = float(np.sum(eigenvalues))
        planarity = float(eigenvalues[0] / total)
        tangent_ratio = float(eigenvalues[1] / eigenvalues[2])
        if (
            not np.isfinite(planarity)
            or planarity > self.max_planarity
            or tangent_ratio < self.min_tangent_ratio
        ):
            surfel = None
        else:
            normal = eigenvectors[:, 0]
            if normal[2] < 0.0:
                normal = -normal
            surfel = Surfel(
                centroid=stats.mean.copy(),
                normal=np.asarray(normal, dtype=np.float64),
                covariance=covariance,
                normal_variance=float(eigenvalues[0]),
                planarity=planarity,
                count=stats.count,
                intensity=stats.intensity_sum / float(stats.count),
                voxel_key=key,
            )
        stats.cached_version = stats.version
        stats.cached_surfel = surfel
        return surfel

    def query(self, point_world: np.ndarray) -> Surfel | None:
        """Return the best nearby plane for one transformed acoustic point."""

        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        if not np.isfinite(point).all():
            return None
        base = self.voxel_key(point)
        best: Surfel | None = None
        best_score = np.inf
        for offset in self._NEIGHBOURS:
            key = (
                base[0] + offset[0],
                base[1] + offset[1],
                base[2] + offset[2],
            )
            stats = self._voxels.get(key)
            if stats is None:
                continue
            surfel = self._surfel_for(key, stats)
            if surfel is None:
                continue
            delta = point - surfel.centroid
            centroid_distance = float(np.linalg.norm(delta))
            if centroid_distance > self.max_centroid_distance:
                continue
            plane_distance = abs(float(surfel.normal @ delta))
            if plane_distance > self.max_plane_distance:
                continue
            score = plane_distance + 0.04 * centroid_distance
            if score < best_score:
                best = surfel
                best_score = score
        return best

    def associate_points(
        self, points_world: np.ndarray
    ) -> SurfelAssociationBatch:
        """Reference batch association with exactly the scalar query rules."""

        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        valid_mask = np.zeros(points.shape[0], dtype=bool)
        centroids = np.zeros((points.shape[0], 3), dtype=np.float64)
        normals = np.zeros((points.shape[0], 3), dtype=np.float64)
        normal_variances = np.zeros(points.shape[0], dtype=np.float64)
        for index, point in enumerate(points):
            surfel = self.query(point)
            if surfel is None:
                continue
            valid_mask[index] = True
            centroids[index] = surfel.centroid
            normals[index] = surfel.normal
            normal_variances[index] = surfel.normal_variance
        return SurfelAssociationBatch(
            valid_mask=valid_mask,
            centroids=centroids,
            normals=normals,
            normal_variances=normal_variances,
        )

    def association_snapshot(self) -> SurfelMapSnapshot:
        """Pack mature surfels in lexicographic voxel-key order.

        The order supports deterministic binary search on accelerators while
        retaining the exact 27-neighbour candidate set used by :meth:`query`.
        """

        surfels: list[Surfel] = []
        for key, stats in self._association_items():
            surfel = self._surfel_for(key, stats)
            if surfel is not None:
                surfels.append(surfel)
        if not surfels:
            return SurfelMapSnapshot(
                revision=self._revision,
                voxel_keys=np.empty((0, 3), dtype=np.int32),
                centroids=np.empty((0, 3), dtype=np.float32),
                normals=np.empty((0, 3), dtype=np.float32),
                normal_variances=np.empty(0, dtype=np.float32),
            )
        keys64 = np.asarray(
            [surfel.voxel_key for surfel in surfels], dtype=np.int64
        )
        int32 = np.iinfo(np.int32)
        if np.any(keys64 < int32.min) or np.any(keys64 > int32.max):
            raise OverflowError("voxel key exceeds the Warp int32 key range")
        order = np.lexsort((keys64[:, 2], keys64[:, 1], keys64[:, 0]))
        return SurfelMapSnapshot(
            revision=self._revision,
            voxel_keys=np.ascontiguousarray(keys64[order], dtype=np.int32),
            centroids=np.ascontiguousarray(
                [surfels[index].centroid for index in order], dtype=np.float32
            ),
            normals=np.ascontiguousarray(
                [surfels[index].normal for index in order], dtype=np.float32
            ),
            normal_variances=np.ascontiguousarray(
                [surfels[index].normal_variance for index in order],
                dtype=np.float32,
            ),
        )

    def surfels(self) -> list[Surfel]:
        output: list[Surfel] = []
        for key, stats in self._voxels.items():
            surfel = self._surfel_for(key, stats)
            if surfel is not None:
                output.append(surfel)
        return output

    def centroids(self, *, mature_only: bool = True) -> np.ndarray:
        if mature_only:
            points = [surfel.centroid for surfel in self.surfels()]
        else:
            points = [stats.mean for stats in self._voxels.values()]
        if not points:
            return np.empty((0, 3), dtype=np.float64)
        return np.ascontiguousarray(points, dtype=np.float64)

    def intensities(self, *, mature_only: bool = True) -> np.ndarray:
        if mature_only:
            values = [surfel.intensity for surfel in self.surfels()]
        else:
            values = [
                stats.intensity_sum / max(1, stats.count)
                for stats in self._voxels.values()
            ]
        return np.asarray(values, dtype=np.float64)
