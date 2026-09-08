"""Incremental local map for direct side-scan intensity alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class IntensityMapSample:
    """Locally fitted intensity and horizontal gradient at one map point."""

    intensity: float
    gradient_xy: np.ndarray
    variance: float
    neighbours: int
    # Newest acquisition that contributed to the fitted reference support.
    # Carrying this lineage into the ESIKF lets the accepted image residual be
    # represented as a binary factor in the global graph instead of being
    # hidden inside an odometry edge.
    reference_observation_id: int
    # Keyframe/submap whose coordinate frame owns all cells used by this
    # sample.  A fitted plane is never assembled from multiple anchors.
    anchor_id: int = -1


@dataclass(slots=True)
class _IntensityCell:
    count: int = 0
    mean_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    mean_intensity: float = 0.0
    intensity_m2: float = 0.0
    first_observation_id: int = -(2**31)
    last_observation_id: int = -(2**31)

    def add(
        self, xy: np.ndarray, intensity: float, observation_id: int | None
    ) -> None:
        if observation_id is not None:
            identifier = int(observation_id)
            if self.count == 0:
                self.first_observation_id = identifier
            else:
                self.first_observation_id = min(
                    self.first_observation_id, identifier
                )
            self.last_observation_id = max(
                self.last_observation_id, identifier
            )
        self.count += 1
        delta_xy = xy - self.mean_xy
        self.mean_xy += delta_xy / float(self.count)
        delta = float(intensity - self.mean_intensity)
        self.mean_intensity += delta / float(self.count)
        self.intensity_m2 += delta * float(intensity - self.mean_intensity)

    def merge(
        self,
        count: int,
        mean_xy: np.ndarray,
        mean_intensity: float,
        intensity_m2: float,
        observation_id: int | None,
    ) -> None:
        """Merge exact sufficient statistics for one acquisition batch.

        Dense side-scan quadrature commonly contributes tens of samples to
        the same map cell in one strip.  Combining their count, mean and M2
        with Chan's parallel-variance formula is mathematically equivalent
        to calling :meth:`add` for every sample, while avoiding a Python hash
        lookup and NumPy allocation for each ray.
        """

        batch_count = int(count)
        if batch_count <= 0:
            return
        batch_mean_xy = np.asarray(mean_xy, dtype=np.float64).reshape(2)
        batch_mean_intensity = float(mean_intensity)
        batch_m2 = max(0.0, float(intensity_m2))
        if observation_id is not None:
            identifier = int(observation_id)
            if self.count == 0:
                self.first_observation_id = identifier
            else:
                self.first_observation_id = min(
                    self.first_observation_id, identifier
                )
            self.last_observation_id = max(
                self.last_observation_id, identifier
            )
        if self.count == 0:
            self.count = batch_count
            self.mean_xy = batch_mean_xy.copy()
            self.mean_intensity = batch_mean_intensity
            self.intensity_m2 = batch_m2
            return

        previous_count = self.count
        combined_count = previous_count + batch_count
        xy_delta = batch_mean_xy - self.mean_xy
        intensity_delta = batch_mean_intensity - self.mean_intensity
        batch_weight = batch_count / float(combined_count)
        self.mean_xy += xy_delta * batch_weight
        self.mean_intensity += intensity_delta * batch_weight
        self.intensity_m2 += batch_m2 + (
            intensity_delta
            * intensity_delta
            * previous_count
            * batch_count
            / float(combined_count)
        )
        self.count = combined_count

    @property
    def variance(self) -> float:
        return self.intensity_m2 / max(1, self.count - 1)


class AcousticIntensityMap:
    """Sparse 2-D side-scan intensity map with local plane interpolation.

    Integrated-sonar backscatter is already registered to the simultaneous
    bathymetric bottom detections. A weighted affine plane fitted to
    neighbouring cells supplies both the predicted intensity and its spatial
    gradient for a direct ESIKF update.
    """

    _OFFSETS = tuple(
        (dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)
    )

    def __init__(
        self,
        voxel_size: float = 3.0,
        *,
        min_cell_points: int = 2,
        min_neighbours: int = 6,
        min_gradient: float = 8.0e-4,
        heading_bins: int = 36,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.min_cell_points = int(min_cell_points)
        self.min_neighbours = int(min_neighbours)
        self.min_gradient = float(min_gradient)
        self.heading_bins = int(heading_bins)
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if (
            self.min_cell_points < 1
            or self.min_neighbours < 3
            or self.heading_bins < 4
        ):
            raise ValueError("intensity-map support thresholds are invalid")
        # Cells are deliberately split by position, acquisition heading and
        # submap/keyframe anchor.  Combining two anchors in one running mean
        # would erase the reference lineage needed by a binary pose-graph
        # factor, even when both observations cover the same seabed voxel.
        self._cells: dict[tuple[int, int, int, int], _IntensityCell] = {}
        # Most survey cells are observed by exactly one submap.  Store that
        # common case as an integer instead of allocating a separate set for
        # every cell; upgrade lazily only when a genuine revisit contributes
        # another anchor.
        self._anchors_by_cell: dict[
            tuple[int, int, int], int | set[int]
        ] = {}
        self._anchor_ids: set[int] = set()
        self._point_count = 0

    def __len__(self) -> int:
        return len(self._cells)

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def anchor_ids(self) -> tuple[int, ...]:
        """Return the submap anchors currently represented in the map."""

        return tuple(sorted(self._anchor_ids))

    def clear(self) -> None:
        self._cells.clear()
        self._anchors_by_cell.clear()
        self._anchor_ids.clear()
        self._point_count = 0

    def _key(self, xy: np.ndarray) -> tuple[int, int]:
        scaled = np.floor(
            np.asarray(xy, dtype=np.float64).reshape(2) / self.voxel_size
        ).astype(np.int64)
        return int(scaled[0]), int(scaled[1])

    def _heading_key(self, heading_rad: float | None) -> int:
        if heading_rad is None:
            return -1
        wrapped = (float(heading_rad) + np.pi) % (2.0 * np.pi)
        return int(
            np.floor(
                wrapped / (2.0 * np.pi) * self.heading_bins + 0.5
            )
        ) % self.heading_bins

    def add_points(
        self,
        points_world: np.ndarray,
        intensities: np.ndarray,
        *,
        heading_rad: float | None = None,
        observation_id: int | None = None,
        anchor_id: int = -1,
    ) -> int:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        values = np.asarray(intensities, dtype=np.float64).reshape(-1)
        if values.size != points.shape[0]:
            raise ValueError("intensities must match points")
        valid = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(values)
            & (values >= 0.0)
        )
        heading_key = self._heading_key(heading_rad)
        anchor_key = int(anchor_id)
        if anchor_key >= 0 and np.any(valid):
            self._anchor_ids.add(anchor_key)
        valid_points = points[valid]
        valid_values = values[valid]
        if valid_points.shape[0] == 0:
            return 0

        spatial_keys, inverse, counts = np.unique(
            np.floor(valid_points[:, :2] / self.voxel_size).astype(np.int64),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        sum_x = np.bincount(inverse, weights=valid_points[:, 0])
        sum_y = np.bincount(inverse, weights=valid_points[:, 1])
        sum_intensity = np.bincount(inverse, weights=valid_values)
        sum_intensity_squared = np.bincount(
            inverse, weights=valid_values * valid_values
        )
        for cell_index, key_values in enumerate(spatial_keys):
            cell_count = int(counts[cell_index])
            mean_xy = np.array(
                [sum_x[cell_index], sum_y[cell_index]], dtype=np.float64
            ) / float(cell_count)
            mean_intensity = sum_intensity[cell_index] / float(cell_count)
            cell_m2 = (
                sum_intensity_squared[cell_index]
                - sum_intensity[cell_index]
                * sum_intensity[cell_index]
                / float(cell_count)
            )
            spatial_key = int(key_values[0]), int(key_values[1])
            spatial_heading_key = (
                spatial_key[0],
                spatial_key[1],
                heading_key,
            )
            key = (*spatial_heading_key, anchor_key)
            cell = self._cells.get(key)
            if cell is None:
                cell = _IntensityCell()
                self._cells[key] = cell
                anchor_support = self._anchors_by_cell.get(
                    spatial_heading_key
                )
                if anchor_support is None:
                    self._anchors_by_cell[spatial_heading_key] = anchor_key
                elif isinstance(anchor_support, int):
                    if anchor_support != anchor_key:
                        self._anchors_by_cell[spatial_heading_key] = {
                            anchor_support,
                            anchor_key,
                        }
                else:
                    anchor_support.add(anchor_key)
            cell.merge(
                cell_count,
                mean_xy,
                mean_intensity,
                cell_m2,
                observation_id,
            )
        count = int(np.count_nonzero(valid))
        self._point_count += count
        return count

    def query(
        self,
        point_world: np.ndarray,
        *,
        heading_rad: float | None = None,
        maximum_observation_id: int | None = None,
        anchor_id: int | None = None,
        _candidate_cache: dict | None = None,
    ) -> IntensityMapSample | None:
        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        if not np.isfinite(point).all():
            return None
        point_x = float(point[0])
        point_y = float(point[1])
        base = (
            math.floor(point_x / self.voxel_size),
            math.floor(point_y / self.voxel_size),
        )
        heading_key = self._heading_key(heading_rad)
        candidates: dict[int, list[_IntensityCell]] = {}
        radius = 2.6 * self.voxel_size
        radius_squared = radius * radius
        observation_cutoff = (
            None
            if maximum_observation_id is None
            else int(maximum_observation_id)
        )
        fixed_anchor_id = None if anchor_id is None else int(anchor_id)
        # Candidate membership depends on the integer query cell, heading,
        # causal cutoff, and fixed anchor, but not on the sub-cell query
        # coordinate.  One iterated ESIKF update evaluates the same immutable
        # map several times, so cache only this exact broad-phase result and
        # repeat the coordinate-dependent radius and plane fits below.
        cache_key = (
            base[0],
            base[1],
            heading_key,
            observation_cutoff,
            fixed_anchor_id,
        )
        candidate_entries = self._candidate_entries(
            base[0],
            base[1],
            heading_key,
            observation_cutoff,
            fixed_anchor_id,
            cache_key=cache_key,
            candidate_cache=_candidate_cache,
        )

        for candidate_key, cell in candidate_entries:
            delta_x = float(cell.mean_xy[0]) - point_x
            delta_y = float(cell.mean_xy[1]) - point_y
            if delta_x * delta_x + delta_y * delta_y > radius_squared:
                continue
            candidate_cells = candidates.get(candidate_key)
            if candidate_cells is None:
                candidates[candidate_key] = [cell]
            else:
                candidate_cells.append(cell)

        # Rank anchors by effective local support, then attempt a plane fit
        # independently for each one.  Trying the next anchor when the first
        # is geometrically degenerate avoids rejecting a valid, slightly
        # smaller submap while preserving single-anchor lineage.
        # An anchor with fewer than ``min_neighbours`` cells is guaranteed to
        # be rejected by ``_fit_anchor_plane``.  Removing it before ranking is
        # therefore mathematically neutral and avoids scoring thousands of
        # impossible candidates along the leading edge of a strip.
        viable_candidates = [
            item
            for item in candidates.items()
            if len(item[1]) >= self.min_neighbours
        ]
        if not viable_candidates:
            return None
        # The overwhelmingly common causal case contains one eligible submap
        # anchor.  Sorting and evaluating its support score cannot change its
        # rank, so bypass that allocation-heavy work without changing the
        # fitted plane or any acceptance threshold.
        if len(viable_candidates) == 1:
            candidate_anchor_id, cells = viable_candidates[0]
            return self._fit_anchor_plane(point, candidate_anchor_id, cells)
        ranked_candidates = sorted(
            viable_candidates,
            key=lambda item: self._anchor_support_score(point, item),
            reverse=True,
        )
        for anchor_id, cells in ranked_candidates:
            sample = self._fit_anchor_plane(point, anchor_id, cells)
            if sample is not None:
                return sample
        return None

    def _anchor_support_score(
        self,
        point: np.ndarray,
        item: tuple[int, list[_IntensityCell]],
    ) -> tuple[float, int, int, int]:
        """Return scalar query()'s deterministic anchor ordering key."""

        anchor_id, cells = item
        distance_squared = np.asarray(
            [
                np.sum((cell.mean_xy - point[:2]) ** 2)
                for cell in cells
            ],
            dtype=np.float64,
        )
        effective_support = float(
            np.sum(
                np.exp(
                    -0.5
                    * distance_squared
                    / (1.25 * self.voxel_size) ** 2
                )
                * np.sqrt(
                    np.minimum(
                        np.asarray(
                            [cell.count for cell in cells],
                            dtype=np.float64,
                        ),
                        12.0,
                    )
                )
            )
        )
        # The final term makes ties deterministic without giving newer
        # anchors preferential treatment.
        return (
            effective_support,
            len(cells),
            sum(cell.count for cell in cells),
            -abs(anchor_id),
        )

    def _candidate_entries(
        self,
        base_x: int,
        base_y: int,
        heading_key: int,
        observation_cutoff: int | None,
        fixed_anchor_id: int | None,
        *,
        cache_key: tuple | None = None,
        candidate_cache: dict | None = None,
    ) -> tuple[tuple[int, _IntensityCell], ...]:
        """Return the exact causal 5 x 5 broad-phase neighbourhood."""

        if cache_key is None:
            cache_key = (
                int(base_x),
                int(base_y),
                int(heading_key),
                observation_cutoff,
                fixed_anchor_id,
            )
        candidate_entries = (
            None
            if candidate_cache is None
            else candidate_cache.get(cache_key)
        )
        if candidate_entries is not None:
            return candidate_entries

        gathered_entries: list[tuple[int, _IntensityCell]] = []
        for offset_x, offset_y in self._OFFSETS:
            spatial_heading_key = (
                int(base_x) + offset_x,
                int(base_y) + offset_y,
                int(heading_key),
            )
            if fixed_anchor_id is None:
                anchor_support = self._anchors_by_cell.get(
                    spatial_heading_key
                )
                if anchor_support is None:
                    anchor_ids = ()
                elif isinstance(anchor_support, int):
                    anchor_ids = (anchor_support,)
                else:
                    anchor_ids = anchor_support
            else:
                anchor_ids = (int(fixed_anchor_id),)
            for candidate_anchor_id in anchor_ids:
                cell = self._cells.get(
                    (*spatial_heading_key, candidate_anchor_id)
                )
                if cell is None or cell.count < self.min_cell_points:
                    continue
                # A cell whose newest contributor is newer than the causal
                # cutoff is rejected as a whole, matching scalar query().
                if (
                    observation_cutoff is not None
                    and cell.last_observation_id > observation_cutoff
                ):
                    continue
                gathered_entries.append((int(candidate_anchor_id), cell))
        result = tuple(gathered_entries)
        if candidate_cache is not None:
            candidate_cache[cache_key] = result
        return result

    def query_batch(
        self,
        points_world: np.ndarray,
        *,
        heading_rad: float | None = None,
        maximum_observation_id: int | None = None,
        _candidate_cache: dict | None = None,
    ) -> list[IntensityMapSample | None]:
        """Evaluate unrestricted queries while batching the one-anchor case.

        Anchor discovery and multi-anchor ranking retain :meth:`query`'s exact
        ordering.  Points having one viable anchor are grouped and evaluated
        by :meth:`query_fixed_anchor_batch`; ambiguous revisit points continue
        through the scalar reference path.
        """

        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        samples: list[IntensityMapSample | None] = [None] * points.shape[0]
        if points.shape[0] == 0:
            return samples

        heading_key = self._heading_key(heading_rad)
        observation_cutoff = (
            None
            if maximum_observation_id is None
            else int(maximum_observation_id)
        )
        radius_squared = (2.6 * self.voxel_size) ** 2
        ranked_anchors_by_point: dict[int, list[int]] = {}
        points_by_anchor: dict[int, list[int]] = {}
        finite_indices = np.flatnonzero(np.isfinite(points).all(axis=1))
        if finite_indices.size == 0:
            return samples
        base_coordinates = np.floor(
            points[finite_indices, :2] / self.voxel_size
        ).astype(np.int64)
        unique_bases, base_inverse = np.unique(
            base_coordinates, axis=0, return_inverse=True
        )
        entries_by_base = [
            self._candidate_entries(
                int(base_x),
                int(base_y),
                heading_key,
                observation_cutoff,
                None,
                candidate_cache=_candidate_cache,
            )
            for base_x, base_y in unique_bases
        ]

        for local_index, point_index in enumerate(finite_indices):
            point_x = float(points[point_index, 0])
            point_y = float(points[point_index, 1])
            entries = entries_by_base[int(base_inverse[local_index])]
            cells_by_anchor: dict[int, list[_IntensityCell]] = {}
            for candidate_anchor_id, cell in entries:
                delta_x = float(cell.mean_xy[0]) - point_x
                delta_y = float(cell.mean_xy[1]) - point_y
                if delta_x * delta_x + delta_y * delta_y > radius_squared:
                    continue
                cells_by_anchor.setdefault(candidate_anchor_id, []).append(
                    cell
                )
            viable_candidates = [
                item
                for item in cells_by_anchor.items()
                if len(item[1]) >= self.min_neighbours
            ]
            if not viable_candidates:
                continue
            if len(viable_candidates) == 1:
                ranked_anchors = [int(viable_candidates[0][0])]
            else:
                ranked_anchors = [
                    int(candidate_anchor_id)
                    for candidate_anchor_id, _cells in sorted(
                        viable_candidates,
                        key=lambda item: self._anchor_support_score(
                            points[point_index], item
                        ),
                        reverse=True,
                    )
                ]
            ranked_anchors_by_point[int(point_index)] = ranked_anchors
            for candidate_anchor_id in ranked_anchors:
                points_by_anchor.setdefault(candidate_anchor_id, []).append(
                    int(point_index)
                )

        sample_by_point_anchor: dict[
            tuple[int, int], IntensityMapSample | None
        ] = {}
        for candidate_anchor_id, point_indices in points_by_anchor.items():
            selected = np.asarray(point_indices, dtype=np.int64)
            selected_samples = self.query_fixed_anchor_batch(
                points[selected],
                heading_rad=heading_rad,
                maximum_observation_id=observation_cutoff,
                anchor_id=candidate_anchor_id,
                _candidate_cache=_candidate_cache,
            )
            for point_index, sample in zip(point_indices, selected_samples):
                sample_by_point_anchor[
                    (point_index, candidate_anchor_id)
                ] = sample

        for point_index, ranked_anchors in ranked_anchors_by_point.items():
            for candidate_anchor_id in ranked_anchors:
                sample = sample_by_point_anchor[
                    (point_index, candidate_anchor_id)
                ]
                if sample is not None:
                    samples[point_index] = sample
                    break
        return samples

    def query_fixed_anchor_batch(
        self,
        points_world: np.ndarray,
        *,
        heading_rad: float | None,
        maximum_observation_id: int | None,
        anchor_id: int,
        _candidate_cache: dict | None = None,
    ) -> list[IntensityMapSample | None]:
        """Evaluate an exact fixed-anchor neighbourhood for many points.

        Once an intensity ESIKF iteration has selected one submap anchor, all
        points use the same immutable map slice.  Packing the 5 x 5 cell
        neighbourhoods and applying the existing plane equations in batches
        removes tens of thousands of tiny NumPy/LAPACK calls.  The spatial
        radius, causal cutoff, support, gradient, and variance gates are the
        same as :meth:`query`; only their execution layout changes.
        """

        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        samples: list[IntensityMapSample | None] = [None] * points.shape[0]
        if points.shape[0] == 0:
            return samples

        finite_points = np.isfinite(points).all(axis=1)
        heading_key = self._heading_key(heading_rad)
        fixed_anchor_id = int(anchor_id)
        observation_cutoff = (
            None
            if maximum_observation_id is None
            else int(maximum_observation_id)
        )
        maximum_cells = len(self._OFFSETS)
        positions = np.zeros(
            (points.shape[0], maximum_cells, 2), dtype=np.float64
        )
        intensities = np.zeros(
            (points.shape[0], maximum_cells), dtype=np.float64
        )
        variances = np.zeros_like(intensities)
        counts = np.zeros_like(intensities)
        observation_ids = np.full(
            (points.shape[0], maximum_cells),
            -(2**31),
            dtype=np.int64,
        )
        support = np.zeros(
            (points.shape[0], maximum_cells), dtype=bool
        )
        radius_squared = (2.6 * self.voxel_size) ** 2

        finite_indices = np.flatnonzero(finite_points)
        base_coordinates = np.floor(
            points[finite_indices, :2] / self.voxel_size
        ).astype(np.int64)
        unique_bases, base_inverse = np.unique(
            base_coordinates, axis=0, return_inverse=True
        )
        entries_by_base = [
            self._candidate_entries(
                int(base_x),
                int(base_y),
                heading_key,
                observation_cutoff,
                fixed_anchor_id,
                candidate_cache=_candidate_cache,
            )
            for base_x, base_y in unique_bases
        ]

        for local_index, point_index in enumerate(finite_indices):
            point_x = float(points[point_index, 0])
            point_y = float(points[point_index, 1])
            entries = entries_by_base[int(base_inverse[local_index])]
            slot = 0
            for _candidate_anchor_id, cell in entries:
                delta_x = float(cell.mean_xy[0]) - point_x
                delta_y = float(cell.mean_xy[1]) - point_y
                if delta_x * delta_x + delta_y * delta_y > radius_squared:
                    continue
                positions[point_index, slot] = cell.mean_xy
                intensities[point_index, slot] = cell.mean_intensity
                variances[point_index, slot] = cell.variance
                counts[point_index, slot] = cell.count
                observation_ids[point_index, slot] = (
                    cell.last_observation_id
                )
                support[point_index, slot] = True
                slot += 1

        neighbour_counts = np.count_nonzero(support, axis=1)
        eligible = np.flatnonzero(
            neighbour_counts >= self.min_neighbours
        )
        if eligible.size == 0:
            return samples

        packed_support = support[eligible]
        packed_positions = positions[eligible]
        packed_intensities = intensities[eligible]
        packed_variances = variances[eligible]
        packed_counts = counts[eligible]
        packed_observation_ids = observation_ids[eligible]
        delta = packed_positions - points[eligible, None, :2]
        delta[~packed_support] = 0.0
        covariance = np.swapaxes(delta, 1, 2) @ delta
        covariance /= neighbour_counts[eligible, None, None]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance_supported = (
            eigenvalues[:, 0] >= (0.12 * self.voxel_size) ** 2
        )

        local_coordinates = delta @ eigenvectors
        local_minimum = np.min(
            np.where(
                packed_support[:, :, None],
                local_coordinates,
                np.inf,
            ),
            axis=1,
        )
        local_maximum = np.max(
            np.where(
                packed_support[:, :, None],
                local_coordinates,
                -np.inf,
            ),
            axis=1,
        )
        support_margin = 0.12 * self.voxel_size
        surrounded = np.all(
            (local_minimum < -support_margin)
            & (local_maximum > support_margin),
            axis=1,
        )

        distance_squared = np.sum(delta * delta, axis=2)
        weights = np.exp(
            -0.5
            * distance_squared
            / (1.25 * self.voxel_size) ** 2
        )
        weights *= np.sqrt(np.minimum(packed_counts, 12.0))
        weights *= packed_support
        design = np.empty(
            (eligible.size, maximum_cells, 3), dtype=np.float64
        )
        design[:, :, 0] = packed_support
        design[:, :, 1:3] = delta
        normal = np.swapaxes(design, 1, 2) @ (
            weights[:, :, None] * design
        )
        normal[:, 0, 0] += 1.0e-6
        normal[:, 1, 1] += 2.0e-3
        normal[:, 2, 2] += 2.0e-3
        right = np.swapaxes(design, 1, 2) @ (
            weights * packed_intensities
        )[:, :, None]
        coefficients = np.linalg.solve(normal, right)[:, :, 0]
        gradients = coefficients[:, 1:3]
        fitted = np.sum(design * coefficients[:, None, :], axis=2)
        weight_sum = np.sum(weights, axis=1)
        residual_variance = np.sum(
            weights * (packed_intensities - fitted) ** 2, axis=1
        ) / weight_sum
        cell_variance = np.sum(
            weights * packed_variances, axis=1
        ) / weight_sum
        latest_observation_ids = np.max(
            packed_observation_ids, axis=1
        )
        fit_supported = (
            covariance_supported
            & surrounded
            & np.isfinite(coefficients).all(axis=1)
            & (
                np.linalg.norm(gradients, axis=1)
                >= self.min_gradient
            )
        )

        for packed_index in np.flatnonzero(fit_supported):
            point_index = int(eligible[packed_index])
            samples[point_index] = IntensityMapSample(
                intensity=float(coefficients[packed_index, 0]),
                gradient_xy=np.asarray(
                    gradients[packed_index], dtype=np.float64
                ).copy(),
                variance=max(
                    float(
                        residual_variance[packed_index]
                        + cell_variance[packed_index]
                    ),
                    2.5e-3,
                ),
                neighbours=int(neighbour_counts[point_index]),
                reference_observation_id=int(
                    latest_observation_ids[packed_index]
                ),
                anchor_id=fixed_anchor_id,
            )
        return samples

    def _fit_anchor_plane(
        self,
        point: np.ndarray,
        anchor_id: int,
        cells: list[_IntensityCell],
    ) -> IntensityMapSample | None:
        """Fit one local plane using cells owned by exactly one anchor."""

        if len(cells) < self.min_neighbours:
            return None
        # Materialize each sufficient statistic once.  The previous code
        # repeatedly rebuilt these same tiny arrays for weighting, fitting,
        # and variance estimation on every ESIKF iteration.
        positions = np.asarray(
            [cell.mean_xy for cell in cells], dtype=np.float64
        )
        intensities = np.fromiter(
            (cell.mean_intensity for cell in cells),
            dtype=np.float64,
            count=len(cells),
        )
        variances = np.fromiter(
            (cell.variance for cell in cells),
            dtype=np.float64,
            count=len(cells),
        )
        counts = np.fromiter(
            (cell.count for cell in cells),
            dtype=np.float64,
            count=len(cells),
        )
        latest_observation_id = max(
            cell.last_observation_id for cell in cells
        )

        delta = positions - point[:2]
        spatial_covariance = delta.T @ delta / float(len(positions))
        eigenvalues, eigenvectors = np.linalg.eigh(spatial_covariance)
        if eigenvalues[0] < (0.12 * self.voxel_size) ** 2:
            return None
        # A side-scan line observes a new cross-track strip.  At the leading
        # edge of a survey, all old samples lie behind that strip; fitting a
        # plane there is extrapolation and systematically pulls the estimate
        # back toward the old ping.  Direct updates are valid only where the
        # query is bracketed by previously mapped samples in both local map
        # directions (normally at swath overlap or a revisit).
        local_coordinates = delta @ eigenvectors
        support_margin = 0.12 * self.voxel_size
        surrounded = np.all(
            (np.min(local_coordinates, axis=0) < -support_margin)
            & (np.max(local_coordinates, axis=0) > support_margin)
        )
        if not surrounded:
            return None
        distance_squared = np.sum(delta * delta, axis=1)
        weights = np.exp(
            -0.5 * distance_squared / (1.25 * self.voxel_size) ** 2
        )
        weights *= np.sqrt(np.minimum(counts, 12.0))
        design = np.empty((len(cells), 3), dtype=np.float64)
        design[:, 0] = 1.0
        design[:, 1:3] = delta
        normal = design.T @ (weights[:, None] * design)
        normal[0, 0] += 1.0e-6
        normal[1, 1] += 2.0e-3
        normal[2, 2] += 2.0e-3
        right = design.T @ (weights * intensities)
        try:
            coefficients = np.linalg.solve(normal, right)
        except np.linalg.LinAlgError:
            return None
        gradient = coefficients[1:3]
        if (
            not np.isfinite(coefficients).all()
            or np.linalg.norm(gradient) < self.min_gradient
        ):
            return None
        fitted = design @ coefficients
        residual_variance = float(
            np.average((intensities - fitted) ** 2, weights=weights)
        )
        cell_variance = float(np.average(variances, weights=weights))
        return IntensityMapSample(
            intensity=float(coefficients[0]),
            gradient_xy=np.asarray(gradient, dtype=np.float64),
            variance=max(residual_variance + cell_variance, 2.5e-3),
            neighbours=len(positions),
            reference_observation_id=latest_observation_id,
            anchor_id=int(anchor_id),
        )
