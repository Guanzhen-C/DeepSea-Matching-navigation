"""Truth-free whole-map terrain/backscatter relocalization.

The ordinary terrain matcher deliberately searches only a small neighbourhood
around the current navigation solution.  That is appropriate while tracking,
but it cannot recover after the navigation error has grown beyond the bounded
search window.  This module supplies a coarse, whole-map place-recognition
stage for that exceptional case.

The search is intentionally only a proposal generator.  Every spatially
separated candidate must still pass bounded three-dimensional occupancy
verification, metric :class:`BathymetricTerrainMatcher` refinement, and the
co-registered backscatter check.  A persistent multi-hypothesis tracker then
propagates all surviving basins with local odometry and requires repeated,
non-overlapping acquisition windows before one candidate can become a
pose-graph factor.  Simulator truth is never an input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .acoustic_radiometry import log_backscatter, uses_relative_power
from .esikf import InertialState
from .math3d import rotation_from_rpy, rpy_from_rotation, wrap_angle
from .real_bathymetry import MeasuredBathymetryGrid


def _readonly_vector(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(shape).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GlobalTerrainRelocalizerConfig:
    """Configuration for the coarse whole-map proposal search.

    ``pyramid_stride`` is expressed in source-map cells.  The Monterey product
    has a 5 m grid, so the default stride four creates a 20 m proposal layer.
    This coarse layer is never reported as the final pose; the local 5 m
    matcher performs the subsequent metric refinement.
    """

    pyramid_stride: int = 4
    yaw_half_window_deg: float = 45.0
    yaw_step_deg: float = 5.0
    maximum_candidates: int = 8
    candidates_per_yaw: int = 24
    basin_translation_separation_m: float = 100.0
    basin_yaw_separation_deg: float = 10.0
    minimum_bathymetry_bins: int = 24
    minimum_backscatter_bins: int = 24
    minimum_valid_fraction: float = 0.999
    minimum_bathymetry_std_m: float = 0.10
    maximum_bathymetry_rmse_m: float = 6.0
    bathymetry_absolute_weight: float = 0.50
    minimum_backscatter_log_std: float = 0.015
    bathymetry_weight: float = 0.35
    backscatter_weight: float = 0.65
    minimum_joint_correlation: float = 0.30
    minimum_basin_score_gap: float = 0.08
    backscatter_log_floor: float = 0.01
    require_backscatter: bool = True

    def __post_init__(self) -> None:
        for name in (
            "pyramid_stride",
            "maximum_candidates",
            "candidates_per_yaw",
            "minimum_bathymetry_bins",
            "minimum_backscatter_bins",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "yaw_half_window_deg",
            "yaw_step_deg",
            "basin_translation_separation_m",
            "basin_yaw_separation_deg",
            "minimum_bathymetry_std_m",
            "maximum_bathymetry_rmse_m",
            "minimum_backscatter_log_std",
            "minimum_joint_correlation",
            "minimum_basin_score_gap",
            "backscatter_log_floor",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.yaw_half_window_deg > 180.0:
            raise ValueError("yaw_half_window_deg must not exceed 180 degrees")
        if self.yaw_step_deg > 2.0 * self.yaw_half_window_deg:
            raise ValueError("yaw_step_deg exceeds the complete yaw search span")
        if self.basin_yaw_separation_deg > 180.0:
            raise ValueError(
                "basin_yaw_separation_deg must not exceed 180 degrees"
            )
        valid_fraction = float(self.minimum_valid_fraction)
        if not np.isfinite(valid_fraction) or not 0.0 < valid_fraction <= 1.0:
            raise ValueError("minimum_valid_fraction must lie in (0, 1]")
        for name in ("bathymetry_weight", "backscatter_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.bathymetry_weight + self.backscatter_weight <= 0.0:
            raise ValueError("at least one global-search modality needs weight")
        absolute_weight = float(self.bathymetry_absolute_weight)
        if (
            not np.isfinite(absolute_weight)
            or not 0.0 <= absolute_weight <= 1.0
        ):
            raise ValueError(
                "bathymetry_absolute_weight must lie in [0, 1]"
            )
        if not isinstance(self.require_backscatter, (bool, np.bool_)):
            raise ValueError("require_backscatter must be boolean")
        if self.require_backscatter and self.backscatter_weight <= 0.0:
            raise ValueError(
                "require_backscatter needs a positive backscatter weight"
            )


@dataclass(frozen=True)
class GlobalTerrainCandidate:
    """One spatially distinct whole-map pose proposal."""

    pose2: np.ndarray
    joint_correlation: float
    bathymetry_correlation: float
    bathymetry_evidence_score: float
    bathymetry_rmse_m: float
    backscatter_correlation: float
    yaw_offset_deg: float
    bathymetry_bins: int
    backscatter_bins: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose2", _readonly_vector(self.pose2, (3,)))
        for name in (
            "joint_correlation",
            "bathymetry_correlation",
            "bathymetry_evidence_score",
            "bathymetry_rmse_m",
            "backscatter_correlation",
            "yaw_offset_deg",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("bathymetry_bins", "backscatter_bins"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class GlobalTerrainSearchResult:
    """Auditable result of one whole-map proposal search."""

    accepted: bool
    candidates: tuple[GlobalTerrainCandidate, ...]
    rejection_reason: str | None
    best_score: float
    second_basin_score: float
    basin_score_gap: float
    evaluated_yaws: int
    finite_lattice_candidates: int
    state_timestamp: float
    bathymetry_observation_std_m: float = 0.0
    backscatter_observation_log_std: float = 0.0
    bathymetry_observability: float = 0.0
    backscatter_observability: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, (bool, np.bool_)):
            raise ValueError("accepted must be boolean")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.accepted and not self.candidates:
            raise ValueError("an accepted search must contain a candidate")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("an accepted search cannot have a rejection reason")
        if not self.accepted and not isinstance(self.rejection_reason, str):
            raise ValueError("a rejected search needs a rejection reason")
        for name in ("evaluated_yaws", "finite_lattice_candidates"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if not np.isfinite(float(self.state_timestamp)):
            raise ValueError("state_timestamp must be finite")
        for name in (
            "bathymetry_observation_std_m",
            "backscatter_observation_log_std",
            "bathymetry_observability",
            "backscatter_observability",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name.endswith("observability") and value > 1.0:
                raise ValueError(f"{name} must not exceed one")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class _RawCandidate:
    pose2: np.ndarray
    score: float
    bathymetry_score: float
    bathymetry_evidence_score: float
    bathymetry_rmse_m: float
    backscatter_score: float
    yaw_offset_deg: float
    bathymetry_bins: int
    backscatter_bins: int


class GlobalTerrainRelocalizer:
    """Propose absolute poses by joint bathymetry/backscatter correlation.

    The implementation uses OpenCV's optimized cross-correlation primitive,
    but constructs zero-mean normalized correlation explicitly from correlation
    sums.  This avoids the non-finite values produced by masked normalized
    template matching when a candidate map window has vanishing variance.
    """

    def __init__(
        self,
        terrain: MeasuredBathymetryGrid,
        config: GlobalTerrainRelocalizerConfig | None = None,
    ) -> None:
        if not isinstance(terrain, MeasuredBathymetryGrid):
            raise TypeError("terrain must be a MeasuredBathymetryGrid")
        self.terrain = terrain
        self.config = config or GlobalTerrainRelocalizerConfig()
        if self.config.require_backscatter and not terrain.has_measured_backscatter:
            raise ValueError(
                "whole-map joint relocalization requires measured backscatter"
            )
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "whole-map terrain relocalization requires OpenCV (cv2)"
            ) from exc
        self._cv2 = cv2
        stride = self.config.pyramid_stride
        self.resolution_m = float(terrain.resolution_m * stride)
        # Each coarse value represents the centre of a stride-by-stride source
        # block, not the first pixel in that block.  Keeping this half-block
        # origin makes proposal coordinates metrically consistent.
        coarse_origin_offset_m = 0.5 * (stride - 1) * terrain.resolution_m
        self._coarse_x0_m = -0.5 * terrain.width + coarse_origin_offset_m
        self._coarse_y0_m = -0.5 * terrain.length + coarse_origin_offset_m
        self._bathymetry, self._bathymetry_valid = self._coarse_grid(
            terrain.navigation_elevation_m,
            terrain.valid_mask,
            logarithmic=False,
        )
        if terrain.has_measured_backscatter:
            self._backscatter, self._backscatter_valid = self._coarse_grid(
                terrain.backscatter,
                terrain.backscatter_valid_mask,
                logarithmic=True,
            )
        else:
            self._backscatter = None
            self._backscatter_valid = None

    def _coarse_grid(
        self,
        values: np.ndarray,
        valid: np.ndarray,
        *,
        logarithmic: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        stride = self.config.pyramid_stride
        rows = (values.shape[0] // stride) * stride
        columns = (values.shape[1] // stride) * stride
        selected = np.asarray(values[:rows, :columns], dtype=np.float32)
        source_valid = np.asarray(valid[:rows, :columns], dtype=bool).copy()
        source_valid &= np.isfinite(selected)
        if logarithmic:
            if uses_relative_power(self.terrain.metadata):
                # Reference maps have no receiver noise. Retain the full
                # documented dB contrast across sections with different gains;
                # a single whole-map absolute floor would erase dark regions.
                selected = np.log(np.maximum(selected, np.finfo(np.float32).tiny))
            else:
                selected = np.log(
                    np.clip(selected, self.config.backscatter_log_floor, None)
                )
            selected = selected.astype(np.float32, copy=False)
        # A true area pyramid suppresses source-grid aliasing.  A coarse cell
        # is valid only when its complete source block is measured, preventing
        # NoData boundaries from being averaged into plausible terrain.
        selected = np.where(source_valid, selected, 0.0).reshape(
            rows // stride,
            stride,
            columns // stride,
            stride,
        )
        block_valid = source_valid.reshape(
            rows // stride,
            stride,
            columns // stride,
            stride,
        )
        selected = selected.mean(axis=(1, 3), dtype=np.float64).astype(
            np.float32
        )
        selected_valid = np.all(block_valid, axis=(1, 3))
        selected[~selected_valid] = 0.0
        return np.ascontiguousarray(selected), np.ascontiguousarray(
            selected_valid
        )

    @staticmethod
    def _yaw_samples(center: float, half_window_deg: float, step_deg: float) -> np.ndarray:
        half = math.radians(float(half_window_deg))
        step = math.radians(float(step_deg))
        count = int(math.floor(half / step))
        offsets = np.arange(-count, count + 1, dtype=np.float64) * step
        offsets = np.unique(np.concatenate((offsets, [-half, 0.0, half])))
        return np.asarray(wrap_angle(float(center) + offsets), dtype=np.float64)

    @staticmethod
    def _detrended_log_backscatter(
        points_body: np.ndarray,
        values: np.ndarray,
        floor: float,
        relative_power: bool = False,
    ) -> np.ndarray:
        cross_track = np.abs(points_body[:, 1])
        scale = max(float(np.percentile(cross_track, 95.0)), 1.0)
        normalized_range = np.clip(cross_track / scale, 0.0, 2.0)
        side = np.sign(points_body[:, 1])
        design = np.column_stack(
            (
                np.ones(points_body.shape[0], dtype=np.float64),
                side,
                normalized_range,
                normalized_range**2,
                side * normalized_range,
            )
        )
        logged = log_backscatter(values, floor, relative_power=relative_power)
        try:
            coefficients = np.linalg.lstsq(design, logged, rcond=None)[0]
        except np.linalg.LinAlgError:
            coefficients = np.zeros(design.shape[1], dtype=np.float64)
            coefficients[0] = float(np.mean(logged))
        return logged - design @ coefficients

    @staticmethod
    def _bin_samples(
        offsets_world: np.ndarray,
        values: np.ndarray,
        resolution_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        cells = np.rint(offsets_world[:, :2] / resolution_m).astype(np.int64)
        unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        sums = np.bincount(inverse, weights=values, minlength=unique.shape[0])
        counts = np.bincount(inverse, minlength=unique.shape[0])
        means = sums / np.maximum(counts, 1)
        finite = np.isfinite(means)
        return unique[finite], means[finite]

    def _correlation_surface(
        self,
        image: np.ndarray,
        image_valid: np.ndarray,
        template: np.ndarray,
        template_mask: np.ndarray,
        minimum_std: float,
        *,
        calculate_rmse: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cv2 = self._cv2
        weights = np.asarray(template_mask, dtype=np.float32)
        sample_count = float(np.sum(weights))
        if sample_count < 2.0:
            shape = (
                image.shape[0] - template.shape[0] + 1,
                image.shape[1] - template.shape[1] + 1,
            )
            return (
                np.full(shape, -np.inf, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
                np.full(shape, np.inf, dtype=np.float32),
            )
        mean = float(np.sum(template * weights) / sample_count)
        centered = (template - mean) * weights
        centered_energy = float(np.sum(centered**2))
        template_std = math.sqrt(centered_energy / sample_count)
        if not np.isfinite(template_std) or template_std < minimum_std:
            shape = (
                image.shape[0] - template.shape[0] + 1,
                image.shape[1] - template.shape[1] + 1,
            )
            return (
                np.full(shape, -np.inf, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
                np.full(shape, np.inf, dtype=np.float32),
            )

        # Shifting both operands by the template mean leaves correlation and
        # absolute residuals unchanged, while avoiding catastrophic float32
        # cancellation for kilometre-scale negative depth datums.
        analysis_image = np.where(image_valid, image - mean, 0.0).astype(
            np.float32, copy=False
        )
        numerator = cv2.matchTemplate(
            analysis_image, centered, cv2.TM_CCORR
        )
        local_sum = cv2.matchTemplate(
            analysis_image, weights, cv2.TM_CCORR
        )
        local_square_sum = cv2.matchTemplate(
            analysis_image * analysis_image, weights, cv2.TM_CCORR
        )
        overlap = cv2.matchTemplate(
            image_valid.astype(np.float32, copy=False), weights, cv2.TM_CCORR
        )
        local_energy = np.maximum(
            local_square_sum - local_sum * local_sum / sample_count,
            0.0,
        )
        denominator = np.sqrt(centered_energy * local_energy)
        score = np.full(numerator.shape, -np.inf, dtype=np.float32)
        required_overlap = self.config.minimum_valid_fraction * sample_count
        usable = (
            (overlap >= required_overlap - 1.0e-3)
            & np.isfinite(numerator)
            & np.isfinite(denominator)
            & (denominator > 1.0e-12)
        )
        score[usable] = numerator[usable] / denominator[usable]
        score[~np.isfinite(score)] = -np.inf
        np.clip(score, -1.0, 1.0, out=score)
        rmse = np.full(score.shape, np.inf, dtype=np.float32)
        if calculate_rmse:
            square_error = np.maximum(
                local_square_sum - 2.0 * numerator + centered_energy,
                0.0,
            )
            rmse[usable] = np.sqrt(square_error[usable] / sample_count)
        return score, overlap / sample_count, rmse

    @staticmethod
    def _make_template(
        cells: np.ndarray,
        values: np.ndarray,
        bounds: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        minimum_x, maximum_x, minimum_y, maximum_y = bounds
        shape = (maximum_y - minimum_y + 1, maximum_x - minimum_x + 1)
        template = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.float32)
        if cells.size:
            rows = cells[:, 1] - minimum_y
            columns = cells[:, 0] - minimum_x
            template[rows, columns] = values.astype(np.float32, copy=False)
            mask[rows, columns] = 1.0
        return template, mask

    def _candidate_is_separate(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> bool:
        distance = float(np.linalg.norm(first[:2] - second[:2]))
        yaw_difference_deg = abs(
            math.degrees(float(wrap_angle(first[2] - second[2])))
        )
        normalized = (
            distance / self.config.basin_translation_separation_m
        ) ** 2 + (
            yaw_difference_deg / self.config.basin_yaw_separation_deg
        ) ** 2
        return normalized >= 1.0 - 1.0e-12

    def _reject(
        self,
        reason: str,
        state: InertialState,
        *,
        candidates: tuple[GlobalTerrainCandidate, ...] = (),
        best_score: float = float("nan"),
        second_score: float = float("nan"),
        gap: float = float("nan"),
        evaluated_yaws: int = 0,
        lattice_candidates: int = 0,
        bathymetry_observation_std_m: float = 0.0,
        backscatter_observation_log_std: float = 0.0,
        bathymetry_observability: float = 0.0,
        backscatter_observability: float = 0.0,
    ) -> GlobalTerrainSearchResult:
        return GlobalTerrainSearchResult(
            accepted=False,
            candidates=candidates,
            rejection_reason=reason,
            best_score=best_score,
            second_basin_score=second_score,
            basin_score_gap=gap,
            evaluated_yaws=evaluated_yaws,
            finite_lattice_candidates=lattice_candidates,
            state_timestamp=float(state.timestamp),
            bathymetry_observation_std_m=bathymetry_observation_std_m,
            backscatter_observation_log_std=backscatter_observation_log_std,
            bathymetry_observability=bathymetry_observability,
            backscatter_observability=backscatter_observability,
        )

    def search(
        self,
        bathymetry_points_body: np.ndarray,
        state: InertialState,
        *,
        backscatter_points_body: np.ndarray | None = None,
        backscatter_values: np.ndarray | None = None,
        yaw_center_rad: float | None = None,
    ) -> GlobalTerrainSearchResult:
        """Search the measured map and return spatially distinct pose proposals."""

        if not isinstance(state, InertialState):
            raise TypeError("state must be an InertialState")
        bathymetry = np.asarray(
            bathymetry_points_body, dtype=np.float64
        ).reshape(-1, 3)
        bathymetry = bathymetry[np.isfinite(bathymetry).all(axis=1)]
        if bathymetry.shape[0] < self.config.minimum_bathymetry_bins:
            return self._reject("insufficient_bathymetry_points", state)

        backscatter_points = np.empty((0, 3), dtype=np.float64)
        backscatter_texture = np.empty(0, dtype=np.float64)
        if backscatter_points_body is not None or backscatter_values is not None:
            if backscatter_points_body is None or backscatter_values is None:
                raise ValueError(
                    "backscatter points and values must be supplied together"
                )
            raw_points = np.asarray(
                backscatter_points_body, dtype=np.float64
            ).reshape(-1, 3)
            raw_values = np.asarray(backscatter_values, dtype=np.float64).reshape(-1)
            if raw_points.shape[0] != raw_values.size:
                raise ValueError("backscatter points and values must have equal length")
            finite = np.isfinite(raw_points).all(axis=1) & np.isfinite(raw_values)
            backscatter_points = raw_points[finite]
            finite_values = raw_values[finite]
            if finite_values.size:
                backscatter_texture = self._detrended_log_backscatter(
                    backscatter_points,
                    finite_values,
                    self.config.backscatter_log_floor,
                    relative_power=uses_relative_power(self.terrain.metadata),
                )
        if self.config.require_backscatter and (
            backscatter_points.shape[0] < self.config.minimum_backscatter_bins
        ):
            return self._reject("insufficient_backscatter_points", state)

        # These are observation-side information indicators, not acceptance
        # gates.  They let the persistent multi-hypothesis tracker reduce the
        # influence of a flat MBES strip or low-texture side-scan strip instead
        # of applying one fixed bathymetry/backscatter weight everywhere.
        bathymetry_observation_std_m = float(
            np.std(bathymetry[:, 2], dtype=np.float64)
        )
        backscatter_observation_log_std = (
            float(np.std(backscatter_texture, dtype=np.float64))
            if backscatter_texture.size
            else 0.0
        )
        bathymetry_observability = float(
            np.clip(
                bathymetry_observation_std_m
                / max(10.0 * self.config.minimum_bathymetry_std_m, 1.0e-12),
                0.0,
                1.0,
            )
        )
        backscatter_observability = float(
            np.clip(
                backscatter_observation_log_std
                / max(10.0 * self.config.minimum_backscatter_log_std, 1.0e-12),
                0.0,
                1.0,
            )
        )

        roll, pitch, state_yaw = rpy_from_rotation(state.rotation)
        yaw_center = state_yaw if yaw_center_rad is None else float(yaw_center_rad)
        if not np.isfinite(yaw_center):
            raise ValueError("yaw_center_rad must be finite")
        yaw_samples = self._yaw_samples(
            yaw_center,
            self.config.yaw_half_window_deg,
            self.config.yaw_step_deg,
        )
        raw_candidates: list[_RawCandidate] = []
        lattice_candidates = 0
        evaluated_yaws = 0

        for yaw in yaw_samples:
            rotation = rotation_from_rpy(float(roll), float(pitch), float(yaw))
            bathymetry_offsets = bathymetry @ rotation.T
            bathymetry_cells, bathymetry_values = self._bin_samples(
                bathymetry_offsets,
                state.position[2] + bathymetry_offsets[:, 2],
                self.resolution_m,
            )
            if bathymetry_cells.shape[0] < self.config.minimum_bathymetry_bins:
                continue

            backscatter_cells = np.empty((0, 2), dtype=np.int64)
            backscatter_binned = np.empty(0, dtype=np.float64)
            if backscatter_points.size:
                backscatter_offsets = backscatter_points @ rotation.T
                backscatter_cells, backscatter_binned = self._bin_samples(
                    backscatter_offsets,
                    backscatter_texture,
                    self.resolution_m,
                )
            if self.config.require_backscatter and (
                backscatter_cells.shape[0] < self.config.minimum_backscatter_bins
            ):
                continue

            all_cells = (
                bathymetry_cells
                if backscatter_cells.size == 0
                else np.vstack((bathymetry_cells, backscatter_cells))
            )
            minimum_x, minimum_y = np.min(all_cells, axis=0)
            maximum_x, maximum_y = np.max(all_cells, axis=0)
            bounds = (
                int(minimum_x),
                int(maximum_x),
                int(minimum_y),
                int(maximum_y),
            )
            if (
                maximum_y - minimum_y + 1 > self._bathymetry.shape[0]
                or maximum_x - minimum_x + 1 > self._bathymetry.shape[1]
            ):
                continue

            bathymetry_template, bathymetry_mask = self._make_template(
                bathymetry_cells, bathymetry_values, bounds
            )
            bathymetry_score, _, bathymetry_rmse = self._correlation_surface(
                self._bathymetry,
                self._bathymetry_valid,
                bathymetry_template,
                bathymetry_mask,
                self.config.minimum_bathymetry_std_m,
                calculate_rmse=True,
            )
            active_weight = 0.0
            joint = np.zeros_like(bathymetry_score)
            jointly_finite = np.ones(bathymetry_score.shape, dtype=bool)
            if self.config.bathymetry_weight > 0.0:
                absolute_sigma_m = (
                    0.5 * self.config.maximum_bathymetry_rmse_m
                )
                absolute_score = np.exp(
                    -0.5 * (bathymetry_rmse / absolute_sigma_m) ** 2
                )
                absolute_weight = self.config.bathymetry_absolute_weight
                bathymetry_evidence = (
                    (1.0 - absolute_weight) * bathymetry_score
                    + absolute_weight * absolute_score
                )
                active_weight += float(self.config.bathymetry_weight)
                jointly_finite &= np.isfinite(bathymetry_score)
                jointly_finite &= (
                    bathymetry_rmse
                    <= self.config.maximum_bathymetry_rmse_m
                )
                joint += self.config.bathymetry_weight * np.where(
                    np.isfinite(bathymetry_evidence),
                    bathymetry_evidence,
                    0.0,
                )
            else:
                bathymetry_evidence = bathymetry_score

            backscatter_score = np.full_like(bathymetry_score, -np.inf)
            if backscatter_cells.size and self._backscatter is not None:
                backscatter_template, backscatter_mask = self._make_template(
                    backscatter_cells, backscatter_binned, bounds
                )
                backscatter_score, _, _ = self._correlation_surface(
                    self._backscatter,
                    self._backscatter_valid,
                    backscatter_template,
                    backscatter_mask,
                    self.config.minimum_backscatter_log_std,
                )
                if self.config.backscatter_weight > 0.0:
                    active_weight += float(self.config.backscatter_weight)
                    jointly_finite &= np.isfinite(backscatter_score)
                    joint += self.config.backscatter_weight * np.where(
                        np.isfinite(backscatter_score), backscatter_score, 0.0
                    )
            if active_weight <= 0.0:
                continue
            joint /= active_weight
            joint[~jointly_finite | ~np.isfinite(joint)] = -np.inf
            finite_indices = np.flatnonzero(np.isfinite(joint))
            lattice_candidates += int(finite_indices.size)
            evaluated_yaws += 1
            if finite_indices.size == 0:
                continue
            keep = min(self.config.candidates_per_yaw, finite_indices.size)
            if keep < finite_indices.size:
                selected_local = np.argpartition(
                    joint.ravel()[finite_indices], -keep
                )[-keep:]
                selected = finite_indices[selected_local]
            else:
                selected = finite_indices
            selected = selected[
                np.argsort(joint.ravel()[selected], kind="stable")[::-1]
            ]
            yaw_offset_deg = math.degrees(float(wrap_angle(yaw - yaw_center)))
            for flat_index in selected:
                top, left = np.unravel_index(int(flat_index), joint.shape)
                center_row = int(top - minimum_y)
                center_column = int(left - minimum_x)
                x = self._coarse_x0_m + center_column * self.resolution_m
                y = self._coarse_y0_m + center_row * self.resolution_m
                raw_candidates.append(
                    _RawCandidate(
                        pose2=np.array([x, y, float(yaw)], dtype=np.float64),
                        score=float(joint[top, left]),
                        bathymetry_score=float(bathymetry_score[top, left]),
                        bathymetry_evidence_score=float(
                            bathymetry_evidence[top, left]
                        ),
                        bathymetry_rmse_m=float(
                            bathymetry_rmse[top, left]
                        ),
                        backscatter_score=(
                            float(backscatter_score[top, left])
                            if np.isfinite(backscatter_score[top, left])
                            else -1.0
                        ),
                        yaw_offset_deg=yaw_offset_deg,
                        bathymetry_bins=int(bathymetry_cells.shape[0]),
                        backscatter_bins=int(backscatter_cells.shape[0]),
                    )
                )

        if not raw_candidates:
            return self._reject(
                "no_finite_global_candidates",
                state,
                evaluated_yaws=evaluated_yaws,
                lattice_candidates=lattice_candidates,
                bathymetry_observation_std_m=bathymetry_observation_std_m,
                backscatter_observation_log_std=(
                    backscatter_observation_log_std
                ),
                bathymetry_observability=bathymetry_observability,
                backscatter_observability=backscatter_observability,
            )
        raw_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        separated: list[_RawCandidate] = []
        # Keep one additional basin so the acceptance gap remains auditable even
        # when only ``maximum_candidates`` proposals are returned to the caller.
        for candidate in raw_candidates:
            if all(
                self._candidate_is_separate(candidate.pose2, previous.pose2)
                for previous in separated
            ):
                separated.append(candidate)
                if len(separated) >= self.config.maximum_candidates + 1:
                    break

        public = tuple(
            GlobalTerrainCandidate(
                pose2=candidate.pose2,
                joint_correlation=candidate.score,
                bathymetry_correlation=candidate.bathymetry_score,
                bathymetry_evidence_score=(
                    candidate.bathymetry_evidence_score
                ),
                bathymetry_rmse_m=candidate.bathymetry_rmse_m,
                backscatter_correlation=candidate.backscatter_score,
                yaw_offset_deg=candidate.yaw_offset_deg,
                bathymetry_bins=candidate.bathymetry_bins,
                backscatter_bins=candidate.backscatter_bins,
            )
            for candidate in separated[: self.config.maximum_candidates]
        )
        best_score = float(separated[0].score)
        second_score = (
            float(separated[1].score) if len(separated) >= 2 else float("-inf")
        )
        gap = (
            best_score - second_score
            if np.isfinite(second_score)
            else float("inf")
        )
        if best_score < self.config.minimum_joint_correlation:
            return self._reject(
                "global_correlation_below_threshold",
                state,
                candidates=public,
                best_score=best_score,
                second_score=second_score,
                gap=gap,
                evaluated_yaws=evaluated_yaws,
                lattice_candidates=lattice_candidates,
                bathymetry_observation_std_m=bathymetry_observation_std_m,
                backscatter_observation_log_std=(
                    backscatter_observation_log_std
                ),
                bathymetry_observability=bathymetry_observability,
                backscatter_observability=backscatter_observability,
            )
        if gap < self.config.minimum_basin_score_gap:
            return self._reject(
                "ambiguous_global_match",
                state,
                candidates=public,
                best_score=best_score,
                second_score=second_score,
                gap=gap,
                evaluated_yaws=evaluated_yaws,
                lattice_candidates=lattice_candidates,
                bathymetry_observation_std_m=bathymetry_observation_std_m,
                backscatter_observation_log_std=(
                    backscatter_observation_log_std
                ),
                bathymetry_observability=bathymetry_observability,
                backscatter_observability=backscatter_observability,
            )
        return GlobalTerrainSearchResult(
            accepted=True,
            candidates=public,
            rejection_reason=None,
            best_score=best_score,
            second_basin_score=second_score,
            basin_score_gap=gap,
            evaluated_yaws=evaluated_yaws,
            finite_lattice_candidates=lattice_candidates,
            state_timestamp=float(state.timestamp),
            bathymetry_observation_std_m=bathymetry_observation_std_m,
            backscatter_observation_log_std=backscatter_observation_log_std,
            bathymetry_observability=bathymetry_observability,
            backscatter_observability=backscatter_observability,
        )
