"""Direct MBES/side-scan map likelihoods for cohort particle filtering.

The terrain matchers in :mod:`oceansim.slam.terrain_matching` deliberately
return a compact Gaussian pose result for graph optimization.  A particle
filter must not use that compact result as its measurement: doing so destroys
the multi-modal likelihood that makes particle filtering useful in flat or
repetitive terrain.  This module evaluates the motion-compensated raw sonar
window independently at every map-frame ``[x, y, yaw]`` particle.

Depth, roll and pitch remain supplied by the local acoustic/inertial ESIKF.
The likelihood therefore estimates only the planar ``map -> odom`` gauge and
does not introduce camera, GNSS, gravity, or DVL scale states.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .esikf import InertialState
from .math3d import rpy_from_rotation
from .real_bathymetry import MeasuredBathymetryGrid
from .terrain_matching import (
    information_with_covariance_floor,
    reporting_covariance_from_information,
)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _uniform_indices(size: int, maximum: int) -> np.ndarray:
    if size <= maximum:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, maximum, dtype=np.int64)


def _remove_linear_trend(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    try:
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    except np.linalg.LinAlgError:
        coefficients = np.zeros(design.shape[1], dtype=np.float64)
        coefficients[0] = float(np.mean(values))
    return values - design @ coefficients


@dataclass(frozen=True)
class TerrainParticleLikelihoodConfig:
    """Calibrated, continuously weighted raw-sonar likelihood policy."""

    bathymetry_measurement_std_m: float = 1.5
    bathymetry_student_t_dof: float = 4.0
    bathymetry_effective_independent_samples: int = 32
    bathymetry_reference_normalized_loss: float = 1.5
    bathymetry_fit_temperature: float = 0.75
    backscatter_effective_independent_samples: int = 16
    backscatter_reference_correlation: float = 0.30
    backscatter_correlation_temperature: float = 0.12
    backscatter_texture_reference_std: float = 0.05
    observation_quality_log_odds_scale: float = 4.0
    observability_reference_information: float = 0.01
    yaw_rank_scale_m: float = 20.0
    radiometric_log_floor: float = 0.01
    maximum_bathymetry_points: int = 768
    maximum_backscatter_points: int = 768
    particle_chunk_size: int = 32
    local_hessian_translation_step_m: float = 0.5
    local_hessian_yaw_step_deg: float = 0.25
    local_measurement_horizontal_std_floor_m: float = 2.5
    local_measurement_yaw_std_floor_deg: float = 1.5
    unobservable_variance_scaled_m2: float = 1.0e6

    def __post_init__(self) -> None:
        positive = (
            "bathymetry_measurement_std_m",
            "bathymetry_student_t_dof",
            "bathymetry_reference_normalized_loss",
            "bathymetry_fit_temperature",
            "backscatter_correlation_temperature",
            "backscatter_texture_reference_std",
            "observation_quality_log_odds_scale",
            "observability_reference_information",
            "yaw_rank_scale_m",
            "radiometric_log_floor",
            "local_hessian_translation_step_m",
            "local_hessian_yaw_step_deg",
            "local_measurement_horizontal_std_floor_m",
            "local_measurement_yaw_std_floor_deg",
            "unobservable_variance_scaled_m2",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.bathymetry_student_t_dof <= 2.0:
            raise ValueError("bathymetry_student_t_dof must exceed two")
        correlation = float(self.backscatter_reference_correlation)
        if not np.isfinite(correlation) or not -1.0 < correlation < 1.0:
            raise ValueError(
                "backscatter_reference_correlation must lie in (-1, 1)"
            )
        object.__setattr__(
            self, "backscatter_reference_correlation", correlation
        )
        for name in (
            "bathymetry_effective_independent_samples",
            "backscatter_effective_independent_samples",
            "maximum_bathymetry_points",
            "maximum_backscatter_points",
            "particle_chunk_size",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class TerrainParticleLikelihoodDiagnostics:
    """Diagnostics for one cohort evaluation, never simulator-truth based."""

    particle_count: int
    bathymetry_points: int
    backscatter_points: int
    bathymetry_observability: float
    backscatter_observability: float
    best_bathymetry_normalized_loss: float
    best_backscatter_correlation: float
    best_bathymetry_valid_fraction: float
    best_backscatter_valid_fraction: float
    log_likelihood_ratio_min: float
    log_likelihood_ratio_max: float
    log_likelihood_ratio_std: float


@dataclass(frozen=True)
class TerrainLikelihoodGaussianApproximation:
    """Current-window-only Laplace approximation inside one PF basin.

    The finite covariance is a reporting/sampling surrogate.  ``information``
    retains the exact local observable subspace and is the quantity intended
    for a factor graph.
    """

    pose2: np.ndarray
    information: np.ndarray
    covariance: np.ndarray
    scaled_information_eigenvalues: np.ndarray
    observable_rank: int
    diagnostics: TerrainParticleLikelihoodDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose2", _readonly(self.pose2).reshape(3))
        object.__setattr__(
            self,
            "information",
            _readonly(self.information).reshape(3, 3),
        )
        object.__setattr__(
            self,
            "covariance",
            _readonly(self.covariance).reshape(3, 3),
        )
        object.__setattr__(
            self,
            "scaled_information_eigenvalues",
            _readonly(self.scaled_information_eigenvalues).reshape(3),
        )


class AcousticTerrainParticleLikelihood:
    """Evaluate one motion-compensated sonar window at arbitrary particles."""

    def __init__(
        self,
        terrain: MeasuredBathymetryGrid,
        reference_state: InertialState,
        bathymetry_points_body: np.ndarray,
        *,
        backscatter_terrain: MeasuredBathymetryGrid | None = None,
        backscatter_points_body: np.ndarray | None = None,
        backscatter_values: np.ndarray | None = None,
        config: TerrainParticleLikelihoodConfig | None = None,
    ) -> None:
        if not isinstance(terrain, MeasuredBathymetryGrid):
            raise TypeError("terrain must be a MeasuredBathymetryGrid")
        if not isinstance(reference_state, InertialState):
            raise TypeError("reference_state must be an InertialState")
        self.terrain = terrain
        if backscatter_terrain is not None and not isinstance(
            backscatter_terrain, MeasuredBathymetryGrid
        ):
            raise TypeError(
                "backscatter_terrain must be a MeasuredBathymetryGrid"
            )
        self.backscatter_terrain = backscatter_terrain or terrain
        self.reference_state = reference_state.copy()
        self.config = config or TerrainParticleLikelihoodConfig()
        if not (
            np.isfinite(self.reference_state.rotation).all()
            and np.isfinite(self.reference_state.position).all()
        ):
            raise ValueError("reference_state pose must be finite")

        bathymetry = np.asarray(
            bathymetry_points_body, dtype=np.float64
        ).reshape(-1, 3)
        bathymetry = bathymetry[np.isfinite(bathymetry).all(axis=1)]
        selected = _uniform_indices(
            bathymetry.shape[0], self.config.maximum_bathymetry_points
        )
        self._bathymetry_points_body = bathymetry[selected]
        self._bathymetry_offsets_world = (
            self._bathymetry_points_body @ self.reference_state.rotation.T
        )

        if (backscatter_points_body is None) != (backscatter_values is None):
            raise ValueError(
                "backscatter points and values must be supplied together"
            )
        backscatter_points = np.empty((0, 3), dtype=np.float64)
        intensity = np.empty(0, dtype=np.float64)
        if backscatter_points_body is not None:
            backscatter_points = np.asarray(
                backscatter_points_body, dtype=np.float64
            ).reshape(-1, 3)
            intensity = np.asarray(
                backscatter_values, dtype=np.float64
            ).reshape(-1)
            if backscatter_points.shape[0] != intensity.size:
                raise ValueError(
                    "backscatter points and values must have equal size"
                )
            valid = (
                np.isfinite(backscatter_points).all(axis=1)
                & np.isfinite(intensity)
                & (intensity > 0.0)
            )
            backscatter_points = backscatter_points[valid]
            intensity = intensity[valid]
            selected = _uniform_indices(
                backscatter_points.shape[0],
                self.config.maximum_backscatter_points,
            )
            backscatter_points = backscatter_points[selected]
            intensity = intensity[selected]
        self._backscatter_points_body = backscatter_points
        self._backscatter_offsets_world = (
            backscatter_points @ self.reference_state.rotation.T
        )
        self._backscatter_values = intensity
        self._backscatter_design = self._radiometric_design(
            backscatter_points
        )
        if intensity.size:
            observed_log = np.log(
                np.clip(
                    intensity,
                    self.config.radiometric_log_floor,
                    None,
                )
            )
            self._backscatter_observed_texture = _remove_linear_trend(
                observed_log, self._backscatter_design
            )
        else:
            self._backscatter_observed_texture = np.empty(
                0, dtype=np.float64
            )
        self._base_yaw = float(
            rpy_from_rotation(self.reference_state.rotation)[2]
        )
        self.last_diagnostics: TerrainParticleLikelihoodDiagnostics | None = (
            None
        )

    @staticmethod
    def _radiometric_design(points_body: np.ndarray) -> np.ndarray:
        if points_body.shape[0] == 0:
            return np.empty((0, 5), dtype=np.float64)
        cross_track = np.abs(points_body[:, 1])
        scale = max(float(np.percentile(cross_track, 95.0)), 1.0)
        normalized_range = np.clip(cross_track / scale, 0.0, 2.0)
        side = np.sign(points_body[:, 1])
        return np.column_stack(
            (
                np.ones(points_body.shape[0], dtype=np.float64),
                side,
                normalized_range,
                normalized_range**2,
                side * normalized_range,
            )
        )

    def _world_coordinates(
        self, particles: np.ndarray, offsets_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        delta_yaw = particles[:, 2] - self._base_yaw
        cosine = np.cos(delta_yaw)[:, None]
        sine = np.sin(delta_yaw)[:, None]
        offset_x = (
            cosine * offsets_world[None, :, 0]
            - sine * offsets_world[None, :, 1]
        )
        offset_y = (
            sine * offsets_world[None, :, 0]
            + cosine * offsets_world[None, :, 1]
        )
        x = particles[:, 0, None] + offset_x
        y = particles[:, 1, None] + offset_y
        z = (
            self.reference_state.position[2]
            + offsets_world[None, :, 2]
        )
        return x, y, z, offset_x, offset_y

    def _bathymetry_scores(
        self, particles: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        count = self._bathymetry_offsets_world.shape[0]
        if count == 0:
            return np.zeros(particles.shape[0]), 0.0, float("nan"), 0.0
        mean_loss = np.full(particles.shape[0], np.inf, dtype=np.float64)
        coverage = np.zeros(particles.shape[0], dtype=np.float64)
        nu = self.config.bathymetry_student_t_dof
        sigma = self.config.bathymetry_measurement_std_m
        for start in range(0, particles.shape[0], self.config.particle_chunk_size):
            stop = min(start + self.config.particle_chunk_size, particles.shape[0])
            x, y, z, _, _ = self._world_coordinates(
                particles[start:stop], self._bathymetry_offsets_world
            )
            predicted = np.asarray(
                self.terrain.seabed_height(x.reshape(-1), y.reshape(-1)),
                dtype=np.float64,
            ).reshape(stop - start, count)
            valid = np.isfinite(predicted)
            residual = z - predicted
            robust_loss = 0.5 * (nu + 1.0) * np.log1p(
                (residual / sigma) ** 2 / nu
            )
            valid_count = np.sum(valid, axis=1)
            coverage[start:stop] = valid_count / float(count)
            summed = np.sum(np.where(valid, robust_loss, 0.0), axis=1)
            mean_loss[start:stop] = np.where(
                valid_count > 0,
                summed / np.maximum(valid_count, 1),
                np.inf,
            )

        finite = np.isfinite(mean_loss)
        if not np.any(finite):
            return np.zeros(particles.shape[0]), 0.0, float("nan"), 0.0
        best_index = int(np.argmin(mean_loss))
        best_loss = float(mean_loss[best_index])
        best_coverage = float(coverage[best_index])
        relative = np.zeros_like(mean_loss)
        evidence_scale = math.sqrt(
            float(self.config.bathymetry_effective_independent_samples)
        )
        relative[finite] = evidence_scale * (
            best_loss - mean_loss[finite]
        )
        relative[finite] += evidence_scale * np.log(
            np.clip(coverage[finite], 1.0e-6, 1.0)
            / max(best_coverage, 1.0e-6)
        )
        relative[~finite] = -10.0 * evidence_scale

        observability = self._bathymetry_observability(
            particles[best_index]
        )
        fit_quality = math.tanh(
            (
                self.config.bathymetry_reference_normalized_loss
                - best_loss
            )
            / self.config.bathymetry_fit_temperature
        )
        base_log_ratio = relative + (
            self.config.observation_quality_log_odds_scale * fit_quality
        )
        return (
            base_log_ratio,
            float(np.clip(observability * best_coverage, 0.0, 1.0)),
            best_loss,
            best_coverage,
        )

    def _bathymetry_observability(self, particle: np.ndarray) -> float:
        if self._bathymetry_offsets_world.shape[0] == 0:
            return 0.0
        x, y, z, offset_x, offset_y = self._world_coordinates(
            particle.reshape(1, 3), self._bathymetry_offsets_world
        )
        predicted = np.asarray(
            self.terrain.seabed_height(x[0], y[0]), dtype=np.float64
        )
        normals = np.asarray(self.terrain.normal(x[0], y[0]), dtype=np.float64)
        valid = (
            np.isfinite(predicted)
            & np.isfinite(normals).all(axis=1)
            & (np.abs(normals[:, 2]) > 1.0e-9)
        )
        if not np.any(valid):
            return 0.0
        gradient_x = -normals[valid, 0] / normals[valid, 2]
        gradient_y = -normals[valid, 1] / normals[valid, 2]
        residual = z[0, valid] - predicted[valid]
        normalized_squared = (
            residual / self.config.bathymetry_measurement_std_m
        ) ** 2
        nu = self.config.bathymetry_student_t_dof
        robust_weight = (nu + 1.0) / (nu + normalized_squared)
        jacobian = np.column_stack(
            (
                -gradient_x,
                -gradient_y,
                gradient_x * offset_y[0, valid]
                - gradient_y * offset_x[0, valid],
            )
        )
        independent_scale = min(
            1.0,
            self.config.bathymetry_effective_independent_samples
            / float(jacobian.shape[0]),
        )
        information = (
            independent_scale
            / self.config.bathymetry_measurement_std_m**2
            * (jacobian.T @ (robust_weight[:, None] * jacobian))
        )
        q_from_scaled = np.diag(
            [1.0, 1.0, 1.0 / self.config.yaw_rank_scale_m]
        )
        scaled = q_from_scaled.T @ information @ q_from_scaled
        try:
            eigenvalues = np.maximum(np.linalg.eigvalsh(scaled), 0.0)
            translation_eigenvalues = np.maximum(
                np.linalg.eigvalsh(information[:2, :2]), 0.0
            )
        except np.linalg.LinAlgError:
            return 0.0
        reference = self.config.observability_reference_information
        translation_fraction = translation_eigenvalues / (
            translation_eigenvalues + reference
        )
        translation_observability = math.sqrt(
            float(np.prod(translation_fraction))
        )
        yaw_fraction = float(eigenvalues[0] / (eigenvalues[0] + reference))
        return float(
            np.clip(
                0.75 * translation_observability + 0.25 * yaw_fraction,
                0.0,
                1.0,
            )
        )

    def _backscatter_scores(
        self, particles: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        count = self._backscatter_offsets_world.shape[0]
        if count < 2 or not self.backscatter_terrain.has_measured_backscatter:
            return np.zeros(particles.shape[0]), 0.0, float("nan"), 0.0
        correlations = np.full(particles.shape[0], np.nan, dtype=np.float64)
        coverage = np.zeros(particles.shape[0], dtype=np.float64)
        map_texture_std = np.zeros(particles.shape[0], dtype=np.float64)
        for start in range(0, particles.shape[0], self.config.particle_chunk_size):
            stop = min(start + self.config.particle_chunk_size, particles.shape[0])
            x, y, _, _, _ = self._world_coordinates(
                particles[start:stop], self._backscatter_offsets_world
            )
            predicted = np.asarray(
                self.backscatter_terrain.measured_reflectivity(
                    x.reshape(-1), y.reshape(-1)
                ),
                dtype=np.float64,
            ).reshape(stop - start, count)
            for local_index in range(stop - start):
                output_index = start + local_index
                valid = np.isfinite(predicted[local_index])
                valid_count = int(np.count_nonzero(valid))
                coverage[output_index] = valid_count / float(count)
                if valid_count < 2:
                    continue
                design = self._backscatter_design[valid]
                map_texture = _remove_linear_trend(
                    np.log(
                        np.clip(
                            predicted[local_index, valid],
                            self.config.radiometric_log_floor,
                            None,
                        )
                    ),
                    design,
                )
                observed_texture = self._backscatter_observed_texture[valid]
                denominator = float(
                    np.linalg.norm(observed_texture)
                    * np.linalg.norm(map_texture)
                )
                if denominator <= 1.0e-12:
                    continue
                correlations[output_index] = float(
                    np.clip(
                        np.dot(observed_texture, map_texture) / denominator,
                        -0.999,
                        0.999,
                    )
                )
                map_texture_std[output_index] = float(
                    np.sqrt(np.mean(map_texture**2))
                )

        finite = np.isfinite(correlations)
        if not np.any(finite):
            return np.zeros(particles.shape[0]), 0.0, float("nan"), 0.0
        best_index = int(np.nanargmax(correlations))
        best_correlation = float(correlations[best_index])
        best_coverage = float(coverage[best_index])
        fisher = np.full_like(correlations, -10.0)
        fisher[finite] = np.arctanh(correlations[finite])
        best_fisher = float(fisher[best_index])
        evidence_scale = math.sqrt(
            max(
                1.0,
                float(
                    self.config.backscatter_effective_independent_samples - 3
                ),
            )
        )
        relative = evidence_scale * (fisher - best_fisher)
        relative[finite] += evidence_scale * np.log(
            np.clip(coverage[finite], 1.0e-6, 1.0)
            / max(best_coverage, 1.0e-6)
        )
        observed_std = float(
            np.sqrt(np.mean(self._backscatter_observed_texture**2))
        )
        texture_reference = self.config.backscatter_texture_reference_std
        texture_observability = (
            observed_std / (observed_std + texture_reference)
            * map_texture_std[best_index]
            / (map_texture_std[best_index] + texture_reference)
        )
        fit_quality = math.tanh(
            (
                best_correlation
                - self.config.backscatter_reference_correlation
            )
            / self.config.backscatter_correlation_temperature
        )
        base_log_ratio = relative + (
            self.config.observation_quality_log_odds_scale * fit_quality
        )
        return (
            base_log_ratio,
            float(np.clip(texture_observability * best_coverage, 0.0, 1.0)),
            best_correlation,
            best_coverage,
        )

    def evaluate(self, particles: np.ndarray) -> np.ndarray:
        poses = np.asarray(particles, dtype=np.float64).reshape(-1, 3)
        if poses.shape[0] == 0:
            raise ValueError("particles must not be empty")
        if not np.isfinite(poses).all():
            raise ValueError("particles must be finite")
        bathymetry, bathymetry_observability, bathymetry_loss, bathymetry_valid = (
            self._bathymetry_scores(poses)
        )
        backscatter, backscatter_observability, backscatter_correlation, backscatter_valid = (
            self._backscatter_scores(poses)
        )
        # A logarithmic opinion pool is conservative for the co-registered
        # MBES and side-scan channels: their weights never sum above one, so
        # shared trajectory/acoustic errors are not counted twice.
        normalization = max(
            1.0,
            bathymetry_observability + backscatter_observability,
        )
        combined = (
            bathymetry_observability * bathymetry
            + backscatter_observability * backscatter
        ) / normalization
        combined = np.nan_to_num(
            combined, nan=0.0, posinf=50.0, neginf=-50.0
        )
        self.last_diagnostics = TerrainParticleLikelihoodDiagnostics(
            particle_count=int(poses.shape[0]),
            bathymetry_points=int(self._bathymetry_points_body.shape[0]),
            backscatter_points=int(self._backscatter_points_body.shape[0]),
            bathymetry_observability=float(bathymetry_observability),
            backscatter_observability=float(backscatter_observability),
            best_bathymetry_normalized_loss=float(bathymetry_loss),
            best_backscatter_correlation=float(backscatter_correlation),
            best_bathymetry_valid_fraction=float(bathymetry_valid),
            best_backscatter_valid_fraction=float(backscatter_valid),
            log_likelihood_ratio_min=float(np.min(combined)),
            log_likelihood_ratio_max=float(np.max(combined)),
            log_likelihood_ratio_std=float(np.std(combined)),
        )
        return _readonly(combined)

    def _refine_pose_with_stages(
        self,
        seed_pose2: np.ndarray,
        stages: tuple[tuple[float, float, int, int], ...],
    ) -> np.ndarray:
        centre = np.asarray(seed_pose2, dtype=np.float64).reshape(3).copy()
        if not np.isfinite(centre).all():
            raise ValueError("seed_pose2 must be finite")
        for translation_step_m, yaw_step_deg, xy_half_steps, yaw_half_steps in stages:
            offsets = []
            for x_index in range(-xy_half_steps, xy_half_steps + 1):
                for y_index in range(-xy_half_steps, xy_half_steps + 1):
                    for yaw_index in range(-yaw_half_steps, yaw_half_steps + 1):
                        offsets.append(
                            (
                                x_index * translation_step_m,
                                y_index * translation_step_m,
                                math.radians(yaw_index * yaw_step_deg),
                            )
                        )
            offsets.sort(
                key=lambda value: (
                    (value[0] / translation_step_m) ** 2
                    + (value[1] / translation_step_m) ** 2
                    + (
                        value[2] / math.radians(yaw_step_deg)
                    )
                    ** 2,
                    abs(value[2]),
                    abs(value[0]) + abs(value[1]),
                    value,
                )
            )
            candidates = centre[None, :] + np.asarray(
                offsets, dtype=np.float64
            )
            candidates[:, 2] = (
                candidates[:, 2] + math.pi
            ) % (2.0 * math.pi) - math.pi
            scores = self.evaluate(candidates)
            centre = candidates[int(np.argmax(scores))].copy()
        return _readonly(centre)

    def refine_pose(self, seed_pose2: np.ndarray) -> np.ndarray:
        """Find a raw-likelihood mode around a new compact-map proposal.

        The search is deterministic and centre-first.  Consequently a fully
        flat likelihood leaves the supplied seed unchanged, while observable
        bathymetry or side-scan texture can move the compact match to the raw
        nonlinear MAP.  No simulator truth, residual gate, or previous
        particle weight enters this estimate.
        """

        return self._refine_pose_with_stages(
            seed_pose2,
            (
                (4.0, 2.0, 3, 4),
                (1.0, 0.5, 2, 2),
                (0.25, 0.125, 2, 2),
            ),
        )

    def refine_pose_in_basin(self, seed_pose2: np.ndarray) -> np.ndarray:
        """Refine the current mode selected by a persistent PF cohort.

        A cohort seed already identifies one data-association basin.  This
        deliberately smaller, centre-first lattice estimates the current
        acquisition's local likelihood maximum without jumping to a distant
        alias.  The seed chooses the basin only; all movement within it is
        driven by the current raw sonar likelihood.
        """

        return self._refine_pose_with_stages(
            seed_pose2,
            (
                (1.0, 0.5, 2, 1),
                (0.25, 0.125, 2, 2),
            ),
        )

    def approximate_local_measurement(
        self, seed_pose2: np.ndarray
    ) -> TerrainLikelihoodGaussianApproximation:
        """Fit a rank-aware Gaussian to this acquisition's local mode.

        The propagated cohort chooses only the data-association basin.  The
        mean and curvature are then computed from the current raw MBES/side-
        scan likelihood, without using recursive particle weights.  A flat
        likelihood therefore returns exactly zero information rather than a
        falsely precise pose factor.
        """

        mode = np.asarray(
            self.refine_pose_in_basin(seed_pose2), dtype=np.float64
        ).reshape(3)
        steps = np.array(
            [
                self.config.local_hessian_translation_step_m,
                self.config.local_hessian_translation_step_m,
                math.radians(self.config.local_hessian_yaw_step_deg),
            ],
            dtype=np.float64,
        )

        offsets: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
        diagonal_indices: list[tuple[int, int]] = []
        for axis in range(3):
            positive = np.zeros(3, dtype=np.float64)
            positive[axis] = steps[axis]
            positive_index = len(offsets)
            offsets.append(positive)
            negative_index = len(offsets)
            offsets.append(-positive)
            diagonal_indices.append((positive_index, negative_index))

        cross_indices: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for first in range(3):
            for second in range(first + 1, 3):
                indices = []
                for first_sign, second_sign in (
                    (1.0, 1.0),
                    (1.0, -1.0),
                    (-1.0, 1.0),
                    (-1.0, -1.0),
                ):
                    offset = np.zeros(3, dtype=np.float64)
                    offset[first] = first_sign * steps[first]
                    offset[second] = second_sign * steps[second]
                    indices.append(len(offsets))
                    offsets.append(offset)
                cross_indices[(first, second)] = tuple(indices)

        candidates = mode[None, :] + np.asarray(offsets, dtype=np.float64)
        candidates[:, 2] = (
            candidates[:, 2] + math.pi
        ) % (2.0 * math.pi) - math.pi
        log_likelihood = np.asarray(
            self.evaluate(candidates), dtype=np.float64
        )
        diagnostics = self.last_diagnostics
        if diagnostics is None:
            raise RuntimeError("raw likelihood diagnostics were not produced")

        hessian = np.zeros((3, 3), dtype=np.float64)
        centre_value = float(log_likelihood[0])
        for axis, (positive_index, negative_index) in enumerate(
            diagonal_indices
        ):
            hessian[axis, axis] = (
                float(log_likelihood[positive_index])
                - 2.0 * centre_value
                + float(log_likelihood[negative_index])
            ) / steps[axis] ** 2
        for (first, second), indices in cross_indices.items():
            pp, pn, np_index, nn = indices
            value = (
                float(log_likelihood[pp])
                - float(log_likelihood[pn])
                - float(log_likelihood[np_index])
                + float(log_likelihood[nn])
            ) / (4.0 * steps[first] * steps[second])
            hessian[first, second] = value
            hessian[second, first] = value

        raw_information = -0.5 * (hessian + hessian.T)
        q_from_scaled = np.diag(
            [1.0, 1.0, 1.0 / self.config.yaw_rank_scale_m]
        )
        scaled_information = (
            q_from_scaled.T @ raw_information @ q_from_scaled
        )
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (scaled_information + scaled_information.T)
            )
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(3, dtype=np.float64)
            eigenvectors = np.eye(3, dtype=np.float64)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        numerical_tolerance = 1.0e-10 * scale
        retained_eigenvalues = np.where(
            eigenvalues > numerical_tolerance, eigenvalues, 0.0
        )
        projected_scaled_information = (
            eigenvectors * retained_eigenvalues
        ) @ eigenvectors.T
        scaled_from_q = np.diag(
            [1.0, 1.0, self.config.yaw_rank_scale_m]
        )
        projected_information = (
            scaled_from_q.T
            @ projected_scaled_information
            @ scaled_from_q
        )
        model_covariance = np.diag(
            [
                self.config.local_measurement_horizontal_std_floor_m**2,
                self.config.local_measurement_horizontal_std_floor_m**2,
                math.radians(
                    self.config.local_measurement_yaw_std_floor_deg
                )
                ** 2,
            ]
        )
        information = information_with_covariance_floor(
            projected_information, model_covariance
        )

        final_scaled_information = (
            q_from_scaled.T @ information @ q_from_scaled
        )
        try:
            final_eigenvalues, final_eigenvectors = np.linalg.eigh(
                0.5
                * (final_scaled_information + final_scaled_information.T)
            )
        except np.linalg.LinAlgError:
            final_eigenvalues = np.zeros(3, dtype=np.float64)
            final_eigenvectors = np.eye(3, dtype=np.float64)
        final_eigenvalues = np.maximum(final_eigenvalues, 0.0)
        final_scale = max(float(np.max(final_eigenvalues)), 1.0)
        observable = final_eigenvalues > 1.0e-10 * final_scale
        covariance = reporting_covariance_from_information(
            information,
            yaw_scale_m=self.config.yaw_rank_scale_m,
            unobservable_variance_scaled_m2=(
                self.config.unobservable_variance_scaled_m2
            ),
        )

        return TerrainLikelihoodGaussianApproximation(
            pose2=mode,
            information=information,
            covariance=covariance,
            scaled_information_eigenvalues=final_eigenvalues,
            observable_rank=int(np.count_nonzero(observable)),
            diagnostics=diagnostics,
        )

    def __call__(self, particles: np.ndarray) -> np.ndarray:
        return self.evaluate(particles)
