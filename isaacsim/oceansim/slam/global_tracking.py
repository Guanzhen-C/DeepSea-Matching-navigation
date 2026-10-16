"""Persistent multi-hypothesis tracking for whole-map sonar relocalization.

The coarse whole-map matcher can legitimately return several spatially
separated peaks in flat or repetitive terrain.  Collapsing those peaks to a
single pose before another independent strip arrives is unsafe: a repeatable
but false terrain alias then looks like a precise absolute fix.  This module
keeps the alternatives as persistent ``cohorts`` and accumulates evidence
over time, following the cohort-filtering idea used by Global UTAN.

The operational update follows the established PMF/PF terrain-navigation
pattern used by systems such as MBARI TRN: odometry propagates a map-frame
posterior and raw multibeam residuals update it against a prior bathymetric
map.  The implementation is independent OceanSim code; no GPL MB-System
source is included.

This is deliberately a policy and probability layer, not another matcher.
Each :class:`GlobalPoseEvidence` is an absolute map pose proposed by MBES and
side-scan processing. Every cohort contains a small SE(2) particle subset,
but all subsets share one raw-likelihood normalization and therefore form one
multimodal point-mass posterior. Those particles are propagated by the
relative motion of the local acoustic/inertial odometry and updated without
simulator truth. A separate null hypothesis represents the scientifically
important outcome "none of the proposals is a valid absolute fix".

Only acquisition windows that do not reuse pings count as independent
evidence.  Overlapping rolling windows may refine a cohort, but their
likelihood and covariance contributions are discounted so they cannot create
false confidence merely by processing the same sonar data repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Iterable

import numpy as np


def select_terrain_navigation_output_source(
    *,
    global_active_cohorts: int,
    ordinary_active_cohorts: int,
    global_confirmed_map_available: bool,
    ordinary_confirmed_map_available: bool,
) -> str:
    """Select the validated online pose source without inspecting truth.

    Independently confirmed whole-map recovery hypotheses have priority.
    A confirmed ordinary terrain cohort may provide a temporary output while
    several ordinary cohorts remain active, but an unconfirmed particle MAP
    is always a shadow state. This prevents a weakly separated mode in flat or
    repetitive terrain from moving ``map->odom`` merely because it currently
    outranks the explicit null hypothesis. GTSAM supplies the public pose when
    neither tracker has a confirmed publishable cohort.
    """
    counts = {
        "global_active_cohorts": global_active_cohorts,
        "ordinary_active_cohorts": ordinary_active_cohorts,
    }
    for name, value in counts.items():
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    availability = {
        "global_confirmed_map_available": global_confirmed_map_available,
        "ordinary_confirmed_map_available": (
            ordinary_confirmed_map_available
        ),
    }
    for name, value in availability.items():
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean")

    if int(global_active_cohorts) > 0 and bool(
        global_confirmed_map_available
    ):
        return "global_terrain_particle_map"
    if int(ordinary_active_cohorts) > 1 and bool(
        ordinary_confirmed_map_available
    ):
        return "ordinary_terrain_particle_map"
    return "pose_graph"


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _pose2(values: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    try:
        pose = np.asarray(values, dtype=np.float64).reshape(3).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} must contain three finite values")
    pose[2] = _wrap_angle(float(pose[2]))
    return pose


def _readonly(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(shape).copy()
    result.setflags(write=False)
    return result


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("logsumexp requires at least one value")
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _observable_information_rank(information: np.ndarray) -> int:
    """Return the numerical rank of a positive-semidefinite 3-D factor."""

    matrix = np.asarray(information, dtype=np.float64).reshape(3, 3)
    matrix = 0.5 * (matrix + matrix.T)
    if not np.isfinite(matrix).all():
        return 0
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return 0
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) < -1.0e-10 * scale:
        return 0
    return int(np.count_nonzero(eigenvalues > 1.0e-10 * scale))


def _weighted_pose_statistics(
    particles: np.ndarray, log_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the SE(2) weighted mean and covariance of one particle cohort."""

    samples = np.asarray(particles, dtype=np.float64).reshape(-1, 3)
    normalized_log_weights = (
        np.asarray(log_weights, dtype=np.float64).reshape(-1)
        - _logsumexp(log_weights)
    )
    if samples.shape[0] != normalized_log_weights.size:
        raise ValueError("particle and weight counts must agree")
    weights = np.exp(normalized_log_weights)
    sine = float(np.sum(weights * np.sin(samples[:, 2])))
    cosine = float(np.sum(weights * np.cos(samples[:, 2])))
    # A vanishing circular resultant is a genuinely unobservable yaw.  Keep a
    # deterministic representative while the covariance records the spread.
    yaw = math.atan2(sine, cosine) if math.hypot(sine, cosine) > 1.0e-12 else 0.0
    mean = np.array(
        [
            float(np.sum(weights * samples[:, 0])),
            float(np.sum(weights * samples[:, 1])),
            _wrap_angle(yaw),
        ],
        dtype=np.float64,
    )
    residuals = samples - mean
    residuals[:, 2] = (
        residuals[:, 2] + math.pi
    ) % (2.0 * math.pi) - math.pi
    covariance = (residuals * weights[:, None]).T @ residuals
    covariance = 0.5 * (covariance + covariance.T)
    covariance += np.diag([1.0e-9, 1.0e-9, 1.0e-12])
    return mean, covariance


def _compose_pose2(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cosine = math.cos(float(first[2]))
    sine = math.sin(float(first[2]))
    return np.array(
        [
            first[0] + cosine * second[0] - sine * second[1],
            first[1] + sine * second[0] + cosine * second[1],
            _wrap_angle(float(first[2] + second[2])),
        ],
        dtype=np.float64,
    )


def _between_pose2(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = second[:2] - first[:2]
    cosine = math.cos(float(first[2]))
    sine = math.sin(float(first[2]))
    return np.array(
        [
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
            _wrap_angle(float(second[2] - first[2])),
        ],
        dtype=np.float64,
    )


def _map_to_odom(map_pose2: np.ndarray, odom_pose2: np.ndarray) -> np.ndarray:
    """Return ``T_map_odom = T_map_body * inverse(T_odom_body)``."""

    yaw = _wrap_angle(float(map_pose2[2] - odom_pose2[2]))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotated_odom = np.array(
        [
            cosine * odom_pose2[0] - sine * odom_pose2[1],
            sine * odom_pose2[0] + cosine * odom_pose2[1],
        ],
        dtype=np.float64,
    )
    return np.array(
        [
            map_pose2[0] - rotated_odom[0],
            map_pose2[1] - rotated_odom[1],
            yaw,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class GlobalPoseEvidence:
    """One absolute-pose proposal and its modality-specific evidence.

    Scores are normalized correlation-like values in ``[-1, 1]``.  The two
    observability values lie in ``[0, 1]`` and determine how much each modality
    contributes on this particular patch; they are not fixed fusion weights.
    ``geometry_score`` is optional because the coarse appearance search may
    precede three-dimensional registration.
    """

    map_pose2: np.ndarray
    bathymetry_score: float
    backscatter_score: float
    bathymetry_observability: float = 1.0
    backscatter_observability: float = 1.0
    geometry_score: float | None = None
    cross_modal_consistent: bool = True
    position_std_m: float = 20.0
    yaw_std_deg: float = 5.0
    patch_start_timestamp_s: float = 0.0
    patch_end_timestamp_s: float = 0.0
    patch_start_observation_id: int = 0
    patch_end_observation_id: int = 0
    source_index: int = 0
    covariance_matrix: np.ndarray | None = None
    # A proposal refined from a persistent cohort's propagated particles has
    # deterministic provenance.  Keeping that identity prevents a broad,
    # distant whole-map proposal from stealing the cohort merely because its
    # inflated covariance produces a smaller normalized innovation.  ``None``
    # denotes an unassigned whole-map proposal that remains free to associate
    # with, or create, a cohort.
    target_cohort_id: int | None = None
    # ``covariance_matrix`` may use a large finite variance to report an
    # unobservable direction.  The optional information matrix preserves the
    # exact rank for graph handoff instead of recovering it with a numerically
    # fragile inverse of that reporting surrogate.
    information_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "map_pose2", _readonly(_pose2(self.map_pose2, "map_pose2"), (3,))
        )
        for name in ("bathymetry_score", "backscatter_score"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [-1, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "bathymetry_observability",
            "backscatter_observability",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        if self.geometry_score is not None:
            geometry = float(self.geometry_score)
            if not np.isfinite(geometry) or not 0.0 <= geometry <= 1.0:
                raise ValueError("geometry_score must lie in [0, 1] or be None")
            object.__setattr__(self, "geometry_score", geometry)
        if not isinstance(self.cross_modal_consistent, (bool, np.bool_)):
            raise ValueError("cross_modal_consistent must be boolean")
        object.__setattr__(
            self, "cross_modal_consistent", bool(self.cross_modal_consistent)
        )
        for name in ("position_std_m", "yaw_std_deg"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        start_time = float(self.patch_start_timestamp_s)
        end_time = float(self.patch_end_timestamp_s)
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError("patch timestamps must be finite")
        if start_time > end_time:
            raise ValueError("patch timestamps must be ordered")
        object.__setattr__(self, "patch_start_timestamp_s", start_time)
        object.__setattr__(self, "patch_end_timestamp_s", end_time)
        identifiers: list[int] = []
        for name in ("patch_start_observation_id", "patch_end_observation_id"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError("patch observation IDs must be nonnegative integers")
            identifiers.append(int(value))
        if identifiers[0] > identifiers[1]:
            raise ValueError("patch observation IDs must be ordered")
        object.__setattr__(self, "patch_start_observation_id", identifiers[0])
        object.__setattr__(self, "patch_end_observation_id", identifiers[1])
        if (
            isinstance(self.source_index, (bool, np.bool_))
            or int(self.source_index) != self.source_index
            or int(self.source_index) < 0
        ):
            raise ValueError("source_index must be a nonnegative integer")
        object.__setattr__(self, "source_index", int(self.source_index))
        if self.target_cohort_id is not None:
            if (
                isinstance(self.target_cohort_id, (bool, np.bool_))
                or int(self.target_cohort_id) != self.target_cohort_id
                or int(self.target_cohort_id) < 1
            ):
                raise ValueError("target_cohort_id must be a positive integer or None")
            object.__setattr__(
                self, "target_cohort_id", int(self.target_cohort_id)
            )
        if self.covariance_matrix is not None:
            covariance = np.asarray(
                self.covariance_matrix, dtype=np.float64
            ).reshape(3, 3)
            covariance = 0.5 * (covariance + covariance.T)
            if not np.isfinite(covariance).all():
                raise ValueError("covariance_matrix must be finite")
            try:
                minimum_eigenvalue = float(
                    np.linalg.eigvalsh(covariance)[0]
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "covariance_matrix eigendecomposition failed"
                ) from exc
            if minimum_eigenvalue <= 0.0:
                raise ValueError(
                    "covariance_matrix must be positive definite"
                )
            object.__setattr__(
                self, "covariance_matrix", _readonly(covariance, (3, 3))
            )
        if self.information_matrix is not None:
            information = np.asarray(
                self.information_matrix, dtype=np.float64
            ).reshape(3, 3)
            information = 0.5 * (information + information.T)
            if not np.isfinite(information).all():
                raise ValueError("information_matrix must be finite")
            try:
                eigenvalues = np.linalg.eigvalsh(information)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "information_matrix eigendecomposition failed"
                ) from exc
            scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
            if float(eigenvalues[0]) < -1.0e-10 * scale:
                raise ValueError(
                    "information_matrix must be positive semidefinite"
                )
            object.__setattr__(
                self, "information_matrix", _readonly(information, (3, 3))
            )

    @property
    def covariance(self) -> np.ndarray:
        if self.covariance_matrix is not None:
            return self.covariance_matrix.copy()
        yaw_std = math.radians(self.yaw_std_deg)
        return np.diag(
            [self.position_std_m**2, self.position_std_m**2, yaw_std**2]
        )

    @property
    def information(self) -> np.ndarray:
        if self.information_matrix is not None:
            return self.information_matrix.copy()
        return np.linalg.inv(self.covariance)


@dataclass(frozen=True)
class ParticleLikelihoodObservation:
    """One raw, independent sonar window evaluated at arbitrary particles.

    ``evaluator`` receives an ``N x 3`` array of current map-frame vehicle
    poses and returns ``N`` log likelihood ratios against the invalid/null
    observation model.  Keeping the callback at this probability-layer
    boundary lets MBES and side-scan retain their native nonlinear map
    likelihood instead of first collapsing the window to one Gaussian pose.
    """

    evaluator: Callable[[np.ndarray], np.ndarray]
    patch_start_timestamp_s: float
    patch_end_timestamp_s: float
    patch_start_observation_id: int
    patch_end_observation_id: int

    def __post_init__(self) -> None:
        if not callable(self.evaluator):
            raise ValueError("evaluator must be callable")
        start_time = float(self.patch_start_timestamp_s)
        end_time = float(self.patch_end_timestamp_s)
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError("patch timestamps must be finite")
        if start_time > end_time:
            raise ValueError("patch timestamps must be ordered")
        object.__setattr__(self, "patch_start_timestamp_s", start_time)
        object.__setattr__(self, "patch_end_timestamp_s", end_time)
        identifiers: list[int] = []
        for name in (
            "patch_start_observation_id",
            "patch_end_observation_id",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError(
                    "patch observation IDs must be nonnegative integers"
                )
            identifiers.append(int(value))
        if identifiers[0] > identifiers[1]:
            raise ValueError("patch observation IDs must be ordered")
        object.__setattr__(
            self, "patch_start_observation_id", identifiers[0]
        )
        object.__setattr__(
            self, "patch_end_observation_id", identifiers[1]
        )

    def evaluate(self, particles: np.ndarray) -> np.ndarray:
        samples = np.asarray(particles, dtype=np.float64).reshape(-1, 3)
        values = np.asarray(self.evaluator(samples), dtype=np.float64).reshape(
            -1
        )
        if values.size != samples.shape[0]:
            raise ValueError(
                "particle likelihood evaluator must return one value per "
                "particle"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                "particle likelihood evaluator must return finite values"
            )
        return values


@dataclass(frozen=True)
class GlobalCohortTrackerConfig:
    """Statistical policy for persistent global-pose cohorts."""

    maximum_cohorts: int = 48
    particles_per_cohort: int = 256
    random_seed: int = 31
    resample_effective_sample_fraction: float = 0.50
    resample_kernel_bandwidth: float = 0.15
    association_max_nis: float = 16.27  # 99.9% chi-square quantile, 3 DoF.
    association_max_position_m: float = 120.0
    association_max_yaw_deg: float = 12.0
    process_translation_std_per_m: float = 0.004
    process_translation_std_per_s: float = 0.01
    process_yaw_std_deg_per_m: float = 0.0008
    process_yaw_std_deg_per_s: float = 0.003
    # Hidden-Markov validity dynamics prevent a once-successful cohort from
    # becoming immortal.  Neutral/flat observations therefore move posterior
    # mass back toward the explicit null state instead of letting confidence
    # monotonically collapse to one.
    cohort_survival_probability: float = 0.97
    null_survival_probability: float = 0.98
    # Per-observation Bayesian change-point mass. With a raw particle update
    # it is drawn from the complete prior (augmented-MCL recovery), not only
    # from p(null), so a previously confident terrain alias is not immortal.
    birth_probability: float = 0.08
    missed_detection_likelihood: float = 0.35
    overlapping_evidence_scale: float = 0.15
    bathymetry_reference_score: float = 0.30
    backscatter_reference_score: float = 0.30
    geometry_reference_score: float = 0.50
    score_temperature: float = 0.12
    maximum_observation_log_likelihood_ratio: float = 4.0
    inconsistent_log_likelihood_ratio: float = -6.0
    minimum_total_evidence: int = 3
    minimum_independent_evidence: int = 3
    confirmation_probability: float = 0.97
    confirmation_odds_over_second: float = 20.0
    confirmation_max_position_std_m: float = 25.0
    confirmation_max_yaw_std_deg: float = 6.0
    # Distinct whole-map proposals can collapse onto the same metric basin
    # after 3D/side-scan refinement.  They share one acquisition and must not
    # create duplicate cohorts or count as independent evidence.
    same_observation_nms_position_m: float = 10.0
    same_observation_nms_yaw_deg: float = 2.0
    maximum_missed_observations: int = 8
    minimum_retained_probability: float = 1.0e-6

    def __post_init__(self) -> None:
        integer_names = (
            "maximum_cohorts",
            "particles_per_cohort",
            "minimum_total_evidence",
            "minimum_independent_evidence",
            "maximum_missed_observations",
        )
        for name in integer_names:
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if (
            isinstance(self.random_seed, (bool, np.bool_))
            or int(self.random_seed) != self.random_seed
            or int(self.random_seed) < 0
        ):
            raise ValueError("random_seed must be a nonnegative integer")
        object.__setattr__(self, "random_seed", int(self.random_seed))
        positive_names = (
            "association_max_nis",
            "association_max_position_m",
            "association_max_yaw_deg",
            "score_temperature",
            "maximum_observation_log_likelihood_ratio",
            "confirmation_odds_over_second",
            "confirmation_max_position_std_m",
            "confirmation_max_yaw_std_deg",
            "same_observation_nms_position_m",
            "same_observation_nms_yaw_deg",
        )
        for name in positive_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        nonnegative_names = (
            "process_translation_std_per_m",
            "process_translation_std_per_s",
            "process_yaw_std_deg_per_m",
            "process_yaw_std_deg_per_s",
        )
        for name in nonnegative_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        unit_interval_names = (
            "birth_probability",
            "missed_detection_likelihood",
            "overlapping_evidence_scale",
            "confirmation_probability",
            "resample_effective_sample_fraction",
            "resample_kernel_bandwidth",
            "cohort_survival_probability",
            "null_survival_probability",
        )
        for name in unit_interval_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
            object.__setattr__(self, name, value)
        for name in (
            "bathymetry_reference_score",
            "backscatter_reference_score",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [-1, 1]")
            object.__setattr__(self, name, value)
        geometry_reference = float(self.geometry_reference_score)
        if not np.isfinite(geometry_reference) or not 0.0 <= geometry_reference <= 1.0:
            raise ValueError("geometry_reference_score must lie in [0, 1]")
        object.__setattr__(self, "geometry_reference_score", geometry_reference)
        inconsistent = float(self.inconsistent_log_likelihood_ratio)
        if not np.isfinite(inconsistent) or inconsistent >= 0.0:
            raise ValueError(
                "inconsistent_log_likelihood_ratio must be finite and negative"
            )
        object.__setattr__(self, "inconsistent_log_likelihood_ratio", inconsistent)
        retained = float(self.minimum_retained_probability)
        if not np.isfinite(retained) or not 0.0 < retained < 1.0:
            raise ValueError("minimum_retained_probability must lie in (0, 1)")
        object.__setattr__(self, "minimum_retained_probability", retained)
        if self.minimum_independent_evidence > self.minimum_total_evidence:
            raise ValueError(
                "minimum_independent_evidence must not exceed minimum_total_evidence"
            )


@dataclass(frozen=True)
class GlobalCohortSnapshot:
    cohort_id: int
    map_pose2: np.ndarray
    map_to_odom_pose2: np.ndarray
    covariance: np.ndarray
    probability: float
    total_evidence: int
    independent_evidence: int
    consecutive_misses: int
    last_source_index: int
    particle_count: int
    effective_sample_size: float
    resampling_events: int
    last_measurement_pose2: np.ndarray
    last_measurement_covariance: np.ndarray
    last_measurement_information: np.ndarray
    last_measurement_observable_rank: int
    last_observation_log_likelihood_ratio: float
    last_update_was_independent: bool
    last_measurement_is_graph_ready: bool
    last_observation_novel_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_pose2", _readonly(self.map_pose2, (3,)))
        object.__setattr__(
            self, "map_to_odom_pose2", _readonly(self.map_to_odom_pose2, (3,))
        )
        object.__setattr__(self, "covariance", _readonly(self.covariance, (3, 3)))
        object.__setattr__(
            self,
            "last_measurement_pose2",
            _readonly(self.last_measurement_pose2, (3,)),
        )
        object.__setattr__(
            self,
            "last_measurement_covariance",
            _readonly(self.last_measurement_covariance, (3, 3)),
        )
        object.__setattr__(
            self,
            "last_measurement_information",
            _readonly(self.last_measurement_information, (3, 3)),
        )
        rank = int(self.last_measurement_observable_rank)
        if not 0 <= rank <= 3:
            raise ValueError(
                "last_measurement_observable_rank must lie in [0, 3]"
            )
        object.__setattr__(self, "last_measurement_observable_rank", rank)


@dataclass(frozen=True)
class GlobalTrackingDecision:
    """Normalized posterior and optional confirmed absolute pose."""

    status: str
    confirmed: bool
    null_probability: float
    best_probability: float
    odds_over_second: float
    cohorts: tuple[GlobalCohortSnapshot, ...]
    confirmed_cohort: GlobalCohortSnapshot | None = None
    confirmed_with_new_independent_evidence: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"no_fix", "tracking", "confirmed"}:
            raise ValueError("unsupported global tracking status")
        if self.confirmed != (self.confirmed_cohort is not None):
            raise ValueError("confirmed flag and confirmed_cohort disagree")
        if self.confirmed != (self.status == "confirmed"):
            raise ValueError("confirmed status is inconsistent")
        if self.confirmed_with_new_independent_evidence and not self.confirmed:
            raise ValueError(
                "new-independent confirmation requires a confirmed cohort"
            )
        object.__setattr__(self, "cohorts", tuple(self.cohorts))


@dataclass
class _Cohort:
    cohort_id: int
    particles: np.ndarray
    log_weights: np.ndarray
    map_pose2: np.ndarray
    covariance: np.ndarray
    log_probability: float
    total_evidence: int
    independent_evidence: int
    consecutive_misses: int
    last_independent_patch_end_timestamp_s: float
    last_independent_patch_end_observation_id: int
    last_source_index: int
    last_effective_sample_size: float
    resampling_events: int
    last_measurement_pose2: np.ndarray
    last_measurement_covariance: np.ndarray
    last_measurement_information: np.ndarray
    last_measurement_observable_rank: int
    last_observation_log_likelihood_ratio: float
    last_update_was_independent: bool
    last_measurement_is_graph_ready: bool
    last_observation_novel_fraction: float
    last_processed_patch_end_timestamp_s: float
    last_processed_patch_end_observation_id: int


@dataclass
class GlobalCohortTracker:
    """Maintain and confirm globally separated sonar pose hypotheses."""

    config: GlobalCohortTrackerConfig = field(
        default_factory=GlobalCohortTrackerConfig
    )
    updates: int = field(default=0, init=False)
    cohorts_created: int = field(default=0, init=False)
    cohorts_pruned: int = field(default=0, init=False)
    confirmations: int = field(default=0, init=False)
    ambiguous_updates: int = field(default=0, init=False)
    duplicate_evidences_suppressed: int = field(default=0, init=False)
    particle_measurement_updates: int = field(default=0, init=False)
    particle_resampling_events: int = field(default=0, init=False)
    direct_map_likelihood_updates: int = field(default=0, init=False)
    joint_particle_likelihood_updates: int = field(default=0, init=False)
    candidate_birth_mass_allocations: int = field(default=0, init=False)
    posterior_mode_switches: int = field(default=0, init=False)
    hypothesis_transition_updates: int = field(default=0, init=False)

    _cohorts: list[_Cohort] = field(default_factory=list, init=False, repr=False)
    _null_log_probability: float = field(default=0.0, init=False, repr=False)
    _previous_timestamp_s: float | None = field(default=None, init=False, repr=False)
    _previous_raw_pose2: np.ndarray | None = field(default=None, init=False, repr=False)
    _current_raw_pose2: np.ndarray | None = field(default=None, init=False, repr=False)
    _next_cohort_id: int = field(default=1, init=False, repr=False)
    _last_leading_cohort_id: int | None = field(
        default=None, init=False, repr=False
    )
    _last_decision: GlobalTrackingDecision | None = field(
        default=None, init=False, repr=False
    )
    _last_joint_log_likelihood_span: float = field(
        default=0.0, init=False, repr=False
    )
    _last_joint_log_likelihood_std: float = field(
        default=0.0, init=False, repr=False
    )
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, GlobalCohortTrackerConfig):
            raise TypeError("config must be a GlobalCohortTrackerConfig")
        self._rng = np.random.default_rng(self.config.random_seed)

    @property
    def null_probability(self) -> float:
        return float(math.exp(self._null_log_probability))

    @property
    def active_cohort_count(self) -> int:
        """Number of currently retained spatial posterior modes."""

        return len(self._cohorts)

    def prediction_seeds(
        self, raw_pose2: np.ndarray
    ) -> tuple[tuple[int, np.ndarray], ...]:
        """Predict current map poses without mutating the particle filter.

        These poses are optimizer seeds for locating the *current* raw-sonar
        likelihood mode associated with each persistent cohort.  They carry no
        measurement weight themselves and therefore cannot be inserted into a
        graph as observations.
        """

        raw = _pose2(raw_pose2, "raw_pose2")
        if self._previous_raw_pose2 is None:
            return ()
        relative = _between_pose2(self._previous_raw_pose2, raw)
        predictions = []
        for cohort in self._cohorts:
            particles = cohort.particles.copy()
            cosine = np.cos(particles[:, 2])
            sine = np.sin(particles[:, 2])
            particles[:, 0] += (
                cosine * relative[0] - sine * relative[1]
            )
            particles[:, 1] += (
                sine * relative[0] + cosine * relative[1]
            )
            particles[:, 2] = (
                particles[:, 2] + relative[2] + math.pi
            ) % (2.0 * math.pi) - math.pi
            prediction, _ = _weighted_pose_statistics(
                particles, cohort.log_weights
            )
            prediction.setflags(write=False)
            predictions.append((int(cohort.cohort_id), prediction))
        return tuple(predictions)

    def maximum_a_posteriori_pose(
        self, raw_pose2: np.ndarray
    ) -> np.ndarray | None:
        """Return the current terrain-navigation MAP pose without mutation.

        The recursive particle posterior is a legitimate online navigation
        output even while competing spatial modes have not yet accumulated
        enough independent evidence to authorize a factor-graph prior.  This
        method therefore returns the propagated leading cohort whenever it is
        more probable than the explicit invalid/null hypothesis.  It does not
        imply graph confirmation and it never writes into the local odometry
        state.

        ``None`` means that the invalid/null hypothesis is at least as likely
        as every spatial hypothesis, so a caller should retain its previous
        map-frame source (normally the pose graph).
        """

        if not self._cohorts:
            return None
        leading = self._cohorts[0]
        if leading.log_probability <= self._null_log_probability:
            return None
        predictions = self.prediction_seeds(raw_pose2)
        if not predictions:
            return None
        cohort_id, pose = predictions[0]
        if cohort_id != leading.cohort_id:
            raise RuntimeError("leading terrain cohort order is inconsistent")
        result = np.asarray(pose, dtype=np.float64).reshape(3).copy()
        result.setflags(write=False)
        return result

    def confirmed_maximum_a_posteriori_pose(
        self, raw_pose2: np.ndarray
    ) -> np.ndarray | None:
        """Return the leading MAP pose only while it is independently confirmed.

        Whole-map relocalization proposals are deliberately allowed to remain
        multimodal and to outrank the explicit null hypothesis while they
        accumulate evidence.  That makes them useful tracker states, but not
        yet safe navigation outputs: a repetitive terrain alias can otherwise
        move ``map->odom`` before the cohort has passed the same independent-
        evidence, probability, odds, and covariance checks required for a
        graph prior.

        This accessor reuses the most recent immutable tracking decision and
        propagates its confirmed leading cohort with the current raw odometry
        increment. Both whole-map recovery and ordinary terrain trackers must
        use this accessor before allowing a particle state to affect the
        published navigation pose. The provisional MAP remains available via
        :meth:`maximum_a_posteriori_pose` for diagnostics and continued
        hypothesis tracking only.
        """

        decision = self._last_decision
        if decision is None or decision.confirmed_cohort is None:
            return None
        if not self._cohorts:
            return None
        confirmed_cohort_id = decision.confirmed_cohort.cohort_id
        if self._cohorts[0].cohort_id != confirmed_cohort_id:
            raise RuntimeError(
                "confirmed terrain cohort is not the leading posterior mode"
            )
        predictions = self.prediction_seeds(raw_pose2)
        if not predictions:
            return None
        cohort_id, pose = predictions[0]
        if cohort_id != confirmed_cohort_id:
            raise RuntimeError(
                "confirmed terrain cohort prediction is inconsistent"
            )
        result = np.asarray(pose, dtype=np.float64).reshape(3).copy()
        result.setflags(write=False)
        return result

    @staticmethod
    def _residual(reference: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        residual = np.asarray(measurement - reference, dtype=np.float64)
        residual[2] = _wrap_angle(float(residual[2]))
        return residual

    def _draw_centered_gaussian(
        self, covariance: np.ndarray, count: int
    ) -> np.ndarray:
        """Draw deterministic-seed, zero-sample-mean Gaussian perturbations."""

        symmetric = 0.5 * (
            np.asarray(covariance, dtype=np.float64).reshape(3, 3)
            + np.asarray(covariance, dtype=np.float64).reshape(3, 3).T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        standard = self._rng.standard_normal((int(count), 3))
        if count > 1:
            standard -= np.mean(standard, axis=0, keepdims=True)
        return standard @ root.T

    def _refresh_particle_statistics(self, cohort: _Cohort) -> None:
        cohort.map_pose2, cohort.covariance = _weighted_pose_statistics(
            cohort.particles, cohort.log_weights
        )

    def _predict_hypothesis_probabilities(self) -> None:
        """Apply the valid/null Markov transition before a new observation."""

        if not self._cohorts:
            self._null_log_probability = 0.0
            return
        null_probability = math.exp(self._null_log_probability)
        cohort_probabilities = np.asarray(
            [math.exp(item.log_probability) for item in self._cohorts],
            dtype=np.float64,
        )
        valid_probability = float(np.sum(cohort_probabilities))
        cohort_survival = self.config.cohort_survival_probability
        null_survival = self.config.null_survival_probability
        predicted_null = (
            null_probability * null_survival
            + valid_probability * (1.0 - cohort_survival)
        )
        reacquisition_share = (
            null_probability
            * (1.0 - null_survival)
            / float(len(self._cohorts))
        )
        predicted_cohorts = (
            cohort_survival * cohort_probabilities + reacquisition_share
        )
        total = predicted_null + float(np.sum(predicted_cohorts))
        predicted_null /= total
        predicted_cohorts /= total
        self._null_log_probability = math.log(
            max(predicted_null, 1.0e-300)
        )
        for cohort, probability in zip(
            self._cohorts, predicted_cohorts, strict=True
        ):
            cohort.log_probability = math.log(max(float(probability), 1.0e-300))
        self.hypothesis_transition_updates += 1

    def _regularized_systematic_resample(self, cohort: _Cohort) -> None:
        """Systematic resampling followed by covariance-preserving jitter.

        A Liu-West-style shrink/jitter step prevents a small Global-UTAN
        cohort from collapsing to duplicate particles after a sharp terrain
        observation while approximately preserving its first two moments.
        """

        count = cohort.particles.shape[0]
        weights = np.exp(cohort.log_weights - _logsumexp(cohort.log_weights))
        positions = (
            float(self._rng.uniform(0.0, 1.0 / count))
            + np.arange(count, dtype=np.float64) / count
        )
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, positions, side="right")

        mean, covariance = _weighted_pose_statistics(
            cohort.particles, cohort.log_weights
        )
        selected = cohort.particles[indices].copy()
        residuals = selected - mean
        residuals[:, 2] = (
            residuals[:, 2] + math.pi
        ) % (2.0 * math.pi) - math.pi
        bandwidth = self.config.resample_kernel_bandwidth
        shrinkage = math.sqrt(max(0.0, 1.0 - bandwidth**2))
        selected = mean + shrinkage * residuals
        selected += self._draw_centered_gaussian(
            covariance * bandwidth**2, count
        )
        selected[:, 2] = (
            selected[:, 2] + math.pi
        ) % (2.0 * math.pi) - math.pi
        cohort.particles = selected
        cohort.log_weights = np.full(count, -math.log(count), dtype=np.float64)
        cohort.resampling_events += 1
        self.particle_resampling_events += 1
        self._refresh_particle_statistics(cohort)

    def _predict(self, timestamp_s: float, raw_pose2: np.ndarray) -> None:
        if self._previous_timestamp_s is None or self._previous_raw_pose2 is None:
            return
        delta_time_s = timestamp_s - self._previous_timestamp_s
        relative = _between_pose2(self._previous_raw_pose2, raw_pose2)
        travelled_m = float(np.linalg.norm(relative[:2]))
        translation_std = math.hypot(
            self.config.process_translation_std_per_m * travelled_m,
            self.config.process_translation_std_per_s * delta_time_s,
        )
        yaw_std = math.radians(
            math.hypot(
                self.config.process_yaw_std_deg_per_m * travelled_m,
                self.config.process_yaw_std_deg_per_s * delta_time_s,
            )
        )
        process_covariance = np.diag(
            [translation_std**2, translation_std**2, yaw_std**2]
        )
        for cohort in self._cohorts:
            dx, dy = float(relative[0]), float(relative[1])
            yaw = cohort.particles[:, 2].copy()
            cosine = np.cos(yaw)
            sine = np.sin(yaw)
            cohort.particles[:, 0] += cosine * dx - sine * dy
            cohort.particles[:, 1] += sine * dx + cosine * dy
            cohort.particles[:, 2] += float(relative[2])
            if translation_std > 0.0 or yaw_std > 0.0:
                cohort.particles += self._draw_centered_gaussian(
                    process_covariance, cohort.particles.shape[0]
                )
            cohort.particles[:, 2] = (
                cohort.particles[:, 2] + math.pi
            ) % (2.0 * math.pi) - math.pi
            self._refresh_particle_statistics(cohort)

    def _evidence_log_likelihood_ratio(
        self, evidence: GlobalPoseEvidence
    ) -> float:
        if not evidence.cross_modal_consistent:
            return self.config.inconsistent_log_likelihood_ratio
        weighted_margin = 0.0
        active_weight = 0.0
        modalities = (
            (
                evidence.bathymetry_score,
                evidence.bathymetry_observability,
                self.config.bathymetry_reference_score,
            ),
            (
                evidence.backscatter_score,
                evidence.backscatter_observability,
                self.config.backscatter_reference_score,
            ),
        )
        for score, observability, reference in modalities:
            if observability <= 0.0:
                continue
            weighted_margin += observability * (score - reference)
            active_weight += observability
        if evidence.geometry_score is not None:
            # 3D occupancy verification and the bathymetry appearance score
            # share the same MBES acquisition, so they are correlated checks,
            # not independent observations.  Keep geometry as a consistency
            # channel, then normalize by the total active weight below; this
            # lets it veto a weak basin without multiplying the evidence count.
            weighted_margin += (
                evidence.geometry_score
                - self.config.geometry_reference_score
            )
            active_weight += 1.0
        if active_weight <= 0.0:
            return self.config.inconsistent_log_likelihood_ratio
        ratio = weighted_margin / (
            active_weight * self.config.score_temperature
        )
        limit = self.config.maximum_observation_log_likelihood_ratio
        return float(np.clip(ratio, -limit, limit))

    def _deduplicate_same_observation(
        self, evidences: tuple[GlobalPoseEvidence, ...]
    ) -> tuple[GlobalPoseEvidence, ...]:
        """Retain one proposal after several seeds reach one metric basin.

        This is deliberately performed *after* independent MBES/backscatter
        verification.  A coarse yaw or lattice variant is not a second sensor
        observation merely because it has a different source index.
        """

        def quality(evidence: GlobalPoseEvidence) -> tuple[float, ...]:
            covariance = evidence.covariance
            sign, log_determinant = np.linalg.slogdet(covariance)
            precision_tiebreak = (
                -float(log_determinant) if sign > 0.0 else -1.0e300
            )
            return (
                1.0 if evidence.target_cohort_id is not None else 0.0,
                self._evidence_log_likelihood_ratio(evidence),
                -1.0 if evidence.geometry_score is None else evidence.geometry_score,
                precision_tiebreak,
                -float(evidence.source_index),
            )

        ordered = sorted(evidences, key=quality, reverse=True)
        retained: list[GlobalPoseEvidence] = []
        for evidence in ordered:
            duplicate = False
            for previous in retained:
                # Distinct persistent cohorts each need the current raw
                # acquisition evaluated in their own basin.  They may have
                # converged spatially, but same-observation NMS must not erase
                # either identity before the tracker can update/merge them.
                if (
                    evidence.target_cohort_id is not None
                    and previous.target_cohort_id is not None
                    and evidence.target_cohort_id
                    != previous.target_cohort_id
                ):
                    continue
                position_separation = float(
                    np.linalg.norm(
                        evidence.map_pose2[:2] - previous.map_pose2[:2]
                    )
                )
                yaw_separation_deg = abs(
                    math.degrees(
                        _wrap_angle(
                            float(
                                evidence.map_pose2[2]
                                - previous.map_pose2[2]
                            )
                        )
                    )
                )
                if (
                    position_separation
                    <= self.config.same_observation_nms_position_m
                    and yaw_separation_deg
                    <= self.config.same_observation_nms_yaw_deg
                ):
                    duplicate = True
                    break
            if duplicate:
                self.duplicate_evidences_suppressed += 1
            else:
                retained.append(evidence)
        retained.sort(key=lambda item: item.source_index)
        return tuple(retained)

    @staticmethod
    def _window_is_disjoint(
        cohort: _Cohort,
        *,
        patch_start_timestamp_s: float,
        patch_start_observation_id: int,
    ) -> bool:
        return bool(
            patch_start_timestamp_s
            > cohort.last_independent_patch_end_timestamp_s + 1.0e-9
            and patch_start_observation_id
            > cohort.last_independent_patch_end_observation_id
        )

    @classmethod
    def _windows_are_disjoint(
        cls, cohort: _Cohort, evidence: GlobalPoseEvidence
    ) -> bool:
        return cls._window_is_disjoint(
            cohort,
            patch_start_timestamp_s=evidence.patch_start_timestamp_s,
            patch_start_observation_id=evidence.patch_start_observation_id,
        )

    @staticmethod
    def _novel_observation_fraction(
        cohort: _Cohort,
        *,
        patch_start_observation_id: int,
        patch_end_observation_id: int,
    ) -> float:
        """Return the fraction of pings not used by the preceding update."""

        if cohort.last_processed_patch_end_observation_id < 0:
            return 1.0
        if (
            patch_start_observation_id
            > cohort.last_processed_patch_end_observation_id
        ):
            return 1.0
        window_count = max(
            1,
            patch_end_observation_id - patch_start_observation_id + 1,
        )
        novel_count = max(
            0,
            patch_end_observation_id
            - cohort.last_processed_patch_end_observation_id,
        )
        return float(np.clip(novel_count / window_count, 0.0, 1.0))

    def _association_candidates(
        self, evidences: tuple[GlobalPoseEvidence, ...]
    ) -> list[tuple[float, int, int]]:
        pairs: list[tuple[float, int, int]] = []
        for cohort_index, cohort in enumerate(self._cohorts):
            for evidence_index, evidence in enumerate(evidences):
                if evidence.target_cohort_id is not None:
                    continue
                residual = self._residual(cohort.map_pose2, evidence.map_pose2)
                position_error = float(np.linalg.norm(residual[:2]))
                yaw_error_deg = abs(math.degrees(float(residual[2])))
                if (
                    position_error > self.config.association_max_position_m
                    or yaw_error_deg > self.config.association_max_yaw_deg
                ):
                    continue
                innovation = cohort.covariance + evidence.covariance
                try:
                    nis = float(residual @ np.linalg.solve(innovation, residual))
                except np.linalg.LinAlgError:
                    continue
                if np.isfinite(nis) and nis <= self.config.association_max_nis:
                    pairs.append((nis, cohort_index, evidence_index))
        pairs.sort(key=lambda item: item[0])
        return pairs

    def _update_cohort(
        self,
        cohort: _Cohort,
        evidence: GlobalPoseEvidence,
        _association_nis: float,
    ) -> bool:
        independent = self._windows_are_disjoint(cohort, evidence)
        scale = self._novel_observation_fraction(
            cohort,
            patch_start_observation_id=(
                evidence.patch_start_observation_id
            ),
            patch_end_observation_id=evidence.patch_end_observation_id,
        )
        measurement_covariance = evidence.covariance / max(scale, 1.0e-12)
        prior_covariance = cohort.covariance.copy()
        try:
            inverse_measurement_covariance = np.linalg.inv(
                measurement_covariance
            )
        except np.linalg.LinAlgError:
            inverse_measurement_covariance = np.linalg.pinv(
                measurement_covariance
            )
        residuals = evidence.map_pose2[None, :] - cohort.particles
        residuals[:, 2] = (
            residuals[:, 2] + math.pi
        ) % (2.0 * math.pi) - math.pi
        squared_mahalanobis = np.einsum(
            "ni,ij,nj->n",
            residuals,
            inverse_measurement_covariance,
            residuals,
        )
        particle_log_likelihood = -0.5 * np.maximum(
            squared_mahalanobis, 0.0
        )
        posterior_log_weights = (
            cohort.log_weights + particle_log_likelihood
        )
        predictive_log_likelihood = _logsumexp(posterior_log_weights)
        cohort.log_weights = (
            posterior_log_weights - predictive_log_likelihood
        )
        weights = np.exp(cohort.log_weights)
        effective_sample_size = 1.0 / float(np.sum(weights**2))
        cohort.last_effective_sample_size = effective_sample_size
        self._refresh_particle_statistics(cohort)

        evidence_ratio = self._evidence_log_likelihood_ratio(evidence)
        innovation_covariance = prior_covariance + measurement_covariance
        measurement_sign, measurement_log_determinant = np.linalg.slogdet(
            measurement_covariance
        )
        innovation_sign, innovation_log_determinant = np.linalg.slogdet(
            innovation_covariance
        )
        if measurement_sign > 0.0 and innovation_sign > 0.0:
            expected_consistent_log_likelihood = 0.5 * (
                measurement_log_determinant - innovation_log_determinant
            )
        else:
            expected_consistent_log_likelihood = 0.0
        # The proposal-quality likelihood determines how plausible the basin
        # is against the explicit null state.  The particle predictive
        # likelihood additionally scores whether this acquisition agrees with
        # the propagated cohort.  Center it by the likelihood expected for a
        # zero-residual Gaussian cohort; otherwise a broad but correct cohort
        # would lose probability merely because its particles represent the
        # required uncertainty volume.
        observation_log_likelihood_ratio = (
            scale * evidence_ratio
            + predictive_log_likelihood
            - expected_consistent_log_likelihood
        )
        cohort.log_probability += observation_log_likelihood_ratio
        self.particle_measurement_updates += 1
        if (
            effective_sample_size
            < self.config.resample_effective_sample_fraction
            * cohort.particles.shape[0]
        ):
            self._regularized_systematic_resample(cohort)
        cohort.total_evidence += 1
        if independent:
            cohort.independent_evidence += 1
            # Only a newly independent acquisition advances the independence
            # anchor.  Moving this boundary for every overlapping rolling
            # window would make a continuous correct track wait forever: the
            # next window start would always trail the immediately preceding
            # window end.
            cohort.last_independent_patch_end_timestamp_s = (
                evidence.patch_end_timestamp_s
            )
            cohort.last_independent_patch_end_observation_id = (
                evidence.patch_end_observation_id
            )
        cohort.consecutive_misses = 0
        cohort.last_source_index = evidence.source_index
        cohort.last_measurement_pose2 = evidence.map_pose2.copy()
        cohort.last_measurement_covariance = measurement_covariance.copy()
        cohort.last_measurement_information = evidence.information
        cohort.last_measurement_observable_rank = (
            _observable_information_rank(evidence.information)
        )
        cohort.last_observation_log_likelihood_ratio = float(
            observation_log_likelihood_ratio
        )
        cohort.last_update_was_independent = bool(independent)
        cohort.last_measurement_is_graph_ready = bool(
            independent and cohort.last_measurement_observable_rank >= 2
        )
        cohort.last_observation_novel_fraction = float(scale)
        cohort.last_processed_patch_end_timestamp_s = max(
            cohort.last_processed_patch_end_timestamp_s,
            evidence.patch_end_timestamp_s,
        )
        cohort.last_processed_patch_end_observation_id = max(
            cohort.last_processed_patch_end_observation_id,
            evidence.patch_end_observation_id,
        )
        return bool(independent)

    def _update_cohort_from_particle_likelihood(
        self,
        cohort: _Cohort,
        observation: ParticleLikelihoodObservation,
        *,
        associated_evidence: GlobalPoseEvidence | None,
        precomputed_log_likelihood_ratio: np.ndarray | None = None,
    ) -> bool:
        """Apply one raw sonar-map likelihood without a Gaussian pose proxy."""

        independent = self._window_is_disjoint(
            cohort,
            patch_start_timestamp_s=observation.patch_start_timestamp_s,
            patch_start_observation_id=(
                observation.patch_start_observation_id
            ),
        )
        scale = self._novel_observation_fraction(
            cohort,
            patch_start_observation_id=(
                observation.patch_start_observation_id
            ),
            patch_end_observation_id=(
                observation.patch_end_observation_id
            ),
        )
        if precomputed_log_likelihood_ratio is None:
            raw_log_likelihood_ratio = observation.evaluate(cohort.particles)
        else:
            raw_log_likelihood_ratio = np.asarray(
                precomputed_log_likelihood_ratio, dtype=np.float64
            ).reshape(-1)
            if raw_log_likelihood_ratio.size != cohort.particles.shape[0]:
                raise ValueError(
                    "precomputed particle likelihood count must match cohort"
                )
            if not np.isfinite(raw_log_likelihood_ratio).all():
                raise ValueError(
                    "precomputed particle likelihoods must be finite"
                )
        log_likelihood_ratio = scale * raw_log_likelihood_ratio

        # This distribution contains only the *current* observation.  It is
        # retained separately from the recursive posterior so a downstream
        # factor graph never mistakes accumulated PF information for a fresh,
        # statistically independent unary factor.
        measurement_log_weights = (
            log_likelihood_ratio - _logsumexp(log_likelihood_ratio)
        )
        measurement_pose2, measurement_covariance = (
            _weighted_pose_statistics(
                cohort.particles, measurement_log_weights
            )
        )

        posterior_log_weights = cohort.log_weights + log_likelihood_ratio
        predictive_log_likelihood_ratio = _logsumexp(
            posterior_log_weights
        )
        cohort.log_weights = (
            posterior_log_weights - predictive_log_likelihood_ratio
        )
        weights = np.exp(cohort.log_weights)
        effective_sample_size = 1.0 / float(np.sum(weights**2))
        cohort.last_effective_sample_size = effective_sample_size
        self._refresh_particle_statistics(cohort)
        cohort.log_probability += predictive_log_likelihood_ratio
        self.particle_measurement_updates += 1
        self.direct_map_likelihood_updates += 1
        if (
            effective_sample_size
            < self.config.resample_effective_sample_fraction
            * cohort.particles.shape[0]
        ):
            self._regularized_systematic_resample(cohort)

        cohort.total_evidence += 1
        if independent:
            cohort.independent_evidence += 1
            cohort.last_independent_patch_end_timestamp_s = (
                observation.patch_end_timestamp_s
            )
            cohort.last_independent_patch_end_observation_id = (
                observation.patch_end_observation_id
            )
        cohort.consecutive_misses = 0
        if associated_evidence is not None:
            cohort.last_source_index = int(associated_evidence.source_index)
            # The associated compact match is derived from this acquisition
            # alone, so it is a valid unary graph measurement.  The raw
            # particle likelihood is still used for the recursive PF update,
            # but its support was proposed by the previous posterior and must
            # not be mistaken for an independent covariance estimate.
            cohort.last_measurement_pose2 = (
                associated_evidence.map_pose2.copy()
            )
            cohort.last_measurement_covariance = (
                associated_evidence.covariance.copy()
            )
            cohort.last_measurement_information = (
                associated_evidence.information
            )
        else:
            cohort.last_measurement_pose2 = measurement_pose2
            cohort.last_measurement_covariance = measurement_covariance
            cohort.last_measurement_information = np.linalg.pinv(
                measurement_covariance, rcond=1.0e-12, hermitian=True
            )
        cohort.last_measurement_observable_rank = (
            _observable_information_rank(
                cohort.last_measurement_information
            )
        )
        cohort.last_observation_log_likelihood_ratio = float(
            predictive_log_likelihood_ratio
        )
        cohort.last_update_was_independent = bool(independent)
        cohort.last_measurement_is_graph_ready = bool(
            independent
            and associated_evidence is not None
            and cohort.last_measurement_observable_rank >= 2
        )
        cohort.last_observation_novel_fraction = float(scale)
        cohort.last_processed_patch_end_timestamp_s = max(
            cohort.last_processed_patch_end_timestamp_s,
            observation.patch_end_timestamp_s,
        )
        cohort.last_processed_patch_end_observation_id = max(
            cohort.last_processed_patch_end_observation_id,
            observation.patch_end_observation_id,
        )
        return bool(independent)

    def _create_cohort(
        self,
        evidence: GlobalPoseEvidence,
        candidate_count: int,
        prior_null_log_probability: float,
        *,
        count_evidence: bool = True,
        include_evidence_quality: bool = True,
        initial_log_probability: float | None = None,
    ) -> None:
        birth_share = self.config.birth_probability / max(1, candidate_count)
        log_probability = (
            (
                prior_null_log_probability + math.log(birth_share)
                if initial_log_probability is None
                else float(initial_log_probability)
            )
            + (
                self._evidence_log_likelihood_ratio(evidence)
                if include_evidence_quality
                else 0.0
            )
        )
        count = self.config.particles_per_cohort
        particles = evidence.map_pose2[None, :] + self._draw_centered_gaussian(
            evidence.covariance, count
        )
        particles[:, 2] = (
            particles[:, 2] + math.pi
        ) % (2.0 * math.pi) - math.pi
        log_weights = np.full(count, -math.log(count), dtype=np.float64)
        map_pose2, covariance = _weighted_pose_statistics(
            particles, log_weights
        )
        self._cohorts.append(
            _Cohort(
                cohort_id=self._next_cohort_id,
                particles=particles,
                log_weights=log_weights,
                map_pose2=map_pose2,
                covariance=covariance,
                log_probability=log_probability,
                total_evidence=1 if count_evidence else 0,
                independent_evidence=1 if count_evidence else 0,
                consecutive_misses=0,
                last_independent_patch_end_timestamp_s=(
                    evidence.patch_end_timestamp_s
                    if count_evidence
                    else float("-inf")
                ),
                last_independent_patch_end_observation_id=(
                    evidence.patch_end_observation_id
                    if count_evidence
                    else -1
                ),
                last_source_index=evidence.source_index,
                last_effective_sample_size=float(count),
                resampling_events=0,
                last_measurement_pose2=evidence.map_pose2.copy(),
                last_measurement_covariance=evidence.covariance.copy(),
                last_measurement_information=evidence.information,
                last_measurement_observable_rank=(
                    _observable_information_rank(evidence.information)
                ),
                last_observation_log_likelihood_ratio=(
                    self._evidence_log_likelihood_ratio(evidence)
                    if include_evidence_quality
                    else 0.0
                ),
                last_update_was_independent=bool(count_evidence),
                last_measurement_is_graph_ready=bool(
                    count_evidence
                    and _observable_information_rank(evidence.information)
                    >= 2
                ),
                last_observation_novel_fraction=(
                    1.0 if count_evidence else 0.0
                ),
                last_processed_patch_end_timestamp_s=(
                    evidence.patch_end_timestamp_s
                    if count_evidence
                    else float("-inf")
                ),
                last_processed_patch_end_observation_id=(
                    evidence.patch_end_observation_id
                    if count_evidence
                    else -1
                ),
            )
        )
        self._next_cohort_id += 1
        self.cohorts_created += 1

    def _normalize_and_prune(self) -> None:
        values = [self._null_log_probability]
        values.extend(cohort.log_probability for cohort in self._cohorts)
        maximum = max(values)
        normalizer = maximum + math.log(
            sum(math.exp(value - maximum) for value in values)
        )
        self._null_log_probability -= normalizer
        for cohort in self._cohorts:
            cohort.log_probability -= normalizer

        retained: list[_Cohort] = []
        for cohort in self._cohorts:
            probability = math.exp(cohort.log_probability)
            if (
                cohort.consecutive_misses
                > self.config.maximum_missed_observations
                or probability < self.config.minimum_retained_probability
            ):
                self.cohorts_pruned += 1
                continue
            retained.append(cohort)
        retained.sort(key=lambda item: item.log_probability, reverse=True)
        if len(retained) > self.config.maximum_cohorts:
            self.cohorts_pruned += len(retained) - self.config.maximum_cohorts
            retained = retained[: self.config.maximum_cohorts]
        self._cohorts = retained

        # Pruning removes normalized probability mass.  Normalize once more so
        # the returned posterior (including the null state) sums exactly to one.
        values = [self._null_log_probability]
        values.extend(cohort.log_probability for cohort in self._cohorts)
        maximum = max(values)
        normalizer = maximum + math.log(
            sum(math.exp(value - maximum) for value in values)
        )
        self._null_log_probability -= normalizer
        for cohort in self._cohorts:
            cohort.log_probability -= normalizer

    def _snapshot(self, cohort: _Cohort) -> GlobalCohortSnapshot:
        if self._current_raw_pose2 is None:
            raise RuntimeError("tracker has no current raw pose")
        return GlobalCohortSnapshot(
            cohort_id=cohort.cohort_id,
            map_pose2=cohort.map_pose2,
            map_to_odom_pose2=_map_to_odom(
                cohort.map_pose2, self._current_raw_pose2
            ),
            covariance=cohort.covariance,
            probability=math.exp(cohort.log_probability),
            total_evidence=cohort.total_evidence,
            independent_evidence=cohort.independent_evidence,
            consecutive_misses=cohort.consecutive_misses,
            last_source_index=cohort.last_source_index,
            particle_count=int(cohort.particles.shape[0]),
            effective_sample_size=float(cohort.last_effective_sample_size),
            resampling_events=int(cohort.resampling_events),
            last_measurement_pose2=cohort.last_measurement_pose2,
            last_measurement_covariance=(
                cohort.last_measurement_covariance
            ),
            last_measurement_information=(
                cohort.last_measurement_information
            ),
            last_measurement_observable_rank=int(
                cohort.last_measurement_observable_rank
            ),
            last_observation_log_likelihood_ratio=float(
                cohort.last_observation_log_likelihood_ratio
            ),
            last_update_was_independent=bool(
                cohort.last_update_was_independent
            ),
            last_measurement_is_graph_ready=bool(
                cohort.last_measurement_is_graph_ready
            ),
            last_observation_novel_fraction=float(
                cohort.last_observation_novel_fraction
            ),
        )

    def _decision(
        self, independently_updated_cohort_ids: set[int] | None = None
    ) -> GlobalTrackingDecision:
        independently_updated_cohort_ids = (
            set()
            if independently_updated_cohort_ids is None
            else independently_updated_cohort_ids
        )
        snapshots = tuple(self._snapshot(cohort) for cohort in self._cohorts)
        if not snapshots:
            return GlobalTrackingDecision(
                status="no_fix",
                confirmed=False,
                null_probability=self.null_probability,
                best_probability=0.0,
                odds_over_second=0.0,
                cohorts=(),
                confirmed_with_new_independent_evidence=False,
            )
        best = snapshots[0]
        runner_up_probability = max(
            self.null_probability,
            snapshots[1].probability if len(snapshots) > 1 else 0.0,
        )
        odds = best.probability / max(runner_up_probability, 1.0e-300)
        position_covariance = best.covariance[:2, :2]
        maximum_position_std = math.sqrt(
            max(0.0, float(np.linalg.eigvalsh(position_covariance)[-1]))
        )
        yaw_std_deg = math.degrees(
            math.sqrt(max(0.0, float(best.covariance[2, 2])))
        )
        confirmed = bool(
            best.total_evidence >= self.config.minimum_total_evidence
            and best.independent_evidence
            >= self.config.minimum_independent_evidence
            and best.probability >= self.config.confirmation_probability
            and odds >= self.config.confirmation_odds_over_second
            and maximum_position_std
            <= self.config.confirmation_max_position_std_m
            and yaw_std_deg <= self.config.confirmation_max_yaw_std_deg
        )
        if confirmed:
            self.confirmations += 1
            status = "confirmed"
            confirmed_cohort = best
        else:
            status = "tracking" if best.probability > self.null_probability else "no_fix"
            confirmed_cohort = None
            if len(snapshots) > 1 and odds < self.config.confirmation_odds_over_second:
                self.ambiguous_updates += 1
        return GlobalTrackingDecision(
            status=status,
            confirmed=confirmed,
            null_probability=self.null_probability,
            best_probability=best.probability,
            odds_over_second=odds,
            cohorts=snapshots,
            confirmed_cohort=confirmed_cohort,
            confirmed_with_new_independent_evidence=bool(
                confirmed_cohort is not None
                and confirmed_cohort.cohort_id
                in independently_updated_cohort_ids
            ),
        )

    def update(
        self,
        *,
        timestamp_s: float,
        raw_pose2: np.ndarray,
        evidences: Iterable[GlobalPoseEvidence],
        particle_observation: ParticleLikelihoodObservation | None = None,
    ) -> GlobalTrackingDecision:
        """Propagate, associate and update all cohorts for one sonar epoch."""

        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if (
            self._previous_timestamp_s is not None
            and timestamp <= self._previous_timestamp_s
        ):
            raise ValueError("tracker timestamps must increase")
        raw = _pose2(raw_pose2, "raw_pose2")
        supplied_evidences = tuple(evidences)
        if not all(
            isinstance(item, GlobalPoseEvidence) for item in supplied_evidences
        ):
            raise TypeError("evidences must contain GlobalPoseEvidence instances")
        source_indices = [item.source_index for item in supplied_evidences]
        if len(source_indices) != len(set(source_indices)):
            raise ValueError("source_index values must be unique within one update")
        if (
            particle_observation is not None
            and not isinstance(
                particle_observation, ParticleLikelihoodObservation
            )
        ):
            raise TypeError(
                "particle_observation must be a "
                "ParticleLikelihoodObservation"
            )
        evidence_tuple = self._deduplicate_same_observation(
            supplied_evidences
        )

        self._predict(timestamp, raw)
        self._current_raw_pose2 = raw.copy()
        self._predict_hypothesis_probabilities()
        prior_null_log_probability = self._null_log_probability
        for cohort in self._cohorts:
            cohort.last_update_was_independent = False
            cohort.last_measurement_is_graph_ready = False

        assigned_cohorts: set[int] = set()
        assigned_evidences: set[int] = set()
        associated_evidences: dict[int, GlobalPoseEvidence] = {}
        independently_updated_cohort_ids: set[int] = set()
        new_cohort_ids: set[int] = set()

        # Reserve provenance-labelled proposals before metric association.
        # The label was created from this exact cohort's non-mutating
        # prediction, so it is a data-flow identity rather than an error gate.
        cohort_indices_by_id = {
            cohort.cohort_id: index
            for index, cohort in enumerate(self._cohorts)
        }
        for evidence_index, evidence in enumerate(evidence_tuple):
            target_id = evidence.target_cohort_id
            if target_id is None:
                continue
            cohort_index = cohort_indices_by_id.get(target_id)
            if cohort_index is None or cohort_index in assigned_cohorts:
                continue
            cohort = self._cohorts[cohort_index]
            associated_evidences[cohort.cohort_id] = evidence
            if particle_observation is None:
                independent = self._update_cohort(cohort, evidence, 0.0)
                if independent:
                    independently_updated_cohort_ids.add(cohort.cohort_id)
            assigned_cohorts.add(cohort_index)
            assigned_evidences.add(evidence_index)

        for nis, cohort_index, evidence_index in self._association_candidates(
            evidence_tuple
        ):
            if cohort_index in assigned_cohorts or evidence_index in assigned_evidences:
                continue
            cohort = self._cohorts[cohort_index]
            evidence = evidence_tuple[evidence_index]
            associated_evidences[cohort.cohort_id] = evidence
            if particle_observation is None:
                independent = self._update_cohort(cohort, evidence, nis)
                if independent:
                    independently_updated_cohort_ids.add(cohort.cohort_id)
            assigned_cohorts.add(cohort_index)
            assigned_evidences.add(evidence_index)

        unmatched = [
            evidence
            for index, evidence in enumerate(evidence_tuple)
            if (
                index not in assigned_evidences
                and evidence.target_cohort_id is None
            )
        ]
        particle_birth_log_probability: float | None = None
        if particle_observation is not None and unmatched:
            # Augmented-MCL-style recovery mass is taken from the complete
            # prior, not only from the null hypothesis.  Restricting births to
            # ``p(null)`` made an old, once-confident alias effectively
            # immortal: a correct whole-map proposal could start hundreds of
            # times less likely before either candidate had seen the current
            # raw sonar window.  This is the Bayesian change-point branch of
            # the motion model.  The raw observation below still decides which
            # branch survives; proposal scores are not counted a second time.
            retained_log_mass = math.log(
                1.0 - self.config.birth_probability
            )
            self._null_log_probability += retained_log_mass
            for cohort in self._cohorts:
                cohort.log_probability += retained_log_mass
            particle_birth_log_probability = math.log(
                self.config.birth_probability / float(len(unmatched))
            )
            self.candidate_birth_mass_allocations += 1
        for evidence in unmatched:
            self._create_cohort(
                evidence,
                candidate_count=len(unmatched),
                prior_null_log_probability=prior_null_log_probability,
                count_evidence=particle_observation is None,
                include_evidence_quality=particle_observation is None,
                initial_log_probability=particle_birth_log_probability,
            )
            new_cohort_ids.add(self._cohorts[-1].cohort_id)
            if particle_observation is None:
                independently_updated_cohort_ids.add(
                    self._cohorts[-1].cohort_id
                )
            else:
                associated_evidences[self._cohorts[-1].cohort_id] = evidence

        if particle_observation is None:
            missed_log_likelihood = math.log(
                self.config.missed_detection_likelihood
            )
            for index, cohort in enumerate(self._cohorts):
                if (
                    index in assigned_cohorts
                    or cohort.cohort_id in new_cohort_ids
                ):
                    continue
                cohort.log_probability += missed_log_likelihood
                cohort.consecutive_misses += 1
        else:
            # Evaluate every spatially separated hypothesis in one callback.
            # The acoustic likelihood normalizes its within-window residuals
            # against the best particle supplied to that callback.  Calling it
            # once per cohort therefore erased the scientifically essential
            # likelihood ratio *between* an old track and a newly injected
            # whole-map candidate.  A joint point-mass update retains that
            # ratio and is equivalent to one multimodal particle population.
            particle_counts = [
                cohort.particles.shape[0] for cohort in self._cohorts
            ]
            joint_log_likelihood_ratio = np.empty(0, dtype=np.float64)
            if self._cohorts:
                joint_particles = np.concatenate(
                    [cohort.particles for cohort in self._cohorts], axis=0
                )
                joint_log_likelihood_ratio = particle_observation.evaluate(
                    joint_particles
                )
                self._last_joint_log_likelihood_span = float(
                    np.ptp(joint_log_likelihood_ratio)
                )
                self._last_joint_log_likelihood_std = float(
                    np.std(joint_log_likelihood_ratio)
                )
                self.joint_particle_likelihood_updates += 1
            offset = 0
            for cohort, particle_count in zip(
                self._cohorts, particle_counts, strict=True
            ):
                cohort_log_likelihood_ratio = (
                    joint_log_likelihood_ratio[
                        offset : offset + particle_count
                    ]
                )
                offset += particle_count
                independent = self._update_cohort_from_particle_likelihood(
                    cohort,
                    particle_observation,
                    associated_evidence=associated_evidences.get(
                        cohort.cohort_id
                    ),
                    precomputed_log_likelihood_ratio=(
                        cohort_log_likelihood_ratio
                    ),
                )
                if independent:
                    independently_updated_cohort_ids.add(cohort.cohort_id)

        self._normalize_and_prune()
        leading_cohort_id = (
            None if not self._cohorts else self._cohorts[0].cohort_id
        )
        if (
            self._last_leading_cohort_id is not None
            and leading_cohort_id is not None
            and leading_cohort_id != self._last_leading_cohort_id
        ):
            self.posterior_mode_switches += 1
        self._last_leading_cohort_id = leading_cohort_id
        self.updates += 1
        self._previous_timestamp_s = timestamp
        self._previous_raw_pose2 = raw.copy()
        decision = self._decision(independently_updated_cohort_ids)
        self._last_decision = decision
        return decision

    def reset(self) -> None:
        """Clear all cohorts after a verified factor has re-anchored the graph."""

        self._cohorts.clear()
        self._null_log_probability = 0.0
        self._previous_timestamp_s = None
        self._previous_raw_pose2 = None
        self._current_raw_pose2 = None
        self._last_leading_cohort_id = None
        self._last_decision = None
        self._last_joint_log_likelihood_span = 0.0
        self._last_joint_log_likelihood_std = 0.0
        self._rng = np.random.default_rng(self.config.random_seed)

    def diagnostics(self) -> dict[str, float | int]:
        """Return JSON-safe tracker counters and current posterior summary."""

        probabilities = sorted(
            (math.exp(item.log_probability) for item in self._cohorts),
            reverse=True,
        )
        posterior_probabilities = np.asarray(
            [self.null_probability, *probabilities], dtype=np.float64
        )
        positive = posterior_probabilities > 0.0
        hypothesis_entropy = -float(
            np.sum(
                posterior_probabilities[positive]
                * np.log(posterior_probabilities[positive])
            )
        )
        hypothesis_effective_count = 1.0 / float(
            np.sum(posterior_probabilities**2)
        )
        return {
            "updates": int(self.updates),
            "active_cohorts": int(len(self._cohorts)),
            "cohorts_created": int(self.cohorts_created),
            "cohorts_pruned": int(self.cohorts_pruned),
            "confirmations": int(self.confirmations),
            "ambiguous_updates": int(self.ambiguous_updates),
            "duplicate_evidences_suppressed": int(
                self.duplicate_evidences_suppressed
            ),
            "particles_per_cohort": int(self.config.particles_per_cohort),
            "active_particles": int(
                len(self._cohorts) * self.config.particles_per_cohort
            ),
            "particle_measurement_updates": int(
                self.particle_measurement_updates
            ),
            "direct_map_likelihood_updates": int(
                self.direct_map_likelihood_updates
            ),
            "joint_particle_likelihood_updates": int(
                self.joint_particle_likelihood_updates
            ),
            "candidate_birth_mass_allocations": int(
                self.candidate_birth_mass_allocations
            ),
            "posterior_mode_switches": int(self.posterior_mode_switches),
            "last_joint_log_likelihood_span": float(
                self._last_joint_log_likelihood_span
            ),
            "last_joint_log_likelihood_std": float(
                self._last_joint_log_likelihood_std
            ),
            "hypothesis_entropy_nats": float(hypothesis_entropy),
            "hypothesis_effective_count": float(
                hypothesis_effective_count
            ),
            "hypothesis_transition_updates": int(
                self.hypothesis_transition_updates
            ),
            "particle_resampling_events": int(
                self.particle_resampling_events
            ),
            "minimum_effective_sample_size": float(
                min(
                    (
                        item.last_effective_sample_size
                        for item in self._cohorts
                    ),
                    default=0.0,
                )
            ),
            "null_probability": float(self.null_probability),
            "best_probability": float(probabilities[0]) if probabilities else 0.0,
            "second_probability": float(probabilities[1]) if len(probabilities) > 1 else 0.0,
            "confirmed_navigation_output_available": bool(
                self._last_decision is not None
                and self._last_decision.confirmed_cohort is not None
            ),
        }


class TerrainNavigationFilter(GlobalCohortTracker):
    """Operational bathymetric TRN policy used behind the local front end.

    The implementation is an independent OceanSim adaptation of established
    point-mass/particle terrain-navigation practice: local odometry supplies
    motion, raw beams are scored against a prior map, all modes share one
    joint likelihood normalization, low ESS triggers regularized resampling,
    and an explicit change-point branch admits global reinitialization
    candidates.  ``GlobalCohortTracker`` remains as a compatibility name for
    existing experiments built around the earlier Global-UTAN terminology.
    """


TerrainNavigationFilterConfig = GlobalCohortTrackerConfig
