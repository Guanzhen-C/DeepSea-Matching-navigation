"""Absolute bathymetric terrain matching against a measured prior map.

The local FAST-LIVO2-inspired estimator aligns acoustic returns to its own
surfel map.  That is a relative constraint and cannot, by itself, recover the
georeferenced horizontal gauge after drift.  This module supplies the
complementary absolute constraint: a motion-compensated MBES patch is aligned
directly to a measured bathymetry raster without consulting simulator truth.

Only world ``x``, world ``y``, and ZYX yaw are optimized.  Vehicle depth,
roll, and pitch are deliberately held fixed because pressure and gravity are
the stronger sources for those states.  The residual for bottom return ``i``
is

``r_i = z_i_world - terrain_height(x_i_world, y_i_world)``.

The implementation is dependency-light (NumPy only), honours the raster's
four-corner validity mask, uses the exact gradient of its bilinear cell, and
combines a bounded coarse search with robust iterated Gauss--Newton.  A
terrain match is rejected unless it has enough overlap, two observable
horizontal directions, acceptable post-fit residuals, a solution away from
the search boundary, and (by default) a correction consistent with the seed
covariance.  Yaw is emitted only when the patch makes it independently
observable; otherwise yaw is fixed and the returned information has rank two.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from .acoustic_radiometry import log_backscatter, uses_relative_power
from .esikf import AcousticESIKF, InertialState
from .math3d import project_to_so3, so3_exp, so3_log
from .real_bathymetry import MeasuredBathymetryGrid


def information_with_covariance_floor(
    information: np.ndarray,
    floor_covariance: np.ndarray,
) -> np.ndarray:
    """Add model uncertainty without inventing unobservable information.

    A pseudoinverse represents an exactly unobservable direction with zero
    covariance.  Adding a diagonal floor to that matrix and inverting it
    would therefore turn the null direction into a precise measurement.  This
    routine instead performs the covariance addition only in the observable
    eigenspace of the supplied information matrix and lifts the result back
    with the original nullspace unchanged.
    """

    source = np.asarray(information, dtype=np.float64)
    floor = np.asarray(floor_covariance, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("information must be a square matrix")
    if floor.shape != source.shape:
        raise ValueError("floor_covariance must match information shape")
    source = 0.5 * (source + source.T)
    floor = 0.5 * (floor + floor.T)
    if not np.isfinite(source).all() or not np.isfinite(floor).all():
        raise ValueError("information and floor covariance must be finite")
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(source)
        floor_eigenvalues = np.linalg.eigvalsh(floor)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance-floor eigendecomposition failed") from exc
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = 1.0e-10 * scale
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError("information must be positive semidefinite")
    floor_scale = max(float(np.max(np.abs(floor_eigenvalues))), 1.0)
    if float(floor_eigenvalues[0]) < -1.0e-10 * floor_scale:
        raise ValueError("floor_covariance must be positive semidefinite")
    observable = eigenvalues > tolerance
    if not np.any(observable):
        return np.zeros_like(source)
    basis = eigenvectors[:, observable]
    observable_covariance = np.diag(1.0 / eigenvalues[observable])
    observable_covariance += basis.T @ floor @ basis
    reduced_information = np.linalg.pinv(
        observable_covariance, rcond=1.0e-12, hermitian=True
    )
    result = basis @ reduced_information @ basis.T
    return 0.5 * (result + result.T)


def reporting_covariance_from_information(
    information: np.ndarray,
    *,
    yaw_scale_m: float = 20.0,
    unobservable_variance_scaled_m2: float = 1.0e6,
) -> np.ndarray:
    """Return a finite positive-definite covariance for a rank-aware factor.

    ``pinv(information)`` is not a covariance when ``information`` is rank
    deficient: it assigns *zero* variance to every unobservable direction.
    That is the opposite of the intended statement and can also fail code
    which quite reasonably requires a positive-definite sampling covariance.

    This helper preserves the exact observable subspace in ``information``
    while constructing a separate, finite covariance suitable for reporting
    and particle proposal sampling.  Translation and yaw are first expressed
    in the same metre-like coordinates used by the terrain observability
    analysis.  Null-space directions receive a deliberately broad variance;
    observable directions receive the reciprocal information eigenvalue.
    The returned matrix must never be inverted to recover the factor -- pass
    the original information matrix to the graph instead.
    """

    source = np.asarray(information, dtype=np.float64)
    if source.shape != (3, 3):
        raise ValueError("information must have shape (3, 3)")
    source = 0.5 * (source + source.T)
    if not np.isfinite(source).all():
        raise ValueError("information must be finite")
    yaw_scale = float(yaw_scale_m)
    unobservable_variance = float(unobservable_variance_scaled_m2)
    if not np.isfinite(yaw_scale) or yaw_scale <= 0.0:
        raise ValueError("yaw_scale_m must be positive and finite")
    if (
        not np.isfinite(unobservable_variance)
        or unobservable_variance <= 0.0
    ):
        raise ValueError(
            "unobservable_variance_scaled_m2 must be positive and finite"
        )

    q_from_scaled = np.diag([1.0, 1.0, 1.0 / yaw_scale])
    scaled_information = q_from_scaled.T @ source @ q_from_scaled
    scaled_information = 0.5 * (
        scaled_information + scaled_information.T
    )
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(scaled_information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("information eigendecomposition failed") from exc
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = 1.0e-10 * scale
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError("information must be positive semidefinite")
    observable = eigenvalues > tolerance
    variances = np.full(3, unobservable_variance, dtype=np.float64)
    variances[observable] = 1.0 / eigenvalues[observable]
    covariance_scaled = (eigenvectors * variances) @ eigenvectors.T
    covariance = q_from_scaled @ covariance_scaled @ q_from_scaled.T
    covariance = 0.5 * (covariance + covariance.T)

    # Very different observable/null-space scales can leave a tiny negative
    # eigenvalue from round-off.  Repair only that numerical residue.  This is
    # a reporting covariance; the exact rank remains in ``information``.
    covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(
        covariance
    )
    numerical_floor = (
        np.finfo(np.float64).eps
        * max(float(covariance_eigenvalues[-1]), 1.0)
        * 16.0
    )
    covariance_eigenvalues = np.maximum(
        covariance_eigenvalues, numerical_floor
    )
    covariance = (
        covariance_eigenvectors * covariance_eigenvalues
    ) @ covariance_eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def covariance_intersection_pose2(
    first_pose2: np.ndarray,
    first_information: np.ndarray,
    second_pose2: np.ndarray,
    second_information: np.ndarray,
    *,
    first_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fuse correlated planar estimates with covariance intersection.

    When ``first_weight`` is omitted, the scalar CI weight is selected by
    minimizing the log-determinant of the fused covariance.  The derivative
    of ``log(det(w I1 + (1-w) I2))`` is monotone, so a deterministic bisection
    finds the global optimum without assuming that MBES and side-scan errors
    are independent.
    """

    first_pose = np.asarray(first_pose2, dtype=np.float64).reshape(3)
    second_pose = np.asarray(second_pose2, dtype=np.float64).reshape(3)
    first_info = np.asarray(
        first_information, dtype=np.float64
    ).reshape(3, 3)
    second_info = np.asarray(
        second_information, dtype=np.float64
    ).reshape(3, 3)
    first_info = 0.5 * (first_info + first_info.T)
    second_info = 0.5 * (second_info + second_info.T)
    if not (
        np.isfinite(first_pose).all()
        and np.isfinite(second_pose).all()
        and np.isfinite(first_info).all()
        and np.isfinite(second_info).all()
    ):
        raise ValueError("CI poses and information matrices must be finite")
    try:
        first_eigenvalues, first_eigenvectors = np.linalg.eigh(first_info)
        second_eigenvalues, second_eigenvectors = np.linalg.eigh(second_info)
        scale = max(
            float(np.max(np.abs(first_eigenvalues))),
            float(np.max(np.abs(second_eigenvalues))),
            1.0,
        )
        tolerance = 1.0e-10 * scale
        if (
            float(first_eigenvalues[0]) < -tolerance
            or float(second_eigenvalues[0]) < -tolerance
        ):
            raise ValueError(
                "CI information matrices must be positive semidefinite"
            )
        first_info = (
            first_eigenvectors
            @ np.diag(np.maximum(first_eigenvalues, 0.0))
            @ first_eigenvectors.T
        )
        second_info = (
            second_eigenvectors
            @ np.diag(np.maximum(second_eigenvalues, 0.0))
            @ second_eigenvectors.T
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("CI information eigendecomposition failed") from exc

    if first_weight is None:
        difference = first_info - second_info
        # This tiny term is used only while selecting the scalar weight.  The
        # returned information remains unregularized, so an actually
        # unobservable direction stays unobservable.
        numerical_regularizer = 1.0e-12 * scale * np.eye(3)

        def derivative(weight: float) -> float:
            combined = (
                second_info + weight * difference + numerical_regularizer
            )
            return float(np.trace(np.linalg.solve(combined, difference)))

        derivative_at_zero = derivative(0.0)
        derivative_at_one = derivative(1.0)
        if derivative_at_zero <= 0.0:
            weight = 0.0
        elif derivative_at_one >= 0.0:
            weight = 1.0
        else:
            lower = 0.0
            upper = 1.0
            for _ in range(60):
                midpoint = 0.5 * (lower + upper)
                if derivative(midpoint) > 0.0:
                    lower = midpoint
                else:
                    upper = midpoint
            weight = 0.5 * (lower + upper)
    else:
        weight = float(first_weight)
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("first_weight must lie in [0, 1] or be None")

    second_weight = 1.0 - weight
    combined_information = (
        weight * first_info + second_weight * second_info
    )
    delta = second_pose - first_pose
    delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
    right_hand_side = second_weight * second_info @ delta
    try:
        fused_delta = np.linalg.solve(
            combined_information, right_hand_side
        )
    except np.linalg.LinAlgError:
        fused_delta = np.linalg.pinv(
            combined_information, rcond=1.0e-12, hermitian=True
        ) @ right_hand_side
    fused_pose = first_pose + fused_delta
    fused_pose[2] = math.atan2(
        math.sin(fused_pose[2]), math.cos(fused_pose[2])
    )
    return fused_pose, combined_information, float(weight)


@dataclass(frozen=True)
class TerrainMatcherConfig:
    """Numerical and statistical gates for bathymetric terrain matching.

    ``measurement_std_m`` is the combined vertical uncertainty of the MBES
    return, sound-speed/ray model, and prior bathymetry.  Dense beams from one
    or a few pings are correlated, so ``effective_independent_points`` caps
    the information contributed by a patch even when many rays are supplied.
    The coarse-seed separation radii only distribute the finite refinement
    budget across the search window.  The (smaller) basin-separation radii are
    used after refinement to decide whether two solutions are genuinely
    distinct.  Keeping those concepts separate is safety-critical for wide
    recovery searches: aggressive seed suppression must never merge two
    nearby, equally plausible final locations.  The
    ``minimum_basin_log_likelihood_ratio`` then requires the best refined basin
    to beat the runner-up after correlation and measurement noise are applied.
    The yaw scale is used only for rank/conditioning checks; returned
    information remains in physical ``[m, m, rad]`` coordinates.
    """

    measurement_std_m: float = 0.50
    huber_delta_m: float = 0.75
    inlier_threshold_m: float = 1.25
    minimum_points: int = 30
    minimum_inlier_ratio: float = 0.65
    minimum_valid_fraction: float = 0.75
    maximum_points: int = 768
    effective_independent_points: int = 96

    maximum_translation_m: float = 30.0
    maximum_yaw_deg: float = 8.0
    coarse_translation_step_m: float = 5.0
    coarse_yaw_step_deg: float = 2.0
    coarse_top_k: int = 7
    # Coarse lattice seeds are suppressed in this normalized ``[x, y, yaw]``
    # neighbourhood so adjacent samples of one broad minimum do not consume
    # the complete refinement budget.
    coarse_seed_translation_separation_m: float = 8.0
    coarse_seed_yaw_separation_deg: float = 2.0
    # Refined solutions farther apart than these much more local radii count as
    # distinct basins for the ambiguity test.  These values must not be enlarged
    # merely to accelerate a wider coarse search.
    minimum_basin_translation_separation_m: float = 5.0
    minimum_basin_yaw_separation_deg: float = 1.0
    # For Gaussian residuals this is the required log likelihood advantage
    # of the best distinct basin over the runner-up.  log(100) is deliberately
    # conservative: an absolute fix needs at least 100:1 evidence before it
    # is allowed to remove the global position gauge.
    minimum_basin_log_likelihood_ratio: float = math.log(100.0)

    maximum_iterations: int = 15
    maximum_iteration_translation_step_m: float = 4.0
    maximum_iteration_yaw_step_deg: float = 1.5
    translation_convergence_m: float = 1.0e-3
    yaw_convergence_deg: float = 2.0e-3
    levenberg_damping: float = 1.0e-5

    yaw_rank_scale_m: float = 20.0
    minimum_scaled_information_eigenvalue: float = 1.0e-3
    minimum_information_eigen_ratio: float = 3.0e-3
    maximum_reduced_chi_square: float = 9.0
    maximum_innovation_nis: float = 25.0
    enable_innovation_nis_gate: bool = True
    require_yaw_observable: bool = False
    # Finite reporting surrogate for an unobservable direction.  Fusion uses
    # the rank/projector rather than this deliberately enormous variance.
    unobservable_variance_scaled_m2: float = 1.0e12
    boundary_translation_margin_m: float = 0.25
    boundary_yaw_margin_deg: float = 0.10

    def __post_init__(self) -> None:
        positive_floats = (
            "measurement_std_m",
            "huber_delta_m",
            "inlier_threshold_m",
            "maximum_translation_m",
            "maximum_yaw_deg",
            "coarse_translation_step_m",
            "coarse_yaw_step_deg",
            "coarse_seed_translation_separation_m",
            "coarse_seed_yaw_separation_deg",
            "minimum_basin_translation_separation_m",
            "minimum_basin_yaw_separation_deg",
            "minimum_basin_log_likelihood_ratio",
            "maximum_iteration_translation_step_m",
            "maximum_iteration_yaw_step_deg",
            "translation_convergence_m",
            "yaw_convergence_deg",
            "yaw_rank_scale_m",
            "maximum_reduced_chi_square",
            "maximum_innovation_nis",
            "unobservable_variance_scaled_m2",
        )
        for name in positive_floats:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        nonnegative_floats = (
            "levenberg_damping",
            "minimum_scaled_information_eigenvalue",
            "minimum_information_eigen_ratio",
            "boundary_translation_margin_m",
            "boundary_yaw_margin_deg",
        )
        for name in nonnegative_floats:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        positive_integers = (
            "minimum_points",
            "maximum_points",
            "effective_independent_points",
            "coarse_top_k",
            "maximum_iterations",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_points < self.minimum_points:
            raise ValueError("maximum_points must not be below minimum_points")
        if self.coarse_top_k < 2:
            raise ValueError(
                "coarse_top_k must be at least two to test match uniqueness"
            )
        for name in ("minimum_inlier_ratio", "minimum_valid_fraction"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.minimum_information_eigen_ratio >= 1.0:
            raise ValueError(
                "minimum_information_eigen_ratio must be smaller than one"
            )
        for name in (
            "enable_innovation_nis_gate",
            "require_yaw_observable",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if self.boundary_translation_margin_m >= self.maximum_translation_m:
            raise ValueError(
                "boundary_translation_margin_m must be below the search bound"
            )
        if self.boundary_yaw_margin_deg >= self.maximum_yaw_deg:
            raise ValueError(
                "boundary_yaw_margin_deg must be below the yaw search bound"
            )
        if (
            self.coarse_seed_translation_separation_m
            < self.coarse_translation_step_m
        ):
            raise ValueError(
                "coarse_seed_translation_separation_m must not be below "
                "coarse_translation_step_m"
            )
        if (
            self.coarse_seed_yaw_separation_deg
            < self.coarse_yaw_step_deg
        ):
            raise ValueError(
                "coarse_seed_yaw_separation_deg must not be below "
                "coarse_yaw_step_deg"
            )
        if (
            self.coarse_seed_translation_separation_m
            > 2.0 * self.maximum_translation_m
        ):
            raise ValueError(
                "coarse_seed_translation_separation_m must not exceed "
                "the search diameter"
            )
        if (
            self.coarse_seed_yaw_separation_deg
            > 2.0 * self.maximum_yaw_deg
        ):
            raise ValueError(
                "coarse_seed_yaw_separation_deg must not exceed the yaw "
                "search span"
            )
        if (
            self.minimum_basin_translation_separation_m
            > 2.0 * self.maximum_translation_m
        ):
            raise ValueError(
                "minimum_basin_translation_separation_m must not exceed "
                "the search diameter"
            )
        if (
            self.minimum_basin_yaw_separation_deg
            > 2.0 * self.maximum_yaw_deg
        ):
            raise ValueError(
                "minimum_basin_yaw_separation_deg must not exceed the yaw "
                "search span"
            )


def terrain_prior_prefit_bypass_allowed(
    *, mode: str, initializes_global_gauge: bool
) -> bool:
    """Return whether a confirmed absolute factor may bypass graph prefit.

    A wider local ``recovery`` search is still centred on the drifting graph
    seed and is not a relocalization proof.  It must therefore pass the normal
    pose-graph innovation test.  Only initial two-strip gauge establishment,
    or an ``emergency``/``global`` hypothesis already confirmed by the
    disjoint-strip controller, may deliberately re-anchor a stale graph.
    """

    normalized_mode = str(mode).strip().lower()
    allowed_modes = {"normal", "recovery", "emergency", "global"}
    if normalized_mode not in allowed_modes:
        raise ValueError(f"unsupported terrain factor mode: {mode!r}")
    return bool(initializes_global_gauge) or normalized_mode in {
        "emergency",
        "global",
    }


def make_terrain_recovery_config(
    normal_config: TerrainMatcherConfig,
    *,
    maximum_translation_m: float,
) -> TerrainMatcherConfig:
    """Return the strict-rank-three wide-window recovery policy.

    Recovery deliberately uses the same patch, prior surface, residual gates,
    yaw window, and 100:1 ambiguity gate as normal matching.  Only the wider
    translation search and its coarse-search geometry change.  Requiring the
    normal matcher to be strict rank three prevents an accidental rank-two
    absolute fix from entering either path.
    """

    if not isinstance(normal_config, TerrainMatcherConfig):
        raise TypeError("normal_config must be a TerrainMatcherConfig")
    if not normal_config.require_yaw_observable:
        raise ValueError("terrain recovery requires strict rank-three matching")
    recovery_window = float(maximum_translation_m)
    if (
        not np.isfinite(recovery_window)
        or recovery_window <= normal_config.maximum_translation_m
    ):
        raise ValueError(
            "recovery translation window must be finite and wider than normal"
        )
    return replace(
        normal_config,
        maximum_translation_m=recovery_window,
        coarse_translation_step_m=10.0,
        coarse_top_k=16,
        coarse_seed_translation_separation_m=20.0,
        # One 5 m prior-map pixel is the largest distance that may still be
        # treated as the same refined solution.  In particular, the wide-search
        # seed NMS radius above must not hide 10--19 m aliases.
        minimum_basin_translation_separation_m=5.0,
        boundary_translation_margin_m=5.0,
    )


@dataclass
class TerrainRecoveryController:
    """Track observation failures and gate wide-window recovery attempts.

    An observation is complete only after both matching and pose-graph prefit.
    Consequently, callers must pass the *actual prior insertion* result to
    :meth:`should_attempt_recovery` and :meth:`complete_observation`.  The
    threshold includes the current failed normal attempt: with a threshold of
    three, recovery is first invoked on the third consecutive observation
    that fails to add a terrain prior.

    ``recovery_events`` counts recovery episodes, not matcher invocations.  A
    failed episode remains active across observations until a prior is truly
    inserted; a later episode starts only after such a success reset.
    """

    enabled: bool = False
    after_rejections: int = 3
    consecutive_failed_observations: int = 0
    maximum_consecutive_failed_observations: int = 0
    recovery_events: int = 0
    recovery_active: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise ValueError("enabled must be boolean")
        if (
            isinstance(self.after_rejections, bool)
            or int(self.after_rejections) != self.after_rejections
            or self.after_rejections < 1
        ):
            raise ValueError("after_rejections must be a positive integer")
        self.after_rejections = int(self.after_rejections)
        for name in (
            "consecutive_failed_observations",
            "maximum_consecutive_failed_observations",
            "recovery_events",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            setattr(self, name, int(value))
        if not isinstance(self.recovery_active, (bool, np.bool_)):
            raise ValueError("recovery_active must be boolean")
        if not self.enabled and self.recovery_active:
            raise ValueError("disabled recovery cannot start active")

    def should_attempt_recovery(self, *, normal_prior_added: bool) -> bool:
        """Return whether this observation needs the second matcher call."""

        if not isinstance(normal_prior_added, (bool, np.bool_)):
            raise ValueError("normal_prior_added must be boolean")
        return bool(
            self.enabled
            and not normal_prior_added
            and self.consecutive_failed_observations + 1
            >= self.after_rejections
        )

    def complete_observation(
        self,
        *,
        prior_added: bool,
        recovery_invoked: bool,
    ) -> None:
        """Commit one observation outcome; only insertion resets the streak."""

        if not isinstance(prior_added, (bool, np.bool_)):
            raise ValueError("prior_added must be boolean")
        if not isinstance(recovery_invoked, (bool, np.bool_)):
            raise ValueError("recovery_invoked must be boolean")
        if recovery_invoked and not self.enabled:
            raise ValueError("recovery_invoked cannot be true when disabled")
        if recovery_invoked and not self.recovery_active:
            self.recovery_events += 1
        if prior_added:
            self.consecutive_failed_observations = 0
            self.recovery_active = False
            return
        self.consecutive_failed_observations += 1
        self.maximum_consecutive_failed_observations = max(
            self.maximum_consecutive_failed_observations,
            self.consecutive_failed_observations,
        )
        if recovery_invoked:
            self.recovery_active = True


@dataclass(frozen=True)
class TerrainEmergencyRecoveryRequest:
    """Decision returned after one normal terrain-matching observation.

    ``translation_window_m`` and ``strict_rank_three`` make the intended
    recovery matcher invocation explicit at the integration boundary.  The
    controller itself is pure policy: it never runs a matcher and never
    consults simulator truth.
    """

    should_attempt: bool
    reason: str
    observation_index: int
    translation_window_m: float
    strict_rank_three: bool = True
    bypassed_throttle: bool = False


@dataclass(frozen=True)
class TerrainEmergencyRecoveryConfirmation:
    """Outcome of feeding one strict wide-window result to the controller.

    Only ``fuse_current_match=True`` authorizes a factor insertion.  A first
    successful match is retained solely as a pending hypothesis and is never
    returned for fusion.
    """

    fuse_current_match: bool
    status: str
    observation_index: int
    pending: bool
    position_disagreement_m: float = float("nan")
    yaw_disagreement_deg: float = float("nan")
    reference_observation_index: int | None = None
    acquisition_windows_disjoint: bool = False


@dataclass(frozen=True)
class _PendingTerrainEmergencyMatch:
    source: str
    observation_index: int
    timestamp: float
    raw_arc_m: float
    seed_pose2: tuple[float, float, float]
    matched_pose2: tuple[float, float, float]
    patch_start_timestamp: float
    patch_end_timestamp: float
    patch_start_observation_id: int
    patch_end_observation_id: int


@dataclass
class TerrainEmergencyRecoveryController:
    """Truth-free safety policy for rare wide-window terrain recovery.

    The normal matcher and the existing +/-100 m recovery remain the primary
    absolute-localization path.  Call :meth:`observe_preceding_recovery_outcome`
    once that complete preceding chain has finished; the reason supplied to
    this emergency layer is the final +/-100 m rejection whenever that search
    ran.  A +/-150 m strict-rank-three attempt is requested only after all
    three outage indicators agree: seven consecutive observations without an
    inserted terrain prior, 600 m of estimated raw-trajectory arc, and 400 s
    since the most recent insertion (or since tracking began).

    Ordinary weak-terrain causes are rate-limited to one attempt per three
    observations.  ``search_boundary_reached`` bypasses that rate limit so a
    drifting seed can immediately use the wider window, but it does not
    bypass the outage thresholds or a safety cooldown.  Ambiguity, map
    overlap, and every unlisted failure are fail-closed.

    A strict recovery result is never fused immediately.  It becomes a
    pending hypothesis; a second strict result must arrive in the next five
    observations after at least 60 m of estimated raw arc and 20 s.  In
    addition, its actual first acquisition timestamp and ping ID must both be
    later than the first patch's final timestamp and ID.  Thus empty returns
    cannot make two nominally separated rolling windows reuse old pings.  The
    first matched pose is propagated by the SE(2) increment between the two
    *seed* poses.
    Only when that prediction and the second matched pose agree within 20 m
    and 5 degrees is the second result returned for fusion.  An inconsistent
    or expired hypothesis starts a three-observation cooldown.

    Calling :meth:`consider_recovery_match` does not reset outage state, even
    after confirmation.  Only an actual graph-factor insertion, reported via
    :meth:`record_prior_insertion` or ``prior_added=True``, resets it.
    """

    enabled: bool = True
    failed_prior_threshold: int = 7
    minimum_raw_arc_m: float = 600.0
    minimum_time_since_prior_s: float = 400.0
    ordinary_attempt_interval_observations: int = 3
    recovery_translation_window_m: float = 150.0
    confirmation_minimum_raw_arc_m: float = 60.0
    confirmation_minimum_time_separation_s: float = 20.0
    confirmation_max_future_observations: int = 5
    confirmation_position_tolerance_m: float = 20.0
    confirmation_yaw_tolerance_deg: float = 5.0
    cooldown_observations: int = 3

    observation_index: int = field(default=0, init=False)
    consecutive_without_prior: int = field(default=0, init=False)
    raw_arc_since_prior_m: float = field(default=0.0, init=False)
    recovery_attempts: int = field(default=0, init=False)
    pending_hypotheses: int = field(default=0, init=False)
    confirmed_recoveries: int = field(default=0, init=False)
    inconsistent_hypotheses: int = field(default=0, init=False)
    expired_hypotheses: int = field(default=0, init=False)
    preceding_factor_rejections: int = field(default=0, init=False)

    _tracking_start_timestamp: float | None = field(
        default=None, init=False, repr=False
    )
    _last_prior_timestamp: float | None = field(
        default=None, init=False, repr=False
    )
    _current_timestamp: float | None = field(
        default=None, init=False, repr=False
    )
    _current_raw_pose2: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _previous_raw_pose2: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _last_attempt_observation: int | None = field(
        default=None, init=False, repr=False
    )
    _requested_observation: int | None = field(
        default=None, init=False, repr=False
    )
    _evaluated_observation: int | None = field(
        default=None, init=False, repr=False
    )
    _confirmed_factor_awaiting_outcome: bool = field(
        default=False, init=False, repr=False
    )
    _cooldown_until_observation: int = field(
        default=0, init=False, repr=False
    )
    _pending: _PendingTerrainEmergencyMatch | None = field(
        default=None, init=False, repr=False
    )

    _IMMEDIATE_REASON = "search_boundary_reached"
    _THROTTLED_REASONS = frozenset(
        {
            "yaw_unobservable",
            "unobservable_terrain",
            "insufficient_inliers",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise ValueError("enabled must be boolean")
        for name in (
            "failed_prior_threshold",
            "ordinary_attempt_interval_observations",
            "confirmation_max_future_observations",
            "cooldown_observations",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise ValueError(f"{name} must be a positive integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        for name in (
            "minimum_raw_arc_m",
            "minimum_time_since_prior_s",
            "recovery_translation_window_m",
            "confirmation_minimum_raw_arc_m",
            "confirmation_minimum_time_separation_s",
            "confirmation_position_tolerance_m",
            "confirmation_yaw_tolerance_deg",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, name, value)
        if self.recovery_translation_window_m > 150.0:
            raise ValueError(
                "recovery_translation_window_m must not exceed 150 m"
            )

    @staticmethod
    def _pose2(values: np.ndarray, name: str) -> np.ndarray:
        try:
            pose = np.asarray(values, dtype=np.float64).reshape(3).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must have three finite values") from exc
        if not np.isfinite(pose).all():
            raise ValueError(f"{name} must have three finite values")
        pose[2] = (pose[2] + np.pi) % (2.0 * np.pi) - np.pi
        return pose

    @staticmethod
    def _boolean(value: bool, name: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
        return bool(value)

    def _request(
        self,
        should_attempt: bool,
        reason: str,
        *,
        bypassed_throttle: bool = False,
    ) -> TerrainEmergencyRecoveryRequest:
        return TerrainEmergencyRecoveryRequest(
            should_attempt=bool(should_attempt),
            reason=str(reason),
            observation_index=self.observation_index,
            translation_window_m=self.recovery_translation_window_m,
            strict_rank_three=True,
            bypassed_throttle=bool(bypassed_throttle),
        )

    def _confirmation(
        self,
        fuse_current_match: bool,
        status: str,
        *,
        position_disagreement_m: float = float("nan"),
        yaw_disagreement_deg: float = float("nan"),
        reference_observation_index: int | None = None,
        acquisition_windows_disjoint: bool = False,
    ) -> TerrainEmergencyRecoveryConfirmation:
        reference = reference_observation_index
        if reference is None and self._pending is not None:
            reference = self._pending.observation_index
        return TerrainEmergencyRecoveryConfirmation(
            fuse_current_match=bool(fuse_current_match),
            status=str(status),
            observation_index=self.observation_index,
            pending=self._pending is not None,
            position_disagreement_m=float(position_disagreement_m),
            yaw_disagreement_deg=float(yaw_disagreement_deg),
            reference_observation_index=reference,
            acquisition_windows_disjoint=bool(
                acquisition_windows_disjoint
            ),
        )

    def _patch_interval(
        self,
        *,
        start_timestamp: float,
        end_timestamp: float,
        start_observation_id: int,
        end_observation_id: int,
    ) -> tuple[float, float, int, int]:
        try:
            start_time = float(start_timestamp)
            end_time = float(end_timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("patch timestamps must be finite") from exc
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError("patch timestamps must be finite")
        if start_time > end_time:
            raise ValueError("patch timestamps must be ordered")
        if (
            self._current_timestamp is None
            or end_time > self._current_timestamp + 1.0e-9
        ):
            raise ValueError("patch cannot end after the current observation")

        identifiers: list[int] = []
        for value in (start_observation_id, end_observation_id):
            try:
                invalid = (
                    isinstance(value, (bool, np.bool_))
                    or int(value) != value
                    or int(value) < 0
                )
            except (TypeError, ValueError, OverflowError):
                invalid = True
            if invalid:
                raise ValueError(
                    "patch observation IDs must be nonnegative integers"
                )
            identifiers.append(int(value))
        if identifiers[0] > identifiers[1]:
            raise ValueError("patch observation IDs must be ordered")
        return start_time, end_time, identifiers[0], identifiers[1]

    @property
    def has_pending_hypothesis(self) -> bool:
        return self._pending is not None

    @property
    def pending_confirmation_source(self) -> str | None:
        """Return the matcher family that created the pending hypothesis."""

        return None if self._pending is None else self._pending.source

    @property
    def last_prior_timestamp(self) -> float | None:
        return self._last_prior_timestamp

    @property
    def time_since_prior_s(self) -> float:
        if self._current_timestamp is None:
            return 0.0
        anchor = (
            self._last_prior_timestamp
            if self._last_prior_timestamp is not None
            else self._tracking_start_timestamp
        )
        if anchor is None:
            return 0.0
        return float(self._current_timestamp - anchor)

    @property
    def cooldown_remaining_observations(self) -> int:
        return max(
            0, self._cooldown_until_observation - self.observation_index
        )

    def _advance_observation(
        self, timestamp: float, raw_pose2: np.ndarray
    ) -> None:
        if self._confirmed_factor_awaiting_outcome:
            raise RuntimeError(
                "report confirmed factor insertion or rejection before the "
                "next observation"
            )
        try:
            current_timestamp = float(timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("timestamp must be finite") from exc
        if not np.isfinite(current_timestamp):
            raise ValueError("timestamp must be finite")
        if (
            self._current_timestamp is not None
            and current_timestamp <= self._current_timestamp
        ):
            raise ValueError("observation timestamps must increase")
        current_pose = self._pose2(raw_pose2, "raw_pose2")
        if self._tracking_start_timestamp is None:
            self._tracking_start_timestamp = current_timestamp
        if self._previous_raw_pose2 is not None:
            self.raw_arc_since_prior_m += float(
                np.linalg.norm(
                    current_pose[:2] - self._previous_raw_pose2[:2]
                )
            )
        self.observation_index += 1
        self._current_timestamp = current_timestamp
        self._current_raw_pose2 = current_pose
        self._previous_raw_pose2 = current_pose.copy()
        self._requested_observation = None
        self._evaluated_observation = None

    def _start_cooldown(self) -> None:
        self._cooldown_until_observation = max(
            self._cooldown_until_observation,
            self.observation_index + self.cooldown_observations,
        )

    def _reset_after_insertion(self) -> None:
        if self._current_timestamp is None or self._current_raw_pose2 is None:
            raise RuntimeError("no observation is available for prior insertion")
        self.consecutive_without_prior = 0
        self.raw_arc_since_prior_m = 0.0
        self._last_prior_timestamp = self._current_timestamp
        self._pending = None
        self._last_attempt_observation = None
        self._requested_observation = None
        self._evaluated_observation = None
        self._confirmed_factor_awaiting_outcome = False
        self._cooldown_until_observation = 0

    def record_prior_insertion(self) -> None:
        """Reset outage state after the caller actually inserted a factor."""

        self._reset_after_insertion()

    def record_factor_rejection(self) -> None:
        """Report that a confirmed emergency factor was not inserted.

        Pose-graph pre-fit and insertion are deliberately outside this pure
        policy object.  A rejection consumes the confirmed candidate and
        starts a three-observation cooldown, but it cannot reset the outage
        counters, distance, or time.
        """

        if not self._confirmed_factor_awaiting_outcome:
            raise RuntimeError("no confirmed emergency factor awaits an outcome")
        self._confirmed_factor_awaiting_outcome = False
        self._pending = None
        self._start_cooldown()

    def observe_preceding_recovery_outcome(
        self,
        *,
        timestamp: float,
        raw_pose2: np.ndarray,
        prior_added: bool,
        preceding_recovery_invoked: bool,
        preceding_recovery_rejection_reason: str | None,
        preceding_factor_rejected: bool = False,
    ) -> TerrainEmergencyRecoveryRequest:
        """Record the preceding +/-30 then +/-100 chain and gate emergency.

        ``prior_added`` means an actual graph insertion by either preceding
        matcher, not merely matcher acceptance.  Once the +/-100 m recovery
        has run, ``preceding_recovery_rejection_reason`` must be its final
        reason and ``preceding_recovery_invoked`` must be true; this explicit
        evidence prevents the emergency layer from bypassing +/-100 m even
        when its cadence is configured later than this controller's seventh
        failure threshold.  Set ``preceding_factor_rejected`` when a normal
        or +/-100 m matcher succeeded but pose-graph pre-fit/insertion
        rejected its candidate.  That outcome invalidates any older pending
        hypothesis and cools the emergency path without resetting outage
        state.
        """

        factor_added = self._boolean(prior_added, "prior_added")
        preceding_invoked = self._boolean(
            preceding_recovery_invoked, "preceding_recovery_invoked"
        )
        preceding_rejected = self._boolean(
            preceding_factor_rejected, "preceding_factor_rejected"
        )
        if factor_added and preceding_rejected:
            raise ValueError(
                "prior_added and preceding_factor_rejected cannot both be true"
            )
        if preceding_recovery_rejection_reason is not None and not isinstance(
            preceding_recovery_rejection_reason, str
        ):
            raise ValueError(
                "preceding_recovery_rejection_reason must be a string or None"
            )
        self._advance_observation(timestamp, raw_pose2)

        if factor_added:
            self._reset_after_insertion()
            return self._request(False, "prior_inserted")

        self.consecutive_without_prior += 1
        if preceding_rejected:
            self._pending = None
            self.preceding_factor_rejections += 1
            self._start_cooldown()
            return self._request(False, "preceding_factor_rejected")

        expired_now = False
        if self._pending is not None and (
            self.observation_index - self._pending.observation_index
            > self.confirmation_max_future_observations
        ):
            self._pending = None
            self.expired_hypotheses += 1
            self._start_cooldown()
            expired_now = True

        if not preceding_invoked:
            return self._request(False, "preceding_recovery_not_invoked")
        if not self.enabled:
            return self._request(False, "disabled")
        if expired_now:
            return self._request(False, "pending_expired")
        if self.consecutive_without_prior < self.failed_prior_threshold:
            return self._request(False, "failure_threshold")
        if self.raw_arc_since_prior_m < self.minimum_raw_arc_m:
            return self._request(False, "raw_arc_threshold")
        if self.time_since_prior_s < self.minimum_time_since_prior_s:
            return self._request(False, "time_threshold")
        if self.observation_index <= self._cooldown_until_observation:
            return self._request(False, "cooldown")

        reason = preceding_recovery_rejection_reason
        if reason == self._IMMEDIATE_REASON:
            bypassed = bool(
                self._last_attempt_observation is not None
                and self.observation_index - self._last_attempt_observation
                < self.ordinary_attempt_interval_observations
            )
        elif reason in self._THROTTLED_REASONS:
            # Once a first strict hypothesis exists, every allowed observation
            # in its short five-observation lifetime is useful confirmation
            # evidence.  The pending lifetime remains the rate bound here.
            if self._pending is not None:
                bypassed = bool(
                    self._last_attempt_observation is not None
                    and self.observation_index
                    - self._last_attempt_observation
                    < self.ordinary_attempt_interval_observations
                )
            elif self._last_attempt_observation is not None and (
                self.observation_index - self._last_attempt_observation
                < self.ordinary_attempt_interval_observations
            ):
                return self._request(False, "ordinary_throttle")
            else:
                bypassed = False
        else:
            return self._request(False, "reason_not_allowed")

        self._last_attempt_observation = self.observation_index
        self._requested_observation = self.observation_index
        self.recovery_attempts += 1
        return self._request(
            True,
            "search_boundary_immediate"
            if reason == self._IMMEDIATE_REASON
            else (
                "pending_confirmation"
                if self._pending is not None
                else "ordinary_allowed"
            ),
            bypassed_throttle=bypassed,
        )

    @staticmethod
    def _propagate_matched_pose(
        first_seed: np.ndarray,
        second_seed: np.ndarray,
        first_matched: np.ndarray,
    ) -> np.ndarray:
        """Propagate a matched pose by the seed trajectory's SE(2) increment."""

        delta_world = second_seed[:2] - first_seed[:2]
        cosine = math.cos(float(first_seed[2]))
        sine = math.sin(float(first_seed[2]))
        delta_local = np.array(
            [
                cosine * delta_world[0] + sine * delta_world[1],
                -sine * delta_world[0] + cosine * delta_world[1],
            ],
            dtype=np.float64,
        )
        matched_cosine = math.cos(float(first_matched[2]))
        matched_sine = math.sin(float(first_matched[2]))
        propagated = np.empty(3, dtype=np.float64)
        propagated[:2] = first_matched[:2] + np.array(
            [
                matched_cosine * delta_local[0]
                - matched_sine * delta_local[1],
                matched_sine * delta_local[0]
                + matched_cosine * delta_local[1],
            ]
        )
        seed_yaw_delta = (
            second_seed[2] - first_seed[2] + np.pi
        ) % (2.0 * np.pi) - np.pi
        propagated[2] = (
            first_matched[2] + seed_yaw_delta + np.pi
        ) % (2.0 * np.pi) - np.pi
        return propagated

    def consider_recovery_match(
        self,
        *,
        accepted: bool,
        observable_rank: int,
        translation_observable_rank: int,
        yaw_observable: bool,
        seed_pose2: np.ndarray,
        matched_pose2: np.ndarray,
        patch_start_timestamp: float,
        patch_end_timestamp: float,
        patch_start_observation_id: int,
        patch_end_observation_id: int,
        confirmation_position_tolerance_m: float | None = None,
        confirmation_yaw_tolerance_deg: float | None = None,
        confirmation_source: str = "emergency",
        externally_confirmed_independent_evidence: int | None = None,
    ) -> TerrainEmergencyRecoveryConfirmation:
        """Validate one requested strict result and possibly confirm fusion.

        The optional per-match tolerances support a longer, independently
        configured whole-map strip.  Omitting them preserves the controller's
        ordinary bounded-recovery policy.  A separate persistent
        multi-hypothesis policy may supply
        ``externally_confirmed_independent_evidence``.  Two or more genuinely
        disjoint windows then replace this controller's older two-strip hold;
        the strict rank-three checks still apply and the graph insertion must
        still report its outcome.
        """

        if self._requested_observation != self.observation_index:
            raise RuntimeError(
                "a recovery match was not requested for this observation"
            )
        if self._evaluated_observation == self.observation_index:
            raise RuntimeError("recovery result already evaluated")
        self._evaluated_observation = self.observation_index
        match_accepted = self._boolean(accepted, "accepted")
        yaw_is_observable = self._boolean(
            yaw_observable, "yaw_observable"
        )
        if (
            isinstance(observable_rank, (bool, np.bool_))
            or int(observable_rank) != observable_rank
            or not 0 <= int(observable_rank) <= 3
        ):
            raise ValueError("observable_rank must be an integer from zero to three")
        rank = int(observable_rank)
        if (
            isinstance(translation_observable_rank, (bool, np.bool_))
            or int(translation_observable_rank)
            != translation_observable_rank
            or not 0 <= int(translation_observable_rank) <= 2
        ):
            raise ValueError(
                "translation_observable_rank must be an integer from zero to two"
            )
        translation_rank = int(translation_observable_rank)
        seed = self._pose2(seed_pose2, "seed_pose2")
        matched = self._pose2(matched_pose2, "matched_pose2")
        if not isinstance(confirmation_source, str) or not confirmation_source:
            raise ValueError("confirmation_source must be a nonempty string")
        external_evidence = None
        if externally_confirmed_independent_evidence is not None:
            value = externally_confirmed_independent_evidence
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 2
            ):
                raise ValueError(
                    "externally confirmed evidence must contain at least two "
                    "independent windows"
                )
            external_evidence = int(value)
        position_tolerance_m = (
            self.confirmation_position_tolerance_m
            if confirmation_position_tolerance_m is None
            else float(confirmation_position_tolerance_m)
        )
        yaw_tolerance_deg = (
            self.confirmation_yaw_tolerance_deg
            if confirmation_yaw_tolerance_deg is None
            else float(confirmation_yaw_tolerance_deg)
        )
        if not np.isfinite(position_tolerance_m) or position_tolerance_m <= 0.0:
            raise ValueError(
                "confirmation_position_tolerance_m must be positive and finite"
            )
        if not np.isfinite(yaw_tolerance_deg) or yaw_tolerance_deg <= 0.0:
            raise ValueError(
                "confirmation_yaw_tolerance_deg must be positive and finite"
            )
        (
            patch_start_time,
            patch_end_time,
            patch_start_id,
            patch_end_id,
        ) = self._patch_interval(
            start_timestamp=patch_start_timestamp,
            end_timestamp=patch_end_timestamp,
            start_observation_id=patch_start_observation_id,
            end_observation_id=patch_end_observation_id,
        )

        if not match_accepted:
            return self._confirmation(False, "recovery_rejected")
        if rank != 3 or translation_rank != 2 or not yaw_is_observable:
            return self._confirmation(False, "strict_rank_three_required")

        if external_evidence is not None:
            # The external cohort filter has already propagated and compared
            # each disjoint window and evaluated the explicit null state.  Do
            # not demand a fourth strip through the legacy pending-hypothesis
            # mechanism.  Mark an outcome as outstanding so a rejected graph
            # insertion still enters the ordinary cooldown path.
            self._pending = None
            self.confirmed_recoveries += 1
            self._confirmed_factor_awaiting_outcome = True
            return self._confirmation(
                True,
                "confirmed_by_persistent_cohort",
                acquisition_windows_disjoint=True,
            )

        if self._pending is None:
            self._pending = _PendingTerrainEmergencyMatch(
                source=confirmation_source,
                observation_index=self.observation_index,
                timestamp=float(self._current_timestamp),
                raw_arc_m=self.raw_arc_since_prior_m,
                seed_pose2=tuple(float(value) for value in seed),
                matched_pose2=tuple(float(value) for value in matched),
                patch_start_timestamp=patch_start_time,
                patch_end_timestamp=patch_end_time,
                patch_start_observation_id=patch_start_id,
                patch_end_observation_id=patch_end_id,
            )
            self.pending_hypotheses += 1
            return self._confirmation(False, "pending_created")

        if confirmation_source != self._pending.source:
            return self._confirmation(False, "pending_wait_matching_source")

        time_separation_s = float(
            self._current_timestamp - self._pending.timestamp
        )
        if time_separation_s < self.confirmation_minimum_time_separation_s:
            return self._confirmation(False, "pending_wait_independent_strip")

        windows_disjoint = bool(
            patch_start_time > self._pending.patch_end_timestamp
            and patch_start_id > self._pending.patch_end_observation_id
        )
        if not windows_disjoint:
            return self._confirmation(
                False, "pending_wait_nonoverlapping_patch"
            )

        raw_arc_separation = (
            self.raw_arc_since_prior_m - self._pending.raw_arc_m
        )
        if raw_arc_separation < self.confirmation_minimum_raw_arc_m:
            return self._confirmation(False, "pending_wait_distance")

        first_seed = np.asarray(self._pending.seed_pose2, dtype=np.float64)
        first_matched = np.asarray(
            self._pending.matched_pose2, dtype=np.float64
        )
        predicted = self._propagate_matched_pose(
            first_seed, seed, first_matched
        )
        position_disagreement = float(
            np.linalg.norm(predicted[:2] - matched[:2])
        )
        yaw_disagreement = float(
            abs(
                (predicted[2] - matched[2] + np.pi)
                % (2.0 * np.pi)
                - np.pi
            )
        )
        yaw_disagreement_deg = float(np.rad2deg(yaw_disagreement))
        if (
            position_disagreement
            > position_tolerance_m
            or yaw_disagreement_deg
            > yaw_tolerance_deg
        ):
            self._pending = None
            self.inconsistent_hypotheses += 1
            self._start_cooldown()
            return self._confirmation(
                False,
                "confirmation_inconsistent",
                position_disagreement_m=position_disagreement,
                yaw_disagreement_deg=yaw_disagreement_deg,
            )

        reference_observation_index = self._pending.observation_index
        self._pending = None
        self.confirmed_recoveries += 1
        self._confirmed_factor_awaiting_outcome = True
        return self._confirmation(
            True,
            "confirmed_current_match",
            position_disagreement_m=position_disagreement,
            yaw_disagreement_deg=yaw_disagreement_deg,
            reference_observation_index=reference_observation_index,
            acquisition_windows_disjoint=windows_disjoint,
        )


def _readonly_array(values, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(shape).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TerrainMatchResult:
    """Diagnostics and planar pseudo-measurement from one terrain match.

    ``correction`` is ordered ``[dx_world_m, dy_world_m, dyaw_world_rad]``.
    ``covariance`` and ``information`` use the same order.  The base pose is
    retained so :func:`apply_terrain_match` can reject a result if IMU/DVL
    propagation changed the filter between matching and fusion.

    Basin costs are robust mean losses in squared-metre units.  Their gap is
    converted to ``basin_log_likelihood_ratio`` using the configured vertical
    measurement variance and effective independent point count.  An infinite
    second cost means all retained coarse hypotheses converged to one basin.
    """

    accepted: bool
    correction: np.ndarray
    covariance: np.ndarray
    information: np.ndarray
    correspondences: int
    inliers: int
    iterations: int
    coarse_candidates: int
    distinct_basin_count: int
    best_basin_cost: float
    second_best_basin_cost: float
    basin_cost_gap: float
    basin_log_likelihood_ratio: float
    rmse_before: float
    rmse_after: float
    reduced_chi_square: float
    innovation_nis: float
    observable_rank: int
    translation_observable_rank: int
    yaw_observable: bool
    observable_projector_scaled: np.ndarray
    scaled_information_eigenvalues: np.ndarray
    information_condition: float
    reached_search_boundary: bool
    rejection_reason: str | None
    state_timestamp: float
    base_position: np.ndarray
    base_rotation: np.ndarray
    maximum_innovation_nis: float
    innovation_nis_gate_enabled: bool
    yaw_rank_scale_m: float
    minimum_scaled_information_eigenvalue: float
    minimum_information_eigen_ratio: float
    minimum_basin_log_likelihood_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "correction", _readonly_array(self.correction, (3,))
        )
        object.__setattr__(
            self, "covariance", _readonly_array(self.covariance, (3, 3))
        )
        object.__setattr__(
            self, "information", _readonly_array(self.information, (3, 3))
        )
        object.__setattr__(
            self,
            "observable_projector_scaled",
            _readonly_array(self.observable_projector_scaled, (3, 3)),
        )
        object.__setattr__(
            self,
            "scaled_information_eigenvalues",
            _readonly_array(self.scaled_information_eigenvalues, (3,)),
        )
        object.__setattr__(
            self, "base_position", _readonly_array(self.base_position, (3,))
        )
        object.__setattr__(
            self, "base_rotation", _readonly_array(self.base_rotation, (3, 3))
        )

    @property
    def translation_correction_m(self) -> float:
        return float(np.linalg.norm(self.correction[:2]))

    @property
    def yaw_correction_deg(self) -> float:
        return float(np.rad2deg(self.correction[2]))

    @property
    def matched_pose2(self) -> np.ndarray:
        """Return absolute ``[x_world, y_world, yaw_world]`` for a graph prior."""

        base_yaw = math.atan2(self.base_rotation[1, 0], self.base_rotation[0, 0])
        yaw = (base_yaw + self.correction[2] + np.pi) % (2.0 * np.pi) - np.pi
        return np.array(
            [
                self.base_position[0] + self.correction[0],
                self.base_position[1] + self.correction[1],
                yaw,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class BackscatterMatcherConfig:
    """Configuration for direct georeferenced side-scan texture matching.

    The map stores normalized survey backscatter rather than calibrated target
    strength.  Matching therefore uses zero-mean normalized correlation after
    projecting both observation and map samples away from a smooth
    port/starboard and slant-range trend.  The reported covariance has explicit
    floors tied to the 5 m registered map and is not inferred from the number
    of densely correlated image pixels.
    """

    maximum_points: int = 2048
    minimum_points: int = 120
    minimum_valid_fraction: float = 0.85
    maximum_translation_m: float = 10.0
    maximum_yaw_deg: float = 2.5
    coarse_translation_step_m: float = 2.5
    coarse_yaw_step_deg: float = 0.5
    refinement_translation_step_m: float = 0.5
    refinement_yaw_step_deg: float = 0.1
    minimum_correlation: float = 0.12
    minimum_basin_correlation_gap: float = 0.015
    basin_translation_separation_m: float = 5.0
    basin_yaw_separation_deg: float = 1.0
    minimum_texture_std: float = 0.025
    radiometric_log_floor: float = 0.01
    horizontal_std_m: float = 2.5
    yaw_std_deg: float = 1.5
    # Optional scale-space/context scoring for long global strips.  Empty
    # scales preserve the original pointwise matcher exactly.  This adapts the
    # reliability principles of SC-Match to our different input contract
    # (geocoded reflectivity samples against a prior mosaic), rather than
    # claiming to reproduce its image-feature pipeline.
    scale_space_cell_sizes_m: tuple[float, ...] = ()
    scale_space_weight: float = 0.0
    context_consistency_weight: float = 0.0
    context_length_m: float = 40.0
    minimum_scale_cells: int = 12
    minimum_context_points: int = 24

    def __post_init__(self) -> None:
        for name in (
            "maximum_points",
            "minimum_points",
            "minimum_scale_cells",
            "minimum_context_points",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_points < self.minimum_points:
            raise ValueError("maximum_points must not be below minimum_points")
        positive = (
            "maximum_translation_m",
            "maximum_yaw_deg",
            "coarse_translation_step_m",
            "coarse_yaw_step_deg",
            "refinement_translation_step_m",
            "refinement_yaw_step_deg",
            "basin_translation_separation_m",
            "basin_yaw_separation_deg",
            "minimum_texture_std",
            "radiometric_log_floor",
            "horizontal_std_m",
            "yaw_std_deg",
            "context_length_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        fractions = (
            "minimum_valid_fraction",
            "minimum_correlation",
            "minimum_basin_correlation_gap",
        )
        for name in fractions:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        if self.coarse_translation_step_m > self.maximum_translation_m:
            raise ValueError("coarse translation step exceeds search window")
        if self.coarse_yaw_step_deg > self.maximum_yaw_deg:
            raise ValueError("coarse yaw step exceeds search window")
        try:
            scales = tuple(float(value) for value in self.scale_space_cell_sizes_m)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "scale_space_cell_sizes_m must contain finite positive values"
            ) from exc
        if any(not np.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError(
                "scale_space_cell_sizes_m must contain finite positive values"
            )
        if tuple(sorted(set(scales))) != scales:
            raise ValueError(
                "scale_space_cell_sizes_m must be unique and increasing"
            )
        object.__setattr__(self, "scale_space_cell_sizes_m", scales)
        for name in ("scale_space_weight", "context_consistency_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
            object.__setattr__(self, name, value)
        if self.scale_space_weight > 0.0 and not scales:
            raise ValueError("positive scale_space_weight requires scale sizes")
        if self.scale_space_weight + self.context_consistency_weight >= 1.0:
            raise ValueError(
                "scale-space and context weights must sum to less than one"
            )


@dataclass(frozen=True)
class BackscatterMatchResult:
    """Direct side-scan-to-survey-map registration result."""

    accepted: bool
    correction: np.ndarray
    matched_pose2: np.ndarray
    information: np.ndarray
    covariance: np.ndarray
    correlation_before: float
    correlation_after: float
    second_basin_correlation: float
    basin_correlation_gap: float
    correspondences: int
    valid_fraction: float
    observation_texture_std: float
    map_texture_std: float
    evaluated_candidates: int
    fine_correlation_after: float = float("nan")
    scale_space_correlation_after: float = float("nan")
    context_consistency_after: float = float("nan")
    valid_contexts_after: int = 0
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "correction", _readonly_array(self.correction, (3,))
        )
        object.__setattr__(
            self, "matched_pose2", _readonly_array(self.matched_pose2, (3,))
        )
        object.__setattr__(
            self, "information", _readonly_array(self.information, (3, 3))
        )
        object.__setattr__(
            self, "covariance", _readonly_array(self.covariance, (3, 3))
        )


class BackscatterMapMatcher:
    """Align motion-compensated side-scan reflectivity to a prior mosaic."""

    def __init__(
        self,
        terrain: MeasuredBathymetryGrid,
        config: BackscatterMatcherConfig | None = None,
    ) -> None:
        if not isinstance(terrain, MeasuredBathymetryGrid):
            raise TypeError("terrain must be a MeasuredBathymetryGrid")
        if not terrain.has_measured_backscatter:
            raise ValueError("terrain has no measured backscatter")
        self.terrain = terrain
        self.config = config or BackscatterMatcherConfig()
        self._relative_power = uses_relative_power(terrain.metadata)

    @staticmethod
    def _axis_samples(limit: float, step: float) -> np.ndarray:
        count = int(math.floor(limit / step))
        values = np.arange(-count, count + 1, dtype=np.float64) * step
        return np.unique(np.concatenate((values, [-limit, 0.0, limit])))

    @staticmethod
    def _radiometric_design(points_body: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _remove_trend(values: np.ndarray, design: np.ndarray) -> np.ndarray:
        try:
            coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        except np.linalg.LinAlgError:
            coefficients = np.zeros(design.shape[1], dtype=np.float64)
            coefficients[0] = float(np.mean(values))
        return values - design @ coefficients

    @staticmethod
    def _spatial_group_ids(
        points_body: np.ndarray, cell_size_m: float
    ) -> np.ndarray:
        cells = np.floor(points_body[:, :2] / float(cell_size_m)).astype(
            np.int64
        )
        return np.unique(cells, axis=0, return_inverse=True)[1]

    @staticmethod
    def _context_group_ids(
        points_body: np.ndarray, context_length_m: float
    ) -> np.ndarray:
        along_track = points_body[:, 0]
        start = float(np.min(along_track))
        segment = np.floor(
            (along_track - start) / float(context_length_m)
        ).astype(np.int64)
        side = (points_body[:, 1] >= 0.0).astype(np.int64)
        pairs = np.column_stack((segment, side))
        return np.unique(pairs, axis=0, return_inverse=True)[1]

    @staticmethod
    def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
        first_centered = first - float(np.mean(first))
        second_centered = second - float(np.mean(second))
        denominator = float(
            np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
        )
        if denominator <= 1.0e-12:
            return float("nan")
        return float(np.dot(first_centered, second_centered) / denominator)

    def _scale_space_correlation(
        self,
        observation_texture: np.ndarray,
        map_texture: np.ndarray,
        scale_group_ids: tuple[np.ndarray, ...],
        valid: np.ndarray,
    ) -> float:
        if self.config.scale_space_weight <= 0.0:
            return float("nan")
        correlations: list[float] = []
        for all_group_ids in scale_group_ids:
            group_ids = all_group_ids[valid]
            group_count = int(np.max(group_ids)) + 1 if group_ids.size else 0
            if group_count < self.config.minimum_scale_cells:
                continue
            counts = np.bincount(group_ids, minlength=group_count)
            usable = counts >= 2
            if np.count_nonzero(usable) < self.config.minimum_scale_cells:
                continue
            observation_means = np.bincount(
                group_ids,
                weights=observation_texture,
                minlength=group_count,
            ) / np.maximum(counts, 1)
            map_means = np.bincount(
                group_ids,
                weights=map_texture,
                minlength=group_count,
            ) / np.maximum(counts, 1)
            correlation = self._normalized_correlation(
                observation_means[usable], map_means[usable]
            )
            if np.isfinite(correlation):
                correlations.append(correlation)
        if not correlations:
            return float("nan")
        # A median prevents one resolution from dominating simply because a
        # repeated texture happens to correlate at that cell size.
        return float(np.median(correlations))

    def _context_consistency(
        self,
        observation_texture: np.ndarray,
        map_texture: np.ndarray,
        all_context_ids: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[float, int]:
        if self.config.context_consistency_weight <= 0.0:
            return float("nan"), 0
        context_ids = all_context_ids[valid]
        correlations: list[float] = []
        for context_id in np.unique(context_ids):
            selected = context_ids == context_id
            if np.count_nonzero(selected) < self.config.minimum_context_points:
                continue
            correlation = self._normalized_correlation(
                observation_texture[selected], map_texture[selected]
            )
            if np.isfinite(correlation):
                correlations.append(correlation)
        if not correlations:
            return float("nan"), 0
        values = np.asarray(correlations, dtype=np.float64)
        positive_fraction = float(np.mean(values > 0.0))
        consistency = 0.5 * float(np.median(values)) + 0.5 * (
            2.0 * positive_fraction - 1.0
        )
        return float(np.clip(consistency, -1.0, 1.0)), int(values.size)

    def _score(
        self,
        correction: np.ndarray,
        state: InertialState,
        offsets_world: np.ndarray,
        observed: np.ndarray,
        design: np.ndarray,
        scale_group_ids: tuple[np.ndarray, ...],
        context_group_ids: np.ndarray,
    ) -> tuple[float, int, float, float, float, float, float, float, int]:
        yaw = float(correction[2])
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        offset_x = cosine * offsets_world[:, 0] - sine * offsets_world[:, 1]
        offset_y = sine * offsets_world[:, 0] + cosine * offsets_world[:, 1]
        x = state.position[0] + correction[0] + offset_x
        y = state.position[1] + correction[1] + offset_y
        predicted = np.asarray(
            self.terrain.measured_reflectivity(x, y), dtype=np.float64
        ).reshape(-1)
        valid = np.isfinite(predicted) & np.isfinite(observed)
        count = int(np.count_nonzero(valid))
        valid_fraction = count / max(observed.size, 1)
        if (
            count < self.config.minimum_points
            or valid_fraction < self.config.minimum_valid_fraction
        ):
            return (
                float("-inf"), count, valid_fraction, 0.0, 0.0,
                float("nan"), float("nan"), float("nan"), 0,
            )
        selected_design = design[valid]
        # The fast sonar model is multiplicative: reflectivity is multiplied
        # by incidence, spreading, absorption and a per-ping normalization.
        # Taking log intensity turns those nuisance gains into additive smooth
        # trends that the design matrix can remove.  A documented floor limits
        # the influence of weak returns dominated by additive receiver noise.
        observation_texture = self._remove_trend(
            log_backscatter(observed[valid], self.config.radiometric_log_floor,
                            relative_power=self._relative_power),
            selected_design,
        )
        map_texture = self._remove_trend(
            log_backscatter(predicted[valid], self.config.radiometric_log_floor,
                            relative_power=self._relative_power),
            selected_design,
        )
        observation_std = float(np.sqrt(np.mean(observation_texture**2)))
        map_std = float(np.sqrt(np.mean(map_texture**2)))
        if (
            observation_std < self.config.minimum_texture_std
            or map_std < self.config.minimum_texture_std
        ):
            return (
                float("-inf"),
                count,
                valid_fraction,
                observation_std,
                map_std,
                float("nan"),
                float("nan"),
                float("nan"),
                0,
            )
        denominator = max(
            float(
                np.linalg.norm(observation_texture)
                * np.linalg.norm(map_texture)
            ),
            1.0e-12,
        )
        fine_correlation = float(
            np.dot(observation_texture, map_texture) / denominator
        )
        scale_correlation = self._scale_space_correlation(
            observation_texture,
            map_texture,
            scale_group_ids,
            valid,
        )
        context_consistency, valid_contexts = self._context_consistency(
            observation_texture,
            map_texture,
            context_group_ids,
            valid,
        )
        scale_weight = (
            self.config.scale_space_weight
            if np.isfinite(scale_correlation)
            else 0.0
        )
        context_weight = (
            self.config.context_consistency_weight
            if np.isfinite(context_consistency)
            else 0.0
        )
        fine_weight = 1.0 - scale_weight - context_weight
        correlation = fine_weight * fine_correlation
        if scale_weight:
            correlation += scale_weight * scale_correlation
        if context_weight:
            correlation += context_weight * context_consistency
        return (
            float(np.clip(correlation, -1.0, 1.0)),
            count,
            valid_fraction,
            observation_std,
            map_std,
            fine_correlation,
            scale_correlation,
            context_consistency,
            valid_contexts,
        )

    def _rejected(
        self,
        reason: str,
        base_pose2: np.ndarray,
        *,
        correction: np.ndarray | None = None,
        correlation_before: float = float("nan"),
        correlation_after: float = float("nan"),
        second_basin_correlation: float = float("nan"),
        basin_correlation_gap: float = float("nan"),
        correspondences: int = 0,
        valid_fraction: float = 0.0,
        observation_texture_std: float = 0.0,
        map_texture_std: float = 0.0,
        evaluated_candidates: int = 0,
        fine_correlation_after: float = float("nan"),
        scale_space_correlation_after: float = float("nan"),
        context_consistency_after: float = float("nan"),
        valid_contexts_after: int = 0,
    ) -> BackscatterMatchResult:
        rejected_correction = (
            np.zeros(3, dtype=np.float64)
            if correction is None
            else np.asarray(correction, dtype=np.float64).reshape(3).copy()
        )
        rejected_pose2 = np.asarray(base_pose2, dtype=np.float64).copy()
        rejected_pose2 += rejected_correction
        rejected_pose2[2] = math.atan2(
            math.sin(rejected_pose2[2]), math.cos(rejected_pose2[2])
        )
        covariance = np.diag(
            [
                self.config.horizontal_std_m**2,
                self.config.horizontal_std_m**2,
                np.deg2rad(self.config.yaw_std_deg) ** 2,
            ]
        )
        return BackscatterMatchResult(
            accepted=False,
            correction=rejected_correction,
            matched_pose2=rejected_pose2,
            information=np.linalg.inv(covariance),
            covariance=covariance,
            correlation_before=correlation_before,
            correlation_after=correlation_after,
            second_basin_correlation=second_basin_correlation,
            basin_correlation_gap=basin_correlation_gap,
            correspondences=correspondences,
            valid_fraction=valid_fraction,
            observation_texture_std=observation_texture_std,
            map_texture_std=map_texture_std,
            evaluated_candidates=evaluated_candidates,
            fine_correlation_after=fine_correlation_after,
            scale_space_correlation_after=scale_space_correlation_after,
            context_consistency_after=context_consistency_after,
            valid_contexts_after=valid_contexts_after,
            rejection_reason=reason,
        )

    def match(
        self,
        points_body: np.ndarray,
        intensities: np.ndarray,
        state: InertialState,
    ) -> BackscatterMatchResult:
        """Register one body-frame strip around a bathymetric pose seed."""

        if not isinstance(state, InertialState):
            raise TypeError("state must be an InertialState")
        points = np.asarray(points_body, dtype=np.float64).reshape(-1, 3)
        values = np.asarray(intensities, dtype=np.float64).reshape(-1)
        if points.shape[0] != values.size:
            raise ValueError("points_body and intensities must have equal size")
        valid = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(values)
            & (values > 0.0)
        )
        points = points[valid]
        values = values[valid]
        base_yaw = math.atan2(state.rotation[1, 0], state.rotation[0, 0])
        base_pose2 = np.array(
            [state.position[0], state.position[1], base_yaw], dtype=np.float64
        )
        if points.shape[0] < self.config.minimum_points:
            return self._rejected("insufficient_points", base_pose2)
        if points.shape[0] > self.config.maximum_points:
            selected = np.linspace(
                0,
                points.shape[0] - 1,
                self.config.maximum_points,
                dtype=np.int64,
            )
            points = points[selected]
            values = values[selected]

        design = self._radiometric_design(points)
        scale_group_ids = tuple(
            self._spatial_group_ids(points, cell_size_m)
            for cell_size_m in self.config.scale_space_cell_sizes_m
        )
        context_group_ids = self._context_group_ids(
            points, self.config.context_length_m
        )
        offsets_world = points @ state.rotation.T
        zero = np.zeros(3, dtype=np.float64)
        before = self._score(
            zero,
            state,
            offsets_world,
            values,
            design,
            scale_group_ids,
            context_group_ids,
        )[0]
        translation_values = self._axis_samples(
            self.config.maximum_translation_m,
            self.config.coarse_translation_step_m,
        )
        yaw_values = np.deg2rad(
            self._axis_samples(
                self.config.maximum_yaw_deg,
                self.config.coarse_yaw_step_deg,
            )
        )
        candidates: list[tuple[float, np.ndarray, tuple]] = []
        evaluated = 0
        supported_diagnostics = None
        for yaw in yaw_values:
            for dx in translation_values:
                for dy in translation_values:
                    if math.hypot(float(dx), float(dy)) > (
                        self.config.maximum_translation_m + 1.0e-12
                    ):
                        continue
                    correction = np.array([dx, dy, yaw], dtype=np.float64)
                    diagnostics = self._score(
                        correction,
                        state,
                        offsets_world,
                        values,
                        design,
                        scale_group_ids,
                        context_group_ids,
                    )
                    evaluated += 1
                    if (
                        diagnostics[1] >= self.config.minimum_points
                        and diagnostics[2] >= self.config.minimum_valid_fraction
                        and (
                            supported_diagnostics is None
                            or min(diagnostics[3:5]) > min(supported_diagnostics[3:5])
                        )
                    ):
                        supported_diagnostics = diagnostics
                    if np.isfinite(diagnostics[0]):
                        candidates.append(
                            (float(diagnostics[0]), correction, diagnostics)
                        )
        if not candidates:
            support = supported_diagnostics
            return self._rejected(
                "insufficient_backscatter_texture" if support is not None
                else "insufficient_map_overlap",
                base_pose2,
                correlation_before=before,
                correspondences=0 if support is None else support[1],
                valid_fraction=0.0 if support is None else support[2],
                observation_texture_std=0.0 if support is None else support[3],
                map_texture_std=0.0 if support is None else support[4],
                evaluated_candidates=evaluated,
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        coarse_best = candidates[0][1]

        refinement_xy = np.arange(-4, 5, dtype=np.float64) * (
            self.config.refinement_translation_step_m
        )
        refinement_yaw = np.deg2rad(
            np.arange(-5, 6, dtype=np.float64)
            * self.config.refinement_yaw_step_deg
        )
        refined: list[tuple[float, np.ndarray, tuple]] = []
        for dyaw in refinement_yaw:
            for dx in refinement_xy:
                for dy in refinement_xy:
                    correction = coarse_best + np.array(
                        [dx, dy, dyaw], dtype=np.float64
                    )
                    if (
                        np.linalg.norm(correction[:2])
                        > self.config.maximum_translation_m + 1.0e-12
                        or abs(np.rad2deg(correction[2]))
                        > self.config.maximum_yaw_deg + 1.0e-12
                    ):
                        continue
                    diagnostics = self._score(
                        correction,
                        state,
                        offsets_world,
                        values,
                        design,
                        scale_group_ids,
                        context_group_ids,
                    )
                    evaluated += 1
                    if np.isfinite(diagnostics[0]):
                        refined.append(
                            (float(diagnostics[0]), correction, diagnostics)
                        )
        candidates.extend(refined)
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_correction, best_diagnostics = candidates[0]
        second_score = float("-inf")
        for score, correction, _diagnostics in candidates[1:]:
            translation_separated = (
                np.linalg.norm(correction[:2] - best_correction[:2])
                >= self.config.basin_translation_separation_m
            )
            yaw_separated = (
                abs(
                    np.rad2deg(
                        math.atan2(
                            math.sin(correction[2] - best_correction[2]),
                            math.cos(correction[2] - best_correction[2]),
                        )
                    )
                )
                >= self.config.basin_yaw_separation_deg
            )
            if translation_separated or yaw_separated:
                second_score = float(score)
                break
        gap = (
            float("inf")
            if not np.isfinite(second_score)
            else float(best_score - second_score)
        )
        count, valid_fraction, observation_std, map_std = (
            int(best_diagnostics[1]),
            float(best_diagnostics[2]),
            float(best_diagnostics[3]),
            float(best_diagnostics[4]),
        )
        rejection = None
        at_boundary = bool(
            np.linalg.norm(best_correction[:2])
            >= self.config.maximum_translation_m
            - 0.5 * self.config.refinement_translation_step_m
            or abs(np.rad2deg(best_correction[2]))
            >= self.config.maximum_yaw_deg
            - 0.5 * self.config.refinement_yaw_step_deg
        )
        if at_boundary:
            rejection = "search_boundary_reached"
        elif best_score < self.config.minimum_correlation:
            rejection = "correlation_below_threshold"
        elif gap < self.config.minimum_basin_correlation_gap:
            rejection = "ambiguous_backscatter_match"
        if rejection is not None:
            return self._rejected(
                rejection,
                base_pose2,
                correction=best_correction,
                correlation_before=before,
                correlation_after=best_score,
                second_basin_correlation=second_score,
                basin_correlation_gap=gap,
                correspondences=count,
                valid_fraction=valid_fraction,
                observation_texture_std=observation_std,
                map_texture_std=map_std,
                evaluated_candidates=evaluated,
                fine_correlation_after=float(best_diagnostics[5]),
                scale_space_correlation_after=float(best_diagnostics[6]),
                context_consistency_after=float(best_diagnostics[7]),
                valid_contexts_after=int(best_diagnostics[8]),
            )

        covariance = np.diag(
            [
                self.config.horizontal_std_m**2,
                self.config.horizontal_std_m**2,
                np.deg2rad(self.config.yaw_std_deg) ** 2,
            ]
        )
        matched_pose2 = base_pose2 + best_correction
        matched_pose2[2] = math.atan2(
            math.sin(matched_pose2[2]), math.cos(matched_pose2[2])
        )
        return BackscatterMatchResult(
            accepted=True,
            correction=best_correction,
            matched_pose2=matched_pose2,
            information=np.linalg.inv(covariance),
            covariance=covariance,
            correlation_before=before,
            correlation_after=best_score,
            second_basin_correlation=second_score,
            basin_correlation_gap=gap,
            correspondences=count,
            valid_fraction=valid_fraction,
            observation_texture_std=observation_std,
            map_texture_std=map_std,
            evaluated_candidates=evaluated,
            fine_correlation_after=float(best_diagnostics[5]),
            scale_space_correlation_after=float(best_diagnostics[6]),
            context_consistency_after=float(best_diagnostics[7]),
            valid_contexts_after=int(best_diagnostics[8]),
        )


@dataclass(frozen=True)
class _Linearization:
    residual: np.ndarray
    jacobian: np.ndarray
    source_indices: np.ndarray


@dataclass(order=True)
class _CoarseCandidate:
    score: float
    correction: np.ndarray = field(compare=False)


@dataclass(order=True)
class _RefinedCandidate:
    score: float
    correction: np.ndarray = field(compare=False)
    iterations: int = field(compare=False)


@dataclass(frozen=True)
class _SolveResult:
    correction: np.ndarray
    iterations: int
    coarse_candidates: int
    distinct_basin_count: int
    best_basin_cost: float
    second_best_basin_cost: float
    basin_cost_gap: float
    basin_log_likelihood_ratio: float


class BathymetricTerrainMatcher:
    """Align motion-compensated MBES bottom points to a measured grid."""

    def __init__(
        self,
        terrain: MeasuredBathymetryGrid,
        config: TerrainMatcherConfig | None = None,
    ) -> None:
        if not isinstance(terrain, MeasuredBathymetryGrid):
            raise TypeError("terrain must be a MeasuredBathymetryGrid")
        self.terrain = terrain
        self.config = config or TerrainMatcherConfig()

    @staticmethod
    def _axis_samples(limit: float, step: float) -> np.ndarray:
        count = int(math.floor(limit / step))
        positive = np.arange(count + 1, dtype=np.float64) * step
        values = np.concatenate((-positive[1:][::-1], positive))
        values = np.concatenate((values, np.array([-limit, 0.0, limit])))
        return np.unique(np.clip(values, -limit, limit))

    def _height_and_gradient(
        self, x_m: np.ndarray, y_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample height and the exact derivative of each bilinear cell.

        All four corners must be measured and finite.  This is intentionally
        stricter than accepting a value exactly on a cell edge: a derivative
        that crosses NoData is not a defensible terrain constraint.
        """

        terrain = self.terrain
        x = np.asarray(x_m, dtype=np.float64).reshape(-1)
        y = np.asarray(y_m, dtype=np.float64).reshape(-1)
        if x.size != y.size:
            raise ValueError("x_m and y_m must have equal size")
        height = np.full(x.size, np.nan, dtype=np.float64)
        gradient_x = np.full(x.size, np.nan, dtype=np.float64)
        gradient_y = np.full(x.size, np.nan, dtype=np.float64)
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= -0.5 * terrain.width)
            & (x <= 0.5 * terrain.width)
            & (y >= -0.5 * terrain.length)
            & (y <= 0.5 * terrain.length)
        )
        selected = np.flatnonzero(valid)
        if selected.size == 0:
            return height, gradient_x, gradient_y, valid

        resolution = terrain.resolution_m
        column = (x[selected] + 0.5 * terrain.width) / resolution
        row = (y[selected] + 0.5 * terrain.length) / resolution
        column0 = np.minimum(
            np.floor(column).astype(np.int64),
            terrain.elevation_m.shape[1] - 2,
        )
        row0 = np.minimum(
            np.floor(row).astype(np.int64),
            terrain.elevation_m.shape[0] - 2,
        )
        column_fraction = column - column0
        row_fraction = row - row0

        elevation = terrain.elevation_m
        mask = terrain.valid_mask
        z00 = np.asarray(elevation[row0, column0], dtype=np.float64)
        z01 = np.asarray(elevation[row0, column0 + 1], dtype=np.float64)
        z10 = np.asarray(elevation[row0 + 1, column0], dtype=np.float64)
        z11 = np.asarray(elevation[row0 + 1, column0 + 1], dtype=np.float64)
        support = (
            mask[row0, column0]
            & mask[row0, column0 + 1]
            & mask[row0 + 1, column0]
            & mask[row0 + 1, column0 + 1]
            & np.isfinite(z00)
            & np.isfinite(z01)
            & np.isfinite(z10)
            & np.isfinite(z11)
        )
        supported = selected[support]
        valid[selected[~support]] = False
        if supported.size == 0:
            return height, gradient_x, gradient_y, valid

        u = column_fraction[support]
        v = row_fraction[support]
        a = z00[support]
        b = z01[support]
        c = z10[support]
        d = z11[support]
        height[supported] = (
            (1.0 - v) * ((1.0 - u) * a + u * b)
            + v * ((1.0 - u) * c + u * d)
        )
        gradient_x[supported] = (
            (1.0 - v) * (b - a) + v * (d - c)
        ) / resolution
        gradient_y[supported] = (
            (1.0 - u) * (c - a) + u * (d - b)
        ) / resolution
        return height, gradient_x, gradient_y, valid

    def _linearize(
        self,
        correction: np.ndarray,
        state: InertialState,
        offsets_world: np.ndarray,
    ) -> _Linearization:
        correction = np.asarray(correction, dtype=np.float64).reshape(3)
        cosine = math.cos(float(correction[2]))
        sine = math.sin(float(correction[2]))
        offset_x = cosine * offsets_world[:, 0] - sine * offsets_world[:, 1]
        offset_y = sine * offsets_world[:, 0] + cosine * offsets_world[:, 1]
        x = state.position[0] + correction[0] + offset_x
        y = state.position[1] + correction[1] + offset_y
        z = state.position[2] + offsets_world[:, 2]
        height, gradient_x, gradient_y, valid = self._height_and_gradient(x, y)
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            return _Linearization(
                np.empty(0, dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                indices,
            )
        residual = z[indices] - height[indices]
        gx = gradient_x[indices]
        gy = gradient_y[indices]
        jacobian = np.column_stack(
            (
                -gx,
                -gy,
                gx * offset_y[indices] - gy * offset_x[indices],
            )
        )
        finite = np.isfinite(residual) & np.isfinite(jacobian).all(axis=1)
        return _Linearization(
            residual[finite], jacobian[finite], indices[finite]
        )

    def _robust_weights_and_cost(
        self, residual: np.ndarray
    ) -> tuple[np.ndarray, float]:
        absolute = np.abs(residual)
        delta = self.config.huber_delta_m
        weights = np.ones(residual.size, dtype=np.float64)
        outside = absolute > delta
        weights[outside] = delta / np.maximum(absolute[outside], 1.0e-12)
        loss = np.where(
            outside,
            delta * (absolute - 0.5 * delta),
            0.5 * residual**2,
        )
        return weights, float(np.mean(loss)) if loss.size else float("inf")

    def _candidate_score(
        self,
        correction: np.ndarray,
        state: InertialState,
        offsets_world: np.ndarray,
    ) -> float:
        linearization = self._linearize(correction, state, offsets_world)
        valid_fraction = linearization.residual.size / offsets_world.shape[0]
        if (
            linearization.residual.size < self.config.minimum_points
            or valid_fraction < self.config.minimum_valid_fraction
        ):
            return float("inf")
        _, cost = self._robust_weights_and_cost(linearization.residual)
        # A solution cannot improve its score merely by moving returns into a
        # NoData gap.  The penalty is modest after the hard overlap gate but
        # makes equal-cost candidates with more measured support preferable.
        missing_penalty = (
            2.0
            * self.config.measurement_std_m**2
            * (1.0 - valid_fraction)
        )
        return cost + missing_penalty

    def _coarse_search(
        self,
        state: InertialState,
        offsets_world: np.ndarray,
        *,
        optimize_yaw: bool,
    ) -> list[_CoarseCandidate]:
        translation = self._axis_samples(
            self.config.maximum_translation_m,
            self.config.coarse_translation_step_m,
        )
        yaw = (
            np.deg2rad(
                self._axis_samples(
                    self.config.maximum_yaw_deg,
                    self.config.coarse_yaw_step_deg,
                )
            )
            if optimize_yaw
            else np.zeros(1, dtype=np.float64)
        )
        candidates: list[_CoarseCandidate] = []
        for yaw_value in yaw:
            for x_value in translation:
                for y_value in translation:
                    if math.hypot(float(x_value), float(y_value)) > (
                        self.config.maximum_translation_m + 1.0e-12
                    ):
                        continue
                    correction = np.array(
                        [x_value, y_value, yaw_value], dtype=np.float64
                    )
                    score = self._candidate_score(
                        correction, state, offsets_world
                    )
                    if np.isfinite(score):
                        candidates.append(_CoarseCandidate(score, correction))
        candidates.sort()
        # Do not let several adjacent lattice samples from one broad minimum
        # consume the complete refinement budget.  This is parameter-space
        # non-maximum suppression: selected seeds are separated in a metric
        # whose translation and yaw radii are explicit configuration values.
        return self._separated_candidates(
            candidates,
            optimize_yaw=optimize_yaw,
            translation_separation_m=(
                self.config.coarse_seed_translation_separation_m
            ),
            yaw_separation_deg=self.config.coarse_seed_yaw_separation_deg,
            maximum_count=self.config.coarse_top_k,
        )

    def _basins_are_separated(
        self,
        first: np.ndarray,
        second: np.ndarray,
        *,
        optimize_yaw: bool,
        translation_separation_m: float,
        yaw_separation_deg: float,
    ) -> bool:
        first = np.asarray(first, dtype=np.float64).reshape(3)
        second = np.asarray(second, dtype=np.float64).reshape(3)
        translation_distance = float(np.linalg.norm(first[:2] - second[:2]))
        normalized_squared = (
            translation_distance / float(translation_separation_m)
        ) ** 2
        if optimize_yaw:
            yaw_difference = float(
                (first[2] - second[2] + np.pi) % (2.0 * np.pi) - np.pi
            )
            normalized_squared += (
                np.rad2deg(abs(yaw_difference))
                / float(yaw_separation_deg)
            ) ** 2
        return normalized_squared >= 1.0 - 1.0e-12

    def _separated_candidates(
        self,
        candidates: list[_CoarseCandidate] | list[_RefinedCandidate],
        *,
        optimize_yaw: bool,
        translation_separation_m: float,
        yaw_separation_deg: float,
        maximum_count: int | None = None,
    ) -> list[_CoarseCandidate] | list[_RefinedCandidate]:
        selected: list[_CoarseCandidate] | list[_RefinedCandidate] = []
        for candidate in candidates:
            if all(
                self._basins_are_separated(
                    candidate.correction,
                    previous.correction,
                    optimize_yaw=optimize_yaw,
                    translation_separation_m=translation_separation_m,
                    yaw_separation_deg=yaw_separation_deg,
                )
                for previous in selected
            ):
                selected.append(candidate)
                if maximum_count is not None and len(selected) >= maximum_count:
                    break
        return selected

    def _bounded(self, correction: np.ndarray) -> np.ndarray:
        bounded = np.asarray(correction, dtype=np.float64).reshape(3).copy()
        translation_norm = float(np.linalg.norm(bounded[:2]))
        if translation_norm > self.config.maximum_translation_m:
            bounded[:2] *= self.config.maximum_translation_m / translation_norm
        yaw_limit = np.deg2rad(self.config.maximum_yaw_deg)
        bounded[2] = np.clip(bounded[2], -yaw_limit, yaw_limit)
        return bounded

    def _refine(
        self,
        initial: np.ndarray,
        state: InertialState,
        offsets_world: np.ndarray,
        *,
        optimize_yaw: bool,
    ) -> tuple[np.ndarray, float, int]:
        correction = self._bounded(initial)
        if not optimize_yaw:
            correction[2] = 0.0
        best_score = self._candidate_score(correction, state, offsets_world)
        iterations = 0
        for iteration in range(self.config.maximum_iterations):
            linearization = self._linearize(correction, state, offsets_world)
            if linearization.residual.size < self.config.minimum_points:
                break
            weights, _ = self._robust_weights_and_cost(
                linearization.residual
            )
            active = np.array([0, 1, 2] if optimize_yaw else [0, 1])
            jacobian = linearization.jacobian[:, active]
            hessian = jacobian.T @ (weights[:, None] * jacobian)
            gradient = jacobian.T @ (weights * linearization.residual)
            diagonal = np.maximum(np.diag(hessian), 1.0)
            damped = hessian + self.config.levenberg_damping * np.diag(diagonal)
            try:
                step = -np.linalg.solve(damped, gradient)
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(step).all():
                break
            full_step = np.zeros(3, dtype=np.float64)
            full_step[active] = step
            step = full_step
            translation_norm = float(np.linalg.norm(step[:2]))
            maximum_translation_step = (
                self.config.maximum_iteration_translation_step_m
            )
            if translation_norm > maximum_translation_step:
                step[:2] *= maximum_translation_step / translation_norm
            maximum_yaw_step = np.deg2rad(
                self.config.maximum_iteration_yaw_step_deg
            )
            step[2] = np.clip(step[2], -maximum_yaw_step, maximum_yaw_step)

            accepted_step = False
            applied_step = np.zeros(3, dtype=np.float64)
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                proposal = self._bounded(correction + scale * step)
                if not optimize_yaw:
                    proposal[2] = 0.0
                score = self._candidate_score(proposal, state, offsets_world)
                if score <= best_score - 1.0e-14:
                    applied_step = proposal - correction
                    correction = proposal
                    best_score = score
                    accepted_step = True
                    break
            iterations = iteration + 1
            if not accepted_step:
                break
            if (
                np.linalg.norm(applied_step[:2])
                <= self.config.translation_convergence_m
                and abs(applied_step[2])
                <= np.deg2rad(self.config.yaw_convergence_deg)
            ):
                break
        return correction, best_score, iterations

    def _solve(
        self,
        state: InertialState,
        offsets_world: np.ndarray,
        *,
        optimize_yaw: bool,
    ) -> _SolveResult | None:
        candidates = self._coarse_search(
            state, offsets_world, optimize_yaw=optimize_yaw
        )
        if not candidates:
            return None
        refined: list[_RefinedCandidate] = []
        for candidate in candidates:
            correction, score, iterations = self._refine(
                candidate.correction,
                state,
                offsets_world,
                optimize_yaw=optimize_yaw,
            )
            refined.append(_RefinedCandidate(score, correction, iterations))
        refined.sort()
        basins = self._separated_candidates(
            refined,
            optimize_yaw=optimize_yaw,
            translation_separation_m=(
                self.config.minimum_basin_translation_separation_m
            ),
            yaw_separation_deg=self.config.minimum_basin_yaw_separation_deg,
        )
        if not basins:
            return None
        best = basins[0]
        second_best_cost = (
            float(basins[1].score) if len(basins) > 1 else float("inf")
        )
        basin_cost_gap = max(0.0, second_best_cost - float(best.score))
        effective_count = min(
            float(offsets_world.shape[0]),
            float(self.config.effective_independent_points),
        )
        # ``_candidate_score`` is the mean robust negative log-likelihood in
        # squared-metre units (0.5*r^2 in the Gaussian core).  Scaling its gap
        # by independent samples and vertical variance produces the log of the
        # likelihood ratio between distinct refined hypotheses.
        log_likelihood_ratio = (
            basin_cost_gap
            * effective_count
            / self.config.measurement_std_m**2
        )
        return _SolveResult(
            correction=best.correction,
            iterations=best.iterations,
            coarse_candidates=len(candidates),
            distinct_basin_count=len(basins),
            best_basin_cost=float(best.score),
            second_best_basin_cost=second_best_cost,
            basin_cost_gap=basin_cost_gap,
            basin_log_likelihood_ratio=log_likelihood_ratio,
        )

    def _information_diagnostics(
        self,
        residual: np.ndarray,
        jacobian: np.ndarray,
        *,
        optimize_yaw: bool,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        int,
        float,
        float,
    ]:
        """Build correlation-capped information and its observable subspace."""

        robust_weights, _ = self._robust_weights_and_cost(residual)
        independent_scale = min(
            1.0,
            self.config.effective_independent_points / float(residual.size),
        )
        inverse_variance = 1.0 / self.config.measurement_std_m**2
        used_jacobian = np.asarray(jacobian, dtype=np.float64).copy()
        if not optimize_yaw:
            used_jacobian[:, 2] = 0.0
        information = (
            independent_scale
            * inverse_variance
            * (
                used_jacobian.T
                @ (robust_weights[:, None] * used_jacobian)
            )
        )
        information = 0.5 * (information + information.T)

        # u = diag(1, 1, yaw_rank_scale_m) q has metre-like units.  This keeps
        # rank thresholds from depending on whether yaw is expressed in rad.
        q_from_u = np.diag(
            [1.0, 1.0, 1.0 / self.config.yaw_rank_scale_m]
        )
        scaled_information = q_from_u.T @ information @ q_from_u
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(scaled_information)
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(3, dtype=np.float64)
            eigenvectors = np.eye(3, dtype=np.float64)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        largest = float(eigenvalues[-1])
        threshold = max(
            self.config.minimum_scaled_information_eigenvalue,
            self.config.minimum_information_eigen_ratio * largest,
        )
        observable = eigenvalues >= threshold
        observable_rank = int(np.count_nonzero(observable))
        basis = eigenvectors[:, observable]
        projector = (
            basis @ basis.T
            if basis.size
            else np.zeros((3, 3), dtype=np.float64)
        )

        translation_information = information[:2, :2]
        try:
            translation_eigenvalues = np.maximum(
                np.linalg.eigvalsh(translation_information), 0.0
            )
        except np.linalg.LinAlgError:
            translation_eigenvalues = np.zeros(2, dtype=np.float64)
        translation_largest = float(translation_eigenvalues[-1])
        translation_threshold = max(
            self.config.minimum_scaled_information_eigenvalue,
            self.config.minimum_information_eigen_ratio * translation_largest,
        )
        translation_rank = int(
            np.count_nonzero(translation_eigenvalues >= translation_threshold)
        )
        positive = eigenvalues[observable]
        condition = (
            float(positive[-1] / positive[0])
            if positive.size and positive[0] > 0.0
            else float("inf")
        )

        # Report a conservative finite covariance even for rank-two output.
        # Consumers forming graph factors should use ``information`` directly.
        variance_u = np.full(3, self.config.unobservable_variance_scaled_m2)
        variance_u[observable] = 1.0 / eigenvalues[observable]
        covariance_u = (eigenvectors * variance_u) @ eigenvectors.T
        covariance = q_from_u @ covariance_u @ q_from_u.T
        covariance = 0.5 * (covariance + covariance.T)
        effective_count = max(
            1.0, independent_scale * float(residual.size) - observable_rank
        )
        reduced_chi_square = float(
            independent_scale
            * np.sum(robust_weights * residual**2)
            * inverse_variance
            / effective_count
        )
        return (
            information,
            covariance,
            eigenvalues,
            projector,
            observable_rank,
            translation_rank,
            condition,
            reduced_chi_square,
        )

    def _rejected(
        self,
        reason: str,
        state: InertialState,
        *,
        correction: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
        information: np.ndarray | None = None,
        correspondences: int = 0,
        inliers: int = 0,
        iterations: int = 0,
        coarse_candidates: int = 0,
        distinct_basin_count: int = 0,
        best_basin_cost: float = float("nan"),
        second_best_basin_cost: float = float("nan"),
        basin_cost_gap: float = float("nan"),
        basin_log_likelihood_ratio: float = float("nan"),
        rmse_before: float = float("nan"),
        rmse_after: float = float("nan"),
        reduced_chi_square: float = float("nan"),
        innovation_nis: float = float("nan"),
        observable_rank: int = 0,
        translation_rank: int = 0,
        yaw_observable: bool = False,
        projector: np.ndarray | None = None,
        eigenvalues: np.ndarray | None = None,
        condition: float = float("inf"),
        boundary: bool = False,
    ) -> TerrainMatchResult:
        return TerrainMatchResult(
            accepted=False,
            correction=(
                np.zeros(3, dtype=np.float64)
                if correction is None
                else correction
            ),
            covariance=(
                np.full((3, 3), np.nan, dtype=np.float64)
                if covariance is None
                else covariance
            ),
            information=(
                np.zeros((3, 3), dtype=np.float64)
                if information is None
                else information
            ),
            correspondences=correspondences,
            inliers=inliers,
            iterations=iterations,
            coarse_candidates=coarse_candidates,
            distinct_basin_count=distinct_basin_count,
            best_basin_cost=best_basin_cost,
            second_best_basin_cost=second_best_basin_cost,
            basin_cost_gap=basin_cost_gap,
            basin_log_likelihood_ratio=basin_log_likelihood_ratio,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            reduced_chi_square=reduced_chi_square,
            innovation_nis=innovation_nis,
            observable_rank=observable_rank,
            translation_observable_rank=translation_rank,
            yaw_observable=yaw_observable,
            observable_projector_scaled=(
                np.zeros((3, 3), dtype=np.float64)
                if projector is None
                else projector
            ),
            scaled_information_eigenvalues=(
                np.zeros(3, dtype=np.float64)
                if eigenvalues is None
                else eigenvalues
            ),
            information_condition=condition,
            reached_search_boundary=boundary,
            rejection_reason=reason,
            state_timestamp=float(state.timestamp),
            base_position=state.position,
            base_rotation=state.rotation,
            maximum_innovation_nis=self.config.maximum_innovation_nis,
            innovation_nis_gate_enabled=self.config.enable_innovation_nis_gate,
            yaw_rank_scale_m=self.config.yaw_rank_scale_m,
            minimum_scaled_information_eigenvalue=(
                self.config.minimum_scaled_information_eigenvalue
            ),
            minimum_information_eigen_ratio=(
                self.config.minimum_information_eigen_ratio
            ),
            minimum_basin_log_likelihood_ratio=(
                self.config.minimum_basin_log_likelihood_ratio
            ),
        )

    def match(
        self,
        points_sensor: np.ndarray,
        state: InertialState,
        *,
        sensor_rotation_body: np.ndarray | None = None,
        sensor_translation_body: np.ndarray | None = None,
        points_are_body_frame: bool = False,
    ) -> TerrainMatchResult:
        """Match a motion-compensated MBES patch to the prior map.

        Points must all refer to one common body epoch.  For sensor-frame
        points, pass the rigid ``sensor -> body`` extrinsics.  If a caller has
        already motion-compensated and transformed the patch into body
        coordinates, set ``points_are_body_frame=True``; non-identity sensor
        extrinsics are then rejected to avoid applying them twice.
        """

        if not isinstance(state, InertialState):
            raise TypeError("state must be an InertialState")
        state = state.copy()
        if (
            not np.isfinite(state.position).all()
            or not np.isfinite(state.rotation).all()
            or not np.allclose(
                state.rotation.T @ state.rotation,
                np.eye(3),
                atol=1.0e-6,
                rtol=0.0,
            )
            or np.linalg.det(state.rotation) <= 0.0
        ):
            raise ValueError("state pose must be finite and lie on SO(3)")
        points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < self.config.minimum_points:
            return self._rejected("insufficient_points", state)

        sensor_rotation = (
            np.eye(3, dtype=np.float64)
            if sensor_rotation_body is None
            else project_to_so3(sensor_rotation_body)
        )
        sensor_translation = (
            np.zeros(3, dtype=np.float64)
            if sensor_translation_body is None
            else np.asarray(sensor_translation_body, dtype=np.float64).reshape(3)
        )
        if not np.isfinite(sensor_translation).all():
            raise ValueError("sensor_translation_body must be finite")
        if points_are_body_frame:
            if (
                not np.allclose(sensor_rotation, np.eye(3), atol=1.0e-12)
                or not np.allclose(sensor_translation, 0.0, atol=1.0e-12)
            ):
                raise ValueError(
                    "sensor extrinsics must be identity for body-frame points"
                )
            points_body = points
        else:
            points_body = points @ sensor_rotation.T + sensor_translation

        # Deterministic spatially uniform subsampling preserves both ends of
        # the swath and makes tests/runs reproducible.
        if points_body.shape[0] > self.config.maximum_points:
            selected = np.linspace(
                0,
                points_body.shape[0] - 1,
                self.config.maximum_points,
                dtype=np.int64,
            )
            points_body = points_body[selected]
        offsets_world = points_body @ state.rotation.T

        initial = self._linearize(np.zeros(3), state, offsets_world)
        initial_valid_fraction = initial.residual.size / points_body.shape[0]
        if (
            initial.residual.size < self.config.minimum_points
            or initial_valid_fraction < self.config.minimum_valid_fraction
        ):
            return self._rejected(
                "insufficient_initial_map_overlap",
                state,
                correspondences=initial.residual.size,
            )

        solution = self._solve(state, offsets_world, optimize_yaw=True)
        if solution is None:
            return self._rejected(
                "coarse_search_failed",
                state,
                correspondences=initial.residual.size,
            )

        best_correction = solution.correction
        best_iterations = solution.iterations
        coarse_candidates = solution.coarse_candidates
        distinct_basin_count = solution.distinct_basin_count
        best_basin_cost = solution.best_basin_cost
        second_best_basin_cost = solution.second_best_basin_cost
        basin_cost_gap = solution.basin_cost_gap
        basin_log_likelihood_ratio = solution.basin_log_likelihood_ratio
        optimize_yaw = True

        # Decide whether yaw is independently observable at the unrestricted
        # solution.  If not, solve again with yaw fixed at the seed; retaining
        # an arbitrary yaw from a rank-deficient fit would contaminate x/y.
        preliminary = self._linearize(
            best_correction, state, offsets_world
        )
        preliminary_inlier = (
            np.abs(preliminary.residual) <= self.config.inlier_threshold_m
        )
        preliminary_rank = 0
        preliminary_translation_rank = 0
        if np.count_nonzero(preliminary_inlier) >= self.config.minimum_points:
            preliminary_diagnostics = self._information_diagnostics(
                preliminary.residual[preliminary_inlier],
                preliminary.jacobian[preliminary_inlier],
                optimize_yaw=True,
            )
            preliminary_rank = preliminary_diagnostics[4]
            preliminary_translation_rank = preliminary_diagnostics[5]
        if preliminary_rank < 3 or preliminary_translation_rank < 2:
            translation_solution = self._solve(
                state, offsets_world, optimize_yaw=False
            )
            if translation_solution is not None:
                best_correction = translation_solution.correction
                best_iterations = translation_solution.iterations
                coarse_candidates = translation_solution.coarse_candidates
                distinct_basin_count = (
                    translation_solution.distinct_basin_count
                )
                best_basin_cost = translation_solution.best_basin_cost
                second_best_basin_cost = (
                    translation_solution.second_best_basin_cost
                )
                basin_cost_gap = translation_solution.basin_cost_gap
                basin_log_likelihood_ratio = (
                    translation_solution.basin_log_likelihood_ratio
                )
                optimize_yaw = False

        basin_diagnostics = {
            "distinct_basin_count": distinct_basin_count,
            "best_basin_cost": best_basin_cost,
            "second_best_basin_cost": second_best_basin_cost,
            "basin_cost_gap": basin_cost_gap,
            "basin_log_likelihood_ratio": basin_log_likelihood_ratio,
        }

        final = self._linearize(best_correction, state, offsets_world)
        correspondences = final.residual.size
        valid_fraction = correspondences / points_body.shape[0]
        if (
            correspondences < self.config.minimum_points
            or valid_fraction < self.config.minimum_valid_fraction
        ):
            return self._rejected(
                "insufficient_final_map_overlap",
                state,
                correction=best_correction,
                correspondences=correspondences,
                iterations=best_iterations,
                coarse_candidates=coarse_candidates,
                **basin_diagnostics,
            )

        inlier_mask = np.abs(final.residual) <= self.config.inlier_threshold_m
        inliers = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inliers / correspondences
        initial_lookup = np.full(points_body.shape[0], np.nan, dtype=np.float64)
        initial_lookup[initial.source_indices] = initial.residual
        common_before = initial_lookup[final.source_indices]
        common = np.isfinite(common_before)
        rmse_before = (
            float(np.sqrt(np.mean(common_before[common] ** 2)))
            if np.any(common)
            else float("nan")
        )
        rmse_after = float(np.sqrt(np.mean(final.residual**2)))
        if inliers < self.config.minimum_points or (
            inlier_ratio < self.config.minimum_inlier_ratio
        ):
            return self._rejected(
                "insufficient_inliers",
                state,
                correction=best_correction,
                correspondences=correspondences,
                inliers=inliers,
                iterations=best_iterations,
                coarse_candidates=coarse_candidates,
                rmse_before=rmse_before,
                rmse_after=rmse_after,
                **basin_diagnostics,
            )

        residual = final.residual[inlier_mask]
        jacobian = final.jacobian[inlier_mask]
        (
            information,
            covariance,
            eigenvalues,
            projector,
            observable_rank,
            translation_rank,
            condition,
            reduced_chi_square,
        ) = self._information_diagnostics(
            residual, jacobian, optimize_yaw=optimize_yaw
        )
        yaw_observable = bool(optimize_yaw and observable_rank == 3)
        required_rank = 3 if yaw_observable else 2
        if translation_rank < 2 or observable_rank < required_rank:
            return self._rejected(
                "unobservable_terrain",
                state,
                correction=best_correction,
                covariance=covariance,
                information=information,
                correspondences=correspondences,
                inliers=inliers,
                iterations=best_iterations,
                coarse_candidates=coarse_candidates,
                rmse_before=rmse_before,
                rmse_after=rmse_after,
                observable_rank=observable_rank,
                translation_rank=translation_rank,
                yaw_observable=yaw_observable,
                projector=projector,
                eigenvalues=eigenvalues,
                condition=condition,
                **basin_diagnostics,
            )

        if self.config.require_yaw_observable and not yaw_observable:
            # A rank-two solution can be a useful horizontal-only update for
            # backwards-compatible callers.  Whole-map absolute localization
            # uses the stricter mode: without independent yaw information a
            # unary [x, y, yaw] prior is not admitted to the global graph.
            innovation_nis = _terrain_match_nis(
                best_correction,
                information,
                state,
                yaw_rank_scale_m=self.config.yaw_rank_scale_m,
                minimum_scaled_information_eigenvalue=(
                    self.config.minimum_scaled_information_eigenvalue
                ),
                minimum_information_eigen_ratio=(
                    self.config.minimum_information_eigen_ratio
                ),
            )
            return self._rejected(
                "yaw_unobservable",
                state,
                correction=best_correction,
                covariance=covariance,
                information=information,
                correspondences=correspondences,
                inliers=inliers,
                iterations=best_iterations,
                coarse_candidates=coarse_candidates,
                rmse_before=rmse_before,
                rmse_after=rmse_after,
                reduced_chi_square=reduced_chi_square,
                innovation_nis=innovation_nis,
                observable_rank=observable_rank,
                translation_rank=translation_rank,
                yaw_observable=yaw_observable,
                projector=projector,
                eigenvalues=eigenvalues,
                condition=condition,
                **basin_diagnostics,
            )

        yaw_limit = np.deg2rad(self.config.maximum_yaw_deg)
        boundary = bool(
            np.linalg.norm(best_correction[:2])
            >= self.config.maximum_translation_m
            - self.config.boundary_translation_margin_m
            or (
                yaw_observable
                and abs(best_correction[2])
                >= yaw_limit - np.deg2rad(self.config.boundary_yaw_margin_deg)
            )
        )
        if boundary:
            return self._rejected(
                "search_boundary_reached",
                state,
                correction=best_correction,
                covariance=covariance,
                information=information,
                correspondences=correspondences,
                inliers=inliers,
                iterations=best_iterations,
                coarse_candidates=coarse_candidates,
                rmse_before=rmse_before,
                rmse_after=rmse_after,
                reduced_chi_square=reduced_chi_square,
                observable_rank=observable_rank,
                translation_rank=translation_rank,
                yaw_observable=yaw_observable,
                projector=projector,
                eigenvalues=eigenvalues,
                condition=condition,
                boundary=True,
                **basin_diagnostics,
            )
        if reduced_chi_square > self.config.maximum_reduced_chi_square:
            return self._rejected(
                "residual_chi_square",
                state,
                correction=best_correction,
                covariance=covariance,
                information=information,
                correspondences=correspondences,
                inliers=inliers,
                iterations=best_iterations,
                coarse_candidates=coarse_candidates,
                rmse_before=rmse_before,
                rmse_after=rmse_after,
                reduced_chi_square=reduced_chi_square,
                observable_rank=observable_rank,
                translation_rank=translation_rank,
                yaw_observable=yaw_observable,
                projector=projector,
                eigenvalues=eigenvalues,
                condition=condition,
                **basin_diagnostics,
            )

        if distinct_basin_count >= 2 and (
            not np.isfinite(basin_log_likelihood_ratio)
            or basin_log_likelihood_ratio
            < self.config.minimum_basin_log_likelihood_ratio
        ):
            return self._rejected(
                "ambiguous_match",
                state,
                correction=best_correction,
                covariance=covariance,
                information=information,
                correspondences=correspondences,
                inliers=inliers,
                iterations=best_iterations,
                coarse_candidates=coarse_candidates,
                rmse_before=rmse_before,
                rmse_after=rmse_after,
                reduced_chi_square=reduced_chi_square,
                observable_rank=observable_rank,
                translation_rank=translation_rank,
                yaw_observable=yaw_observable,
                projector=projector,
                eigenvalues=eigenvalues,
                condition=condition,
                **basin_diagnostics,
            )

        innovation_nis = _terrain_match_nis(
            best_correction,
            information,
            state,
            yaw_rank_scale_m=self.config.yaw_rank_scale_m,
            minimum_scaled_information_eigenvalue=(
                self.config.minimum_scaled_information_eigenvalue
            ),
            minimum_information_eigen_ratio=(
                self.config.minimum_information_eigen_ratio
            ),
        )
        if self.config.enable_innovation_nis_gate and (
            not np.isfinite(innovation_nis)
            or innovation_nis > self.config.maximum_innovation_nis
        ):
            return self._rejected(
                "innovation_nis",
                state,
                correction=best_correction,
                covariance=covariance,
                information=information,
                correspondences=correspondences,
                inliers=inliers,
                iterations=best_iterations,
                coarse_candidates=coarse_candidates,
                rmse_before=rmse_before,
                rmse_after=rmse_after,
                reduced_chi_square=reduced_chi_square,
                innovation_nis=innovation_nis,
                observable_rank=observable_rank,
                translation_rank=translation_rank,
                yaw_observable=yaw_observable,
                projector=projector,
                eigenvalues=eigenvalues,
                condition=condition,
                **basin_diagnostics,
            )

        return TerrainMatchResult(
            accepted=True,
            correction=best_correction,
            covariance=covariance,
            information=information,
            correspondences=correspondences,
            inliers=inliers,
            iterations=best_iterations,
            coarse_candidates=coarse_candidates,
            distinct_basin_count=distinct_basin_count,
            best_basin_cost=best_basin_cost,
            second_best_basin_cost=second_best_basin_cost,
            basin_cost_gap=basin_cost_gap,
            basin_log_likelihood_ratio=basin_log_likelihood_ratio,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            reduced_chi_square=reduced_chi_square,
            innovation_nis=innovation_nis,
            observable_rank=observable_rank,
            translation_observable_rank=translation_rank,
            yaw_observable=yaw_observable,
            observable_projector_scaled=projector,
            scaled_information_eigenvalues=eigenvalues,
            information_condition=condition,
            reached_search_boundary=False,
            rejection_reason=None,
            state_timestamp=float(state.timestamp),
            base_position=state.position,
            base_rotation=state.rotation,
            maximum_innovation_nis=self.config.maximum_innovation_nis,
            innovation_nis_gate_enabled=self.config.enable_innovation_nis_gate,
            yaw_rank_scale_m=self.config.yaw_rank_scale_m,
            minimum_scaled_information_eigenvalue=(
                self.config.minimum_scaled_information_eigenvalue
            ),
            minimum_information_eigen_ratio=(
                self.config.minimum_information_eigen_ratio
            ),
            minimum_basin_log_likelihood_ratio=(
                self.config.minimum_basin_log_likelihood_ratio
            ),
        )


def _observable_measurement(
    correction: np.ndarray,
    information: np.ndarray,
    state: InertialState,
    *,
    yaw_rank_scale_m: float,
    minimum_scaled_information_eigenvalue: float,
    minimum_information_eigen_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Convert rank-deficient planar information into observable rows."""

    correction = np.asarray(correction, dtype=np.float64).reshape(3)
    information = np.asarray(information, dtype=np.float64).reshape(3, 3)
    information = 0.5 * (information + information.T)
    if not np.isfinite(correction).all() or not np.isfinite(information).all():
        return None
    q_from_u = np.diag([1.0, 1.0, 1.0 / yaw_rank_scale_m])
    u_from_q = np.diag([1.0, 1.0, yaw_rank_scale_m])
    information_u = q_from_u.T @ information @ q_from_u
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(information_u)
    except np.linalg.LinAlgError:
        return None
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest = float(eigenvalues[-1])
    threshold = max(
        minimum_scaled_information_eigenvalue,
        minimum_information_eigen_ratio * largest,
    )
    observable = eigenvalues >= threshold
    if not np.any(observable):
        return None
    basis = eigenvectors[:, observable]

    planar_observation = np.zeros((3, 15), dtype=np.float64)
    planar_observation[0, 3] = 1.0
    planar_observation[1, 4] = 1.0
    # ESIKF attitude errors are right perturbations in body coordinates.  A
    # world-Z yaw perturbation therefore projects as e_z^T R_world_body.
    planar_observation[2, 0:3] = (
        np.array([0.0, 0.0, 1.0]) @ state.rotation
    )
    residual = basis.T @ u_from_q @ correction
    observation = basis.T @ u_from_q @ planar_observation
    covariance = np.diag(1.0 / eigenvalues[observable])
    return residual, observation, covariance


def _terrain_match_nis(
    correction: np.ndarray,
    information: np.ndarray,
    state: InertialState,
    *,
    yaw_rank_scale_m: float,
    minimum_scaled_information_eigenvalue: float,
    minimum_information_eigen_ratio: float,
) -> float:
    measurement = _observable_measurement(
        correction,
        information,
        state,
        yaw_rank_scale_m=yaw_rank_scale_m,
        minimum_scaled_information_eigenvalue=(
            minimum_scaled_information_eigenvalue
        ),
        minimum_information_eigen_ratio=minimum_information_eigen_ratio,
    )
    if measurement is None:
        return float("inf")
    residual, observation, covariance = measurement
    innovation = observation @ state.covariance @ observation.T + covariance
    try:
        return float(residual @ np.linalg.solve(innovation, residual))
    except np.linalg.LinAlgError:
        return float("inf")


def motion_compensate_to_body(
    points_sensor: np.ndarray,
    acquisition_state: InertialState,
    reference_state: InertialState,
    *,
    sensor_rotation_body: np.ndarray | None = None,
    sensor_translation_body: np.ndarray | None = None,
) -> np.ndarray:
    """Transform one historical MBES scan into a reference body frame.

    Both poses must come from navigation/SLAM estimates in the same frame;
    simulator truth is neither needed nor accepted separately.  Concatenate
    the result for successive scans to form a terrain-matching patch::

        patch = np.vstack([
            motion_compensate_to_body(scan.points, scan.state, global_seed)
            for scan in recent_scans
        ])
        result = matcher.match(
            patch, global_seed, points_are_body_frame=True
        )

    The rigid extrinsics map ``sensor -> body``.  Non-finite misses are
    discarded.  Motion compensation quality bounds the useful patch duration;
    this helper does not hide intra-ping timing distortion.
    """

    if not isinstance(acquisition_state, InertialState) or not isinstance(
        reference_state, InertialState
    ):
        raise TypeError("acquisition_state and reference_state must be inertial states")
    points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    sensor_rotation = (
        np.eye(3, dtype=np.float64)
        if sensor_rotation_body is None
        else project_to_so3(sensor_rotation_body)
    )
    sensor_translation = (
        np.zeros(3, dtype=np.float64)
        if sensor_translation_body is None
        else np.asarray(sensor_translation_body, dtype=np.float64).reshape(3)
    )
    if not np.isfinite(sensor_translation).all():
        raise ValueError("sensor extrinsics must be finite")
    points_body = points @ sensor_rotation.T + sensor_translation
    points_world = (
        points_body @ acquisition_state.rotation.T
        + acquisition_state.position
    )
    return (points_world - reference_state.position) @ reference_state.rotation


def fixed_lag_fused_deskew_states(
    acoustic_states: list[InertialState] | tuple[InertialState, ...],
    inertial_states: list[InertialState] | tuple[InertialState, ...],
) -> list[InertialState]:
    """Return causal, endpoint-constrained states for sonar-patch deskew.

    Direct acoustic updates improve the low-frequency displacement of a
    terrain patch, but applying every scan-to-map correction as an
    instantaneous pose jump can emboss estimator update noise into a dense
    10 Hz surface.  Over the already-complete fixed-lag window, this routine
    retains the smooth IMU/DVL relative motion and interpolates the acoustic
    correction between the two window endpoints on SE(3).  Both endpoints
    therefore equal the causal acoustic predictions exactly, every interior
    state remains independent of simulator truth, and no observation later
    than the terrain-factor timestamp is used.
    """

    acoustic = list(acoustic_states)
    inertial = list(inertial_states)
    if len(acoustic) != len(inertial):
        raise ValueError("acoustic and inertial state sequences must have equal length")
    if not acoustic:
        return []
    if not all(isinstance(state, InertialState) for state in acoustic + inertial):
        raise TypeError("deskew inputs must contain only inertial states")

    acoustic_times = np.asarray(
        [state.timestamp for state in acoustic], dtype=np.float64
    )
    inertial_times = np.asarray(
        [state.timestamp for state in inertial], dtype=np.float64
    )
    if (
        not np.isfinite(acoustic_times).all()
        or not np.isfinite(inertial_times).all()
        or np.any(np.diff(acoustic_times) < 0.0)
        or not np.allclose(acoustic_times, inertial_times, atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError("deskew state timestamps must be finite, ordered, and paired")
    if len(acoustic) == 1:
        return [acoustic[0].copy()]

    duration = float(acoustic_times[-1] - acoustic_times[0])
    if duration <= 0.0:
        fractions = np.linspace(0.0, 1.0, len(acoustic), dtype=np.float64)
    else:
        fractions = (acoustic_times - acoustic_times[0]) / duration

    position_correction_start = (
        acoustic[0].position - inertial[0].position
    )
    position_correction_end = (
        acoustic[-1].position - inertial[-1].position
    )
    rotation_correction_start = project_to_so3(
        acoustic[0].rotation @ inertial[0].rotation.T
    )
    rotation_correction_end = project_to_so3(
        acoustic[-1].rotation @ inertial[-1].rotation.T
    )
    correction_increment = so3_log(
        rotation_correction_start.T @ rotation_correction_end
    )

    fused: list[InertialState] = []
    for fraction, acoustic_state, inertial_state in zip(
        fractions, acoustic, inertial
    ):
        alpha = float(np.clip(fraction, 0.0, 1.0))
        state = acoustic_state.copy()
        state.position = inertial_state.position + (
            (1.0 - alpha) * position_correction_start
            + alpha * position_correction_end
        )
        interpolated_rotation_correction = project_to_so3(
            rotation_correction_start @ so3_exp(alpha * correction_increment)
        )
        state.rotation = project_to_so3(
            interpolated_rotation_correction @ inertial_state.rotation
        )
        fused.append(state)

    # Avoid allowing round-off in the Lie interpolation to move either
    # physical endpoint used by the terrain factor.
    fused[0].position = acoustic[0].position.copy()
    fused[0].rotation = acoustic[0].rotation.copy()
    fused[-1].position = acoustic[-1].position.copy()
    fused[-1].rotation = acoustic[-1].rotation.copy()
    return fused


def uniform_full_swath_scan_indices(
    scan_point_counts: np.ndarray | list[int] | tuple[int, ...],
    maximum_points: int,
) -> np.ndarray:
    """Select uniformly spaced complete sonar swaths within a point budget.

    Flattening a dense rolling MBES patch and then selecting uniformly by point
    index changes the spatial experiment as ping rate changes: a low-rate patch
    can retain complete cross-track fans while a high-rate patch retains only a
    handful of beams from every fan.  This helper first distributes the budget
    over acquisition epochs and keeps each selected fan complete.  If even one
    fan exceeds the budget, that single fan is returned and the matcher's
    ordinary point-level limit remains the final safety net.
    """

    if isinstance(maximum_points, (bool, np.bool_)):
        raise ValueError("maximum_points must be a positive integer")
    try:
        point_budget = int(maximum_points)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("maximum_points must be a positive integer") from exc
    if point_budget < 1 or point_budget != maximum_points:
        raise ValueError("maximum_points must be a positive integer")

    supplied = np.asarray(scan_point_counts)
    if supplied.ndim != 1:
        raise ValueError("scan_point_counts must be one-dimensional")
    if supplied.size == 0:
        return np.empty(0, dtype=np.int64)
    try:
        counts_float = supplied.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scan point counts must be non-negative integers") from exc
    if (
        not np.isfinite(counts_float).all()
        or np.any(counts_float < 0.0)
        or np.any(counts_float != np.floor(counts_float))
    ):
        raise ValueError("scan point counts must be non-negative integers")
    counts = counts_float.astype(np.int64)
    nonempty = np.flatnonzero(counts > 0)
    if nonempty.size == 0:
        return np.empty(0, dtype=np.int64)
    if int(np.sum(counts[nonempty])) <= point_budget:
        return nonempty.astype(np.int64, copy=False)

    maximum_scan_points = int(np.max(counts[nonempty]))
    retained_count = min(
        int(nonempty.size),
        max(1, point_budget // maximum_scan_points),
    )
    positions = np.linspace(
        0,
        nonempty.size - 1,
        retained_count,
        dtype=np.int64,
    )
    return nonempty[positions].astype(np.int64, copy=False)


def voxel_average_terrain_patch(
    points_body: np.ndarray,
    maximum_points: int,
    *,
    minimum_voxel_size_m: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Average all motion-compensated returns into a bounded x/y voxel set.

    Dense overlapping pings should improve a terrain patch through repeated
    observation, not be discarded merely because a nonlinear matcher has a
    finite point budget.  This deterministic reducer chooses the smallest
    practical horizontal voxel size that fits the budget (and is no smaller
    than ``minimum_voxel_size_m``), then returns the centroid of *every*
    occupied cell. Consequently every finite input point contributes to one
    output point while cross-track and along-track coverage are both retained.
    A nonzero minimum lets callers match the observation support to a smoothed
    prior-map scale. The second return value is the selected voxel size in
    metres, or zero when no reduction was required.
    """

    if isinstance(maximum_points, (bool, np.bool_)):
        raise ValueError("maximum_points must be a positive integer")
    try:
        point_budget = int(maximum_points)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("maximum_points must be a positive integer") from exc
    if point_budget < 1 or point_budget != maximum_points:
        raise ValueError("maximum_points must be a positive integer")
    try:
        minimum_voxel_size = float(minimum_voxel_size_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "minimum_voxel_size_m must be finite and non-negative"
        ) from exc
    if not np.isfinite(minimum_voxel_size) or minimum_voxel_size < 0.0:
        raise ValueError(
            "minimum_voxel_size_m must be finite and non-negative"
        )

    try:
        points = np.asarray(points_body, dtype=np.float64).reshape(-1, 3)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("points_body must be an Nx3 numeric array") from exc
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] <= point_budget and minimum_voxel_size == 0.0:
        return points.copy(), 0.0

    horizontal = points[:, :2]
    origin = np.min(horizontal, axis=0)
    extent = np.ptp(horizontal, axis=0)
    maximum_extent = float(np.max(extent))
    area = float(max(extent[0] * extent[1], 0.0))
    voxel_size = max(
        math.sqrt(max(area, 1.0e-12) / point_budget),
        maximum_extent / point_budget,
        minimum_voxel_size,
        1.0e-6,
    )

    inverse = None
    voxel_count = points.shape[0]
    for _ in range(64):
        keys = np.floor((horizontal - origin) / voxel_size).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        voxel_count = int(np.max(inverse)) + 1
        if voxel_count <= point_budget:
            break
        # The square root is the 2-D occupancy scaling.  A small margin avoids
        # oscillating one or two cells above the requested integer budget.
        voxel_size *= max(
            1.02,
            1.01 * math.sqrt(voxel_count / float(point_budget)),
        )
    else:  # pragma: no cover - geometric growth guarantees convergence
        raise RuntimeError("terrain voxel reduction failed to reach its budget")

    assert inverse is not None
    sums = np.zeros((voxel_count, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)
    np.add.at(sums, inverse, points)
    averaged = sums / counts[:, None]
    return averaged, float(voxel_size)


def voxel_average_backscatter_patch(
    points_body: np.ndarray,
    intensities: np.ndarray,
    maximum_points: int,
    *,
    minimum_voxel_size_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Average every registered intensity sample in horizontal body voxels."""

    if isinstance(maximum_points, (bool, np.bool_)):
        raise ValueError("maximum_points must be a positive integer")
    try:
        point_budget = int(maximum_points)
        minimum_voxel_size = float(minimum_voxel_size_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid backscatter voxel configuration") from exc
    if point_budget < 1 or point_budget != maximum_points:
        raise ValueError("maximum_points must be a positive integer")
    if not np.isfinite(minimum_voxel_size) or minimum_voxel_size < 0.0:
        raise ValueError(
            "minimum_voxel_size_m must be finite and non-negative"
        )
    try:
        points = np.asarray(points_body, dtype=np.float64).reshape(-1, 3)
        values = np.asarray(intensities, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("backscatter samples must be numeric arrays") from exc
    if points.shape[0] != values.size:
        raise ValueError("points_body and intensities must have equal size")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(values)
    points = points[finite]
    values = values[finite]
    if points.shape[0] == 0:
        return points.copy(), values.copy(), 0.0
    if points.shape[0] <= point_budget and minimum_voxel_size == 0.0:
        return points.copy(), values.copy(), 0.0

    horizontal = points[:, :2]
    origin = np.min(horizontal, axis=0)
    extent = np.ptp(horizontal, axis=0)
    voxel_size = max(
        math.sqrt(max(float(extent[0] * extent[1]), 1.0e-12) / point_budget),
        float(np.max(extent)) / point_budget,
        minimum_voxel_size,
        1.0e-6,
    )
    inverse = None
    voxel_count = points.shape[0]
    for _ in range(64):
        keys = np.floor((horizontal - origin) / voxel_size).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        voxel_count = int(np.max(inverse)) + 1
        if voxel_count <= point_budget:
            break
        voxel_size *= max(
            1.02,
            1.01 * math.sqrt(voxel_count / float(point_budget)),
        )
    else:  # pragma: no cover - geometric growth guarantees convergence
        raise RuntimeError("backscatter voxel reduction failed")

    assert inverse is not None
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)
    point_sums = np.zeros((voxel_count, 3), dtype=np.float64)
    intensity_sums = np.zeros(voxel_count, dtype=np.float64)
    np.add.at(point_sums, inverse, points)
    np.add.at(intensity_sums, inverse, values)
    return (
        point_sums / counts[:, None],
        intensity_sums / counts,
        float(voxel_size),
    )


def match_bathymetric_terrain(
    terrain: MeasuredBathymetryGrid,
    points_sensor: np.ndarray,
    state: InertialState,
    *,
    config: TerrainMatcherConfig | None = None,
    sensor_rotation_body: np.ndarray | None = None,
    sensor_translation_body: np.ndarray | None = None,
    points_are_body_frame: bool = False,
) -> TerrainMatchResult:
    """Stateless convenience wrapper around :class:`BathymetricTerrainMatcher`."""

    return BathymetricTerrainMatcher(terrain, config).match(
        points_sensor,
        state,
        sensor_rotation_body=sensor_rotation_body,
        sensor_translation_body=sensor_translation_body,
        points_are_body_frame=points_are_body_frame,
    )


def apply_terrain_match(
    estimator: AcousticESIKF,
    result: TerrainMatchResult,
    *,
    pose_tolerance: float = 1.0e-9,
) -> bool:
    """Fuse one accepted, current terrain match as a planar EKF measurement.

    This is intentionally a covariance-aware measurement update, not a pose
    overwrite.  It never uses ground truth.  Rejected/non-finite results and
    results computed from an earlier filter pose return ``False`` without
    mutating the estimator.
    """

    if not isinstance(estimator, AcousticESIKF):
        raise TypeError("estimator must be an AcousticESIKF")
    if not isinstance(result, TerrainMatchResult):
        raise TypeError("result must be a TerrainMatchResult")
    if not estimator.initialized or not result.accepted:
        return False
    state = estimator.state
    if (
        abs(float(state.timestamp) - result.state_timestamp) > pose_tolerance
        or not np.allclose(
            state.position, result.base_position, atol=pose_tolerance, rtol=0.0
        )
        or not np.allclose(
            state.rotation, result.base_rotation, atol=pose_tolerance, rtol=0.0
        )
    ):
        return False
    measurement = _observable_measurement(
        result.correction,
        result.information,
        state,
        yaw_rank_scale_m=result.yaw_rank_scale_m,
        minimum_scaled_information_eigenvalue=(
            result.minimum_scaled_information_eigenvalue
        ),
        minimum_information_eigen_ratio=(
            result.minimum_information_eigen_ratio
        ),
    )
    if measurement is None:
        return False
    residual, observation, covariance = measurement
    # The match result already passed this same NIS gate, but retain it here
    # as defence against accidental result corruption or a changed covariance.
    return estimator._kalman_update(
        residual,
        observation,
        covariance,
        mahalanobis_limit=result.maximum_innovation_nis,
    )
