"""Keyframe and loop-closure back end for acoustic SLAM.

The high-rate estimator in :mod:`esikf` deliberately remains a local
odometry front end, just like FAST-LIO/FAST-LIVO.  This module adds the part
needed to make a long underwater survey a SLAM system: repeat-place
recognition, geometric MBES verification, and a robust planar pose graph.

Survey AUVs are normally roll/pitch stabilised and have pressure depth, so the
graph optimises the weakly drifting ``x, y, yaw`` states while the filter
continues to own roll, pitch, depth, velocity, and IMU biases.  The historical
NumPy batch solver remains the default reference, while a selectable GTSAM
iSAM2 incremental fixed-lag backend supports controlled A/B experiments.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .math3d import rotation_from_rpy, rpy_from_rotation, wrap_angle


@dataclass
class PoseGraphConfig:
    enabled: bool = True
    keyframe_distance_m: float = 15.0
    keyframe_max_interval_s: float = 2.0
    loop_min_separation_s: float = 120.0
    loop_search_radius_m: float = 60.0
    loop_max_yaw_difference_deg: float = 80.0
    loop_descriptor_threshold: float = 0.24
    loop_registration_score_threshold: float = 0.32
    loop_max_candidates: int = 6
    # A stricter appearance threshold also permits place proposals outside
    # the current odometric search radius. MBES registration and consecutive
    # confirmation still have to verify them, providing true relocalization
    # instead of making loop closure depend entirely on the drifting pose.
    loop_global_descriptor_threshold: float = 0.12
    loop_global_max_candidates: int = 2
    loop_relocalization_seed_threshold_m: float = 10.0
    loop_relocalization_consistency_m: float = 3.0
    loop_relocalization_consistency_deg: float = 5.0
    loop_period_tolerance_s: float = 6.0
    loop_confirmation_count: int = 2
    loop_sequence_max_gap: int = 3
    icp_max_points: int = 280
    icp_iterations: int = 12
    icp_max_correspondence_m: float = 2.4
    icp_max_rmse_m: float = 0.34
    ordered_registration_max_rmse_m: float = 0.28
    icp_min_inliers: int = 70
    icp_max_translation_correction_m: float = 60.0
    icp_max_yaw_correction_deg: float = 30.0
    # The following deviations describe accumulated random-walk error over a
    # fixed physical duration, not over "one keyframe".  Leaving the reference
    # unset preserves the historical behaviour for existing callers; survey
    # applications should calibrate and set it explicitly so changing graph
    # cadence cannot silently change the information accumulated per minute.
    odometry_reference_interval_s: float | None = None
    odometry_xy_std_m: float = 0.32
    odometry_yaw_std_deg: float = 0.8
    loop_xy_std_m: float = 0.16
    loop_yaw_std_deg: float = 0.4
    robust_loop_delta: float = 3.0
    # A direct backscatter alignment that passes graph-export validation is
    # retained as an explicit, conservatively weighted binary factor. The
    # lower bounds include
    # intensity-map interpolation and the short propagation from the image
    # update to the next acoustic keyframe; they prevent thousands of pixels
    # from being mistaken for thousands of independent measurements.
    intensity_factor_enabled: bool = True
    # The prepared public backscatter raster has 5 m cells and the online
    # intensity surface uses 3 m voxels.  Sub-cell alignment is possible, but
    # treating it as centimetre-level independent information is not.  These
    # floors also cover reference-submap pose uncertainty, which the compact
    # graph does not explicitly marginalize into each image factor.
    intensity_factor_min_xy_std_m: float = 1.0
    intensity_factor_min_yaw_std_deg: float = 2.0
    intensity_factor_max_xy_std_m: float = 5.0
    intensity_factor_max_yaw_std_deg: float = 10.0
    # Pre-fit normalized-residual limits at 99% for chi-square variables with
    # two and three observable planar directions.  This is deliberately named
    # a compatibility test rather than a full NIS: the compact graph does not
    # form the joint marginal covariance of the two endpoint poses here.
    intensity_factor_prefit_chi2_rank2: float = 9.210
    intensity_factor_prefit_chi2_rank3: float = 11.345
    robust_intensity_delta: float = 2.5
    # A match against a surveyed bathymetry grid is an absolute world-frame
    # observation, unlike the binary self-map factors above.  Keep it behind
    # its own switch/cadence and robustify its *whitened* residual so one
    # perceptually aliased terrain match cannot drag the whole trajectory.
    terrain_prior_enabled: bool = True
    terrain_prior_robust_kernel: str = "huber"
    robust_terrain_prior_delta: float = 2.5
    # A confirmed whole-map relocalization is first re-seeded to zero
    # residual, then kept as a redescending factor in case later independent
    # evidence contradicts it.  This policy is deliberately separate from
    # ordinary local terrain matches, which retain the historical Huber loss.
    relocalization_prior_robust_kernel: str = "cauchy"
    robust_relocalization_prior_delta: float = 2.5
    terrain_prior_optimize_interval: int = 1
    # Terrain-map curvature can make the local optimum look extremely
    # precise even though the graph seed has accumulated metres of systematic
    # odometry drift.  Consequently this is a deliberately broad pre-fit
    # compatibility guard, not a classical NIS gate: the latter would require
    # a graph marginal covariance that this compact backend does not retain.
    # The defaults preserve legitimate corrections throughout the matcher's
    # nominal +/-30 m search basin while still rejecting a confidently wrong
    # multi-modal jump.  Tighter deployments can override the rank-specific
    # limits after calibrating prediction consistency on representative runs.
    terrain_prior_prefit_chi2_rank2: float = 100_000.0
    terrain_prior_prefit_chi2_rank3: float = 150_000.0
    # Convert yaw to a metre-like coordinate before numerical rank testing.
    # The quadratic form itself is invariant under this change of units.
    terrain_prior_prefit_yaw_scale_m: float = 20.0
    # Direct updates can arrive much faster than geometry keyframes.  Without
    # a cadence limit, every accepted update may create a pose-only node and
    # defeat the bounded geometry-keyframe spacing on a long survey.  Zero
    # preserves the unrestricted research default; long-running deployments
    # can retain periodic independent image factors at a controlled rate.
    intensity_factor_min_interval_s: float = 0.0
    intensity_factor_optimize_interval: int = 1
    intensity_seed_search_radius_m: float = 15.0
    intensity_seed_max_yaw_difference_deg: float = 25.0
    intensity_seed_max_candidates: int = 6
    optimizer_iterations: int = 12
    sidescan_descriptor_bins: int = 32
    # The solver is explicit so a run can change only the back end while
    # preserving the acoustic front end and every generated factor.  NumPy is
    # kept as the historical default; the GTSAM candidate must be requested.
    optimizer_backend: str = "numpy_batch"
    fixed_lag_seconds: float = 180.0
    isam_relinearize_threshold: float = 0.01
    isam_relinearize_skip: int = 1


@dataclass(frozen=True)
class PoseGraphEdge:
    source: int
    target: int
    measurement: np.ndarray
    information: np.ndarray
    kind: str = "odometry"
    score: float = 0.0


@dataclass(frozen=True)
class PoseGraphPrior:
    """Unary absolute pose observation expressed in the world frame."""

    target: int
    measurement: np.ndarray
    information: np.ndarray
    kind: str = "terrain_prior"
    score: float = 0.0
    # Ordinary single-patch terrain matches retain a bounded Huber influence.
    # A separately confirmed relocalization may disable that kernel after it
    # has first re-seeded the graph into the verified absolute basin.
    robust: bool = True
    # Optional per-factor policy.  ``None`` uses the ordinary terrain-prior
    # settings from :class:`PoseGraphConfig` and keeps old factor streams
    # backward compatible.
    robust_kernel: str | None = None
    robust_delta: float | None = None


@dataclass
class AcousticKeyframe:
    index: int
    timestamp: float
    raw_position: np.ndarray
    raw_rotation: np.ndarray
    points_body: np.ndarray
    intensities: np.ndarray | None
    descriptor: np.ndarray
    observation_id: int
    # Pose-only direct-intensity nodes belong to the optimization chain but
    # must not alter geometry-keyframe cadence or loop-sequence counting.
    loop_eligible: bool = True
    geometry_sequence_index: int = -1

    @property
    def raw_pose2(self) -> np.ndarray:
        yaw = float(rpy_from_rotation(self.raw_rotation)[2])
        return np.array(
            [self.raw_position[0], self.raw_position[1], yaw],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RegistrationResult:
    accepted: bool
    rotation_target_source: np.ndarray
    translation_target_source: np.ndarray
    rmse: float
    inliers: int
    translation_correction: float
    yaw_correction_deg: float


@dataclass(frozen=True)
class PoseGraphDiagnostics:
    keyframes: int
    odometry_edges: int
    loop_edges: int
    relocalization_edges: int
    intensity_edges: int
    rejected_intensity_factors: int
    throttled_intensity_factors: int
    intensity_factor_rejection_reasons: dict[str, int]
    last_intensity_factor_rejection_reason: str | None
    last_intensity_prefit_performed: bool
    last_intensity_prefit_chi2: float
    last_intensity_prefit_dof: int
    pending_intensity_factors: int
    optimization_runs: int
    initial_cost: float
    final_cost: float
    # Appended defaults preserve positional construction compatibility for
    # callers written before absolute terrain factors were introduced.
    terrain_prior_count: int = 0
    accepted_terrain_priors: int = 0
    rejected_terrain_priors: int = 0
    terrain_prior_rejection_reasons: dict[str, int] = field(
        default_factory=dict
    )
    last_terrain_prior_rejection_reason: str | None = None
    last_terrain_prefit_performed: bool = False
    last_terrain_prefit_bypassed_for_relocalization: bool = False
    last_terrain_prefit_chi2: float = 0.0
    last_terrain_prefit_dof: int = 0
    terrain_relocalization_seeds: int = 0
    last_terrain_relocalization_translation_m: float = 0.0
    last_terrain_relocalization_yaw_deg: float = 0.0
    last_terrain_relocalization_anchor_index: int | None = None
    optimizer_backend: str = "numpy_batch"
    isam_updates: int = 0
    fixed_lag_updates: int = 0
    fixed_lag_active_variables: int = 0
    fixed_lag_marginalized_variables: int = 0
    fixed_lag_global_only_factors: int = 0
    global_anchor_bridge_updates: int = 0
    global_anchor_bridge_max_translation_m: float = 0.0
    global_anchor_bridge_max_abs_yaw_deg: float = 0.0
    optimizer_rebuilds: int = 0
    global_isam_initial_cost: float = 0.0
    global_isam_final_cost: float = 0.0


def _relative_pose2(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """Return pose B in the planar coordinate frame of pose A."""

    xa, ya, yaw_a = np.asarray(pose_a, dtype=np.float64).reshape(3)
    xb, yb, yaw_b = np.asarray(pose_b, dtype=np.float64).reshape(3)
    dx, dy = xb - xa, yb - ya
    c, s = np.cos(yaw_a), np.sin(yaw_a)
    return np.array(
        [c * dx + s * dy, -s * dx + c * dy, wrap_angle(yaw_b - yaw_a)],
        dtype=np.float64,
    )


def _compose_pose2(pose_a: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """Apply a local planar increment to pose A."""

    xa, ya, yaw_a = np.asarray(pose_a, dtype=np.float64).reshape(3)
    dx, dy, dyaw = np.asarray(relative, dtype=np.float64).reshape(3)
    c, s = np.cos(yaw_a), np.sin(yaw_a)
    return np.array(
        [
            xa + c * dx - s * dy,
            ya + s * dx + c * dy,
            wrap_angle(yaw_a + dyaw),
        ],
        dtype=np.float64,
    )


def _edge_error_and_jacobians(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    measurement: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xa, ya, yaw_a = pose_a
    xb, yb, yaw_b = pose_b
    dx, dy = xb - xa, yb - ya
    c, s = np.cos(yaw_a), np.sin(yaw_a)
    predicted_x = c * dx + s * dy
    predicted_y = -s * dx + c * dy
    error = np.array(
        [
            predicted_x - measurement[0],
            predicted_y - measurement[1],
            wrap_angle(yaw_b - yaw_a - measurement[2]),
        ],
        dtype=np.float64,
    )
    jacobian_a = np.array(
        [
            [-c, -s, predicted_y],
            [s, -c, -predicted_x],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    jacobian_b = np.array(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return error, jacobian_a, jacobian_b


def _prior_error(pose: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    """Return a world-frame pose residual with a wrapped heading error."""

    error = np.asarray(pose, dtype=np.float64).reshape(3) - np.asarray(
        measurement, dtype=np.float64
    ).reshape(3)
    error[2] = wrap_angle(error[2])
    return error


def _validated_psd_information(information: np.ndarray) -> np.ndarray:
    """Return a symmetric PSD information matrix or raise ``ValueError``.

    A small negative eigenvalue can arise from floating-point assembly of a
    normal equation, so values within a scale-relative tolerance are clipped
    to zero.  Materially indefinite matrices are rejected rather than being
    silently turned into a different measurement model.
    """

    try:
        supplied = np.asarray(information, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError) as exc:
        raise ValueError("prior information must have shape (3, 3)") from exc
    if not np.isfinite(supplied).all():
        raise ValueError("prior information must be finite")
    scale = max(1.0, float(np.max(np.abs(supplied))))
    if float(np.max(np.abs(supplied - supplied.T))) > 1.0e-9 * scale:
        raise ValueError("prior information must be symmetric")
    symmetric = 0.5 * (supplied + supplied.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise ValueError("prior information eigendecomposition failed") from exc
    tolerance = 1.0e-9 * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("prior information must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    validated = (eigenvectors * eigenvalues) @ eigenvectors.T
    return 0.5 * (validated + validated.T)


class PlanarPoseGraph:
    """Small dense Gauss-Newton graph suitable for hundreds of keyframes."""

    def __init__(self, config: PoseGraphConfig | None = None) -> None:
        self.config = config or PoseGraphConfig()
        for name in (
            "terrain_prior_robust_kernel",
            "relocalization_prior_robust_kernel",
        ):
            if getattr(self.config, name) not in ("huber", "cauchy"):
                raise ValueError(f"{name} must be 'huber' or 'cauchy'")
        for name in (
            "robust_terrain_prior_delta",
            "robust_relocalization_prior_delta",
        ):
            value = float(getattr(self.config, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.poses = np.empty((0, 3), dtype=np.float64)
        self.edges: list[PoseGraphEdge] = []
        self.priors: list[PoseGraphPrior] = []
        self.optimization_runs = 0
        self.initial_cost = 0.0
        self.final_cost = 0.0
        self.isam_updates = 0
        self.fixed_lag_updates = 0
        self.fixed_lag_active_variables = 0
        self.fixed_lag_marginalized_variables = 0
        self._timestamps: list[float] = []
        self._pose_initializations: list[np.ndarray] = []
        self.optimization_boundaries: list[tuple[int, int, int]] = []

    @property
    def backend_name(self) -> str:
        return "numpy_batch"

    def add_pose(
        self, pose: np.ndarray, *, timestamp: float | None = None
    ) -> int:
        value = np.asarray(pose, dtype=np.float64).reshape(3).copy()
        if not np.isfinite(value).all():
            raise ValueError("pose must be finite")
        time_value = (
            self._timestamps[-1] + 1.0
            if timestamp is None and self._timestamps
            else float(len(self._timestamps))
            if timestamp is None
            else float(timestamp)
        )
        if not np.isfinite(time_value):
            raise ValueError("pose timestamp must be finite")
        if self._timestamps and time_value < self._timestamps[-1] - 1.0e-9:
            raise ValueError("pose timestamps must be monotonic")
        value[2] = wrap_angle(value[2])
        self._timestamps.append(time_value)
        self._pose_initializations.append(value.copy())
        value = value.reshape(1, 3)
        self.poses = np.concatenate((self.poses, value), axis=0)
        return self.poses.shape[0] - 1

    def truncate_poses(self, count: int) -> None:
        retained = int(count)
        if retained < 0 or retained > self.poses.shape[0]:
            raise ValueError("retained pose count is invalid")
        self.poses = self.poses[:retained].copy()
        del self._timestamps[retained:]
        del self._pose_initializations[retained:]

    def factor_stream_payload(self) -> dict[str, object]:
        """Return a JSON-compatible solver-independent factor stream."""

        return {
            "schema": "oceansim.planar_pose_graph_factor_stream.v1",
            "source_backend": self.backend_name,
            "pose_initializations": [
                pose.tolist() for pose in self._pose_initializations
            ],
            "pose_timestamps_s": [float(value) for value in self._timestamps],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "measurement": edge.measurement.tolist(),
                    "information": edge.information.tolist(),
                    "kind": edge.kind,
                    "score": edge.score,
                }
                for edge in self.edges
            ],
            "priors": [
                {
                    "target": prior.target,
                    "measurement": prior.measurement.tolist(),
                    "information": prior.information.tolist(),
                    "kind": prior.kind,
                    "score": prior.score,
                    "robust": prior.robust,
                    "robust_kernel": prior.robust_kernel,
                    "robust_delta": prior.robust_delta,
                }
                for prior in self.priors
            ],
            "optimization_boundaries": [
                {
                    "poses": poses,
                    "edges": edges,
                    "priors": priors,
                }
                for poses, edges, priors in self.optimization_boundaries
            ],
            "final_online_poses": self.poses.tolist(),
            "gauge_pose": (
                self.poses[0].tolist() if self.poses.shape[0] else None
            ),
            "full_history_diagnostic_poses": getattr(
                self, "global_isam_poses", np.empty((0, 3), dtype=np.float64)
            ).tolist(),
            "global_anchor_bridge": {
                "updates": int(
                    getattr(self, "global_anchor_bridge_updates", 0)
                ),
                "maximum_translation_m": float(
                    getattr(
                        self,
                        "global_anchor_bridge_max_translation_m",
                        0.0,
                    )
                ),
                "maximum_abs_yaw_deg": float(
                    getattr(
                        self,
                        "global_anchor_bridge_max_abs_yaw_deg",
                        0.0,
                    )
                ),
            },
            "objective_configuration": {
                "terrain_prior_robust_kernel": (
                    self.config.terrain_prior_robust_kernel
                ),
                "robust_terrain_prior_delta": float(
                    self.config.robust_terrain_prior_delta
                ),
                "relocalization_prior_robust_kernel": (
                    self.config.relocalization_prior_robust_kernel
                ),
                "robust_relocalization_prior_delta": float(
                    self.config.robust_relocalization_prior_delta
                ),
                "robust_loop_delta": float(self.config.robust_loop_delta),
                "robust_intensity_delta": float(
                    self.config.robust_intensity_delta
                ),
                "optimizer_iterations": int(self.config.optimizer_iterations),
                "fixed_lag_seconds": float(self.config.fixed_lag_seconds),
                "isam_relinearize_threshold": float(
                    self.config.isam_relinearize_threshold
                ),
                "isam_relinearize_skip": int(
                    self.config.isam_relinearize_skip
                ),
            },
        }

    def add_edge(
        self,
        source: int,
        target: int,
        measurement: np.ndarray,
        information: np.ndarray,
        *,
        kind: str,
        score: float = 0.0,
    ) -> None:
        if source < 0 or target <= source or target >= self.poses.shape[0]:
            raise ValueError("pose graph edge indices are invalid")
        self.edges.append(
            PoseGraphEdge(
                source=source,
                target=target,
                measurement=np.asarray(measurement, dtype=np.float64).reshape(3),
                information=np.asarray(information, dtype=np.float64).reshape(3, 3),
                kind=str(kind),
                score=float(score),
            )
        )

    def add_prior(
        self,
        target: int,
        measurement: np.ndarray,
        information: np.ndarray,
        *,
        kind: str = "terrain_prior",
        score: float = 0.0,
        robust: bool = True,
        robust_kernel: str | None = None,
        robust_delta: float | None = None,
    ) -> None:
        """Add a unary absolute ``x, y, yaw`` observation in world axes.

        Node zero remains the graph's fixed gauge anchor.  A verified global
        relocalization can explicitly rebase that gauge before this factor is
        added.  Ordinary terrain priors are robustified in standardized
        (information-whitened) residual units; a caller may disable the
        kernel only after an independent confirmation policy has run.
        """

        if target < 0 or target >= self.poses.shape[0]:
            raise ValueError("pose graph prior target index is invalid")
        try:
            value = np.asarray(measurement, dtype=np.float64).reshape(3).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError("prior measurement must have shape (3,)") from exc
        if not np.isfinite(value).all():
            raise ValueError("prior measurement must be finite")
        value[2] = wrap_angle(value[2])
        factor_kind = str(kind)
        if factor_kind != "terrain_prior":
            raise ValueError("only terrain_prior unary factors are supported")
        use_robust = bool(robust)
        if (robust_kernel is None) != (robust_delta is None):
            raise ValueError(
                "prior robust_kernel and robust_delta must be supplied together"
            )
        factor_robust_kernel = None
        factor_robust_delta = None
        if robust_kernel is not None:
            if not use_robust:
                raise ValueError(
                    "a non-robust prior cannot override a robust kernel"
                )
            factor_robust_kernel = str(robust_kernel)
            if factor_robust_kernel not in ("huber", "cauchy"):
                raise ValueError("unsupported prior robust kernel")
            factor_robust_delta = float(robust_delta)
            if not np.isfinite(factor_robust_delta) or factor_robust_delta <= 0.0:
                raise ValueError("prior robust delta must be finite and positive")
        self.priors.append(
            PoseGraphPrior(
                target=int(target),
                measurement=value,
                information=_validated_psd_information(information),
                kind=factor_kind,
                score=float(score),
                robust=use_robust,
                robust_kernel=factor_robust_kernel,
                robust_delta=factor_robust_delta,
            )
        )

    def _prior_robust_policy(
        self, prior: PoseGraphPrior
    ) -> tuple[str | None, float | None]:
        """Resolve the robust policy for one absolute factor."""

        if not prior.robust:
            return None, None
        if prior.robust_kernel is not None:
            return prior.robust_kernel, float(prior.robust_delta)
        return (
            self.config.terrain_prior_robust_kernel,
            float(self.config.robust_terrain_prior_delta),
        )

    def _cost(self, poses: np.ndarray) -> float:
        cost = 0.0
        for edge in self.edges:
            error, _, _ = _edge_error_and_jacobians(
                poses[edge.source], poses[edge.target], edge.measurement
            )
            squared = float(error @ edge.information @ error)
            if edge.kind in ("loop", "intensity"):
                norm = np.sqrt(max(squared, 0.0))
                delta = (
                    self.config.robust_loop_delta
                    if edge.kind == "loop"
                    else self.config.robust_intensity_delta
                )
                if norm > delta:
                    squared = 2.0 * delta * norm - delta * delta
            cost += squared
        for prior in self.priors:
            error = _prior_error(poses[prior.target], prior.measurement)
            squared = float(error @ prior.information @ error)
            if prior.robust:
                norm = np.sqrt(max(squared, 0.0))
                kernel, configured_delta = self._prior_robust_policy(prior)
                delta = max(float(configured_delta), 1.0e-12)
                if kernel == "huber":
                    if norm > delta:
                        squared = 2.0 * delta * norm - delta * delta
                else:
                    squared = delta * delta * np.log1p(
                        squared / (delta * delta)
                    )
            cost += squared
        return cost

    def optimize(self) -> np.ndarray:
        count = self.poses.shape[0]
        if count < 2 or (not self.edges and not self.priors):
            return self.poses.copy()
        poses = self.poses.copy()
        self.initial_cost = self._cost(poses)
        variable_count = 3 * (count - 1)  # Pose zero is the gauge anchor.
        for _ in range(self.config.optimizer_iterations):
            hessian = np.eye(variable_count, dtype=np.float64) * 1.0e-8
            gradient = np.zeros(variable_count, dtype=np.float64)
            for edge in self.edges:
                error, jacobian_a, jacobian_b = _edge_error_and_jacobians(
                    poses[edge.source], poses[edge.target], edge.measurement
                )
                information = edge.information
                if edge.kind in ("loop", "intensity"):
                    whitened_norm = np.sqrt(
                        max(float(error @ information @ error), 0.0)
                    )
                    robust_delta = (
                        self.config.robust_loop_delta
                        if edge.kind == "loop"
                        else self.config.robust_intensity_delta
                    )
                    if whitened_norm > robust_delta:
                        information = information * (
                            robust_delta / whitened_norm
                        )

                blocks = ((edge.source, jacobian_a), (edge.target, jacobian_b))
                for node_i, jacobian_i in blocks:
                    if node_i == 0:
                        continue
                    slice_i = slice(3 * (node_i - 1), 3 * node_i)
                    gradient[slice_i] += jacobian_i.T @ information @ error
                    for node_j, jacobian_j in blocks:
                        if node_j == 0:
                            continue
                        slice_j = slice(3 * (node_j - 1), 3 * node_j)
                        hessian[slice_i, slice_j] += (
                            jacobian_i.T @ information @ jacobian_j
                        )
            for prior in self.priors:
                # Pose zero is deliberately fixed even when an absolute
                # observation is attached to it.
                if prior.target == 0:
                    continue
                error = _prior_error(poses[prior.target], prior.measurement)
                information = prior.information
                if prior.robust:
                    whitened_squared = max(
                        float(error @ information @ error), 0.0
                    )
                    whitened_norm = np.sqrt(whitened_squared)
                    robust_kernel, configured_delta = (
                        self._prior_robust_policy(prior)
                    )
                    robust_delta = max(float(configured_delta), 1.0e-12)
                    if robust_kernel == "huber":
                        if whitened_norm > robust_delta:
                            information = information * (
                                robust_delta / whitened_norm
                            )
                    else:
                        information = information * (
                            1.0
                            / (
                                1.0
                                + whitened_squared
                                / (robust_delta * robust_delta)
                            )
                        )
                target_slice = slice(
                    3 * (prior.target - 1), 3 * prior.target
                )
                # The unary residual Jacobian is identity in world-frame
                # pose coordinates (with yaw locally linear after wrapping).
                gradient[target_slice] += information @ error
                hessian[target_slice, target_slice] += information
            try:
                step = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, -gradient, rcond=1.0e-10)[0]
            if not np.isfinite(step).all():
                break
            poses[1:, :] += step.reshape(-1, 3)
            poses[:, 2] = wrap_angle(poses[:, 2])
            if float(np.max(np.abs(step))) < 1.0e-5:
                break
        self.poses = poses
        self.final_cost = self._cost(poses)
        self.optimization_runs += 1
        self.optimization_boundaries.append(
            (self.poses.shape[0], len(self.edges), len(self.priors))
        )
        return poses.copy()


class GTSAMPlanarPoseGraph(PlanarPoseGraph):
    """Incremental GTSAM backend with a fixed-lag and a global iSAM2 graph.

    The fixed-lag smoother is the bounded real-time estimator and supplies the
    public ``poses`` used by online ``map->odom`` correction.  Once a pose is
    marginalized, its last fixed-lag estimate remains in that array.  A second
    iSAM2 instance retains the complete graph for old-node constraints and a
    separately reported full-history diagnostic solution.  If a newly arrived
    factor spans beyond the lag, only its solved rigid map anchor is transferred
    to the live window; the complete trajectory is never substituted for the
    fixed-lag online estimate.

    No silent NumPy fallback is provided.  If GTSAM is unavailable, creating
    the production acoustic graph fails with an actionable dependency error.
    """

    BACKEND_NAME = "gtsam_isam2_incremental_fixed_lag"

    def __init__(self, config: PoseGraphConfig | None = None) -> None:
        super().__init__(config)
        if (
            not np.isfinite(self.config.fixed_lag_seconds)
            or self.config.fixed_lag_seconds <= 0.0
        ):
            raise ValueError("fixed_lag_seconds must be finite and positive")
        if (
            not np.isfinite(self.config.isam_relinearize_threshold)
            or self.config.isam_relinearize_threshold <= 0.0
        ):
            raise ValueError(
                "isam_relinearize_threshold must be finite and positive"
            )
        if int(self.config.isam_relinearize_skip) < 1:
            raise ValueError("isam_relinearize_skip must be at least one")

        try:
            import gtsam  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "The production acoustic pose graph requires GTSAM. Install "
                "requirements-slam.txt before running the SLAM demo."
            ) from exc
        self._gtsam = gtsam
        if hasattr(gtsam, "IncrementalFixedLagSmoother"):
            # GTSAM 4.3 promotes the fixed-lag API into the main module.
            self._fixed_lag_api = gtsam
        else:
            try:
                import gtsam_unstable  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "This GTSAM build does not expose an incremental "
                    "fixed-lag smoother"
                ) from exc
            if not hasattr(gtsam_unstable, "IncrementalFixedLagSmoother"):
                raise ImportError(
                    "This GTSAM build does not expose an incremental "
                    "fixed-lag smoother"
                )
            self._fixed_lag_api = gtsam_unstable
        self._synced_pose_count = 0
        self._synced_edge_count = 0
        self._synced_prior_count = 0
        self._synced_pose_snapshot = np.empty((0, 3), dtype=np.float64)
        self._synced_edge_ids: tuple[int, ...] = ()
        self._synced_prior_ids: tuple[int, ...] = ()
        self.isam_updates = 0
        self.fixed_lag_updates = 0
        self.fixed_lag_active_variables = 0
        self.fixed_lag_marginalized_variables = 0
        self.global_isam_poses = np.empty((0, 3), dtype=np.float64)
        self.global_initial_cost = 0.0
        self.global_final_cost = 0.0
        self.rebuilds = 0
        self.global_only_factor_count = 0
        self._global_only_factor_ids: set[tuple[str, int]] = set()
        self.global_anchor_bridge_updates = 0
        self.global_anchor_bridge_max_translation_m = 0.0
        self.global_anchor_bridge_max_abs_yaw_deg = 0.0
        self._new_solvers()

    @property
    def backend_name(self) -> str:
        return self.BACKEND_NAME

    def _new_isam_params(self):
        params = self._gtsam.ISAM2Params()
        params.setRelinearizeThreshold(
            float(self.config.isam_relinearize_threshold)
        )
        params.relinearizeSkip = int(self.config.isam_relinearize_skip)
        params.enableDetailedResults = True
        params.evaluateNonlinearError = True
        return params

    def _new_solvers(self) -> None:
        self._global_isam = self._gtsam.ISAM2(self._new_isam_params())
        self._fixed_lag = self._fixed_lag_api.IncrementalFixedLagSmoother(
            float(self.config.fixed_lag_seconds), self._new_isam_params()
        )
        self._fixed_lag_active_keys: set[int] = set()

    def add_pose(
        self, pose: np.ndarray, *, timestamp: float | None = None
    ) -> int:
        return super().add_pose(pose, timestamp=timestamp)

    def _key(self, index: int) -> int:
        return int(self._gtsam.symbol("x", int(index)))

    def _pose2(self, pose: np.ndarray):
        x, y, yaw = np.asarray(pose, dtype=np.float64).reshape(3)
        return self._gtsam.Pose2(float(x), float(y), float(yaw))

    @staticmethod
    def _pose_array(pose) -> np.ndarray:
        return np.array(
            [float(pose.x()), float(pose.y()), float(pose.theta())],
            dtype=np.float64,
        )

    @staticmethod
    def _local_delta_to_world(pose: np.ndarray) -> np.ndarray:
        """Map a GTSAM Pose2 tangent perturbation to additive world axes."""

        yaw = float(np.asarray(pose, dtype=np.float64).reshape(3)[2])
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _noise_model(
        self,
        information: np.ndarray,
        *,
        robust_kernel: str | None = None,
        robust_delta: float | None = None,
    ):
        validated = _validated_psd_information(information)
        eigenvalues, eigenvectors = np.linalg.eigh(validated)
        # Supplying the symmetric square root directly preserves rank-deficient
        # acoustic factors.  Adding an arbitrary diagonal floor here would turn
        # an unobservable direction into invented information.
        square_root = (
            eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
        ) @ eigenvectors.T
        base = self._gtsam.noiseModel.Gaussian.SqrtInformation(
            np.ascontiguousarray(square_root, dtype=np.float64), False
        )
        if robust_kernel is None:
            return base
        delta = float(robust_delta)
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("robust loss delta must be finite and positive")
        if robust_kernel == "huber":
            loss = self._gtsam.noiseModel.mEstimator.Huber.Create(delta)
        elif robust_kernel == "cauchy":
            loss = self._gtsam.noiseModel.mEstimator.Cauchy.Create(delta)
        else:
            raise ValueError(f"unsupported GTSAM robust kernel: {robust_kernel}")
        return self._gtsam.noiseModel.Robust.Create(loss, base)

    def _edge_factor(self, edge: PoseGraphEdge):
        robust_kernel = None
        robust_delta = None
        if edge.kind == "loop":
            robust_kernel = "huber"
            robust_delta = self.config.robust_loop_delta
        elif edge.kind == "intensity":
            robust_kernel = "huber"
            robust_delta = self.config.robust_intensity_delta
        source_key = self._key(edge.source)
        target_key = self._key(edge.target)

        def error_function(_factor, values, jacobians):
            source = self._pose_array(values.atPose2(source_key))
            target = self._pose_array(values.atPose2(target_key))
            error, jacobian_source, jacobian_target = (
                _edge_error_and_jacobians(source, target, edge.measurement)
            )
            if jacobians is not None:
                jacobians[0] = np.asfortranarray(
                    jacobian_source @ self._local_delta_to_world(source)
                )
                jacobians[1] = np.asfortranarray(
                    jacobian_target @ self._local_delta_to_world(target)
                )
            return error

        return self._gtsam.CustomFactor(
            self._noise_model(
                edge.information,
                robust_kernel=robust_kernel,
                robust_delta=robust_delta,
            ),
            [source_key, target_key],
            error_function,
        )

    def _prior_factor(self, prior: PoseGraphPrior):
        robust_kernel, robust_delta = self._prior_robust_policy(prior)
        target_key = self._key(prior.target)

        def error_function(_factor, values, jacobians):
            target = self._pose_array(values.atPose2(target_key))
            error = _prior_error(target, prior.measurement)
            if jacobians is not None:
                jacobians[0] = np.asfortranarray(
                    self._local_delta_to_world(target)
                )
            return error

        return self._gtsam.CustomFactor(
            self._noise_model(
                prior.information,
                robust_kernel=robust_kernel,
                robust_delta=robust_delta,
            ),
            [target_key],
            error_function,
        )

    def _gauge_factor(self, pose: np.ndarray):
        return self._gtsam.PriorFactorPose2(
            self._key(0),
            self._pose2(pose),
            self._gtsam.noiseModel.Constrained.All(3),
        )

    def _normalise_timestamp_storage(self) -> None:
        pose_count = int(self.poses.shape[0])
        if len(self._timestamps) > pose_count:
            del self._timestamps[pose_count:]
        if len(self._pose_initializations) > pose_count:
            del self._pose_initializations[pose_count:]
        while len(self._timestamps) < pose_count:
            next_value = (
                self._timestamps[-1] + 1.0
                if self._timestamps
                else float(len(self._timestamps))
            )
            self._timestamps.append(next_value)
        while len(self._pose_initializations) < pose_count:
            self._pose_initializations.append(
                self.poses[len(self._pose_initializations)].copy()
            )

    def _existing_state_was_edited(self) -> bool:
        if self.poses.shape[0] < self._synced_pose_count:
            return True
        if len(self.edges) < self._synced_edge_count:
            return True
        if len(self.priors) < self._synced_prior_count:
            return True
        if self._synced_pose_count and not np.array_equal(
            self.poses[: self._synced_pose_count],
            self._synced_pose_snapshot[: self._synced_pose_count],
        ):
            return True
        if tuple(id(edge) for edge in self.edges[: self._synced_edge_count]) != (
            self._synced_edge_ids
        ):
            return True
        if tuple(
            id(prior) for prior in self.priors[: self._synced_prior_count]
        ) != self._synced_prior_ids:
            return True
        return False

    def _mark_synchronised(self) -> None:
        self._synced_pose_count = int(self.poses.shape[0])
        self._synced_edge_count = len(self.edges)
        self._synced_prior_count = len(self.priors)
        self._synced_pose_snapshot = self.poses.copy()
        self._synced_edge_ids = tuple(id(edge) for edge in self.edges)
        self._synced_prior_ids = tuple(id(prior) for prior in self.priors)

    def _pose_array_from_values(
        self, values, initial: np.ndarray
    ) -> np.ndarray:
        result = np.asarray(initial, dtype=np.float64).reshape(-1, 3).copy()
        for index in range(result.shape[0]):
            key = self._key(index)
            if values.exists(key):
                result[index] = self._pose_array(values.atPose2(key))
        result[:, 2] = wrap_angle(result[:, 2])
        return result

    def _store_fixed_lag_values(self, values) -> None:
        # Values contains only the live smoother window.  Updating it after
        # every chronological backlog step preserves a pose's last estimate
        # immediately before marginalization instead of leaving its raw seed.
        self.poses = self._pose_array_from_values(values, self.poses)

    def _mark_global_only_factor(
        self, category: str, factor: PoseGraphEdge | PoseGraphPrior
    ) -> None:
        """Record a unique factor that cannot enter the live lag window."""

        self._global_only_factor_ids.add((category, id(factor)))
        self.global_only_factor_count = len(self._global_only_factor_ids)

    def _bridge_global_anchor_to_fixed_lag_window(self) -> None:
        """Apply a cross-window global constraint as a rigid map anchor.

        Fixed-lag marginalization cannot later revive an old loop endpoint.
        The complete iSAM2 graph can still solve that constraint.  When such a
        factor has actually arrived, transfer only the latest global-to-window
        SE(2) transform to all live smoother poses.  Their relative geometry is
        unchanged and historical marginalized poses are not overwritten.
        """

        if self.global_only_factor_count <= 0 or not self._fixed_lag_active_keys:
            return
        active_indices = [
            index
            for index in range(self.poses.shape[0])
            if self._key(index) in self._fixed_lag_active_keys
        ]
        if not active_indices or self.global_isam_poses.shape != self.poses.shape:
            return
        anchor = active_indices[-1]
        fixed_anchor = self.poses[anchor].copy()
        global_anchor = self.global_isam_poses[anchor]
        yaw_delta = float(wrap_angle(global_anchor[2] - fixed_anchor[2]))
        c, s = np.cos(yaw_delta), np.sin(yaw_delta)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        translation = global_anchor[:2] - rotation @ fixed_anchor[:2]
        for index in active_indices:
            self.poses[index, :2] = rotation @ self.poses[index, :2] + translation
            self.poses[index, 2] = wrap_angle(
                self.poses[index, 2] + yaw_delta
            )
        self.global_anchor_bridge_updates += 1
        self.global_anchor_bridge_max_translation_m = max(
            self.global_anchor_bridge_max_translation_m,
            float(np.linalg.norm(translation)),
        )
        self.global_anchor_bridge_max_abs_yaw_deg = max(
            self.global_anchor_bridge_max_abs_yaw_deg,
            abs(float(np.rad2deg(yaw_delta))),
        )

    def _incremental_global_update(self):
        graph = self._gtsam.NonlinearFactorGraph()
        values = self._gtsam.Values()
        touched_keys: set[int] = set()
        if self._synced_pose_count == 0 and self.poses.shape[0]:
            graph.add(self._gauge_factor(self.poses[0]))
            touched_keys.add(self._key(0))
        for edge in self.edges[self._synced_edge_count :]:
            graph.add(self._edge_factor(edge))
            touched_keys.update((self._key(edge.source), self._key(edge.target)))
        for prior in self.priors[self._synced_prior_count :]:
            graph.add(self._prior_factor(prior))
            touched_keys.add(self._key(prior.target))
        for index in range(self._synced_pose_count, self.poses.shape[0]):
            key = self._key(index)
            if key not in touched_keys:
                raise RuntimeError(
                    f"new GTSAM pose x{index} has no connecting factor"
                )
            values.insert(key, self._pose2(self.poses[index]))
        result = self._global_isam.update(graph, values)
        # A second standard iSAM2 update lets variables selected by the
        # relinearization policy settle after a newly inserted nonlinear loop
        # or terrain factor.  It is not a hand-written batch iteration.
        self._global_isam.update()
        self.isam_updates += 1
        return result

    def _incremental_fixed_lag_update(self) -> None:
        # Insert a backlog chronologically, one timestamp at a time.  GTSAM's
        # IncrementalFixedLagSmoother cannot safely receive a single batch whose
        # timestamp span exceeds the lag: it may try to marginalize the first
        # new key while later factors in that same update still reference it,
        # which the Python binding reports as ``IndexError: map::at``.  This
        # situation is normal here because terrain factors are intentionally
        # confirmed sparsely and graph optimization can be idle for several
        # keyframes.  Replaying the pending chain in timestamp order is exactly
        # the online sequence the smoother is defined to consume.
        pending_edges = list(
            enumerate(
                self.edges[self._synced_edge_count :],
                start=self._synced_edge_count,
            )
        )
        pending_priors = list(
            enumerate(
                self.priors[self._synced_prior_count :],
                start=self._synced_prior_count,
            )
        )
        emitted_edges: set[int] = set()
        emitted_priors: set[int] = set()

        def update_once(graph, values, timestamps) -> None:
            has_new_data = bool(values.size() or graph.size())
            self._fixed_lag.update(graph, values, timestamps)
            self.fixed_lag_updates += 1
            if has_new_data:
                # Match the two-step iSAM2 policy used by the retained global
                # diagnostic graph: the first call inserts nonlinear factors,
                # while the second gives the selected variables one immediate
                # relinearization/solve pass without adding information.
                self._fixed_lag.update(
                    self._gtsam.NonlinearFactorGraph(),
                    self._gtsam.Values(),
                    self._fixed_lag_api.FixedLagSmootherKeyTimestampMap(),
                )
                self.fixed_lag_updates += 1
            fixed_values = self._fixed_lag.calculateEstimate()
            self._store_fixed_lag_values(fixed_values)
            self._fixed_lag_active_keys = {
                int(active_key) for active_key in fixed_values.keys()
            }

        for index in range(self._synced_pose_count, self.poses.shape[0]):
            key = self._key(index)
            available = set(self._fixed_lag_active_keys)
            available.add(key)
            graph = self._gtsam.NonlinearFactorGraph()
            values = self._gtsam.Values()
            timestamps = self._fixed_lag_api.FixedLagSmootherKeyTimestampMap()
            touched_keys: set[int] = set()
            if index == 0 and self._synced_pose_count == 0:
                graph.add(self._gauge_factor(self.poses[0]))
                touched_keys.add(key)
            for edge_index, edge in pending_edges:
                if edge_index in emitted_edges:
                    continue
                keys = {self._key(edge.source), self._key(edge.target)}
                if keys.issubset(available):
                    graph.add(self._edge_factor(edge))
                    touched_keys.update(keys)
                    emitted_edges.add(edge_index)
            for prior_index, prior in pending_priors:
                if prior_index in emitted_priors:
                    continue
                prior_key = self._key(prior.target)
                if prior_key in available:
                    graph.add(self._prior_factor(prior))
                    touched_keys.add(prior_key)
                    emitted_priors.add(prior_index)
            if key not in touched_keys:
                # A long-baseline loop can reference a key already marginalized
                # from this window.  Anchor the new window state to the global
                # iSAM2 marginal instead of reviving a deleted variable.
                estimate = self._global_isam.calculateEstimatePose2(key)
                covariance = np.asarray(
                    self._global_isam.marginalCovariance(key),
                    dtype=np.float64,
                )
                information = np.linalg.pinv(
                    0.5 * (covariance + covariance.T), rcond=1.0e-12
                )
                graph.add(
                    self._gtsam.PriorFactorPose2(
                        key,
                        estimate,
                        self._noise_model(information),
                    )
                )
            values.insert(key, self._pose2(self.poses[index]))
            timestamps.insert((key, float(self._timestamps[index])))
            update_once(graph, values, timestamps)

        # Factors added without a new pose (for example a late absolute prior)
        # are still useful when all their variables remain inside the window.
        # Factors touching a marginalized variable stay in global iSAM2 only.
        graph = self._gtsam.NonlinearFactorGraph()
        values = self._gtsam.Values()
        timestamps = self._fixed_lag_api.FixedLagSmootherKeyTimestampMap()
        available = set(self._fixed_lag_active_keys)
        for edge_index, edge in pending_edges:
            if edge_index in emitted_edges:
                continue
            keys = {self._key(edge.source), self._key(edge.target)}
            if keys.issubset(available):
                graph.add(self._edge_factor(edge))
                emitted_edges.add(edge_index)
            else:
                self._mark_global_only_factor("edge", edge)
        for prior_index, prior in pending_priors:
            if prior_index in emitted_priors:
                continue
            prior_key = self._key(prior.target)
            if prior_key in available:
                graph.add(self._prior_factor(prior))
                emitted_priors.add(prior_index)
            else:
                self._mark_global_only_factor("prior", prior)
        update_once(graph, values, timestamps)
        self.fixed_lag_active_variables = len(self._fixed_lag_active_keys)
        self.fixed_lag_marginalized_variables = max(
            0, int(self.poses.shape[0]) - self.fixed_lag_active_variables
        )

    def _rebuild(self) -> None:
        self.rebuilds += 1
        self._new_solvers()
        # Recompute this set from the rebuilt live smoother so repeated
        # external edits cannot double-count old factors.
        self._global_only_factor_ids.clear()
        self.global_only_factor_count = 0
        all_graph = self._gtsam.NonlinearFactorGraph()
        all_values = self._gtsam.Values()
        if self.poses.shape[0]:
            all_graph.add(self._gauge_factor(self.poses[0]))
        for edge in self.edges:
            all_graph.add(self._edge_factor(edge))
        for prior in self.priors:
            all_graph.add(self._prior_factor(prior))
        for index, pose in enumerate(self.poses):
            all_values.insert(self._key(index), self._pose2(pose))
        if self.poses.shape[0]:
            self._global_isam.update(all_graph, all_values)
            self._global_isam.update()
            self.isam_updates += 1

        # Replay chronologically so IncrementalFixedLagSmoother performs real
        # marginalization and creates its boundary marginal factors itself.
        self._fixed_lag_active_keys.clear()
        edges_by_target: dict[int, list[PoseGraphEdge]] = {}
        priors_by_target: dict[int, list[PoseGraphPrior]] = {}
        for edge in self.edges:
            edges_by_target.setdefault(edge.target, []).append(edge)
        for prior in self.priors:
            priors_by_target.setdefault(prior.target, []).append(prior)
        for index, pose in enumerate(self.poses):
            key = self._key(index)
            graph = self._gtsam.NonlinearFactorGraph()
            values = self._gtsam.Values()
            timestamps = self._fixed_lag_api.FixedLagSmootherKeyTimestampMap()
            values.insert(key, self._pose2(pose))
            timestamps.insert((key, float(self._timestamps[index])))
            if index == 0:
                graph.add(self._gauge_factor(pose))
            available = set(self._fixed_lag_active_keys)
            available.add(key)
            for edge in edges_by_target.get(index, ()):
                keys = {self._key(edge.source), self._key(edge.target)}
                if keys.issubset(available):
                    graph.add(self._edge_factor(edge))
                else:
                    self._mark_global_only_factor("edge", edge)
            for prior in priors_by_target.get(index, ()):
                graph.add(self._prior_factor(prior))
            if graph.size() == 0:
                estimate = self._global_isam.calculateEstimatePose2(key)
                covariance = np.asarray(
                    self._global_isam.marginalCovariance(key),
                    dtype=np.float64,
                )
                graph.add(
                    self._gtsam.PriorFactorPose2(
                        key,
                        estimate,
                        self._noise_model(
                            np.linalg.pinv(
                                0.5 * (covariance + covariance.T),
                                rcond=1.0e-12,
                            )
                        ),
                    )
                )
            self._fixed_lag.update(graph, values, timestamps)
            self.fixed_lag_updates += 1
            self._fixed_lag.update(
                self._gtsam.NonlinearFactorGraph(),
                self._gtsam.Values(),
                self._fixed_lag_api.FixedLagSmootherKeyTimestampMap(),
            )
            self.fixed_lag_updates += 1
            estimate_values = self._fixed_lag.calculateEstimate()
            self._store_fixed_lag_values(estimate_values)
            self._fixed_lag_active_keys = set(
                int(active_key) for active_key in estimate_values.keys()
            )
        self.fixed_lag_active_variables = len(self._fixed_lag_active_keys)
        self.fixed_lag_marginalized_variables = max(
            0, int(self.poses.shape[0]) - self.fixed_lag_active_variables
        )

    def optimize(self) -> np.ndarray:
        count = int(self.poses.shape[0])
        if count < 2 or (not self.edges and not self.priors):
            return self.poses.copy()
        self._normalise_timestamp_storage()
        self.initial_cost = self._cost(self.poses)
        self.global_initial_cost = self.initial_cost
        if self._existing_state_was_edited():
            self._rebuild()
        else:
            self._incremental_global_update()
            self._incremental_fixed_lag_update()
        global_values = self._global_isam.calculateEstimate()
        self.global_isam_poses = self._pose_array_from_values(
            global_values, self.poses
        )
        self.global_final_cost = self._cost(self.global_isam_poses)
        # ``_incremental_fixed_lag_update`` and ``_rebuild`` already copied
        # each live-window estimate into public poses. A long-baseline factor
        # that cannot enter the bounded window contributes only one rigid map
        # anchor; it never replaces the window's relative trajectory shape.
        self._bridge_global_anchor_to_fixed_lag_window()
        self.final_cost = self._cost(self.poses)
        self.optimization_runs += 1
        self.optimization_boundaries.append(
            (self.poses.shape[0], len(self.edges), len(self.priors))
        )
        self._mark_synchronised()
        return self.poses.copy()


def _make_planar_pose_graph(config: PoseGraphConfig) -> PlanarPoseGraph:
    backend = str(config.optimizer_backend).strip().lower()
    if backend == "numpy_batch":
        return PlanarPoseGraph(config)
    if backend == "gtsam_fixed_lag_isam2":
        return GTSAMPlanarPoseGraph(config)
    raise ValueError(
        "optimizer_backend must be 'numpy_batch' or "
        "'gtsam_fixed_lag_isam2'"
    )


def acoustic_descriptor(
    points_body: np.ndarray,
    intensities: np.ndarray | None,
    *,
    bins: int = 32,
) -> np.ndarray:
    """Build a compact cross-track bathymetry/reflectivity descriptor."""

    points = np.asarray(points_body, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] == 0:
        return np.zeros(2 * bins, dtype=np.float64)
    angle = np.arctan2(points[:, 1], np.maximum(-points[:, 2], 1.0e-6))
    angle_extent = max(float(np.percentile(np.abs(angle), 98.0)), 0.25)
    indices = np.floor(
        (angle / (2.0 * angle_extent) + 0.5) * bins
    ).astype(np.int64)
    indices = np.clip(indices, 0, bins - 1)
    ranges = np.linalg.norm(points, axis=1)
    range_profile = np.full(bins, np.nan, dtype=np.float64)
    intensity_profile = np.full(bins, np.nan, dtype=np.float64)
    if intensities is None:
        intensity_values = np.zeros(points.shape[0], dtype=np.float64)
    else:
        supplied = np.asarray(intensities, dtype=np.float64).reshape(-1)
        if supplied.size == finite.size:
            intensity_values = supplied[finite]
        elif supplied.size == points.shape[0]:
            intensity_values = supplied
        else:
            intensity_values = np.zeros(points.shape[0], dtype=np.float64)
    for index in range(bins):
        selected = indices == index
        if np.any(selected):
            range_profile[index] = np.median(ranges[selected])
            intensity_profile[index] = np.median(intensity_values[selected])

    def fill(profile: np.ndarray) -> np.ndarray:
        valid = np.flatnonzero(np.isfinite(profile))
        if valid.size == 0:
            return np.zeros_like(profile)
        return np.interp(np.arange(profile.size), valid, profile[valid])

    range_profile = fill(range_profile)
    intensity_profile = fill(intensity_profile)
    # Preserve altitude and relief, while keeping the two modalities on
    # comparable scales.  Absolute range is useful for rejecting a visually
    # similar seabed at a different vehicle height.
    range_feature = np.clip(range_profile / 45.0, 0.0, 2.0)
    intensity_scale = max(float(np.percentile(np.abs(intensity_profile), 90.0)), 0.05)
    intensity_feature = np.clip(intensity_profile / intensity_scale, 0.0, 2.0)
    return np.concatenate((range_feature, intensity_feature))


def descriptor_distance(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first) - np.asarray(second)
    return float(np.sqrt(np.mean(difference * difference)))


def sidescan_descriptor(
    port: np.ndarray,
    starboard: np.ndarray,
    *,
    bins: int = 32,
) -> np.ndarray:
    """Create a gain-invariant descriptor from one side-scan intensity line."""

    def resample(values: np.ndarray) -> np.ndarray:
        line = np.nan_to_num(
            np.asarray(values, dtype=np.float64).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if line.size == 0:
            return np.zeros(bins, dtype=np.float64)
        source = np.linspace(0.0, 1.0, line.size)
        target = np.linspace(0.0, 1.0, bins)
        sampled = np.interp(target, source, np.maximum(line, 0.0))
        sampled = np.log1p(8.0 * sampled)
        scale = max(float(np.percentile(sampled, 90.0)), 1.0e-6)
        return np.clip(sampled / scale, 0.0, 2.0)

    return np.concatenate((resample(port), resample(starboard)))


def _sample_points(points: np.ndarray, maximum: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] <= maximum:
        return points.copy()
    indices = np.linspace(0, points.shape[0] - 1, maximum, dtype=np.int64)
    return points[indices]


def _nearest_neighbours(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Arrays are deliberately capped below 300 points, so this dependency-free
    # brute-force query is quicker than importing a large spatial-tree stack.
    squared = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    indices = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(source.shape[0]), indices])
    return indices, distances


def register_keyframes(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_rotation_target_source: np.ndarray,
    initial_translation_target_source: np.ndarray,
    config: PoseGraphConfig,
) -> RegistrationResult:
    """Verify a candidate loop with trimmed, yaw-constrained 3-D ICP."""

    source = _sample_points(source_points, config.icp_max_points)
    target = _sample_points(target_points, config.icp_max_points)
    rotation = np.asarray(
        initial_rotation_target_source, dtype=np.float64
    ).reshape(3, 3).copy()
    translation = np.asarray(
        initial_translation_target_source, dtype=np.float64
    ).reshape(3).copy()
    rejected = RegistrationResult(
        False, rotation, translation, np.inf, 0, np.inf, np.inf
    )
    if source.shape[0] < config.icp_min_inliers or target.shape[0] < config.icp_min_inliers:
        return rejected

    initial_rotation = rotation.copy()
    initial_translation = translation.copy()

    # A grouped MBES patch is ordered by ping and beam.  On a same-direction
    # revisit, equal-sized groups therefore provide much stronger
    # correspondences than generic nearest-neighbour ICP, which can slide
    # freely along a smooth seabed plane.  Try this geometry-preserving
    # hypothesis first and retain ICP as the fallback for unequal/asynchronous
    # groups.
    if source.shape[0] == target.shape[0]:
        source_mean = np.mean(source, axis=0)
        target_mean = np.mean(target, axis=0)
        cross = (source - source_mean).T @ (target - target_mean)
        try:
            u, _, vt = np.linalg.svd(cross)
            full_rotation = vt.T @ u.T
            if np.linalg.det(full_rotation) < 0.0:
                vt[-1, :] *= -1.0
                full_rotation = vt.T @ u.T
            # Keep the full rigid alignment for geometric verification.  The
            # graph later consumes only x/y/yaw, but allowing roll and pitch
            # here prevents an accumulated inertial tilt error from appearing
            # as metres of false MBES range residual.
            ordered_rotation = full_rotation
            ordered_translation = target_mean - ordered_rotation @ source_mean
            ordered_residual = np.linalg.norm(
                source @ ordered_rotation.T
                + ordered_translation
                - target,
                axis=1,
            )
            ordered_limit = float(np.percentile(ordered_residual, 90.0))
            ordered_inliers = ordered_residual <= ordered_limit
            ordered_rmse = float(
                np.sqrt(np.mean(ordered_residual[ordered_inliers] ** 2))
            )
            ordered_rotation_correction = initial_rotation.T @ ordered_rotation
            ordered_yaw_correction = abs(
                float(rpy_from_rotation(ordered_rotation_correction)[2])
            )
            ordered_translation_correction = float(
                np.linalg.norm(ordered_translation - initial_translation)
            )
            if (
                np.count_nonzero(ordered_inliers) >= config.icp_min_inliers
                and ordered_rmse <= config.ordered_registration_max_rmse_m
                and ordered_translation_correction
                <= config.icp_max_translation_correction_m
                and ordered_yaw_correction
                <= np.deg2rad(config.icp_max_yaw_correction_deg)
            ):
                return RegistrationResult(
                    accepted=True,
                    rotation_target_source=ordered_rotation,
                    translation_target_source=ordered_translation,
                    rmse=ordered_rmse,
                    inliers=int(np.count_nonzero(ordered_inliers)),
                    translation_correction=ordered_translation_correction,
                    yaw_correction_deg=float(np.rad2deg(ordered_yaw_correction)),
                )
        except np.linalg.LinAlgError:
            pass

    for _ in range(config.icp_iterations):
        transformed = source @ rotation.T + translation
        nearest, distances = _nearest_neighbours(transformed, target)
        valid = distances <= config.icp_max_correspondence_m
        if np.count_nonzero(valid) < config.icp_min_inliers:
            return rejected
        valid_distances = distances[valid]
        trim_limit = float(np.percentile(valid_distances, 82.0))
        valid &= distances <= trim_limit
        if np.count_nonzero(valid) < config.icp_min_inliers:
            return rejected
        moving = transformed[valid]
        fixed = target[nearest[valid]]
        moving_mean = np.mean(moving, axis=0)
        fixed_mean = np.mean(fixed, axis=0)
        cross = (moving - moving_mean).T @ (fixed - fixed_mean)
        try:
            u, _, vt = np.linalg.svd(cross)
        except np.linalg.LinAlgError:
            return rejected
        full_delta = vt.T @ u.T
        if np.linalg.det(full_delta) < 0.0:
            vt[-1, :] *= -1.0
            full_delta = vt.T @ u.T
        yaw_delta = float(rpy_from_rotation(full_delta)[2])
        delta_rotation = full_delta
        delta_translation = fixed_mean - delta_rotation @ moving_mean
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
        if (
            abs(yaw_delta) < np.deg2rad(0.005)
            and np.linalg.norm(delta_translation) < 1.0e-3
        ):
            break

    transformed = source @ rotation.T + translation
    nearest, distances = _nearest_neighbours(transformed, target)
    valid = distances <= config.icp_max_correspondence_m
    if np.count_nonzero(valid) < config.icp_min_inliers:
        return rejected
    trim_limit = float(np.percentile(distances[valid], 82.0))
    valid &= distances <= trim_limit
    inliers = int(np.count_nonzero(valid))
    rmse = float(np.sqrt(np.mean(distances[valid] ** 2)))
    correction_rotation = initial_rotation.T @ rotation
    yaw_correction = abs(float(rpy_from_rotation(correction_rotation)[2]))
    translation_correction = float(np.linalg.norm(translation - initial_translation))
    accepted = (
        inliers >= config.icp_min_inliers
        and rmse <= config.icp_max_rmse_m
        and translation_correction <= config.icp_max_translation_correction_m
        and yaw_correction <= np.deg2rad(config.icp_max_yaw_correction_deg)
    )
    return RegistrationResult(
        accepted=accepted,
        rotation_target_source=rotation,
        translation_target_source=translation,
        rmse=rmse,
        inliers=inliers,
        translation_correction=translation_correction,
        yaw_correction_deg=float(np.rad2deg(yaw_correction)),
    )


class AcousticPoseGraph:
    """Manage acoustic keyframes, loop verification, and graph corrections."""

    def __init__(self, config: PoseGraphConfig | None = None) -> None:
        self.config = config or PoseGraphConfig()
        self.graph = _make_planar_pose_graph(self.config)
        self.keyframes: list[AcousticKeyframe] = []
        self.loop_edges: list[PoseGraphEdge] = []
        self._pending_loops: list[PoseGraphEdge] = []
        self._committed_loop_pairs: set[tuple[int, int]] = set()
        self._intensity_factor_count = 0
        self._rejected_intensity_factor_count = 0
        self._throttled_intensity_factor_count = 0
        self._last_intensity_factor_timestamp = -np.inf
        self._intensity_factor_rejection_reasons: Counter[str] = Counter()
        self._last_intensity_factor_rejection_reason: str | None = None
        self._last_intensity_prefit_performed = False
        self._last_intensity_prefit_chi2 = 0.0
        self._last_intensity_prefit_dof = 0
        self._terrain_prior_count = 0
        self._rejected_terrain_prior_count = 0
        self._terrain_prior_rejection_reasons: Counter[str] = Counter()
        self._last_terrain_prior_rejection_reason: str | None = None
        self._last_terrain_prefit_performed = False
        self._last_terrain_prefit_bypassed_for_relocalization = False
        self._last_terrain_prefit_chi2 = 0.0
        self._last_terrain_prefit_dof = 0
        self._terrain_prior_observations: set[int] = set()
        self._terrain_relocalization_seed_count = 0
        self._last_terrain_relocalization_translation_m = 0.0
        self._last_terrain_relocalization_yaw_deg = 0.0
        self._last_terrain_relocalization_anchor_index: int | None = None
        self._geometry_keyframe_count = 0

    def reset(self) -> None:
        self.graph = _make_planar_pose_graph(self.config)
        self.keyframes.clear()
        self.loop_edges.clear()
        self._pending_loops.clear()
        self._committed_loop_pairs.clear()
        self._intensity_factor_count = 0
        self._rejected_intensity_factor_count = 0
        self._throttled_intensity_factor_count = 0
        self._last_intensity_factor_timestamp = -np.inf
        self._intensity_factor_rejection_reasons.clear()
        self._last_intensity_factor_rejection_reason = None
        self._last_intensity_prefit_performed = False
        self._last_intensity_prefit_chi2 = 0.0
        self._last_intensity_prefit_dof = 0
        self._terrain_prior_count = 0
        self._rejected_terrain_prior_count = 0
        self._terrain_prior_rejection_reasons.clear()
        self._last_terrain_prior_rejection_reason = None
        self._last_terrain_prefit_performed = False
        self._last_terrain_prefit_bypassed_for_relocalization = False
        self._last_terrain_prefit_chi2 = 0.0
        self._last_terrain_prefit_dof = 0
        self._terrain_prior_observations.clear()
        self._terrain_relocalization_seed_count = 0
        self._last_terrain_relocalization_translation_m = 0.0
        self._last_terrain_relocalization_yaw_deg = 0.0
        self._last_terrain_relocalization_anchor_index = None
        self._geometry_keyframe_count = 0

    def factor_stream_payload(self) -> dict[str, object]:
        """Return solver factors plus the untouched local keyframe states.

        Solver pose initializations may already contain a ``map->odom`` graph
        correction.  They therefore cannot be used to audit the front-end
        trajectory.  Recording each keyframe's raw ESIKF posterior separately
        keeps backend replay and local-odometry evaluation unambiguous.
        """

        if len(self.keyframes) != self.graph.poses.shape[0]:
            raise RuntimeError("keyframes and graph poses diverged")
        payload = self.graph.factor_stream_payload()
        payload["raw_local_keyframe_pose2"] = [
            frame.raw_pose2.tolist() for frame in self.keyframes
        ]
        payload["raw_local_keyframe_positions_xyz"] = [
            frame.raw_position.tolist() for frame in self.keyframes
        ]
        payload["raw_local_keyframe_rotations_world_body"] = [
            frame.raw_rotation.tolist() for frame in self.keyframes
        ]
        payload["keyframe_observation_ids"] = [
            int(frame.observation_id) for frame in self.keyframes
        ]
        payload["keyframe_loop_eligible"] = [
            bool(frame.loop_eligible) for frame in self.keyframes
        ]
        return payload

    def _reject_intensity_factor(self, reason: str) -> bool:
        """Record a rejected attempted factor and preserve the bool API."""

        self._rejected_intensity_factor_count += 1
        self._intensity_factor_rejection_reasons[str(reason)] += 1
        self._last_intensity_factor_rejection_reason = str(reason)
        return False

    def _reject_terrain_prior(self, reason: str) -> bool:
        """Record a rejected map-match without changing graph topology."""

        self._rejected_terrain_prior_count += 1
        self._terrain_prior_rejection_reasons[str(reason)] += 1
        self._last_terrain_prior_rejection_reason = str(reason)
        return False

    @staticmethod
    def _information(xy_std: float, yaw_std_deg: float) -> np.ndarray:
        deviations = np.array(
            [xy_std, xy_std, np.deg2rad(yaw_std_deg)], dtype=np.float64
        )
        return np.diag(1.0 / np.maximum(deviations * deviations, 1.0e-12))

    def _needs_keyframe(self, timestamp: float, position: np.ndarray) -> bool:
        previous = next(
            (frame for frame in reversed(self.keyframes) if frame.loop_eligible),
            None,
        )
        if previous is None:
            return True
        elapsed = float(timestamp - previous.timestamp)
        distance = float(np.linalg.norm(position[:2] - previous.raw_position[:2]))
        return (
            distance >= self.config.keyframe_distance_m
            or elapsed >= self.config.keyframe_max_interval_s
        )

    def add_if_needed(
        self,
        timestamp: float,
        rotation_world_body: np.ndarray,
        position_world: np.ndarray,
        points_body: np.ndarray,
        intensities: np.ndarray | None = None,
        appearance_descriptor: np.ndarray | None = None,
        *,
        observation_id: int | None = None,
        force: bool = False,
        odometry_pose2: np.ndarray | None = None,
        detect_loops: bool = True,
    ) -> bool:
        if not self.config.enabled:
            return False
        position = np.asarray(position_world, dtype=np.float64).reshape(3)
        if not force and not self._needs_keyframe(timestamp, position):
            return False
        points = _sample_points(points_body, self.config.icp_max_points)
        # Ordinary geometry keyframes need enough returns for loop
        # verification.  A forced image-update node, however, is a pose-only
        # graph node: its direct-intensity factor may be observable with fewer
        # samples and loop detection is explicitly disabled by the caller.
        if points.shape[0] < self.config.icp_min_inliers and not force:
            return False
        values = None
        if intensities is not None:
            supplied = np.asarray(intensities, dtype=np.float64).reshape(-1)
            if supplied.size == np.asarray(points_body).reshape(-1, 3).shape[0]:
                if supplied.size > self.config.icp_max_points:
                    indices = np.linspace(
                        0, supplied.size - 1, self.config.icp_max_points, dtype=np.int64
                    )
                    values = supplied[indices]
                else:
                    values = supplied.copy()
        geometry_descriptor = acoustic_descriptor(points, values)
        appearance_size = 2 * self.config.sidescan_descriptor_bins
        if appearance_descriptor is None:
            appearance = np.zeros(appearance_size, dtype=np.float64)
        else:
            appearance = np.asarray(
                appearance_descriptor, dtype=np.float64
            ).reshape(-1)
            if appearance.size != appearance_size:
                raise ValueError(
                    f"side-scan descriptor must contain {appearance_size} values"
                )
        frame = AcousticKeyframe(
            index=len(self.keyframes),
            timestamp=float(timestamp),
            raw_position=position.copy(),
            raw_rotation=np.asarray(rotation_world_body, dtype=np.float64).reshape(3, 3).copy(),
            points_body=points,
            intensities=values,
            descriptor=np.concatenate((geometry_descriptor, appearance)),
            observation_id=(
                len(self.keyframes)
                if observation_id is None
                else int(observation_id)
            ),
            loop_eligible=bool(detect_loops),
            geometry_sequence_index=(
                self._geometry_keyframe_count if detect_loops else -1
            ),
        )
        raw_pose = frame.raw_pose2
        odometry_pose = (
            raw_pose
            if odometry_pose2 is None
            else np.asarray(odometry_pose2, dtype=np.float64).reshape(3)
        )
        graph_pose = raw_pose
        odometry_measurement = None
        if self.keyframes:
            previous = self.keyframes[-1]
            odometry_measurement = _relative_pose2(
                previous.raw_pose2, odometry_pose
            )
            # Once a loop has optimized the graph, initialise every subsequent
            # node by propagating the last optimized pose. Adding it at the raw
            # location would reset the newest correction to zero until another
            # loop happens, making the reported global pose jump back to local
            # odometry between loop events.
            if self.graph.optimization_runs > 0:
                graph_prior = _compose_pose2(
                    self.graph.poses[-1], odometry_measurement
                )
                graph_pose = _compose_pose2(
                    graph_prior,
                    _relative_pose2(odometry_pose, raw_pose),
                )
        node = self.graph.add_pose(graph_pose, timestamp=float(timestamp))
        if node != frame.index:
            raise RuntimeError("keyframe and graph indices diverged")
        if self.keyframes:
            previous = self.keyframes[-1]
            elapsed = max(
                float(frame.timestamp - previous.timestamp), 1.0e-6
            )
            configured_reference = self.config.odometry_reference_interval_s
            reference_interval = max(
                float(
                    self.config.keyframe_max_interval_s
                    if configured_reference is None
                    else configured_reference
                ),
                1.0e-6,
            )
            random_walk_scale = np.sqrt(elapsed / reference_interval)
            self.graph.add_edge(
                previous.index,
                frame.index,
                odometry_measurement,
                self._information(
                    self.config.odometry_xy_std_m * random_walk_scale,
                    self.config.odometry_yaw_std_deg * random_walk_scale,
                ),
                kind="odometry",
            )
        self.keyframes.append(frame)
        if detect_loops:
            self._geometry_keyframe_count += 1
            loop = self._find_and_verify_loop(frame)
            confirmed = self._confirm_loop_sequence(frame.index, loop)
            if confirmed:
                globally_relocalized = self._seed_global_relocalization(confirmed)
                for edge in confirmed:
                    committed_edge = edge
                    if globally_relocalized:
                        # This is no longer an unconfirmed, isolated loop
                        # hypothesis. Keeping it outside the single-loop robust
                        # kernel lets the consistent sequence overcome the large
                        # accumulated odometry residual it was designed to fix.
                        committed_edge = PoseGraphEdge(
                            source=edge.source,
                            target=edge.target,
                            measurement=edge.measurement,
                            information=edge.information,
                            kind="relocalization",
                            score=edge.score,
                        )
                    self.graph.edges.append(committed_edge)
                    self.loop_edges.append(committed_edge)
                self.graph.optimize()
        return True

    @property
    def latest_keyframe_index(self) -> int | None:
        return None if not self.keyframes else self.keyframes[-1].index

    def _seed_terrain_relocalization(
        self, target_index: int, matched_pose2: np.ndarray
    ) -> None:
        """Re-seed a confirmed absolute recovery without bending old map data.

        Before the first accepted absolute terrain prior, every graph pose is
        defined only up to one common SE(2) gauge.  Applying the verified
        target correction rigidly to the complete graph therefore preserves
        every odometry/loop residual while moving node zero into the surveyed
        map frame.

        After an earlier terrain prior exists, the historical graph already
        has an absolute anchor.  In that case the correction is introduced
        smoothly from the latest earlier prior to the current target, using
        accumulated graph distance as the interpolation coordinate.  Poses
        after the target receive the full rigid transform.  The subsequent
        non-robust confirmed prior and ordinary optimizer then distribute the
        accumulated drift over only the unanchored outage segment.
        """

        target = int(target_index)
        if target < 0 or target >= self.graph.poses.shape[0]:
            raise ValueError("terrain relocalization target is invalid")
        matched = np.asarray(matched_pose2, dtype=np.float64).reshape(3)
        predicted = self.graph.poses[target].copy()
        yaw_delta = float(wrap_angle(matched[2] - predicted[2]))
        endpoint_translation = matched[:2] - predicted[:2]

        earlier_prior_targets = [
            int(prior.target)
            for prior in self.graph.priors
            if prior.kind == "terrain_prior" and prior.target < target
        ]
        anchor = max(earlier_prior_targets, default=None)
        pose_count = self.graph.poses.shape[0]
        if anchor is None:
            alphas = np.ones(pose_count, dtype=np.float64)
        else:
            alphas = np.zeros(pose_count, dtype=np.float64)
            segment = self.graph.poses[anchor : target + 1, :2].copy()
            if segment.shape[0] > 1:
                cumulative = np.r_[
                    0.0,
                    np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1)),
                ]
                total = float(cumulative[-1])
                if total > 1.0e-9:
                    alphas[anchor : target + 1] = cumulative / total
                else:
                    alphas[anchor : target + 1] = np.linspace(
                        0.0, 1.0, target - anchor + 1
                    )
            alphas[target:] = 1.0

        original = self.graph.poses.copy()
        for index, alpha in enumerate(alphas):
            if alpha <= 0.0:
                continue
            angle = alpha * yaw_delta
            c = float(np.cos(angle))
            s = float(np.sin(angle))
            rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
            self.graph.poses[index, :2] = (
                rotation @ (original[index, :2] - predicted[:2])
                + predicted[:2]
                + alpha * endpoint_translation
            )
            self.graph.poses[index, 2] = wrap_angle(
                original[index, 2] + angle
            )

        # Make the endpoint exact despite floating-point interpolation.  This
        # is only an optimizer seed; the confirmed factor remains the source
        # of absolute information in the graph objective.
        self.graph.poses[target] = matched
        self._terrain_relocalization_seed_count += 1
        self._last_terrain_relocalization_translation_m = float(
            np.linalg.norm(endpoint_translation)
        )
        self._last_terrain_relocalization_yaw_deg = float(
            np.rad2deg(yaw_delta)
        )
        self._last_terrain_relocalization_anchor_index = anchor

    def _terrain_prior_prefit_is_consistent(
        self,
        predicted_pose2: np.ndarray,
        matched_pose2: np.ndarray,
        information_world: np.ndarray,
    ) -> bool:
        """Apply the catastrophic-jump gate to an ordinary unary factor."""

        prefit_error = _prior_error(predicted_pose2, matched_pose2)
        try:
            yaw_scale = float(self.config.terrain_prior_prefit_yaw_scale_m)
            if not np.isfinite(yaw_scale) or yaw_scale <= 0.0:
                raise ValueError
            # q = Q u, with u_yaw expressed in metre-like units.  This is the
            # same congruence used by the matcher for observability analysis.
            q_from_u = np.diag([1.0, 1.0, 1.0 / yaw_scale])
            scaled_information = (
                q_from_u.T @ information_world @ q_from_u
            )
            scaled_information = 0.5 * (
                scaled_information + scaled_information.T
            )
            eigenvalues, eigenvectors = np.linalg.eigh(scaled_information)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            rank_tolerance = max(1.0, float(eigenvalues[-1])) * 1.0e-9
            observable = eigenvalues > rank_tolerance
            observable_rank = int(np.count_nonzero(observable))
            scaled_error = prefit_error.copy()
            scaled_error[2] *= yaw_scale
            observable_error = eigenvectors[:, observable].T @ scaled_error
            prefit_chi2 = float(
                np.sum(eigenvalues[observable] * observable_error**2)
            )
            threshold = float(
                self.config.terrain_prior_prefit_chi2_rank2
                if observable_rank == 2
                else self.config.terrain_prior_prefit_chi2_rank3
            )
        except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
            observable_rank = 0
            prefit_chi2 = float("inf")
            threshold = float("nan")
        self._last_terrain_prefit_performed = True
        self._last_terrain_prefit_chi2 = prefit_chi2
        self._last_terrain_prefit_dof = observable_rank
        return bool(
            observable_rank in (2, 3)
            and np.isfinite(prefit_chi2)
            and np.isfinite(threshold)
            and threshold > 0.0
            and prefit_chi2 <= threshold
        )

    def add_terrain_prior(
        self,
        *,
        timestamp: float,
        observation_id: int,
        prior_pose2: np.ndarray,
        raw_rotation_world_body: np.ndarray,
        raw_position_world: np.ndarray,
        matched_pose2: np.ndarray,
        information_world: np.ndarray,
        points_body: np.ndarray | None = None,
        optimize_interval: int | None = None,
        relocalization_seed: bool = False,
        relocalization_seed_pose2: np.ndarray | None = None,
        relocalization_prior_robust: bool = False,
    ) -> bool:
        """Commit one independently matched world-frame terrain pose.

        ``prior_pose2`` is the local odometry state immediately *before* the
        terrain match and is used only by a newly-created incoming odometry
        edge.  ``matched_pose2`` is the independent map-match optimum and is
        represented only by a unary absolute factor.  Keeping those two
        inputs separate avoids applying the same terrain residual once in
        odometry and again as an absolute constraint.

        If the acoustic observation already created a keyframe at the same
        timestamp, that physical pose node is reused. Otherwise a pose-only
        node is appended. All validation precedes this append, so rejected
        inputs do not leave a node or edge behind. One terrain prior is
        accepted per observation id. ``relocalization_seed`` is reserved for
        a match that an external confirmation policy has already verified; it
        re-seeds the outage segment and, by default, disables the ordinary
        single-match robust kernel for this factor.  A persistent
        multi-hypothesis policy with an explicit null state may set
        ``relocalization_prior_robust`` after it has selected and re-seeded the
        verified basin; the factor then starts at zero residual but retains
        bounded influence if later evidence contradicts it.
        ``relocalization_seed_pose2`` may carry
        a separately estimated gauge pose (for example, yaw derived from two
        absolute-position fixes) while ``matched_pose2`` remains the actual
        unary measurement. Omitting it seeds directly from ``matched_pose2``.
        """

        if not self.config.enabled or not self.config.terrain_prior_enabled:
            return False
        self._last_terrain_prior_rejection_reason = None
        self._last_terrain_prefit_performed = False
        self._last_terrain_prefit_bypassed_for_relocalization = False
        self._last_terrain_prefit_chi2 = 0.0
        self._last_terrain_prefit_dof = 0

        try:
            match_timestamp = float(timestamp)
            match_observation_id = int(observation_id)
            if not isinstance(relocalization_seed, (bool, np.bool_)):
                raise TypeError
            use_relocalization_seed = bool(relocalization_seed)
            if not isinstance(
                relocalization_prior_robust, (bool, np.bool_)
            ):
                raise TypeError
            robust_relocalization_prior = bool(
                relocalization_prior_robust
            )
            prior = np.asarray(prior_pose2, dtype=np.float64).reshape(3).copy()
            raw_rotation = np.asarray(
                raw_rotation_world_body, dtype=np.float64
            ).reshape(3, 3).copy()
            raw_position = np.asarray(
                raw_position_world, dtype=np.float64
            ).reshape(3).copy()
            matched = np.asarray(
                matched_pose2, dtype=np.float64
            ).reshape(3).copy()
            seed_matched = (
                matched.copy()
                if relocalization_seed_pose2 is None
                else np.asarray(
                    relocalization_seed_pose2, dtype=np.float64
                ).reshape(3).copy()
            )
            points = (
                np.empty((0, 3), dtype=np.float64)
                if points_body is None
                else np.asarray(points_body, dtype=np.float64).reshape(-1, 3)
            )
        except (TypeError, ValueError, OverflowError):
            return self._reject_terrain_prior("invalid_shape_or_type")
        if (
            not np.isfinite(match_timestamp)
            or not np.isfinite(prior).all()
            or not np.isfinite(raw_rotation).all()
            or not np.isfinite(raw_position).all()
            or not np.isfinite(matched).all()
            or not np.isfinite(seed_matched).all()
            or not np.isfinite(points).all()
        ):
            return self._reject_terrain_prior("nonfinite_input")
        prior[2] = wrap_angle(prior[2])
        matched[2] = wrap_angle(matched[2])
        seed_matched[2] = wrap_angle(seed_matched[2])
        if relocalization_seed_pose2 is not None and not use_relocalization_seed:
            return self._reject_terrain_prior("unused_relocalization_seed_pose")
        if robust_relocalization_prior and not use_relocalization_seed:
            return self._reject_terrain_prior(
                "robust_relocalization_prior_without_seed"
            )

        try:
            factor_information = _validated_psd_information(
                information_world
            )
        except ValueError:
            return self._reject_terrain_prior("invalid_information")
        information_scale = max(
            1.0, float(np.max(np.abs(factor_information)))
        )
        if not np.any(
            np.linalg.eigvalsh(factor_information)
            > 1.0e-12 * information_scale
        ):
            return self._reject_terrain_prior("zero_information")

        try:
            if optimize_interval is None:
                interval = int(
                    self.config.terrain_prior_optimize_interval
                )
            else:
                interval = int(optimize_interval)
        except (TypeError, ValueError, OverflowError):
            return self._reject_terrain_prior("invalid_optimize_interval")
        if interval < 1:
            return self._reject_terrain_prior("invalid_optimize_interval")

        if match_observation_id in self._terrain_prior_observations:
            return self._reject_terrain_prior("duplicate_observation")

        same_observation = [
            frame
            for frame in self.keyframes
            if frame.observation_id == match_observation_id
        ]
        exact_nodes = [
            frame
            for frame in same_observation
            if abs(frame.timestamp - match_timestamp) <= 1.0e-9
        ]
        if same_observation and not exact_nodes:
            return self._reject_terrain_prior(
                "observation_timestamp_mismatch"
            )
        if len(exact_nodes) > 1:
            return self._reject_terrain_prior("ambiguous_observation")

        target = exact_nodes[0] if exact_nodes else None
        if (
            target is None
            and self.keyframes
            and match_timestamp <= self.keyframes[-1].timestamp + 1.0e-9
        ):
            return self._reject_terrain_prior("nonmonotonic_timestamp")

        # Predict the target from graph state without consulting the current
        # terrain residual.  For a prospective node, propagate the incoming
        # local-odometry increment from the latest graph pose, then retain any
        # independent prior-to-raw frontend correction exactly as
        # ``add_if_needed`` will when the node is eventually committed.  This
        # is intentionally evaluated before node creation so a failed match is
        # topology-transactional.
        if target is not None:
            predicted_target = self.graph.poses[target.index].copy()
        else:
            raw_pose = np.array(
                [
                    raw_position[0],
                    raw_position[1],
                    rpy_from_rotation(raw_rotation)[2],
                ],
                dtype=np.float64,
            )
            if self.keyframes:
                incoming_odometry = _relative_pose2(
                    self.keyframes[-1].raw_pose2, prior
                )
                predicted_target = _compose_pose2(
                    self.graph.poses[-1], incoming_odometry
                )
                predicted_target = _compose_pose2(
                    predicted_target, _relative_pose2(prior, raw_pose)
                )
            else:
                predicted_target = raw_pose

        # A normal single match is compared with the current graph estimate.
        # A confirmed relocalization exists specifically because that graph
        # estimate has left the correct basin; comparing the two here would
        # deterministically reject every successful long-range recovery.  The
        # caller may request this bypass only after its external independent-
        # strip confirmation policy has accepted the match.
        if use_relocalization_seed:
            self._last_terrain_prefit_bypassed_for_relocalization = True
        elif not self._terrain_prior_prefit_is_consistent(
            predicted_target,
            matched,
            factor_information,
        ):
            return self._reject_terrain_prior("prefit_inconsistent")

        if target is None:
            # ``add_if_needed`` has no externally observable side effects
            # beyond these containers when loop detection is disabled. Keep
            # snapshots as a final transactional guard against an unexpected
            # node-construction failure.
            pose_count = self.graph.poses.shape[0]
            edge_count = len(self.graph.edges)
            frame_count = len(self.keyframes)
            try:
                created = self.add_if_needed(
                    match_timestamp,
                    raw_rotation,
                    raw_position,
                    points,
                    observation_id=match_observation_id,
                    force=True,
                    odometry_pose2=prior,
                    detect_loops=False,
                )
            except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError):
                self.graph.truncate_poses(pose_count)
                del self.graph.edges[edge_count:]
                del self.keyframes[frame_count:]
                return self._reject_terrain_prior("node_creation_failed")
            if not created:
                self.graph.truncate_poses(pose_count)
                del self.graph.edges[edge_count:]
                del self.keyframes[frame_count:]
                return self._reject_terrain_prior("node_creation_failed")
            target = self.keyframes[-1]

        # Inputs were fully validated before any pose-only node could be
        # appended; adding this prevalidated factor therefore completes the
        # transaction without changing the raw local-odometry baseline.
        if use_relocalization_seed:
            self._seed_terrain_relocalization(target.index, seed_matched)

        self.graph.add_prior(
            target.index,
            matched,
            factor_information,
            kind="terrain_prior",
            robust=(
                not use_relocalization_seed
                or robust_relocalization_prior
            ),
            robust_kernel=(
                self.config.relocalization_prior_robust_kernel
                if use_relocalization_seed
                and robust_relocalization_prior
                else None
            ),
            robust_delta=(
                self.config.robust_relocalization_prior_delta
                if use_relocalization_seed
                and robust_relocalization_prior
                else None
            ),
        )
        self._terrain_prior_count += 1
        self._terrain_prior_observations.add(match_observation_id)
        if self._terrain_prior_count % interval == 0:
            self.graph.optimize()
        return True

    def intensity_seed_candidates(
        self,
        *,
        rotation_world_body: np.ndarray,
        position_world: np.ndarray,
        anchor_ids: tuple[int, ...] | list[int],
        maximum_observation_id: int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Map the current global graph estimate into old anchor coordinates.

        Direct intensity interpolation is deliberately local and therefore
        cannot bootstrap itself after metres of local-odometry drift.  Once
        the pose graph places the vehicle near an old submap, this method
        produces conservative same-heading seeds expressed in each candidate
        anchor's raw map coordinates.  The image residual must still pass all
        overlap, correlation, rank, and correction gates.
        """

        if not self.keyframes:
            return []
        corrected_rotation, corrected_position = self.corrected_pose(
            rotation_world_body, position_world
        )
        current_global = np.array(
            [
                corrected_position[0],
                corrected_position[1],
                rpy_from_rotation(corrected_rotation)[2],
            ],
            dtype=np.float64,
        )
        maximum_distance = max(
            0.0, float(self.config.intensity_seed_search_radius_m)
        )
        maximum_yaw = np.deg2rad(
            max(
                0.0,
                float(
                    self.config.intensity_seed_max_yaw_difference_deg
                ),
            )
        )
        ranked: list[tuple[float, float, int, np.ndarray]] = []
        for anchor_id in anchor_ids:
            index = int(anchor_id)
            if not 0 <= index < len(self.keyframes):
                continue
            anchor = self.keyframes[index]
            if (
                maximum_observation_id is not None
                and anchor.observation_id > int(maximum_observation_id)
            ):
                continue
            anchor_global = self.graph.poses[index]
            distance = float(
                np.linalg.norm(current_global[:2] - anchor_global[:2])
            )
            yaw_difference = abs(
                float(wrap_angle(current_global[2] - anchor_global[2]))
            )
            if distance > maximum_distance or yaw_difference > maximum_yaw:
                continue
            relative = _relative_pose2(anchor_global, current_global)
            raw_seed = _compose_pose2(anchor.raw_pose2, relative)
            ranked.append((distance, yaw_difference, index, raw_seed))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        limit = max(1, int(self.config.intensity_seed_max_candidates))
        return [
            (index, pose.copy())
            for _, _, index, pose in ranked[:limit]
        ]

    def _intensity_information_in_source(
        self, information_world: np.ndarray, source_yaw: float
    ) -> np.ndarray | None:
        """Regularize and rotate a measurement-only image Hessian."""

        information = np.asarray(
            information_world, dtype=np.float64
        ).reshape(3, 3)
        information = 0.5 * (information + information.T)
        if not np.isfinite(information).all():
            return None
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(information)
        except np.linalg.LinAlgError:
            return None
        negative_tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * 1.0e-9
        if np.min(eigenvalues) < -negative_tolerance:
            return None
        eigenvalues = np.maximum(eigenvalues, 0.0)
        if np.count_nonzero(eigenvalues > 1.0e-8) < 2:
            return None
        information = (eigenvectors * eigenvalues) @ eigenvectors.T

        maximum_deviation = np.array(
            [
                self.config.intensity_factor_max_xy_std_m,
                self.config.intensity_factor_max_xy_std_m,
                np.deg2rad(self.config.intensity_factor_max_yaw_std_deg),
            ],
            dtype=np.float64,
        )
        c, s = np.cos(source_yaw), np.sin(source_yaw)
        source_to_world = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        information_source = (
            source_to_world.T @ information @ source_to_world
        )
        information_source = 0.5 * (
            information_source + information_source.T
        )

        # Reject image normal equations that are not useful in at least two
        # planar directions at the configured maximum uncertainty.  Do not
        # add a weak diagonal before inversion: that would manufacture finite
        # information in a true photometric nullspace.
        maximum_scale = np.diag(maximum_deviation)
        weak_normalized = (
            maximum_scale @ information_source @ maximum_scale
        )
        weak_eigenvalues = np.linalg.eigvalsh(
            0.5 * (weak_normalized + weak_normalized.T)
        )
        if np.count_nonzero(weak_eigenvalues >= 1.0) < 2:
            return None

        # Cap only excessive information, in dimensionless coordinates, so
        # no image direction is allowed to claim a standard deviation below
        # the conservative sensor/model floor. Exact zero modes stay zero.
        minimum_deviation = np.array(
            [
                self.config.intensity_factor_min_xy_std_m,
                self.config.intensity_factor_min_xy_std_m,
                np.deg2rad(self.config.intensity_factor_min_yaw_std_deg),
            ],
            dtype=np.float64,
        )
        minimum_scale = np.diag(minimum_deviation)
        normalized = minimum_scale @ information_source @ minimum_scale
        normalized = 0.5 * (normalized + normalized.T)
        capped_eigenvalues, capped_eigenvectors = np.linalg.eigh(normalized)
        capped_eigenvalues = np.clip(capped_eigenvalues, 0.0, 1.0)
        capped = (
            capped_eigenvectors * capped_eigenvalues
        ) @ capped_eigenvectors.T
        inverse_minimum_scale = np.diag(1.0 / minimum_deviation)
        regularized = (
            inverse_minimum_scale @ capped @ inverse_minimum_scale
        )
        return 0.5 * (regularized + regularized.T)

    def add_intensity_observation(
        self,
        *,
        timestamp: float,
        observation_id: int,
        anchor_keyframe_index: int,
        prior_pose2: np.ndarray,
        posterior_rotation_world_body: np.ndarray,
        posterior_position_world: np.ndarray,
        photometric_pose2: np.ndarray,
        information_world: np.ndarray,
        points_body: np.ndarray,
        intensities: np.ndarray | None,
        appearance_descriptor: np.ndarray | None,
        score: float,
    ) -> bool:
        """Add separate odometry and anchored direct-intensity edges.

        ``prior_pose2`` is the state immediately before the image update and
        supplies the incoming odometry edge. ``photometric_pose2`` is the
        verified local-quadratic pose from the measurement-only projected
        normal equation, not the ESIKF posterior. Consequently no part of the same
        image residual is counted in both edges. The self-built reference
        submap can still be correlated with older odometry, which is why the
        image information is conservatively floored and compatibility-gated.
        """

        if not self.config.enabled or not self.config.intensity_factor_enabled:
            return False
        self._last_intensity_factor_rejection_reason = None
        self._last_intensity_prefit_performed = False
        self._last_intensity_prefit_chi2 = 0.0
        self._last_intensity_prefit_dof = 0
        minimum_interval = max(
            0.0, float(self.config.intensity_factor_min_interval_s)
        )
        if (
            minimum_interval > 0.0
            and float(timestamp) - self._last_intensity_factor_timestamp
            < minimum_interval
        ):
            self._throttled_intensity_factor_count += 1
            return False
        if not (0 <= int(anchor_keyframe_index) < len(self.keyframes)):
            return self._reject_intensity_factor("invalid_anchor")
        prior = np.asarray(prior_pose2, dtype=np.float64).reshape(3)
        photo_pose = np.asarray(
            photometric_pose2, dtype=np.float64
        ).reshape(3)
        if not np.isfinite(prior).all() or not np.isfinite(photo_pose).all():
            return self._reject_intensity_factor("nonfinite_pose")

        posterior_rotation = np.asarray(
            posterior_rotation_world_body, dtype=np.float64
        ).reshape(3, 3)
        posterior_position = np.asarray(
            posterior_position_world, dtype=np.float64
        ).reshape(3)
        posterior_pose = np.array(
            [
                posterior_position[0],
                posterior_position[1],
                rpy_from_rotation(posterior_rotation)[2],
            ],
            dtype=np.float64,
        )
        if (
            not np.isfinite(posterior_rotation).all()
            or not np.isfinite(posterior_pose).all()
        ):
            return self._reject_intensity_factor("nonfinite_pose")
        source = self.keyframes[int(anchor_keyframe_index)]
        factor_information = self._intensity_information_in_source(
            information_world, source.raw_pose2[2]
        )
        if factor_information is None:
            return self._reject_intensity_factor("invalid_or_weak_information")

        # Geometry and intensity acquired at the same timestamp represent one
        # physical pose variable. Reuse that node when it exists. Otherwise
        # insert a pose-only chain node whose incoming odometry terminates at
        # the pre-image prior; it is excluded from loop cadence/candidates.
        target = self.keyframes[-1]
        reuse_latest = bool(
            target.observation_id == int(observation_id)
            and abs(target.timestamp - float(timestamp)) <= 1.0e-9
        )
        prospective_target_index = (
            target.index if reuse_latest else len(self.keyframes)
        )
        if source.index >= prospective_target_index:
            return self._reject_intensity_factor("invalid_topology")

        # Compare the image-only measurement with the target predicted from
        # the already-existing graph and the pre-image odometry, which is
        # independent of this current image residual.
        # Crucially, this happens before creating a pose-only node or mutating
        # an existing raw baseline, so a rejected factor is transactional.
        measurement = _relative_pose2(source.raw_pose2, photo_pose)
        if reuse_latest:
            predicted_target = self.graph.poses[target.index].copy()
        else:
            incoming_odometry = _relative_pose2(target.raw_pose2, prior)
            predicted_target = _compose_pose2(
                self.graph.poses[target.index], incoming_odometry
            )
        prefit_error, _, _ = _edge_error_and_jacobians(
            self.graph.poses[source.index], predicted_target, measurement
        )
        eigenvalues = np.linalg.eigvalsh(factor_information)
        rank_tolerance = max(1.0, float(np.max(eigenvalues))) * 1.0e-9
        observable_rank = int(np.count_nonzero(eigenvalues > rank_tolerance))
        prefit_chi2 = float(
            prefit_error @ factor_information @ prefit_error
        )
        self._last_intensity_prefit_chi2 = prefit_chi2
        self._last_intensity_prefit_dof = observable_rank
        self._last_intensity_prefit_performed = True
        threshold = (
            float(self.config.intensity_factor_prefit_chi2_rank2)
            if observable_rank == 2
            else float(self.config.intensity_factor_prefit_chi2_rank3)
        )
        if (
            observable_rank not in (2, 3)
            or not np.isfinite(prefit_chi2)
            or not np.isfinite(threshold)
            or threshold <= 0.0
            or prefit_chi2 > threshold
        ):
            return self._reject_intensity_factor("prefit_inconsistent")

        if not reuse_latest:
            created = self.add_if_needed(
                timestamp,
                posterior_rotation,
                posterior_position,
                np.empty((0, 3), dtype=np.float64),
                None,
                None,
                observation_id=observation_id,
                force=True,
                odometry_pose2=prior,
                detect_loops=False,
            )
            if not created:
                return self._reject_intensity_factor("node_creation_failed")
            target = self.keyframes[-1]

        if source.index >= target.index:
            return self._reject_intensity_factor("invalid_topology")
        self.graph.add_edge(
            source.index,
            target.index,
            measurement,
            factor_information,
            kind="intensity",
            score=float(score),
        )
        # Raw trajectory corrections are defined relative to the local ESIKF
        # posterior, while the incoming odometry factor above remains the
        # separate pre-image prior. This makes the two estimates fuse once
        # in the graph rather than reapplying the image correction when the
        # optimized correction is interpolated onto the local trajectory.
        target.raw_position = posterior_position.copy()
        target.raw_rotation = posterior_rotation.copy()
        self._intensity_factor_count += 1
        self._last_intensity_factor_timestamp = float(timestamp)
        interval = max(
            1, int(self.config.intensity_factor_optimize_interval)
        )
        if self._intensity_factor_count % interval == 0:
            self.graph.optimize()
        return True

    def absorb_latest_local_posterior(
        self,
        *,
        timestamp: float,
        observation_id: int,
        posterior_rotation_world_body: np.ndarray,
        posterior_position_world: np.ndarray,
    ) -> bool:
        """Fold an unexported local image update into the latest odometry.

        This fallback is used only when an ESIKF intensity update was accepted
        but could not produce a valid separate anchored graph factor.  If a
        geometry node was created at the same ping, leaving its raw baseline
        at the pre-image pose would move the correction discontinuity into the
        next odometry edge.  Rewriting the still-terminal incoming edge makes
        the graph represent the local posterior exactly once.
        """

        if not self.keyframes:
            return False
        target = self.keyframes[-1]
        if (
            target.observation_id != int(observation_id)
            or abs(target.timestamp - float(timestamp)) > 1.0e-9
        ):
            return False
        rotation = np.asarray(
            posterior_rotation_world_body, dtype=np.float64
        ).reshape(3, 3)
        position = np.asarray(
            posterior_position_world, dtype=np.float64
        ).reshape(3)
        posterior_pose = np.array(
            [
                position[0],
                position[1],
                rpy_from_rotation(rotation)[2],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(posterior_pose).all():
            return False

        replacement_index: int | None = None
        if target.index > 0:
            replacement_index = next(
                (
                    index
                    for index in range(len(self.graph.edges) - 1, -1, -1)
                    if self.graph.edges[index].kind == "odometry"
                    and self.graph.edges[index].target == target.index
                ),
                None,
            )
            if replacement_index is None:
                return False
        old_raw_pose = target.raw_pose2
        local_increment = _relative_pose2(old_raw_pose, posterior_pose)
        self.graph.poses[target.index] = _compose_pose2(
            self.graph.poses[target.index], local_increment
        )
        if target.index > 0:
            previous = self.keyframes[target.index - 1]
            assert replacement_index is not None
            old_edge = self.graph.edges[replacement_index]
            self.graph.edges[replacement_index] = PoseGraphEdge(
                source=old_edge.source,
                target=old_edge.target,
                measurement=_relative_pose2(
                    previous.raw_pose2, posterior_pose
                ),
                information=old_edge.information,
                kind=old_edge.kind,
                score=old_edge.score,
            )
        target.raw_position = position.copy()
        target.raw_rotation = rotation.copy()
        return True

    def optimize(self) -> np.ndarray:
        """Force all currently queued graph factors into the solution."""

        return self.graph.optimize()

    def _seed_global_relocalization(
        self, confirmed: list[PoseGraphEdge]
    ) -> bool:
        """Re-seed a badly drifted graph branch from a loop sequence.

        A robust loop kernel intentionally suppresses a single very large
        residual. After two independently verified adjacent matches, that
        behaviour would also suppress a legitimate global relocalization.
        Consistent loop transforms are therefore used to initialize only the
        newer branch before the normal robust optimizer is run.
        """

        if len(confirmed) < max(2, self.config.loop_confirmation_count):
            return False
        translations: list[np.ndarray] = []
        yaw_deltas: list[float] = []
        targets: list[int] = []
        for edge in confirmed:
            expected = _compose_pose2(
                self.graph.poses[edge.source], edge.measurement
            )
            current = self.graph.poses[edge.target]
            yaw_delta = float(wrap_angle(expected[2] - current[2]))
            c, s = np.cos(yaw_delta), np.sin(yaw_delta)
            rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
            translation = expected[:2] - rotation @ current[:2]
            translations.append(translation)
            yaw_deltas.append(yaw_delta)
            targets.append(edge.target)

        translations_array = np.asarray(translations, dtype=np.float64)
        translation = np.median(translations_array, axis=0)
        yaw_unwrapped = np.unwrap(np.asarray(yaw_deltas, dtype=np.float64))
        yaw_delta = float(wrap_angle(np.median(yaw_unwrapped)))
        correction_norm = float(np.linalg.norm(translation))
        if correction_norm < self.config.loop_relocalization_seed_threshold_m:
            return False
        translation_spread = float(
            np.max(
                np.linalg.norm(
                    translations_array - translation[None, :], axis=1
                )
            )
        )
        yaw_spread = float(
            np.max(np.abs(wrap_angle(yaw_unwrapped - yaw_delta)))
        )
        if (
            translation_spread
            > self.config.loop_relocalization_consistency_m
            or yaw_spread
            > np.deg2rad(self.config.loop_relocalization_consistency_deg)
        ):
            return False

        c, s = np.cos(yaw_delta), np.sin(yaw_delta)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        first_target = min(targets)
        self.graph.poses[first_target:, :2] = (
            self.graph.poses[first_target:, :2] @ rotation.T
            + translation
        )
        self.graph.poses[first_target:, 2] = wrap_angle(
            self.graph.poses[first_target:, 2] + yaw_delta
        )
        return True

    def _confirm_loop_sequence(
        self,
        current_index: int,
        candidate: PoseGraphEdge | None,
    ) -> list[PoseGraphEdge]:
        """Require neighbouring place matches before mutating the graph.

        Underwater bathymetry often contains long, locally similar slopes.  A
        single geometrically plausible match is therefore held as a proposal;
        the source and target indices must advance together before the loop is
        accepted.  Short gaps are allowed for feature-poor pings.
        """

        maximum_gap = max(1, int(self.config.loop_sequence_max_gap))
        current_sequence = self.keyframes[
            current_index
        ].geometry_sequence_index
        if candidate is None:
            if (
                self._pending_loops
                and current_sequence
                - self.keyframes[
                    self._pending_loops[-1].target
                ].geometry_sequence_index
                > maximum_gap
            ):
                self._pending_loops.clear()
            return []
        if self._pending_loops:
            previous = self._pending_loops[-1]
            source_step = (
                self.keyframes[candidate.source].geometry_sequence_index
                - self.keyframes[previous.source].geometry_sequence_index
            )
            target_step = (
                self.keyframes[candidate.target].geometry_sequence_index
                - self.keyframes[previous.target].geometry_sequence_index
            )
            consistent = (
                0 < source_step <= maximum_gap
                and 0 < target_step <= maximum_gap
                and abs(source_step - target_step) <= 1
            )
            if not consistent:
                self._pending_loops = [candidate]
            else:
                self._pending_loops.append(candidate)
        else:
            self._pending_loops = [candidate]

        if len(self._pending_loops) < self.config.loop_confirmation_count:
            return []
        confirmed: list[PoseGraphEdge] = []
        for edge in self._pending_loops:
            pair = (edge.source, edge.target)
            if pair not in self._committed_loop_pairs:
                confirmed.append(edge)
                self._committed_loop_pairs.add(pair)
        # Keep only recent context; committed pairs remain in the graph.
        self._pending_loops = self._pending_loops[-maximum_gap:]
        return confirmed

    def _find_and_verify_loop(
        self, current: AcousticKeyframe
    ) -> PoseGraphEdge | None:
        local_candidates: list[
            tuple[tuple[float, float, float], float, AcousticKeyframe]
        ] = []
        global_candidates: list[
            tuple[tuple[float, float, float], float, AcousticKeyframe]
        ] = []
        current_pose = current.raw_pose2
        maximum_yaw = np.deg2rad(self.config.loop_max_yaw_difference_deg)
        period_samples = [
            self.keyframes[edge.target].timestamp
            - self.keyframes[edge.source].timestamp
            for edge in (*self.loop_edges, *self._pending_loops)
        ]
        expected_period = (
            float(np.median(period_samples)) if period_samples else None
        )
        for candidate in self.keyframes[:-1]:
            if not candidate.loop_eligible:
                continue
            if current.timestamp - candidate.timestamp < self.config.loop_min_separation_s:
                continue
            spatial_distance = float(
                np.linalg.norm(current.raw_position[:2] - candidate.raw_position[:2])
            )
            yaw_difference = abs(
                float(wrap_angle(current_pose[2] - candidate.raw_pose2[2]))
            )
            distance = descriptor_distance(current.descriptor, candidate.descriptor)
            is_local = (
                spatial_distance <= self.config.loop_search_radius_m
                and yaw_difference <= maximum_yaw
                and distance <= self.config.loop_descriptor_threshold
            )
            is_global = (
                distance <= self.config.loop_global_descriptor_threshold
            )
            if not is_local and not is_global:
                continue
            if expected_period is None:
                ranking = (1.0, 0.0, distance)
            else:
                period_error = abs(
                    current.timestamp - candidate.timestamp - expected_period
                )
                ranking = (
                    0.0
                    if period_error <= self.config.loop_period_tolerance_s
                    else 1.0,
                    period_error,
                    distance,
                )
            entry = (ranking, distance, candidate)
            if is_local:
                local_candidates.append(entry)
            elif is_global:
                global_candidates.append(entry)
        local_candidates.sort(key=lambda item: item[0])
        global_candidates.sort(key=lambda item: item[0])
        candidates = local_candidates[: self.config.loop_max_candidates]
        candidates.extend(
            global_candidates[: self.config.loop_global_max_candidates]
        )

        best: tuple[float, AcousticKeyframe, RegistrationResult] | None = None
        for _, descriptor_score, candidate in candidates:
            rotation_target_source = candidate.raw_rotation.T @ current.raw_rotation
            translation_target_source = candidate.raw_rotation.T @ (
                current.raw_position - candidate.raw_position
            )
            registration = register_keyframes(
                current.points_body,
                candidate.points_body,
                rotation_target_source,
                translation_target_source,
                self.config,
            )
            if not registration.accepted:
                continue
            combined_score = registration.rmse + 0.25 * descriptor_score
            if combined_score > self.config.loop_registration_score_threshold:
                continue
            if best is None or combined_score < best[0]:
                best = (combined_score, candidate, registration)
        if best is None:
            return None
        score, candidate, registration = best
        measurement = np.array(
            [
                registration.translation_target_source[0],
                registration.translation_target_source[1],
                rpy_from_rotation(registration.rotation_target_source)[2],
            ],
            dtype=np.float64,
        )
        loop_xy_std = max(
            self.config.loop_xy_std_m,
            1.25 * registration.rmse,
        )
        return PoseGraphEdge(
            source=candidate.index,
            target=current.index,
            measurement=measurement,
            information=self._information(
                loop_xy_std, self.config.loop_yaw_std_deg
            ),
            kind="loop",
            score=score,
        )

    def diagnostics(self) -> PoseGraphDiagnostics:
        odometry_edges = sum(edge.kind == "odometry" for edge in self.graph.edges)
        intensity_edges = sum(
            edge.kind == "intensity" for edge in self.graph.edges
        )
        terrain_prior_count = sum(
            prior.kind == "terrain_prior" for prior in self.graph.priors
        )
        return PoseGraphDiagnostics(
            keyframes=len(self.keyframes),
            odometry_edges=int(odometry_edges),
            loop_edges=len(self.loop_edges),
            relocalization_edges=sum(
                edge.kind == "relocalization" for edge in self.loop_edges
            ),
            intensity_edges=intensity_edges,
            terrain_prior_count=int(terrain_prior_count),
            accepted_terrain_priors=self._terrain_prior_count,
            rejected_terrain_priors=(
                self._rejected_terrain_prior_count
            ),
            terrain_prior_rejection_reasons=dict(
                self._terrain_prior_rejection_reasons
            ),
            last_terrain_prior_rejection_reason=(
                self._last_terrain_prior_rejection_reason
            ),
            last_terrain_prefit_performed=(
                self._last_terrain_prefit_performed
            ),
            last_terrain_prefit_bypassed_for_relocalization=(
                self._last_terrain_prefit_bypassed_for_relocalization
            ),
            last_terrain_prefit_chi2=float(
                self._last_terrain_prefit_chi2
            ),
            last_terrain_prefit_dof=self._last_terrain_prefit_dof,
            terrain_relocalization_seeds=(
                self._terrain_relocalization_seed_count
            ),
            last_terrain_relocalization_translation_m=(
                self._last_terrain_relocalization_translation_m
            ),
            last_terrain_relocalization_yaw_deg=(
                self._last_terrain_relocalization_yaw_deg
            ),
            last_terrain_relocalization_anchor_index=(
                self._last_terrain_relocalization_anchor_index
            ),
            rejected_intensity_factors=(
                self._rejected_intensity_factor_count
            ),
            throttled_intensity_factors=(
                self._throttled_intensity_factor_count
            ),
            intensity_factor_rejection_reasons=dict(
                self._intensity_factor_rejection_reasons
            ),
            last_intensity_factor_rejection_reason=(
                self._last_intensity_factor_rejection_reason
            ),
            last_intensity_prefit_performed=(
                self._last_intensity_prefit_performed
            ),
            last_intensity_prefit_chi2=(
                float(self._last_intensity_prefit_chi2)
            ),
            last_intensity_prefit_dof=self._last_intensity_prefit_dof,
            pending_intensity_factors=0,
            optimization_runs=self.graph.optimization_runs,
            initial_cost=float(self.graph.initial_cost),
            final_cost=float(self.graph.final_cost),
            optimizer_backend=self.graph.backend_name,
            isam_updates=self.graph.isam_updates,
            fixed_lag_updates=self.graph.fixed_lag_updates,
            fixed_lag_active_variables=(
                self.graph.fixed_lag_active_variables
            ),
            fixed_lag_marginalized_variables=(
                self.graph.fixed_lag_marginalized_variables
            ),
            fixed_lag_global_only_factors=getattr(
                self.graph, "global_only_factor_count", 0
            ),
            optimizer_rebuilds=getattr(self.graph, "rebuilds", 0),
            global_isam_initial_cost=float(
                getattr(self.graph, "global_initial_cost", 0.0)
            ),
            global_isam_final_cost=float(
                getattr(self.graph, "global_final_cost", 0.0)
            ),
            global_anchor_bridge_updates=getattr(
                self.graph, "global_anchor_bridge_updates", 0
            ),
            global_anchor_bridge_max_translation_m=float(
                getattr(
                    self.graph,
                    "global_anchor_bridge_max_translation_m",
                    0.0,
                )
            ),
            global_anchor_bridge_max_abs_yaw_deg=float(
                getattr(
                    self.graph,
                    "global_anchor_bridge_max_abs_yaw_deg",
                    0.0,
                )
            ),
        )

    def optimized_keyframe_positions(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.keyframes:
            return np.empty(0), np.empty((0, 3))
        timestamps = np.array([frame.timestamp for frame in self.keyframes])
        positions = np.array([frame.raw_position for frame in self.keyframes])
        positions[:, :2] = self.graph.poses[:, :2]
        return timestamps, positions

    def _corrections(
        self, optimized_poses: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamps = np.array([frame.timestamp for frame in self.keyframes])
        optimized_values = (
            self.graph.poses
            if optimized_poses is None
            else np.asarray(optimized_poses, dtype=np.float64).reshape(-1, 3)
        )
        if optimized_values.shape[0] != len(self.keyframes):
            raise ValueError(
                "optimized pose count must match acoustic keyframe count"
            )
        yaw_delta = np.empty(len(self.keyframes), dtype=np.float64)
        translation = np.empty((len(self.keyframes), 2), dtype=np.float64)
        for index, frame in enumerate(self.keyframes):
            raw = frame.raw_pose2
            optimized = optimized_values[index]
            yaw_delta[index] = float(wrap_angle(optimized[2] - raw[2]))
            c, s = np.cos(yaw_delta[index]), np.sin(yaw_delta[index])
            rotated_raw = np.array(
                [c * raw[0] - s * raw[1], s * raw[0] + c * raw[1]]
            )
            translation[index] = optimized[:2] - rotated_raw
        # A direct-intensity update deliberately creates a second graph node
        # at the same sensor timestamp as the preceding geometry keyframe.
        # np.interp expects strictly increasing abscissae; retain the later
        # (post-image-update) correction from every equal-time run.
        if timestamps.size > 1:
            keep = np.r_[timestamps[1:] != timestamps[:-1], True]
            timestamps = timestamps[keep]
            yaw_delta = yaw_delta[keep]
            translation = translation[keep]
        return timestamps, yaw_delta, translation

    def correct_trajectory(
        self,
        timestamps: np.ndarray,
        raw_positions: np.ndarray,
        *,
        optimized_poses: np.ndarray | None = None,
    ) -> np.ndarray:
        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        positions = np.asarray(raw_positions, dtype=np.float64).reshape(-1, 3)
        if not self.keyframes or times.size == 0:
            return positions.copy()
        key_times, key_yaw_delta, key_translation = self._corrections(
            optimized_poses
        )
        # Unwrap before interpolation so corrections cross +/-pi smoothly.
        yaw_delta = np.interp(times, key_times, np.unwrap(key_yaw_delta))
        tx = np.interp(times, key_times, key_translation[:, 0])
        ty = np.interp(times, key_times, key_translation[:, 1])
        c, s = np.cos(yaw_delta), np.sin(yaw_delta)
        corrected = positions.copy()
        corrected[:, 0] = c * positions[:, 0] - s * positions[:, 1] + tx
        corrected[:, 1] = s * positions[:, 0] + c * positions[:, 1] + ty
        return corrected

    def corrected_pose(
        self, rotation_world_body: np.ndarray, position_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.keyframes:
            return np.asarray(rotation_world_body).copy(), np.asarray(position_world).copy()
        _, yaw_delta, translation = self._corrections()
        delta = yaw_delta[-1]
        correction_rotation = rotation_from_rpy(0.0, 0.0, delta)
        position = np.asarray(position_world, dtype=np.float64).reshape(3)
        corrected = position.copy()
        corrected[:2] = (
            correction_rotation[:2, :2] @ position[:2] + translation[-1]
        )
        rotation = correction_rotation @ np.asarray(rotation_world_body).reshape(3, 3)
        return rotation, corrected

    def optimized_map_points(self) -> np.ndarray:
        clouds: list[np.ndarray] = []
        for index, frame in enumerate(self.keyframes):
            if not frame.loop_eligible or frame.points_body.size == 0:
                continue
            raw_yaw = frame.raw_pose2[2]
            optimized_yaw = self.graph.poses[index, 2]
            correction = rotation_from_rpy(0.0, 0.0, optimized_yaw - raw_yaw)
            rotation = correction @ frame.raw_rotation
            position = frame.raw_position.copy()
            position[:2] = self.graph.poses[index, :2]
            clouds.append(frame.points_body @ rotation.T + position)
        if not clouds:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(clouds, axis=0)
