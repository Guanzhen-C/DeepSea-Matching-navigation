"""Error-state iterated Kalman filter for underwater acoustic navigation.

The architecture adapts FAST-LIVO2 to an acoustic sensor: high-rate IMU
propagation, sequential heterogeneous geometry/intensity updates, and direct
point-to-map iteration. The observation model is changed
for sonar by using polar anisotropic uncertainty, an absolute depth update,
and explicit geometric-degeneracy reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from .intensity_map import AcousticIntensityMap
from .math3d import project_to_so3, skew, so3_exp, transform_points
from .surfel_map import SurfelAssociationBatch, SurfelVoxelMap


@dataclass
class ESIKFConfig:
    gravity_mps2: float = 9.80665
    # Continuous-time white-noise densities used by covariance propagation.
    # These correspond to the demo's 50 Hz per-sample noise (0.0022 rad/s
    # and 0.025 m/s^2), rather than incorrectly treating sample deviation as
    # a spectral density.
    gyro_noise_std: float = 3.2e-4
    accel_noise_std: float = 3.6e-3
    gyro_bias_rw_std: float = 2.0e-5
    accel_bias_rw_std: float = 3.0e-4
    # Optional calibration-envelope constraint.  Leave infinite for generic
    # hardware; a vehicle runner can set this from its IMU data sheet.
    gyro_bias_max_norm: float = float("inf")
    dvl_velocity_std: float = 0.045
    dvl_huber_mahalanobis: float = 16.0
    dvl_max_covariance_scale: float = 100.0
    dvl_max_residual_mps: float = 2.5
    # A bottom-track DVL measures velocity in the vehicle frame; one sample
    # contains no absolute-position or global-heading observation.  Preserve
    # those pose gauge directions during the DVL correction while still
    # updating velocity, IMU biases, and the observable roll/pitch subspace.
    # This is an observability-constrained partial update, not a residual gate
    # or a reduction of the three-axis DVL measurement information.
    dvl_preserve_pose_gauge: bool = True
    depth_std: float = 0.04
    sonar_range_std: float = 0.035
    sonar_angle_std_deg: float = 0.25
    # This includes unmodelled sound-speed, footprint, and map-association
    # error; it is intentionally larger than the raw range repeatability.
    sonar_noise_floor: float = 0.30
    sonar_huber_delta: float = 0.22
    sonar_max_iterations: int = 5
    sonar_min_correspondences: int = 18
    sonar_max_points: int = 128
    sonar_convergence_rotation: float = 2.0e-5
    sonar_convergence_translation: float = 2.0e-4
    sonar_max_rotation_step_deg: float = 0.35
    sonar_max_translation_step: float = 0.12
    sonar_max_velocity_step: float = 0.08
    sonar_max_gyro_bias_step: float = 2.0e-5
    sonar_max_accel_bias_step: float = 0.002
    intensity_noise_std: float = 0.09
    intensity_huber_delta: float = 0.14
    intensity_min_correspondences: int = 18
    intensity_max_points: int = 384
    intensity_max_iterations: int = 3
    intensity_max_rotation_step_deg: float = 0.18
    intensity_max_translation_step: float = 0.06
    # The graph consumes a verified local quadratic image-only normal
    # equation, not the prior-regularized ESIKF posterior. Reject implausibly
    # distant local optima rather than turning them into a strong factor.
    intensity_graph_max_translation_m: float = 0.75
    intensity_graph_max_yaw_deg: float = 3.0
    intensity_min_yaw_information: float = 8.0
    # Direct intensity alignment is used only when a substantial part of the
    # strip sees an older reference and the two signals agree after affine
    # gain/offset normalization.  This prevents a low-residual fit to a small,
    # repetitive seabed patch from being counted as useful information.
    intensity_min_overlap_ratio: float = 0.45
    intensity_min_normalized_correlation: float = 0.35
    intensity_min_inlier_ratio: float = 0.55
    intensity_min_planar_rank: int = 2
    intensity_min_singular_ratio: float = 2.0e-3
    # Dense neighbouring rays reuse the same fitted map cells and are not
    # statistically independent. Cap their aggregate weight at this many
    # independent-equivalent samples in the local filter.
    intensity_effective_independent_samples: int = 96
    # The intensity and depth samples are emitted by the same integrated
    # acoustic ping.  A photometric match that disagrees with the locally
    # fitted seabed elevation is therefore continuously down-weighted with a
    # Cauchy response.  This is a scale, not an accept/reject threshold.
    intensity_surface_consistency_std: float = 0.75
    # With backscatter measured on the same bottom detections as bathymetry,
    # the reliable direct observables are horizontal position and yaw. Roll,
    # pitch, and depth remain constrained by IMU/MBES/pressure respectively.
    intensity_planar_only: bool = True
    degeneracy_eigenvalue: float = 500.0
    degeneracy_rotation_eigenvalue: float = 500.0
    degeneracy_min_eigen_ratio: float = 0.003
    degeneracy_min_weight: float = 0.15
    # Shape each rotational/translational information eigendirection
    # independently instead of multiplying the complete acoustic update by
    # the weakest translation direction.  For eigenvalue lambda and the
    # configured information knee k, the retained information amplitude is
    # lambda / (lambda + k).  This response is continuous, has no arbitrary
    # accept/reject boundary, and tends exactly to zero in a true null space.
    # ``degeneracy_min_weight`` remains only for reproducible legacy runs when
    # both this option and hard subspace projection are disabled.
    sonar_continuous_observability_weighting: bool = True
    # A scalar weight cannot distinguish the well-constrained surface-normal
    # direction from the unconstrained tangent directions of an MBES strip.
    # FAST-LIVO2's direct update is retained, but weak eigendirections are
    # removed from the point-to-plane Jacobian before solving the ESIKF step.
    # Disabled until an acoustic intensity update can supply the weak
    # tangent/yaw directions.  It remains available for geometry-only
    # ablations and is enabled selectively when that complementary channel
    # is present; hard projection alone regresses the two-lap benchmark.
    sonar_observable_subspace_projection: bool = False
    sonar_min_observable_pose_rank: int = 1
    covariance_floor: float = 1.0e-10


@dataclass
class InertialState:
    timestamp: float = 0.0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    gyro_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    accel_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    covariance: np.ndarray = field(
        default_factory=lambda: np.eye(15, dtype=np.float64) * 1.0e-3
    )

    def copy(self) -> "InertialState":
        return InertialState(
            timestamp=float(self.timestamp),
            rotation=self.rotation.copy(),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            gyro_bias=self.gyro_bias.copy(),
            accel_bias=self.accel_bias.copy(),
            covariance=self.covariance.copy(),
        )

    def pose_matrix(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.rotation
        pose[:3, 3] = self.position
        return pose


@dataclass(frozen=True)
class SonarUpdateResult:
    accepted: bool
    correspondences: int
    inliers: int
    iterations: int
    rmse_before: float
    rmse_after: float
    minimum_translation_eigenvalue: float
    information_condition: float
    adaptive_weight: float
    rotation_observable_rank: int
    translation_observable_rank: int
    rotation_correction_deg: float
    translation_correction: float
    rejection_reason: str | None = None
    information_scale: float = 1.0

    @property
    def degenerate(self) -> bool:
        return (
            self.rotation_observable_rank + self.translation_observable_rank
            < 6
            or self.adaptive_weight < 0.999
        )


@dataclass(frozen=True)
class IntensityUpdateResult:
    """Diagnostics for one direct side-scan intensity update."""

    accepted: bool
    correspondences: int
    inliers: int
    iterations: int
    rmse_before: float
    rmse_after: float
    yaw_information: float
    overlap_ratio: float
    normalized_correlation: float
    planar_observable_rank: int
    gain: float
    offset: float
    rotation_correction_deg: float
    translation_correction: float
    rejection_reason: str | None = None
    # Exact reference-submap lineage and measurement-only planar normal
    # equation used by the global graph. Pose order is
    # [x_world, y_world, yaw_world]. Rejected updates carry no anchor and a
    # zero information matrix.
    reference_observation_id: int | None = None
    anchor_keyframe_index: int | None = None
    photometric_pose2: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    information_pose2: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )


class AcousticESIKF:
    """15-state acoustic/inertial iterated error-state filter.

    Error ordering is ``[dtheta, dp, dv, dbg, dba]``.  Rotations use a right
    perturbation, ``R_true = R_nominal Exp(dtheta)``.
    """

    STATE_SIZE = 15

    def __init__(
        self,
        config: ESIKFConfig | None = None,
        *,
        sensor_rotation_body: np.ndarray | None = None,
        sensor_translation_body: np.ndarray | None = None,
        dvl_rotation_body: np.ndarray | None = None,
        dvl_translation_body: np.ndarray | None = None,
        surfel_associator=None,
    ) -> None:
        self.config = config or ESIKFConfig()
        self.sensor_rotation_body = (
            np.eye(3, dtype=np.float64)
            if sensor_rotation_body is None
            else project_to_so3(sensor_rotation_body)
        )
        self.sensor_translation_body = (
            np.zeros(3, dtype=np.float64)
            if sensor_translation_body is None
            else np.asarray(sensor_translation_body, dtype=np.float64).reshape(3)
        )
        # Extrinsics map DVL-frame vectors/points into the IMU body frame.
        # They are fixed calibration inputs rather than weakly observable
        # online states.  Identity/zero retains the historical body-origin
        # measurement contract exactly.
        self.dvl_rotation_body = (
            np.eye(3, dtype=np.float64)
            if dvl_rotation_body is None
            else project_to_so3(dvl_rotation_body)
        )
        self.dvl_translation_body = (
            np.zeros(3, dtype=np.float64)
            if dvl_translation_body is None
            else np.asarray(dvl_translation_body, dtype=np.float64).reshape(3)
        )
        if not np.isfinite(self.dvl_translation_body).all():
            raise ValueError("DVL translation must be finite")
        self.state = InertialState()
        self.initialized = False
        self._last_gyro = np.zeros(3, dtype=np.float64)
        self._last_accel = np.zeros(3, dtype=np.float64)
        self.surfel_associator = surfel_associator
        self._association_calls = 0
        self._last_association_ms = 0.0
        self._mean_association_ms = 0.0

    @property
    def association_calls(self) -> int:
        return self._association_calls

    @property
    def last_association_ms(self) -> float:
        return self._last_association_ms

    @property
    def mean_association_ms(self) -> float:
        return self._mean_association_ms

    @property
    def gravity(self) -> np.ndarray:
        return np.array(
            [0.0, 0.0, -self.config.gravity_mps2], dtype=np.float64
        )

    def initialize(
        self,
        timestamp: float,
        *,
        rotation: np.ndarray | None = None,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        gyro_bias: np.ndarray | None = None,
        accel_bias: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
    ) -> None:
        self.state = InertialState(
            timestamp=float(timestamp),
            rotation=(
                np.eye(3, dtype=np.float64)
                if rotation is None
                else project_to_so3(rotation)
            ),
            position=(
                np.zeros(3, dtype=np.float64)
                if position is None
                else np.asarray(position, dtype=np.float64).reshape(3).copy()
            ),
            velocity=(
                np.zeros(3, dtype=np.float64)
                if velocity is None
                else np.asarray(velocity, dtype=np.float64).reshape(3).copy()
            ),
            gyro_bias=(
                np.zeros(3, dtype=np.float64)
                if gyro_bias is None
                else np.asarray(gyro_bias, dtype=np.float64).reshape(3).copy()
            ),
            accel_bias=(
                np.zeros(3, dtype=np.float64)
                if accel_bias is None
                else np.asarray(accel_bias, dtype=np.float64).reshape(3).copy()
            ),
            covariance=(
                self._default_covariance()
                if covariance is None
                else self._regularize_covariance(covariance)
            ),
        )
        self.initialized = True
        self._association_calls = 0
        self._last_association_ms = 0.0
        self._mean_association_ms = 0.0

    @staticmethod
    def _default_covariance() -> np.ndarray:
        standard_deviations = np.array(
            [
                0.03,
                0.03,
                0.06,
                0.25,
                0.25,
                0.10,
                0.15,
                0.15,
                0.08,
                0.003,
                0.003,
                0.006,
                0.04,
                0.04,
                0.04,
            ],
            dtype=np.float64,
        )
        return np.diag(standard_deviations**2)

    def _regularize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        covariance = np.asarray(covariance, dtype=np.float64).reshape(15, 15)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, self.config.covariance_floor)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    @staticmethod
    def _observable_projector(
        jacobian_block: np.ndarray,
        weights: np.ndarray,
        *,
        absolute_threshold: float,
        relative_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return the information-supported subspace of a 3-DoF block.

        MBES measurements form narrow cross-track strips.  On a locally
        planar seabed their point-to-plane Jacobians contain strong
        information along the surface normal but almost none along the two
        tangent directions.  Scaling the complete update by one degeneracy
        number still lets numerical association noise move those tangent
        states.  Projecting onto the eigenvectors that pass both an absolute
        and relative information test preserves the direct raw-point update
        while leaving genuinely unobservable states to the inertial prior.
        """

        block = np.asarray(jacobian_block, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if block.shape[0] != weights.size:
            raise ValueError("Jacobian rows and weights must match")
        information = block.T @ (weights[:, None] * block)
        information = 0.5 * (information + information.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(information)
        except np.linalg.LinAlgError:
            return np.zeros((3, 3), dtype=np.float64), np.zeros(3), 0
        eigenvalues = np.maximum(eigenvalues, 0.0)
        largest = float(max(eigenvalues[-1], 0.0))
        threshold = max(
            float(absolute_threshold),
            float(relative_threshold) * largest,
        )
        observable = eigenvalues >= threshold
        rank = int(np.count_nonzero(observable))
        if rank == 0:
            projector = np.zeros((3, 3), dtype=np.float64)
        else:
            basis = eigenvectors[:, observable]
            projector = basis @ basis.T
        return projector, eigenvalues, rank

    @staticmethod
    def _continuous_observability_transform(
        jacobian_block: np.ndarray,
        weights: np.ndarray,
        *,
        absolute_knee: float,
        relative_knee: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a smooth square-root information transform for 3 DoF.

        A hard rank decision is useful for diagnostics, but applying it to a
        live filter makes the posterior discontinuous as terrain changes.  A
        scalar frame weight has the opposite defect: one weak tangent can
        suppress a well-observed surface-normal direction while still
        leaking numerical information into the tangent.  This spectral
        response preserves strong eigenvectors and continuously removes weak
        ones.  Right-multiplying a Jacobian block by the returned transform
        applies the response once to its normal matrix.
        """

        block = np.asarray(jacobian_block, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if block.shape[0] != weights.size:
            raise ValueError("Jacobian rows and weights must match")
        information = block.T @ (weights[:, None] * block)
        information = 0.5 * (information + information.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(information)
        except np.linalg.LinAlgError:
            return (
                np.zeros((3, 3), dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        eigenvalues = np.maximum(eigenvalues, 0.0)
        largest = float(max(eigenvalues[-1], 0.0))
        knee = max(
            0.0,
            float(absolute_knee),
            float(relative_knee) * largest,
        )
        if knee <= 0.0:
            response = np.ones(3, dtype=np.float64)
        else:
            response = eigenvalues / (eigenvalues + knee)
        transform = (eigenvectors * np.sqrt(response)) @ eigenvectors.T
        return transform, eigenvalues, response

    def propagate(
        self,
        timestamp: float,
        gyro_rad_s: np.ndarray,
        specific_force_mps2: np.ndarray,
    ) -> bool:
        """Propagate the nominal state and covariance with one IMU sample."""

        if not self.initialized:
            raise RuntimeError("filter must be initialized before propagation")
        timestamp = float(timestamp)
        dt = timestamp - self.state.timestamp
        if dt <= 0.0:
            return False
        if dt > 0.25:
            raise ValueError(f"IMU gap {dt:.3f}s is too large")

        gyro = np.asarray(gyro_rad_s, dtype=np.float64).reshape(3)
        accel = np.asarray(specific_force_mps2, dtype=np.float64).reshape(3)
        if not np.isfinite(gyro).all() or not np.isfinite(accel).all():
            raise ValueError("IMU samples must be finite")
        omega = gyro - self.state.gyro_bias
        specific_force = accel - self.state.accel_bias

        rotation_before = self.state.rotation
        acceleration_world = rotation_before @ specific_force + self.gravity
        self.state.position = (
            self.state.position
            + self.state.velocity * dt
            + 0.5 * acceleration_world * dt * dt
        )
        self.state.velocity = self.state.velocity + acceleration_world * dt
        self.state.rotation = project_to_so3(
            rotation_before @ so3_exp(omega * dt)
        )

        f = np.zeros((15, 15), dtype=np.float64)
        f[0:3, 0:3] = -skew(omega)
        f[0:3, 9:12] = -np.eye(3)
        f[3:6, 6:9] = np.eye(3)
        f[6:9, 0:3] = -rotation_before @ skew(specific_force)
        f[6:9, 12:15] = -rotation_before
        g = np.zeros((15, 12), dtype=np.float64)
        g[0:3, 0:3] = -np.eye(3)
        g[6:9, 3:6] = -rotation_before
        g[9:12, 6:9] = np.eye(3)
        g[12:15, 9:12] = np.eye(3)
        q = np.diag(
            np.repeat(
                [
                    self.config.gyro_noise_std**2,
                    self.config.accel_noise_std**2,
                    self.config.gyro_bias_rw_std**2,
                    self.config.accel_bias_rw_std**2,
                ],
                3,
            )
        )
        phi = np.eye(15) + f * dt + 0.5 * (f @ f) * dt * dt
        process_noise = g @ q @ g.T * dt
        self.state.covariance = self._regularize_covariance(
            phi @ self.state.covariance @ phi.T + process_noise
        )
        self.state.timestamp = timestamp
        self._last_gyro = gyro
        self._last_accel = accel
        return True

    def deskew_sensor_points(
        self,
        points_sensor: np.ndarray,
        sample_time_offsets_s: np.ndarray,
    ) -> np.ndarray:
        """Express asynchronously sampled acoustic points at the state epoch.

        ``sample_time_offsets_s[i]`` is the pose-sampling time of point ``i``
        relative to ``state.timestamp``.  A constant body angular rate and
        constant world velocity are used across the short ping.  Fixed sonar
        lever-arm motion is included.  The operation is exact identity for a
        synchronous ping, which is the current simulator contract.

        Acoustic two-way travel time is not a pose timestamp and must not be
        passed here without an instrument-specific transmit/receive model.
        """

        points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
        offsets = np.asarray(
            sample_time_offsets_s, dtype=np.float64
        ).reshape(-1)
        if offsets.size != points.shape[0]:
            raise ValueError("sample time offsets must match sensor points")
        if not np.isfinite(offsets).all():
            raise ValueError("sample time offsets must be finite")
        if points.shape[0] == 0 or not np.any(offsets):
            return points.copy()

        omega_body = self._last_gyro - self.state.gyro_bias
        body_velocity_reference = self.state.rotation.T @ self.state.velocity
        rotation_bs = self.sensor_rotation_body
        translation_bs = self.sensor_translation_body
        points_body_sample = points @ rotation_bs.T + translation_bs
        deskewed_body = np.empty_like(points_body_sample)
        unique_offsets, inverse = np.unique(offsets, return_inverse=True)
        for offset_index, offset in enumerate(unique_offsets):
            relative_rotation = so3_exp(omega_body * float(offset))
            selected = inverse == offset_index
            deskewed_body[selected] = (
                points_body_sample[selected] @ relative_rotation.T
                + body_velocity_reference * float(offset)
            )
        return (
            deskewed_body - translation_bs
        ) @ rotation_bs

    def _state_with_error(
        self, base: InertialState, error: np.ndarray
    ) -> InertialState:
        error = np.asarray(error, dtype=np.float64).reshape(15)
        candidate = base.copy()
        candidate.rotation = project_to_so3(
            base.rotation @ so3_exp(error[0:3])
        )
        candidate.position = base.position + error[3:6]
        candidate.velocity = base.velocity + error[6:9]
        candidate.gyro_bias = base.gyro_bias + error[9:12]
        bias_limit = float(self.config.gyro_bias_max_norm)
        bias_norm = float(np.linalg.norm(candidate.gyro_bias))
        if np.isfinite(bias_limit) and bias_limit >= 0.0 and bias_norm > bias_limit:
            candidate.gyro_bias *= bias_limit / bias_norm
        candidate.accel_bias = base.accel_bias + error[12:15]
        return candidate

    def _inject(self, error: np.ndarray) -> None:
        old_covariance = self.state.covariance
        self.state = self._state_with_error(self.state, error)
        reset = self._error_reset_jacobian(error)
        self.state.covariance = self._regularize_covariance(
            reset @ old_covariance @ reset.T
        )

    @staticmethod
    def _error_reset_jacobian(error: np.ndarray) -> np.ndarray:
        """Map posterior covariance into the retracted nominal tangent space.

        The filter uses a right attitude perturbation,
        ``R_true = R_nominal Exp(dtheta)``.  After injecting an estimated
        error, the attitude-error coordinates change.  The first-order reset
        Jacobian is therefore ``I - 1/2 [dtheta]x``.  Applying this transform
        to the *posterior* covariance is essential; applying it before a
        Joseph update and then overwriting it silently discards the reset.
        """

        error = np.asarray(error, dtype=np.float64).reshape(15)
        reset = np.eye(15, dtype=np.float64)
        reset[0:3, 0:3] -= 0.5 * skew(error[0:3])
        return reset

    def _dvl_correction_projector(self) -> np.ndarray:
        """Return the observable-state projector for one DVL update.

        In this filter's right/body attitude coordinates, an infinitesimal
        rotation about the fixed world vertical is ``R.T @ e_z``.  DVL body
        velocity cannot observe that global-heading gauge, nor can it observe
        the vehicle's accumulated position.  Projecting the Kalman correction
        (rather than rejecting or weakening the measurement) prevents a
        velocity residual from retracting the live local-map frame.
        """

        projector = np.eye(15, dtype=np.float64)
        world_vertical_body = self.state.rotation.T @ np.array(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
        world_vertical_body /= max(
            float(np.linalg.norm(world_vertical_body)), 1.0e-12
        )
        projector[0:3, 0:3] -= np.outer(
            world_vertical_body, world_vertical_body
        )
        projector[3:6, 3:6] = 0.0
        return projector

    def _kalman_update(
        self,
        residual: np.ndarray,
        jacobian: np.ndarray,
        measurement_covariance: np.ndarray,
        *,
        mahalanobis_limit: float | None = None,
        correction_projector: np.ndarray | None = None,
    ) -> bool:
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        jacobian = np.asarray(jacobian, dtype=np.float64).reshape(
            residual.size, 15
        )
        measurement_covariance = np.asarray(
            measurement_covariance, dtype=np.float64
        ).reshape(residual.size, residual.size)
        innovation = (
            jacobian @ self.state.covariance @ jacobian.T
            + measurement_covariance
        )
        if mahalanobis_limit is not None:
            try:
                mahalanobis = float(
                    residual @ np.linalg.solve(innovation, residual)
                )
            except np.linalg.LinAlgError:
                return False
            if mahalanobis > mahalanobis_limit:
                return False
        try:
            gain = np.linalg.solve(
                innovation,
                jacobian @ self.state.covariance,
            ).T
        except np.linalg.LinAlgError:
            return False
        if correction_projector is not None:
            correction_projector = np.asarray(
                correction_projector, dtype=np.float64
            ).reshape(15, 15)
            gain = correction_projector @ gain
        prior_covariance = self.state.covariance.copy()
        error = gain @ residual
        identity_minus = np.eye(15) - gain @ jacobian
        posterior_covariance = (
            identity_minus @ prior_covariance @ identity_minus.T
            + gain @ measurement_covariance @ gain.T
        )
        candidate = self._state_with_error(self.state, error)
        reset = self._error_reset_jacobian(error)
        candidate.covariance = self._regularize_covariance(
            reset @ posterior_covariance @ reset.T
        )
        self.state = candidate
        return True

    def update_dvl(
        self,
        velocity_body_mps: np.ndarray,
        covariance: np.ndarray | None = None,
    ) -> bool:
        """Fuse a bottom-track DVL velocity at its calibrated phase centre.

        The historical argument name is retained for API compatibility.  With
        identity DVL rotation and zero translation it is a body-origin vector.
        Otherwise the measurement is expressed in DVL axes, and the predicted
        velocity includes the rigid-body lever-arm term ``omega x t_bd``.

        Bottom-track velocity is the estimator's most dependable short-term
        motion constraint.  A hard statistical gate is brittle after an
        acoustic pose correction: even a valid two-degree attitude change can
        momentarily create a several-sigma body-velocity innovation.  We use
        covariance inflation (a multivariate Huber update) for that case and
        retain only a generous physical outlier gate.  This lets DVL pull the
        inertial state back into agreement without giving a single sample
        unlimited influence.
        """

        measurement = np.asarray(
            velocity_body_mps, dtype=np.float64
        ).reshape(3)
        velocity_body_origin = self.state.rotation.T @ self.state.velocity
        angular_velocity_body = self._last_gyro - self.state.gyro_bias
        velocity_body_dvl_origin = velocity_body_origin + np.cross(
            angular_velocity_body, self.dvl_translation_body
        )
        rotation_db = self.dvl_rotation_body.T
        predicted = rotation_db @ velocity_body_dvl_origin
        jacobian = np.zeros((3, 15), dtype=np.float64)
        jacobian[:, 0:3] = rotation_db @ skew(velocity_body_origin)
        jacobian[:, 6:9] = rotation_db @ self.state.rotation.T
        jacobian[:, 9:12] = rotation_db @ skew(
            self.dvl_translation_body
        )
        if covariance is None:
            covariance = np.eye(3) * self.config.dvl_velocity_std**2
        covariance = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
        residual = measurement - predicted
        if np.linalg.norm(residual) > self.config.dvl_max_residual_mps:
            return False

        innovation = (
            jacobian @ self.state.covariance @ jacobian.T + covariance
        )
        try:
            mahalanobis = float(residual @ np.linalg.solve(innovation, residual))
        except np.linalg.LinAlgError:
            return False
        robust_limit = max(self.config.dvl_huber_mahalanobis, 1.0e-6)
        covariance_scale = float(
            np.clip(
                mahalanobis / robust_limit,
                1.0,
                self.config.dvl_max_covariance_scale,
            )
        )
        return self._kalman_update(
            residual,
            jacobian,
            covariance * covariance_scale,
            correction_projector=(
                self._dvl_correction_projector()
                if self.config.dvl_preserve_pose_gauge
                else None
            ),
        )

    def update_depth(
        self, depth_m: float, variance: float | None = None
    ) -> bool:
        """Fuse positive-down pressure depth in a Z-up world frame."""

        if not np.isfinite(depth_m):
            return False
        predicted = -float(self.state.position[2])
        jacobian = np.zeros((1, 15), dtype=np.float64)
        jacobian[0, 5] = -1.0
        measurement_variance = (
            self.config.depth_std**2 if variance is None else float(variance)
        )
        return self._kalman_update(
            np.array([float(depth_m) - predicted]),
            jacobian,
            np.array([[measurement_variance]]),
            mahalanobis_limit=16.0,
        )

    def points_sensor_to_world(
        self, points_sensor: np.ndarray, state: InertialState | None = None
    ) -> np.ndarray:
        state = self.state if state is None else state
        points_body = transform_points(
            self.sensor_rotation_body,
            self.sensor_translation_body,
            points_sensor,
        )
        return transform_points(state.rotation, state.position, points_body)

    def _sonar_linearization(
        self,
        state: InertialState,
        points_sensor: np.ndarray,
        surfel_map: SurfelVoxelMap,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points_body = transform_points(
            self.sensor_rotation_body,
            self.sensor_translation_body,
            points_sensor,
        )
        points_world = transform_points(
            state.rotation, state.position, points_body
        )
        world_sensor_rotation = state.rotation @ self.sensor_rotation_body
        angle_std = np.deg2rad(self.config.sonar_angle_std_deg)

        association_started = time.perf_counter()
        associations: SurfelAssociationBatch | None = None
        if self.surfel_associator is not None:
            associations = self.surfel_associator.associate(
                points_world, surfel_map
            )
        if associations is None:
            associations = surfel_map.associate_points(points_world)
        association_ms = 1.0e3 * (
            time.perf_counter() - association_started
        )
        self._association_calls += 1
        self._last_association_ms = association_ms
        self._mean_association_ms += (
            association_ms - self._mean_association_ms
        ) / float(self._association_calls)

        ranges = np.linalg.norm(points_sensor, axis=1)
        valid = np.asarray(associations.valid_mask, dtype=bool) & (
            ranges >= 1.0e-6
        )
        if not np.any(valid):
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, 15), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        selected_sensor = points_sensor[valid]
        selected_body = points_body[valid]
        selected_world = points_world[valid]
        centroids = associations.centroids[valid]
        normals_world = associations.normals[valid]
        selected_ranges = ranges[valid]

        residuals = np.einsum(
            "ij,ij->i", normals_world, selected_world - centroids
        )
        jacobians = np.zeros((residuals.size, 15), dtype=np.float64)
        # n^T(-R[p]_x) == (p x R^T n)^T for the right/body-frame
        # attitude perturbation used by this filter.
        normals_body = normals_world @ state.rotation
        jacobians[:, 0:3] = np.cross(selected_body, normals_body)
        jacobians[:, 3:6] = normals_world

        directions_sensor = selected_sensor / selected_ranges[:, None]
        normals_sensor = normals_world @ world_sensor_rotation
        radial_projection = np.einsum(
            "ij,ij->i", normals_sensor, directions_sensor
        )
        normal_norm_squared = np.einsum(
            "ij,ij->i", normals_sensor, normals_sensor
        )
        tangential_projection_squared = np.maximum(
            0.0, normal_norm_squared - radial_projection**2
        )
        variances = (
            self.config.sonar_range_std**2 * radial_projection**2
            + (angle_std * selected_ranges) ** 2
            * tangential_projection_squared
            + np.maximum(0.0, associations.normal_variances[valid])
            + self.config.sonar_noise_floor**2
        )
        return (
            np.ascontiguousarray(residuals, dtype=np.float64),
            np.ascontiguousarray(jacobians, dtype=np.float64),
            np.ascontiguousarray(np.maximum(variances, 1.0e-8)),
        )

    def update_sonar(
        self,
        points_sensor: np.ndarray,
        surfel_map: SurfelVoxelMap,
        *,
        information_scale: float = 1.0,
    ) -> SonarUpdateResult:
        """Perform a robust iterated point-to-surfel acoustic update.

        ``information_scale`` tempers a likelihood whose source samples are
        intentionally reused by overlapping sliding windows.  It scales the
        complete acoustic information continuously; it is not a residual or
        acceptance gate.
        """

        information_scale = float(information_scale)
        if not np.isfinite(information_scale) or information_scale <= 0.0:
            raise ValueError("sonar information scale must be finite and positive")

        points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] > self.config.sonar_max_points:
            indices = np.linspace(
                0,
                points.shape[0] - 1,
                self.config.sonar_max_points,
                dtype=np.int64,
            )
            points = points[indices]
        def rejected_result(
            reason: str,
            *,
            correspondences: int = 0,
            iterations: int = 0,
            rmse_before_value: float = np.nan,
            rmse_after_value: float = np.nan,
            minimum_eigenvalue: float = 0.0,
            condition: float = np.inf,
            rotation_rank: int = 0,
            translation_rank: int = 0,
            correction_value: np.ndarray | None = None,
        ) -> SonarUpdateResult:
            correction_value = (
                np.zeros(15, dtype=np.float64)
                if correction_value is None
                else np.asarray(correction_value, dtype=np.float64).reshape(15)
            )
            return SonarUpdateResult(
                accepted=False,
                correspondences=int(correspondences),
                inliers=0,
                iterations=int(iterations),
                rmse_before=float(rmse_before_value),
                rmse_after=float(rmse_after_value),
                minimum_translation_eigenvalue=float(minimum_eigenvalue),
                information_condition=float(condition),
                adaptive_weight=0.0,
                rotation_observable_rank=int(rotation_rank),
                translation_observable_rank=int(translation_rank),
                rotation_correction_deg=float(
                    np.rad2deg(np.linalg.norm(correction_value[0:3]))
                ),
                translation_correction=float(
                    np.linalg.norm(correction_value[3:6])
                ),
                rejection_reason=reason,
                information_scale=information_scale,
            )
        if points.shape[0] < self.config.sonar_min_correspondences:
            return rejected_result("insufficient_input_points")

        base = self.state.copy()
        prior_covariance = self._regularize_covariance(base.covariance)
        prior_information = np.linalg.pinv(prior_covariance, rcond=1.0e-10)
        correction = np.zeros(15, dtype=np.float64)
        rmse_before = np.nan
        adaptive_weight = 1.0
        minimum_translation_eigenvalue = 0.0
        information_condition = np.inf
        rotation_observable_rank = 0
        translation_observable_rank = 0
        iterations = 0
        last_count = 0

        for iteration in range(self.config.sonar_max_iterations):
            candidate = self._state_with_error(base, correction)
            residual, jacobian, variance = self._sonar_linearization(
                candidate, points, surfel_map
            )
            last_count = residual.size
            if residual.size < self.config.sonar_min_correspondences:
                return rejected_result(
                    "insufficient_correspondences",
                    correspondences=last_count,
                    iterations=iterations,
                    rmse_before_value=rmse_before,
                    minimum_eigenvalue=minimum_translation_eigenvalue,
                    condition=information_condition,
                    rotation_rank=rotation_observable_rank,
                    translation_rank=translation_observable_rank,
                    correction_value=correction,
                )
            if iteration == 0:
                rmse_before = float(np.sqrt(np.mean(residual**2)))

            normalized = np.abs(residual) / self.config.sonar_huber_delta
            robust_weight = np.ones_like(normalized)
            outside = normalized > 1.0
            robust_weight[outside] = 1.0 / normalized[outside]
            weights = robust_weight / variance

            rotation_projector, _, rotation_observable_rank = (
                self._observable_projector(
                    jacobian[:, 0:3],
                    weights,
                    absolute_threshold=(
                        self.config.degeneracy_rotation_eigenvalue
                    ),
                    relative_threshold=self.config.degeneracy_min_eigen_ratio,
                )
            )
            (
                translation_projector,
                translation_eigenvalues,
                translation_observable_rank,
            ) = self._observable_projector(
                jacobian[:, 3:6],
                weights,
                absolute_threshold=self.config.degeneracy_eigenvalue,
                relative_threshold=self.config.degeneracy_min_eigen_ratio,
            )
            minimum_translation_eigenvalue = float(
                translation_eigenvalues[0]
            )
            largest = float(max(translation_eigenvalues[-1], 1.0e-12))
            information_condition = largest / max(
                minimum_translation_eigenvalue, 1.0e-12
            )
            if self.config.sonar_observable_subspace_projection:
                observable_rank = (
                    rotation_observable_rank + translation_observable_rank
                )
                if observable_rank < self.config.sonar_min_observable_pose_rank:
                    return rejected_result(
                        "insufficient_observable_rank",
                        correspondences=last_count,
                        iterations=iterations,
                        rmse_before_value=rmse_before,
                        minimum_eigenvalue=minimum_translation_eigenvalue,
                        condition=information_condition,
                        rotation_rank=rotation_observable_rank,
                        translation_rank=translation_observable_rank,
                        correction_value=correction,
                    )
                jacobian = jacobian.copy()
                jacobian[:, 0:3] = (
                    jacobian[:, 0:3] @ rotation_projector
                )
                jacobian[:, 3:6] = (
                    jacobian[:, 3:6] @ translation_projector
                )
                adaptive_weight = 1.0
            elif self.config.sonar_continuous_observability_weighting:
                rotation_transform, _, rotation_response = (
                    self._continuous_observability_transform(
                        jacobian[:, 0:3],
                        weights,
                        absolute_knee=(
                            self.config.degeneracy_rotation_eigenvalue
                        ),
                        relative_knee=self.config.degeneracy_min_eigen_ratio,
                    )
                )
                translation_transform, _, translation_response = (
                    self._continuous_observability_transform(
                        jacobian[:, 3:6],
                        weights,
                        absolute_knee=self.config.degeneracy_eigenvalue,
                        relative_knee=self.config.degeneracy_min_eigen_ratio,
                    )
                )
                jacobian = jacobian.copy()
                jacobian[:, 0:3] = (
                    jacobian[:, 0:3] @ rotation_transform
                )
                jacobian[:, 3:6] = (
                    jacobian[:, 3:6] @ translation_transform
                )
                adaptive_weight = float(
                    min(
                        np.min(rotation_response),
                        np.min(translation_response),
                    )
                )
            else:
                absolute_observability = (
                    minimum_translation_eigenvalue
                    / max(self.config.degeneracy_eigenvalue, 1.0e-12)
                )
                relative_observability = (
                    minimum_translation_eigenvalue
                    / largest
                    / max(self.config.degeneracy_min_eigen_ratio, 1.0e-12)
                )
                adaptive_weight = float(
                    np.clip(
                        min(absolute_observability, relative_observability),
                        self.config.degeneracy_min_weight,
                        1.0,
                    )
                )
                weights *= adaptive_weight

            # Observability is a property of the complete spatial patch.  The
            # overlap correction belongs to the likelihood strength after
            # that geometry has been analysed; applying it before the smooth
            # spectral response would attenuate the same evidence twice.
            weights *= information_scale

            acoustic_information = (
                jacobian.T @ (weights[:, None] * jacobian)
            )
            acoustic_gradient = jacobian.T @ (weights * residual)
            system = prior_information + acoustic_information
            right_hand_side = (
                -prior_information @ correction
                - acoustic_gradient
            )
            try:
                step = np.linalg.solve(system, right_hand_side)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(
                    system, right_hand_side, rcond=1.0e-10
                )[0]

            max_rotation = np.deg2rad(
                self.config.sonar_max_rotation_step_deg
            )
            rotation_norm = float(np.linalg.norm(step[0:3]))
            if rotation_norm > max_rotation:
                step[0:3] *= max_rotation / rotation_norm
            translation_norm = float(np.linalg.norm(step[3:6]))
            if translation_norm > self.config.sonar_max_translation_step:
                step[3:6] *= (
                    self.config.sonar_max_translation_step / translation_norm
                )
            for state_slice, maximum in (
                (slice(6, 9), self.config.sonar_max_velocity_step),
                (slice(9, 12), self.config.sonar_max_gyro_bias_step),
                (slice(12, 15), self.config.sonar_max_accel_bias_step),
            ):
                block_norm = float(np.linalg.norm(step[state_slice]))
                if block_norm > maximum:
                    step[state_slice] *= maximum / block_norm
            correction += step
            iterations = iteration + 1
            if (
                np.linalg.norm(step[0:3])
                < self.config.sonar_convergence_rotation
                and np.linalg.norm(step[3:6])
                < self.config.sonar_convergence_translation
            ):
                break

        final_state = self._state_with_error(base, correction)
        final_residual, final_jacobian, final_variance = self._sonar_linearization(
            final_state, points, surfel_map
        )
        if final_residual.size < self.config.sonar_min_correspondences:
            return rejected_result(
                "insufficient_final_correspondences",
                correspondences=int(final_residual.size),
                iterations=iterations,
                rmse_before_value=rmse_before,
                minimum_eigenvalue=minimum_translation_eigenvalue,
                condition=information_condition,
                rotation_rank=rotation_observable_rank,
                translation_rank=translation_observable_rank,
                correction_value=correction,
            )
        rmse_after = float(np.sqrt(np.mean(final_residual**2)))
        total_rotation = float(np.linalg.norm(correction[0:3]))
        total_translation = float(np.linalg.norm(correction[3:6]))
        rejection_reason = None
        if not np.isfinite(rmse_after):
            rejection_reason = "non_finite_residual"
        elif rmse_after > rmse_before + 1.0e-4:
            rejection_reason = "residual_not_improved"
        elif total_rotation > 2.0 * np.deg2rad(
            self.config.sonar_max_rotation_step_deg
        ):
            rejection_reason = "excessive_rotation_correction"
        elif total_translation > 2.0 * self.config.sonar_max_translation_step:
            rejection_reason = "excessive_translation_correction"
        if rejection_reason is not None:
            return rejected_result(
                rejection_reason,
                correspondences=int(final_residual.size),
                iterations=iterations,
                rmse_before_value=rmse_before,
                rmse_after_value=rmse_after,
                minimum_eigenvalue=minimum_translation_eigenvalue,
                condition=information_condition,
                rotation_rank=rotation_observable_rank,
                translation_rank=translation_observable_rank,
                correction_value=correction,
            )

        normalized = np.abs(final_residual) / self.config.sonar_huber_delta
        robust_weight = np.ones_like(normalized)
        outside = normalized > 1.0
        robust_weight[outside] = 1.0 / normalized[outside]
        final_weights = robust_weight / final_variance
        if self.config.sonar_observable_subspace_projection:
            rotation_projector, _, rotation_observable_rank = (
                self._observable_projector(
                    final_jacobian[:, 0:3],
                    final_weights,
                    absolute_threshold=(
                        self.config.degeneracy_rotation_eigenvalue
                    ),
                    relative_threshold=self.config.degeneracy_min_eigen_ratio,
                )
            )
            (
                translation_projector,
                translation_eigenvalues,
                translation_observable_rank,
            ) = self._observable_projector(
                final_jacobian[:, 3:6],
                final_weights,
                absolute_threshold=self.config.degeneracy_eigenvalue,
                relative_threshold=self.config.degeneracy_min_eigen_ratio,
            )
            minimum_translation_eigenvalue = float(
                translation_eigenvalues[0]
            )
            largest = float(max(translation_eigenvalues[-1], 1.0e-12))
            information_condition = largest / max(
                minimum_translation_eigenvalue, 1.0e-12
            )
            final_jacobian = final_jacobian.copy()
            final_jacobian[:, 0:3] = (
                final_jacobian[:, 0:3] @ rotation_projector
            )
            final_jacobian[:, 3:6] = (
                final_jacobian[:, 3:6] @ translation_projector
            )
        elif self.config.sonar_continuous_observability_weighting:
            rotation_transform, _, rotation_response = (
                self._continuous_observability_transform(
                    final_jacobian[:, 0:3],
                    final_weights,
                    absolute_knee=(
                        self.config.degeneracy_rotation_eigenvalue
                    ),
                    relative_knee=self.config.degeneracy_min_eigen_ratio,
                )
            )
            translation_transform, _, translation_response = (
                self._continuous_observability_transform(
                    final_jacobian[:, 3:6],
                    final_weights,
                    absolute_knee=self.config.degeneracy_eigenvalue,
                    relative_knee=self.config.degeneracy_min_eigen_ratio,
                )
            )
            final_jacobian = final_jacobian.copy()
            final_jacobian[:, 0:3] = (
                final_jacobian[:, 0:3] @ rotation_transform
            )
            final_jacobian[:, 3:6] = (
                final_jacobian[:, 3:6] @ translation_transform
            )
            adaptive_weight = float(
                min(
                    np.min(rotation_response),
                    np.min(translation_response),
                )
            )
        else:
            final_weights *= adaptive_weight
        final_weights *= information_scale
        final_information = prior_information + final_jacobian.T @ (
            final_weights[:, None] * final_jacobian
        )
        try:
            final_covariance = np.linalg.inv(final_information)
        except np.linalg.LinAlgError:
            final_covariance = np.linalg.pinv(final_information, rcond=1.0e-10)
        reset = self._error_reset_jacobian(correction)
        final_state.covariance = self._regularize_covariance(
            reset @ final_covariance @ reset.T
        )
        self.state = final_state

        return SonarUpdateResult(
            accepted=True,
            correspondences=int(last_count),
            inliers=int(
                np.count_nonzero(
                    np.abs(final_residual)
                    <= 2.5 * self.config.sonar_huber_delta
                )
            ),
            iterations=iterations,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            minimum_translation_eigenvalue=minimum_translation_eigenvalue,
            information_condition=information_condition,
            adaptive_weight=adaptive_weight,
            rotation_observable_rank=rotation_observable_rank,
            translation_observable_rank=translation_observable_rank,
            rotation_correction_deg=float(
                np.rad2deg(np.linalg.norm(correction[0:3]))
            ),
            translation_correction=float(np.linalg.norm(correction[3:6])),
            rejection_reason=None,
            information_scale=information_scale,
        )

    def _intensity_linearization(
        self,
        state: InertialState,
        points_sensor: np.ndarray,
        measured_intensity: np.ndarray,
        intensity_map: AcousticIntensityMap,
        maximum_observation_id: int | None = None,
        anchor_id: int | None = None,
        affine_group_ids: np.ndarray | None = None,
        candidate_cache: dict | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        points_body = transform_points(
            self.sensor_rotation_body,
            self.sensor_translation_body,
            points_sensor,
        )
        points_world = transform_points(
            state.rotation, state.position, points_body
        )
        measurements: list[float] = []
        predictions: list[float] = []
        jacobians: list[np.ndarray] = []
        pose2_jacobians: list[np.ndarray] = []
        variances: list[float] = []
        reference_observation_ids: list[int] = []
        anchor_ids: list[int] = []
        if affine_group_ids is None:
            input_group_ids = np.zeros(
                measured_intensity.size, dtype=np.int64
            )
        else:
            input_group_ids = np.asarray(
                affine_group_ids, dtype=np.int64
            ).reshape(-1)
            if input_group_ids.size != measured_intensity.size:
                raise ValueError(
                    "affine group IDs must match measured intensities"
                )
        matched_group_ids: list[int] = []
        heading = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
        # The ESIKF uses a right/body-frame attitude perturbation. A unit
        # world-yaw perturbation therefore has body coordinates R^T e_z.
        # Projecting through this axis avoids treating body dtheta_z as world
        # yaw when the AUV has non-zero roll or pitch.
        world_yaw_axis_body = state.rotation.T @ np.array(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
        if anchor_id is None:
            intensity_samples = intensity_map.query_batch(
                points_world,
                heading_rad=heading,
                maximum_observation_id=maximum_observation_id,
                _candidate_cache=candidate_cache,
            )
        else:
            intensity_samples = intensity_map.query_fixed_anchor_batch(
                points_world,
                heading_rad=heading,
                maximum_observation_id=maximum_observation_id,
                anchor_id=anchor_id,
                _candidate_cache=candidate_cache,
            )
        for point_body, point_world, measured, group_id, sample in zip(
            points_body,
            points_world,
            measured_intensity,
            input_group_ids,
            intensity_samples,
        ):
            if sample is None:
                continue
            gradient = np.array(
                [sample.gradient_xy[0], sample.gradient_xy[1], 0.0],
                dtype=np.float64,
            )
            point_jacobian = np.zeros(15, dtype=np.float64)
            rotation_jacobian = gradient @ (
                -state.rotation @ skew(point_body)
            )
            yaw_jacobian = float(
                rotation_jacobian @ world_yaw_axis_body
            )
            if self.config.intensity_planar_only:
                point_jacobian[0:3] = (
                    yaw_jacobian * world_yaw_axis_body
                )
                point_jacobian[3:5] = gradient[:2]
            else:
                point_jacobian[0:3] = rotation_jacobian
                point_jacobian[3:6] = gradient
            measurements.append(float(measured))
            predictions.append(sample.intensity)
            jacobians.append(point_jacobian)
            pose2_jacobians.append(
                np.array(
                    [gradient[0], gradient[1], yaw_jacobian],
                    dtype=np.float64,
                )
            )
            photometric_variance = (
                self.config.intensity_noise_std**2 + sample.variance
            )
            if np.isfinite(sample.surface_z):
                surface_scale_squared = (
                    self.config.intensity_surface_consistency_std**2
                    + max(0.0, float(sample.surface_variance))
                )
                elevation_residual = float(point_world[2] - sample.surface_z)
                surface_weight = surface_scale_squared / (
                    surface_scale_squared + elevation_residual**2
                )
                photometric_variance /= max(surface_weight, 1.0e-12)
            variances.append(photometric_variance)
            reference_observation_ids.append(
                int(sample.reference_observation_id)
            )
            anchor_ids.append(int(sample.anchor_id))
            matched_group_ids.append(int(group_id))
        if not measurements:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty((0, 15), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        return (
            np.asarray(measurements, dtype=np.float64),
            np.asarray(predictions, dtype=np.float64),
            np.asarray(jacobians, dtype=np.float64),
            np.asarray(pose2_jacobians, dtype=np.float64),
            np.asarray(variances, dtype=np.float64),
            np.asarray(reference_observation_ids, dtype=np.int64),
            np.asarray(anchor_ids, dtype=np.int64),
            np.asarray(matched_group_ids, dtype=np.int64),
        )

    @staticmethod
    def _affine_intensity_residual(
        measured: np.ndarray,
        predicted: np.ndarray,
        jacobian: np.ndarray,
        pose2_jacobian: np.ndarray,
        variance: np.ndarray,
        huber_delta: float = 0.14,
        affine_group_ids: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        np.ndarray,
    ]:
        """Eliminate per-strip gain/offset from residual and information.

        Merely fitting gain and offset and then retaining the original pose
        Jacobian overstates image information.  The two nuisance directions
        are projected out in the same weighted metric (the Schur complement
        of their joint normal equation) before either the ESIKF or pose graph
        sees the residual.
        """

        if affine_group_ids is None:
            group_inverse = np.zeros(predicted.size, dtype=np.int64)
        else:
            supplied_groups = np.asarray(
                affine_group_ids, dtype=np.int64
            ).reshape(-1)
            if supplied_groups.size != predicted.size:
                raise ValueError(
                    "affine group IDs must match intensity residuals"
                )
            _, group_inverse = np.unique(
                supplied_groups, return_inverse=True
            )
        group_count = int(np.max(group_inverse, initial=-1) + 1)
        design = np.zeros(
            (predicted.size, 2 * group_count), dtype=np.float64
        )
        rows = np.arange(predicted.size, dtype=np.int64)
        design[rows, 2 * group_inverse] = predicted
        design[rows, 2 * group_inverse + 1] = 1.0
        inverse_variance = 1.0 / np.maximum(variance, 1.0e-8)
        weights = inverse_variance.copy()
        gains = np.ones(group_count, dtype=np.float64)
        offsets = np.zeros(group_count, dtype=np.float64)

        def fit_affine(metric: np.ndarray):
            normal_matrix = design.T @ (metric[:, None] * design)
            normal_matrix += np.eye(2 * group_count) * 1.0e-8
            right_hand_side = design.T @ (metric * measured)
            try:
                coefficients = np.linalg.solve(
                    normal_matrix, right_hand_side
                )
            except np.linalg.LinAlgError:
                coefficients = np.zeros(2 * group_count, dtype=np.float64)
                coefficients[0::2] = 1.0
            fitted_gains = np.clip(coefficients[0::2], 0.45, 2.2)
            fitted_offsets = np.clip(coefficients[1::2], -0.45, 0.45)
            fitted = (
                fitted_gains[group_inverse] * predicted
                + fitted_offsets[group_inverse]
            )
            return (
                normal_matrix,
                fitted_gains,
                fitted_offsets,
                measured - fitted,
            )

        # Robust affine exposure estimation and nuisance elimination must use
        # one metric. Projecting with inverse-variance weights and only later
        # applying Huber weights destroys Schur-complement monotonicity and
        # can spuriously increase pose information in an outlier direction.
        for _ in range(4):
            normal, gains, offsets, residual = fit_affine(weights)
            normalized = np.abs(residual) / max(float(huber_delta), 1.0e-8)
            robust_weight = np.ones_like(normalized)
            outside = normalized > 1.0
            robust_weight[outside] = 1.0 / normalized[outside]
            new_weights = inverse_variance * robust_weight
            if np.max(np.abs(new_weights - weights)) <= (
                1.0e-5 * max(float(np.max(weights)), 1.0)
            ):
                weights = new_weights
                break
            weights = new_weights

        # One final affine solve makes gain/offset, the returned residual, and
        # the nuisance projector use exactly the returned robust metric.
        normal, gains, offsets, residual = fit_affine(weights)
        gain_per_sample = gains[group_inverse]
        scaled_jacobian = gain_per_sample[:, None] * jacobian
        scaled_pose2_jacobian = (
            gain_per_sample[:, None] * pose2_jacobian
        )
        nuisance_to_state = np.linalg.solve(
            normal,
            design.T @ (weights[:, None] * scaled_jacobian),
        )
        nuisance_to_pose2 = np.linalg.solve(
            normal,
            design.T @ (weights[:, None] * scaled_pose2_jacobian),
        )
        projected_jacobian = scaled_jacobian - design @ nuisance_to_state
        projected_pose2_jacobian = (
            scaled_pose2_jacobian - design @ nuisance_to_pose2
        )
        return (
            residual,
            projected_jacobian,
            projected_pose2_jacobian,
            float(np.median(gains)),
            float(np.median(offsets)),
            weights,
        )

    @staticmethod
    def _normalized_correlation(
        measured: np.ndarray, predicted: np.ndarray
    ) -> float:
        """Return zero-mean normalized correlation without hiding flat data."""

        first = np.asarray(measured, dtype=np.float64).reshape(-1)
        second = np.asarray(predicted, dtype=np.float64).reshape(-1)
        if first.size < 3 or second.size != first.size:
            return 0.0
        first = first - np.mean(first)
        second = second - np.mean(second)
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator < 1.0e-10:
            return 0.0
        return float(np.clip((first @ second) / denominator, -1.0, 1.0))

    def _intensity_planar_rank(
        self, pose2_jacobian: np.ndarray, weights: np.ndarray
    ) -> int:
        """Measure independent yaw/x/y directions in a side-scan strip."""

        if pose2_jacobian.shape[0] == 0:
            return 0
        # Scale yaw by a representative side-scan lever arm so radians and
        # metres have comparable numerical meaning in the SVD.
        planar = np.asarray(pose2_jacobian, dtype=np.float64).copy()
        planar[:, 2] /= 20.0
        whitened = np.sqrt(np.maximum(weights, 0.0))[:, None] * planar
        try:
            singular_values = np.linalg.svd(
                whitened, compute_uv=False
            )
        except np.linalg.LinAlgError:
            return 0
        if singular_values.size == 0 or singular_values[0] <= 1.0e-10:
            return 0
        threshold = max(
            1.0e-6,
            float(self.config.intensity_min_singular_ratio)
            * float(singular_values[0]),
        )
        return int(np.count_nonzero(singular_values >= threshold))

    def update_intensity(
        self,
        points_sensor: np.ndarray,
        intensities: np.ndarray,
        intensity_map: AcousticIntensityMap,
        *,
        maximum_observation_id: int | None = None,
        initial_state: InertialState | None = None,
        fixed_anchor_id: int | None = None,
        commit_state: bool = True,
        affine_group_ids: np.ndarray | None = None,
    ) -> IntensityUpdateResult:
        """Perform a FAST-LIVO2-style direct side-scan intensity update.

        ``initial_state``/``fixed_anchor_id`` support a graph-seeded,
        non-committing alignment on a verified revisit.  In that mode the
        returned measurement-only factor refines the global graph without
        jumping the live local inertial state.
        """

        points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
        values = np.asarray(intensities, dtype=np.float64).reshape(-1)
        if values.size != points.shape[0]:
            raise ValueError("intensities must match side-scan support points")
        if affine_group_ids is None:
            groups = np.zeros(points.shape[0], dtype=np.int64)
        else:
            groups = np.asarray(
                affine_group_ids, dtype=np.int64
            ).reshape(-1)
            if groups.size != points.shape[0]:
                raise ValueError(
                    "affine group IDs must match side-scan support points"
                )
        valid = np.isfinite(points).all(axis=1) & np.isfinite(values)
        points, values, groups = points[valid], values[valid], groups[valid]
        if points.shape[0] > self.config.intensity_max_points:
            indices = np.linspace(
                0,
                points.shape[0] - 1,
                self.config.intensity_max_points,
                dtype=np.int64,
            )
            points, values, groups = (
                points[indices],
                values[indices],
                groups[indices],
            )
        def rejected_result(
            reason: str,
            *,
            correspondences: int = 0,
            iterations: int = 0,
            rmse_before_value: float = np.nan,
            rmse_after_value: float = np.nan,
            yaw_information_value: float = 0.0,
            overlap_ratio_value: float = 0.0,
            correlation_value: float = 0.0,
            planar_rank_value: int = 0,
            gain_value: float = 1.0,
            offset_value: float = 0.0,
            correction_value: np.ndarray | None = None,
        ) -> IntensityUpdateResult:
            correction_value = (
                np.zeros(15, dtype=np.float64)
                if correction_value is None
                else np.asarray(correction_value, dtype=np.float64).reshape(15)
            )
            return IntensityUpdateResult(
                accepted=False,
                correspondences=int(correspondences),
                inliers=0,
                iterations=int(iterations),
                rmse_before=float(rmse_before_value),
                rmse_after=float(rmse_after_value),
                yaw_information=float(yaw_information_value),
                overlap_ratio=float(overlap_ratio_value),
                normalized_correlation=float(correlation_value),
                planar_observable_rank=int(planar_rank_value),
                gain=float(gain_value),
                offset=float(offset_value),
                rotation_correction_deg=float(
                    np.rad2deg(np.linalg.norm(correction_value[0:3]))
                ),
                translation_correction=float(
                    np.linalg.norm(correction_value[3:6])
                ),
                rejection_reason=reason,
            )
        if points.shape[0] < self.config.intensity_min_correspondences:
            return rejected_result("insufficient_input_points")

        base = (
            self.state.copy()
            if initial_state is None
            else initial_state.copy()
        )
        prior_covariance = self._regularize_covariance(base.covariance)
        prior_information = np.linalg.pinv(prior_covariance, rcond=1.0e-10)
        correction = np.zeros(15, dtype=np.float64)
        rmse_before = np.nan
        yaw_information = 0.0
        overlap_ratio = 0.0
        normalized_correlation = 0.0
        planar_observable_rank = 0
        iterations = 0
        count = 0
        gain, offset = 1.0, 0.0
        selected_anchor_id: int | None = (
            None if fixed_anchor_id is None else int(fixed_anchor_id)
        )
        # The intensity map is immutable throughout this iterated update.
        # Reuse only exact broad-phase cell memberships; every sub-cell range
        # gate, local plane fit, residual, and Jacobian is still recomputed at
        # each candidate state.
        intensity_candidate_cache: dict = {}

        for iteration in range(self.config.intensity_max_iterations):
            candidate = self._state_with_error(base, correction)
            (
                measured,
                predicted,
                jacobian,
                pose2_jacobian,
                variance,
                _reference_observation_ids,
                anchor_ids,
                matched_group_ids,
            ) = self._intensity_linearization(
                candidate,
                points,
                values,
                intensity_map,
                maximum_observation_id,
                selected_anchor_id,
                groups,
                intensity_candidate_cache,
            )
            if selected_anchor_id is None and anchor_ids.size:
                unique_anchors, anchor_counts = np.unique(
                    anchor_ids, return_counts=True
                )
                selected_anchor_id = int(
                    unique_anchors[np.argmax(anchor_counts)]
                )
                (
                    measured,
                    predicted,
                    jacobian,
                    pose2_jacobian,
                    variance,
                    _reference_observation_ids,
                    anchor_ids,
                    matched_group_ids,
                ) = self._intensity_linearization(
                    candidate,
                    points,
                    values,
                    intensity_map,
                    maximum_observation_id,
                    selected_anchor_id,
                    groups,
                    intensity_candidate_cache,
                )
            count = measured.size
            overlap_ratio = float(count / max(1, points.shape[0]))
            if count < self.config.intensity_min_correspondences:
                return rejected_result(
                    "insufficient_correspondences",
                    correspondences=count,
                    iterations=iterations,
                    rmse_before_value=rmse_before,
                    yaw_information_value=yaw_information,
                    overlap_ratio_value=overlap_ratio,
                    correlation_value=normalized_correlation,
                    planar_rank_value=planar_observable_rank,
                    gain_value=gain,
                    offset_value=offset,
                    correction_value=correction,
                )
            if overlap_ratio < self.config.intensity_min_overlap_ratio:
                return rejected_result(
                    "insufficient_overlap",
                    correspondences=count,
                    iterations=iterations,
                    rmse_before_value=rmse_before,
                    yaw_information_value=yaw_information,
                    overlap_ratio_value=overlap_ratio,
                    correlation_value=normalized_correlation,
                    planar_rank_value=planar_observable_rank,
                    gain_value=gain,
                    offset_value=offset,
                    correction_value=correction,
                )
            (
                residual,
                jacobian,
                pose2_jacobian,
                gain,
                offset,
                weights,
            ) = self._affine_intensity_residual(
                measured,
                predicted,
                jacobian,
                pose2_jacobian,
                variance,
                self.config.intensity_huber_delta,
                matched_group_ids,
            )
            weights *= min(
                1.0,
                float(self.config.intensity_effective_independent_samples)
                / max(1, weights.size),
            )
            normalized_correlation = self._normalized_correlation(
                measured, predicted
            )
            if iteration == 0:
                rmse_before = float(np.sqrt(np.mean(residual**2)))
            planar_observable_rank = self._intensity_planar_rank(
                pose2_jacobian, weights
            )
            if (
                planar_observable_rank
                < self.config.intensity_min_planar_rank
            ):
                return rejected_result(
                    "insufficient_planar_rank",
                    correspondences=count,
                    iterations=iterations,
                    rmse_before_value=rmse_before,
                    yaw_information_value=yaw_information,
                    overlap_ratio_value=overlap_ratio,
                    correlation_value=normalized_correlation,
                    planar_rank_value=planar_observable_rank,
                    gain_value=gain,
                    offset_value=offset,
                    correction_value=correction,
                )
            yaw_information = float(
                np.sum(
                    weights
                    * pose2_jacobian[:, 2]
                    * pose2_jacobian[:, 2]
                )
            )
            if yaw_information < self.config.intensity_min_yaw_information:
                return rejected_result(
                    "insufficient_yaw_information",
                    correspondences=count,
                    iterations=iterations,
                    rmse_before_value=rmse_before,
                    yaw_information_value=yaw_information,
                    overlap_ratio_value=overlap_ratio,
                    correlation_value=normalized_correlation,
                    planar_rank_value=planar_observable_rank,
                    gain_value=gain,
                    offset_value=offset,
                    correction_value=correction,
                )
            information = jacobian.T @ (weights[:, None] * jacobian)
            gradient = jacobian.T @ (weights * residual)
            system = prior_information + information
            right_hand_side = -prior_information @ correction + gradient
            try:
                step = np.linalg.solve(system, right_hand_side)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(
                    system, right_hand_side, rcond=1.0e-10
                )[0]

            max_rotation = np.deg2rad(
                self.config.intensity_max_rotation_step_deg
            )
            rotation_norm = float(np.linalg.norm(step[0:3]))
            if rotation_norm > max_rotation:
                step[0:3] *= max_rotation / rotation_norm
            translation_norm = float(np.linalg.norm(step[3:6]))
            if translation_norm > self.config.intensity_max_translation_step:
                step[3:6] *= (
                    self.config.intensity_max_translation_step / translation_norm
                )
            for state_slice, maximum in (
                (slice(6, 9), self.config.sonar_max_velocity_step),
                (slice(9, 12), self.config.sonar_max_gyro_bias_step),
                (slice(12, 15), self.config.sonar_max_accel_bias_step),
            ):
                block_norm = float(np.linalg.norm(step[state_slice]))
                if block_norm > maximum:
                    step[state_slice] *= maximum / block_norm
            correction += step
            iterations = iteration + 1
            if (
                np.linalg.norm(step[0:3])
                < self.config.sonar_convergence_rotation
                and np.linalg.norm(step[3:6])
                < self.config.sonar_convergence_translation
            ):
                break

        final_state = self._state_with_error(base, correction)
        (
            measured,
            predicted,
            final_jacobian,
            final_pose2_jacobian,
            final_variance,
            final_reference_observation_ids,
            final_anchor_ids,
            final_group_ids,
        ) = (
            self._intensity_linearization(
                final_state,
                points,
                values,
                intensity_map,
                maximum_observation_id,
                selected_anchor_id,
                groups,
                intensity_candidate_cache,
            )
        )
        if measured.size < self.config.intensity_min_correspondences:
            return rejected_result(
                "insufficient_final_correspondences",
                correspondences=int(measured.size),
                iterations=iterations,
                rmse_before_value=rmse_before,
                yaw_information_value=yaw_information,
                overlap_ratio_value=float(
                    measured.size / max(1, points.shape[0])
                ),
                correlation_value=normalized_correlation,
                planar_rank_value=planar_observable_rank,
                gain_value=gain,
                offset_value=offset,
                correction_value=correction,
            )
        (
            final_residual,
            final_jacobian,
            final_pose2_jacobian,
            gain,
            offset,
            final_weights,
        ) = self._affine_intensity_residual(
                measured,
                predicted,
                final_jacobian,
                final_pose2_jacobian,
                final_variance,
                self.config.intensity_huber_delta,
                final_group_ids,
        )
        final_weights *= min(
            1.0,
            float(self.config.intensity_effective_independent_samples)
            / max(1, final_weights.size),
        )
        overlap_ratio = float(measured.size / max(1, points.shape[0]))
        normalized_correlation = self._normalized_correlation(
            measured, predicted
        )
        rmse_after = float(np.sqrt(np.mean(final_residual**2)))
        total_rotation = float(np.linalg.norm(correction[0:3]))
        total_translation = float(np.linalg.norm(correction[3:6]))
        planar_observable_rank = self._intensity_planar_rank(
            final_pose2_jacobian, final_weights
        )
        yaw_information = float(
            np.sum(
                final_weights
                * final_pose2_jacobian[:, 2]
                * final_pose2_jacobian[:, 2]
            )
        )
        inlier_ratio = float(
            np.count_nonzero(
                np.abs(final_residual)
                <= 2.5 * self.config.intensity_huber_delta
            )
            / max(1, final_residual.size)
        )
        rejection_reason = None
        if not np.isfinite(rmse_after):
            rejection_reason = "non_finite_residual"
        elif rmse_after > rmse_before * 1.03 + 2.0e-3:
            rejection_reason = "residual_not_improved"
        elif overlap_ratio < self.config.intensity_min_overlap_ratio:
            rejection_reason = "insufficient_overlap"
        elif (
            normalized_correlation
            < self.config.intensity_min_normalized_correlation
        ):
            rejection_reason = "insufficient_correlation"
        elif inlier_ratio < self.config.intensity_min_inlier_ratio:
            rejection_reason = "insufficient_inlier_ratio"
        elif planar_observable_rank < self.config.intensity_min_planar_rank:
            rejection_reason = "insufficient_planar_rank"
        elif yaw_information < self.config.intensity_min_yaw_information:
            rejection_reason = "insufficient_yaw_information"
        elif total_rotation > 2.0 * np.deg2rad(
            self.config.intensity_max_rotation_step_deg
        ):
            rejection_reason = "excessive_rotation_correction"
        elif total_translation > 2.0 * self.config.intensity_max_translation_step:
            rejection_reason = "excessive_translation_correction"
        if rejection_reason is not None:
            return rejected_result(
                rejection_reason,
                correspondences=int(measured.size),
                iterations=iterations,
                rmse_before_value=rmse_before,
                rmse_after_value=rmse_after,
                yaw_information_value=yaw_information,
                overlap_ratio_value=overlap_ratio,
                correlation_value=normalized_correlation,
                planar_rank_value=planar_observable_rank,
                gain_value=gain,
                offset_value=offset,
                correction_value=correction,
            )

        final_information = prior_information + final_jacobian.T @ (
            final_weights[:, None] * final_jacobian
        )
        try:
            final_covariance = np.linalg.inv(final_information)
        except np.linalg.LinAlgError:
            final_covariance = np.linalg.pinv(final_information, rcond=1.0e-10)
        reset = self._error_reset_jacobian(correction)
        final_state.covariance = self._regularize_covariance(
            reset @ final_covariance @ reset.T
        )
        # Solve the measurement-only projected normal equation in coordinates
        # where one radian of yaw is represented by a 20 m side-scan lever
        # arm. This keeps the eigenspace test dimensionally meaningful.
        pose_scale = np.diag([1.0, 1.0, 1.0 / 20.0])
        scaled_jacobian = final_pose2_jacobian @ pose_scale
        scaled_information = scaled_jacobian.T @ (
            final_weights[:, None] * scaled_jacobian
        )
        scaled_gradient = scaled_jacobian.T @ (
            final_weights * final_residual
        )
        scaled_information = 0.5 * (
            scaled_information + scaled_information.T
        )
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(
                scaled_information
            )
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(3, dtype=np.float64)
            eigenvectors = np.eye(3, dtype=np.float64)
        threshold = max(
            1.0e-8,
            float(self.config.intensity_min_singular_ratio) ** 2
            * max(float(eigenvalues[-1]), 0.0),
        )
        observable = eigenvalues >= threshold
        photo_delta_scaled = np.zeros(3, dtype=np.float64)
        projected_scaled_information = np.zeros((3, 3), dtype=np.float64)
        if np.any(observable):
            projected_gradient = eigenvectors.T @ scaled_gradient
            photo_delta_scaled = eigenvectors[:, observable] @ (
                projected_gradient[observable] / eigenvalues[observable]
            )
            observable_vectors = eigenvectors[:, observable]
            projected_scaled_information = (
                observable_vectors * eigenvalues[observable]
            ) @ observable_vectors.T
        photometric_delta = pose_scale @ photo_delta_scaled
        # Convert the *same projected Hessian* back to [m, m, rad]
        # coordinates. Returning the unprojected Hessian here would give the
        # graph information in directions that were intentionally excluded
        # from the image-only local quadratic solve.
        inverse_pose_scale = np.diag([1.0, 1.0, 20.0])
        information_pose2 = (
            inverse_pose_scale
            @ projected_scaled_information
            @ inverse_pose_scale
        )
        information_pose2 = 0.5 * (
            information_pose2 + information_pose2.T
        )
        graph_measurement_valid = bool(
            selected_anchor_id is not None
            and selected_anchor_id >= 0
            and final_anchor_ids.size > 0
            and np.all(final_anchor_ids == selected_anchor_id)
            and np.linalg.norm(photometric_delta[:2])
            <= self.config.intensity_graph_max_translation_m
            and abs(photometric_delta[2])
            <= np.deg2rad(self.config.intensity_graph_max_yaw_deg)
        )
        final_yaw = float(
            np.arctan2(final_state.rotation[1, 0], final_state.rotation[0, 0])
        )
        photometric_pose2 = np.array(
            [
                final_state.position[0] + photometric_delta[0],
                final_state.position[1] + photometric_delta[1],
                np.arctan2(
                    np.sin(final_yaw + photometric_delta[2]),
                    np.cos(final_yaw + photometric_delta[2]),
                ),
            ],
            dtype=np.float64,
        )
        if graph_measurement_valid:
            # Validate the local quadratic image-only step against a fresh
            # map query. A Hessian at the ESIKF MAP point is not, by itself,
            # evidence that its unconstrained step remains valid after cell
            # associations change.
            photo_state = final_state.copy()
            photo_state.position[:2] += photometric_delta[:2]
            c_photo, s_photo = (
                np.cos(photometric_delta[2]),
                np.sin(photometric_delta[2]),
            )
            photo_state.rotation = np.array(
                [
                    [c_photo, -s_photo, 0.0],
                    [s_photo, c_photo, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ) @ photo_state.rotation
            (
                photo_measured,
                photo_predicted,
                photo_jacobian,
                photo_pose2_jacobian,
                photo_variance,
                photo_reference_ids,
                photo_anchor_ids,
                photo_group_ids,
            ) = self._intensity_linearization(
                photo_state,
                points,
                values,
                intensity_map,
                maximum_observation_id,
                selected_anchor_id,
                groups,
            )
            minimum_verified_count = max(
                self.config.intensity_min_correspondences,
                int(np.ceil(0.9 * measured.size)),
            )
            if photo_measured.size >= minimum_verified_count:
                (
                    photo_residual,
                    _photo_jacobian,
                    _photo_pose2_jacobian,
                    _photo_gain,
                    _photo_offset,
                    photo_weights,
                ) = self._affine_intensity_residual(
                    photo_measured,
                    photo_predicted,
                    photo_jacobian,
                    photo_pose2_jacobian,
                    photo_variance,
                    self.config.intensity_huber_delta,
                    photo_group_ids,
                )
                base_cost = float(
                    np.sum(final_weights * final_residual**2)
                    / max(np.sum(final_weights), 1.0e-12)
                )
                photo_cost = float(
                    np.sum(photo_weights * photo_residual**2)
                    / max(np.sum(photo_weights), 1.0e-12)
                )
                photo_correlation = self._normalized_correlation(
                    photo_measured, photo_predicted
                )
                photo_inlier_ratio = float(
                    np.mean(
                        np.abs(photo_residual)
                        <= 2.5 * self.config.intensity_huber_delta
                    )
                )
                graph_measurement_valid = bool(
                    np.all(photo_anchor_ids == selected_anchor_id)
                    and photo_measured.size / max(1, points.shape[0])
                    >= self.config.intensity_min_overlap_ratio
                    and photo_correlation
                    >= self.config.intensity_min_normalized_correlation
                    and photo_inlier_ratio
                    >= self.config.intensity_min_inlier_ratio
                    and np.isfinite(photo_cost)
                    and photo_cost <= 1.01 * base_cost + 1.0e-10
                )
                if graph_measurement_valid:
                    final_reference_observation_ids = photo_reference_ids
            else:
                graph_measurement_valid = False
        reference_observation_id = int(
            np.max(final_reference_observation_ids)
        )
        if commit_state:
            self.state = final_state
        return IntensityUpdateResult(
            accepted=True,
            correspondences=int(measured.size),
            inliers=int(
                np.count_nonzero(
                    np.abs(final_residual)
                    <= 2.5 * self.config.intensity_huber_delta
                )
            ),
            iterations=iterations,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            yaw_information=yaw_information,
            overlap_ratio=overlap_ratio,
            normalized_correlation=normalized_correlation,
            planar_observable_rank=planar_observable_rank,
            gain=gain,
            offset=offset,
            rotation_correction_deg=float(np.rad2deg(total_rotation)),
            translation_correction=total_translation,
            rejection_reason=None,
            reference_observation_id=reference_observation_id,
            anchor_keyframe_index=(
                int(selected_anchor_id)
                if graph_measurement_valid
                else None
            ),
            photometric_pose2=photometric_pose2,
            information_pose2=(
                information_pose2
                if graph_measurement_valid
                else np.zeros((3, 3), dtype=np.float64)
            ),
        )
