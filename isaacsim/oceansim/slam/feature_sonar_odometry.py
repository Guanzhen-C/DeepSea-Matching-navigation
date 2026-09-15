"""Feature-based relative odometry for co-registered MBES/side-scan pings.

The V2 acoustic head publishes several simultaneous along-track rows on both
sides.  Consecutive physical pings therefore contain genuinely repeated
seabed observations even though a conventional, single-row side-scan line is
underdetermined.  This module uses the intensity image only to establish
correspondence; every accepted image feature is lifted back to the exact XYZ
return that generated its pixel before a planar rigid transform is estimated.

No prior terrain map is used here.  The correspondence fit retains a full
prediction-centred SE(3) transform and covariance for diagnostics.  In the
default, observability-aware policy only its along-track position projection is
fed to the local filter; cross-track, attitude and depth remain constrained by
the DVL/IMU/MBES branches.  This frontend is deliberately separate from the
absolute terrain matcher in the Global backend.  OpenCV is imported lazily so
the default NumPy-only direct frontend retains its historical dependency
contract.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np

from .math3d import project_to_so3, so3_exp, wrap_angle

if TYPE_CHECKING:
    from ..utils.acoustic_sonar import BathymetricSideScanPing


@dataclass
class FeatureSonarOdometryConfig:
    """Configuration for causal multi-ping feature odometry."""

    # One update is formed after this many distinct physical acquisitions.
    # With the documented 8 rows/ping, 4 pings at 5 Hz span about 2.1 m while
    # preserving the native 75% inter-ping footprint overlap.
    subframe_pings: int = 4
    maximum_features: int = 900
    association_method: str = "klt"
    state_update_policy: str = "along_track"
    descriptor_ratio: float = 0.90
    descriptor_maximum_distance: float = 1.15
    minimum_matches: int = 24
    effective_independent_matches: int = 24
    ransac_iterations: int = 256
    ransac_residual_m: float = 0.70
    along_track_residual_weight: float = 10.0
    vertical_residual_weight: float = 2.0
    association_along_track_std_m: float = 0.12
    association_cross_track_std_m: float = 0.15
    association_vertical_std_m: float = 0.50
    association_maximum_normalized_distance: float = 4.0
    association_geometry_weight: float = 0.20
    image_row_scale: int = 4
    # The raw V2 image has 12,500 samples/side but the simulated ray model has
    # only 256 independent angular returns/row.  A fixed metric feature raster
    # avoids asking SIFT to process thousands of empty interpolation cells;
    # raw data and all XYZ returns remain untouched.
    canonical_range_bins_per_side: int = 512
    minimum_pair_baseline_m: float = 1.0
    minimum_information_eigenvalue: float = 1.0e-3
    # KLT reliably observes the row-shift (vehicle-forward) component.  A
    # single eight-row ping does not independently support cross-track or yaw
    # corrections on smooth terrain, so those components remain with
    # DVL/IMU/MBES and carry deliberately weak acoustic information.
    position_noise_floor_m: float = 0.15
    cross_track_position_noise_floor_m: float = 5.0
    yaw_noise_floor_deg: float = 5.0
    klt_window_width: int = 25
    klt_window_height: int = 15
    klt_maximum_level: int = 1
    klt_forward_backward_limit_px: float = 1.0
    klt_lift_radius_px: float = 4.0
    prediction_correction_scale_m: float = 0.08
    anchor_covariance_scale: float = 1.5
    # The images are only 32 x 1024 pixels, so OpenCV's CPU GFTT/KLT kernels
    # are already faster than a host/device round trip.  The expensive stage is
    # the 256-hypothesis geometric consensus.  ``cuda`` batches those
    # hypotheses in one PyTorch CUDA launch and retains the double-precision
    # CPU consensus refinement.  ``auto`` may fall back to CPU if CUDA is not
    # available or later fails; explicit ``cuda`` fails loudly instead.
    compute_backend: str = "cpu"


@dataclass(frozen=True)
class FeatureOdometryResult:
    """Result of one physical ping entering the feature frontend."""

    pair_accepted: bool
    subframe_ready: bool
    rejection_reason: str | None
    descriptor_matches: int
    inliers: int
    inlier_ratio: float
    pair_rmse_m: float
    subframe_ping_count: int
    relative_pose_sensor: np.ndarray | None
    relative_covariance: np.ndarray | None
    measured_position_xy_world: np.ndarray | None
    measured_yaw_world: float | None
    measurement_covariance_world: np.ndarray | None
    measured_along_track_axis_world: np.ndarray | None = None
    measured_along_track_projection_m: float | None = None


@dataclass(frozen=True)
class FeatureOdometryDiagnostics:
    frames: int
    pair_attempts: int
    accepted_pairs: int
    rejected_pairs: int
    completed_subframes: int
    rejection_reasons: dict[str, int]
    mean_pair_inlier_ratio: float
    mean_pair_rmse_m: float
    requested_compute_backend: str
    active_compute_backend: str
    compute_device: str
    cuda_ransac_calls: int
    cpu_ransac_calls: int
    cuda_fallbacks: int
    mean_frame_build_ms: float
    mean_pair_estimation_ms: float
    mean_ransac_ms: float


@dataclass
class _FeatureFrame:
    timestamp: float
    image: np.ndarray
    keypoint_pixels: np.ndarray
    keypoint_points_sensor: np.ndarray
    descriptors: np.ndarray
    # Sensor-to-world rotation at this acquisition, evaluated from the
    # propagated inertial state.  It is estimator state, never simulation
    # truth, and lets the feature residual use the physical gravity axis when
    # the vehicle is rolled or pitched.
    rotation_world_sensor: np.ndarray | None = None
    # Fraction of along-track gradient energy that survives cross-range
    # smoothing.  Independent speckle/electronic noise collapses under this
    # operation; coherent seabed texture remains.  It is computed solely from
    # the received image and is therefore available on a real vehicle.
    photometric_observability: float = 0.0
    along_track_row_centres_m: np.ndarray | None = None
    maximum_ground_range_m: float | None = None
    canonical_bins_per_side: int | None = None


@dataclass
class _PairEstimate:
    accepted: bool
    reason: str | None
    matches: int
    inliers: int
    rmse_m: float
    transform_previous_sensor_current_sensor: np.ndarray | None
    covariance: np.ndarray | None


def _pose_matrix(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = project_to_so3(rotation)
    pose[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return pose


def _pose2_from_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    return np.array(
        [
            pose[0, 3],
            pose[1, 3],
            math.atan2(pose[1, 0], pose[0, 0]),
        ],
        dtype=np.float64,
    )


def _matrix_from_planar_pose(pose: np.ndarray, z_translation: float = 0.0) -> np.ndarray:
    x, y, yaw = np.asarray(pose, dtype=np.float64).reshape(3)
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform[:3, 3] = [x, y, float(z_translation)]
    return transform


def _gravity_yaw_correction(
    correction: np.ndarray, gravity_axis_sensor: np.ndarray
) -> np.ndarray:
    """Build a local x/y plus gravity-yaw SE(3) correction.

    ``correction`` is a tangent increment in the *previous* sensor frame.
    The yaw component rotates around physical vertical expressed in that
    frame, rather than around sensor z.  That distinction is essential on a
    terrain-following AUV with non-zero roll and pitch.
    """

    delta = np.asarray(correction, dtype=np.float64).reshape(3)
    gravity_axis = np.asarray(
        gravity_axis_sensor, dtype=np.float64
    ).reshape(3)
    norm = float(np.linalg.norm(gravity_axis))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("gravity axis must be finite and non-zero")
    gravity_axis = gravity_axis / norm
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = so3_exp(gravity_axis * float(delta[2]))
    transform[:3, 3] = [float(delta[0]), float(delta[1]), 0.0]
    return transform


def _planar_covariance_from_state(
    rotation_world_body: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """Extract x/y/world-yaw covariance from a right-error inertial state."""

    rotation = project_to_so3(rotation_world_body)
    covariance = np.asarray(covariance, dtype=np.float64).reshape(15, 15)
    yaw_axis_body = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    yaw_axis_body /= max(float(np.linalg.norm(yaw_axis_body)), 1.0e-12)
    selector = np.zeros((3, 15), dtype=np.float64)
    selector[0:2, 3:5] = np.eye(2, dtype=np.float64)
    selector[2, 0:3] = yaw_axis_body
    planar = selector @ covariance @ selector.T
    return 0.5 * (planar + planar.T)


def _fit_planar_rigid(
    current_points: np.ndarray,
    previous_points: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Least-squares transform mapping current points into previous frame."""

    current = np.asarray(current_points, dtype=np.float64).reshape(-1, 3)
    previous = np.asarray(previous_points, dtype=np.float64).reshape(-1, 3)
    if current.shape != previous.shape or current.shape[0] < 2:
        raise ValueError("at least two paired 3-D points are required")
    if weights is None:
        normalized_weights = np.full(
            current.shape[0], 1.0 / current.shape[0], dtype=np.float64
        )
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if normalized_weights.size != current.shape[0]:
            raise ValueError("rigid-fit weights must match paired points")
        if not np.isfinite(normalized_weights).all() or np.any(
            normalized_weights < 0.0
        ):
            raise ValueError("rigid-fit weights must be finite and non-negative")
        total_weight = float(np.sum(normalized_weights))
        if total_weight <= 0.0:
            raise ValueError("rigid-fit weights must have positive sum")
        normalized_weights = normalized_weights / total_weight
    current_mean = np.sum(
        normalized_weights[:, None] * current[:, :2], axis=0
    )
    previous_mean = np.sum(
        normalized_weights[:, None] * previous[:, :2], axis=0
    )
    current_centered = current[:, :2] - current_mean
    previous_centered = previous[:, :2] - previous_mean
    cross = current_centered.T @ (
        normalized_weights[:, None] * previous_centered
    )
    u, _, vt = np.linalg.svd(cross)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = previous_mean - rotation @ current_mean
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    z_translation = float(np.median(previous[:, 2] - current[:, 2]))
    return np.array([translation[0], translation[1], yaw]), z_translation


def _transform_residuals(
    current_points: np.ndarray,
    previous_points: np.ndarray,
    pose: np.ndarray,
    z_translation: float,
    *,
    along_track_weight: float = 1.0,
    vertical_weight: float = 1.0,
) -> np.ndarray:
    x, y, yaw = np.asarray(pose, dtype=np.float64).reshape(3)
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    predicted_xy = np.asarray(current_points)[:, :2] @ rotation.T + [x, y]
    predicted_z = np.asarray(current_points)[:, 2] + float(z_translation)
    delta_xy = np.asarray(previous_points)[:, :2] - predicted_xy
    delta_z = np.asarray(previous_points)[:, 2] - predicted_z
    return np.sqrt(
        (float(along_track_weight) * delta_xy[:, 0]) ** 2
        + delta_xy[:, 1] ** 2
        + (float(vertical_weight) * delta_z) ** 2
    )


def _estimate_planar_rigid_ransac(
    current_points: np.ndarray,
    previous_points: np.ndarray,
    *,
    iterations: int,
    residual_limit_m: float,
    minimum_baseline_m: float,
    random_seed: int,
    along_track_weight: float = 1.0,
    vertical_weight: float = 1.0,
) -> tuple[np.ndarray | None, float, np.ndarray]:
    """Deterministic two-point RANSAC followed by all-inlier refinement."""

    current = np.asarray(current_points, dtype=np.float64).reshape(-1, 3)
    previous = np.asarray(previous_points, dtype=np.float64).reshape(-1, 3)
    count = current.shape[0]
    if count < 2 or previous.shape != current.shape:
        return None, 0.0, np.zeros(count, dtype=bool)
    rng = np.random.default_rng(int(random_seed) & 0xFFFFFFFF)
    best_inliers = np.zeros(count, dtype=bool)
    best_rmse = float("inf")
    for _ in range(max(1, int(iterations))):
        pair = rng.choice(count, size=2, replace=False)
        if (
            np.linalg.norm(current[pair[1], :2] - current[pair[0], :2])
            < minimum_baseline_m
            or np.linalg.norm(
                previous[pair[1], :2] - previous[pair[0], :2]
            )
            < minimum_baseline_m
        ):
            continue
        pose, z_translation = _fit_planar_rigid(
            current[pair], previous[pair]
        )
        residuals = _transform_residuals(
            current,
            previous,
            pose,
            z_translation,
            along_track_weight=along_track_weight,
            vertical_weight=vertical_weight,
        )
        inliers = residuals <= residual_limit_m
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 2:
            continue
        rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        if inlier_count > int(np.count_nonzero(best_inliers)) or (
            inlier_count == int(np.count_nonzero(best_inliers))
            and rmse < best_rmse
        ):
            best_inliers = inliers
            best_rmse = rmse
    if np.count_nonzero(best_inliers) < 2:
        return None, 0.0, best_inliers

    # Refit twice so the final consensus is consistent with the refined pose.
    for _ in range(2):
        pose, z_translation = _fit_planar_rigid(
            current[best_inliers], previous[best_inliers]
        )
        residuals = _transform_residuals(
            current,
            previous,
            pose,
            z_translation,
            along_track_weight=along_track_weight,
            vertical_weight=vertical_weight,
        )
        refined = residuals <= residual_limit_m
        if np.count_nonzero(refined) < 2:
            break
        best_inliers = refined
    selected_current = current[best_inliers]
    selected_previous = previous[best_inliers]
    # Equal total weight for port and starboard prevents a high-texture side
    # from moving the effective rotation centre tens of metres off nadir.
    # That imbalance otherwise aliases a tiny yaw error into lateral motion.
    port = selected_current[:, 1] >= 0.0
    starboard = ~port
    balanced_weights = np.ones(selected_current.shape[0], dtype=np.float64)
    if np.any(port) and np.any(starboard):
        balanced_weights[port] = 0.5 / np.count_nonzero(port)
        balanced_weights[starboard] = 0.5 / np.count_nonzero(starboard)
    pose, z_translation = _fit_planar_rigid(
        selected_current, selected_previous, balanced_weights
    )
    residuals = _transform_residuals(
        current,
        previous,
        pose,
        z_translation,
        along_track_weight=along_track_weight,
        vertical_weight=vertical_weight,
    )
    rmse = float(np.sqrt(np.mean(residuals[best_inliers] ** 2)))
    transform = _matrix_from_planar_pose(pose, z_translation)
    return transform, rmse, best_inliers


def _prediction_centered_residuals(
    predicted_points: np.ndarray,
    previous_points: np.ndarray,
    correction: np.ndarray,
    gravity_axis_sensor: np.ndarray,
    *,
    along_track_weight: float,
    vertical_weight: float,
) -> np.ndarray:
    correction_transform = _gravity_yaw_correction(
        correction, gravity_axis_sensor
    )
    transformed = (
        np.asarray(predicted_points, dtype=np.float64)
        @ correction_transform[:3, :3].T
        + correction_transform[:3, 3]
    )
    delta = np.asarray(previous_points, dtype=np.float64) - transformed
    component_weights = np.array(
        [along_track_weight, 1.0, vertical_weight], dtype=np.float64
    )
    return np.linalg.norm(delta * component_weights, axis=1)


def _fit_prediction_centered_correction(
    predicted_points: np.ndarray,
    previous_points: np.ndarray,
    gravity_axis_sensor: np.ndarray,
    *,
    along_track_weight: float,
    vertical_weight: float,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray | None:
    """Gauss-Newton fit of x/y/gravity-yaw around an inertial prediction."""

    predicted = np.asarray(predicted_points, dtype=np.float64).reshape(-1, 3)
    previous = np.asarray(previous_points, dtype=np.float64).reshape(-1, 3)
    if predicted.shape != previous.shape or predicted.shape[0] < 2:
        return None
    axis = np.asarray(gravity_axis_sensor, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1.0e-12:
        return None
    axis /= axis_norm
    component_weights = np.array(
        [along_track_weight, 1.0, vertical_weight], dtype=np.float64
    )
    if sample_weights is None:
        square_root_sample_weights = np.ones(
            predicted.shape[0], dtype=np.float64
        )
    else:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if (
            weights.size != predicted.shape[0]
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            return None
        square_root_sample_weights = np.sqrt(
            weights / float(np.mean(weights))
        )
    correction = np.zeros(3, dtype=np.float64)
    for _ in range(6):
        correction_transform = _gravity_yaw_correction(correction, axis)
        rotated = predicted @ correction_transform[:3, :3].T
        transformed = rotated + correction_transform[:3, 3]
        jacobian = np.zeros((predicted.shape[0], 3, 3), dtype=np.float64)
        jacobian[:, 0, 0] = 1.0
        jacobian[:, 1, 1] = 1.0
        jacobian[:, :, 2] = np.cross(
            np.broadcast_to(axis, rotated.shape), rotated
        )
        weighted_jacobian = (
            jacobian
            * component_weights[None, :, None]
            * square_root_sample_weights[:, None, None]
        ).reshape(-1, 3)
        weighted_residual = (
            (previous - transformed)
            * component_weights
            * square_root_sample_weights[:, None]
        ).reshape(-1)
        try:
            increment, _, rank, _ = np.linalg.lstsq(
                weighted_jacobian, weighted_residual, rcond=None
            )
        except np.linalg.LinAlgError:
            return None
        if rank < 3 or not np.isfinite(increment).all():
            return None
        correction += increment
        correction[2] = float(wrap_angle(correction[2]))
        if float(np.linalg.norm(increment)) < 1.0e-8:
            break
    return correction


def _draw_two_point_ransac_samples(
    count: int, iterations: int, random_seed: int
) -> np.ndarray:
    """Draw the historical deterministic two-point hypothesis sequence."""

    rng = np.random.default_rng(int(random_seed) & 0xFFFFFFFF)
    return np.stack(
        [
            rng.choice(int(count), size=2, replace=False)
            for _ in range(max(1, int(iterations)))
        ],
        axis=0,
    ).astype(np.int64, copy=False)


def _prediction_centered_consensus_cpu(
    predicted_points: np.ndarray,
    previous_points: np.ndarray,
    gravity_axis_sensor: np.ndarray,
    sample_pairs: np.ndarray,
    *,
    residual_limit_m: float,
    minimum_baseline_m: float,
    along_track_weight: float,
    vertical_weight: float,
) -> np.ndarray:
    """Score prediction-centred RANSAC hypotheses serially on the CPU."""

    count = int(predicted_points.shape[0])
    best_inliers = np.zeros(count, dtype=bool)
    best_rmse = float("inf")
    for pair in np.asarray(sample_pairs, dtype=np.int64).reshape(-1, 2):
        if (
            np.linalg.norm(
                predicted_points[pair[1], :2]
                - predicted_points[pair[0], :2]
            )
            < minimum_baseline_m
            or np.linalg.norm(
                previous_points[pair[1], :2]
                - previous_points[pair[0], :2]
            )
            < minimum_baseline_m
        ):
            continue
        correction = _fit_prediction_centered_correction(
            predicted_points[pair],
            previous_points[pair],
            gravity_axis_sensor,
            along_track_weight=along_track_weight,
            vertical_weight=vertical_weight,
        )
        if correction is None:
            continue
        residuals = _prediction_centered_residuals(
            predicted_points,
            previous_points,
            correction,
            gravity_axis_sensor,
            along_track_weight=along_track_weight,
            vertical_weight=vertical_weight,
        )
        inliers = residuals <= residual_limit_m
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 2:
            continue
        rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        if inlier_count > int(np.count_nonzero(best_inliers)) or (
            inlier_count == int(np.count_nonzero(best_inliers))
            and rmse < best_rmse
        ):
            best_inliers = inliers
            best_rmse = rmse
    return best_inliers


def _prediction_centered_consensus_cuda(
    predicted_points: np.ndarray,
    previous_points: np.ndarray,
    gravity_axis_sensor: np.ndarray,
    sample_pairs: np.ndarray,
    *,
    residual_limit_m: float,
    minimum_baseline_m: float,
    along_track_weight: float,
    vertical_weight: float,
    torch_module=None,
) -> np.ndarray:
    """Batch all prediction-centred hypotheses on CUDA.

    Only hypothesis generation and scoring move to the device.  Float64 is
    intentional: it makes threshold/tie behaviour agree with the historical
    NumPy implementation, while batching removes the Python/lstsq loop.  The
    selected consensus is still refined below by the unchanged CPU estimator.
    """

    if torch_module is None:  # Lazy optional dependency for CPU-only users.
        try:
            import torch as torch_module
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "CUDA feature RANSAC requires a CUDA-enabled PyTorch install"
            ) from exc
    torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA feature RANSAC requested but CUDA is unavailable")

    predicted = np.ascontiguousarray(predicted_points, dtype=np.float64)
    previous = np.ascontiguousarray(previous_points, dtype=np.float64)
    samples = np.ascontiguousarray(sample_pairs, dtype=np.int64)
    axis = np.asarray(gravity_axis_sensor, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1.0e-12:
        return np.zeros(predicted.shape[0], dtype=bool)
    axis /= axis_norm

    device = torch.device("cuda")
    dtype = torch.float64
    with torch.inference_mode():
        predicted_tensor = torch.as_tensor(
            predicted, dtype=dtype, device=device
        )
        previous_tensor = torch.as_tensor(
            previous, dtype=dtype, device=device
        )
        sample_tensor = torch.as_tensor(samples, device=device)
        axis_tensor = torch.as_tensor(axis, dtype=dtype, device=device)
        component_weights = torch.tensor(
            [along_track_weight, 1.0, vertical_weight],
            dtype=dtype,
            device=device,
        )
        hypotheses = int(samples.shape[0])
        sampled_predicted = predicted_tensor[sample_tensor]
        sampled_previous = previous_tensor[sample_tensor]
        correction = torch.zeros(
            (hypotheses, 3), dtype=dtype, device=device
        )
        valid = (
            torch.linalg.vector_norm(
                sampled_predicted[:, 1, :2]
                - sampled_predicted[:, 0, :2],
                dim=1,
            )
            >= float(minimum_baseline_m)
        ) & (
            torch.linalg.vector_norm(
                sampled_previous[:, 1, :2]
                - sampled_previous[:, 0, :2],
                dim=1,
            )
            >= float(minimum_baseline_m)
        )
        active = valid.clone()
        axis_view = axis_tensor.reshape(1, 1, 3)

        def rotate_about_gravity(points, angles):
            cosine = torch.cos(angles).reshape(-1, 1, 1)
            sine = torch.sin(angles).reshape(-1, 1, 1)
            projected = torch.sum(points * axis_view, dim=2, keepdim=True)
            return (
                points * cosine
                + torch.cross(
                    axis_view.expand_as(points), points, dim=2
                )
                * sine
                + axis_view * projected * (1.0 - cosine)
            )

        # Match _fit_prediction_centered_correction: at most six Gauss-Newton
        # iterations, with converged hypotheses frozen independently.
        for _ in range(6):
            rotated = rotate_about_gravity(
                sampled_predicted, correction[:, 2]
            )
            translation = torch.stack(
                (
                    correction[:, 0],
                    correction[:, 1],
                    torch.zeros_like(correction[:, 0]),
                ),
                dim=1,
            )
            transformed = rotated + translation[:, None, :]
            jacobian = torch.zeros(
                (hypotheses, 2, 3, 3), dtype=dtype, device=device
            )
            jacobian[:, :, 0, 0] = 1.0
            jacobian[:, :, 1, 1] = 1.0
            jacobian[:, :, :, 2] = torch.cross(
                axis_view.expand_as(rotated), rotated, dim=2
            )
            weighted_jacobian = (
                jacobian * component_weights.reshape(1, 1, 3, 1)
            ).reshape(hypotheses, -1, 3)
            weighted_residual = (
                (sampled_previous - transformed) * component_weights
            ).reshape(hypotheses, -1)
            normal = (
                weighted_jacobian.transpose(1, 2) @ weighted_jacobian
            )
            rhs = (
                weighted_jacobian.transpose(1, 2)
                @ weighted_residual.unsqueeze(2)
            )
            increment, solve_info = torch.linalg.solve_ex(normal, rhs)
            increment = increment[:, :, 0]
            # The two-point XY baseline test removes the routine degenerate
            # hypotheses; solve_ex catches singular normal matrices, and the
            # unchanged all-inlier information-eigenvalue test below remains
            # the final observability check.  A batched SVD at every inner
            # iteration would duplicate that check at substantial cost.
            solvable = (
                (solve_info == 0)
                & torch.isfinite(increment).all(dim=1)
            )
            failed = active & ~solvable
            valid &= ~failed
            update = active & solvable
            correction = torch.where(
                update[:, None], correction + increment, correction
            )
            correction[:, 2] = torch.atan2(
                torch.sin(correction[:, 2]), torch.cos(correction[:, 2])
            )
            converged = (
                torch.linalg.vector_norm(increment, dim=1) < 1.0e-8
            )
            active = update & ~converged

        all_predicted = predicted_tensor.unsqueeze(0).expand(
            hypotheses, -1, -1
        )
        rotated = rotate_about_gravity(all_predicted, correction[:, 2])
        translation = torch.stack(
            (
                correction[:, 0],
                correction[:, 1],
                torch.zeros_like(correction[:, 0]),
            ),
            dim=1,
        )
        transformed = rotated + translation[:, None, :]
        residuals = torch.linalg.vector_norm(
            (previous_tensor.unsqueeze(0) - transformed)
            * component_weights,
            dim=2,
        )
        inliers = (residuals <= float(residual_limit_m)) & valid[:, None]
        inlier_counts = torch.sum(inliers, dim=1)
        squared_sum = torch.sum(residuals.square() * inliers, dim=1)
        rmses = torch.sqrt(squared_sum / inlier_counts.clamp_min(1))
        infinity = torch.full_like(rmses, float("inf"))
        rmses = torch.where(inlier_counts >= 2, rmses, infinity)
        maximum_count = torch.max(inlier_counts)
        if int(maximum_count.item()) < 2:
            return np.zeros(predicted.shape[0], dtype=bool)
        ranked_rmse = torch.where(
            inlier_counts == maximum_count, rmses, infinity
        )
        # torch.argmin returns the first index, preserving the serial CPU tie
        # rule for exactly equal consensus count and RMSE.
        best_index = torch.argmin(ranked_rmse)
        return (
            inliers[best_index]
            .to(device="cpu", dtype=torch.bool)
            .numpy()
            .astype(bool, copy=False)
        )


def _estimate_prediction_centered_ransac(
    current_points: np.ndarray,
    previous_points: np.ndarray,
    predicted_transform: np.ndarray,
    gravity_axis_sensor: np.ndarray,
    *,
    iterations: int,
    residual_limit_m: float,
    minimum_baseline_m: float,
    random_seed: int,
    along_track_weight: float,
    vertical_weight: float,
    compute_backend: str = "cpu",
    torch_module=None,
) -> tuple[np.ndarray | None, float, np.ndarray]:
    """Robustly estimate only the acoustic residual about IMU/DVL motion.

    The nominal relative transform retains all propagated roll, pitch, depth,
    and lever-arm motion.  Acoustic correspondences solve three observable
    corrections: previous-sensor x/y and yaw about physical gravity.
    """

    current = np.asarray(current_points, dtype=np.float64).reshape(-1, 3)
    previous = np.asarray(previous_points, dtype=np.float64).reshape(-1, 3)
    predicted = np.asarray(predicted_transform, dtype=np.float64).reshape(4, 4)
    count = current.shape[0]
    if count < 2 or previous.shape != current.shape:
        return None, 0.0, np.zeros(count, dtype=bool)
    predicted_points = current @ predicted[:3, :3].T + predicted[:3, 3]
    sample_pairs = _draw_two_point_ransac_samples(
        count, iterations, random_seed
    )
    if compute_backend == "cuda":
        best_inliers = _prediction_centered_consensus_cuda(
            predicted_points,
            previous,
            gravity_axis_sensor,
            sample_pairs,
            residual_limit_m=residual_limit_m,
            minimum_baseline_m=minimum_baseline_m,
            along_track_weight=along_track_weight,
            vertical_weight=vertical_weight,
            torch_module=torch_module,
        )
    elif compute_backend == "cpu":
        best_inliers = _prediction_centered_consensus_cpu(
            predicted_points,
            previous,
            gravity_axis_sensor,
            sample_pairs,
            residual_limit_m=residual_limit_m,
            minimum_baseline_m=minimum_baseline_m,
            along_track_weight=along_track_weight,
            vertical_weight=vertical_weight,
        )
    else:
        raise ValueError("prediction-centred RANSAC backend must be cpu or cuda")
    if np.count_nonzero(best_inliers) < 2:
        return None, 0.0, best_inliers

    correction = None
    for _ in range(3):
        correction = _fit_prediction_centered_correction(
            predicted_points[best_inliers],
            previous[best_inliers],
            gravity_axis_sensor,
            along_track_weight=along_track_weight,
            vertical_weight=vertical_weight,
        )
        if correction is None:
            return None, 0.0, best_inliers
        residuals = _prediction_centered_residuals(
            predicted_points,
            previous,
            correction,
            gravity_axis_sensor,
            along_track_weight=along_track_weight,
            vertical_weight=vertical_weight,
        )
        refined = residuals <= residual_limit_m
        if np.count_nonzero(refined) < 2 or np.array_equal(
            refined, best_inliers
        ):
            break
        best_inliers = refined
    if correction is None:
        return None, 0.0, best_inliers
    selected_predicted = predicted_points[best_inliers]
    selected_previous = previous[best_inliers]
    port = selected_predicted[:, 1] >= 0.0
    starboard = ~port
    balanced_weights = np.ones(
        selected_predicted.shape[0], dtype=np.float64
    )
    if np.any(port) and np.any(starboard):
        balanced_weights[port] = 0.5 / float(np.count_nonzero(port))
        balanced_weights[starboard] = 0.5 / float(
            np.count_nonzero(starboard)
        )
    balanced_correction = _fit_prediction_centered_correction(
        selected_predicted,
        selected_previous,
        gravity_axis_sensor,
        along_track_weight=along_track_weight,
        vertical_weight=vertical_weight,
        sample_weights=balanced_weights,
    )
    if balanced_correction is not None:
        correction = balanced_correction
    residuals = _prediction_centered_residuals(
        predicted_points,
        previous,
        correction,
        gravity_axis_sensor,
        along_track_weight=along_track_weight,
        vertical_weight=vertical_weight,
    )
    rmse = float(np.sqrt(np.mean(residuals[best_inliers] ** 2)))
    transform = (
        _gravity_yaw_correction(correction, gravity_axis_sensor) @ predicted
    )
    return transform, rmse, best_inliers


def _prediction_centered_covariance(
    transformed_current_points: np.ndarray,
    gravity_axis_sensor: np.ndarray,
    inlier_rmse_m: float,
    *,
    position_floor_m: float,
    cross_track_position_floor_m: float,
    yaw_floor_rad: float,
    effective_independent_points: int | None,
    along_track_weight: float,
    vertical_weight: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Covariance in the local x/y/gravity-yaw correction tangent."""

    points = np.asarray(
        transformed_current_points, dtype=np.float64
    ).reshape(-1, 3)
    if points.shape[0] < 2:
        return None, np.zeros(3, dtype=np.float64)
    axis = np.asarray(gravity_axis_sensor, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    jacobian = np.zeros((points.shape[0], 3, 3), dtype=np.float64)
    jacobian[:, 0, 0] = 1.0
    jacobian[:, 1, 1] = 1.0
    jacobian[:, :, 2] = np.cross(
        np.broadcast_to(axis, points.shape), points
    )
    component_weights = np.array(
        [along_track_weight, 1.0, vertical_weight], dtype=np.float64
    )
    weighted = (
        jacobian * component_weights[None, :, None]
    ).reshape(-1, 3)
    information = weighted.T @ weighted
    information = 0.5 * (information + information.T)
    try:
        eigenvalues = np.maximum(np.linalg.eigvalsh(information), 0.0)
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return None, np.zeros(3, dtype=np.float64)
    sigma = max(float(inlier_rmse_m), 0.5 * float(position_floor_m))
    correlation_scale = (
        1.0
        if effective_independent_points is None
        else max(1.0, points.shape[0] / float(effective_independent_points))
    )
    covariance *= sigma * sigma * correlation_scale
    covariance += np.diag(
        [
            position_floor_m**2,
            cross_track_position_floor_m**2,
            yaw_floor_rad**2,
        ]
    )
    return 0.5 * (covariance + covariance.T), eigenvalues


def _relative_covariance(
    current_points: np.ndarray,
    transform: np.ndarray,
    inlier_rmse_m: float,
    *,
    position_floor_m: float,
    yaw_floor_rad: float,
    effective_independent_points: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Return covariance and eigenvalues for planar point correspondences."""

    current = np.asarray(current_points, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(transform, dtype=np.float64)[:2, :2]
    rotated = current[:, :2] @ rotation.T
    jacobian = np.zeros((2 * current.shape[0], 3), dtype=np.float64)
    jacobian[0::2, 0] = 1.0
    jacobian[1::2, 1] = 1.0
    jacobian[0::2, 2] = -rotated[:, 1]
    jacobian[1::2, 2] = rotated[:, 0]
    information = jacobian.T @ jacobian
    information = 0.5 * (information + information.T)
    try:
        eigenvalues = np.maximum(np.linalg.eigvalsh(information), 0.0)
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return None, np.zeros(3, dtype=np.float64)
    sigma = max(float(inlier_rmse_m), 0.5 * float(position_floor_m))
    correlation_scale = (
        1.0
        if effective_independent_points is None
        else max(
            1.0,
            current.shape[0] / float(effective_independent_points),
        )
    )
    covariance *= sigma * sigma * correlation_scale
    covariance += np.diag(
        [position_floor_m**2, position_floor_m**2, yaw_floor_rad**2]
    )
    return 0.5 * (covariance + covariance.T), eigenvalues


def _compose_planar_covariance(
    first_transform: np.ndarray,
    first_covariance: np.ndarray,
    second_transform: np.ndarray,
    second_covariance: np.ndarray,
) -> np.ndarray:
    """First-order covariance for ``first_transform @ second_transform``."""

    first = _pose2_from_matrix(first_transform)
    second = _pose2_from_matrix(second_transform)
    cosine = math.cos(float(first[2]))
    sine = math.sin(float(first[2]))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    derivative_rotation = np.array(
        [[-sine, -cosine], [cosine, -sine]], dtype=np.float64
    )
    jacobian_first = np.eye(3, dtype=np.float64)
    jacobian_first[:2, 2] = derivative_rotation @ second[:2]
    jacobian_second = np.eye(3, dtype=np.float64)
    jacobian_second[:2, :2] = rotation
    covariance = (
        jacobian_first @ first_covariance @ jacobian_first.T
        + jacobian_second @ second_covariance @ jacobian_second.T
    )
    return 0.5 * (covariance + covariance.T)


class FeatureSonarOdometry:
    """Causal side-scan feature tracks lifted into co-registered MBES XYZ."""

    def __init__(
        self,
        config: FeatureSonarOdometryConfig | None = None,
        *,
        sensor_rotation_body: np.ndarray | None = None,
        sensor_translation_body: np.ndarray | None = None,
    ) -> None:
        self.config = config or FeatureSonarOdometryConfig()
        self._validate_config()
        self._torch = None
        self._requested_compute_backend = self.config.compute_backend
        self._active_compute_backend = self._resolve_compute_backend()
        self._compute_device = "CPU"
        if self._active_compute_backend == "cuda":
            self._compute_device = str(
                self._torch.cuda.get_device_name(
                    self._torch.cuda.current_device()
                )
            )
        rotation = (
            np.eye(3, dtype=np.float64)
            if sensor_rotation_body is None
            else project_to_so3(sensor_rotation_body)
        )
        translation = (
            np.zeros(3, dtype=np.float64)
            if sensor_translation_body is None
            else np.asarray(sensor_translation_body, dtype=np.float64).reshape(3)
        )
        self._body_from_sensor = _pose_matrix(rotation, translation)
        self.reset()

    def _validate_config(self) -> None:
        config = self.config
        if int(config.subframe_pings) < 2:
            raise ValueError("feature subframe_pings must be at least two")
        if int(config.maximum_features) < 16:
            raise ValueError("feature maximum_features must be at least 16")
        if config.association_method not in {"descriptor", "klt"}:
            raise ValueError(
                "feature association_method must be descriptor or klt"
            )
        if config.state_update_policy not in {"along_track", "planar"}:
            raise ValueError(
                "feature state_update_policy must be along_track or planar"
            )
        if not 0.0 < float(config.descriptor_ratio) < 1.0:
            raise ValueError("feature descriptor_ratio must lie in (0, 1)")
        if float(config.descriptor_maximum_distance) <= 0.0:
            raise ValueError(
                "feature descriptor_maximum_distance must be positive"
            )
        if int(config.minimum_matches) < 3:
            raise ValueError("feature minimum_matches must be at least three")
        if int(config.effective_independent_matches) < 3:
            raise ValueError(
                "feature effective_independent_matches must be at least three"
            )
        if int(config.ransac_iterations) < 1:
            raise ValueError("feature ransac_iterations must be positive")
        if float(config.ransac_residual_m) <= 0.0:
            raise ValueError("feature ransac_residual_m must be positive")
        if float(config.along_track_residual_weight) < 1.0:
            raise ValueError(
                "feature along_track_residual_weight must be at least one"
            )
        if float(config.vertical_residual_weight) < 1.0:
            raise ValueError(
                "feature vertical_residual_weight must be at least one"
            )
        association_scales = (
            config.association_along_track_std_m,
            config.association_cross_track_std_m,
            config.association_vertical_std_m,
        )
        if not all(float(value) > 0.0 for value in association_scales):
            raise ValueError("feature association scales must be positive")
        if float(config.association_maximum_normalized_distance) <= 0.0:
            raise ValueError(
                "feature association maximum distance must be positive"
            )
        if float(config.association_geometry_weight) < 0.0:
            raise ValueError(
                "feature association geometry weight must be non-negative"
            )
        if int(config.image_row_scale) < 1:
            raise ValueError("feature image_row_scale must be positive")
        if int(config.canonical_range_bins_per_side) < 64:
            raise ValueError(
                "feature canonical_range_bins_per_side must be at least 64"
            )
        if float(config.minimum_pair_baseline_m) <= 0.0:
            raise ValueError("feature minimum_pair_baseline_m must be positive")
        if float(config.minimum_information_eigenvalue) <= 0.0:
            raise ValueError(
                "feature minimum_information_eigenvalue must be positive"
            )
        if float(config.position_noise_floor_m) <= 0.0:
            raise ValueError("feature position_noise_floor_m must be positive")
        if float(config.cross_track_position_noise_floor_m) <= 0.0:
            raise ValueError(
                "feature cross_track_position_noise_floor_m must be positive"
            )
        if float(config.yaw_noise_floor_deg) <= 0.0:
            raise ValueError("feature yaw_noise_floor_deg must be positive")
        if (
            int(config.klt_window_width) < 3
            or int(config.klt_window_height) < 3
            or int(config.klt_maximum_level) < 0
        ):
            raise ValueError("feature KLT window and pyramid settings are invalid")
        if (
            int(config.klt_window_width) % 2 == 0
            or int(config.klt_window_height) % 2 == 0
        ):
            raise ValueError("feature KLT window dimensions must be odd")
        if (
            float(config.klt_forward_backward_limit_px) <= 0.0
            or float(config.klt_lift_radius_px) <= 0.0
            or float(config.prediction_correction_scale_m) <= 0.0
        ):
            raise ValueError(
                "feature KLT and prediction consistency scales must be positive"
            )
        if float(config.anchor_covariance_scale) < 1.0:
            raise ValueError("feature anchor_covariance_scale must be >= 1")
        if config.compute_backend not in {"cpu", "cuda", "auto"}:
            raise ValueError("feature compute_backend must be cpu, cuda, or auto")

    def _resolve_compute_backend(self) -> str:
        requested = self.config.compute_backend
        if requested == "cpu":
            return "cpu"
        try:
            import torch
        except ImportError as exc:
            if requested == "cuda":
                raise RuntimeError(
                    "feature compute backend cuda requires PyTorch"
                ) from exc
            return "cpu"
        if not torch.cuda.is_available():
            if requested == "cuda":
                raise RuntimeError(
                    "feature compute backend cuda requested but CUDA is unavailable"
                )
            return "cpu"
        self._torch = torch
        return "cuda"

    def reset(self) -> None:
        self._previous_frame: _FeatureFrame | None = None
        self._previous_world_body: np.ndarray | None = None
        self._pending_frame: _FeatureFrame | None = None
        self._anchor_world_body: np.ndarray | None = None
        self._anchor_planar_covariance: np.ndarray | None = None
        self._chain_transform = np.eye(4, dtype=np.float64)
        self._chain_covariance = np.zeros((3, 3), dtype=np.float64)
        self._chain_ping_count = 0
        self._restart_anchor_after_finalize = False
        self._frames = 0
        self._pair_attempts = 0
        self._accepted_pairs = 0
        self._rejected_pairs = 0
        self._completed_subframes = 0
        self._rejection_reasons: Counter[str] = Counter()
        self._inlier_ratio_sum = 0.0
        self._pair_rmse_sum = 0.0
        self._cuda_ransac_calls = 0
        self._cpu_ransac_calls = 0
        self._cuda_fallbacks = 0
        self._frame_build_calls = 0
        self._frame_build_wall_s = 0.0
        self._pair_estimation_wall_s = 0.0
        self._ransac_wall_s = 0.0

    @staticmethod
    def _opencv():
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "feature sonar odometry requires OpenCV feature primitives"
            ) from exc
        return cv2

    def _canonical_frame(self, ping: "BathymetricSideScanPing") -> _FeatureFrame:
        cv2 = self._opencv()
        if ping.imaging is None:
            raise ValueError(
                "feature sonar odometry requires point-wise imaging returns"
            )
        imaging = ping.imaging
        row_indices = np.asarray(
            imaging.along_track_row_indices, dtype=np.int64
        ).reshape(-1)
        if row_indices.size == 0:
            raise ValueError("feature sonar frame has no along-track rows")
        rows = int(np.max(row_indices)) + 1
        if rows < 2:
            raise ValueError(
                "feature sonar odometry requires at least two simultaneous rows"
            )
        centres = np.asarray(
            ping.side_scan.range_bin_centres, dtype=np.float64
        ).reshape(-1)
        raw_bins = int(centres.size)
        if raw_bins < 16:
            raise ValueError("feature sonar frame requires at least 16 range bins")
        bins = min(
            raw_bins, int(self.config.canonical_range_bins_per_side)
        )
        if raw_bins > 1:
            spacing = float(np.median(np.diff(centres)))
        else:  # unreachable because bins >= 16, retained for completeness
            spacing = max(float(centres[0]), 1.0)
        maximum_ground_range = max(
            float(centres[-1] + 0.5 * spacing), spacing
        )

        intensities = np.asarray(imaging.intensities, dtype=np.float64)
        points = np.asarray(imaging.points_sensor, dtype=np.float64)
        directions = np.asarray(imaging.beam_directions, dtype=np.float64)
        valid_returns = (
            np.asarray(imaging.hit_mask, dtype=bool)
            & np.isfinite(intensities)
            & (intensities > 0.0)
            & np.isfinite(points).all(axis=1)
            & np.isfinite(directions).all(axis=1)
        )
        image = np.zeros((rows, 2 * bins), dtype=np.float64)
        point_raster = np.full((rows, 2 * bins, 3), np.nan, dtype=np.float64)
        sources = np.flatnonzero(valid_returns)
        if sources.size == 0:
            raise ValueError("feature sonar frame has insufficient valid returns")
        source_rows = row_indices[sources]
        ground_ranges = np.abs(points[sources, 1])
        range_bins = np.clip(
            np.floor(ground_ranges / maximum_ground_range * bins).astype(
                np.int64
            ),
            0,
            bins - 1,
        )
        source_columns = np.where(
            directions[sources, 1] >= 0.0,
            bins - 1 - range_bins,
            bins + range_bins,
        )
        raster_width = 2 * bins
        linear_cells = source_rows * raster_width + source_columns
        image_flat = image.reshape(-1)
        np.maximum.at(image_flat, linear_cells, intensities[sources])

        # Preserve the old strict-maximum rule exactly: within each raster
        # cell choose the highest intensity and, for a tie, the earliest raw
        # return.  This removes a 4096-element Python loop without changing
        # which physical XYZ point owns a side-scan pixel.
        winner_order = np.lexsort(
            (sources, -intensities[sources], linear_cells)
        )
        ordered_cells = linear_cells[winner_order]
        first_in_cell = np.empty(ordered_cells.size, dtype=bool)
        first_in_cell[0] = True
        first_in_cell[1:] = ordered_cells[1:] != ordered_cells[:-1]
        winners = sources[winner_order[first_in_cell]]
        winner_cells = linear_cells[winner_order[first_in_cell]]
        point_raster.reshape(-1, 3)[winner_cells] = points[winners]

        support = image > 0.0
        if np.count_nonzero(support) < self.config.minimum_matches:
            raise ValueError("feature sonar frame has insufficient valid returns")

        # Canonical radiometry: log compression, slowly varying range-gain
        # removal, robust normalization and local contrast equalization.  XYZ
        # lineage remains attached to the unfiltered raster throughout.
        compressed = np.log1p(12.0 * image) / math.log(13.0)
        # Exact per-column median for the eight physical rows, evaluated as
        # one vector operation instead of one NumPy call per range cell.
        sortable_columns = np.sort(
            np.where(support, compressed, np.inf), axis=0
        )
        column_counts = np.count_nonzero(support, axis=0)
        column_indices = np.arange(2 * bins)
        lower_indices = np.maximum((column_counts - 1) // 2, 0)
        upper_indices = np.maximum(column_counts // 2, 0)
        column_gain = 0.5 * (
            sortable_columns[lower_indices, column_indices]
            + sortable_columns[upper_indices, column_indices]
        )
        column_gain[column_counts == 0] = 0.0
        known = column_gain > 0.0
        if np.count_nonzero(known) >= 2:
            columns = np.arange(2 * bins, dtype=np.float64)
            column_gain = np.interp(columns, columns[known], column_gain[known])
        else:
            column_gain.fill(float(np.median(compressed[support])))
        column_gain = cv2.GaussianBlur(
            column_gain.reshape(1, -1), (0, 0), sigmaX=9.0
        ).reshape(-1)
        canonical = np.zeros_like(compressed)
        canonical[support] = compressed[support] / np.maximum(
            np.broadcast_to(column_gain, compressed.shape)[support], 1.0e-3
        )
        low, high = np.percentile(canonical[support], [2.0, 98.0])
        if high <= low + 1.0e-9:
            raise ValueError("feature sonar frame has no radiometric contrast")
        canonical = np.clip((canonical - low) / (high - low), 0.0, 1.0)
        cross_range_smoothed = cv2.GaussianBlur(
            canonical,
            (9, 1),
            sigmaX=2.0,
            sigmaY=0.0,
            borderType=cv2.BORDER_REFLECT,
        )
        adjacent_support = support[1:] & support[:-1]
        raw_row_difference = np.diff(canonical, axis=0)
        coherent_row_difference = np.diff(
            cross_range_smoothed, axis=0
        )
        if np.any(adjacent_support):
            raw_energy = float(
                np.mean(raw_row_difference[adjacent_support] ** 2)
            )
            coherent_energy = float(
                np.mean(coherent_row_difference[adjacent_support] ** 2)
            )
            photometric_observability = float(
                np.clip(
                    coherent_energy / max(raw_energy, 1.0e-12),
                    0.0,
                    1.0,
                )
            )
        else:
            photometric_observability = 0.0
        image_u8 = np.rint(canonical * 255.0).astype(np.uint8)
        scale = int(self.config.image_row_scale)
        image_u8 = cv2.resize(
            image_u8,
            (image_u8.shape[1], image_u8.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        mask = cv2.resize(
            support.astype(np.uint8) * 255,
            (support.shape[1], support.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        # Close only descriptor-image pinholes.  The point lookup below still
        # requires a nearby, physically registered raw return.
        image_u8 = cv2.morphologyEx(
            image_u8, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
        )
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 2))
        image_u8 = clahe.apply(image_u8)
        # A physical ping is only eight rows high.  Generic scale-space SIFT
        # finds very few stable extrema in such an anisotropic image, even
        # after vertical interpolation.  Detect acoustic texture corners and
        # describe normalized 2-D patches instead; the subsequent estimator
        # remains invariant to gain/offset and obtains metric geometry from
        # the registered XYZ rather than from pixel displacement.
        corners = cv2.goodFeaturesToTrack(
            image_u8,
            maxCorners=int(self.config.maximum_features),
            qualityLevel=0.003,
            minDistance=4.0,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        corner_pixels = (
            np.empty((0, 2), dtype=np.float64)
            if corners is None
            else np.asarray(corners[:, 0, :], dtype=np.float64)
        )
        patch_radius = 9
        padded_image = np.pad(
            image_u8.astype(np.float32), patch_radius, mode="reflect"
        )
        accepted_pixels: list[tuple[float, float]] = []
        accepted_points: list[np.ndarray] = []
        accepted_descriptors: list[np.ndarray] = []
        if corner_pixels.size:
            rounded_columns = np.rint(corner_pixels[:, 0]).astype(np.int64)
            rounded_image_rows = np.rint(corner_pixels[:, 1]).astype(
                np.int64
            )
            rounded_rows = np.rint(corner_pixels[:, 1] / scale).astype(
                np.int64
            )
            row_offsets, column_offsets = np.meshgrid(
                np.arange(-1, 2, dtype=np.int64),
                np.arange(-2, 3, dtype=np.int64),
                indexing="ij",
            )
            candidate_rows = rounded_rows[:, None] + row_offsets.reshape(1, -1)
            candidate_columns = (
                rounded_columns[:, None] + column_offsets.reshape(1, -1)
            )
            candidate_bounds = (
                (candidate_rows >= 0)
                & (candidate_rows < rows)
                & (candidate_columns >= 0)
                & (candidate_columns < 2 * bins)
            )
            safe_rows = np.clip(candidate_rows, 0, rows - 1)
            safe_columns = np.clip(candidate_columns, 0, 2 * bins - 1)
            candidate_points = point_raster[safe_rows, safe_columns]
            candidate_valid = candidate_bounds & np.isfinite(
                candidate_points
            ).all(axis=2)
            candidate_distances = (
                (candidate_rows * scale - corner_pixels[:, 1, None]) ** 2
                + (candidate_columns - corner_pixels[:, 0, None]) ** 2
            )
            candidate_distances[~candidate_valid] = np.inf
            best_offsets = np.argmin(candidate_distances, axis=1)
            corner_indices = np.arange(corner_pixels.shape[0])
            best_distances = candidate_distances[corner_indices, best_offsets]
            best_rows = safe_rows[corner_indices, best_offsets]
            best_columns = safe_columns[corner_indices, best_offsets]
            usable = np.flatnonzero(np.isfinite(best_distances))
        else:
            rounded_columns = np.empty(0, dtype=np.int64)
            rounded_image_rows = np.empty(0, dtype=np.int64)
            best_rows = np.empty(0, dtype=np.int64)
            best_columns = np.empty(0, dtype=np.int64)
            usable = np.empty(0, dtype=np.int64)
        used_return_cells: set[tuple[int, int]] = set()
        for corner_index in usable:
            pixel = corner_pixels[corner_index]
            column = int(rounded_columns[corner_index])
            image_row = int(rounded_image_rows[corner_index])
            return_cell = (
                int(best_rows[corner_index]),
                int(best_columns[corner_index]),
            )
            if return_cell in used_return_cells:
                continue
            patch = padded_image[
                image_row : image_row + 2 * patch_radius + 1,
                column : column + 2 * patch_radius + 1,
            ]
            descriptor = cv2.resize(
                patch, (8, 8), interpolation=cv2.INTER_AREA
            ).reshape(-1)
            descriptor -= float(np.mean(descriptor))
            deviation = float(np.std(descriptor))
            if deviation <= 1.0e-6:
                continue
            descriptor /= deviation
            descriptor /= max(float(np.linalg.norm(descriptor)), 1.0e-12)
            accepted_pixels.append((float(pixel[0]), float(pixel[1])))
            accepted_points.append(
                point_raster[
                    best_rows[corner_index], best_columns[corner_index]
                ].copy()
            )
            accepted_descriptors.append(descriptor.copy())
            used_return_cells.add(return_cell)
        descriptor_array = (
            np.asarray(accepted_descriptors, dtype=np.float32)
            if accepted_descriptors
            else np.empty((0, 64), dtype=np.float32)
        )
        origins = getattr(imaging, "ray_origins_sensor", None)
        if origins is not None:
            origin_x = np.asarray(origins, dtype=np.float64).reshape(-1, 3)[
                :, 0
            ]
        else:
            # Compatibility for recorded/synthetic messages that predate
            # explicit phase-centre publication.  Parallel-row V2 returns
            # have zero along-track steering, so point x is the row origin.
            origin_x = points[:, 0]
        row_centres = np.empty(rows, dtype=np.float64)
        for row in range(rows):
            selected = origin_x[(row_indices == row) & np.isfinite(origin_x)]
            row_centres[row] = (
                float(np.median(selected)) if selected.size else float(row)
            )
        return _FeatureFrame(
            timestamp=float(ping.timestamp),
            image=image_u8,
            keypoint_pixels=np.asarray(accepted_pixels, dtype=np.float64).reshape(-1, 2),
            keypoint_points_sensor=np.asarray(
                accepted_points, dtype=np.float64
            ).reshape(-1, 3),
            descriptors=descriptor_array,
            photometric_observability=photometric_observability,
            along_track_row_centres_m=row_centres,
            maximum_ground_range_m=maximum_ground_range,
            canonical_bins_per_side=bins,
        )

    def _klt_pairs(
        self,
        previous: _FeatureFrame,
        current: _FeatureFrame,
        predicted_transform: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Track current acoustic corners back to the previous V2 image."""

        if (
            previous.along_track_row_centres_m is None
            or previous.maximum_ground_range_m is None
            or previous.canonical_bins_per_side is None
        ):
            return []
        centres = np.asarray(
            previous.along_track_row_centres_m, dtype=np.float64
        ).reshape(-1)
        if centres.size < 2:
            return []
        spacing = float(np.median(np.diff(centres)))
        if not np.isfinite(spacing) or abs(spacing) <= 1.0e-9:
            return []
        predicted = np.asarray(
            predicted_transform, dtype=np.float64
        ).reshape(4, 4)
        predicted_points = (
            current.keypoint_points_sensor @ predicted[:3, :3].T
            + predicted[:3, 3]
        )
        bins = int(previous.canonical_bins_per_side)
        maximum_range = float(previous.maximum_ground_range_m)
        columns = np.where(
            predicted_points[:, 1] >= 0.0,
            bins
            - 1.0
            - np.abs(predicted_points[:, 1]) / maximum_range * bins,
            bins
            + np.abs(predicted_points[:, 1]) / maximum_range * bins,
        )
        rows = (
            (predicted_points[:, 0] - centres[0])
            / spacing
            * float(self.config.image_row_scale)
        )
        initial_pixels = np.column_stack((columns, rows))
        height, width = previous.image.shape
        valid_initial = (
            np.isfinite(initial_pixels).all(axis=1)
            & (initial_pixels[:, 0] >= 0.0)
            & (initial_pixels[:, 0] < width)
            & (initial_pixels[:, 1] >= 0.0)
            & (initial_pixels[:, 1] < height)
        )
        source_indices = np.flatnonzero(valid_initial)
        if source_indices.size < int(self.config.minimum_matches):
            return []
        source = np.asarray(
            current.keypoint_pixels[source_indices], dtype=np.float32
        ).reshape(-1, 1, 2)
        initial = np.asarray(
            initial_pixels[source_indices], dtype=np.float32
        ).reshape(-1, 1, 2)
        cv2 = self._opencv()
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )
        window = (
            int(self.config.klt_window_width),
            int(self.config.klt_window_height),
        )
        target, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            current.image,
            previous.image,
            source,
            initial.copy(),
            winSize=window,
            maxLevel=int(self.config.klt_maximum_level),
            criteria=criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            minEigThreshold=1.0e-4,
        )
        if target is None or forward_status is None:
            return []
        returned, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
            previous.image,
            current.image,
            target,
            source.copy(),
            winSize=window,
            maxLevel=int(self.config.klt_maximum_level),
            criteria=criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            minEigThreshold=1.0e-4,
        )
        if returned is None or reverse_status is None:
            return []
        target_pixels = target[:, 0].astype(np.float64)
        round_trip = np.linalg.norm(
            returned[:, 0].astype(np.float64) - source[:, 0], axis=1
        )
        valid = (
            (forward_status[:, 0] > 0)
            & (reverse_status[:, 0] > 0)
            & (
                round_trip
                <= float(self.config.klt_forward_backward_limit_px)
            )
            & np.isfinite(target_pixels).all(axis=1)
            & (target_pixels[:, 0] >= 0.0)
            & (target_pixels[:, 0] < width)
            & (target_pixels[:, 1] >= 0.0)
            & (target_pixels[:, 1] < height)
        )
        candidates = np.flatnonzero(valid)
        candidates = candidates[
            np.argsort(round_trip[candidates], kind="stable")
        ]
        previous_pixels = np.asarray(
            previous.keypoint_pixels, dtype=np.float64
        )
        used_previous: set[int] = set()
        pairs: list[tuple[int, int]] = []
        lift_radius = float(self.config.klt_lift_radius_px)
        for candidate in candidates:
            current_index = int(source_indices[candidate])
            tracked = target_pixels[candidate]
            same_side = (
                (previous_pixels[:, 0] < bins) == (tracked[0] < bins)
            )
            distances = np.linalg.norm(
                previous_pixels - tracked[None, :], axis=1
            )
            distances[~same_side] = np.inf
            previous_index = int(np.argmin(distances))
            if (
                not np.isfinite(distances[previous_index])
                or distances[previous_index] > lift_radius
                or previous_index in used_previous
            ):
                continue
            pairs.append((current_index, previous_index))
            used_previous.add(previous_index)
        return pairs

    def _estimate_pair(
        self,
        previous: _FeatureFrame,
        current: _FeatureFrame,
        predicted_transform: np.ndarray | None = None,
    ) -> _PairEstimate:
        minimum = int(self.config.minimum_matches)
        if previous.descriptors.shape[0] < minimum or current.descriptors.shape[0] < minimum:
            return _PairEstimate(False, "insufficient_features", 0, 0, 0.0, None, None)
        current_descriptors = np.ascontiguousarray(
            current.descriptors, dtype=np.float32
        )
        previous_descriptors = np.ascontiguousarray(
            previous.descriptors, dtype=np.float32
        )
        if predicted_transform is None:
            cv2 = self._opencv()
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            forward_knn = matcher.knnMatch(
                current_descriptors, previous_descriptors, k=2
            )
            reverse_knn = matcher.knnMatch(
                previous_descriptors, current_descriptors, k=2
            )

            def ratio_matches(knn):
                output = {}
                for candidates in knn:
                    if len(candidates) < 2:
                        continue
                    best, second = candidates
                    if (
                        best.distance
                        < float(self.config.descriptor_ratio) * second.distance
                        and best.distance
                        <= float(self.config.descriptor_maximum_distance)
                    ):
                        output[int(best.queryIdx)] = int(best.trainIdx)
                return output

            forward = ratio_matches(forward_knn)
            reverse = ratio_matches(reverse_knn)
            pairs = [
                (current_index, previous_index)
                for current_index, previous_index in forward.items()
                if reverse.get(previous_index) == current_index
            ]
        elif self.config.association_method == "klt":
            pairs = self._klt_pairs(
                previous, current, predicted_transform
            )
        else:
            predicted = np.asarray(
                predicted_transform, dtype=np.float64
            ).reshape(4, 4)
            predicted_current = (
                current.keypoint_points_sensor @ predicted[:3, :3].T
                + predicted[:3, 3]
            )
            delta = (
                predicted_current[:, None, :]
                - previous.keypoint_points_sensor[None, :, :]
            )
            scales = np.array(
                [
                    self.config.association_along_track_std_m,
                    self.config.association_cross_track_std_m,
                    self.config.association_vertical_std_m,
                ],
                dtype=np.float64,
            )
            geometry_distance = np.linalg.norm(delta / scales, axis=2)
            # Patch descriptors are unit-normalized.  The matrix expression
            # is both faster and deterministic compared with nested matchers.
            descriptor_squared = np.maximum(
                0.0,
                2.0
                - 2.0
                * np.asarray(current_descriptors, dtype=np.float64)
                @ np.asarray(previous_descriptors, dtype=np.float64).T,
            )
            descriptor_distance = np.sqrt(descriptor_squared)
            valid = (
                geometry_distance
                <= self.config.association_maximum_normalized_distance
            ) & (
                descriptor_distance
                <= self.config.descriptor_maximum_distance
            )
            cost = descriptor_squared + (
                self.config.association_geometry_weight
                * geometry_distance**2
            )
            cost[~valid] = np.inf

            def best_assignments(matrix: np.ndarray) -> dict[int, int]:
                assignments: dict[int, int] = {}
                for query_index, row in enumerate(matrix):
                    finite = np.flatnonzero(np.isfinite(row))
                    if finite.size == 0:
                        continue
                    order = finite[np.argsort(row[finite], kind="stable")]
                    best = int(order[0])
                    if order.size > 1 and not (
                        row[best]
                        < float(self.config.descriptor_ratio) ** 2
                        * row[int(order[1])]
                    ):
                        continue
                    assignments[int(query_index)] = best
                return assignments

            forward = best_assignments(cost)
            reverse = best_assignments(cost.T)
            pairs = [
                (current_index, previous_index)
                for current_index, previous_index in forward.items()
                if reverse.get(previous_index) == current_index
            ]
        if len(pairs) < minimum:
            return _PairEstimate(
                False, "insufficient_mutual_matches", len(pairs), 0, 0.0, None, None
            )
        current_points = np.asarray(
            [current.keypoint_points_sensor[index] for index, _ in pairs]
        )
        previous_points = np.asarray(
            [previous.keypoint_points_sensor[index] for _, index in pairs]
        )
        seed = int(round(previous.timestamp * 1.0e6)) ^ (
            int(round(current.timestamp * 1.0e6)) << 1
        )
        gravity_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if previous.rotation_world_sensor is not None:
            gravity_axis = (
                project_to_so3(previous.rotation_world_sensor).T
                @ gravity_axis
            )
        ransac_started = time.perf_counter()
        if predicted_transform is None:
            self._cpu_ransac_calls += 1
            transform, rmse, inlier_mask = _estimate_planar_rigid_ransac(
                current_points,
                previous_points,
                iterations=self.config.ransac_iterations,
                residual_limit_m=self.config.ransac_residual_m,
                minimum_baseline_m=self.config.minimum_pair_baseline_m,
                random_seed=seed,
                along_track_weight=self.config.along_track_residual_weight,
                vertical_weight=self.config.vertical_residual_weight,
            )
        else:
            backend = self._active_compute_backend
            if backend == "cuda":
                self._cuda_ransac_calls += 1
            else:
                self._cpu_ransac_calls += 1
            try:
                transform, rmse, inlier_mask = (
                    _estimate_prediction_centered_ransac(
                        current_points,
                        previous_points,
                        predicted_transform,
                        gravity_axis,
                        iterations=self.config.ransac_iterations,
                        residual_limit_m=self.config.ransac_residual_m,
                        minimum_baseline_m=self.config.minimum_pair_baseline_m,
                        random_seed=seed,
                        along_track_weight=(
                            self.config.along_track_residual_weight
                        ),
                        vertical_weight=self.config.vertical_residual_weight,
                        compute_backend=backend,
                        torch_module=self._torch,
                    )
                )
            except RuntimeError as exc:
                if (
                    self._requested_compute_backend != "auto"
                    or backend != "cuda"
                ):
                    raise
                self._cuda_fallbacks += 1
                self._active_compute_backend = "cpu"
                self._compute_device = "CPU (automatic CUDA fallback)"
                self._cpu_ransac_calls += 1
                warnings.warn(
                    "feature CUDA RANSAC failed; auto backend is falling back "
                    f"to CPU: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                transform, rmse, inlier_mask = (
                    _estimate_prediction_centered_ransac(
                        current_points,
                        previous_points,
                        predicted_transform,
                        gravity_axis,
                        iterations=self.config.ransac_iterations,
                        residual_limit_m=self.config.ransac_residual_m,
                        minimum_baseline_m=self.config.minimum_pair_baseline_m,
                        random_seed=seed,
                        along_track_weight=(
                            self.config.along_track_residual_weight
                        ),
                        vertical_weight=self.config.vertical_residual_weight,
                        compute_backend="cpu",
                    )
                )
        self._ransac_wall_s += time.perf_counter() - ransac_started
        inlier_count = int(np.count_nonzero(inlier_mask))
        if transform is None or inlier_count < minimum:
            return _PairEstimate(
                False,
                "insufficient_geometric_inliers",
                len(pairs),
                inlier_count,
                rmse,
                None,
                None,
            )
        if (
            predicted_transform is not None
            and self.config.state_update_policy == "along_track"
        ):
            # KLT consensus may fit nuisance cross-track/yaw components to
            # explain sampling on a continuous seabed surface.  They remain
            # useful for identifying inliers, but the eight-row V2 aperture
            # does not make them independent state observations.  Project the
            # accepted mean back onto the one supported acoustic direction.
            fitted_correction = transform @ np.linalg.inv(
                np.asarray(predicted_transform, dtype=np.float64).reshape(4, 4)
            )
            along_track_correction = float(fitted_correction[0, 3])
            cauchy_scale = float(
                self.config.prediction_correction_scale_m
            )
            consistency_weight = 1.0 / (
                1.0 + (along_track_correction / cauchy_scale) ** 2
            )
            along_track_correction *= consistency_weight
            transform = (
                _gravity_yaw_correction(
                    np.array([along_track_correction, 0.0, 0.0]),
                    gravity_axis,
                )
                @ predicted_transform
            )
        if predicted_transform is None:
            covariance, eigenvalues = _relative_covariance(
                current_points[inlier_mask],
                transform,
                rmse,
                position_floor_m=self.config.position_noise_floor_m,
                yaw_floor_rad=math.radians(
                    self.config.yaw_noise_floor_deg
                ),
                effective_independent_points=(
                    self.config.effective_independent_matches
                ),
            )
        else:
            transformed_current = (
                current_points[inlier_mask] @ transform[:3, :3].T
                + transform[:3, 3]
            )
            covariance, eigenvalues = _prediction_centered_covariance(
                transformed_current,
                gravity_axis,
                rmse,
                position_floor_m=self.config.position_noise_floor_m,
                cross_track_position_floor_m=(
                    self.config.cross_track_position_noise_floor_m
                ),
                yaw_floor_rad=math.radians(
                    self.config.yaw_noise_floor_deg
                ),
                effective_independent_points=(
                    self.config.effective_independent_matches
                ),
                along_track_weight=(
                    self.config.along_track_residual_weight
                ),
                vertical_weight=self.config.vertical_residual_weight,
            )
        if (
            covariance is None
            or eigenvalues[0] < self.config.minimum_information_eigenvalue
        ):
            return _PairEstimate(
                False,
                "rank_deficient_geometry",
                len(pairs),
                inlier_count,
                rmse,
                None,
                None,
            )
        return _PairEstimate(
            True,
            None,
            len(pairs),
            inlier_count,
            rmse,
            transform,
            covariance,
        )

    def _absolute_measurement(
        self, relative_sensor: np.ndarray, relative_covariance: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray]:
        if self._anchor_world_body is None or self._anchor_planar_covariance is None:
            raise RuntimeError("feature odometry anchor is not initialized")
        sensor_from_body = np.linalg.inv(self._body_from_sensor)
        anchor_world_sensor = (
            self._anchor_world_body @ self._body_from_sensor
        )
        gravity_axis = (
            anchor_world_sensor[:3, :3].T
            @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )

        def evaluate(correction: np.ndarray) -> np.ndarray:
            # The nominal chain is full SE(3): propagated roll, pitch, depth,
            # and sensor lever arm are never flattened.  Covariance lives in
            # a left perturbation tangent at the subframe anchor.
            sensor_transform = (
                _gravity_yaw_correction(correction, gravity_axis)
                @ relative_sensor
            )
            body_relative = (
                self._body_from_sensor @ sensor_transform @ sensor_from_body
            )
            return _pose2_from_matrix(self._anchor_world_body @ body_relative)

        nominal_correction = np.zeros(3, dtype=np.float64)
        measured = evaluate(nominal_correction)
        jacobian = np.zeros((3, 3), dtype=np.float64)
        steps = np.array([1.0e-4, 1.0e-4, 1.0e-6], dtype=np.float64)
        for index, step in enumerate(steps):
            plus = nominal_correction.copy()
            minus = nominal_correction.copy()
            plus[index] += step
            minus[index] -= step
            difference = evaluate(plus) - evaluate(minus)
            difference[2] = float(wrap_angle(difference[2]))
            jacobian[:, index] = difference / (2.0 * step)
        covariance = jacobian @ relative_covariance @ jacobian.T
        covariance += (
            float(self.config.anchor_covariance_scale)
            * self._anchor_planar_covariance
        )
        covariance = 0.5 * (covariance + covariance.T)
        return measured[:2], float(measured[2]), covariance

    def process(
        self,
        ping: "BathymetricSideScanPing",
        *,
        rotation_world_body: np.ndarray,
        position_world_body: np.ndarray,
        state_covariance: np.ndarray,
    ) -> FeatureOdometryResult:
        """Consume one ping; call :meth:`finalize` after state fusion."""

        if self._pending_frame is not None:
            raise RuntimeError("finalize must be called before the next feature ping")
        frame_started = time.perf_counter()
        try:
            frame = self._canonical_frame(ping)
        except (ValueError, RuntimeError) as exc:
            reason = str(exc)
            self._frames += 1
            self._rejection_reasons[reason] += 1
            self._pending_frame = None
            self._previous_frame = None
            self._restart_anchor_after_finalize = True
            return FeatureOdometryResult(
                False, False, reason, 0, 0, 0.0, 0.0, 0, None, None, None, None, None
            )
        finally:
            self._frame_build_calls += 1
            self._frame_build_wall_s += time.perf_counter() - frame_started

        self._frames += 1
        self._pending_frame = frame
        predicted_world_body = _pose_matrix(
            rotation_world_body, position_world_body
        )
        current_world_sensor = (
            predicted_world_body @ self._body_from_sensor
        )
        frame.rotation_world_sensor = current_world_sensor[:3, :3].copy()
        if self._previous_frame is None:
            self._restart_anchor_after_finalize = True
            return FeatureOdometryResult(
                False,
                False,
                "bootstrap",
                0,
                0,
                0.0,
                0.0,
                1,
                None,
                None,
                None,
                None,
                None,
            )

        self._pair_attempts += 1
        predicted_transform = None
        if self._previous_world_body is not None:
            previous_world_sensor = (
                self._previous_world_body @ self._body_from_sensor
            )
            predicted_transform = (
                np.linalg.inv(previous_world_sensor) @ current_world_sensor
            )
        pair_started = time.perf_counter()
        try:
            pair = self._estimate_pair(
                self._previous_frame, frame, predicted_transform
            )
        finally:
            self._pair_estimation_wall_s += (
                time.perf_counter() - pair_started
            )
        if not pair.accepted:
            self._rejected_pairs += 1
            reason = pair.reason or "unspecified"
            self._rejection_reasons[reason] += 1
            self._restart_anchor_after_finalize = True
            return FeatureOdometryResult(
                False,
                False,
                reason,
                pair.matches,
                pair.inliers,
                pair.inliers / max(1, pair.matches),
                pair.rmse_m,
                1,
                None,
                None,
                None,
                None,
                None,
            )

        self._accepted_pairs += 1
        inlier_ratio = pair.inliers / max(1, pair.matches)
        self._inlier_ratio_sum += inlier_ratio
        self._pair_rmse_sum += pair.rmse_m
        if self._chain_ping_count == 0:
            # The anchor is the finalized state belonging to previous_frame.
            self._chain_ping_count = 1
        self._chain_covariance = _compose_planar_covariance(
            self._chain_transform,
            self._chain_covariance,
            pair.transform_previous_sensor_current_sensor,
            pair.covariance,
        )
        self._chain_transform = (
            self._chain_transform @ pair.transform_previous_sensor_current_sensor
        )
        self._chain_ping_count += 1
        ready = self._chain_ping_count >= int(self.config.subframe_pings)
        if not ready:
            return FeatureOdometryResult(
                True,
                False,
                None,
                pair.matches,
                pair.inliers,
                inlier_ratio,
                pair.rmse_m,
                self._chain_ping_count,
                self._chain_transform.copy(),
                self._chain_covariance.copy(),
                None,
                None,
                None,
            )

        position_xy, yaw, covariance_world = self._absolute_measurement(
            self._chain_transform, self._chain_covariance
        )
        if self._anchor_world_body is None:
            raise RuntimeError("feature odometry anchor is not initialized")
        anchor_world_sensor = (
            self._anchor_world_body @ self._body_from_sensor
        )
        along_track_axis = anchor_world_sensor[:2, 0].copy()
        along_track_axis /= max(
            float(np.linalg.norm(along_track_axis)), 1.0e-12
        )
        along_track_projection = float(along_track_axis @ position_xy)
        self._completed_subframes += 1
        self._restart_anchor_after_finalize = True
        return FeatureOdometryResult(
            True,
            True,
            None,
            pair.matches,
            pair.inliers,
            inlier_ratio,
            pair.rmse_m,
            self._chain_ping_count,
            self._chain_transform.copy(),
            self._chain_covariance.copy(),
            position_xy,
            yaw,
            covariance_world,
            along_track_axis,
            along_track_projection,
        )

    def finalize(
        self,
        *,
        rotation_world_body: np.ndarray,
        position_world_body: np.ndarray,
        state_covariance: np.ndarray,
    ) -> None:
        """Commit the current frame after the caller finishes state fusion."""

        if self._restart_anchor_after_finalize:
            self._anchor_world_body = _pose_matrix(
                rotation_world_body, position_world_body
            )
            self._anchor_planar_covariance = _planar_covariance_from_state(
                rotation_world_body, state_covariance
            )
            self._chain_transform = np.eye(4, dtype=np.float64)
            self._chain_covariance = np.zeros((3, 3), dtype=np.float64)
            self._chain_ping_count = 1 if self._pending_frame is not None else 0
        if self._pending_frame is not None:
            finalized_world_sensor = (
                _pose_matrix(rotation_world_body, position_world_body)
                @ self._body_from_sensor
            )
            self._pending_frame.rotation_world_sensor = (
                finalized_world_sensor[:3, :3].copy()
            )
        self._previous_frame = self._pending_frame
        self._previous_world_body = _pose_matrix(
            rotation_world_body, position_world_body
        )
        self._pending_frame = None
        self._restart_anchor_after_finalize = False

    @property
    def active_compute_backend(self) -> str:
        """Resolved feature-consensus backend used by subsequent pings."""

        return self._active_compute_backend

    @property
    def compute_device(self) -> str:
        """Human-readable compute device for logs and experiment metadata."""

        return self._compute_device

    def diagnostics(self) -> FeatureOdometryDiagnostics:
        ransac_calls = (
            self._cuda_ransac_calls
            + self._cpu_ransac_calls
            - self._cuda_fallbacks
        )
        return FeatureOdometryDiagnostics(
            frames=self._frames,
            pair_attempts=self._pair_attempts,
            accepted_pairs=self._accepted_pairs,
            rejected_pairs=self._rejected_pairs,
            completed_subframes=self._completed_subframes,
            rejection_reasons=dict(self._rejection_reasons),
            mean_pair_inlier_ratio=(
                self._inlier_ratio_sum / max(1, self._accepted_pairs)
            ),
            mean_pair_rmse_m=(
                self._pair_rmse_sum / max(1, self._accepted_pairs)
            ),
            requested_compute_backend=self._requested_compute_backend,
            active_compute_backend=self._active_compute_backend,
            compute_device=self._compute_device,
            cuda_ransac_calls=self._cuda_ransac_calls,
            cpu_ransac_calls=self._cpu_ransac_calls,
            cuda_fallbacks=self._cuda_fallbacks,
            mean_frame_build_ms=(
                1.0e3
                * self._frame_build_wall_s
                / max(1, self._frame_build_calls)
            ),
            mean_pair_estimation_ms=(
                1.0e3
                * self._pair_estimation_wall_s
                / max(1, self._pair_attempts)
            ),
            mean_ransac_ms=(
                1.0e3 * self._ransac_wall_s / max(1, ransac_calls)
            ),
        )


__all__ = [
    "FeatureOdometryDiagnostics",
    "FeatureOdometryResult",
    "FeatureSonarOdometry",
    "FeatureSonarOdometryConfig",
]
