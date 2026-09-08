"""Sensor-only initialization helpers for acoustic-inertial navigation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math3d import rotation_from_rpy


@dataclass(frozen=True)
class NavigationInitialization:
    """Initial pose and velocity expressed in the chosen local gauge."""

    rotation: np.ndarray
    position: np.ndarray
    velocity: np.ndarray


def initialize_from_imu_dvl_depth(
    specific_force_body_mps2: np.ndarray,
    velocity_body_mps: np.ndarray,
    depth_m: float | np.ndarray,
    *,
    yaw_gauge_rad: float = 0.0,
    horizontal_origin_m: np.ndarray | tuple[float, float] = (0.0, 0.0),
) -> NavigationInitialization:
    """Initialize roll/pitch, velocity, and depth from allowed AUV sensors.

    ``specific_force_body_mps2`` and ``velocity_body_mps`` may contain one
    sample or an ``Nx3`` startup window.  The acceleration window must be
    quasi-static so its mean direction represents gravity.  Absolute yaw and
    horizontal translation are unobservable without a compass, GNSS, or an
    external map; callers therefore choose those local-frame gauge values.
    """

    specific_force = np.asarray(
        specific_force_body_mps2, dtype=np.float64
    ).reshape(-1, 3)
    body_velocity = np.asarray(
        velocity_body_mps, dtype=np.float64
    ).reshape(-1, 3)
    depths = np.asarray(depth_m, dtype=np.float64).reshape(-1)
    if (
        specific_force.shape[0] == 0
        or body_velocity.shape[0] == 0
        or depths.size == 0
        or not np.isfinite(specific_force).all()
        or not np.isfinite(body_velocity).all()
        or not np.isfinite(depths).all()
    ):
        raise ValueError("initialization samples must be finite and non-empty")

    mean_force = np.mean(specific_force, axis=0)
    force_norm = float(np.linalg.norm(mean_force))
    if force_norm < 1.0:
        raise ValueError("specific-force window does not contain gravity")
    up_body = mean_force / force_norm
    roll = float(np.arctan2(up_body[1], up_body[2]))
    pitch = float(np.arcsin(np.clip(-up_body[0], -1.0, 1.0)))
    rotation = rotation_from_rpy(roll, pitch, float(yaw_gauge_rad))

    origin = np.asarray(horizontal_origin_m, dtype=np.float64).reshape(2)
    if not np.isfinite(origin).all():
        raise ValueError("horizontal origin must be finite")
    position = np.array(
        [origin[0], origin[1], -float(np.median(depths))],
        dtype=np.float64,
    )
    velocity = rotation @ np.mean(body_velocity, axis=0)
    return NavigationInitialization(rotation, position, velocity)


def gauge_anchored_initial_covariance(
    *,
    gyro_bias_std_rad_s: float = 0.003,
    roll_pitch_std_rad: float = 0.005,
    yaw_gauge_std_rad: float = 0.001,
    horizontal_gauge_std_m: float = 0.01,
    depth_std_m: float = 0.03,
    velocity_std_mps: float = 0.03,
    accel_bias_std_mps2: float = 0.04,
) -> np.ndarray:
    """Return a 15-state covariance for sensor-only startup.

    Small yaw/x/y entries fix the otherwise unobservable local-frame gauge;
    they must not be interpreted as absolute heading or position accuracy.
    The remaining entries describe an averaged IMU/DVL/depth startup window.
    """

    values = np.asarray(
        [
            roll_pitch_std_rad,
            roll_pitch_std_rad,
            yaw_gauge_std_rad,
            horizontal_gauge_std_m,
            horizontal_gauge_std_m,
            depth_std_m,
            velocity_std_mps,
            velocity_std_mps,
            velocity_std_mps,
            gyro_bias_std_rad_s,
            gyro_bias_std_rad_s,
            gyro_bias_std_rad_s,
            accel_bias_std_mps2,
            accel_bias_std_mps2,
            accel_bias_std_mps2,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("initial standard deviations must be positive and finite")
    return np.diag(values**2)
