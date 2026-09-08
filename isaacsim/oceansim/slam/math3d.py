"""Small Lie-group helpers shared by the underwater SLAM implementation."""

from __future__ import annotations

import math

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the matrix that implements ``vector x other``."""

    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def so3_exp(tangent: np.ndarray) -> np.ndarray:
    """Exponential map from a rotation vector to SO(3)."""

    tangent = np.asarray(tangent, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(tangent))
    tangent_skew = skew(tangent)
    if theta < 1.0e-8:
        return (
            np.eye(3)
            + tangent_skew
            + 0.5 * tangent_skew @ tangent_skew
        )
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3) + a * tangent_skew + b * tangent_skew @ tangent_skew


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Logarithmic map from SO(3) to a rotation vector."""

    rotation = project_to_so3(rotation)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1.0e-8:
        return 0.5 * vee
    return 0.5 * theta / math.sin(theta) * vee


def project_to_so3(rotation: np.ndarray) -> np.ndarray:
    """Project a nearly orthogonal matrix back onto SO(3)."""

    u, _, vt = np.linalg.svd(
        np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    )
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build ``R_world_body`` from intrinsic roll, pitch, yaw angles."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rpy_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw from ``R_world_body``."""

    rotation = project_to_so3(rotation)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-7:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def wrap_angle(angle: float | np.ndarray):
    """Wrap radians to the half-open interval ``[-pi, pi)``."""

    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def transform_points(
    rotation: np.ndarray,
    translation: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Apply an SE(3) transform to an ``Nx3`` point array."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ np.asarray(rotation, dtype=np.float64).reshape(3, 3).T + np.asarray(
        translation, dtype=np.float64
    ).reshape(1, 3)


def inverse_transform(
    rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Invert an SE(3) transform represented by rotation and translation."""

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    inverse_rotation = rotation.T
    return inverse_rotation, -(inverse_rotation @ translation)


def relative_pose(
    rotation_a: np.ndarray,
    position_a: np.ndarray,
    rotation_b: np.ndarray,
    position_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pose of frame B expressed in frame A."""

    rotation_a = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3)
    rotation_b = np.asarray(rotation_b, dtype=np.float64).reshape(3, 3)
    position_a = np.asarray(position_a, dtype=np.float64).reshape(3)
    position_b = np.asarray(position_b, dtype=np.float64).reshape(3)
    rotation_ab = rotation_a.T @ rotation_b
    position_ab = rotation_a.T @ (position_b - position_a)
    return rotation_ab, position_ab
