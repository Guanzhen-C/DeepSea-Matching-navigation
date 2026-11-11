"""Dependency-light metrics shared by SLAM simulations and tests."""

from __future__ import annotations

from typing import Any

import numpy as np


def sanitize_for_strict_json(value: Any) -> tuple[Any, int]:
    """Return a JSON-safe copy and its non-finite replacement count.

    Python's JSON encoder emits ``NaN`` and ``Infinity`` by default even
    though neither token is valid JSON.  Long-run diagnostic records can
    legitimately contain unavailable covariance or chi-square values.  Keep
    that distinction explicit as ``null`` without weakening serialization of
    the rest of the run artifact.
    """

    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        replacements = 0
        for key, item in value.items():
            safe_item, item_replacements = sanitize_for_strict_json(item)
            result[key] = safe_item
            replacements += item_replacements
        return result, replacements
    if isinstance(value, (list, tuple)):
        result = []
        replacements = 0
        for item in value:
            safe_item, item_replacements = sanitize_for_strict_json(item)
            result.append(safe_item)
            replacements += item_replacements
        return result, replacements
    if isinstance(value, np.ndarray):
        return sanitize_for_strict_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value), 0
    if isinstance(value, np.integer):
        return int(value), 0
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None, 1
        return numeric, 0
    return value, 0


def se2_relative_pose_error_by_distance(
    estimate_poses: np.ndarray,
    truth_poses: np.ndarray,
    cumulative_truth_distance_m: np.ndarray,
    interval_m: float,
) -> dict[str, float | int | bool | None]:
    """Return gauge-invariant planar relative-pose error by route distance.

    Poses use ``[x, y, yaw]`` with yaw in radians.  Estimate and truth motion
    are each expressed in their own starting body frame, so one constant SE(2)
    transform between the two trajectories has no effect on this metric.
    """

    estimate = np.asarray(estimate_poses, dtype=np.float64).reshape(-1, 3)
    truth = np.asarray(truth_poses, dtype=np.float64).reshape(-1, 3)
    distance = np.asarray(
        cumulative_truth_distance_m, dtype=np.float64
    ).reshape(-1)
    if estimate.shape != truth.shape or estimate.shape[0] != distance.size:
        raise ValueError("SE(2) RPE tracks must have equal length")
    if not np.isfinite(float(interval_m)) or interval_m <= 0.0:
        raise ValueError("SE(2) RPE interval must be positive and finite")
    if (
        distance.size == 0
        or not np.isfinite(distance).all()
        or not np.isfinite(estimate).all()
        or not np.isfinite(truth).all()
        or np.any(np.diff(distance) < 0.0)
    ):
        raise ValueError("SE(2) RPE inputs must be finite and distance monotonic")

    target_distance = distance + float(interval_m)
    ending = np.searchsorted(distance, target_distance, side="left")
    starting = np.arange(distance.size, dtype=np.int64)
    available = ending < distance.size
    starting = starting[available]
    ending = ending[available]
    if starting.size == 0:
        return {
            "available": False,
            "samples": 0,
            "requested_interval_m": float(interval_m),
            "mean_actual_interval_m": None,
            "rmse_translation_m": None,
            "mean_translation_m": None,
            "p95_translation_m": None,
            "max_translation_m": None,
            "rmse_yaw_deg": None,
            "mean_abs_yaw_deg": None,
            "p95_abs_yaw_deg": None,
            "max_abs_yaw_deg": None,
        }

    def relative_motion(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = poses[ending, :2] - poses[starting, :2]
        yaw = poses[starting, 2]
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        translation = np.column_stack(
            (
                cosine * delta[:, 0] + sine * delta[:, 1],
                -sine * delta[:, 0] + cosine * delta[:, 1],
            )
        )
        relative_yaw = (
            poses[ending, 2] - poses[starting, 2] + np.pi
        ) % (2.0 * np.pi) - np.pi
        return translation, relative_yaw

    estimate_translation, estimate_yaw = relative_motion(estimate)
    truth_translation, truth_yaw = relative_motion(truth)
    translation_error = np.linalg.norm(
        estimate_translation - truth_translation, axis=1
    )
    yaw_error_deg = np.abs(
        np.rad2deg(
            (estimate_yaw - truth_yaw + np.pi) % (2.0 * np.pi) - np.pi
        )
    )
    return {
        "available": True,
        "samples": int(starting.size),
        "requested_interval_m": float(interval_m),
        "mean_actual_interval_m": float(
            np.mean(distance[ending] - distance[starting])
        ),
        "rmse_translation_m": float(
            np.sqrt(np.mean(translation_error**2))
        ),
        "mean_translation_m": float(np.mean(translation_error)),
        "p95_translation_m": float(
            np.percentile(translation_error, 95.0)
        ),
        "max_translation_m": float(np.max(translation_error)),
        "rmse_yaw_deg": float(np.sqrt(np.mean(yaw_error_deg**2))),
        "mean_abs_yaw_deg": float(np.mean(yaw_error_deg)),
        "p95_abs_yaw_deg": float(np.percentile(yaw_error_deg, 95.0)),
        "max_abs_yaw_deg": float(np.max(yaw_error_deg)),
    }


def se3_relative_pose_error_by_distance(
    estimate_positions: np.ndarray,
    estimate_rotations: np.ndarray,
    truth_positions: np.ndarray,
    truth_rotations: np.ndarray,
    cumulative_truth_distance_m: np.ndarray,
    interval_m: float,
) -> dict[str, float | int | bool | None]:
    """Return gauge-invariant SE(3) RPE over a truth-distance interval.

    Each motion is expressed in its own starting body frame before the truth
    relative transform is removed.  A single rigid translation/rotation of an
    otherwise exact estimated trajectory therefore produces zero error.
    """

    estimate_xyz = np.asarray(
        estimate_positions, dtype=np.float64
    ).reshape(-1, 3)
    truth_xyz = np.asarray(truth_positions, dtype=np.float64).reshape(-1, 3)
    estimate_rotation = np.asarray(
        estimate_rotations, dtype=np.float64
    ).reshape(-1, 3, 3)
    truth_rotation = np.asarray(
        truth_rotations, dtype=np.float64
    ).reshape(-1, 3, 3)
    distance = np.asarray(
        cumulative_truth_distance_m, dtype=np.float64
    ).reshape(-1)
    count = distance.size
    if (
        estimate_xyz.shape != truth_xyz.shape
        or estimate_xyz.shape[0] != count
        or estimate_rotation.shape != truth_rotation.shape
        or estimate_rotation.shape[0] != count
    ):
        raise ValueError("SE(3) RPE tracks must have equal length")
    if not np.isfinite(float(interval_m)) or interval_m <= 0.0:
        raise ValueError("SE(3) RPE interval must be positive and finite")
    if (
        count == 0
        or not np.isfinite(distance).all()
        or not np.isfinite(estimate_xyz).all()
        or not np.isfinite(truth_xyz).all()
        or not np.isfinite(estimate_rotation).all()
        or not np.isfinite(truth_rotation).all()
        or np.any(np.diff(distance) < 0.0)
    ):
        raise ValueError("SE(3) RPE inputs must be finite and distance monotonic")

    target_distance = distance + float(interval_m)
    ending = np.searchsorted(distance, target_distance, side="left")
    starting = np.arange(count, dtype=np.int64)
    available = ending < count
    starting = starting[available]
    ending = ending[available]
    if starting.size == 0:
        return {
            "available": False,
            "samples": 0,
            "requested_interval_m": float(interval_m),
            "mean_actual_interval_m": None,
            "rmse_translation_m": None,
            "mean_translation_m": None,
            "p95_translation_m": None,
            "max_translation_m": None,
            "rmse_rotation_deg": None,
            "mean_rotation_deg": None,
            "p95_rotation_deg": None,
            "max_rotation_deg": None,
        }

    estimate_relative_rotation = np.einsum(
        "nji,njk->nik",
        estimate_rotation[starting],
        estimate_rotation[ending],
    )
    truth_relative_rotation = np.einsum(
        "nji,njk->nik",
        truth_rotation[starting],
        truth_rotation[ending],
    )
    estimate_relative_translation = np.einsum(
        "nji,nj->ni",
        estimate_rotation[starting],
        estimate_xyz[ending] - estimate_xyz[starting],
    )
    truth_relative_translation = np.einsum(
        "nji,nj->ni",
        truth_rotation[starting],
        truth_xyz[ending] - truth_xyz[starting],
    )
    relative_error_rotation = np.einsum(
        "nji,njk->nik",
        truth_relative_rotation,
        estimate_relative_rotation,
    )
    relative_error_translation = np.einsum(
        "nji,nj->ni",
        truth_relative_rotation,
        estimate_relative_translation - truth_relative_translation,
    )
    translation_error = np.linalg.norm(relative_error_translation, axis=1)
    trace = np.trace(relative_error_rotation, axis1=1, axis2=2)
    rotation_error_deg = np.rad2deg(
        np.arccos(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))
    )
    return {
        "available": True,
        "samples": int(starting.size),
        "requested_interval_m": float(interval_m),
        "mean_actual_interval_m": float(
            np.mean(distance[ending] - distance[starting])
        ),
        "rmse_translation_m": float(
            np.sqrt(np.mean(translation_error**2))
        ),
        "mean_translation_m": float(np.mean(translation_error)),
        "p95_translation_m": float(
            np.percentile(translation_error, 95.0)
        ),
        "max_translation_m": float(np.max(translation_error)),
        "rmse_rotation_deg": float(
            np.sqrt(np.mean(rotation_error_deg**2))
        ),
        "mean_rotation_deg": float(np.mean(rotation_error_deg)),
        "p95_rotation_deg": float(
            np.percentile(rotation_error_deg, 95.0)
        ),
        "max_rotation_deg": float(np.max(rotation_error_deg)),
    }


def map_to_odom_change_metrics(
    timestamps: np.ndarray,
    local_positions: np.ndarray,
    local_rotations: np.ndarray,
    global_positions: np.ndarray,
    global_rotations: np.ndarray,
) -> dict[str, float | int | None]:
    """Audit changes in the online planar ``map->odom`` correction."""

    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    local_xyz = np.asarray(local_positions, dtype=np.float64).reshape(-1, 3)
    global_xyz = np.asarray(global_positions, dtype=np.float64).reshape(-1, 3)
    local_rotation = np.asarray(local_rotations, dtype=np.float64).reshape(
        -1, 3, 3
    )
    global_rotation = np.asarray(global_rotations, dtype=np.float64).reshape(
        -1, 3, 3
    )
    count = times.size
    if not (
        local_xyz.shape[0]
        == global_xyz.shape[0]
        == local_rotation.shape[0]
        == global_rotation.shape[0]
        == count
    ):
        raise ValueError("map-to-odom audit tracks must have equal length")
    finite_samples = (
        np.isfinite(times)
        & np.isfinite(local_xyz).all(axis=1)
        & np.isfinite(global_xyz).all(axis=1)
        & np.isfinite(local_rotation).all(axis=(1, 2))
        & np.isfinite(global_rotation).all(axis=(1, 2))
    )
    if count < 2 or not np.all(finite_samples):
        return {
            "samples": int(count),
            "nonfinite_samples": int(np.count_nonzero(~finite_samples)),
            "maximum_translation_change_m": None,
            "p99_translation_change_m": None,
            "maximum_yaw_change_deg": None,
            "p99_yaw_change_deg": None,
            "maximum_global_position_step_m": None,
            "translation_changes_over_5m": 0,
            "yaw_changes_over_1deg": 0,
            "maximum_change_timestamp_s": None,
        }

    correction_rotation = np.einsum(
        "nij,nkj->nik", global_rotation, local_rotation
    )
    correction_yaw = np.arctan2(
        correction_rotation[:, 1, 0], correction_rotation[:, 0, 0]
    )
    cosine = np.cos(correction_yaw)
    sine = np.sin(correction_yaw)
    rotated_local_xy = np.column_stack(
        (
            cosine * local_xyz[:, 0] - sine * local_xyz[:, 1],
            sine * local_xyz[:, 0] + cosine * local_xyz[:, 1],
        )
    )
    correction_translation = global_xyz[:, :2] - rotated_local_xy
    translation_change = np.linalg.norm(
        np.diff(correction_translation, axis=0), axis=1
    )
    yaw_change = np.diff(correction_yaw)
    yaw_change = np.abs(
        (yaw_change + np.pi) % (2.0 * np.pi) - np.pi
    )
    global_position_step = np.linalg.norm(np.diff(global_xyz, axis=0), axis=1)
    maximum_index = int(np.argmax(translation_change))
    return {
        "samples": int(count),
        "nonfinite_samples": 0,
        "maximum_translation_change_m": float(np.max(translation_change)),
        "p99_translation_change_m": float(
            np.percentile(translation_change, 99.0)
        ),
        "maximum_yaw_change_deg": float(np.rad2deg(np.max(yaw_change))),
        "p99_yaw_change_deg": float(
            np.rad2deg(np.percentile(yaw_change, 99.0))
        ),
        "maximum_global_position_step_m": float(
            np.max(global_position_step)
        ),
        "translation_changes_over_5m": int(
            np.count_nonzero(translation_change > 5.0)
        ),
        "yaw_changes_over_1deg": int(
            np.count_nonzero(yaw_change > np.deg2rad(1.0))
        ),
        "maximum_change_timestamp_s": float(times[maximum_index + 1]),
    }
