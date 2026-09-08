"""Dependency-light metrics shared by SLAM simulations and tests."""

from __future__ import annotations

import numpy as np


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
