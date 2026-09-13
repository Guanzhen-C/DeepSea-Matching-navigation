"""Continuous planar ``map -> odom`` correction for online SLAM output.

The acoustic ESIKF owns the locally smooth ``odom -> body`` state.  Loop
closures and absolute terrain factors update ``map -> body`` in the pose
graph, sometimes by many metres at one optimizer update.  This module keeps
those responsibilities separate: it follows the graph correction with a
time-constant and physically interpretable rate limits, without ever writing
the correction back into the ESIKF.

The filter is evaluated at the current vehicle pose rather than by directly
interpolating the translation component of ``T_map_odom``.  That distinction
matters on a map whose coordinates are tens of kilometres from the origin: a
small yaw change can otherwise induce a very large transform translation even
though the desired correction at the vehicle is small.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .math3d import rotation_from_rpy, rpy_from_rotation, wrap_angle


def _validated_pose2(value: np.ndarray, *, name: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(3).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (3,)") from exc
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} must be finite")
    pose[2] = float(wrap_angle(pose[2]))
    return pose


def map_to_odom_from_poses(
    map_pose2: np.ndarray, odom_pose2: np.ndarray
) -> np.ndarray:
    """Return ``T_map_odom = T_map_body * inverse(T_odom_body)``."""

    map_pose = _validated_pose2(map_pose2, name="map_pose2")
    odom_pose = _validated_pose2(odom_pose2, name="odom_pose2")
    yaw = float(wrap_angle(map_pose[2] - odom_pose[2]))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotated_odom = np.array(
        [
            cosine * odom_pose[0] - sine * odom_pose[1],
            sine * odom_pose[0] + cosine * odom_pose[1],
        ],
        dtype=np.float64,
    )
    return np.array(
        [
            map_pose[0] - rotated_odom[0],
            map_pose[1] - rotated_odom[1],
            yaw,
        ],
        dtype=np.float64,
    )


def apply_map_to_odom_pose2(
    map_to_odom_pose2: np.ndarray, odom_pose2: np.ndarray
) -> np.ndarray:
    """Apply a planar ``map -> odom`` transform to an odometry pose."""

    transform = _validated_pose2(
        map_to_odom_pose2, name="map_to_odom_pose2"
    )
    odom_pose = _validated_pose2(odom_pose2, name="odom_pose2")
    cosine = math.cos(float(transform[2]))
    sine = math.sin(float(transform[2]))
    return np.array(
        [
            transform[0]
            + cosine * odom_pose[0]
            - sine * odom_pose[1],
            transform[1]
            + sine * odom_pose[0]
            + cosine * odom_pose[1],
            float(wrap_angle(transform[2] + odom_pose[2])),
        ],
        dtype=np.float64,
    )


def apply_map_to_odom_pose(
    map_to_odom_pose2: np.ndarray,
    rotation_odom_body: np.ndarray,
    position_odom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a planar correction while retaining front-end depth/roll/pitch."""

    transform = _validated_pose2(
        map_to_odom_pose2, name="map_to_odom_pose2"
    )
    try:
        rotation = np.asarray(rotation_odom_body, dtype=np.float64).reshape(
            3, 3
        )
        position = np.asarray(position_odom, dtype=np.float64).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rotation_odom_body and position_odom must have shapes (3,3) "
            "and (3,)"
        ) from exc
    if not np.isfinite(rotation).all() or not np.isfinite(position).all():
        raise ValueError("odometry rotation and position must be finite")
    correction_rotation = rotation_from_rpy(0.0, 0.0, transform[2])
    corrected_rotation = correction_rotation @ rotation
    corrected_position = position.copy()
    corrected_position[:2] = (
        correction_rotation[:2, :2] @ position[:2] + transform[:2]
    )
    return corrected_rotation, corrected_position


@dataclass(frozen=True)
class MapOdomSoftCorrectionConfig:
    """Dynamics of the output-only map-frame correction.

    Time constants shape ordinary convergence.  Rate limits are independent
    safety bounds at the current vehicle pose and therefore retain their
    physical meaning even when global map coordinates are large.
    """

    translation_time_constant_s: float = 5.0
    yaw_time_constant_s: float = 5.0
    maximum_translation_rate_mps: float = 5.0
    maximum_yaw_rate_deg_s: float = 5.0

    def __post_init__(self) -> None:
        positive_fields = {
            "translation_time_constant_s": self.translation_time_constant_s,
            "yaw_time_constant_s": self.yaw_time_constant_s,
            "maximum_translation_rate_mps": (
                self.maximum_translation_rate_mps
            ),
            "maximum_yaw_rate_deg_s": self.maximum_yaw_rate_deg_s,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MapOdomCorrectionSnapshot:
    """Immutable state and convergence diagnostics for one output epoch."""

    timestamp: float | None
    current_map_to_odom_pose2: np.ndarray
    target_map_to_odom_pose2: np.ndarray
    current_map_pose2: np.ndarray
    target_map_pose2: np.ndarray
    remaining_translation_m: float
    remaining_yaw_deg: float
    update_calls: int
    positive_time_updates: int
    translation_rate_limit_activations: int
    yaw_rate_limit_activations: int

    def __post_init__(self) -> None:
        for field_name in (
            "current_map_to_odom_pose2",
            "target_map_to_odom_pose2",
            "current_map_pose2",
            "target_map_pose2",
        ):
            value = _validated_pose2(getattr(self, field_name), name=field_name)
            value.setflags(write=False)
            object.__setattr__(self, field_name, value)


class MapOdomSoftCorrector:
    """Rate-limited first-order follower for the planar graph correction."""

    def __init__(
        self, config: MapOdomSoftCorrectionConfig | None = None
    ) -> None:
        self.config = config or MapOdomSoftCorrectionConfig()
        self.reset()

    def reset(
        self,
        *,
        timestamp: float | None = None,
        map_to_odom_pose2: np.ndarray | None = None,
    ) -> None:
        if timestamp is not None and not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        initial = (
            np.zeros(3, dtype=np.float64)
            if map_to_odom_pose2 is None
            else _validated_pose2(
                map_to_odom_pose2, name="map_to_odom_pose2"
            )
        )
        self._initialized = timestamp is not None
        self._timestamp = None if timestamp is None else float(timestamp)
        self._current_map_to_odom_pose2 = initial.copy()
        self._target_map_to_odom_pose2 = initial.copy()
        self._current_map_pose2 = np.zeros(3, dtype=np.float64)
        self._target_map_pose2 = np.zeros(3, dtype=np.float64)
        self._remaining_translation_m = 0.0
        self._remaining_yaw_deg = 0.0
        self._update_calls = 0
        self._positive_time_updates = 0
        self._translation_rate_limit_activations = 0
        self._yaw_rate_limit_activations = 0

    @property
    def current_map_to_odom_pose2(self) -> np.ndarray:
        value = self._current_map_to_odom_pose2.copy()
        value.setflags(write=False)
        return value

    @property
    def target_map_to_odom_pose2(self) -> np.ndarray:
        value = self._target_map_to_odom_pose2.copy()
        value.setflags(write=False)
        return value

    def update(
        self,
        timestamp: float,
        odom_pose2: np.ndarray,
        target_map_pose2: np.ndarray,
    ) -> np.ndarray:
        """Advance the correction and return the filtered map-frame pose.

        A graph update received at the same timestamp changes the target but
        cannot instantaneously change the published pose.  This property also
        makes repeated UI/diagnostic reads idempotent.
        """

        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        timestamp = float(timestamp)
        odom_pose = _validated_pose2(odom_pose2, name="odom_pose2")
        target_map_pose = _validated_pose2(
            target_map_pose2, name="target_map_pose2"
        )
        target_transform = map_to_odom_from_poses(
            target_map_pose, odom_pose
        )
        self._update_calls += 1

        if not self._initialized:
            # Identity is the continuity-preserving default.  A caller that
            # starts in another established map frame can supply it to reset.
            self._initialized = True
            self._timestamp = timestamp
        elif timestamp < float(self._timestamp) - 1.0e-12:
            raise ValueError("timestamps must be nondecreasing")

        dt = max(0.0, timestamp - float(self._timestamp))
        predicted_map_pose = apply_map_to_odom_pose2(
            self._current_map_to_odom_pose2, odom_pose
        )
        translation_residual = (
            target_map_pose[:2] - predicted_map_pose[:2]
        )
        yaw_residual = float(
            wrap_angle(target_map_pose[2] - predicted_map_pose[2])
        )

        translation_step = np.zeros(2, dtype=np.float64)
        yaw_step = 0.0
        if dt > 0.0:
            self._positive_time_updates += 1
            translation_alpha = -math.expm1(
                -dt / self.config.translation_time_constant_s
            )
            yaw_alpha = -math.expm1(
                -dt / self.config.yaw_time_constant_s
            )
            translation_step = translation_alpha * translation_residual
            maximum_translation_step = (
                self.config.maximum_translation_rate_mps * dt
            )
            translation_step_norm = float(np.linalg.norm(translation_step))
            if translation_step_norm > maximum_translation_step:
                translation_step *= (
                    maximum_translation_step / translation_step_norm
                )
                self._translation_rate_limit_activations += 1

            yaw_step = yaw_alpha * yaw_residual
            maximum_yaw_step = math.radians(
                self.config.maximum_yaw_rate_deg_s
            ) * dt
            if abs(yaw_step) > maximum_yaw_step:
                yaw_step = math.copysign(maximum_yaw_step, yaw_step)
                self._yaw_rate_limit_activations += 1

        current_map_pose = predicted_map_pose.copy()
        current_map_pose[:2] += translation_step
        current_map_pose[2] = float(
            wrap_angle(current_map_pose[2] + yaw_step)
        )
        self._current_map_to_odom_pose2 = map_to_odom_from_poses(
            current_map_pose, odom_pose
        )
        self._target_map_to_odom_pose2 = target_transform
        self._current_map_pose2 = current_map_pose
        self._target_map_pose2 = target_map_pose
        self._remaining_translation_m = float(
            np.linalg.norm(target_map_pose[:2] - current_map_pose[:2])
        )
        self._remaining_yaw_deg = float(
            abs(
                math.degrees(
                    float(
                        wrap_angle(
                            target_map_pose[2] - current_map_pose[2]
                        )
                    )
                )
            )
        )
        self._timestamp = timestamp
        return current_map_pose.copy()

    def correct_pose(
        self,
        timestamp: float,
        rotation_odom_body: np.ndarray,
        position_odom: np.ndarray,
        target_rotation_map_body: np.ndarray,
        target_position_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update from full poses and return the continuously corrected pose."""

        try:
            odom_rotation = np.asarray(
                rotation_odom_body, dtype=np.float64
            ).reshape(3, 3)
            odom_position = np.asarray(
                position_odom, dtype=np.float64
            ).reshape(3)
            target_rotation = np.asarray(
                target_rotation_map_body, dtype=np.float64
            ).reshape(3, 3)
            target_position = np.asarray(
                target_position_map, dtype=np.float64
            ).reshape(3)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "full poses must have rotation (3,3), position (3,)"
            ) from exc
        if not (
            np.isfinite(odom_rotation).all()
            and np.isfinite(odom_position).all()
            and np.isfinite(target_rotation).all()
            and np.isfinite(target_position).all()
        ):
            raise ValueError("full poses must be finite")
        odom_pose2 = np.array(
            [
                odom_position[0],
                odom_position[1],
                rpy_from_rotation(odom_rotation)[2],
            ],
            dtype=np.float64,
        )
        target_map_pose2 = np.array(
            [
                target_position[0],
                target_position[1],
                rpy_from_rotation(target_rotation)[2],
            ],
            dtype=np.float64,
        )
        self.update(timestamp, odom_pose2, target_map_pose2)
        return apply_map_to_odom_pose(
            self._current_map_to_odom_pose2,
            odom_rotation,
            odom_position,
        )

    def snapshot(self) -> MapOdomCorrectionSnapshot:
        return MapOdomCorrectionSnapshot(
            timestamp=self._timestamp,
            current_map_to_odom_pose2=self._current_map_to_odom_pose2,
            target_map_to_odom_pose2=self._target_map_to_odom_pose2,
            current_map_pose2=self._current_map_pose2,
            target_map_pose2=self._target_map_pose2,
            remaining_translation_m=self._remaining_translation_m,
            remaining_yaw_deg=self._remaining_yaw_deg,
            update_calls=self._update_calls,
            positive_time_updates=self._positive_time_updates,
            translation_rate_limit_activations=(
                self._translation_rate_limit_activations
            ),
            yaw_rate_limit_activations=self._yaw_rate_limit_activations,
        )

    def diagnostics(self) -> dict[str, object]:
        snapshot = self.snapshot()
        return {
            "timestamp_s": snapshot.timestamp,
            "current_map_to_odom_pose2": (
                snapshot.current_map_to_odom_pose2.tolist()
            ),
            "target_map_to_odom_pose2": (
                snapshot.target_map_to_odom_pose2.tolist()
            ),
            "remaining_translation_m": snapshot.remaining_translation_m,
            "remaining_yaw_deg": snapshot.remaining_yaw_deg,
            "update_calls": snapshot.update_calls,
            "positive_time_updates": snapshot.positive_time_updates,
            "translation_rate_limit_activations": (
                snapshot.translation_rate_limit_activations
            ),
            "yaw_rate_limit_activations": (
                snapshot.yaw_rate_limit_activations
            ),
        }
