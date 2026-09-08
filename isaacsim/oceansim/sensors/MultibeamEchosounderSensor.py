"""Down-looking multibeam echosounder for bathymetric mapping and SLAM."""

from __future__ import annotations

import numpy as np
import omni.timeline

from isaacsim.core.api.sensors import BaseSensor
from isaacsim.core.utils.rotations import quat_to_rot_matrix

from isaacsim.oceansim.sensors.raycast_backend import PhysXRaycastBackend
from isaacsim.oceansim.utils.acoustic_sonar import (
    MultibeamPing,
    compute_backscatter_intensity,
    multibeam_directions,
    normalize_intensities,
    perturb_ranges,
    validate_range_limits,
)
from isaacsim.oceansim.utils.acoustic_visualization import (
    render_multibeam_swath,
)


class MultibeamEchosounderSensor(BaseSensor):
    """Simulate one bathymetric swath using closest-hit PhysX rays.

    The sensor frame follows ``+X`` forward, ``+Y`` port, ``+Z`` up.  Beams
    span the local Y-Z plane and point down.  Negative steering angles are
    port and positive angles are starboard, matching the raw-angle convention
    in ``marine_acoustic_msgs/SonarDetections``.

    This is a geometric first-return model intended for SLAM and mapping
    development.  It does not yet model sound-speed-profile refraction,
    beamforming, water-column samples, or multipath propagation.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "mbes",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        num_beams: int = 256,
        swath_width_deg: float = 120.0,
        min_range: float = 0.5,
        max_range: float = 75.0,
        sound_speed: float = 1500.0,
        center_frequency_hz: float = 0.0,
        range_noise_std: float = 0.02,
        dropout_probability: float = 0.0,
        attenuation: float = 0.01,
        default_reflectivity: float = 1.0,
        reflectivity_resolver=None,
        random_seed: int | None = None,
        raycast_backend=None,
        frame_id: str | None = None,
        viewport: bool = False,
        draw_in_scene: bool = False,
        viewport_resolution: tuple[int, int] = (900, 520),
    ) -> None:
        super().__init__(
            prim_path,
            name,
            position,
            translation,
            orientation,
            scale,
            visible,
        )

        self._min_range, self._max_range = validate_range_limits(
            min_range, max_range
        )
        self._sound_speed = float(sound_speed)
        self._center_frequency_hz = float(center_frequency_hz)
        self._range_noise_std = float(range_noise_std)
        self._dropout_probability = float(dropout_probability)
        self._attenuation = float(attenuation)
        if self._sound_speed <= 0.0:
            raise ValueError("sound_speed must be > 0")
        if self._center_frequency_hz < 0.0:
            raise ValueError("center_frequency_hz must be >= 0")
        if self._range_noise_std < 0.0:
            raise ValueError("range_noise_std must be >= 0")
        if not 0.0 <= self._dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be in [0, 1]")
        if self._attenuation < 0.0:
            raise ValueError("attenuation must be >= 0")

        self._beam_angles, self._beam_directions = multibeam_directions(
            num_beams=num_beams, swath_width_deg=swath_width_deg
        )
        self._frame_id = frame_id or prim_path.rsplit("/", 1)[-1]
        self._rng = np.random.default_rng(random_seed)
        self._backend = raycast_backend
        self._backend_kwargs = {
            "default_reflectivity": default_reflectivity,
            "reflectivity_resolver": reflectivity_resolver,
        }
        self._last_ping: MultibeamPing | None = None
        self._viewport_enabled = bool(viewport)
        self._draw_in_scene = bool(draw_in_scene)
        self._viewport_resolution = tuple(
            int(value) for value in viewport_resolution
        )
        if (
            len(self._viewport_resolution) != 2
            or min(self._viewport_resolution) < 64
        ):
            raise ValueError(
                "viewport_resolution must be (width, height) >= 64"
            )
        self._visual_viewport = None
        self._scene_visualizer = None

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        if self._backend is None:
            self._backend = PhysXRaycastBackend(**self._backend_kwargs)
        if self._viewport_enabled and self._visual_viewport is None:
            self.make_viewport()
        if self._draw_in_scene and self._scene_visualizer is None:
            from isaacsim.oceansim.sensors.acoustic_visualizer import (
                AcousticSceneVisualizer,
            )

            self._scene_visualizer = AcousticSceneVisualizer(
                self.prim_path,
                line_color=(0.05, 0.72, 0.92),
                point_color=(1.0, 0.55, 0.10),
                max_rays=96,
            )

    @staticmethod
    def _simulation_time() -> float:
        return float(omni.timeline.get_timeline_interface().get_current_time())

    def scan(self, timestamp: float | None = None) -> MultibeamPing:
        """Emit one ping from the current sensor pose."""

        if self._backend is None:
            self._backend = PhysXRaycastBackend(**self._backend_kwargs)

        origin_world, orientation_world = self.get_world_pose()
        origin_world = np.asarray(origin_world, dtype=np.float64)
        rotation_world_sensor = quat_to_rot_matrix(orientation_world)
        directions_world = self._beam_directions @ rotation_world_sensor.T

        accelerated_cast = getattr(self._backend, "cast_with_intensity", None)
        if callable(accelerated_cast):
            hits = accelerated_cast(
                origin_world=origin_world,
                directions_world=directions_world,
                max_range=self._max_range,
                attenuation=self._attenuation,
            )
        else:
            hits = self._backend.cast(
                origin_world=origin_world,
                directions_world=directions_world,
                max_range=self._max_range,
            )
        hit_mask = (
            hits.hit_mask
            & np.isfinite(hits.distances)
            & (hits.distances >= self._min_range)
            & (hits.distances <= self._max_range)
        )
        if self._dropout_probability > 0.0:
            hit_mask &= (
                self._rng.random(hit_mask.size) >= self._dropout_probability
            )

        ranges, _ = perturb_ranges(
            hits.distances,
            hit_mask,
            self._rng,
            gaussian_std=self._range_noise_std,
        )
        hit_mask &= (
            np.isfinite(ranges)
            & (ranges >= self._min_range)
            & (ranges <= self._max_range)
        )
        ranges[~hit_mask] = np.nan

        accelerated_intensities = getattr(hits, "raw_intensities", None)
        if accelerated_intensities is None:
            raw_intensities = compute_backscatter_intensity(
                distances=hits.distances,
                directions_world=directions_world,
                normals_world=hits.normals_world,
                reflectivities=hits.reflectivities,
                hit_mask=hit_mask,
                attenuation=self._attenuation,
            )
        else:
            raw_intensities = np.asarray(
                accelerated_intensities, dtype=np.float64
            ).copy()
            raw_intensities[~hit_mask] = 0.0
        intensities = normalize_intensities(raw_intensities, hit_mask)

        points_sensor = self._beam_directions * ranges[:, None]
        points_world = origin_world + directions_world * ranges[:, None]
        points_sensor[~hit_mask] = np.nan
        points_world[~hit_mask] = np.nan
        two_way_travel_times = 2.0 * ranges / self._sound_speed

        ping = MultibeamPing(
            timestamp=(
                self._simulation_time()
                if timestamp is None
                else float(timestamp)
            ),
            frame_id=self._frame_id,
            sound_speed=self._sound_speed,
            frequency=self._center_frequency_hz,
            beam_angles=self._beam_angles.copy(),
            beam_directions=self._beam_directions.copy(),
            ranges=ranges,
            two_way_travel_times=two_way_travel_times,
            intensities=intensities,
            hit_mask=hit_mask,
            points_sensor=points_sensor,
            points_world=points_world,
            prim_paths=hits.prim_paths,
        )
        self._last_ping = ping
        self._update_visualization(ping, origin_world)
        return ping

    def make_viewport(self) -> None:
        """Open the live port/starboard MBES cross-section window."""

        if self._visual_viewport is not None:
            return
        from isaacsim.oceansim.sensors.acoustic_visualizer import (
            AcousticImageViewport,
        )

        width, height = self._viewport_resolution
        self._visual_viewport = AcousticImageViewport(
            f"{self.name} | Multibeam Swath",
            width=width,
            height=height,
            footer="PORT (+Y)  <---  NADIR  --->  STARBOARD (-Y)",
        )

    def _update_visualization(
        self, ping: MultibeamPing, origin_world: np.ndarray
    ) -> None:
        if self._visual_viewport is not None:
            width, height = self._viewport_resolution
            rgba = render_multibeam_swath(
                ping,
                max_range=self._max_range,
                width=width,
                height=height,
            )
            valid_ranges = ping.ranges[ping.hit_mask]
            if valid_ranges.size:
                range_text = (
                    f"range {np.min(valid_ranges):.2f}-"
                    f"{np.max(valid_ranges):.2f} m"
                )
            else:
                range_text = "no bottom return"
            self._visual_viewport.update(
                rgba,
                f"t={ping.timestamp:.2f} s  |  "
                f"hits {np.count_nonzero(ping.hit_mask)}/"
                f"{ping.hit_mask.size}  |  "
                f"{range_text}",
            )
        if self._scene_visualizer is not None:
            self._scene_visualizer.update(
                origin_world, ping.valid_points(frame="world")
            )

    def close(self) -> None:
        """Close visualization resources created by this sensor."""

        if self._visual_viewport is not None:
            self._visual_viewport.close()
            self._visual_viewport = None
        if self._scene_visualizer is not None:
            self._scene_visualizer.close()
            self._scene_visualizer = None

    def get_last_ping(self) -> MultibeamPing | None:
        return self._last_ping

    def get_ranges(self) -> np.ndarray:
        if self._last_ping is None:
            return np.full(self._beam_angles.size, np.nan, dtype=np.float64)
        return self._last_ping.ranges.copy()

    def get_pointcloud(
        self, frame: str = "sensor", include_intensity: bool = False
    ) -> np.ndarray:
        """Return the most recent valid swath as XYZ or XYZI float32 points."""

        if self._last_ping is None:
            width = 4 if include_intensity else 3
            return np.empty((0, width), dtype=np.float32)
        points = self._last_ping.valid_points(frame=frame)
        if not include_intensity:
            return points
        intensity = self._last_ping.intensities[
            self._last_ping.hit_mask
        ].astype(np.float32)
        return np.ascontiguousarray(
            np.column_stack((points, intensity)), dtype=np.float32
        )

    def get_beam_angles(self) -> np.ndarray:
        return self._beam_angles.copy()

    def get_beam_directions(self) -> np.ndarray:
        return self._beam_directions.copy()
