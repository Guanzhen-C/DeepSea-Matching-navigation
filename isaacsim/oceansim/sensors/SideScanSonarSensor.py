"""Dual-channel side-scan sonar for synthetic waterfall imagery."""

from __future__ import annotations

import numpy as np
import omni.timeline

from isaacsim.core.api.sensors import BaseSensor
from isaacsim.core.utils.rotations import quat_to_rot_matrix

from isaacsim.oceansim.sensors.raycast_backend import PhysXRaycastBackend
from isaacsim.oceansim.utils.acoustic_sonar import (
    SideScanPing,
    add_sonar_noise,
    bin_sidescan_returns,
    compose_sidescan_line,
    compute_backscatter_intensity,
    normalize_intensities,
    perturb_ranges,
    sidescan_directions,
    validate_range_limits,
)
from isaacsim.oceansim.utils.acoustic_visualization import colorize_sidescan


class SideScanSonarSensor(BaseSensor):
    """Simulate port/starboard side-scan lines using closest-hit ray fans.

    The local frame is ``+X`` forward, ``+Y`` port, and ``+Z`` up.  Each ping
    raycasts a wide vertical fan on both sides, bins first-return intensity by
    slant range, then concatenates far-port -> nadir -> far-starboard.  Keeping
    only the closest return along a direction produces first-order acoustic
    shadows; repeated pings are accumulated in a waterfall buffer.

    The implementation is a fast geometric approximation.  It does not yet
    include refraction, coherent interference, sediment volume scattering, or
    multipath reverberation.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "sidescan",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        range_bins: int = 512,
        waterfall_lines: int = 400,
        rays_per_side: int = 512,
        tilt_deg: float = 45.0,
        vertical_beam_width_deg: float = 50.0,
        along_track_beam_width_deg: float = 2.0,
        along_track_samples: int = 1,
        min_range: float = 0.5,
        max_range: float = 75.0,
        sound_speed: float = 1500.0,
        center_frequency_hz: float = 0.0,
        range_noise_std: float = 0.02,
        range_diffusion_scale: float = 0.0,
        multiplicative_noise_std: float = 0.02,
        additive_noise_std: float = 0.01,
        attenuation: float = 0.01,
        gain: float = 1.0,
        default_reflectivity: float = 1.0,
        reflectivity_resolver=None,
        random_seed: int | None = None,
        raycast_backend=None,
        frame_id: str | None = None,
        viewport: bool = False,
        draw_in_scene: bool = False,
        viewport_resolution: tuple[int, int] = (1024, 520),
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
        self._range_bins = int(range_bins)
        self._waterfall_lines = int(waterfall_lines)
        if self._range_bins < 1 or self._waterfall_lines < 1:
            raise ValueError("range_bins and waterfall_lines must be >= 1")

        self._sound_speed = float(sound_speed)
        self._center_frequency_hz = float(center_frequency_hz)
        self._range_noise_std = float(range_noise_std)
        self._range_diffusion_scale = float(range_diffusion_scale)
        self._multiplicative_noise_std = float(multiplicative_noise_std)
        self._additive_noise_std = float(additive_noise_std)
        self._attenuation = float(attenuation)
        self._gain = float(gain)
        if self._sound_speed <= 0.0:
            raise ValueError("sound_speed must be > 0")
        if self._center_frequency_hz < 0.0:
            raise ValueError("center_frequency_hz must be >= 0")
        if any(
            value < 0.0
            for value in (
                self._range_noise_std,
                self._range_diffusion_scale,
                self._multiplicative_noise_std,
                self._additive_noise_std,
                self._attenuation,
                self._gain,
            )
        ):
            raise ValueError(
                "noise, attenuation, and gain values must be >= 0"
            )

        (
            self._port_directions,
            self._starboard_directions,
        ) = sidescan_directions(
            rays_per_side=rays_per_side,
            tilt_deg=tilt_deg,
            vertical_beam_width_deg=vertical_beam_width_deg,
            along_track_beam_width_deg=along_track_beam_width_deg,
            along_track_samples=along_track_samples,
        )
        self._frame_id = frame_id or prim_path.rsplit("/", 1)[-1]
        self._rng = np.random.default_rng(random_seed)
        self._backend = raycast_backend
        self._backend_kwargs = {
            "default_reflectivity": default_reflectivity,
            "reflectivity_resolver": reflectivity_resolver,
        }
        self._waterfall = np.zeros(
            (self._waterfall_lines, 2 * self._range_bins), dtype=np.uint8
        )
        self._waterfall_count = 0
        self._last_ping: SideScanPing | None = None
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
                line_color=(0.12, 0.56, 0.78),
                point_color=(1.0, 0.70, 0.18),
                max_rays=80,
            )

    @staticmethod
    def _simulation_time() -> float:
        return float(omni.timeline.get_timeline_interface().get_current_time())

    def _process_side(self, hits, directions_world):
        hit_mask = (
            hits.hit_mask
            & np.isfinite(hits.distances)
            & (hits.distances >= self._min_range)
            & (hits.distances <= self._max_range)
        )
        measured_ranges, positive_bias = perturb_ranges(
            hits.distances,
            hit_mask,
            self._rng,
            gaussian_std=self._range_noise_std,
            positive_bias_scale=self._range_diffusion_scale,
        )
        hit_mask &= (
            np.isfinite(measured_ranges)
            & (measured_ranges >= self._min_range)
            & (measured_ranges <= self._max_range)
        )
        measured_ranges[~hit_mask] = np.nan

        accelerated_intensities = getattr(hits, "raw_intensities", None)
        if accelerated_intensities is None:
            intensity = compute_backscatter_intensity(
                distances=hits.distances,
                directions_world=directions_world,
                normals_world=hits.normals_world,
                reflectivities=hits.reflectivities,
                hit_mask=hit_mask,
                attenuation=self._attenuation,
            )
        else:
            intensity = np.asarray(
                accelerated_intensities, dtype=np.float64
            ).copy()
            intensity[~hit_mask] = 0.0
        if self._range_diffusion_scale > 0.0:
            intensity *= np.exp(-positive_bias / self._range_diffusion_scale)

        centres, binned = bin_sidescan_returns(
            ranges=measured_ranges,
            intensities=intensity,
            hit_mask=hit_mask,
            min_range=self._min_range,
            max_range=self._max_range,
            num_bins=self._range_bins,
        )
        points_world = hits.positions_world[hit_mask]
        points_world = np.ascontiguousarray(
            points_world, dtype=np.float32
        )
        return centres, binned, points_world

    def scan(self, timestamp: float | None = None) -> SideScanPing:
        """Emit one dual-channel ping and append it to the waterfall."""

        if self._backend is None:
            self._backend = PhysXRaycastBackend(**self._backend_kwargs)

        origin_world, orientation_world = self.get_world_pose()
        origin_world = np.asarray(origin_world, dtype=np.float64)
        rotation_world_sensor = quat_to_rot_matrix(orientation_world)
        port_world = self._port_directions @ rotation_world_sensor.T
        starboard_world = self._starboard_directions @ rotation_world_sensor.T

        directions_world = np.concatenate((port_world, starboard_world), axis=0)
        accelerated_cast = getattr(self._backend, "cast_with_intensity", None)
        if callable(accelerated_cast):
            combined_hits = accelerated_cast(
                origin_world,
                directions_world,
                self._max_range,
                attenuation=self._attenuation,
            )
        else:
            combined_hits = self._backend.cast(
                origin_world, directions_world, self._max_range
            )
        split = port_world.shape[0]
        port_hits = combined_hits.sliced(0, split)
        starboard_hits = combined_hits.sliced(split, directions_world.shape[0])
        centres, port_raw, port_points = self._process_side(
            port_hits, port_world
        )
        _, starboard_raw, starboard_points = self._process_side(
            starboard_hits, starboard_world
        )

        combined = normalize_intensities(
            np.concatenate((port_raw, starboard_raw))
        )
        port = np.clip(combined[:self._range_bins] * self._gain, 0.0, 1.0)
        starboard = np.clip(
            combined[self._range_bins:] * self._gain, 0.0, 1.0
        )
        port = add_sonar_noise(
            port,
            centres,
            self._max_range,
            self._rng,
            multiplicative_std=self._multiplicative_noise_std,
            additive_std=self._additive_noise_std,
        )
        starboard = add_sonar_noise(
            starboard,
            centres,
            self._max_range,
            self._rng,
            multiplicative_std=self._multiplicative_noise_std,
            additive_std=self._additive_noise_std,
        )
        display_line = np.rint(
            compose_sidescan_line(port, starboard) * 255.0
        ).astype(np.uint8)

        self._waterfall[:-1] = self._waterfall[1:]
        self._waterfall[-1] = display_line
        self._waterfall_count = min(
            self._waterfall_count + 1, self._waterfall_lines
        )

        ping = SideScanPing(
            timestamp=(
                self._simulation_time()
                if timestamp is None
                else float(timestamp)
            ),
            frame_id=self._frame_id,
            sound_speed=self._sound_speed,
            frequency=self._center_frequency_hz,
            range_bin_centres=centres,
            port=port,
            starboard=starboard,
            display_line=display_line,
            port_points_world=port_points,
            starboard_points_world=starboard_points,
        )
        self._last_ping = ping
        self._update_visualization(ping, origin_world)
        return ping

    def make_viewport(self) -> None:
        """Open the live colorized side-scan waterfall window."""

        if self._visual_viewport is not None:
            return
        from isaacsim.oceansim.sensors.acoustic_visualizer import (
            AcousticImageViewport,
        )

        width, height = self._viewport_resolution
        self._visual_viewport = AcousticImageViewport(
            f"{self.name} | Side-scan Waterfall",
            width=width,
            height=height,
            footer=(
                "FAR PORT  <---  PORT  |  NADIR  |  "
                "STARBOARD  --->  FAR STARBOARD"
            ),
        )

    def _update_visualization(
        self, ping: SideScanPing, origin_world: np.ndarray
    ) -> None:
        if self._visual_viewport is not None:
            rgba = colorize_sidescan(self._waterfall)
            self._visual_viewport.update(
                rgba,
                f"t={ping.timestamp:.2f} s  |  ping "
                f"{self._waterfall_count}/{self._waterfall_lines}  |  "
                f"range {self._min_range:.1f}-{self._max_range:.1f} m",
            )
        if self._scene_visualizer is not None:
            hit_points = np.concatenate(
                (ping.port_points_world, ping.starboard_points_world), axis=0
            )
            self._scene_visualizer.update(origin_world, hit_points)

    def close(self) -> None:
        """Close visualization resources created by this sensor."""

        if self._visual_viewport is not None:
            self._visual_viewport.close()
            self._visual_viewport = None
        if self._scene_visualizer is not None:
            self._scene_visualizer.close()
            self._scene_visualizer = None

    def get_last_ping(self) -> SideScanPing | None:
        return self._last_ping

    def get_waterfall(self, valid_only: bool = False) -> np.ndarray:
        """Return a copy of the display waterfall as a mono8 image."""

        if not valid_only or self._waterfall_count == self._waterfall_lines:
            return self._waterfall.copy()
        if self._waterfall_count == 0:
            return np.empty((0, 2 * self._range_bins), dtype=np.uint8)
        return self._waterfall[-self._waterfall_count:].copy()

    def clear_waterfall(self) -> None:
        self._waterfall.fill(0)
        self._waterfall_count = 0

    def get_ray_directions(self) -> tuple[np.ndarray, np.ndarray]:
        return self._port_directions.copy(), self._starboard_directions.copy()
