"""Deterministic large-area underwater survey used for SLAM validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.acoustic_sonar import (
    BathymetricSideScanPing,
    MultibeamPing,
    RegisteredSideScanReturns,
    SideScanPing,
    bin_coregistered_sidescan_returns,
    compose_sidescan_line,
    imaging_lineage_to_bathymetry,
    integrated_bathymetric_sidescan_fan,
    multibeam_directions,
    parallel_bathymetric_sidescan_rays,
)
from .math3d import rotation_from_rpy, so3_log


@dataclass(frozen=True)
class GroundTruthState:
    timestamp: float
    rotation: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    angular_velocity_body: np.ndarray
    specific_force_body: np.ndarray


class LargeUnderwaterWorld:
    """Analytic bathymetry for repeatable validation and live demonstrations.

    ``validation`` preserves the original 300 x 220 m quality-gate terrain.
    ``continental_margin`` adds scale-aware shelf, slope, canyon, ridge,
    pockmark, and sand-wave morphology for a more natural large live scene.
    Both profiles are fully deterministic so repeated sensor runs see exactly
    the same seabed.
    """

    def __init__(
        self,
        *,
        width: float = 300.0,
        length: float = 220.0,
        terrain_profile: str = "validation",
    ) -> None:
        self.width = float(width)
        self.length = float(length)
        self.terrain_profile = str(terrain_profile)
        if self.width <= 0.0 or self.length <= 0.0:
            raise ValueError("world dimensions must be positive")
        if self.terrain_profile not in {"validation", "continental_margin"}:
            raise ValueError(
                "terrain_profile must be 'validation' or 'continental_margin'"
            )

    @staticmethod
    def _smooth_box(value: np.ndarray, half_extent: float, softness: float) -> np.ndarray:
        return 0.5 * (
            np.tanh((value + half_extent) / softness)
            - np.tanh((value - half_extent) / softness)
        )

    def seabed_height(self, x, y) -> np.ndarray:
        """Return Z-up seabed elevation for scalar or array coordinates."""

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.terrain_profile == "continental_margin":
            return self._continental_margin_height(x, y)

        height = (
            -34.0
            + 2.2 * np.sin(x / 29.0) * np.cos(y / 24.0)
            + 1.3 * np.sin((x + 0.7 * y) / 17.0)
            + 0.8 * np.cos((x - 1.4 * y) / 12.0)
        )

        # Long ridge, eroded trench, wreck, and pipe give the mapper both
        # gentle bathymetry and sharp repeatable structure over a large area.
        ridge_axis = y - 0.24 * x - 22.0
        height += 3.7 * np.exp(-(ridge_axis / 7.0) ** 2) * np.exp(
            -(x / 105.0) ** 6
        )
        height -= 3.2 * np.exp(-((x + 48.0) / 13.0) ** 2) * np.exp(
            -((y - 28.0) / 62.0) ** 4
        )

        wreck_x = (x - 48.0) * np.cos(0.32) + (y + 27.0) * np.sin(0.32)
        wreck_y = -(x - 48.0) * np.sin(0.32) + (y + 27.0) * np.cos(0.32)
        wreck = self._smooth_box(wreck_x, 10.0, 0.8) * self._smooth_box(
            wreck_y, 3.2, 0.45
        )
        height += 4.6 * wreck
        height += 2.0 * np.exp(
            -((wreck_x - 11.0) / 3.0) ** 2 - (wreck_y / 2.3) ** 2
        )

        pipe_axis = y + 0.31 * x - 42.0
        pipe_window = self._smooth_box(x + 12.0, 92.0, 4.0)
        height += 1.15 * np.exp(-(pipe_axis / 1.25) ** 2) * pipe_window

        for centre_x, centre_y, amplitude, radius in (
            (-104.0, -61.0, 2.8, 7.0),
            (-72.0, 64.0, 3.4, 8.5),
            (2.0, 72.0, 2.5, 6.0),
            (91.0, 51.0, 3.7, 9.0),
            (112.0, -48.0, 2.6, 6.5),
        ):
            height += amplitude * np.exp(
                -((x - centre_x) ** 2 + (y - centre_y) ** 2)
                / (radius * radius)
            )
        return height

    def _continental_margin_height(self, x, y) -> np.ndarray:
        """Return a deterministic, multi-scale continental-margin seabed."""

        # Work in normalized coordinates for the regional landforms while
        # retaining metre-scale wavelengths for sand waves and roughness.
        xn = 2.0 * x / self.width
        yn = 2.0 * y / self.length
        warped_x = xn + 0.075 * np.sin(2.4 * np.pi * yn)
        warped_y = yn + 0.055 * np.sin(2.0 * np.pi * xn + 0.7)

        # A shallow western shelf rolls over into a deeper eastern basin.
        shelf_break = 0.5 * (1.0 + np.tanh((xn + 0.12) / 0.24))
        height = -39.0 - 25.0 * shelf_break
        height += 4.6 * np.sin(1.25 * np.pi * warped_x + 0.35) * np.cos(
            1.05 * np.pi * warped_y - 0.2
        )
        height += 2.3 * np.sin(
            3.1 * np.pi * warped_x + 1.7 * np.pi * warped_y
        )
        height += 1.15 * np.cos(
            6.4 * np.pi * warped_x - 3.8 * np.pi * warped_y + 0.8
        )
        height += 0.48 * np.sin(
            12.0 * np.pi * warped_x + 8.5 * np.pi * warped_y
        )

        # A meandering submarine canyon crosses the slope, with one tributary
        # and asymmetric shoulders produced by nearby depositional levees.
        canyon_centre = (
            -0.12
            + 0.18 * np.sin(1.15 * np.pi * (xn + 0.22))
            + 0.045 * np.sin(4.2 * np.pi * xn)
        )
        canyon_offset = yn - canyon_centre
        canyon_width = 0.050 + 0.018 * (0.5 + 0.5 * np.cos(np.pi * xn))
        canyon_window = 0.5 * (
            np.tanh((xn + 0.92) / 0.07) - np.tanh((xn - 0.95) / 0.07)
        )
        height -= 14.5 * np.exp(-(canyon_offset / canyon_width) ** 2) * canyon_window
        height += 3.4 * np.exp(
            -((canyon_offset - 1.75 * canyon_width) / (0.85 * canyon_width)) ** 2
        ) * canyon_window

        tributary = yn - (0.34 + 0.62 * (xn + 0.68))
        tributary_window = 0.5 * (
            np.tanh((xn + 0.72) / 0.05) - np.tanh((xn + 0.18) / 0.05)
        )
        height -= 5.0 * np.exp(-(tributary / 0.038) ** 2) * tributary_window

        # Folded rock ridge, isolated knoll, slump scar, and pockmarks add the
        # irregular, locally steep structure useful to sonar registration.
        ridge_axis = yn + 0.41 * xn - 0.43 + 0.035 * np.sin(5.0 * np.pi * xn)
        height += 8.5 * np.exp(-(ridge_axis / 0.045) ** 2) * np.exp(
            -((xn + 0.05) / 0.78) ** 6
        )
        height += 15.0 * np.exp(
            -((xn - 0.52) / 0.13) ** 2 - ((yn - 0.53) / 0.18) ** 2
        )
        height -= 7.0 * np.exp(
            -((xn + 0.46) / 0.22) ** 2 - ((yn + 0.61) / 0.13) ** 2
        )
        for centre_x, centre_y, radius, depth in (
            (-0.64, 0.48, 0.035, 2.8),
            (-0.23, -0.54, 0.046, 3.5),
            (0.18, 0.66, 0.032, 2.5),
            (0.61, -0.47, 0.052, 4.2),
            (0.76, 0.12, 0.038, 3.1),
        ):
            height -= depth * np.exp(
                -((xn - centre_x) ** 2 + (yn - centre_y) ** 2)
                / (radius * radius)
            )

        # Sand waves occur only on the shallow shelf. Their curved phase and
        # tapered patch prevent the visibly repetitive whole-map sine pattern.
        sand_patch = np.exp(
            -((xn + 0.63) / 0.34) ** 4 - ((yn + 0.05) / 0.72) ** 6
        )
        sand_phase = 2.0 * np.pi * (
            x / 21.0 + y / 74.0 + 0.28 * np.sin(y / 53.0)
        )
        height += 0.72 * sand_patch * np.sin(sand_phase)
        return height

    def reflectivity(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.terrain_profile == "continental_margin":
            xn = 2.0 * x / self.width
            yn = 2.0 * y / self.length
            texture = (
                0.50
                + 0.10 * np.sin(x / 13.0 + y / 37.0)
                + 0.07 * np.cos(x / 5.8 - y / 9.1)
                + 0.04 * np.sin(x / 2.7 + 0.6 * np.sin(y / 18.0))
            )
            ridge_axis = yn + 0.41 * xn - 0.43
            texture += 0.34 * np.exp(-(ridge_axis / 0.055) ** 2)
            sand = np.exp(
                -((xn + 0.63) / 0.34) ** 4 - ((yn + 0.05) / 0.72) ** 6
            )
            texture += 0.14 * sand
            canyon_centre = -0.12 + 0.18 * np.sin(
                1.15 * np.pi * (xn + 0.22)
            )
            texture -= 0.18 * np.exp(-((yn - canyon_centre) / 0.075) ** 2)
            return np.clip(texture, 0.10, 1.35)

        texture = (
            0.62
            + 0.16 * np.sin(x / 8.0) * np.sin(y / 11.0)
            + 0.12 * np.cos((x - y) / 5.7)
        )
        pipe_axis = y + 0.31 * x - 42.0
        texture += 0.65 * np.exp(-(pipe_axis / 1.6) ** 2)
        return np.clip(texture, 0.12, 1.5)

    def seabed_color(self, x, y) -> np.ndarray:
        """Return RGB sediment colours for a terrain vertex array."""

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        height = self.seabed_height(x, y)
        normal = self.normal(x, y)
        slope = np.clip((1.0 - normal[..., 2]) / 0.12, 0.0, 1.0)
        reflectivity = self.reflectivity(x, y)

        if self.terrain_profile == "continental_margin":
            xn = 2.0 * x / self.width
            yn = 2.0 * y / self.length
            sand = np.exp(
                -((xn + 0.63) / 0.34) ** 4 - ((yn + 0.05) / 0.72) ** 6
            )
            deep = np.clip((-height - 38.0) / 42.0, 0.0, 1.0)
            base = np.stack(
                (
                    0.16 - 0.045 * deep,
                    0.30 - 0.080 * deep,
                    0.26 - 0.025 * deep,
                ),
                axis=-1,
            )
            sand_colour = np.array([0.47, 0.43, 0.29], dtype=np.float64)
            rock_colour = np.array([0.25, 0.30, 0.27], dtype=np.float64)
            colour = base * (1.0 - 0.58 * sand[..., None])
            colour += sand_colour * (0.58 * sand[..., None])
            colour = colour * (1.0 - 0.46 * slope[..., None])
            colour += rock_colour * (0.46 * slope[..., None])
        else:
            colour = np.broadcast_to(
                np.array([0.12, 0.31, 0.28], dtype=np.float64),
                height.shape + (3,),
            ).copy()

        mottling = np.clip(0.82 + 0.27 * reflectivity, 0.78, 1.18)
        return np.clip(colour * mottling[..., None], 0.02, 0.72)

    def normal(self, x, y) -> np.ndarray:
        epsilon = 0.08
        dz_dx = (
            self.seabed_height(np.asarray(x) + epsilon, y)
            - self.seabed_height(np.asarray(x) - epsilon, y)
        ) / (2.0 * epsilon)
        dz_dy = (
            self.seabed_height(x, np.asarray(y) + epsilon)
            - self.seabed_height(x, np.asarray(y) - epsilon)
        ) / (2.0 * epsilon)
        normals = np.stack((-dz_dx, -dz_dy, np.ones_like(dz_dx)), axis=-1)
        return normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    def intersect_rays(
        self,
        origin_world: np.ndarray,
        directions_world: np.ndarray,
        *,
        min_range: float,
        max_range: float,
        iterations: int = 24,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the analytic bathymetry using vector bisection."""

        origin = np.asarray(origin_world, dtype=np.float64).reshape(3)
        directions = np.asarray(directions_world, dtype=np.float64).reshape(-1, 3)
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        low = np.full(directions.shape[0], float(min_range), dtype=np.float64)
        high = np.full(directions.shape[0], float(max_range), dtype=np.float64)

        def signed_height(distance):
            points = origin + directions * distance[:, None]
            return points[:, 2] - self.seabed_height(points[:, 0], points[:, 1])

        f_low = signed_height(low)
        f_high = signed_height(high)
        valid = (
            np.isfinite(f_low)
            & np.isfinite(f_high)
            & (f_low >= 0.0)
            & (f_high <= 0.0)
            & (directions[:, 2] < -1.0e-4)
        )
        for _ in range(int(iterations)):
            middle = 0.5 * (low + high)
            f_middle = signed_height(middle)
            above = f_middle > 0.0
            low = np.where(valid & above, middle, low)
            high = np.where(valid & ~above, middle, high)

        distances = 0.5 * (low + high)
        distances[~valid] = np.nan
        points = origin + directions * distances[:, None]
        points[~valid] = np.nan
        normals = np.zeros_like(points)
        normals[valid] = self.normal(points[valid, 0], points[valid, 1])
        incidence = np.zeros(directions.shape[0], dtype=np.float64)
        incidence[valid] = np.clip(
            np.sum(normals[valid] * -directions[valid], axis=1), 0.0, 1.0
        )
        return valid, distances, points, incidence


class FigureEightSurvey:
    """Smooth two-lap survey spanning roughly 240 x 160 metres."""

    def __init__(
        self,
        *,
        period: float = 180.0,
        x_amplitude: float = 122.0,
        y_amplitude: float = 78.0,
        nominal_depth: float = 8.5,
    ) -> None:
        self.period = float(period)
        self.x_amplitude = float(x_amplitude)
        self.y_amplitude = float(y_amplitude)
        self.nominal_depth = float(nominal_depth)
        self.omega = 2.0 * np.pi / self.period

    def kinematics(
        self, timestamp: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = float(timestamp)
        phase = self.omega * t
        x = self.x_amplitude * np.sin(phase)
        y = self.y_amplitude * np.sin(2.0 * phase)
        z = -self.nominal_depth + 0.55 * np.sin(3.0 * phase)
        vx = self.x_amplitude * self.omega * np.cos(phase)
        vy = 2.0 * self.y_amplitude * self.omega * np.cos(2.0 * phase)
        vz = 1.65 * self.omega * np.cos(3.0 * phase)
        ax = -self.x_amplitude * self.omega**2 * np.sin(phase)
        ay = -4.0 * self.y_amplitude * self.omega**2 * np.sin(2.0 * phase)
        az = -4.95 * self.omega**2 * np.sin(3.0 * phase)
        return (
            np.array([x, y, z], dtype=np.float64),
            np.array([vx, vy, vz], dtype=np.float64),
            np.array([ax, ay, az], dtype=np.float64),
        )

    def rotation(self, timestamp: float) -> np.ndarray:
        _, velocity, _ = self.kinematics(timestamp)
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        yaw = float(np.arctan2(velocity[1], velocity[0]))
        pitch = float(-np.arctan2(velocity[2], max(horizontal_speed, 1.0e-6)))
        # Use an integer harmonic so the complete 6-DoF vehicle state repeats
        # after one survey period.  This matches the roll-stabilised AUV
        # assumption used by the x/y/yaw pose-graph back end and makes a
        # two-lap loop-closure test physically self-consistent.
        roll = float(0.035 * np.sin(2.0 * self.omega * timestamp))
        return rotation_from_rpy(roll, pitch, yaw)

    def state(self, timestamp: float) -> GroundTruthState:
        position, velocity, acceleration = self.kinematics(timestamp)
        rotation = self.rotation(timestamp)
        epsilon = 1.0e-3
        rotation_before = self.rotation(timestamp - epsilon)
        rotation_after = self.rotation(timestamp + epsilon)
        angular_velocity = so3_log(rotation_before.T @ rotation_after) / (
            2.0 * epsilon
        )
        gravity = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
        specific_force = rotation.T @ (acceleration - gravity)
        return GroundTruthState(
            timestamp=float(timestamp),
            rotation=rotation,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            angular_velocity_body=angular_velocity,
            specific_force_body=specific_force,
        )


def unresolved_reflectivity_multiplier(
    x, y, amplitude: float
) -> np.ndarray:
    """Return a deterministic, zero-mean subgrid scattering multiplier."""

    amplitude = float(amplitude)
    if not 0.0 <= amplitude <= 0.75:
        raise ValueError("subgrid reflectivity amplitude must lie in [0, 0.75]")
    x_values, y_values = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )
    if amplitude <= 0.0:
        return np.ones_like(x_values, dtype=np.float64)
    wavelengths = np.array(
        [0.36, 0.53, 0.79, 1.17, 1.73, 2.57], dtype=np.float64
    )
    directions = np.array(
        [0.19, 1.07, 2.11, 2.83, 0.71, 1.69], dtype=np.float64
    )
    phases = np.array(
        [0.37, 2.19, 4.71, 1.43, 5.37, 3.11], dtype=np.float64
    )
    weights = np.array(
        [0.34, 0.27, 0.20, 0.13, 0.08, 0.05], dtype=np.float64
    )
    texture = np.zeros_like(x_values, dtype=np.float64)
    for wavelength, direction, phase, weight in zip(
        wavelengths, directions, phases, weights, strict=True
    ):
        coordinate = (
            np.cos(direction) * x_values + np.sin(direction) * y_values
        )
        texture += weight * np.sin(
            (2.0 * np.pi / wavelength) * coordinate + phase
        )
    texture /= float(np.sqrt(0.5 * np.sum(weights * weights)))
    return np.clip(1.0 + amplitude * texture, 0.25, 1.75)


class LargeScaleSensorSimulator:
    """Generate navigation and shared bathymetry/side-scan measurements."""

    def __init__(
        self,
        world: LargeUnderwaterWorld,
        trajectory: FigureEightSurvey,
        *,
        num_beams: int = 96,
        swath_width_deg: float = 120.0,
        min_range: float = 1.0,
        max_range: float = 62.0,
        sidescan_range_bins: int = 160,
        imaging_rays_per_side: int = 512,
        along_track_rows_per_side: int = 1,
        along_track_sample_spacing_m: float = 0.0,
        subgrid_reflectivity_amplitude: float = 0.0,
        seed: int = 31,
    ) -> None:
        self.world = world
        self.trajectory = trajectory
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.rng = np.random.default_rng(seed)
        self.sidescan_rng = np.random.default_rng(seed + 10_007)
        self.imaging_intensity_rng = np.random.default_rng(seed + 30_013)
        self.sidescan_range_bins = int(sidescan_range_bins)
        self.imaging_rays_per_side = int(imaging_rays_per_side)
        self.along_track_rows_per_side = int(along_track_rows_per_side)
        self.along_track_sample_spacing_m = float(
            along_track_sample_spacing_m
        )
        self.subgrid_reflectivity_amplitude = float(
            subgrid_reflectivity_amplitude
        )
        if self.sidescan_range_bins < 1:
            raise ValueError("sidescan_range_bins must be >= 1")
        if self.imaging_rays_per_side < 1:
            raise ValueError("imaging_rays_per_side must be >= 1")
        if self.along_track_rows_per_side < 1:
            raise ValueError("along_track_rows_per_side must be >= 1")
        if not 0.0 <= self.subgrid_reflectivity_amplitude <= 0.75:
            raise ValueError(
                "subgrid_reflectivity_amplitude must lie in [0, 0.75]"
            )
        if self.along_track_rows_per_side == 1:
            if self.along_track_sample_spacing_m < 0.0:
                raise ValueError(
                    "along_track_sample_spacing_m must be non-negative"
                )
            (
                self.shared_ray_directions_sensor,
                self.bathymetry_source_ray_indices,
                self.imaging_directions_sensor,
                self.imaging_source_ray_indices,
                self.imaging_angles,
            ) = integrated_bathymetric_sidescan_fan(
                num_beams=int(num_beams),
                swath_width_deg=swath_width_deg,
                imaging_rays_per_side=self.imaging_rays_per_side,
            )
            self.shared_ray_origins_sensor = np.zeros(
                (self.shared_ray_directions_sensor.shape[0], 3),
                dtype=np.float64,
            )
        else:
            if self.along_track_sample_spacing_m <= 0.0:
                raise ValueError(
                    "multiple along-track rows require positive sample spacing"
                )
            (
                self.shared_ray_origins_sensor,
                self.shared_ray_directions_sensor,
                self.bathymetry_source_ray_indices,
                _imaging_origins,
                self.imaging_directions_sensor,
                self.imaging_source_ray_indices,
                self.imaging_angles,
            ) = parallel_bathymetric_sidescan_rays(
                num_beams=int(num_beams),
                swath_width_deg=swath_width_deg,
                imaging_rays_per_side=self.imaging_rays_per_side,
                along_track_rows_per_side=self.along_track_rows_per_side,
                along_track_sample_spacing_m=(
                    self.along_track_sample_spacing_m
                ),
            )
        self.imaging_row_indices = np.tile(
            np.repeat(
                np.arange(self.along_track_rows_per_side, dtype=np.int64),
                self.imaging_rays_per_side,
            ),
            2,
        )
        self.beam_angles = self.imaging_angles[
            self.bathymetry_source_ray_indices
        ].copy()
        self.beam_directions_sensor = self.shared_ray_directions_sensor[
            self.bathymetry_source_ray_indices
        ].copy()
        self.last_integrated_ping: BathymetricSideScanPing | None = None
        # The published LAUV INS specification gives a maximum gyro bias of
        # one degree per hour.  Keep the simulated three-axis bias within that
        # envelope instead of the unrealistically large 165 deg/hour formerly
        # used by this demonstration.
        gyro_bias_limit = np.deg2rad(1.0) / 3600.0
        self.gyro_bias = gyro_bias_limit * np.array([0.29, -0.23, 0.87])
        self.accel_bias = np.array([0.018, -0.012, 0.025])
        self.dvl_bias = np.array([0.012, -0.008, 0.004])

    def _acoustic_reflectivity(self, x, y) -> np.ndarray:
        """Measured reflectivity plus optional unresolved spatial texture.

        Public regional backscatter mosaics are metre-scale products, whereas
        the V2 head samples 0.15 m along track.  When explicitly requested,
        this deterministic band-limited field represents unresolved sediment
        scattering below the source-raster cell size.  It is world-fixed (so
        revisits observe the same seabed), zero-mean, and multiplicative; it
        does not alter measured bathymetry or claim to be measured intensity.
        """

        measured = np.asarray(self.world.reflectivity(x, y), dtype=np.float64)
        amplitude = self.subgrid_reflectivity_amplitude
        if amplitude <= 0.0:
            return measured
        # Incommensurate wavelengths/directions form a reproducible,
        # non-axis-aligned narrow-band random-field approximation.  The
        # shortest wavelength remains above twice the 0.15 m standard-mode
        # sampling interval, avoiding synthetic spatial aliasing.
        multiplier = unresolved_reflectivity_multiplier(x, y, amplitude)
        return np.clip(measured * multiplier, 0.0, None)

    def _intersect_sensor_rays(
        self,
        truth: GroundTruthState,
        origins_sensor: np.ndarray,
        directions_sensor: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Intersect possibly displaced simultaneous ray origins."""

        origins_sensor = np.asarray(origins_sensor, dtype=np.float64).reshape(
            -1, 3
        )
        directions_sensor = np.asarray(
            directions_sensor, dtype=np.float64
        ).reshape(-1, 3)
        if origins_sensor.shape != directions_sensor.shape:
            raise ValueError("ray origins and directions must agree")
        origins_world = truth.position + origins_sensor @ truth.rotation.T
        directions_world = directions_sensor @ truth.rotation.T
        hit = np.zeros(directions_sensor.shape[0], dtype=bool)
        distances = np.full(directions_sensor.shape[0], np.nan, dtype=np.float64)
        points_world = np.full(
            directions_sensor.shape, np.nan, dtype=np.float64
        )
        incidence = np.zeros(directions_sensor.shape[0], dtype=np.float64)
        # Both sides of one documented row share a phase centre.  Grouping by
        # that centre keeps the analytic height-field API dependency-light.
        unique_origins, inverse = np.unique(
            origins_sensor, axis=0, return_inverse=True
        )
        for origin_index in range(unique_origins.shape[0]):
            selected = np.flatnonzero(inverse == origin_index)
            (
                selected_hit,
                selected_distances,
                selected_points,
                selected_incidence,
            ) = self.world.intersect_rays(
                origins_world[selected[0]],
                directions_world[selected],
                min_range=self.min_range,
                max_range=self.max_range,
            )
            hit[selected] = selected_hit
            distances[selected] = selected_distances
            points_world[selected] = selected_points
            incidence[selected] = selected_incidence
        return (
            hit,
            distances,
            points_world,
            incidence,
            origins_world,
            directions_world,
        )

    def imu(self, truth: GroundTruthState) -> tuple[np.ndarray, np.ndarray]:
        gyro = (
            truth.angular_velocity_body
            + self.gyro_bias
            + self.rng.normal(0.0, 0.0022, 3)
        )
        accel = (
            truth.specific_force_body
            + self.accel_bias
            + self.rng.normal(0.0, 0.025, 3)
        )
        return gyro, accel

    def dvl(self, truth: GroundTruthState) -> np.ndarray:
        return (
            truth.rotation.T @ truth.velocity
            + self.dvl_bias
            + self.rng.normal(0.0, 0.032, 3)
        )

    def depth(self, truth: GroundTruthState) -> float:
        return float(-truth.position[2] + self.rng.normal(0.0, 0.035))

    def mbes(
        self, truth: GroundTruthState
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        origins_sensor = self.shared_ray_origins_sensor[
            self.bathymetry_source_ray_indices
        ]
        (
            hit,
            distances,
            points_world,
            incidence,
            _origins_world,
            _directions_world,
        ) = self._intersect_sensor_rays(
            truth,
            origins_sensor,
            self.beam_directions_sensor,
        )
        measured_ranges = distances.copy()
        measured_ranges[hit] += self.rng.normal(0.0, 0.028, np.count_nonzero(hit))
        measured_ranges[hit] = np.clip(
            measured_ranges[hit], self.min_range, self.max_range
        )
        points_sensor = (
            origins_sensor
            + self.beam_directions_sensor * measured_ranges[:, None]
        )
        points_sensor[~hit] = np.nan
        intensity = np.zeros(hit.size, dtype=np.float64)
        if np.any(hit):
            reflectivity = self._acoustic_reflectivity(
                points_world[hit, 0], points_world[hit, 1]
            )
            intensity[hit] = (
                reflectivity
                * incidence[hit] ** 1.4
                * np.exp(-0.012 * distances[hit])
            )
            intensity[hit] *= np.clip(
                1.0 + self.rng.normal(0.0, 0.035, np.count_nonzero(hit)),
                0.0,
                None,
            )
        return points_sensor, intensity, hit

    def sidescan(
        self, truth: GroundTruthState
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility wrapper returning one integrated image product.

        New code should call :meth:`bathymetric_sidescan_ping` so both the
        sparse geometry and dense point-wise backscatter remain available.
        """

        _, _, _, port, starboard = self.bathymetric_sidescan(truth)
        return port, starboard

    def _registered_sidescan_lines(
        self,
        points_sensor: np.ndarray,
        intensities: np.ndarray,
        hit: np.ndarray,
        directions_sensor: np.ndarray | None = None,
        origins_sensor: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if directions_sensor is None:
            directions_sensor = self.beam_directions_sensor
        origins = (
            np.zeros_like(points_sensor)
            if origins_sensor is None
            else np.asarray(origins_sensor, dtype=np.float64).reshape(-1, 3)
        )
        measured_ranges = np.linalg.norm(
            np.where(hit[:, None], points_sensor - origins, 0.0), axis=1
        )
        (
            _,
            port,
            starboard,
            port_source,
            starboard_source,
        ) = bin_coregistered_sidescan_returns(
            measured_ranges,
            intensities,
            hit,
            directions_sensor,
            self.min_range,
            self.max_range,
            self.sidescan_range_bins,
        )
        lines = []
        for line in (port, starboard):
            # Pulse length and finite beam footprint spread one registered
            # bottom detection over adjacent range samples for the image
            # product only. Direct SLAM still consumes the original XYZI pair.
            line = np.convolve(
                line, np.array([0.16, 0.68, 0.16]), mode="same"
            )
            line *= np.clip(
                1.0
                + self.sidescan_rng.normal(
                    0.0, 0.035, self.sidescan_range_bins
                ),
                0.0,
                None,
            )
            line += self.sidescan_rng.normal(
                0.0, 0.003, self.sidescan_range_bins
            )
            lines.append(np.maximum(line, 0.0))
        scale = max(
            float(np.percentile(np.concatenate(lines), 99.0)), 1.0e-8
        )
        normalized = tuple(np.clip(line / scale, 0.0, 1.0) for line in lines)
        return (
            normalized[0],
            normalized[1],
            port_source,
            starboard_source,
        )

    def bathymetric_sidescan_ping(
        self, truth: GroundTruthState
    ) -> BathymetricSideScanPing:
        """Return bathymetry and imagery derived from one raw echo grid.

        ``LargeUnderwaterWorld.intersect_rays`` is invoked once.  Range noise
        and radiometric noise are also sampled once per echo; bathymetry is an
        exact output view of those same point-wise XYZ + intensity returns.
        """

        (
            hit,
            distances,
            true_points_world,
            incidence,
            shared_origins_world,
            shared_directions_world,
        ) = self._intersect_sensor_rays(
            truth,
            self.shared_ray_origins_sensor,
            self.shared_ray_directions_sensor,
        )
        shared_hit = np.asarray(hit, dtype=bool).copy()
        shared_ranges = distances.copy()
        shared_ranges[shared_hit] += self.rng.normal(
            0.0, 0.028, np.count_nonzero(shared_hit)
        )
        shared_ranges[shared_hit] = np.clip(
            shared_ranges[shared_hit],
            self.min_range,
            self.max_range,
        )
        shared_ranges[~shared_hit] = np.nan

        raw_intensity = np.zeros(shared_hit.size, dtype=np.float64)
        if np.any(shared_hit):
            reflectivity = self._acoustic_reflectivity(
                true_points_world[shared_hit, 0],
                true_points_world[shared_hit, 1],
            )
            raw_intensity[shared_hit] = (
                reflectivity
                * incidence[shared_hit] ** 1.4
                * np.exp(-0.012 * distances[shared_hit])
            )

        shared_intensity = raw_intensity.copy()
        shared_intensity[shared_hit] *= np.clip(
            1.0
            + self.imaging_intensity_rng.normal(
                0.0, 0.035, np.count_nonzero(shared_hit)
            ),
            0.0,
            None,
        )
        shared_points_sensor = (
            self.shared_ray_origins_sensor
            + self.shared_ray_directions_sensor * shared_ranges[:, None]
        )
        shared_points_sensor[~shared_hit] = np.nan
        shared_points_world = (
            shared_origins_world
            + shared_directions_world * shared_ranges[:, None]
        )
        shared_points_world[~shared_hit] = np.nan

        bathymetry_hit = shared_hit[
            self.bathymetry_source_ray_indices
        ].copy()
        bathymetry_ranges = shared_ranges[
            self.bathymetry_source_ray_indices
        ].copy()
        bathymetry_intensity = shared_intensity[
            self.bathymetry_source_ray_indices
        ].copy()
        bathymetry_points_sensor = shared_points_sensor[
            self.bathymetry_source_ray_indices
        ].copy()
        bathymetry_points_world = shared_points_world[
            self.bathymetry_source_ray_indices
        ].copy()
        bathymetry = MultibeamPing(
            timestamp=float(truth.timestamp),
            frame_id="integrated_sonar",
            sound_speed=1500.0,
            frequency=400_000.0,
            beam_angles=self.beam_angles.copy(),
            beam_directions=self.beam_directions_sensor.copy(),
            ranges=bathymetry_ranges,
            two_way_travel_times=2.0 * bathymetry_ranges / 1500.0,
            intensities=bathymetry_intensity,
            hit_mask=bathymetry_hit,
            points_sensor=bathymetry_points_sensor,
            points_world=bathymetry_points_world,
        )

        imaging_hit = shared_hit[self.imaging_source_ray_indices]
        imaging_ranges = shared_ranges[
            self.imaging_source_ray_indices
        ].copy()
        imaging_intensity = shared_intensity[
            self.imaging_source_ray_indices
        ].copy()
        imaging_points_sensor = shared_points_sensor[
            self.imaging_source_ray_indices
        ].copy()
        imaging_points_world = shared_points_world[
            self.imaging_source_ray_indices
        ].copy()
        imaging = RegisteredSideScanReturns(
            timestamp=float(truth.timestamp),
            frame_id="integrated_sonar",
            origin_world=truth.position.copy(),
            beam_angles=self.imaging_angles.copy(),
            beam_directions=self.imaging_directions_sensor.copy(),
            ranges=imaging_ranges,
            intensities=imaging_intensity,
            hit_mask=imaging_hit,
            points_sensor=imaging_points_sensor,
            points_world=imaging_points_world,
            source_ray_indices=self.imaging_source_ray_indices.copy(),
            ray_origins_sensor=self.shared_ray_origins_sensor[
                self.imaging_source_ray_indices
            ].copy(),
            ray_origins_world=shared_origins_world[
                self.imaging_source_ray_indices
            ].copy(),
            along_track_row_indices=self.imaging_row_indices.copy(),
        )
        port, starboard, port_source, starboard_source = (
            self._registered_sidescan_lines(
                imaging.points_sensor,
                imaging.intensities,
                imaging.hit_mask,
                imaging.beam_directions,
                imaging.ray_origins_sensor,
            )
        )
        centres = np.linspace(
            self.min_range,
            self.max_range,
            self.sidescan_range_bins + 1,
            dtype=np.float64,
        )
        centres = 0.5 * (centres[:-1] + centres[1:])
        port_mask = imaging.hit_mask & (imaging.beam_directions[:, 1] >= 0.0)
        starboard_mask = imaging.hit_mask & (
            imaging.beam_directions[:, 1] < 0.0
        )
        side_scan = SideScanPing(
            timestamp=float(truth.timestamp),
            frame_id="integrated_sonar",
            sound_speed=1500.0,
            frequency=400_000.0,
            range_bin_centres=centres,
            port=port,
            starboard=starboard,
            display_line=np.rint(
                compose_sidescan_line(port, starboard) * 255.0
            ).astype(np.uint8),
            port_points_world=np.ascontiguousarray(
                imaging.points_world[port_mask], dtype=np.float32
            ),
            starboard_points_world=np.ascontiguousarray(
                imaging.points_world[starboard_mask], dtype=np.float32
            ),
        )
        ping = BathymetricSideScanPing(
            bathymetry=bathymetry,
            side_scan=side_scan,
            port_source_beam_indices=imaging_lineage_to_bathymetry(
                port_source,
                self.imaging_source_ray_indices,
                self.bathymetry_source_ray_indices,
            ),
            starboard_source_beam_indices=imaging_lineage_to_bathymetry(
                starboard_source,
                self.imaging_source_ray_indices,
                self.bathymetry_source_ray_indices,
            ),
            origin_world=truth.position.copy(),
            shared_ray_directions=self.shared_ray_directions_sensor.copy(),
            shared_ray_origins_sensor=(
                self.shared_ray_origins_sensor.copy()
            ),
            shared_ray_origins_world=shared_origins_world.copy(),
            bathymetry_source_ray_indices=(
                self.bathymetry_source_ray_indices.copy()
            ),
            imaging=imaging,
            port_source_imaging_indices=port_source,
            starboard_source_imaging_indices=starboard_source,
            products_share_raw_returns=True,
            along_track_coverage_m=(
                self.along_track_rows_per_side
                * self.along_track_sample_spacing_m
                if self.along_track_rows_per_side > 1
                else 0.0
            ),
        )
        self.last_integrated_ping = ping
        return ping

    def bathymetric_sidescan(
        self, truth: GroundTruthState
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Compatibility tuple for existing sparse-geometry consumers.

        New code needing the dense point-wise image returns should use
        :meth:`bathymetric_sidescan_ping` and ``ping.valid_imaging_samples()``.
        """

        ping = self.bathymetric_sidescan_ping(truth)
        return (
            ping.bathymetry.points_sensor,
            ping.bathymetry.intensities,
            ping.bathymetry.hit_mask,
            ping.side_scan.port,
            ping.side_scan.starboard,
        )
