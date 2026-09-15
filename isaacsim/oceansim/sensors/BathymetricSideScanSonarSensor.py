"""Integrated multibeam bathymetry and co-registered side-scan sonar."""

from __future__ import annotations

import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix

from isaacsim.oceansim.sensors.MultibeamEchosounderSensor import (
    MultibeamEchosounderSensor,
)
from isaacsim.oceansim.utils.acoustic_sonar import (
    BathymetricSideScanPing,
    MultibeamPing,
    RegisteredSideScanReturns,
    SideScanPing,
    add_sonar_noise,
    bin_coregistered_sidescan_returns,
    compose_sidescan_line,
    compute_backscatter_intensity,
    imaging_lineage_to_bathymetry,
    integrated_bathymetric_sidescan_fan,
    normalize_intensities,
    parallel_bathymetric_sidescan_rays,
    perturb_ranges,
)
from isaacsim.oceansim.utils.acoustic_visualization import (
    colorize_sidescan,
    resample_sidescan_ground_range,
    tone_map_sidescan_power,
)


class BathymetricSideScanSonarSensor(MultibeamEchosounderSensor):
    """Generate bathymetry and side-scan imagery from one acoustic ping.

    Unlike mounting independent MBES and side-scan simulators together, this
    model launches one down-looking raw-return grid.  Each returned echo has
    one measured range/XYZ and one measured backscatter value.  The bathymetry
    point cloud is a deterministic output view of that grid and the side-scan
    image bins the same samples, so neither product owns an independent ray,
    dropout decision, range perturbation, or timestamp.

    The model remains a first-return geometric acoustic approximation.  It
    does not synthesize coherent phase interferometry, refraction, multipath,
    or water-column beam samples.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "bathymetric_side_scan",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        num_beams: int = 256,
        swath_width_deg: float = 120.0,
        range_bins: int = 320,
        imaging_rays_per_side: int = 512,
        along_track_beams_per_side: int = 1,
        along_track_beam_spacing_deg: float = 0.0,
        along_track_origin_spacing_m: float = 0.0,
        waterfall_lines: int = 260,
        waterfall_decimation: int = 1,
        beam_footprint_bins: int = 1,
        min_range: float = 0.5,
        max_range: float = 75.0,
        sound_speed: float = 1500.0,
        center_frequency_hz: float = 400_000.0,
        range_noise_std: float = 0.02,
        dropout_probability: float = 0.0,
        multiplicative_noise_std: float = 0.02,
        additive_noise_std: float = 0.01,
        attenuation: float = 0.01,
        gain: float = 1.0,
        default_reflectivity: float = 1.0,
        reflectivity_resolver=None,
        spatial_reflectivity_resolver=None,
        random_seed: int | None = None,
        raycast_backend=None,
        frame_id: str | None = None,
        viewport: bool = False,
        draw_in_scene: bool = False,
        viewport_resolution: tuple[int, int] = (900, 520),
        sidescan_viewport_resolution: tuple[int, int] = (1024, 520),
    ) -> None:
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            num_beams=num_beams,
            swath_width_deg=swath_width_deg,
            min_range=min_range,
            max_range=max_range,
            sound_speed=sound_speed,
            center_frequency_hz=center_frequency_hz,
            range_noise_std=range_noise_std,
            dropout_probability=dropout_probability,
            attenuation=attenuation,
            default_reflectivity=default_reflectivity,
            reflectivity_resolver=reflectivity_resolver,
            random_seed=random_seed,
            raycast_backend=raycast_backend,
            frame_id=frame_id,
            viewport=viewport,
            draw_in_scene=draw_in_scene,
            viewport_resolution=viewport_resolution,
        )
        self._range_bins = int(range_bins)
        self._imaging_rays_per_side = int(imaging_rays_per_side)
        self._along_track_beams_per_side = int(
            along_track_beams_per_side
        )
        self._along_track_beam_spacing_deg = float(
            along_track_beam_spacing_deg
        )
        self._along_track_origin_spacing_m = float(
            along_track_origin_spacing_m
        )
        self._waterfall_lines = int(waterfall_lines)
        self._waterfall_decimation = int(waterfall_decimation)
        self._beam_footprint_bins = int(beam_footprint_bins)
        self._multiplicative_noise_std = float(multiplicative_noise_std)
        self._additive_noise_std = float(additive_noise_std)
        self._gain = float(gain)
        if (
            spatial_reflectivity_resolver is not None
            and not callable(spatial_reflectivity_resolver)
        ):
            raise TypeError("spatial_reflectivity_resolver must be callable")
        self._spatial_reflectivity_resolver = spatial_reflectivity_resolver
        # Keep radiometric noise on an independent deterministic stream so
        # enabling the image product cannot change later bathymetric range
        # samples. This is essential for geometry-only/intensity A/B tests.
        intensity_seed = (
            None if random_seed is None else int(random_seed) + 1_000_003
        )
        self._intensity_rng = np.random.default_rng(intensity_seed)
        display_seed = (
            None if random_seed is None else int(random_seed) + 3_000_037
        )
        self._display_rng = np.random.default_rng(display_seed)
        if (
            self._range_bins < 1
            or self._imaging_rays_per_side < 1
            or self._along_track_beams_per_side < 1
            or self._waterfall_lines < 1
            or self._waterfall_decimation < 1
        ):
            raise ValueError(
                "range_bins, imaging rays/beams, waterfall lines, and "
                "waterfall decimation must be >= 1"
            )
        if (
            not np.isfinite(self._along_track_beam_spacing_deg)
            or self._along_track_beam_spacing_deg < 0.0
        ):
            raise ValueError(
                "along_track_beam_spacing_deg must be finite and non-negative"
            )
        if (
            self._along_track_beams_per_side > 1
            and self._along_track_beam_spacing_deg <= 0.0
            and self._along_track_origin_spacing_m <= 0.0
        ):
            raise ValueError(
                "multi-row side-scan requires a positive angular beam-centre "
                "spacing or spatial origin spacing"
            )
        if (
            not np.isfinite(self._along_track_origin_spacing_m)
            or self._along_track_origin_spacing_m < 0.0
        ):
            raise ValueError(
                "along_track_origin_spacing_m must be finite and non-negative"
            )
        if (
            self._along_track_origin_spacing_m > 0.0
            and self._along_track_beam_spacing_deg > 0.0
        ):
            raise ValueError(
                "angular beam-centre spacing and spatial origin spacing are "
                "mutually exclusive"
            )
        if self._beam_footprint_bins < 0:
            raise ValueError("beam_footprint_bins must be >= 0")
        if min(
            self._multiplicative_noise_std,
            self._additive_noise_std,
            self._gain,
        ) < 0.0:
            raise ValueError("intensity noise and gain values must be >= 0")

        self._sidescan_viewport_resolution = tuple(
            int(value) for value in sidescan_viewport_resolution
        )
        if (
            len(self._sidescan_viewport_resolution) != 2
            or min(self._sidescan_viewport_resolution) < 64
        ):
            raise ValueError(
                "sidescan_viewport_resolution must be (width, height) >= 64"
            )
        self._waterfall = np.zeros(
            (
                self._waterfall_lines,
                2
                * int(
                    np.ceil(
                        self._range_bins / self._waterfall_decimation
                    )
                ),
            ),
            dtype=np.uint8,
        )
        # Keep the raw slant-range waterfall for the sensor/ROS interface and
        # maintain a separate, ground-range-corrected operator display.  This
        # prevents presentation processing from changing estimator inputs or
        # claiming false source lineage for interpolated display pixels.
        self._display_waterfall = np.zeros_like(self._waterfall)
        self._waterfall_count = 0
        self._display_waterfall_count = 0
        self._display_reference_power = None
        self._display_altitude_m = float("nan")
        self._display_ground_range_m = float("nan")
        self._display_coverage_fraction = 0.0
        self._sidescan_viewport = None
        self._last_integrated_ping: BathymetricSideScanPing | None = None
        if self._along_track_origin_spacing_m > 0.0:
            (
                self._shared_ray_origins_sensor,
                self._shared_ray_directions,
                self._bathymetry_source_ray_indices,
                self._imaging_origins_sensor,
                self._imaging_directions,
                self._imaging_source_ray_indices,
                self._imaging_angles,
            ) = parallel_bathymetric_sidescan_rays(
                num_beams=self._beam_directions.shape[0],
                swath_width_deg=swath_width_deg,
                imaging_rays_per_side=self._imaging_rays_per_side,
                along_track_rows_per_side=(
                    self._along_track_beams_per_side
                ),
                along_track_sample_spacing_m=(
                    self._along_track_origin_spacing_m
                ),
            )
        else:
            (
                self._shared_ray_directions,
                self._bathymetry_source_ray_indices,
                self._imaging_directions,
                self._imaging_source_ray_indices,
                self._imaging_angles,
            ) = integrated_bathymetric_sidescan_fan(
                num_beams=self._beam_directions.shape[0],
                swath_width_deg=swath_width_deg,
                imaging_rays_per_side=self._imaging_rays_per_side,
                along_track_beams_per_side=(
                    self._along_track_beams_per_side
                ),
                along_track_beam_spacing_deg=(
                    self._along_track_beam_spacing_deg
                ),
            )
            self._shared_ray_origins_sensor = np.zeros_like(
                self._shared_ray_directions
            )
            self._imaging_origins_sensor = self._shared_ray_origins_sensor[
                self._imaging_source_ray_indices
            ].copy()
        # The inherited beam arrays now describe the published bathymetry view
        # selected from the one raw-return grid.  This keeps inherited accessors
        # and visualization consistent without creating an independent fan.
        self._beam_directions = self._shared_ray_directions[
            self._bathymetry_source_ray_indices
        ].copy()
        self._beam_angles = self._imaging_angles[
            self._bathymetry_source_ray_indices
        ].copy()
        one_side_rows = np.repeat(
            np.arange(self._along_track_beams_per_side, dtype=np.int64),
            self._imaging_rays_per_side,
        )
        self._imaging_row_indices = np.concatenate(
            (one_side_rows, one_side_rows)
        )

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        if self._viewport_enabled and self._sidescan_viewport is None:
            self.make_sidescan_viewport()

    @staticmethod
    def _spread_beam_footprints(
        line: np.ndarray, source: np.ndarray, radius: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spread a detection over finite range cells without losing lineage."""

        if radius <= 0:
            return line.copy(), source.copy()
        spread = np.zeros_like(line)
        spread_source = np.full(source.shape, -1, dtype=np.int64)
        populated = np.flatnonzero(source >= 0)
        if populated.size == 0:
            return spread, spread_source

        # Build contributions in the original centre-major, offset-minor
        # order.  maximum.at supplies the same strongest value, while the
        # minimum winning order preserves the scalar loop's strict-`>` tie
        # rule and therefore its exact source lineage.
        offsets = np.arange(-radius, radius + 1, dtype=np.int64)
        indices = populated[:, None] + offsets[None, :]
        weights = 1.0 - np.abs(offsets) / float(radius + 1)
        contributions = np.asarray(line)[populated, None] * weights[None, :]
        source_values = np.broadcast_to(
            np.asarray(source)[populated, None], indices.shape
        )
        valid = (
            (indices >= 0)
            & (indices < line.size)
            & (contributions > 0.0)
        )
        flat_indices = indices.ravel()
        flat_contributions = contributions.ravel()
        flat_sources = source_values.ravel()
        flat_valid = valid.ravel()
        np.maximum.at(
            spread,
            flat_indices[flat_valid],
            flat_contributions[flat_valid],
        )
        order = np.arange(flat_indices.size, dtype=np.int64)
        winners = flat_valid & (
            flat_contributions == spread[np.clip(flat_indices, 0, line.size - 1)]
        )
        first_winner = np.full(line.size, flat_indices.size, dtype=np.int64)
        np.minimum.at(first_winner, flat_indices[winners], order[winners])
        assigned = first_winner < flat_indices.size
        spread_source[assigned] = flat_sources[first_winner[assigned]]
        return spread, spread_source

    def _bin_imaging_beam_rows(
        self,
        ranges: np.ndarray,
        intensities: np.ndarray,
        hit_mask: np.ndarray,
        directions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bin each documented along-track beam without collapsing its row."""

        beams = self._along_track_beams_per_side
        rays = self._imaging_rays_per_side
        expected = 2 * beams * rays
        if np.asarray(ranges).size != expected:
            raise RuntimeError(
                f"expected {expected} imaging rays for {beams} beams/side, "
                f"got {np.asarray(ranges).size}"
            )
        port_rows = []
        starboard_rows = []
        port_sources = []
        starboard_sources = []
        centres = None
        port_count = beams * rays
        for beam in range(beams):
            port_start = beam * rays
            starboard_start = port_count + beam * rays
            selected_indices = np.concatenate(
                (
                    np.arange(port_start, port_start + rays),
                    np.arange(starboard_start, starboard_start + rays),
                )
            )
            (
                row_centres,
                port,
                starboard,
                port_source,
                starboard_source,
            ) = bin_coregistered_sidescan_returns(
                np.asarray(ranges)[selected_indices],
                np.asarray(intensities)[selected_indices],
                np.asarray(hit_mask, dtype=bool)[selected_indices],
                np.asarray(directions)[selected_indices],
                self._min_range,
                self._max_range,
                self._range_bins,
            )
            port_valid = port_source >= 0
            starboard_valid = starboard_source >= 0
            port_source[port_valid] = selected_indices[
                port_source[port_valid]
            ]
            starboard_source[starboard_valid] = selected_indices[
                starboard_source[starboard_valid]
            ]
            port, port_source = self._spread_beam_footprints(
                port, port_source, self._beam_footprint_bins
            )
            starboard, starboard_source = self._spread_beam_footprints(
                starboard,
                starboard_source,
                self._beam_footprint_bins,
            )
            centres = row_centres
            port_rows.append(port)
            starboard_rows.append(starboard)
            port_sources.append(port_source)
            starboard_sources.append(starboard_source)
        return (
            np.asarray(centres, dtype=np.float64),
            np.stack(port_rows),
            np.stack(starboard_rows),
            np.stack(port_sources),
            np.stack(starboard_sources),
        )

    @staticmethod
    def _aggregate_beam_rows(
        rows: np.ndarray, sources: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Make the legacy one-line view while retaining strongest lineage."""

        best_row = np.argmax(rows, axis=0)
        columns = np.arange(rows.shape[1], dtype=np.int64)
        values = rows[best_row, columns]
        lineage = sources[best_row, columns].copy()
        lineage[values <= 0.0] = -1
        return values, lineage

    def _decimate_display_line(self, line: np.ndarray) -> np.ndarray:
        """Max-pool native range samples for a responsive UI waterfall."""

        factor = self._waterfall_decimation
        line = np.asarray(line, dtype=np.uint8).reshape(2, self._range_bins)
        if factor == 1:
            return line.reshape(-1).copy()
        output_bins = int(np.ceil(self._range_bins / factor))
        padded_bins = output_bins * factor
        if padded_bins != self._range_bins:
            line = np.pad(
                line,
                ((0, 0), (0, padded_bins - self._range_bins)),
                mode="constant",
            )
        return line.reshape(2, output_bins, factor).max(axis=2).reshape(-1)

    def _ground_range_display_lines(
        self, imaging: RegisteredSideScanReturns
    ) -> np.ndarray:
        """Build processed UI rows without changing the raw sonar payload."""

        rows = self._along_track_beams_per_side
        rays = self._imaging_rays_per_side
        side_count = rows * rays
        output_bins = self._display_waterfall.shape[1] // 2
        ranges = np.asarray(imaging.ranges, dtype=np.float64)
        intensities = np.asarray(imaging.intensities, dtype=np.float64)
        hits = np.asarray(imaging.hit_mask, dtype=bool)
        directions = np.asarray(imaging.beam_directions, dtype=np.float64)

        port_ranges = ranges[:side_count].reshape(rows, rays)
        starboard_ranges = ranges[side_count:].reshape(rows, rays)
        port_intensities = intensities[:side_count].reshape(rows, rays)
        starboard_intensities = intensities[side_count:].reshape(rows, rays)
        port_hits = hits[:side_count].reshape(rows, rays)
        starboard_hits = hits[side_count:].reshape(rows, rays)
        port_directions = directions[:side_count].reshape(rows, rays, 3)
        starboard_directions = directions[side_count:].reshape(rows, rays, 3)

        power = np.zeros((rows, 2, output_bins), dtype=np.float64)
        coverage = np.zeros((rows, 2, output_bins), dtype=bool)
        altitudes = []
        ground_ranges = []
        channels = (
            (
                port_ranges,
                port_intensities,
                port_hits,
                port_directions,
            ),
            (
                starboard_ranges,
                starboard_intensities,
                starboard_hits,
                starboard_directions,
            ),
        )
        for side, channel in enumerate(channels):
            side_ranges, side_intensities, side_hits, side_directions = channel
            for row in range(rows):
                (
                    power[row, side],
                    coverage[row, side],
                    altitude,
                    maximum_ground_range,
                ) = resample_sidescan_ground_range(
                    side_ranges[row],
                    side_intensities[row],
                    side_hits[row],
                    side_directions[row],
                    max_range=self._max_range,
                    output_bins=output_bins,
                    attenuation=self._attenuation,
                )
                if np.isfinite(altitude):
                    altitudes.append(altitude)
                if maximum_ground_range > 0.0:
                    ground_ranges.append(maximum_ground_range)

        positive = coverage & np.isfinite(power) & (power > 0.0)
        if np.any(positive):
            candidate = float(np.percentile(power[positive], 98.0))
            if self._display_reference_power is None:
                self._display_reference_power = candidate
            else:
                # A real receiver's AGC changes slowly; smoothing prevents
                # distracting line-to-line brightness pumping in the UI.
                self._display_reference_power = (
                    0.92 * self._display_reference_power + 0.08 * candidate
                )
        reference = self._display_reference_power or 1.0
        median_ground_range = (
            float(np.median(ground_ranges)) if ground_ranges else 0.0
        )
        nadir_mute_bins = (
            max(
                1,
                int(np.ceil(output_bins / max(median_ground_range, 1.0))),
            )
            if median_ground_range > 0.0
            else 0
        )
        mono = tone_map_sidescan_power(
            power,
            coverage,
            rng=self._display_rng,
            reference_power=reference,
            dynamic_range_db=36.0,
            speckle_looks=5.0,
            noise_floor_db=-34.0,
            nadir_mute_bins=nadir_mute_bins,
        )
        self._display_altitude_m = (
            float(np.median(altitudes)) if altitudes else float("nan")
        )
        self._display_ground_range_m = (
            median_ground_range if ground_ranges else float("nan")
        )
        self._display_coverage_fraction = float(np.mean(coverage))
        return np.concatenate((mono[:, 0, ::-1], mono[:, 1]), axis=1)

    def scan(
        self,
        timestamp: float | None = None,
        *,
        update_visualization: bool = True,
    ) -> BathymetricSideScanPing:
        """Emit one shared launch and select sparse and dense output products.

        ``update_visualization`` controls only the two live image windows and
        the in-scene ray guides.  Ray casting, noise draws, raw measurements,
        the slant-range waterfall, and the returned ping are unchanged when it
        is false.  Accelerated applications can therefore present only the
        newest ping before a rendered frame without decimating estimator data.
        """

        if self._backend is None:
            from isaacsim.oceansim.sensors.raycast_backend import (
                PhysXRaycastBackend,
            )

            self._backend = PhysXRaycastBackend(**self._backend_kwargs)

        origin_world, orientation_world = self.get_world_pose()
        origin_world = np.asarray(origin_world, dtype=np.float64).reshape(3)
        rotation_world_sensor = quat_to_rot_matrix(orientation_world)
        shared_directions_world = (
            self._shared_ray_directions @ rotation_world_sensor.T
        )
        shared_ray_origins_world = (
            origin_world
            + self._shared_ray_origins_sensor @ rotation_world_sensor.T
        )
        # Preserve the historical single-origin call for zero-offset sensors and
        # third-party backends. Spatial multi-row profiles require the expanded
        # built-in backend contract with one origin per ray.
        backend_origin_world = (
            shared_ray_origins_world
            if np.any(self._shared_ray_origins_sensor != 0.0)
            else origin_world
        )

        accelerated_cast = getattr(self._backend, "cast_with_intensity", None)
        if callable(accelerated_cast):
            hits = accelerated_cast(
                origin_world=backend_origin_world,
                directions_world=shared_directions_world,
                max_range=self._max_range,
                attenuation=self._attenuation,
            )
        else:
            hits = self._backend.cast(
                origin_world=backend_origin_world,
                directions_world=shared_directions_world,
                max_range=self._max_range,
            )

        geometric_hit = (
            hits.hit_mask
            & np.isfinite(hits.distances)
            & (hits.distances >= self._min_range)
            & (hits.distances <= self._max_range)
        )
        # Apply channel dropout and range error once to the raw echo grid.  Both
        # output products select from these exact arrays below; there is no
        # second bathymetry or image measurement draw.
        shared_hit = geometric_hit.copy()
        if self._dropout_probability > 0.0:
            shared_hit &= (
                self._rng.random(shared_hit.size)
                >= self._dropout_probability
            )
        shared_ranges, _ = perturb_ranges(
            hits.distances,
            shared_hit,
            self._rng,
            gaussian_std=self._range_noise_std,
        )
        shared_hit &= (
            np.isfinite(shared_ranges)
            & (shared_ranges >= self._min_range)
            & (shared_ranges <= self._max_range)
        )
        shared_ranges[~shared_hit] = np.nan

        accelerated_intensities = getattr(hits, "raw_intensities", None)
        if self._spatial_reflectivity_resolver is not None:
            resolved_reflectivity = np.asarray(
                self._spatial_reflectivity_resolver(
                    hits.positions_world[:, 0],
                    hits.positions_world[:, 1],
                ),
                dtype=np.float64,
            ).reshape(shared_hit.size)
            resolved_reflectivity = np.nan_to_num(
                resolved_reflectivity, nan=0.0, posinf=0.0, neginf=0.0
            )
            resolved_reflectivity = np.clip(
                resolved_reflectivity, 0.0, None
            )
            raw_intensities = compute_backscatter_intensity(
                distances=hits.distances,
                directions_world=shared_directions_world,
                normals_world=hits.normals_world,
                reflectivities=resolved_reflectivity,
                hit_mask=shared_hit,
                attenuation=self._attenuation,
            )
        elif accelerated_intensities is None:
            raw_intensities = compute_backscatter_intensity(
                distances=hits.distances,
                directions_world=shared_directions_world,
                normals_world=hits.normals_world,
                reflectivities=hits.reflectivities,
                hit_mask=shared_hit,
                attenuation=self._attenuation,
            )
        else:
            raw_intensities = np.asarray(
                accelerated_intensities, dtype=np.float64
            ).copy()
            raw_intensities[~shared_hit] = 0.0

        shared_intensities = normalize_intensities(
            raw_intensities, shared_hit
        )
        shared_intensities = add_sonar_noise(
            shared_intensities,
            np.where(shared_hit, shared_ranges, 0.0),
            self._max_range,
            self._intensity_rng,
            multiplicative_std=self._multiplicative_noise_std,
            additive_std=self._additive_noise_std,
        )
        shared_intensities = np.clip(
            shared_intensities * self._gain, 0.0, 1.0
        )
        shared_intensities[~shared_hit] = 0.0

        bathymetry_hit = shared_hit[
            self._bathymetry_source_ray_indices
        ].copy()
        bathymetry_ranges = shared_ranges[
            self._bathymetry_source_ray_indices
        ].copy()
        bathymetry_intensities = shared_intensities[
            self._bathymetry_source_ray_indices
        ].copy()
        bathymetry_origins_sensor = self._shared_ray_origins_sensor[
            self._bathymetry_source_ray_indices
        ]
        bathymetry_origins_world = shared_ray_origins_world[
            self._bathymetry_source_ray_indices
        ]
        bathymetry_points_sensor = (
            bathymetry_origins_sensor
            + self._beam_directions * bathymetry_ranges[:, None]
        )
        bathymetry_directions_world = shared_directions_world[
            self._bathymetry_source_ray_indices
        ]
        bathymetry_points_world = (
            bathymetry_origins_world
            + bathymetry_directions_world * bathymetry_ranges[:, None]
        )
        bathymetry_points_sensor[~bathymetry_hit] = np.nan
        bathymetry_points_world[~bathymetry_hit] = np.nan
        observation_time = (
            self._simulation_time() if timestamp is None else float(timestamp)
        )
        if hits.prim_paths:
            bathymetry_prim_paths = tuple(
                hits.prim_paths[int(index)]
                for index in self._bathymetry_source_ray_indices
            )
        else:
            bathymetry_prim_paths = ()
        bathymetry = MultibeamPing(
            timestamp=observation_time,
            frame_id=self._frame_id,
            sound_speed=self._sound_speed,
            frequency=self._center_frequency_hz,
            beam_angles=self._beam_angles.copy(),
            beam_directions=self._beam_directions.copy(),
            ranges=bathymetry_ranges,
            two_way_travel_times=(
                2.0 * bathymetry_ranges / self._sound_speed
            ),
            intensities=bathymetry_intensities,
            hit_mask=bathymetry_hit,
            points_sensor=bathymetry_points_sensor,
            points_world=bathymetry_points_world,
            prim_paths=bathymetry_prim_paths,
        )
        self._last_ping = bathymetry
        if update_visualization:
            self._update_visualization(bathymetry, origin_world)

        imaging_hit = shared_hit[self._imaging_source_ray_indices]
        imaging_ranges = shared_ranges[
            self._imaging_source_ray_indices
        ].copy()
        imaging_intensities = shared_intensities[
            self._imaging_source_ray_indices
        ].copy()
        imaging_origins_world = shared_ray_origins_world[
            self._imaging_source_ray_indices
        ]
        imaging_points_sensor = (
            self._imaging_origins_sensor
            + self._imaging_directions * imaging_ranges[:, None]
        )
        imaging_directions_world = shared_directions_world[
            self._imaging_source_ray_indices
        ]
        imaging_points_world = (
            imaging_origins_world
            + imaging_directions_world * imaging_ranges[:, None]
        )
        imaging_points_sensor[~imaging_hit] = np.nan
        imaging_points_world[~imaging_hit] = np.nan
        imaging = RegisteredSideScanReturns(
            timestamp=observation_time,
            frame_id=self._frame_id,
            origin_world=origin_world.copy(),
            beam_angles=self._imaging_angles.copy(),
            beam_directions=self._imaging_directions.copy(),
            ranges=imaging_ranges,
            intensities=imaging_intensities,
            hit_mask=imaging_hit,
            points_sensor=imaging_points_sensor,
            points_world=imaging_points_world,
            source_ray_indices=self._imaging_source_ray_indices.copy(),
            ray_origins_sensor=self._imaging_origins_sensor.copy(),
            ray_origins_world=imaging_origins_world.copy(),
            along_track_row_indices=self._imaging_row_indices.copy(),
        )

        (
            centres,
            port_beams,
            starboard_beams,
            port_beam_sources,
            starboard_beam_sources,
        ) = self._bin_imaging_beam_rows(
            imaging.ranges,
            imaging.intensities,
            imaging.hit_mask,
            imaging.beam_directions,
        )
        port, port_imaging_source = self._aggregate_beam_rows(
            port_beams, port_beam_sources
        )
        starboard, starboard_imaging_source = self._aggregate_beam_rows(
            starboard_beams, starboard_beam_sources
        )
        port_bathymetry_source = imaging_lineage_to_bathymetry(
            port_imaging_source,
            self._imaging_source_ray_indices,
            self._bathymetry_source_ray_indices,
        )
        starboard_bathymetry_source = imaging_lineage_to_bathymetry(
            starboard_imaging_source,
            self._imaging_source_ray_indices,
            self._bathymetry_source_ray_indices,
        )
        display_line = np.rint(
            compose_sidescan_line(port, starboard) * 255.0
        ).astype(np.uint8)
        raw_waterfall_lines = np.stack(
            [
                self._decimate_display_line(
                    np.rint(
                        compose_sidescan_line(port_row, starboard_row)
                        * 255.0
                    ).astype(np.uint8)
                )
                for port_row, starboard_row in zip(
                    port_beams, starboard_beams
                )
            ]
        )
        rows_to_add = min(
            self._waterfall_lines, raw_waterfall_lines.shape[0]
        )
        if rows_to_add < self._waterfall_lines:
            self._waterfall[:-rows_to_add] = self._waterfall[rows_to_add:]
        self._waterfall[-rows_to_add:] = raw_waterfall_lines[-rows_to_add:]
        self._waterfall_count = min(
            self._waterfall_count + raw_waterfall_lines.shape[0],
            self._waterfall_lines,
        )
        if self._viewport_enabled and update_visualization:
            corrected_lines = self._ground_range_display_lines(imaging)
            corrected_rows = min(
                self._waterfall_lines, corrected_lines.shape[0]
            )
            if corrected_rows < self._waterfall_lines:
                self._display_waterfall[:-corrected_rows] = (
                    self._display_waterfall[corrected_rows:]
                )
            self._display_waterfall[-corrected_rows:] = corrected_lines[
                -corrected_rows:
            ]
            self._display_waterfall_count = min(
                self._display_waterfall_count + corrected_lines.shape[0],
                self._waterfall_lines,
            )

        valid = imaging.hit_mask
        port_mask = valid & (imaging.beam_directions[:, 1] >= 0.0)
        starboard_mask = valid & (imaging.beam_directions[:, 1] < 0.0)
        sidescan = SideScanPing(
            timestamp=bathymetry.timestamp,
            frame_id=bathymetry.frame_id,
            sound_speed=bathymetry.sound_speed,
            frequency=bathymetry.frequency,
            range_bin_centres=centres,
            port=port,
            starboard=starboard,
            display_line=display_line,
            port_points_world=np.ascontiguousarray(
                imaging.points_world[port_mask], dtype=np.float32
            ),
            starboard_points_world=np.ascontiguousarray(
                imaging.points_world[starboard_mask], dtype=np.float32
            ),
            port_beams=port_beams,
            starboard_beams=starboard_beams,
        )
        ping = BathymetricSideScanPing(
            bathymetry=bathymetry,
            side_scan=sidescan,
            port_source_beam_indices=port_bathymetry_source,
            starboard_source_beam_indices=starboard_bathymetry_source,
            origin_world=origin_world,
            shared_ray_directions=self._shared_ray_directions.copy(),
            shared_ray_origins_sensor=(
                self._shared_ray_origins_sensor.copy()
            ),
            shared_ray_origins_world=shared_ray_origins_world.copy(),
            bathymetry_source_ray_indices=(
                self._bathymetry_source_ray_indices.copy()
            ),
            imaging=imaging,
            port_source_imaging_indices=port_imaging_source,
            starboard_source_imaging_indices=starboard_imaging_source,
            products_share_raw_returns=True,
            along_track_coverage_m=(
                self._along_track_beams_per_side
                * self._along_track_origin_spacing_m
            ),
        )
        self._last_integrated_ping = ping
        if update_visualization:
            self._update_sidescan_visualization(ping)
        return ping

    def make_sidescan_viewport(self) -> None:
        if self._sidescan_viewport is not None:
            return
        from isaacsim.oceansim.sensors.acoustic_visualizer import (
            AcousticImageViewport,
        )

        width, height = self._sidescan_viewport_resolution
        self._sidescan_viewport = AcousticImageViewport(
            f"{self.name} | Ground-range Side-scan | Amber Power",
            width=width,
            height=height,
            footer=(
                "AMBER POWER LUT + SRC + TVG  |  FAR PORT  <---  TRACK  "
                "--->  FAR STARBOARD"
            ),
        )

    def _update_sidescan_visualization(
        self, ping: BathymetricSideScanPing
    ) -> None:
        if self._sidescan_viewport is None:
            return
        rgba = colorize_sidescan(
            self._display_waterfall,
            gamma=1.0,
            mark_nadir=False,
            palette="amber",
        )
        if ping.side_scan.port_beams is None:
            valid_sources = np.count_nonzero(
                ping.port_source_imaging_indices >= 0
            ) + np.count_nonzero(
                ping.starboard_source_imaging_indices >= 0
            )
            row_text = "1 row/side"
        else:
            valid_sources = np.count_nonzero(
                ping.side_scan.port_beams > 0.0
            ) + np.count_nonzero(ping.side_scan.starboard_beams > 0.0)
            row_text = f"{ping.side_scan.port_beams.shape[0]} rows/side"
        altitude_text = (
            f"{self._display_altitude_m:.1f} m"
            if np.isfinite(self._display_altitude_m)
            else "n/a"
        )
        ground_range_text = (
            f"{self._display_ground_range_m:.0f} m/side"
            if np.isfinite(self._display_ground_range_m)
            else "n/a"
        )
        self._sidescan_viewport.update(
            rgba,
            f"t={ping.timestamp:.2f} s  |  raw/published hits "
            f"{np.count_nonzero(ping.imaging.hit_mask)}/"
            f"{np.count_nonzero(ping.bathymetry.hit_mask)}  |  "
            f"{row_text}  |  bottom {altitude_text}  |  "
            f"ground {ground_range_text}  |  coverage "
            f"{100.0 * self._display_coverage_fraction:.0f}%  |  "
            f"raw cells {valid_sources}",
        )

    def get_last_ping(self) -> BathymetricSideScanPing | None:
        return self._last_integrated_ping

    def get_last_bathymetry_ping(self):
        return self._last_ping

    def get_waterfall(self, valid_only: bool = False) -> np.ndarray:
        if not valid_only or self._waterfall_count == self._waterfall_lines:
            return self._waterfall.copy()
        if self._waterfall_count == 0:
            return np.empty((0, self._waterfall.shape[1]), dtype=np.uint8)
        return self._waterfall[-self._waterfall_count :].copy()

    def get_processed_waterfall(self, valid_only: bool = False) -> np.ndarray:
        """Return the display-only SRC/TVG waterfall as a mono8 image."""

        if (
            not valid_only
            or self._display_waterfall_count == self._waterfall_lines
        ):
            return self._display_waterfall.copy()
        if self._display_waterfall_count == 0:
            return np.empty(
                (0, self._display_waterfall.shape[1]), dtype=np.uint8
            )
        return self._display_waterfall[
            -self._display_waterfall_count :
        ].copy()

    def clear_waterfall(self) -> None:
        self._waterfall.fill(0)
        self._display_waterfall.fill(0)
        self._waterfall_count = 0
        self._display_waterfall_count = 0
        self._display_reference_power = None

    def close(self) -> None:
        if self._sidescan_viewport is not None:
            self._sidescan_viewport.close()
            self._sidescan_viewport = None
        super().close()
