"""Pure numerical utilities shared by OceanSim acoustic sensors.

The functions in this module deliberately do not import Isaac Sim.  Keeping the
geometry, binning, and noise model independent from the simulator makes those
parts testable with a normal Python interpreter and allows alternative raycast
backends to be added later.

Coordinate convention
---------------------
Sensor-local vectors use ``+X`` forward, ``+Y`` port/left, and ``+Z`` up.  A
down-looking multibeam therefore has a boresight of ``[0, 0, -1]``.  Multibeam
steering angles follow the convention used by ``marine_acoustic_msgs``:
positive across-track angles point to starboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _finite_float(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def broadcast_ray_origins(
    origin_world: np.ndarray, ray_count: int
) -> np.ndarray:
    """Return one finite world-frame origin for every ray.

    Acoustic backends historically accepted one ``(3,)`` origin shared by the
    entire fan.  Some multi-row sonars instead expose several simultaneous,
    spatially separated phase centres.  This helper accepts either the legacy
    single origin or an explicit ``(ray_count, 3)`` array and normalizes both to
    the latter representation.
    """

    ray_count = int(ray_count)
    if ray_count < 0:
        raise ValueError("ray_count must be non-negative")
    origins = np.asarray(origin_world, dtype=np.float64)
    if origins.shape == (3,):
        origins = np.broadcast_to(origins, (ray_count, 3)).copy()
    elif origins.shape == (1, 3):
        origins = np.broadcast_to(origins, (ray_count, 3)).copy()
    elif origins.shape == (ray_count, 3):
        origins = origins.copy()
    else:
        raise ValueError(
            "origin_world must have shape (3,), (1, 3), or "
            f"({ray_count}, 3)"
        )
    if not np.isfinite(origins).all():
        raise ValueError("origin_world must contain only finite values")
    return np.ascontiguousarray(origins, dtype=np.float64)


def validate_range_limits(min_range: float, max_range: float) -> tuple[float, float]:
    """Validate and return a sonar's range interval in metres."""

    min_range = _finite_float("min_range", min_range)
    max_range = _finite_float("max_range", max_range)
    if min_range < 0.0:
        raise ValueError(f"min_range must be >= 0, got {min_range}")
    if max_range <= min_range:
        raise ValueError(
            f"max_range must be greater than min_range, got {max_range} <= {min_range}"
        )
    return min_range, max_range


def multibeam_directions(
    num_beams: int, swath_width_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Create a down-looking, across-track multibeam fan.

    Returns:
        ``(angles, directions)`` where angles are in radians, negative is port,
        positive is starboard, and directions are unit vectors with shape
        ``(num_beams, 3)`` in the sensor frame.
    """

    num_beams = _positive_int("num_beams", num_beams)
    swath_width_deg = _finite_float("swath_width_deg", swath_width_deg)
    if not 0.0 < swath_width_deg < 180.0:
        raise ValueError("swath_width_deg must be in the open interval (0, 180)")

    angles = np.deg2rad(
        np.linspace(-0.5 * swath_width_deg, 0.5 * swath_width_deg, num_beams)
    )
    # Positive steering is starboard, which is -Y in the local REP-103 frame.
    directions = np.column_stack(
        (
            np.zeros_like(angles),
            -np.sin(angles),
            -np.cos(angles),
        )
    )
    return angles.astype(np.float64), directions.astype(np.float64)


def _stratified_bathymetry_indices(
    rays_per_side: int,
    rows_per_side: int,
    output_count: int,
) -> np.ndarray:
    """Select a balanced bathymetry view from one raw echo grid.

    The integrated instrument has one beam-major return grid: all port rows
    followed by all starboard rows.  A consumer may request a smaller point
    cloud than that native grid for bandwidth or estimator cost, but those
    points must remain an exact subset of the raw acquisition rather than a
    second independently cast fan.
    """

    rays_per_side = _positive_int("rays_per_side", rays_per_side)
    rows_per_side = _positive_int("rows_per_side", rows_per_side)
    output_count = _positive_int("output_count", output_count)
    samples_per_side = rays_per_side * rows_per_side
    available = 2 * samples_per_side
    if output_count > available:
        raise ValueError(
            "bathymetry output count cannot exceed the unified raw-return "
            f"count ({output_count} > {available})"
        )

    port_count = (output_count + 1) // 2
    starboard_count = output_count // 2

    def cell_centres(count: int) -> np.ndarray:
        if count == 0:
            return np.empty(0, dtype=np.int64)
        return np.floor(
            (np.arange(count, dtype=np.float64) + 0.5)
            * samples_per_side
            / count
        ).astype(np.int64)

    return np.concatenate(
        (
            cell_centres(port_count),
            samples_per_side + cell_centres(starboard_count),
        )
    )


def integrated_bathymetric_sidescan_fan(
    num_beams: int,
    swath_width_deg: float,
    imaging_rays_per_side: int,
    along_track_beams_per_side: int = 1,
    along_track_beam_spacing_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one raw echo fan shared exactly by bathymetry and imagery.

    The native acquisition samples the angular swath densely, with an equal
    number of strictly port and strictly starboard rays.  ``num_beams`` is the
    size of a deterministic, balanced bathymetry *view* selected from that raw
    grid; it never creates a second set of acoustic rays.  By default the grid
    retains the historical, zero-thickness cross-track fan.  When
    ``along_track_beams_per_side`` is greater than one, each side contains a
    beam-major stack of three-dimensional fans.  Adjacent beam centres are
    separated by ``along_track_beam_spacing_deg`` and are symmetric about the
    local Y-Z plane.  The fixed output order is all port beams followed by all
    starboard beams; within each beam, rays retain their historical near/far
    cross-track ordering.

    Consequently every published bathymetric range, point, and intensity has
    an exact source sample in the side-scan return grid.  A backend launches
    each raw direction once and both products are derived from that result.

    Returns:
        ``(shared_directions, bathymetry_indices, imaging_directions,
        imaging_indices, imaging_angles)``.  Angles use the multibeam sign
        convention: negative is port and positive is starboard.
    """

    num_beams = _positive_int("num_beams", num_beams)
    swath_width_deg = _finite_float("swath_width_deg", swath_width_deg)
    if not 0.0 < swath_width_deg < 180.0:
        raise ValueError("swath_width_deg must be in the open interval (0, 180)")
    imaging_rays_per_side = _positive_int(
        "imaging_rays_per_side", imaging_rays_per_side
    )
    along_track_beams_per_side = _positive_int(
        "along_track_beams_per_side", along_track_beams_per_side
    )
    along_track_beam_spacing_deg = _finite_float(
        "along_track_beam_spacing_deg", along_track_beam_spacing_deg
    )
    if along_track_beam_spacing_deg < 0.0:
        raise ValueError("along_track_beam_spacing_deg must be >= 0")
    if (
        along_track_beams_per_side > 1
        and along_track_beam_spacing_deg <= 0.0
    ):
        raise ValueError(
            "along_track_beam_spacing_deg must be > 0 when multiple "
            "along-track beams are requested"
        )

    half_swath = 0.5 * np.deg2rad(float(swath_width_deg))
    # Cell-centred steering avoids assigning the nadir direction to both sides.
    positive = half_swath * (
        np.arange(imaging_rays_per_side, dtype=np.float64) + 0.5
    ) / float(imaging_rays_per_side)
    port_angles = -positive[::-1]
    starboard_angles = positive
    if along_track_beams_per_side == 1:
        # Preserve the exact historical ordering and arithmetic in the default
        # mode.  Existing source-index maps therefore remain bit-for-bit stable.
        imaging_angles = np.concatenate((port_angles, starboard_angles))
        imaging_directions = np.column_stack(
            (
                np.zeros_like(imaging_angles),
                -np.sin(imaging_angles),
                -np.cos(imaging_angles),
            )
        )
    else:
        along_track_angles = np.deg2rad(
            (
                np.arange(along_track_beams_per_side, dtype=np.float64)
                - 0.5 * (along_track_beams_per_side - 1)
            )
            * along_track_beam_spacing_deg
        )

        def side_fans(cross_track_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            repeated_cross_track = np.tile(
                cross_track_angles, along_track_beams_per_side
            )
            repeated_along_track = np.repeat(
                along_track_angles, imaging_rays_per_side
            )
            along_cosine = np.cos(repeated_along_track)
            directions = np.column_stack(
                (
                    np.sin(repeated_along_track),
                    -np.sin(repeated_cross_track) * along_cosine,
                    -np.cos(repeated_cross_track) * along_cosine,
                )
            )
            return repeated_cross_track, directions

        port_imaging_angles, port_directions = side_fans(port_angles)
        starboard_imaging_angles, starboard_directions = side_fans(
            starboard_angles
        )
        imaging_angles = np.concatenate(
            (port_imaging_angles, starboard_imaging_angles)
        )
        imaging_directions = np.concatenate(
            (port_directions, starboard_directions), axis=0
        )

    shared_directions = np.ascontiguousarray(
        imaging_directions, dtype=np.float64
    )
    imaging_indices = np.arange(
        imaging_directions.shape[0], dtype=np.int64
    )
    bathymetry_indices = _stratified_bathymetry_indices(
        imaging_rays_per_side,
        along_track_beams_per_side,
        num_beams,
    )
    return (
        shared_directions,
        bathymetry_indices,
        imaging_directions.astype(np.float64),
        imaging_indices,
        imaging_angles.astype(np.float64),
    )


def parallel_bathymetric_sidescan_rays(
    num_beams: int,
    swath_width_deg: float,
    imaging_rays_per_side: int,
    along_track_rows_per_side: int,
    along_track_sample_spacing_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build simultaneous image rows with parallel, displaced ray origins.

    Every row repeats the same cross-track steering directions but starts at a
    different local-X phase centre.  The bathymetry output is a balanced subset
    of these same raw returns, never an additional centre-origin fan.  Row
    centres are cell-centred within
    ``rows * spacing`` metres, so eight 0.15 m rows have centres from -0.525 m
    through +0.525 m and coverage boundaries at -0.6 m and +0.6 m.

    This spatial construction deliberately does not use horizontal beam width
    as angular steering between row centres.  Consequently the row-centre
    spacing stays constant at every slant range.

    Returns:
        ``(shared_origins_sensor, shared_directions_sensor,
        bathymetry_indices, imaging_origins_sensor,
        imaging_directions_sensor, imaging_indices, imaging_angles)``.
    """

    num_beams = _positive_int("num_beams", num_beams)
    swath_width_deg = _finite_float("swath_width_deg", swath_width_deg)
    if not 0.0 < swath_width_deg < 180.0:
        raise ValueError("swath_width_deg must be in the open interval (0, 180)")
    imaging_rays_per_side = _positive_int(
        "imaging_rays_per_side", imaging_rays_per_side
    )
    along_track_rows_per_side = _positive_int(
        "along_track_rows_per_side", along_track_rows_per_side
    )
    spacing_m = _finite_float(
        "along_track_sample_spacing_m", along_track_sample_spacing_m
    )
    if spacing_m <= 0.0:
        raise ValueError("along_track_sample_spacing_m must be positive")

    half_swath = 0.5 * np.deg2rad(float(swath_width_deg))
    positive = half_swath * (
        np.arange(imaging_rays_per_side, dtype=np.float64) + 0.5
    ) / float(imaging_rays_per_side)
    port_angles = -positive[::-1]
    starboard_angles = positive

    def repeated_side_fan(
        cross_track_angles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        repeated_angles = np.tile(
            cross_track_angles, along_track_rows_per_side
        )
        directions = np.column_stack(
            (
                np.zeros_like(repeated_angles),
                -np.sin(repeated_angles),
                -np.cos(repeated_angles),
            )
        )
        return repeated_angles, directions

    port_imaging_angles, port_directions = repeated_side_fan(port_angles)
    starboard_imaging_angles, starboard_directions = repeated_side_fan(
        starboard_angles
    )
    imaging_angles = np.concatenate(
        (port_imaging_angles, starboard_imaging_angles)
    )
    imaging_directions = np.concatenate(
        (port_directions, starboard_directions), axis=0
    ).astype(np.float64)

    coverage_m = along_track_rows_per_side * spacing_m
    row_centres_x = (
        -0.5 * coverage_m
        + (np.arange(along_track_rows_per_side, dtype=np.float64) + 0.5)
        * spacing_m
    )
    side_origins = np.zeros(
        (along_track_rows_per_side * imaging_rays_per_side, 3),
        dtype=np.float64,
    )
    side_origins[:, 0] = np.repeat(
        row_centres_x, imaging_rays_per_side
    )
    imaging_origins = np.concatenate(
        (side_origins, side_origins.copy()), axis=0
    )

    shared_origins = np.ascontiguousarray(imaging_origins, dtype=np.float64)
    shared_directions = np.ascontiguousarray(
        imaging_directions, dtype=np.float64
    )
    imaging_indices = np.arange(
        imaging_directions.shape[0], dtype=np.int64
    )
    bathymetry_indices = _stratified_bathymetry_indices(
        imaging_rays_per_side,
        along_track_rows_per_side,
        num_beams,
    )
    return (
        shared_origins,
        shared_directions,
        bathymetry_indices,
        imaging_origins,
        imaging_directions,
        imaging_indices,
        imaging_angles.astype(np.float64),
    )


def sidescan_directions(
    rays_per_side: int,
    tilt_deg: float,
    vertical_beam_width_deg: float,
    along_track_beam_width_deg: float = 0.0,
    along_track_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create port and starboard ray fans for a simplified side-scan sonar.

    ``tilt_deg`` is the depression of the beam centre below horizontal.  The
    vertical fan samples depression angles while the optional along-track fan
    samples the narrow horizontal aperture.
    """

    rays_per_side = _positive_int("rays_per_side", rays_per_side)
    along_track_samples = _positive_int("along_track_samples", along_track_samples)
    tilt_deg = _finite_float("tilt_deg", tilt_deg)
    vertical_beam_width_deg = _finite_float(
        "vertical_beam_width_deg", vertical_beam_width_deg
    )
    along_track_beam_width_deg = _finite_float(
        "along_track_beam_width_deg", along_track_beam_width_deg
    )

    if not 0.0 < vertical_beam_width_deg < 180.0:
        raise ValueError("vertical_beam_width_deg must be in (0, 180)")
    if not 0.0 <= along_track_beam_width_deg < 180.0:
        raise ValueError("along_track_beam_width_deg must be in [0, 180)")

    min_depression = tilt_deg - 0.5 * vertical_beam_width_deg
    max_depression = tilt_deg + 0.5 * vertical_beam_width_deg
    if min_depression <= 0.0 or max_depression >= 90.0:
        raise ValueError(
            "side-scan depression fan must remain between 0 and 90 degrees; "
            f"got [{min_depression}, {max_depression}]"
        )

    depression = np.deg2rad(
        np.linspace(min_depression, max_depression, rays_per_side)
    )
    along_track = np.deg2rad(
        np.linspace(
            -0.5 * along_track_beam_width_deg,
            0.5 * along_track_beam_width_deg,
            along_track_samples,
        )
    )
    depression_grid, along_track_grid = np.meshgrid(
        depression, along_track, indexing="ij"
    )

    horizontal = np.cos(depression_grid)
    z = -np.sin(depression_grid)

    def make_side(centre_azimuth: float) -> np.ndarray:
        azimuth = centre_azimuth + along_track_grid
        vectors = np.stack(
            (
                horizontal * np.cos(azimuth),
                horizontal * np.sin(azimuth),
                z,
            ),
            axis=-1,
        )
        return vectors.reshape(-1, 3).astype(np.float64)

    return make_side(np.pi / 2.0), make_side(-np.pi / 2.0)


@dataclass
class RaycastBatch:
    """Closest-hit results returned by a batched raycast backend."""

    hit_mask: np.ndarray
    distances: np.ndarray
    positions_world: np.ndarray
    normals_world: np.ndarray
    reflectivities: np.ndarray
    prim_paths: tuple[str, ...] = ()
    # Accelerated backends may fuse ray intersection and acoustic power
    # evaluation. Geometry-only backends leave this value unset.
    raw_intensities: np.ndarray | None = None
    # Optional explicit launch origins retain phase-centre lineage for sensors
    # whose simultaneous rows do not share one physical/virtual origin.
    ray_origins_world: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.hit_mask = np.asarray(self.hit_mask, dtype=bool).reshape(-1)
        count = self.hit_mask.size
        self.distances = np.asarray(self.distances, dtype=np.float64).reshape(count)
        self.positions_world = np.asarray(
            self.positions_world, dtype=np.float64
        ).reshape(count, 3)
        self.normals_world = np.asarray(
            self.normals_world, dtype=np.float64
        ).reshape(count, 3)
        self.reflectivities = np.asarray(
            self.reflectivities, dtype=np.float64
        ).reshape(count)
        if self.prim_paths and len(self.prim_paths) != count:
            raise ValueError("prim_paths length must match the number of rays")
        if self.raw_intensities is not None:
            self.raw_intensities = np.asarray(
                self.raw_intensities, dtype=np.float64
            ).reshape(count)
        if self.ray_origins_world is not None:
            self.ray_origins_world = broadcast_ray_origins(
                self.ray_origins_world, count
            )

    def sliced(self, start: int, stop: int) -> "RaycastBatch":
        """Return a contiguous view-copy used to split a combined ray launch."""

        start = int(start)
        stop = int(stop)
        if start < 0 or stop < start or stop > self.hit_mask.size:
            raise IndexError("invalid RaycastBatch slice")
        return RaycastBatch(
            hit_mask=self.hit_mask[start:stop],
            distances=self.distances[start:stop],
            positions_world=self.positions_world[start:stop],
            normals_world=self.normals_world[start:stop],
            reflectivities=self.reflectivities[start:stop],
            prim_paths=(self.prim_paths[start:stop] if self.prim_paths else ()),
            raw_intensities=(
                None
                if self.raw_intensities is None
                else self.raw_intensities[start:stop]
            ),
            ray_origins_world=(
                None
                if self.ray_origins_world is None
                else self.ray_origins_world[start:stop]
            ),
        )


@dataclass
class MultibeamPing:
    """One processed multibeam echosounder ping."""

    timestamp: float
    frame_id: str
    sound_speed: float
    frequency: float
    beam_angles: np.ndarray
    beam_directions: np.ndarray
    ranges: np.ndarray
    two_way_travel_times: np.ndarray
    intensities: np.ndarray
    hit_mask: np.ndarray
    points_sensor: np.ndarray
    points_world: np.ndarray
    prim_paths: tuple[str, ...] = ()

    def valid_points(self, frame: str = "sensor") -> np.ndarray:
        """Return valid XYZ points in the sensor or world frame."""

        if frame == "sensor":
            points = self.points_sensor
        elif frame == "world":
            points = self.points_world
        else:
            raise ValueError("frame must be 'sensor' or 'world'")
        return np.ascontiguousarray(points[self.hit_mask], dtype=np.float32)

    def as_sonar_ranges_dict(self) -> dict:
        """Return fields matching ``marine_acoustic_msgs/SonarRanges`` semantics."""

        return {
            "timestamp": float(self.timestamp),
            "frame_id": self.frame_id,
            "ping_info": {
                "frequency": float(self.frequency),
                "sound_speed": float(self.sound_speed),
                "tx_beamwidths": [],
                "rx_beamwidths": [],
            },
            "flags": np.where(self.hit_mask, 0, 1).astype(np.uint8),
            "transmit_delays": np.zeros(self.ranges.size, dtype=np.float32),
            "intensities": np.asarray(self.intensities, dtype=np.float32),
            "beam_unit_vec": np.asarray(self.beam_directions, dtype=np.float32),
            "ranges": np.asarray(self.ranges, dtype=np.float32),
        }

    def as_sonar_detections_dict(self) -> dict:
        """Return temporal fields matching ``marine_acoustic_msgs/SonarDetections``."""

        return {
            "timestamp": float(self.timestamp),
            "frame_id": self.frame_id,
            "ping_info": {
                "frequency": float(self.frequency),
                "sound_speed": float(self.sound_speed),
                "tx_beamwidths": [],
                "rx_beamwidths": [],
            },
            "flags": np.where(self.hit_mask, 0, 1).astype(np.uint8),
            "two_way_travel_times": np.asarray(
                self.two_way_travel_times, dtype=np.float32
            ),
            "tx_delays": np.zeros(self.ranges.size, dtype=np.float32),
            "intensities": np.asarray(self.intensities, dtype=np.float32),
            "tx_angles": np.zeros(self.ranges.size, dtype=np.float32),
            "rx_angles": np.asarray(self.beam_angles, dtype=np.float32),
        }


@dataclass
class SideScanPing:
    """One port/starboard side-scan line plus its display-ready representation."""

    timestamp: float
    frame_id: str
    sound_speed: float
    frequency: float
    range_bin_centres: np.ndarray
    port: np.ndarray
    starboard: np.ndarray
    display_line: np.ndarray
    port_points_world: np.ndarray
    starboard_points_world: np.ndarray
    port_beams: np.ndarray | None = None
    starboard_beams: np.ndarray | None = None

    def _raw_beam_data(self) -> np.ndarray:
        """Return validated beam-major port then starboard image samples."""

        range_count = int(np.asarray(self.range_bin_centres).size)
        if range_count < 1:
            raise ValueError("range_bin_centres must contain at least one sample")
        port = np.asarray(self.port, dtype=np.float32).reshape(-1)
        starboard = np.asarray(self.starboard, dtype=np.float32).reshape(-1)
        if port.size != range_count or starboard.size != range_count:
            raise ValueError(
                "port and starboard lines must match range_bin_centres"
            )

        if (self.port_beams is None) != (self.starboard_beams is None):
            raise ValueError(
                "port_beams and starboard_beams must be supplied together"
            )
        if self.port_beams is None:
            return np.ascontiguousarray(np.stack((port, starboard)))

        port_beams = np.asarray(self.port_beams, dtype=np.float32)
        starboard_beams = np.asarray(self.starboard_beams, dtype=np.float32)
        if port_beams.ndim != 2 or starboard_beams.ndim != 2:
            raise ValueError("port_beams and starboard_beams must be 2-D")
        if port_beams.shape != starboard_beams.shape:
            raise ValueError(
                "port_beams and starboard_beams must have the same shape"
            )
        if port_beams.shape[0] < 1 or port_beams.shape[1] != range_count:
            raise ValueError(
                "beam matrices must have shape (beams, range_bin_centres)"
            )
        return np.ascontiguousarray(
            np.concatenate((port_beams, starboard_beams), axis=0)
        )

    def as_raw_sonar_image_dict(self) -> dict:
        """Return fields matching ``marine_acoustic_msgs/RawSonarImage`` semantics."""

        range_bin_centres = np.asarray(
            self.range_bin_centres, dtype=np.float64
        ).reshape(-1)
        beam_data = self._raw_beam_data()
        beam_count = int(beam_data.shape[0])
        beams_per_side = beam_count // 2
        if range_bin_centres.size > 1:
            range_step = float(
                np.mean(np.diff(range_bin_centres))
            )
        else:
            range_step = 1.0
        if not np.isfinite(range_step) or range_step <= 0.0:
            raise ValueError("range_bin_centres must be strictly increasing")
        sample_rate = float(self.sound_speed / (2.0 * range_step))
        sample0 = int(round(float(range_bin_centres[0]) / range_step - 0.5))
        return {
            "timestamp": float(self.timestamp),
            "frame_id": self.frame_id,
            "ping_info": {
                "frequency": float(self.frequency),
                "sound_speed": float(self.sound_speed),
                "tx_beamwidths": [],
                "rx_beamwidths": [],
            },
            "sample_rate": sample_rate,
            "samples_per_beam": int(range_bin_centres.size),
            "sample0": max(0, sample0),
            "tx_delays": np.zeros(beam_count, dtype=np.float32),
            "tx_angles": np.zeros(beam_count, dtype=np.float32),
            # Beam-major data is [all port beams, all starboard beams]; +Y
            # (port) is positive.
            "rx_angles": np.concatenate(
                (
                    np.full(beams_per_side, np.pi / 2.0, dtype=np.float32),
                    np.full(
                        beams_per_side, -np.pi / 2.0, dtype=np.float32
                    ),
                )
            ),
            "image": {
                "is_bigendian": False,
                "dtype": 8,  # SonarImageData.DTYPE_FLOAT32
                "beam_count": beam_count,
                "data": beam_data.tobytes(),
            },
        }


@dataclass
class RegisteredSideScanReturns:
    """Point-wise geometry and backscatter from one raw acoustic return grid."""

    timestamp: float
    frame_id: str
    origin_world: np.ndarray
    beam_angles: np.ndarray
    beam_directions: np.ndarray
    ranges: np.ndarray
    intensities: np.ndarray
    hit_mask: np.ndarray
    points_sensor: np.ndarray
    points_world: np.ndarray
    source_ray_indices: np.ndarray
    ray_origins_sensor: np.ndarray | None = None
    ray_origins_world: np.ndarray | None = None
    along_track_row_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        if not np.isfinite(self.timestamp):
            raise ValueError("imaging timestamp must be finite")
        self.frame_id = str(self.frame_id)
        self.origin_world = np.asarray(
            self.origin_world, dtype=np.float64
        ).reshape(3)
        if not np.isfinite(self.origin_world).all():
            raise ValueError("imaging origin must be finite")
        self.beam_angles = np.asarray(
            self.beam_angles, dtype=np.float64
        ).reshape(-1)
        count = self.beam_angles.size
        self.beam_directions = np.asarray(
            self.beam_directions, dtype=np.float64
        ).reshape(count, 3)
        self.ranges = np.asarray(self.ranges, dtype=np.float64).reshape(count)
        self.intensities = np.asarray(
            self.intensities, dtype=np.float64
        ).reshape(count)
        self.hit_mask = np.asarray(self.hit_mask, dtype=bool).reshape(count)
        self.points_sensor = np.asarray(
            self.points_sensor, dtype=np.float64
        ).reshape(count, 3)
        self.points_world = np.asarray(
            self.points_world, dtype=np.float64
        ).reshape(count, 3)
        self.source_ray_indices = np.asarray(
            self.source_ray_indices, dtype=np.int64
        ).reshape(count)
        if np.any(self.source_ray_indices < 0):
            raise ValueError("imaging source ray indices must be non-negative")
        if self.ray_origins_sensor is None:
            self.ray_origins_sensor = np.zeros((count, 3), dtype=np.float64)
        else:
            self.ray_origins_sensor = broadcast_ray_origins(
                self.ray_origins_sensor, count
            )
        if self.ray_origins_world is None:
            self.ray_origins_world = broadcast_ray_origins(
                self.origin_world, count
            )
        else:
            self.ray_origins_world = broadcast_ray_origins(
                self.ray_origins_world, count
            )
        if self.along_track_row_indices is None:
            self.along_track_row_indices = np.zeros(count, dtype=np.int64)
        else:
            self.along_track_row_indices = np.asarray(
                self.along_track_row_indices, dtype=np.int64
            ).reshape(count)
        if np.any(self.along_track_row_indices < 0):
            raise ValueError("along-track row indices must be non-negative")

    def valid_samples(
        self, frame: str = "sensor"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(XYZ, backscatter, shared_ray_index)`` for valid returns."""

        if frame == "sensor":
            points = self.points_sensor
        elif frame == "world":
            points = self.points_world
        else:
            raise ValueError("frame must be 'sensor' or 'world'")
        valid = (
            self.hit_mask
            & np.isfinite(points).all(axis=1)
            & np.isfinite(self.intensities)
        )
        return (
            np.ascontiguousarray(points[valid], dtype=np.float32),
            np.ascontiguousarray(self.intensities[valid], dtype=np.float32),
            np.ascontiguousarray(
                self.source_ray_indices[valid], dtype=np.int64
            ),
        )

    def valid_row_indices(self) -> np.ndarray:
        """Return the simultaneous along-track row ID for each valid sample."""

        valid = (
            self.hit_mask
            & np.isfinite(self.points_sensor).all(axis=1)
            & np.isfinite(self.intensities)
        )
        return np.ascontiguousarray(
            self.along_track_row_indices[valid], dtype=np.int64
        )


@dataclass
class BathymetricSideScanPing:
    """One physically shared bathymetry/backscatter acquisition.

    When ``products_share_raw_returns`` is true, ``imaging`` is the unified
    point-wise echo grid and ``bathymetry`` must be an exact subset of it:
    direction, noisy range, hit flag, XYZ, and backscatter are all checked.
    The legacy ``*_source_beam_indices`` fields index the published bathymetry
    view and use ``-1`` when an image cell's raw return was not selected for
    that compact output. ``*_source_imaging_indices`` retain complete raw
    return lineage.
    """

    bathymetry: MultibeamPing
    side_scan: SideScanPing
    port_source_beam_indices: np.ndarray
    starboard_source_beam_indices: np.ndarray
    origin_world: np.ndarray | None = None
    shared_ray_directions: np.ndarray | None = None
    bathymetry_source_ray_indices: np.ndarray | None = None
    imaging: RegisteredSideScanReturns | None = None
    port_source_imaging_indices: np.ndarray | None = None
    starboard_source_imaging_indices: np.ndarray | None = None
    shared_ray_origins_sensor: np.ndarray | None = None
    shared_ray_origins_world: np.ndarray | None = None
    products_share_raw_returns: bool = False
    along_track_coverage_m: float = 0.0

    def __post_init__(self) -> None:
        if not np.isclose(
            self.bathymetry.timestamp,
            self.side_scan.timestamp,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "bathymetry and side-scan products must share one timestamp"
            )
        if self.bathymetry.frame_id != self.side_scan.frame_id:
            raise ValueError(
                "bathymetry and side-scan products must share one frame"
            )

        if self.origin_world is not None:
            self.origin_world = np.asarray(
                self.origin_world, dtype=np.float64
            ).reshape(3)
            if not np.isfinite(self.origin_world).all():
                raise ValueError("shared ping origin must be finite")

        if self.shared_ray_directions is None:
            self.shared_ray_directions = np.asarray(
                self.bathymetry.beam_directions, dtype=np.float64
            ).copy()
        else:
            self.shared_ray_directions = np.asarray(
                self.shared_ray_directions, dtype=np.float64
            ).reshape(-1, 3)
        shared_count = int(self.shared_ray_directions.shape[0])
        if self.shared_ray_origins_sensor is None:
            self.shared_ray_origins_sensor = np.zeros(
                (shared_count, 3), dtype=np.float64
            )
        else:
            self.shared_ray_origins_sensor = broadcast_ray_origins(
                self.shared_ray_origins_sensor, shared_count
            )
        if self.shared_ray_origins_world is not None:
            self.shared_ray_origins_world = broadcast_ray_origins(
                self.shared_ray_origins_world, shared_count
            )

        beam_count = int(self.bathymetry.ranges.size)
        if self.bathymetry_source_ray_indices is None:
            self.bathymetry_source_ray_indices = np.arange(
                beam_count, dtype=np.int64
            )
        else:
            self.bathymetry_source_ray_indices = np.asarray(
                self.bathymetry_source_ray_indices, dtype=np.int64
            ).reshape(-1)
        if self.bathymetry_source_ray_indices.size != beam_count:
            raise ValueError(
                "bathymetry source-ray indices must match bathymetric beams"
            )
        if np.any(
            (self.bathymetry_source_ray_indices < 0)
            | (self.bathymetry_source_ray_indices >= shared_count)
        ):
            raise ValueError("bathymetry source ray index is out of range")

        if self.imaging is not None:
            if not np.isclose(
                self.bathymetry.timestamp,
                self.imaging.timestamp,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError(
                    "bathymetry and dense imaging must share one timestamp"
                )
            if self.bathymetry.frame_id != self.imaging.frame_id:
                raise ValueError(
                    "bathymetry and dense imaging must share one frame"
                )
            if self.origin_world is None:
                self.origin_world = self.imaging.origin_world.copy()
            elif not np.allclose(
                self.origin_world,
                self.imaging.origin_world,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError(
                    "bathymetry and dense imaging must share one origin"
                )
            if np.any(self.imaging.source_ray_indices >= shared_count):
                raise ValueError("imaging source ray index is out of range")

        if self.shared_ray_origins_world is None and self.origin_world is not None:
            self.shared_ray_origins_world = broadcast_ray_origins(
                self.origin_world, shared_count
            )
        if self.imaging is not None:
            imaging_source = self.imaging.source_ray_indices
            if not np.allclose(
                self.imaging.ray_origins_sensor,
                self.shared_ray_origins_sensor[imaging_source],
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "imaging origins must match shared-ray sensor origins"
                )
            if (
                self.shared_ray_origins_world is not None
                and not np.allclose(
                    self.imaging.ray_origins_world,
                    self.shared_ray_origins_world[imaging_source],
                    rtol=0.0,
                    atol=1.0e-9,
                )
            ):
                raise ValueError(
                    "imaging origins must match shared-ray world origins"
                )

        self.products_share_raw_returns = bool(
            self.products_share_raw_returns
        )
        self.along_track_coverage_m = float(self.along_track_coverage_m)
        if (
            not np.isfinite(self.along_track_coverage_m)
            or self.along_track_coverage_m < 0.0
        ):
            raise ValueError(
                "along-track coverage must be finite and non-negative"
            )
        if self.products_share_raw_returns:
            if self.imaging is None:
                raise ValueError(
                    "raw-return sharing requires point-wise acoustic returns"
                )
            imaging_lookup = {
                int(source): local
                for local, source in enumerate(self.imaging.source_ray_indices)
            }
            try:
                imaging_local = np.asarray(
                    [
                        imaging_lookup[int(source)]
                        for source in self.bathymetry_source_ray_indices
                    ],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise ValueError(
                    "every bathymetry sample must come from the raw return grid"
                ) from exc
            comparisons = (
                (
                    self.bathymetry.beam_directions,
                    self.imaging.beam_directions[imaging_local],
                    "directions",
                ),
                (
                    self.bathymetry.ranges,
                    self.imaging.ranges[imaging_local],
                    "ranges",
                ),
                (
                    self.bathymetry.intensities,
                    self.imaging.intensities[imaging_local],
                    "intensities",
                ),
                (
                    self.bathymetry.points_sensor,
                    self.imaging.points_sensor[imaging_local],
                    "sensor points",
                ),
                (
                    self.bathymetry.points_world,
                    self.imaging.points_world[imaging_local],
                    "world points",
                ),
            )
            for bathymetry_values, raw_values, label in comparisons:
                if not np.allclose(
                    bathymetry_values,
                    raw_values,
                    rtol=0.0,
                    atol=1.0e-12,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"bathymetry {label} must be exact raw-return views"
                    )
            if not np.array_equal(
                self.bathymetry.hit_mask,
                self.imaging.hit_mask[imaging_local],
            ):
                raise ValueError(
                    "bathymetry hit flags must be exact raw-return views"
                )

        bin_count = int(self.side_scan.range_bin_centres.size)
        self.port_source_beam_indices = np.asarray(
            self.port_source_beam_indices, dtype=np.int64
        ).reshape(-1)
        self.starboard_source_beam_indices = np.asarray(
            self.starboard_source_beam_indices, dtype=np.int64
        ).reshape(-1)
        if (
            self.port_source_beam_indices.size != bin_count
            or self.starboard_source_beam_indices.size != bin_count
        ):
            raise ValueError(
                "source-index arrays must match the side-scan range bins"
            )
        for source in (
            self.port_source_beam_indices,
            self.starboard_source_beam_indices,
        ):
            if np.any((source < -1) | (source >= beam_count)):
                raise ValueError("side-scan source beam index is out of range")

        if self.port_source_imaging_indices is None:
            self.port_source_imaging_indices = (
                self.port_source_beam_indices.copy()
            )
        else:
            self.port_source_imaging_indices = np.asarray(
                self.port_source_imaging_indices, dtype=np.int64
            ).reshape(-1)
        if self.starboard_source_imaging_indices is None:
            self.starboard_source_imaging_indices = (
                self.starboard_source_beam_indices.copy()
            )
        else:
            self.starboard_source_imaging_indices = np.asarray(
                self.starboard_source_imaging_indices, dtype=np.int64
            ).reshape(-1)
        if (
            self.port_source_imaging_indices.size != bin_count
            or self.starboard_source_imaging_indices.size != bin_count
        ):
            raise ValueError(
                "imaging source-index arrays must match the side-scan bins"
            )
        imaging_count = (
            int(self.imaging.ranges.size)
            if self.imaging is not None
            else beam_count
        )
        for source in (
            self.port_source_imaging_indices,
            self.starboard_source_imaging_indices,
        ):
            if np.any((source < -1) | (source >= imaging_count)):
                raise ValueError(
                    "side-scan source imaging index is out of range"
                )

    @property
    def timestamp(self) -> float:
        return float(self.bathymetry.timestamp)

    @property
    def frame_id(self) -> str:
        return self.bathymetry.frame_id

    @property
    def shared_ray_count(self) -> int:
        return int(self.shared_ray_directions.shape[0])

    @property
    def shared_ray_directions_sensor(self) -> np.ndarray:
        """The de-duplicated launch directions expressed in the sensor frame."""

        return self.shared_ray_directions

    def valid_registered_samples(
        self, frame: str = "sensor"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return co-registered ``(XYZ, backscatter, beam_index)`` samples."""

        if frame == "sensor":
            points = self.bathymetry.points_sensor
        elif frame == "world":
            points = self.bathymetry.points_world
        else:
            raise ValueError("frame must be 'sensor' or 'world'")
        valid = (
            self.bathymetry.hit_mask
            & np.isfinite(points).all(axis=1)
            & np.isfinite(self.bathymetry.intensities)
        )
        indices = np.flatnonzero(valid).astype(np.int64)
        return (
            np.ascontiguousarray(points[valid], dtype=np.float32),
            np.ascontiguousarray(
                self.bathymetry.intensities[valid], dtype=np.float32
            ),
            indices,
        )

    def valid_imaging_samples(
        self, frame: str = "sensor"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return dense ``(XYZ, backscatter, shared_ray_index)`` samples.

        Pings produced by older code do not contain a separate imaging return
        set; for those pings this method safely falls back to the original
        bathymetric registrations.
        """

        if self.imaging is None:
            points, intensities, beam_indices = self.valid_registered_samples(
                frame=frame
            )
            source = self.bathymetry_source_ray_indices[beam_indices]
            return points, intensities, np.ascontiguousarray(source)
        return self.imaging.valid_samples(frame=frame)

    def valid_acoustic_samples(
        self, frame: str = "sensor"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the unified per-echo ``XYZ, intensity, source`` samples.

        This is the preferred estimator input for an integrated bathymetric
        side-scan.  ``valid_imaging_samples`` remains as a compatibility alias.
        """

        return self.valid_imaging_samples(frame=frame)

    def valid_acoustic_row_indices(self) -> np.ndarray:
        """Return synchronous along-track row IDs for valid acoustic samples."""

        if self.imaging is None:
            return np.zeros(
                self.valid_registered_samples()[0].shape[0], dtype=np.int64
            )
        return self.imaging.valid_row_indices()


def compute_backscatter_intensity(
    distances: Sequence[float],
    directions_world: np.ndarray,
    normals_world: np.ndarray,
    reflectivities: Sequence[float],
    hit_mask: Sequence[bool],
    attenuation: float = 0.01,
) -> np.ndarray:
    """Compute a first-order monostatic acoustic return intensity.

    The model combines Lambertian incidence, two-way exponential attenuation,
    and spherical spreading.  It is intentionally a fast geometric model; it
    does not model beamforming, refraction through a sound-speed profile, or
    multipath propagation.
    """

    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    count = distances.size
    directions_world = np.asarray(directions_world, dtype=np.float64).reshape(
        count, 3
    )
    normals_world = np.asarray(normals_world, dtype=np.float64).reshape(count, 3)
    reflectivities = np.asarray(reflectivities, dtype=np.float64).reshape(count)
    hit_mask = np.asarray(hit_mask, dtype=bool).reshape(count)
    attenuation = _finite_float("attenuation", attenuation)
    if attenuation < 0.0:
        raise ValueError("attenuation must be >= 0")

    incidence = np.clip(
        np.einsum("ij,ij->i", -directions_world, normals_world), 0.0, 1.0
    )
    safe_range = np.maximum(np.where(hit_mask, distances, 1.0), 1e-3)
    spreading = 1.0 / np.square(safe_range)
    absorption = np.exp(-2.0 * attenuation * safe_range)
    intensity = (
        np.maximum(reflectivities, 0.0) * incidence * spreading * absorption
    )
    intensity[~hit_mask] = 0.0
    intensity[~np.isfinite(intensity)] = 0.0
    return intensity


def normalize_intensities(
    intensities: Sequence[float],
    valid_mask: Sequence[bool] | None = None,
    percentile: float = 99.0,
) -> np.ndarray:
    """Robustly normalize arbitrary acoustic intensities to ``[0, 1]``."""

    values = np.asarray(intensities, dtype=np.float64)
    if valid_mask is None:
        valid = np.isfinite(values) & (values > 0.0)
    else:
        valid = (
            np.asarray(valid_mask, dtype=bool).reshape(values.shape)
            & np.isfinite(values)
            & (values > 0.0)
        )
    if not np.any(valid):
        return np.zeros_like(values, dtype=np.float64)
    scale = float(np.percentile(values[valid], percentile))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip(values / scale, 0.0, 1.0)


def perturb_ranges(
    ranges: Sequence[float],
    hit_mask: Sequence[bool],
    rng: np.random.Generator,
    gaussian_std: float = 0.0,
    positive_bias_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Add range error while preserving NaNs for missed beams.

    ``positive_bias_scale`` implements the outward-only exponential range
    diffusion used by published HoloOcean sonar models.  The returned second
    array contains that positive bias so callers can attenuate its intensity.
    """

    measured = np.asarray(ranges, dtype=np.float64).copy()
    hit_mask = np.asarray(hit_mask, dtype=bool).reshape(measured.shape)
    gaussian_std = _finite_float("gaussian_std", gaussian_std)
    positive_bias_scale = _finite_float(
        "positive_bias_scale", positive_bias_scale
    )
    if gaussian_std < 0.0 or positive_bias_scale < 0.0:
        raise ValueError("range noise parameters must be >= 0")

    measured[~hit_mask] = np.nan
    if gaussian_std > 0.0:
        measured[hit_mask] += rng.normal(0.0, gaussian_std, np.count_nonzero(hit_mask))
    positive_bias = np.zeros_like(measured)
    if positive_bias_scale > 0.0:
        positive_bias[hit_mask] = rng.exponential(
            positive_bias_scale, np.count_nonzero(hit_mask)
        )
        measured[hit_mask] += positive_bias[hit_mask]
    return measured, positive_bias


def bin_sidescan_returns(
    ranges: Sequence[float],
    intensities: Sequence[float],
    hit_mask: Sequence[bool],
    min_range: float,
    max_range: float,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse side-looking ray returns into slant-range bins using max power."""

    min_range, max_range = validate_range_limits(min_range, max_range)
    num_bins = _positive_int("num_bins", num_bins)
    ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(ranges.shape)
    hit_mask = np.asarray(hit_mask, dtype=bool).reshape(ranges.shape)

    edges = np.linspace(min_range, max_range, num_bins + 1, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    output = np.zeros(num_bins, dtype=np.float64)

    valid = (
        hit_mask
        & np.isfinite(ranges)
        & np.isfinite(intensities)
        & (ranges >= min_range)
        & (ranges <= max_range)
        & (intensities > 0.0)
    )
    if np.any(valid):
        indices = np.floor(
            (ranges[valid] - min_range) / (max_range - min_range) * num_bins
        ).astype(np.int64)
        indices = np.clip(indices, 0, num_bins - 1)
        np.maximum.at(output, indices, intensities[valid])
    return centres, output


def bin_coregistered_sidescan_returns(
    ranges: Sequence[float],
    intensities: Sequence[float],
    hit_mask: Sequence[bool],
    beam_directions: np.ndarray,
    min_range: float,
    max_range: float,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Form port/starboard imagery from the *same* bathymetric detections.

    This is the numerical core of an integrated bathymetric side-scan sonar.
    No second side-looking ray fan is created.  Instead, each valid multibeam
    bottom detection contributes its measured backscatter to a slant-range
    cell on the corresponding side.  Along with both intensity lines, the
    function returns the bathymetric source-beam index for every range cell.
    """

    min_range, max_range = validate_range_limits(min_range, max_range)
    num_bins = _positive_int("num_bins", num_bins)
    ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
    hit_mask = np.asarray(hit_mask, dtype=bool).reshape(-1)
    directions = np.asarray(beam_directions, dtype=np.float64).reshape(-1, 3)
    count = ranges.size
    if (
        intensities.size != count
        or hit_mask.size != count
        or directions.shape[0] != count
    ):
        raise ValueError(
            "ranges, intensities, hit_mask, and beam_directions must agree"
        )

    edges = np.linspace(min_range, max_range, num_bins + 1, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    port = np.zeros(num_bins, dtype=np.float64)
    starboard = np.zeros(num_bins, dtype=np.float64)
    port_source = np.full(num_bins, -1, dtype=np.int64)
    starboard_source = np.full(num_bins, -1, dtype=np.int64)
    base_valid = (
        hit_mask
        & np.isfinite(ranges)
        & np.isfinite(intensities)
        & np.isfinite(directions).all(axis=1)
        & (ranges >= min_range)
        & (ranges <= max_range)
        & (intensities > 0.0)
    )

    def accumulate(side_mask, output, source) -> None:
        beam_indices = np.flatnonzero(base_valid & side_mask)
        if beam_indices.size == 0:
            return
        bin_indices = np.floor(
            (ranges[beam_indices] - min_range)
            / (max_range - min_range)
            * num_bins
        ).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        # A range cell represents the strongest return in that cell.  Keeping
        # its beam index proves which bathymetric sample generated the pixel.
        for beam_index, bin_index in zip(beam_indices, bin_indices):
            value = intensities[beam_index]
            if value > output[bin_index]:
                output[bin_index] = value
                source[bin_index] = beam_index

    # +Y is port and -Y is starboard.  An exactly nadir-looking odd centre
    # beam is assigned to port so that every valid bathymetric return has a
    # deterministic image product; operational configurations use even beam
    # counts and therefore have no duplicated nadir sample.
    accumulate(directions[:, 1] >= 0.0, port, port_source)
    accumulate(directions[:, 1] < 0.0, starboard, starboard_source)
    return centres, port, starboard, port_source, starboard_source


def imaging_lineage_to_bathymetry(
    imaging_lineage: Sequence[int],
    imaging_source_ray_indices: Sequence[int],
    bathymetry_source_ray_indices: Sequence[int],
) -> np.ndarray:
    """Translate dense image lineage to exact coincident sparse beams.

    Dense-only rays intentionally map to ``-1``; no nearest-angle assignment
    is made because that would falsely claim physical co-registration.
    """

    lineage = np.asarray(imaging_lineage, dtype=np.int64).reshape(-1)
    imaging_sources = np.asarray(
        imaging_source_ray_indices, dtype=np.int64
    ).reshape(-1)
    bathymetry_sources = np.asarray(
        bathymetry_source_ray_indices, dtype=np.int64
    ).reshape(-1)
    if np.any((lineage < -1) | (lineage >= imaging_sources.size)):
        raise ValueError("imaging lineage index is out of range")
    if np.any(imaging_sources < 0) or np.any(bathymetry_sources < 0):
        raise ValueError("shared source ray indices must be non-negative")

    output = np.full(lineage.shape, -1, dtype=np.int64)
    valid_cells = np.flatnonzero(lineage >= 0)
    if valid_cells.size == 0 or bathymetry_sources.size == 0:
        return output
    maximum_source = int(
        max(np.max(imaging_sources), np.max(bathymetry_sources))
    )
    bathymetry_by_ray = np.full(maximum_source + 1, -1, dtype=np.int64)
    bathymetry_by_ray[bathymetry_sources] = np.arange(
        bathymetry_sources.size, dtype=np.int64
    )
    output[valid_cells] = bathymetry_by_ray[
        imaging_sources[lineage[valid_cells]]
    ]
    return output


def add_sonar_noise(
    signal: Sequence[float],
    range_bin_centres: Sequence[float],
    max_range: float,
    rng: np.random.Generator,
    multiplicative_std: float = 0.0,
    additive_std: float = 0.0,
) -> np.ndarray:
    """Apply multiplicative normal and range-scaled additive Rayleigh noise."""

    signal = np.asarray(signal, dtype=np.float64).copy()
    ranges = np.asarray(range_bin_centres, dtype=np.float64).reshape(signal.shape)
    max_range = _finite_float("max_range", max_range)
    multiplicative_std = _finite_float("multiplicative_std", multiplicative_std)
    additive_std = _finite_float("additive_std", additive_std)
    if max_range <= 0.0:
        raise ValueError("max_range must be > 0")
    if multiplicative_std < 0.0 or additive_std < 0.0:
        raise ValueError("noise standard deviations must be >= 0")

    if multiplicative_std > 0.0:
        multiplier = 1.0 + rng.normal(0.0, multiplicative_std, signal.shape)
        signal *= np.maximum(multiplier, 0.0)
    if additive_std > 0.0:
        range_scale = np.square(np.clip(ranges / max_range, 0.0, 1.0))
        signal += range_scale * rng.rayleigh(additive_std, signal.shape)
    return np.clip(signal, 0.0, 1.0)


def compose_sidescan_line(port: Sequence[float], starboard: Sequence[float]) -> np.ndarray:
    """Compose far-port -> nadir -> far-starboard side-scan display order."""

    port = np.asarray(port, dtype=np.float64).reshape(-1)
    starboard = np.asarray(starboard, dtype=np.float64).reshape(-1)
    if port.shape != starboard.shape:
        raise ValueError("port and starboard arrays must have the same shape")
    return np.concatenate((port[::-1], starboard))
