"""Dependency-free rasterizers for OceanSim acoustic sensor displays.

The functions in this module only depend on NumPy.  Isaac Sim viewports and
the standalone presentation demo both use the same rendering path, which
keeps the on-screen picture testable without loading Kit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


_BACKGROUND = np.array([3, 12, 24, 255], dtype=np.uint8)
_GRID = (29, 62, 79, 255)
_RAY = (24, 131, 160, 150)
_MISS_RAY = (24, 63, 82, 100)
_BOTTOM = (255, 190, 66, 255)
_SENSOR = (92, 231, 255, 255)


def _canvas(width: int, height: int) -> np.ndarray:
    width = int(width)
    height = int(height)
    if width < 64 or height < 64:
        raise ValueError("visualization width and height must be >= 64")
    return np.broadcast_to(_BACKGROUND, (height, width, 4)).copy()


def _draw_line(
    image: np.ndarray,
    start: Sequence[float],
    end: Sequence[float],
    color: Sequence[int],
    thickness: int = 1,
) -> None:
    """Draw one clipped, alpha-blended line into an RGBA NumPy image."""

    x0, y0 = (float(start[0]), float(start[1]))
    x1, y1 = (float(end[0]), float(end[1]))
    samples = max(1, int(np.ceil(max(abs(x1 - x0), abs(y1 - y0))))) + 1
    x = np.rint(np.linspace(x0, x1, samples)).astype(np.int64)
    y = np.rint(np.linspace(y0, y1, samples)).astype(np.int64)
    radius = max(0, int(thickness) // 2)
    rgba = np.asarray(color, dtype=np.float64)
    alpha = float(rgba[3] / 255.0)

    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            xx = x + offset_x
            yy = y + offset_y
            valid = (
                (xx >= 0)
                & (xx < image.shape[1])
                & (yy >= 0)
                & (yy < image.shape[0])
            )
            if not np.any(valid):
                continue
            xx = xx[valid]
            yy = yy[valid]
            current = image[yy, xx, :3].astype(np.float64)
            image[yy, xx, :3] = np.rint(
                current * (1.0 - alpha) + rgba[:3] * alpha
            ).astype(np.uint8)
            image[yy, xx, 3] = 255


def _draw_disc(
    image: np.ndarray,
    centre: Sequence[float],
    radius: int,
    color: Sequence[int],
) -> None:
    cx, cy = int(round(float(centre[0]))), int(round(float(centre[1])))
    radius = max(1, int(radius))
    y0 = max(0, cy - radius)
    y1 = min(image.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(image.shape[1], cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    region = image[y0:y1, x0:x1]
    region[mask] = np.asarray(color, dtype=np.uint8)


def _return_color(intensity: float) -> tuple[int, int, int, int]:
    value = float(np.clip(intensity, 0.0, 1.0))
    low = np.array([26.0, 177.0, 205.0])
    high = np.array(_BOTTOM[:3], dtype=np.float64)
    rgb = np.rint(low * (1.0 - value) + high * value).astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2]), 255


def render_multibeam_swath(
    ping,
    *,
    max_range: float,
    width: int = 960,
    height: int = 540,
    max_drawn_beams: int = 96,
) -> np.ndarray:
    """Render one MBES ping as a port-to-starboard cross-section.

    The result is contiguous ``uint8`` RGBA.  Port is shown on the left,
    starboard on the right, and depth increases down the image.
    """

    image = _canvas(width, height)
    max_range = float(max_range)
    if not np.isfinite(max_range) or max_range <= 0.0:
        raise ValueError("max_range must be finite and > 0")
    max_drawn_beams = max(1, int(max_drawn_beams))

    margin_x = max(24, width // 24)
    top = max(22, height // 18)
    bottom = max(18, height // 24)
    scale = min(
        (width - 2.0 * margin_x) / (2.0 * max_range),
        (height - top - bottom) / max_range,
    )
    sensor_pixel = np.array([0.5 * width, float(top)])

    for fraction in (0.25, 0.5, 0.75, 1.0):
        depth_y = top + fraction * max_range * scale
        _draw_line(
            image,
            (margin_x, depth_y),
            (width - margin_x, depth_y),
            _GRID,
        )
        offset = fraction * max_range * scale
        _draw_line(
            image,
            (sensor_pixel[0] - offset, top),
            (sensor_pixel[0] - offset, height - bottom),
            _GRID,
        )
        _draw_line(
            image,
            (sensor_pixel[0] + offset, top),
            (sensor_pixel[0] + offset, height - bottom),
            _GRID,
        )

    directions = np.asarray(
        ping.beam_directions, dtype=np.float64
    ).reshape(-1, 3)
    ranges = np.asarray(ping.ranges, dtype=np.float64).reshape(-1)
    intensities = np.asarray(ping.intensities, dtype=np.float64).reshape(-1)
    hit_mask = np.asarray(ping.hit_mask, dtype=bool).reshape(-1)
    if not (
        directions.shape[0]
        == ranges.size
        == intensities.size
        == hit_mask.size
    ):
        raise ValueError("multibeam ping arrays must have matching lengths")

    draw_step = max(1, int(np.ceil(ranges.size / max_drawn_beams)))
    indices = np.arange(0, ranges.size, draw_step, dtype=np.int64)
    for index in indices:
        ray_range = ranges[index] if hit_mask[index] else max_range
        if not np.isfinite(ray_range):
            ray_range = max_range
        endpoint_sensor = directions[index] * min(float(ray_range), max_range)
        endpoint = np.array(
            [
                sensor_pixel[0] - endpoint_sensor[1] * scale,
                sensor_pixel[1] - endpoint_sensor[2] * scale,
            ]
        )
        _draw_line(
            image,
            sensor_pixel,
            endpoint,
            _RAY if hit_mask[index] else _MISS_RAY,
        )

    points_sensor = np.asarray(
        ping.points_sensor, dtype=np.float64
    ).reshape(-1, 3)
    pixels = np.column_stack(
        (
            sensor_pixel[0] - points_sensor[:, 1] * scale,
            sensor_pixel[1] - points_sensor[:, 2] * scale,
        )
    )
    previous_index = None
    finite_hits = hit_mask & np.isfinite(points_sensor).all(axis=1)
    for index in np.flatnonzero(finite_hits):
        if previous_index is not None and index == previous_index + 1:
            _draw_line(
                image,
                pixels[previous_index],
                pixels[index],
                _BOTTOM,
                2,
            )
        _draw_disc(
            image,
            pixels[index],
            radius=2,
            color=_return_color(intensities[index]),
        )
        previous_index = int(index)

    _draw_line(
        image,
        (sensor_pixel[0] - 10, sensor_pixel[1]),
        (sensor_pixel[0] + 10, sensor_pixel[1]),
        _SENSOR,
        3,
    )
    _draw_disc(image, sensor_pixel, radius=4, color=_SENSOR)
    return np.ascontiguousarray(image)


_SIDESCAN_PALETTES = {
    # Monotonic amber/copper power LUT used by many live side-scan operator
    # displays: acoustic shadow stays black, progressively stronger returns
    # move through dark copper and gold, and only saturation reaches cream.
    # Unlike a rainbow map, luminance never reverses or invents class edges.
    "amber": np.asarray(
        [
            [0, 0, 0],
            [25, 9, 1],
            [91, 36, 3],
            [213, 119, 13],
            [255, 241, 181],
        ],
        dtype=np.float32,
    ),
    "ocean": np.asarray(
        [
            [2, 8, 17],
            [0, 31, 50],
            [13, 101, 116],
            [226, 132, 37],
            [255, 246, 195],
        ],
        dtype=np.float32,
    ),
    # Processed side-scan products are conventionally inspected as an
    # achromatic backscatter image.  Keeping all three channels equal also
    # avoids suggesting that the palette represents sediment classes.
    "gray": np.asarray(
        [
            [0, 0, 0],
            [26, 26, 26],
            [91, 91, 91],
            [184, 184, 184],
            [255, 255, 255],
        ],
        dtype=np.float32,
    ),
}


def _interpolate_palette(
    values: np.ndarray, palette: str = "ocean"
) -> np.ndarray:
    stops = np.asarray([0.0, 0.18, 0.42, 0.70, 1.0], dtype=np.float32)
    try:
        colors = _SIDESCAN_PALETTES[str(palette)]
    except KeyError as error:
        choices = ", ".join(sorted(_SIDESCAN_PALETTES))
        raise ValueError(
            f"unknown side-scan palette {palette!r}; choose {choices}"
        ) from error
    flat = values.reshape(-1)
    output = np.empty((flat.size, 3), dtype=np.float32)
    for channel in range(3):
        output[:, channel] = np.interp(flat, stops, colors[:, channel])
    return np.rint(output).astype(np.uint8).reshape(values.shape + (3,))


def resample_sidescan_ground_range(
    ranges: np.ndarray,
    intensities: np.ndarray,
    hit_mask: np.ndarray,
    directions_sensor: np.ndarray,
    *,
    max_range: float,
    output_bins: int,
    attenuation: float = 0.0,
    spreading_power: float = 2.0,
    continuity_factor: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Reconstruct one side as a continuous, ground-range backscatter line.

    The geometric simulator evaluates the receive fan with a finite number of
    quadrature rays.  A real side-scan receiver instead samples a continuous
    echo envelope in time.  This function reconstructs that envelope between
    adjacent ray samples, performs slant-to-ground projection, and applies the
    propagation part of time-varying gain (TVG).  Large discontinuities are
    deliberately left unfilled so geometric acoustic shadows remain dark.

    This is a display product only.  The sparse measurements and their exact
    source lineage remain untouched and are still used by the estimator.

    Returns ``(power, coverage, altitude, maximum_ground_range)``.  ``power``
    is linear (not decibel) intensity and is intentionally not tone mapped.
    """

    ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
    hit_mask = np.asarray(hit_mask, dtype=bool).reshape(-1)
    directions = np.asarray(directions_sensor, dtype=np.float64).reshape(-1, 3)
    if not (
        ranges.size
        == intensities.size
        == hit_mask.size
        == directions.shape[0]
    ):
        raise ValueError("side-scan sample arrays must have matching lengths")
    max_range = float(max_range)
    output_bins = int(output_bins)
    attenuation = float(attenuation)
    spreading_power = float(spreading_power)
    continuity_factor = float(continuity_factor)
    if not np.isfinite(max_range) or max_range <= 0.0:
        raise ValueError("max_range must be finite and > 0")
    if output_bins < 2:
        raise ValueError("output_bins must be >= 2")
    if not np.isfinite(attenuation) or attenuation < 0.0:
        raise ValueError("attenuation must be finite and >= 0")
    if not np.isfinite(spreading_power) or spreading_power < 0.0:
        raise ValueError("spreading_power must be finite and >= 0")
    if not np.isfinite(continuity_factor) or continuity_factor <= 1.0:
        raise ValueError("continuity_factor must be finite and > 1")

    norms = np.linalg.norm(directions, axis=1)
    unit = np.zeros_like(directions)
    direction_valid = np.isfinite(norms) & (norms > 1.0e-12)
    unit[direction_valid] = directions[direction_valid] / norms[
        direction_valid, None
    ]
    down = -unit[:, 2]
    horizontal = np.hypot(unit[:, 0], unit[:, 1])
    angles = np.arctan2(horizontal, down)
    valid = (
        hit_mask
        & direction_valid
        & np.isfinite(ranges)
        & np.isfinite(intensities)
        & (ranges > 0.0)
        & (ranges <= max_range)
        & (down > 1.0e-6)
    )

    output = np.zeros(output_bins, dtype=np.float64)
    coverage = np.zeros(output_bins, dtype=bool)
    if not np.any(valid):
        return output, coverage, float("nan"), 0.0

    valid_indices = np.flatnonzero(valid)
    near_order = valid_indices[np.argsort(angles[valid_indices])]
    all_angles = np.sort(
        np.unique(angles[direction_valid & (down > 1.0e-6)])
    )
    angular_steps = np.diff(all_angles)
    angular_steps = angular_steps[angular_steps > 1.0e-12]
    nominal_angle_step = (
        float(np.median(angular_steps)) if angular_steps.size else np.inf
    )
    near_count = min(8, near_order.size)
    # The vertical component of the closest-to-nadir returns is a robust
    # bottom-track altitude estimate.  Median filtering avoids one isolated
    # rock or range-noise sample moving an entire corrected scan line.
    altitude = float(
        np.median(ranges[near_order[:near_count]] * down[near_order[:near_count]])
    )
    altitude = float(np.clip(altitude, 1.0e-3, max_range))
    range_limited_ground = float(
        np.sqrt(max(0.0, max_range * max_range - altitude * altitude))
    )
    if all_angles.size:
        half_cell = (
            0.5 * nominal_angle_step
            if np.isfinite(nominal_angle_step)
            else 0.0
        )
        fan_edge = min(
            float(all_angles[-1] + half_cell),
            0.5 * np.pi - 1.0e-6,
        )
        angle_limited_ground = altitude * np.tan(fan_edge)
        maximum_ground_range = float(
            min(range_limited_ground, angle_limited_ground)
        )
    else:
        fan_edge = 0.0
        maximum_ground_range = range_limited_ground
    if maximum_ground_range <= 1.0e-9:
        return output, coverage, altitude, maximum_ground_range

    order = near_order
    theta = angles[order]
    sample_ranges = ranges[order]
    ground_ranges = sample_ranges * horizontal[order]
    sample_power = np.maximum(intensities[order], 0.0)

    relative_range = np.maximum(sample_ranges / altitude, 1.0e-6)
    tvg = np.power(relative_range, spreading_power)
    absorption_exponent = np.clip(
        2.0 * attenuation * (sample_ranges - altitude), -20.0, 9.0
    )
    tvg *= np.exp(absorption_exponent)
    sample_power *= np.clip(tvg, 1.0e-4, 1.0e4)

    if order.size > 1:
        angle_gap = np.diff(theta)
        actual_ground_gap = np.diff(ground_ranges)
        actual_slant_gap = np.abs(np.diff(sample_ranges))
        flat_ground = altitude * np.tan(theta)
        flat_slant = altitude / np.maximum(np.cos(theta), 1.0e-6)
        expected_ground_gap = np.abs(np.diff(flat_ground))
        expected_slant_gap = np.abs(np.diff(flat_slant))
        continuity = (
            (angle_gap <= 1.75 * nominal_angle_step)
            & (actual_ground_gap >= -0.25)
            & (
                actual_ground_gap
                <= np.maximum(
                    2.5,
                    continuity_factor * expected_ground_gap + 0.5,
                )
            )
            & (
                actual_slant_gap
                <= np.maximum(
                    2.5,
                    continuity_factor * expected_slant_gap + 0.5,
                )
            )
        )
    else:
        continuity = np.empty(0, dtype=bool)

    ground_axis = np.linspace(
        0.0, maximum_ground_range, output_bins, dtype=np.float64
    )
    boundaries = np.flatnonzero(~continuity) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [order.size]))
    output_spacing = maximum_ground_range / float(output_bins - 1)

    for run_number, (start, stop) in enumerate(zip(starts, stops)):
        run_x = ground_ranges[start:stop]
        run_power = sample_power[start:stop]
        finite = np.isfinite(run_x) & np.isfinite(run_power)
        run_x = run_x[finite]
        run_power = run_power[finite]
        if run_x.size == 0:
            continue

        sort_x = np.argsort(run_x)
        run_x = run_x[sort_x]
        run_power = run_power[sort_x]
        unique_x, inverse = np.unique(run_x, return_inverse=True)
        unique_power = np.zeros(unique_x.size, dtype=np.float64)
        np.maximum.at(unique_power, inverse, run_power)
        run_x = unique_x
        run_power = unique_power

        if (
            run_number == 0
            and theta[start] <= 2.0 * nominal_angle_step
            and run_x[0] <= max(1.0, 3.0 * output_spacing)
        ):
            run_x = np.concatenate(([0.0], run_x))
            run_power = np.concatenate(([run_power[0]], run_power))

        if run_number == len(starts) - 1 and run_x.size >= 2:
            expected_last_step = abs(flat_slant[-1] - flat_slant[-2])
            reaches_range_edge = max_range - sample_ranges[-1] <= max(
                2.5, 2.5 * expected_last_step
            )
            reaches_fan_edge = (
                np.isfinite(nominal_angle_step)
                and fan_edge - theta[-1] <= 1.25 * nominal_angle_step
            )
            if reaches_range_edge or reaches_fan_edge:
                run_x = np.concatenate((run_x, [maximum_ground_range]))
                run_power = np.concatenate((run_power, [run_power[-1]]))

        if run_x.size == 1:
            index = int(np.argmin(np.abs(ground_axis - run_x[0])))
            output[index] = max(output[index], run_power[0])
            coverage[index] = True
            continue

        inside = (ground_axis >= run_x[0]) & (ground_axis <= run_x[-1])
        if not np.any(inside):
            continue
        interpolated = np.interp(ground_axis[inside], run_x, run_power)
        output[inside] = np.maximum(output[inside], interpolated)
        coverage[inside] = True

    return output, coverage, altitude, maximum_ground_range


def tone_map_sidescan_power(
    power: np.ndarray,
    coverage: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    reference_power: float | None = None,
    dynamic_range_db: float = 36.0,
    headroom_db: float = 7.0,
    speckle_looks: float = 5.0,
    noise_floor_db: float = -34.0,
    nadir_mute_bins: int = 0,
) -> np.ndarray:
    """Convert linear ground-range power into a realistic mono8 record.

    Gamma-distributed multiplicative speckle approximates a multi-look
    coherent sonar image.  A low receiver/reverberation floor prevents valid
    low-backscatter seabed from becoming artificial pure-black holes.  The
    final logarithmic mapping is equivalent to a finite-dynamic-range sonar
    display rather than an unconstrained per-pixel linear stretch.
    """

    power = np.asarray(power, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=bool)
    if power.shape != coverage.shape or power.ndim < 1:
        raise ValueError("power and coverage must have the same non-empty shape")
    dynamic_range_db = float(dynamic_range_db)
    headroom_db = float(headroom_db)
    speckle_looks = float(speckle_looks)
    noise_floor_db = float(noise_floor_db)
    nadir_mute_bins = int(nadir_mute_bins)
    if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0.0:
        raise ValueError("dynamic_range_db must be finite and > 0")
    if (
        not np.isfinite(headroom_db)
        or headroom_db < 0.0
        or headroom_db >= dynamic_range_db
    ):
        raise ValueError(
            "headroom_db must be finite, >= 0, and below dynamic_range_db"
        )
    if not np.isfinite(speckle_looks) or speckle_looks <= 0.0:
        raise ValueError("speckle_looks must be finite and > 0")
    if not np.isfinite(noise_floor_db) or noise_floor_db >= 0.0:
        raise ValueError("noise_floor_db must be finite and < 0")
    if nadir_mute_bins < 0:
        raise ValueError("nadir_mute_bins must be >= 0")

    positive = coverage & np.isfinite(power) & (power > 0.0)
    if reference_power is None:
        reference = (
            float(np.percentile(power[positive], 98.0))
            if np.any(positive)
            else 1.0
        )
    else:
        reference = float(reference_power)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_power must be finite and > 0")

    signal = np.maximum(np.nan_to_num(power, nan=0.0), 0.0).copy()
    if rng is None:
        speckle = np.ones(signal.shape, dtype=np.float64)
        receiver_noise = np.ones(signal.shape, dtype=np.float64)
    else:
        speckle = rng.gamma(
            shape=speckle_looks,
            scale=1.0 / speckle_looks,
            size=signal.shape,
        )
        if signal.shape[-1] >= 3:
            # Finite receive bandwidth correlates adjacent range samples.
            speckle = (
                0.18 * np.roll(speckle, 1, axis=-1)
                + 0.64 * speckle
                + 0.18 * np.roll(speckle, -1, axis=-1)
            )
        receiver_noise = rng.rayleigh(scale=1.0, size=signal.shape)
    signal *= speckle

    mute = min(nadir_mute_bins, signal.shape[-1])
    if mute > 0:
        phase = np.linspace(0.0, 1.0, mute, endpoint=False)
        smoothstep = phase * phase * (3.0 - 2.0 * phase)
        signal[..., :mute] *= smoothstep

    noise_power = reference * 10.0 ** (noise_floor_db / 10.0)
    noise_scale = np.where(coverage, 1.0, 0.45)
    displayed_power = signal + noise_power * receiver_noise * noise_scale
    floor = reference * 10.0 ** (-dynamic_range_db / 10.0)
    decibels = (
        10.0 * np.log10(np.maximum(displayed_power, floor) / reference)
        - headroom_db
    )
    normalized = np.clip(
        (decibels + dynamic_range_db) / dynamic_range_db, 0.0, 1.0
    )
    return np.rint(255.0 * normalized).astype(np.uint8)


def colorize_sidescan(
    waterfall: np.ndarray,
    *,
    gamma: float = 0.62,
    mark_nadir: bool = True,
    palette: str = "ocean",
) -> np.ndarray:
    """Convert a mono8 side-scan waterfall to presentation-ready RGBA."""

    mono = np.asarray(waterfall)
    if mono.ndim != 2:
        raise ValueError("waterfall must be a 2-D mono image")
    if mono.shape[0] < 1 or mono.shape[1] < 2:
        raise ValueError(
            "waterfall must contain at least one row and two columns"
        )
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and > 0")

    normalized = np.clip(mono.astype(np.float32) / 255.0, 0.0, 1.0) ** gamma
    rgb = _interpolate_palette(normalized, palette)
    rgba = np.empty(mono.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255

    if mark_nadir:
        centre = mono.shape[1] // 2
        left = max(0, centre - 1)
        right = min(mono.shape[1], centre + 1)
        rgba[:, left:right, :3] = np.asarray(
            [21, 171, 196], dtype=np.uint8
        )
    return np.ascontiguousarray(rgba)


def render_bathymetry_map(
    points_world: np.ndarray,
    track_world: np.ndarray,
    *,
    width: int = 960,
    height: int = 540,
    padding_fraction: float = 0.08,
) -> np.ndarray:
    """Render accumulated MBES points and the vehicle track from above."""

    image = _canvas(width, height)
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    track = np.asarray(track_world, dtype=np.float64).reshape(-1, 3)
    finite_points = points[np.isfinite(points).all(axis=1)]
    finite_track = track[np.isfinite(track).all(axis=1)]
    if finite_points.size == 0 and finite_track.size == 0:
        return image

    xy_parts = []
    if finite_points.size:
        xy_parts.append(finite_points[:, :2])
    if finite_track.size:
        xy_parts.append(finite_track[:, :2])
    xy = np.concatenate(xy_parts)
    lower = np.min(xy, axis=0)
    upper = np.max(xy, axis=0)
    span = np.maximum(upper - lower, 1.0)
    lower -= span * float(padding_fraction)
    upper += span * float(padding_fraction)
    span = upper - lower

    def project(values):
        px = (values[:, 0] - lower[0]) / span[0] * (width - 1)
        py = (1.0 - (values[:, 1] - lower[1]) / span[1]) * (height - 1)
        return np.column_stack((px, py))

    for fraction in (0.25, 0.5, 0.75):
        _draw_line(
            image,
            (fraction * width, 0),
            (fraction * width, height),
            _GRID,
        )
        _draw_line(
            image,
            (0, fraction * height),
            (width, fraction * height),
            _GRID,
        )

    if finite_points.size:
        projected = project(finite_points)
        depth = -finite_points[:, 2]
        depth_span = max(float(np.ptp(depth)), 1e-6)
        normalized = (depth - np.min(depth)) / depth_span
        colors = _interpolate_palette(1.0 - normalized)
        for pixel, color in zip(projected, colors):
            _draw_disc(
                image,
                pixel,
                radius=2,
                color=(int(color[0]), int(color[1]), int(color[2]), 255),
            )

    if finite_track.shape[0] >= 2:
        projected_track = project(finite_track)
        for start, end in zip(projected_track[:-1], projected_track[1:]):
            _draw_line(image, start, end, _SENSOR, 2)
        _draw_disc(image, projected_track[-1], radius=5, color=_SENSOR)
    return np.ascontiguousarray(image)
