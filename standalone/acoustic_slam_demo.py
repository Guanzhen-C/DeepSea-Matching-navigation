#!/usr/bin/env python3
"""Generate a presentation-ready OceanSim acoustic SLAM dashboard.

This preview intentionally runs without Isaac Sim. It exercises the same
single-ping bathymetry/backscatter geometry, binning, noise, and rasterizers as
the live integrated sonar against a deterministic analytic seabed.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sonar = _load_module(
    "oceansim_demo_acoustic_sonar",
    "isaacsim/oceansim/utils/acoustic_sonar.py",
)
visualization = _load_module(
    "oceansim_demo_acoustic_visualization",
    "isaacsim/oceansim/utils/acoustic_visualization.py",
)


def seabed_height(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    undulation = 0.35 * np.sin(0.22 * x) * np.cos(0.30 * y)
    ridge = 1.65 * np.exp(-((x - 3.0) ** 2 / 24.0 + (y + 4.0) ** 2 / 5.0))
    rock = 1.15 * np.exp(-((x + 7.0) ** 2 / 3.0 + (y - 5.0) ** 2 / 2.2))
    trench = -0.85 * np.exp(-((x - 9.0) ** 2 / 16.0 + (y - 3.0) ** 2 / 4.0))
    return -9.2 + undulation + ridge + rock + trench


def seabed_reflectivity(x, y):
    wreck = np.exp(-((x - 4.0) ** 2 / 2.2 + (y + 5.0) ** 2 / 0.7))
    rock = np.exp(-((x + 7.0) ** 2 / 2.0 + (y - 5.0) ** 2 / 1.5))
    ripples = 0.18 * (1.0 + np.sin(1.7 * y + 0.25 * x))
    return 0.72 + ripples + 2.8 * wreck + 1.8 * rock


def intersect_seabed(origin, directions, max_range):
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    downward = directions[:, 2] < -1e-6
    initial_bottom = seabed_height(origin[0], origin[1])
    distance = np.full(directions.shape[0], np.nan, dtype=np.float64)
    distance[downward] = (initial_bottom - origin[2]) / directions[downward, 2]

    for _ in range(10):
        active = downward & np.isfinite(distance)
        points = origin + directions[active] * distance[active, None]
        bottom = seabed_height(points[:, 0], points[:, 1])
        distance[active] = (bottom - origin[2]) / directions[active, 2]

    hit = (
        downward
        & np.isfinite(distance)
        & (distance > 0.3)
        & (distance <= max_range)
    )
    points = np.full((directions.shape[0], 3), np.nan, dtype=np.float64)
    points[hit] = origin + directions[hit] * distance[hit, None]

    epsilon = 0.04
    normals = np.zeros_like(points)
    if np.any(hit):
        px = points[hit, 0]
        py = points[hit, 1]
        dz_dx = (
            seabed_height(px + epsilon, py) - seabed_height(px - epsilon, py)
        ) / (2.0 * epsilon)
        dz_dy = (
            seabed_height(px, py + epsilon) - seabed_height(px, py - epsilon)
        ) / (2.0 * epsilon)
        normal = np.column_stack((-dz_dx, -dz_dy, np.ones_like(dz_dx)))
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        normals[hit] = normal

    reflectivity = np.zeros(directions.shape[0], dtype=np.float64)
    reflectivity[hit] = seabed_reflectivity(points[hit, 0], points[hit, 1])
    return hit, distance, points, normals, reflectivity


def vehicle_pose(frame_index: int, frame_count: int):
    progress = frame_index / max(1, frame_count - 1)
    x = -18.0 + 36.0 * progress
    y = 0.75 * np.sin(progress * 2.0 * np.pi)
    z = -2.1 + 0.12 * np.sin(progress * 4.0 * np.pi)
    return np.array([x, y, z], dtype=np.float64)


def make_multibeam_ping(origin, frame_index, rng):
    max_range = 24.0
    angles, directions = sonar.multibeam_directions(192, 120.0)
    hit, distances, points, normals, reflectivity = intersect_seabed(
        origin, directions, max_range
    )
    measured, _ = sonar.perturb_ranges(
        distances, hit, rng, gaussian_std=0.018
    )
    hit &= np.isfinite(measured) & (measured <= max_range)
    measured[~hit] = np.nan
    raw = sonar.compute_backscatter_intensity(
        distances,
        directions,
        normals,
        reflectivity,
        hit,
        attenuation=0.008,
    )
    intensities = sonar.normalize_intensities(raw, hit)
    points_sensor = directions * measured[:, None]
    points_sensor[~hit] = np.nan
    points[~hit] = np.nan
    return sonar.MultibeamPing(
        timestamp=frame_index / 10.0,
        frame_id="demo_integrated_sonar",
        sound_speed=1500.0,
        frequency=400_000.0,
        beam_angles=angles,
        beam_directions=directions,
        ranges=measured,
        two_way_travel_times=2.0 * measured / 1500.0,
        intensities=intensities,
        hit_mask=hit,
        points_sensor=points_sensor,
        points_world=points,
    )


def make_sidescan_ping(multibeam_ping, rng, range_bins=320):
    """Derive the image product from the exact bathymetric detections."""

    max_range = 24.0
    centres, port, starboard, _, _ = (
        sonar.bin_coregistered_sidescan_returns(
            multibeam_ping.ranges,
            multibeam_ping.intensities,
            multibeam_ping.hit_mask,
            multibeam_ping.beam_directions,
            min_range=0.5,
            max_range=max_range,
            num_bins=range_bins,
        )
    )
    port = sonar.add_sonar_noise(
        port,
        centres,
        max_range,
        rng,
        multiplicative_std=0.05,
        additive_std=0.014,
    )
    starboard = sonar.add_sonar_noise(
        starboard,
        centres,
        max_range,
        rng,
        multiplicative_std=0.05,
        additive_std=0.014,
    )
    display_line = np.rint(
        sonar.compose_sidescan_line(port, starboard) * 255.0
    ).astype(np.uint8)
    return sonar.SideScanPing(
        timestamp=multibeam_ping.timestamp,
        frame_id=multibeam_ping.frame_id,
        sound_speed=multibeam_ping.sound_speed,
        frequency=multibeam_ping.frequency,
        range_bin_centres=centres,
        port=port,
        starboard=starboard,
        display_line=display_line,
        port_points_world=multibeam_ping.points_world[
            multibeam_ping.hit_mask
            & (multibeam_ping.beam_directions[:, 1] >= 0.0)
        ],
        starboard_points_world=multibeam_ping.points_world[
            multibeam_ping.hit_mask
            & (multibeam_ping.beam_directions[:, 1] < 0.0)
        ],
    )


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _panel(canvas, draw, xy, title, image):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=12,
        fill=(5, 17, 29, 255),
        outline=(31, 87, 105, 255),
        width=2,
    )
    draw.text(
        (x0 + 16, y0 + 10),
        title,
        font=_font(18, True),
        fill=(202, 239, 247),
    )
    target_width = x1 - x0 - 20
    target_height = y1 - y0 - 48
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    panel = Image.fromarray(image, mode="RGBA").resize(
        (target_width, target_height), resampling
    )
    canvas.alpha_composite(panel, (x0 + 10, y0 + 38))


def render_dashboard(
    ping, waterfall, map_points, track, frame_index, frame_count
):
    canvas = Image.new("RGBA", (1440, 810), (2, 9, 18, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1440, 68), fill=(4, 21, 36, 255))
    draw.text(
        (34, 17),
        "OCEANSIM  /  SONAR SLAM",
        font=_font(27, True),
        fill=(101, 226, 246),
    )
    draw.text(
        (1120, 20),
        f"PING {frame_index + 1:03d} / {frame_count:03d}",
        font=_font(18, True),
        fill=(255, 181, 70),
    )

    swath = visualization.render_multibeam_swath(
        ping, max_range=24.0, width=650, height=290
    )
    bathymetry = visualization.render_bathymetry_map(
        map_points, track, width=650, height=290
    )
    sidescan = visualization.colorize_sidescan(waterfall)
    _panel(
        canvas,
        draw,
        (30, 86, 710, 424),
        "INTEGRATED SONAR / BATHYMETRY PRODUCT",
        swath,
    )
    _panel(
        canvas,
        draw,
        (30, 442, 710, 782),
        "BATHYMETRIC MAP / SLAM INPUT",
        bathymetry,
    )
    _panel(
        canvas,
        draw,
        (730, 86, 1410, 782),
        "CO-REGISTERED BACKSCATTER / PORT — STARBOARD",
        sidescan,
    )

    valid_ranges = ping.ranges[ping.hit_mask]
    status = (
        f"NAV  x={track[-1, 0]:6.2f} m  y={track[-1, 1]:5.2f} m   "
        f"SHARED RETURNS {np.count_nonzero(ping.hit_mask):3d}/"
        f"{ping.hit_mask.size:3d} hits   "
        f"altitude={np.min(valid_ranges):4.2f} m"
    )
    draw.rounded_rectangle(
        (450, 18, 1095, 54),
        radius=8,
        fill=(8, 38, 52, 255),
        outline=(21, 105, 127, 255),
    )
    draw.text((466, 27), status, font=_font(14), fill=(194, 229, 233))
    return canvas


def generate_demo(
    output_dir: Path, frame_count: int, fps: int, make_gif: bool
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260902)
    waterfall = np.zeros((min(220, frame_count), 640), dtype=np.uint8)
    map_points = []
    track = []
    gif_frames = []
    dashboard = None

    for frame_index in range(frame_count):
        origin = vehicle_pose(frame_index, frame_count)
        ping = make_multibeam_ping(origin, frame_index, rng)
        side_ping = make_sidescan_ping(ping, rng)
        waterfall[:-1] = waterfall[1:]
        waterfall[-1] = side_ping.display_line
        map_points.append(ping.valid_points(frame="world"))
        track.append(origin)
        stacked_points = np.concatenate(map_points, axis=0)
        stacked_track = np.asarray(track)
        valid_lines = min(frame_index + 1, waterfall.shape[0])
        valid_waterfall = waterfall[-valid_lines:]
        dashboard = render_dashboard(
            ping,
            valid_waterfall,
            stacked_points,
            stacked_track,
            frame_index,
            frame_count,
        )
        if make_gif:
            palette_namespace = getattr(Image, "Palette", None)
            adaptive_palette = getattr(
                palette_namespace, "ADAPTIVE", Image.ADAPTIVE
            )
            preview = dashboard.convert("RGB").resize((960, 540)).convert(
                "P", palette=adaptive_palette, colors=128
            )
            gif_frames.append(preview)

    png_path = output_dir / "acoustic_slam_dashboard.png"
    dashboard.convert("RGB").save(png_path, quality=95)
    gif_path = output_dir / "acoustic_slam_dashboard.gif"
    if make_gif:
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=max(20, int(round(1000 / fps))),
            loop=0,
            optimize=False,
            disposal=2,
        )
    return png_path, gif_path if make_gif else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "demo" / "acoustic_slam_preview",
    )
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be >= 2")
    if args.fps < 1:
        parser.error("--fps must be >= 1")

    png_path, gif_path = generate_demo(
        args.output_dir,
        frame_count=args.frames,
        fps=args.fps,
        make_gif=not args.no_gif,
    )
    print(f"PNG: {png_path}")
    if gif_path is not None:
        print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()
