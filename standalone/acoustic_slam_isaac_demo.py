#!/usr/bin/env python3
"""Run a lightweight, live MBES/side-scan demonstration in Isaac Sim.

The scene is generated procedurally, so this entry point remains usable when
the optional ``OceanSim_assets`` repository is not installed.  Close the
Isaac Sim window (or press Ctrl+C in the launching terminal) to stop it.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OceanSim live acoustic-SLAM sensor demonstration"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without the desktop window (intended for smoke tests)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after this many wall-clock seconds; zero runs until closed",
    )
    parser.add_argument(
        "--ping-hz",
        type=float,
        default=5.0,
        help="acoustic ping rate (default: 5 Hz)",
    )
    args, kit_args = parser.parse_known_args()
    if args.duration < 0.0:
        parser.error("--duration must be >= 0")
    if args.ping_hz <= 0.0:
        parser.error("--ping-hz must be > 0")
    sys.argv = [sys.argv[0], *kit_args]
    return args


ARGS = parse_args()
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# SimulationApp must be constructed before importing any Kit/Omniverse APIs.
from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RayTracedLighting",
        "width": 1280,
        "height": 720,
        "window_width": 1600,
        "window_height": 1000,
        "anti_aliasing": 0,
        "multi_gpu": False,
        "sync_loads": True,
    }
)

# The pip package owns the top-level ``isaacsim`` package.  Extend its search
# path so it can also discover this checkout's ``isaacsim.oceansim`` module.
import isaacsim


local_package = str(ROOT / "isaacsim")
if local_package not in isaacsim.__path__:
    isaacsim.__path__.append(local_package)

import carb
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

from isaacsim.oceansim.sensors.BathymetricSideScanSonarSensor import (
    BathymetricSideScanSonarSensor,
)


def _set_xform(
    prim,
    *,
    translation=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    rotation=(0.0, 0.0, 0.0),
):
    xformable = UsdGeom.Xformable(prim)
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*(float(value) for value in translation)))
    if rotation != (0.0, 0.0, 0.0):
        rotate_op = xformable.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(*(float(value) for value in rotation)))
    scale_op = xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3f(*(float(value) for value in scale)))
    return translate_op


def _make_cube(
    stage,
    path: str,
    *,
    position,
    scale,
    color,
    rotation=(0.0, 0.0, 0.0),
    collision=True,
    opacity=1.0,
):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    cube.CreateDisplayOpacityAttr([float(opacity)])
    _set_xform(
        cube.GetPrim(),
        translation=position,
        scale=scale,
        rotation=rotation,
    )
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _make_sphere(
    stage,
    path: str,
    *,
    position,
    radius,
    color,
    collision=True,
):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _set_xform(sphere.GetPrim(), translation=position)
    if collision:
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    return sphere


def _make_status_window():
    if ARGS.headless:
        return None, None
    import omni.ui as ui

    window = ui.Window(
        "OceanSim Acoustic SLAM | LIVE",
        width=430,
        height=150,
        visible=True,
    )
    window.position_x = 18
    window.position_y = 56
    with window.frame:
        with ui.VStack(spacing=5):
            ui.Label(
                "LIVE  |  PROCEDURAL UNDERWATER TEST RANGE",
                height=28,
                style={"color": 0xFF50E6FF, "font_size": 17},
            )
            status = ui.Label(
                "Initialising integrated bathymetric side-scan...",
                height=72,
                word_wrap=True,
                style={"color": 0xFFE3F7FA, "font_size": 15},
            )
            ui.Label(
                "Cyan: beams   Orange: returns   Close Isaac Sim to stop",
                height=22,
                style={"color": 0xFFA3C5CE, "font_size": 13},
            )
    return window, status


def _position_sensor_windows(sonar) -> None:
    """Arrange the live plots below the scene viewport when Kit permits it."""

    if ARGS.headless:
        return
    layouts = ((20, 585), (675, 585))
    for viewport, (x, y) in zip(
        (sonar._visual_viewport, sonar._sidescan_viewport), layouts
    ):
        try:
            window = viewport._window
            window.position_x = x
            window.position_y = y
        except (AttributeError, RuntimeError):
            # Some Kit window managers choose the floating-window position.
            pass


def build_scene(world: World):
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Scope.Define(stage, "/World/Environment")

    _make_cube(
        stage,
        "/World/Environment/Seabed",
        position=(0.0, 0.0, -8.25),
        scale=(52.0, 38.0, 0.5),
        color=(0.16, 0.27, 0.25),
    )

    # A deliberately asymmetric test range makes the MBES cross-section and
    # side-scan shadows easy to recognise while the vehicle moves.
    obstacles = (
        ("RidgePort", (-6.0, 5.0, -7.25), (6.0, 1.7, 1.7), (4.0, 0.0, 9.0)),
        ("RidgeStarboard", (5.5, -6.0, -7.45), (7.5, 1.2, 1.3), (-3.0, 0.0, -12.0)),
        ("WreckBody", (3.0, 4.0, -7.15), (4.8, 1.5, 1.9), (0.0, 8.0, 18.0)),
        ("WreckBow", (5.7, 4.9, -7.45), (1.3, 1.0, 1.3), (0.0, 18.0, 18.0)),
        ("Pipeline", (-1.0, -4.6, -7.55), (12.0, 0.6, 0.6), (0.0, 0.0, -5.0)),
    )
    for index, (name, position, scale, rotation) in enumerate(obstacles):
        color = (0.48, 0.34, 0.18) if index < 2 else (0.52, 0.24, 0.11)
        _make_cube(
            stage,
            f"/World/Environment/{name}",
            position=position,
            scale=scale,
            rotation=rotation,
            color=color,
        )

    for index, (x, y, radius) in enumerate(
        (
            (-10.0, -6.0, 0.8),
            (-5.0, 7.0, 1.0),
            (0.5, 7.5, 0.7),
            (8.5, 5.6, 1.2),
            (11.0, -3.0, 0.9),
        )
    ):
        _make_sphere(
            stage,
            f"/World/Environment/Rock_{index:02d}",
            position=(x, y, -8.0 + radius * 0.72),
            radius=radius,
            color=(0.28, 0.31, 0.25),
        )

    vehicle = UsdGeom.Xform.Define(stage, "/World/Vehicle")
    vehicle_translate = _set_xform(
        vehicle.GetPrim(), translation=(-12.0, 0.0, -2.2)
    )
    _make_cube(
        stage,
        "/World/Vehicle/Hull",
        position=(0.0, 0.0, 0.0),
        scale=(2.8, 1.15, 0.65),
        color=(0.86, 0.48, 0.06),
        collision=False,
    )
    _make_cube(
        stage,
        "/World/Vehicle/Fin",
        position=(-0.7, 0.0, 0.45),
        scale=(0.9, 0.12, 0.8),
        color=(0.96, 0.67, 0.10),
        collision=False,
    )

    distant = UsdLux.DistantLight.Define(stage, "/World/Sun")
    distant.CreateIntensityAttr(850.0)
    distant.CreateColorAttr(Gf.Vec3f(0.46, 0.70, 0.82))
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(35.0, -25.0, -30.0)
    )
    dome = UsdLux.DomeLight.Define(stage, "/World/UnderwaterAmbient")
    dome.CreateIntensityAttr(280.0)
    dome.CreateColorAttr(Gf.Vec3f(0.07, 0.28, 0.36))
    fill = UsdLux.SphereLight.Define(stage, "/World/Vehicle/WorkLight")
    fill.CreateIntensityAttr(9000.0)
    fill.CreateRadiusAttr(1.0)
    fill.CreateColorAttr(Gf.Vec3f(0.30, 0.72, 0.86))
    _set_xform(fill.GetPrim(), translation=(1.0, 0.0, -0.2))

    return stage, vehicle_translate


def main() -> int:
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    _, vehicle_translate = build_scene(world)

    sonar = BathymetricSideScanSonarSensor(
        prim_path="/World/Vehicle/bathymetric_sidescan",
        name="OceanSim Integrated Sonar",
        translation=np.array([0.0, 0.0, -0.52]),
        num_beams=128,
        swath_width_deg=120.0,
        range_bins=300,
        waterfall_lines=240,
        beam_footprint_bins=1,
        min_range=0.5,
        max_range=20.0,
        center_frequency_hz=400_000.0,
        range_noise_std=0.02,
        multiplicative_noise_std=0.035,
        additive_noise_std=0.008,
        attenuation=0.008,
        gain=1.35,
        random_seed=21,
        viewport=not ARGS.headless,
        draw_in_scene=True,
        viewport_resolution=(620, 330),
        sidescan_viewport_resolution=(620, 330),
    )

    status_window, status_label = _make_status_window()
    try:
        world.reset()
        sonar.initialize()
        _position_sensor_windows(sonar)
        set_camera_view(
            eye=np.array([14.0, 16.0, -0.4]),
            target=np.array([0.0, 0.0, -6.0]),
            camera_prim_path="/OmniverseKit_Persp",
        )

        for _ in range(12):
            world.step(render=True)
        world.play()

        started_at = time.monotonic()
        last_ping_at = -1.0
        ping_period = 1.0 / ARGS.ping_hz
        ping_count = 0
        print(
            "[OceanSim] Isaac Sim UI is ready; integrated sonar is running.",
            flush=True,
        )

        while simulation_app.is_running():
            elapsed = time.monotonic() - started_at
            if ARGS.duration and elapsed >= ARGS.duration:
                break

            if elapsed - last_ping_at >= ping_period:
                phase = elapsed * 0.18
                x = -12.0 + math.fmod(elapsed * 1.15, 24.0)
                y = 1.15 * math.sin(phase)
                z = -2.2 + 0.12 * math.sin(phase * 1.8)
                vehicle_translate.Set(Gf.Vec3d(x, y, z))

                sonar_ping = sonar.scan(timestamp=elapsed)
                ping_count += 1
                last_ping_at = elapsed

                if status_label is not None:
                    shared_hits = int(
                        np.count_nonzero(sonar_ping.bathymetry.hit_mask)
                    )
                    status_label.text = (
                        f"Ping {ping_count:05d}  |  {ARGS.ping_hz:.1f} Hz\n"
                        f"AUV  x={x:+6.2f}  y={y:+5.2f}  depth={-z:4.2f} m\n"
                        f"Shared bathymetry/backscatter returns "
                        f"{shared_hits:03d}/128"
                    )

            world.step(render=True)
    except KeyboardInterrupt:
        carb.log_info("[OceanSim] interrupted by user")
    finally:
        try:
            world.stop()
        except Exception:
            pass
        sonar.close()
        if status_window is not None:
            try:
                status_window.destroy()
            except Exception:
                pass
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
