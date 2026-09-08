"""Isaac Sim UI and scene helpers for acoustic sensor visualization."""

from __future__ import annotations

import re

import numpy as np


class AcousticImageViewport:
    """Small Kit window backed by a CPU RGBA NumPy image."""

    def __init__(
        self,
        title: str,
        *,
        width: int,
        height: int,
        footer: str = "",
    ) -> None:
        import omni.ui as ui

        self._ui = ui
        self._width = int(width)
        self._height = int(height)
        self._provider = ui.ByteImageProvider()
        self._window = ui.Window(
            title,
            width=self._width + 20,
            height=self._height + 92,
            visible=True,
        )
        with self._window.frame:
            with ui.VStack(spacing=4):
                self._status = ui.Label(
                    "Waiting for the first ping...",
                    height=24,
                    style={"color": 0xFFD8F4FF, "font_size": 16},
                )
                with ui.ZStack(height=self._height):
                    ui.Rectangle(style={"background_color": 0xFF180C03})
                    self._image = ui.ImageWithProvider(
                        self._provider,
                        width=self._width,
                        height=self._height,
                        style={
                            "fill_policy": ui.FillPolicy.PRESERVE_ASPECT_FIT,
                            "alignment": ui.Alignment.CENTER,
                        },
                    )
                self._footer = ui.Label(
                    footer,
                    height=20,
                    style={"color": 0xFF8FB8C8, "font_size": 13},
                )

    def update(self, rgba: np.ndarray, status: str) -> None:
        image = np.ascontiguousarray(rgba, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError("viewport image must be an HxWx4 RGBA array")
        self._provider.set_data_array(
            image.reshape(-1), [int(image.shape[1]), int(image.shape[0])]
        )
        self._status.text = str(status)

    def close(self) -> None:
        for element in (
            self._image,
            self._provider,
            self._status,
            self._footer,
            self._window,
        ):
            try:
                element.destroy()
            except Exception:
                pass


class AcousticSceneVisualizer:
    """Draw sensor rays and hit points as isolated USD guide geometry.

    USD curves are used instead of clearing the global debug-draw interface,
    so enabling an acoustic sensor cannot erase DVL or user debug drawings.
    """

    def __init__(
        self,
        sensor_prim_path: str,
        *,
        line_color=(0.08, 0.65, 0.82),
        point_color=(1.0, 0.57, 0.12),
        line_width: float = 0.012,
        point_width: float = 0.07,
        opacity: float = 0.62,
        max_rays: int = 96,
    ) -> None:
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, Vt

        self._Gf = Gf
        self._Sdf = Sdf
        self._UsdGeom = UsdGeom
        self._stage = omni.usd.get_context().get_stage()
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", sensor_prim_path.strip("/"))
        self._root_path = f"/World/OceanSimVisuals/{safe_name}"
        UsdGeom.Scope.Define(self._stage, "/World/OceanSimVisuals")
        UsdGeom.Scope.Define(self._stage, self._root_path)
        self._curves = UsdGeom.BasisCurves.Define(
            self._stage, f"{self._root_path}/rays"
        )
        self._points = UsdGeom.Points.Define(
            self._stage, f"{self._root_path}/returns"
        )
        self._curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        self._curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        self._curves.CreateDisplayColorAttr(
            [Gf.Vec3f(*(float(value) for value in line_color))]
        )
        self._curves.CreateDisplayOpacityAttr([float(opacity)])
        self._curves.CreateWidthsAttr(
            Vt.FloatArray([float(line_width)])
        )
        self._curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        self._points.CreateDisplayColorAttr(
            [Gf.Vec3f(*(float(value) for value in point_color))]
        )
        self._line_width = float(line_width)
        self._point_width = float(point_width)
        self._max_rays = max(1, int(max_rays))

    def update(
        self, origin_world: np.ndarray, hit_points_world: np.ndarray
    ) -> None:
        points = np.asarray(hit_points_world, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] > self._max_rays:
            indices = np.linspace(
                0, points.shape[0] - 1, self._max_rays, dtype=np.int64
            )
            points = points[indices]

        origin = np.asarray(origin_world, dtype=np.float64).reshape(3)
        gf_points = [
            self._Gf.Vec3f(float(value[0]), float(value[1]), float(value[2]))
            for value in points
        ]
        curve_points = []
        origin_gf = self._Gf.Vec3f(
            float(origin[0]), float(origin[1]), float(origin[2])
        )
        for endpoint in gf_points:
            curve_points.extend((origin_gf, endpoint))

        count = len(gf_points)
        self._curves.CreatePointsAttr().Set(curve_points)
        self._curves.CreateCurveVertexCountsAttr().Set([2] * count)
        self._points.CreatePointsAttr().Set(gf_points)
        self._points.CreateWidthsAttr().Set(
            [self._point_width] * count
        )

    def clear(self) -> None:
        self._curves.CreatePointsAttr().Set([])
        self._curves.CreateCurveVertexCountsAttr().Set([])
        self._points.CreatePointsAttr().Set([])

    def close(self) -> None:
        if (
            self._stage is not None
            and self._stage.GetPrimAtPath(self._root_path)
        ):
            self._stage.RemovePrim(self._Sdf.Path(self._root_path))
        self._stage = None
