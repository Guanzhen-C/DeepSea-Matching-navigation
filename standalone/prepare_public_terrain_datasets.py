#!/usr/bin/env python3
"""Prepare downloaded public acoustic maps for the production SLAM runner.

The source products remain untouched.  This program writes one ``prepared``
directory per dataset using the memory-mappable map contract consumed by
``MeasuredBathymetryGrid``.  NoData stays invalid, acoustic layers are only
registered where they were measured, and every generated route is checked
against the production 250 m sonar-clearance requirement.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import ctypes
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import sys
import warnings

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
import tifffile


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/underwater_terrain_datasets"
sys.path.insert(0, str(ROOT / "isaacsim"))

from oceansim.slam.real_bathymetry import (  # noqa: E402
    MeasuredBathymetryGrid,
    WaypointTerrainFollowingSurvey,
)


DATASET_ALIASES = {
    "chelan": "lake_chelan_usgs_2023_3m",
    "jenkinson": "jenkinson_lake_usgs_2023",
    "wallops": "wallops_assawoman_usgs_2024_1m",
    "sheboygan": "lake_michigan_sheboygan_noaa_2018",
}
REQUIRED_OUTPUTS = (
    "elevation_m.npy",
    "navigation_elevation_m.npy",
    "bathymetry_valid_mask.npy",
    "backscatter.npy",
    "backscatter_valid_mask.npy",
    "backscatter_source_mask.npy",
    "metadata.json",
    "full_map_traversal_route.json",
    "sha256sums.txt",
)


@dataclass(frozen=True)
class RasterInfo:
    path: Path
    shape: tuple[int, int]
    dtype: str
    world: tuple[float, float, float, float, float, float]
    nodata: float | None

    @property
    def dx(self) -> float:
        return float(self.world[0])

    @property
    def dy(self) -> float:
        return float(self.world[3])

    @property
    def x0(self) -> float:
        return float(self.world[4])

    @property
    def y0(self) -> float:
        return float(self.world[5])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        rows, columns = self.shape
        west = self.x0 - 0.5 * self.dx
        east = self.x0 + (columns - 0.5) * self.dx
        north = self.y0 - 0.5 * self.dy
        south = self.y0 + (rows - 0.5) * self.dy
        return float(west), float(east), float(south), float(north)


@dataclass(frozen=True)
class GridSpec:
    resolution_m: float
    rows: int
    columns: int
    x0: float
    y0: float

    @property
    def world(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.resolution_m,
            0.0,
            0.0,
            -self.resolution_m,
            self.x0,
            self.y0,
        )

    @property
    def scene_extent_m(self) -> list[float]:
        return [
            float((self.columns - 1) * self.resolution_m),
            float((self.rows - 1) * self.resolution_m),
        ]

    @property
    def bounds(self) -> dict[str, float]:
        half = 0.5 * self.resolution_m
        return {
            "west": float(self.x0 - half),
            "east": float(
                self.x0 + (self.columns - 1) * self.resolution_m + half
            ),
            "north": float(self.y0 + half),
            "south": float(
                self.y0 - (self.rows - 1) * self.resolution_m - half
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the downloaded Chelan, Jenkinson, Wallops, and "
            "Sheboygan acoustic maps to OceanSim's production map format"
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASET_ALIASES),
        default="all",
        help="dataset to prepare; default prepares every downloaded dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing complete prepared directory",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _world_file(path: Path) -> tuple[float, float, float, float, float, float]:
    candidate = path.with_suffix(".tfw")
    if candidate.is_file():
        values = tuple(float(value) for value in candidate.read_text().split())
        if len(values) != 6:
            raise RuntimeError(f"invalid world file: {candidate}")
        return values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with tifffile.TiffFile(path) as tiff:
            page = tiff.pages[0]
            scale_tag = page.tags.get("ModelPixelScaleTag")
            tie_tag = page.tags.get("ModelTiepointTag")
            if scale_tag is None or tie_tag is None:
                raise RuntimeError(f"GeoTIFF has no usable transform: {path}")
            scale = scale_tag.value
            tie = tie_tag.value
    dx = float(scale[0])
    dy = -float(scale[1])
    raster_i, raster_j = float(tie[0]), float(tie[1])
    model_x, model_y = float(tie[3]), float(tie[4])
    x_corner = model_x - raster_i * dx
    y_corner = model_y - raster_j * dy
    return (dx, 0.0, 0.0, dy, x_corner + 0.5 * dx, y_corner + 0.5 * dy)


def _raster_info(path: Path) -> RasterInfo:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with tifffile.TiffFile(path) as tiff:
            page = tiff.pages[0]
            if len(page.shape) != 2:
                raise RuntimeError(f"expected one-band raster: {path}")
            nodata_tag = page.tags.get("GDAL_NODATA")
            try:
                nodata = (
                    None if nodata_tag is None else float(nodata_tag.value)
                )
            except (TypeError, ValueError, OverflowError):
                nodata = None
            shape = tuple(int(value) for value in page.shape)
            dtype = str(page.dtype)
    world = _world_file(path)
    dx, rx, ry, dy, _, _ = world
    if max(abs(rx), abs(ry)) > 1.0e-8 or dx <= 0.0 or dy >= 0.0:
        raise RuntimeError(f"only north-up rasters are supported: {path}")
    return RasterInfo(path, shape, dtype, world, nodata)


def _read_raster(info: RasterInfo) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = tifffile.imread(info.path)
    except ValueError as error:
        if "imagecodecs" not in str(error):
            raise
        values = cv2.imread(str(info.path), cv2.IMREAD_UNCHANGED)
        if values is None:
            raise RuntimeError(
                f"OpenCV could not decode compressed GeoTIFF: {info.path}"
            ) from error
    if values.shape != info.shape:
        raise RuntimeError(
            f"raster shape changed while reading {info.path}: {values.shape}"
        )
    return np.asarray(values)


def _valid_mask(values: np.ndarray, nodata: float | None) -> np.ndarray:
    valid = np.isfinite(values)
    if nodata is not None and np.isfinite(nodata):
        valid &= values != nodata
    if np.issubdtype(values.dtype, np.floating):
        valid &= np.abs(values) < 1.0e20
    return valid


def _grid_from_rasters(
    rasters: list[RasterInfo], resolution_m: float
) -> GridSpec:
    west = min(info.bounds[0] for info in rasters)
    east = max(info.bounds[1] for info in rasters)
    south = min(info.bounds[2] for info in rasters)
    north = max(info.bounds[3] for info in rasters)
    columns = int(math.ceil((east - west) / resolution_m))
    rows = int(math.ceil((north - south) / resolution_m))
    return GridSpec(
        float(resolution_m),
        rows,
        columns,
        float(west + 0.5 * resolution_m),
        float(north - 0.5 * resolution_m),
    )


def _grid_from_point_bounds(
    minimum_x: float,
    maximum_x: float,
    minimum_y: float,
    maximum_y: float,
    resolution_m: float,
) -> GridSpec:
    west = float(minimum_x - 0.5)
    east = float(maximum_x + 0.5)
    south = float(minimum_y - 0.5)
    north = float(maximum_y + 0.5)
    columns = int(math.ceil((east - west) / resolution_m))
    rows = int(math.ceil((north - south) / resolution_m))
    return GridSpec(
        float(resolution_m),
        rows,
        columns,
        float(west + 0.5 * resolution_m),
        float(north - 0.5 * resolution_m),
    )


def _target_coordinates(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    x = grid.x0 + np.arange(grid.columns, dtype=np.float64) * grid.resolution_m
    y = grid.y0 - np.arange(grid.rows, dtype=np.float64) * grid.resolution_m
    return x, y


def _registered_region(
    source: np.ndarray,
    source_valid: np.ndarray,
    info: RasterInfo,
    target: GridSpec,
) -> tuple[slice, slice, np.ndarray, np.ndarray] | None:
    target_x, target_y = _target_coordinates(target)
    west, east, south, north = info.bounds
    target_columns = np.flatnonzero(
        (target_x >= west) & (target_x <= east)
    )
    target_rows = np.flatnonzero(
        (target_y >= south) & (target_y <= north)
    )
    if target_columns.size == 0 or target_rows.size == 0:
        return None

    source_columns = np.rint(
        (target_x[target_columns] - info.x0) / info.dx
    ).astype(np.int64)
    source_rows = np.rint(
        (target_y[target_rows] - info.y0) / info.dy
    ).astype(np.int64)
    column_inside = (source_columns >= 0) & (source_columns < info.shape[1])
    row_inside = (source_rows >= 0) & (source_rows < info.shape[0])
    source_columns = np.clip(source_columns, 0, info.shape[1] - 1)
    source_rows = np.clip(source_rows, 0, info.shape[0] - 1)
    sampled = source[np.ix_(source_rows, source_columns)]
    sampled_valid = source_valid[np.ix_(source_rows, source_columns)].copy()
    sampled_valid &= row_inside[:, None] & column_inside[None, :]
    row_slice = slice(int(target_rows[0]), int(target_rows[-1]) + 1)
    column_slice = slice(
        int(target_columns[0]), int(target_columns[-1]) + 1
    )
    return row_slice, column_slice, np.asarray(sampled), sampled_valid


def _mosaic_bathymetry(
    sources: list[RasterInfo], target: GridSpec
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    total = np.zeros((target.rows, target.columns), dtype=np.float32)
    count = np.zeros((target.rows, target.columns), dtype=np.uint8)
    layers: list[dict] = []
    for info in sources:
        values = _read_raster(info)
        valid = _valid_mask(values, info.nodata)
        registered = _registered_region(values, valid, info, target)
        if registered is None:
            raise RuntimeError(f"bathymetry does not overlap target: {info.path}")
        rows, columns, sampled, sampled_valid = registered
        region_total = total[rows, columns]
        region_count = count[rows, columns]
        region_total[sampled_valid] += sampled[sampled_valid].astype(np.float32)
        region_count[sampled_valid] += 1
        layers.append(
            {
                "file": str(info.path.relative_to(ROOT)),
                "source_shape_pixels": list(info.shape),
                "source_resolution_m": [info.dx, abs(info.dy)],
                "source_valid_samples": int(np.count_nonzero(valid)),
                "registered_samples": int(np.count_nonzero(sampled_valid)),
            }
        )
        del values, valid, sampled, sampled_valid
    valid = count > 0
    bathymetry = np.full(total.shape, np.nan, dtype=np.float32)
    bathymetry[valid] = total[valid] / count[valid].astype(np.float32)
    return bathymetry, valid, layers


def _normalize_registered_layer(
    values: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, float, float]:
    measured = values[valid].astype(np.float64)
    if measured.size == 0:
        raise RuntimeError("registered acoustic layer is empty")
    low, high = np.percentile(measured, (1.0, 99.0))
    if not high > low:
        raise RuntimeError("registered acoustic layer has no dynamic range")
    unit = np.full(values.shape, np.nan, dtype=np.float32)
    unit[valid] = np.clip(
        (values[valid].astype(np.float32) - float(low)) / float(high - low),
        0.0,
        1.0,
    )
    return unit, float(low), float(high)


def _relative_power_registered_layer(
    values: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Decode a documented dB intensity layer without display contrast stretching.

    Each survey section may have an unknown constant gain. Subtracting its
    median is a constant gain change in power, preserving every within-section
    dB difference. The resulting values are relative power, not calibrated
    seafloor scattering strength. No percentile clipping is performed.
    """

    measured = np.asarray(values[valid], dtype=np.float64)
    if not measured.size or not np.isfinite(measured).all():
        raise ValueError("dB backscatter must contain finite measured samples")
    reference_db = float(np.median(measured))
    exponent = (measured - reference_db) / 10.0
    # Reject unsupported data instead of silently clipping physical contrast.
    limit = np.log10(np.finfo(np.float32).max)
    if np.any(np.abs(exponent) >= limit):
        raise ValueError("dB dynamic range cannot be represented as float32 power")
    power = np.full(values.shape, np.nan, dtype=np.float32)
    power[valid] = np.power(10.0, exponent).astype(np.float32)
    return power, {
        "native_units": "dB intensity (as documented by the source)",
        "registered_reference_db": reference_db,
        "radiometric_conversion": "10 ** ((source_db - registered_reference_db) / 10)",
        "radiometric_reference": "per-section median; unknown constant survey gain",
        "percentile_clipping": False,
        "registered_percentile_1_99": np.percentile(measured, (1, 99)).tolist(),
    }


def _mosaic_backscatter(
    sources: list[tuple[str, RasterInfo]],
    target: GridSpec,
    bathymetry_valid: np.ndarray,
    *,
    documented_db: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    total = np.zeros((target.rows, target.columns), dtype=np.float32)
    count = np.zeros((target.rows, target.columns), dtype=np.uint8)
    source_mask = np.zeros((target.rows, target.columns), dtype=np.uint16)
    layers: list[dict] = []
    for index, (label_name, info) in enumerate(sources):
        values = _read_raster(info)
        valid = _valid_mask(values, info.nodata)
        registered = _registered_region(values, valid, info, target)
        if registered is None:
            continue
        rows, columns, sampled, sampled_valid = registered
        sampled_valid &= bathymetry_valid[rows, columns]
        if not np.any(sampled_valid):
            continue
        if documented_db:
            unit, radiometry = _relative_power_registered_layer(sampled, sampled_valid)
        else:
            unit, low, high = _normalize_registered_layer(sampled, sampled_valid)
            radiometry = {"registered_percentile_1_99": [low, high]}
        region_total = total[rows, columns]
        region_count = count[rows, columns]
        region_sources = source_mask[rows, columns]
        region_total[sampled_valid] += unit[sampled_valid]
        region_count[sampled_valid] += 1
        region_sources[sampled_valid] |= np.uint16(1 << index)
        layers.append(
            {
                "label": label_name,
                "file": str(info.path.relative_to(ROOT)),
                "source_shape_pixels": list(info.shape),
                "source_resolution_m": [info.dx, abs(info.dy)],
                "registered_samples": int(np.count_nonzero(sampled_valid)),
                **radiometry,
            }
        )
        del values, valid, sampled, sampled_valid, unit
    registered_valid = (count > 0) & bathymetry_valid
    backscatter = np.full(total.shape, np.nan, dtype=np.float32)
    backscatter[registered_valid] = (
        total[registered_valid] / count[registered_valid].astype(np.float32)
    )
    source_mask[~registered_valid] = 0
    return backscatter, registered_valid, source_mask, layers


def _masked_navigation_surface(
    elevation: np.ndarray, valid: np.ndarray, sigma: float = 2.2
) -> np.ndarray:
    weighted = gaussian_filter(
        np.where(valid, elevation, 0.0).astype(np.float32),
        sigma=sigma,
        mode="nearest",
    )
    weights = gaussian_filter(
        valid.astype(np.float32), sigma=sigma, mode="nearest"
    )
    navigation = np.full(elevation.shape, np.nan, dtype=np.float32)
    supported = valid & (weights > 1.0e-6)
    navigation[supported] = weighted[supported] / weights[supported]
    return navigation


def _elevation_from_datum(
    source: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, float]:
    maximum = float(np.max(source[valid]))
    offset = -5.0 - maximum
    result = source.astype(np.float32, copy=True)
    result[valid] += np.float32(offset)
    result[~valid] = np.nan
    return result, float(offset)


def _elevation_from_positive_depth(
    source: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, float]:
    result = np.full(source.shape, np.nan, dtype=np.float32)
    result[valid] = -source[valid].astype(np.float32)
    return result, 0.0


def _flip_local(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.flipud(array).copy() for array in arrays)


def _base_metadata(
    *,
    title: str,
    dataset_id: str,
    grid: GridSpec,
    bathymetry_layers: list[dict],
    acoustic_layers: list[dict],
    source_kind: str,
    vertical_transform: str,
    vertical_offset_m: float,
) -> dict:
    return {
        "schema": "oceansim.measured-acoustic-map/v2",
        "title": title,
        "dataset_id": dataset_id,
        "source_directory": str((DATA_ROOT / dataset_id).relative_to(ROOT)),
        "source_kind": source_kind,
        "bathymetry_layers": bathymetry_layers,
        "backscatter_layers": acoustic_layers,
        "resolution_m": float(grid.resolution_m),
        "source_shape_pixels": [int(grid.rows), int(grid.columns)],
        "scene_extent_m": grid.scene_extent_m,
        "bounds_projected_m": grid.bounds,
        "vertical_transform": vertical_transform,
        "vertical_offset_m": float(vertical_offset_m),
        "vertical_scale": 1.0,
        "is_geographically_cropped": False,
        "processing": (
            "nearest registration to a common north-up grid; source NoData "
            "preserved; acoustic layers independently clipped at their "
            "registered 1st/99th percentiles and mapped monotonically to "
            "[0,1]; rows reversed for OceanSim local +Y"
        ),
        "backscatter_mosaic_units": "unitless normalized measured intensity [0,1]",
        "access": "public scientific data; not for navigation",
    }


def _prepare_chelan() -> tuple[tuple[np.ndarray, ...], dict]:
    dataset_id = DATASET_ALIASES["chelan"]
    raw = DATA_ROOT / dataset_id / "raw"
    bath_paths = sorted(raw.glob("Chelan_*_3m_bathy_NAD83_NAVD88.tif"))
    acoustic_paths = sorted(raw.glob("Chelan_*_3m_backscatter_NAD83/*.tif"))
    if len(bath_paths) != 5 or len(acoustic_paths) != 5:
        raise FileNotFoundError("Lake Chelan requires five bathymetry and five backscatter tiles")
    bath_info = [_raster_info(path) for path in bath_paths]
    acoustic_info = [_raster_info(path) for path in acoustic_paths]
    grid = _grid_from_rasters(bath_info, resolution_m=6.0)
    bathymetry, bath_valid, bath_layers = _mosaic_bathymetry(bath_info, grid)
    backscatter, back_valid, source_mask, acoustic_layers = _mosaic_backscatter(
        [(path.parent.name.split("_3m_")[0], info) for path, info in zip(acoustic_paths, acoustic_info)],
        grid,
        bath_valid,
    )
    elevation, offset = _elevation_from_datum(bathymetry, bath_valid)
    navigation = _masked_navigation_surface(elevation, bath_valid)
    elevation, navigation, bath_valid, backscatter, back_valid, source_mask = _flip_local(
        elevation, navigation, bath_valid, backscatter, back_valid, source_mask
    )
    metadata = _base_metadata(
        title="Lake Chelan measured multibeam bathymetry and backscatter",
        dataset_id=dataset_id,
        grid=grid,
        bathymetry_layers=bath_layers,
        acoustic_layers=acoustic_layers,
        source_kind="NORBIT WINGHEAD i67 multibeam bathymetry/backscatter",
        vertical_transform="NAVD88 elevation translated so highest valid seabed is -5 m",
        vertical_offset_m=offset,
    )
    metadata.update(
        {
            "source_doi": "https://doi.org/10.5066/P1GE3YDR",
            "source_horizontal_crs": "NAD83(2011) / UTM zone 10N",
            "source_vertical_datum": "NAVD88",
            "output_resolution_note": "two-by-two reduction from native 3 m grids to a 6 m validation grid",
        }
    )
    return (
        (elevation, navigation, bath_valid, backscatter, back_valid, source_mask),
        metadata,
    )


def _prepare_jenkinson() -> tuple[tuple[np.ndarray, ...], dict]:
    dataset_id = DATASET_ALIASES["jenkinson"]
    raw = DATA_ROOT / dataset_id / "raw"
    bath_path = raw / "JenkinsonLake_July_2023_all_1m_bathymetry_NAD83_NAVD88.tif"
    acoustic_path = raw / "JenkinsonLake_July_2023_all_1m_backscatter_NAD83.tif"
    bath_info = [_raster_info(bath_path)]
    acoustic_info = _raster_info(acoustic_path)
    grid = _grid_from_rasters(bath_info, resolution_m=2.0)
    bathymetry, bath_valid, bath_layers = _mosaic_bathymetry(bath_info, grid)
    backscatter, back_valid, source_mask, acoustic_layers = _mosaic_backscatter(
        [("SWATHplus-M backscatter", acoustic_info)], grid, bath_valid
    )
    elevation, offset = _elevation_from_datum(bathymetry, bath_valid)
    navigation = _masked_navigation_surface(elevation, bath_valid)
    elevation, navigation, bath_valid, backscatter, back_valid, source_mask = _flip_local(
        elevation, navigation, bath_valid, backscatter, back_valid, source_mask
    )
    metadata = _base_metadata(
        title="Jenkinson Lake measured interferometric bathymetry and backscatter",
        dataset_id=dataset_id,
        grid=grid,
        bathymetry_layers=bath_layers,
        acoustic_layers=acoustic_layers,
        source_kind="SWATHplus-M interferometric bathymetric side-scan sonar",
        vertical_transform="NAVD88 elevation translated so highest valid seabed is -5 m",
        vertical_offset_m=offset,
    )
    metadata.update(
        {
            "source_doi": "https://doi.org/10.5066/P13BSSDY",
            "source_vertical_datum": "NAVD88",
            "output_resolution_note": "two-by-two reduction from the complete-lake nominal 1 m grids",
        }
    )
    return (
        (elevation, navigation, bath_valid, backscatter, back_valid, source_mask),
        metadata,
    )


def _wallops_bathymetry(
    path: Path, grid: GridSpec
) -> tuple[np.ndarray, np.ndarray, dict]:
    total = np.zeros((grid.rows, grid.columns), dtype=np.float32)
    count = np.zeros((grid.rows, grid.columns), dtype=np.uint16)
    bounds = grid.bounds
    used = 0
    seen = 0
    minimum = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maximum = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    with path.open("r", encoding="ascii") as stream:
        header = stream.readline().strip()
        if header != "WGS84_18_X WGS84_18_Y WGS84_Z Uncertainty":
            raise RuntimeError(f"unexpected Wallops XYZ header: {header}")
        while True:
            lines = list(islice(stream, 500_000))
            if not lines:
                break
            flat = np.fromstring("".join(lines), sep=" ", dtype=np.float64)
            if flat.size != 4 * len(lines):
                raise RuntimeError("malformed Wallops XYZ row")
            chunk = flat.reshape(-1, 4)
            xyz = chunk[:, :3]
            minimum = np.minimum(minimum, np.min(xyz, axis=0))
            maximum = np.maximum(maximum, np.max(xyz, axis=0))
            columns = np.floor(
                (chunk[:, 0] - bounds["west"]) / grid.resolution_m
            ).astype(np.int64)
            rows = np.floor(
                (bounds["north"] - chunk[:, 1]) / grid.resolution_m
            ).astype(np.int64)
            inside = (
                (rows >= 0)
                & (rows < grid.rows)
                & (columns >= 0)
                & (columns < grid.columns)
                & np.isfinite(chunk[:, 2])
            )
            flat_index = rows[inside] * grid.columns + columns[inside]
            np.add.at(total.reshape(-1), flat_index, chunk[inside, 2].astype(np.float32))
            np.add.at(count.reshape(-1), flat_index, 1)
            used += int(np.count_nonzero(inside))
            seen += int(chunk.shape[0])
    valid = count > 0
    elevation = np.full(total.shape, np.nan, dtype=np.float32)
    elevation[valid] = total[valid] / count[valid].astype(np.float32)
    metadata = {
        "file": str(path.relative_to(ROOT)),
        "source_points": int(seen),
        "registered_points": int(used),
        "registered_cells": int(np.count_nonzero(valid)),
        "source_xyz_min": minimum.tolist(),
        "source_xyz_max": maximum.tolist(),
        "aggregation": "arithmetic mean of measured 1 m CUBE nodes in each 2 m output cell",
    }
    return elevation, valid, metadata


def _libtiff_scanline_sample(
    info: RasterInfo, target: GridSpec
) -> np.ndarray:
    if info.dtype != "float32":
        raise RuntimeError("Wallops scanline reader expects float32")
    target_x, target_y = _target_coordinates(target)
    source_columns = np.rint((target_x - info.x0) / info.dx).astype(np.int64)
    source_rows = np.rint((target_y - info.y0) / info.dy).astype(np.int64)
    columns_inside = (source_columns >= 0) & (source_columns < info.shape[1])
    rows_inside = (source_rows >= 0) & (source_rows < info.shape[0])
    output = np.full((target.rows, target.columns), np.nan, dtype=np.float32)

    library = ctypes.CDLL("libtiff.so")
    library.TIFFOpen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.TIFFOpen.restype = ctypes.c_void_p
    library.TIFFClose.argtypes = [ctypes.c_void_p]
    library.TIFFScanlineSize64.argtypes = [ctypes.c_void_p]
    library.TIFFScanlineSize64.restype = ctypes.c_uint64
    library.TIFFReadScanline.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint16,
    ]
    library.TIFFReadScanline.restype = ctypes.c_int
    handle = library.TIFFOpen(str(info.path).encode(), b"r")
    if not handle:
        raise RuntimeError(f"libtiff could not open {info.path}")
    try:
        scanline_bytes = int(library.TIFFScanlineSize64(handle))
        expected_bytes = info.shape[1] * np.dtype(np.float32).itemsize
        if scanline_bytes < expected_bytes:
            raise RuntimeError("Wallops TIFF scanline is shorter than expected")
        buffer = ctypes.create_string_buffer(scanline_bytes)
        selected_columns = source_columns[columns_inside]
        for target_row in np.flatnonzero(rows_inside):
            source_row = int(source_rows[target_row])
            if library.TIFFReadScanline(handle, buffer, source_row, 0) < 0:
                raise RuntimeError(f"failed reading Wallops TIFF row {source_row}")
            scanline = np.frombuffer(
                buffer, dtype=np.float32, count=info.shape[1]
            )
            output[target_row, columns_inside] = scanline[selected_columns]
    finally:
        library.TIFFClose(handle)
    return output


def _prepare_wallops() -> tuple[tuple[np.ndarray, ...], dict]:
    dataset_id = DATASET_ALIASES["wallops"]
    raw = DATA_ROOT / dataset_id / "raw"
    xyz_path = raw / "2024-310-FA_MBES_Wallops_1m_WGS84_UTM18N.xyz"
    acoustic_path = raw / "2024-310-FA_Wallops_MBES_1m_Backscatter.tiff"
    grid = _grid_from_point_bounds(
        453332.5, 464689.5, 4178775.5, 4194493.5, resolution_m=2.0
    )
    bathymetry, bath_valid, bath_layer = _wallops_bathymetry(xyz_path, grid)
    acoustic_info = _raster_info(acoustic_path)
    raw_backscatter = _libtiff_scanline_sample(acoustic_info, grid)
    back_valid = _valid_mask(raw_backscatter, acoustic_info.nodata) & bath_valid
    backscatter, low, high = _normalize_registered_layer(
        raw_backscatter, back_valid
    )
    source_mask = np.zeros(back_valid.shape, dtype=np.uint16)
    source_mask[back_valid] = 1
    elevation, offset = _elevation_from_datum(bathymetry, bath_valid)
    navigation = _masked_navigation_surface(elevation, bath_valid)
    elevation, navigation, bath_valid, backscatter, back_valid, source_mask = _flip_local(
        elevation, navigation, bath_valid, backscatter, back_valid, source_mask
    )
    acoustic_layers = [
        {
            "label": "SeaBat T50-P multibeam backscatter",
            "file": str(acoustic_path.relative_to(ROOT)),
            "source_shape_pixels": list(acoustic_info.shape),
            "source_pixel_spacing_m": [acoustic_info.dx, abs(acoustic_info.dy)],
            "registered_samples": int(np.count_nonzero(back_valid)),
            "registered_percentile_1_99": [low, high],
            "note": "official 1 m CUBE-derived product stored on a sparse 0.2 m TIFF support grid",
        }
    ]
    metadata = _base_metadata(
        title="Wallops and Assawoman measured multibeam map",
        dataset_id=dataset_id,
        grid=grid,
        bathymetry_layers=[bath_layer],
        acoustic_layers=acoustic_layers,
        source_kind="dual-head Teledyne Reson SeaBat T50-P multibeam",
        vertical_transform="WGS84 G2139 ellipsoid elevation translated so highest valid seabed is -5 m",
        vertical_offset_m=offset,
    )
    metadata.update(
        {
            "source_doi": "https://doi.org/10.5066/P14SGUMT",
            "source_horizontal_crs": "WGS84 (G2139) / UTM zone 18N",
            "source_vertical_datum": "WGS84 (G2139) ellipsoid",
            "output_resolution_note": "measured 1 m bathymetry nodes aggregated to 2 m for tractable whole-map validation",
        }
    )
    return (
        (elevation, navigation, bath_valid, backscatter, back_valid, source_mask),
        metadata,
    )


def _aligned_crop_grid(
    bathymetry: RasterInfo, coverage: RasterInfo
) -> tuple[GridSpec, slice, slice]:
    west, east, south, north = coverage.bounds
    columns = np.arange(bathymetry.shape[1], dtype=np.int64)
    rows = np.arange(bathymetry.shape[0], dtype=np.int64)
    x = bathymetry.x0 + columns * bathymetry.dx
    y = bathymetry.y0 + rows * bathymetry.dy
    selected_columns = columns[(x >= west) & (x <= east)]
    selected_rows = rows[(y >= south) & (y <= north)]
    if selected_columns.size < 2 or selected_rows.size < 2:
        raise RuntimeError("Sheboygan bathymetry and side-scan do not overlap")
    row_slice = slice(int(selected_rows[0]), int(selected_rows[-1]) + 1)
    column_slice = slice(
        int(selected_columns[0]), int(selected_columns[-1]) + 1
    )
    grid = GridSpec(
        float(bathymetry.dx),
        int(selected_rows.size),
        int(selected_columns.size),
        float(x[selected_columns[0]]),
        float(y[selected_rows[0]]),
    )
    return grid, row_slice, column_slice


def _prepare_sheboygan() -> tuple[tuple[np.ndarray, ...], dict]:
    dataset_id = DATASET_ALIASES["sheboygan"]
    raw = DATA_ROOT / dataset_id / "raw/Sheboygan_2018"
    bath_path = raw / "LakeMichigan_RVStorm_2018_EM2040c_Bathymetry_Shallow_2m.tif"
    mbes_path = raw / "LakeMichigan_RVStorm_2018_EM2040c_Backscatter_Shallow_1m.tif"
    side_path = raw / "LakeMichigan_RVStorm_2018_EM2040c_SidescanLF_Shallow_50cm.tif"
    bath_info = _raster_info(bath_path)
    mbes_info = _raster_info(mbes_path)
    side_info = _raster_info(side_path)
    grid, row_slice, column_slice = _aligned_crop_grid(bath_info, mbes_info)
    bath_values = _read_raster(bath_info)[row_slice, column_slice].copy()
    bath_valid = _valid_mask(bath_values, bath_info.nodata)
    elevation, offset = _elevation_from_positive_depth(bath_values, bath_valid)
    del bath_values
    backscatter, back_valid, source_mask, acoustic_layers = _mosaic_backscatter(
        [
            ("Kongsberg EM2040C multibeam backscatter", mbes_info),
            ("Klein System 3000 low-frequency side-scan", side_info),
        ],
        grid,
        bath_valid,
    )
    navigation = _masked_navigation_surface(elevation, bath_valid)
    elevation, navigation, bath_valid, backscatter, back_valid, source_mask = _flip_local(
        elevation, navigation, bath_valid, backscatter, back_valid, source_mask
    )
    bath_layers = [
        {
            "file": str(bath_path.relative_to(ROOT)),
            "source_shape_pixels": list(bath_info.shape),
            "source_resolution_m": [bath_info.dx, abs(bath_info.dy)],
            "cropped_shape_pixels": [grid.rows, grid.columns],
            "registered_samples": int(np.count_nonzero(bath_valid)),
        }
    ]
    for layer in acoustic_layers:
        if layer["label"].startswith("Klein"):
            layer["frequency_khz"] = 100
    metadata = _base_metadata(
        title="Lake Michigan Sheboygan W00463 multibeam and side-scan map",
        dataset_id=dataset_id,
        grid=grid,
        bathymetry_layers=bath_layers,
        acoustic_layers=acoustic_layers,
        source_kind="Kongsberg EM2040C bathymetry plus Klein System 3000 side-scan",
        vertical_transform="published positive depth converted to local Z-up elevation",
        vertical_offset_m=offset,
    )
    metadata.update(
        {
            "source_accession": "NOAA NCEI 0222391",
            "source_survey": "W00463",
            "source_horizontal_crs": "NAD83 / UTM zone 16N",
            "source_vertical_datum": "NAVD88 depth in metres",
            "is_geographically_cropped": True,
            "crop_note": "bathymetry cropped only to the measured shallow-water EM2040C backscatter extent; NoData inside the crop remains invalid",
            "excluded_intensity_layers": [
                "500 kHz side-scan is preserved raw but uses NAD83 / UTM zone 18N and requires a true reprojection before fusion",
            ],
        }
    )
    return (
        (elevation, navigation, bath_valid, backscatter, back_valid, source_mask),
        metadata,
    )


def _route_from_joint_coverage(
    world: MeasuredBathymetryGrid,
    dataset_id: str,
    output: Path,
) -> tuple[dict, dict]:
    joint = np.asarray(
        world.valid_mask & world.backscatter_valid_mask, dtype=bool
    )
    clearance = distance_transform_edt(joint, sampling=world.resolution_m)
    maximum_clearance = float(np.max(clearance, initial=0.0))
    required_clearance = 250.0
    base_margin = max(50.0, 5.0 * world.resolution_m)
    thresholds = []
    for value in (
        required_clearance + base_margin,
        350.0,
        450.0,
        550.0,
        0.8 * maximum_clearance,
    ):
        value = float(value)
        if required_clearance < value < maximum_clearance - world.resolution_m:
            if not any(abs(value - old) < 0.5 * world.resolution_m for old in thresholds):
                thresholds.append(value)
    thresholds.sort()
    if not thresholds:
        raise RuntimeError(
            f"{dataset_id}: joint acoustic coverage has only "
            f"{maximum_clearance:.1f} m clearance; production requires 250 m"
        )

    failures: list[str] = []
    chosen = None
    for threshold in thresholds:
        safe = clearance >= threshold
        components, component_count = label(
            safe, structure=np.ones((3, 3), dtype=np.uint8)
        )
        if component_count == 0:
            continue
        component_sizes = np.bincount(components.reshape(-1))
        component_sizes[0] = 0
        component_id = int(np.argmax(component_sizes))
        component = np.asarray(components == component_id, dtype=np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        contour = max(
            contours, key=lambda candidate: cv2.arcLength(candidate, True)
        )
        epsilon_pixels = max(1.0, 20.0 / world.resolution_m)
        simplified = cv2.approxPolyDP(contour, epsilon_pixels, True)
        pixels = simplified.reshape(-1, 2).astype(np.float64)
        if pixels.shape[0] < 4:
            continue
        if pixels.shape[0] > 600:
            indices = np.linspace(
                0, pixels.shape[0], 600, endpoint=False, dtype=np.int64
            )
            pixels = pixels[indices]
        waypoints = np.column_stack(
            (
                (pixels[:, 0] - 0.5 * (joint.shape[1] - 1))
                * world.resolution_m,
                (pixels[:, 1] - 0.5 * (joint.shape[0] - 1))
                * world.resolution_m,
            )
        )
        for smoothing_iterations in (2, 1, 0):
            try:
                survey = WaypointTerrainFollowingSurvey(
                    world,
                    waypoints,
                    speed_mps=1.5,
                    altitude_m=40.0,
                    smoothing_iterations=smoothing_iterations,
                    maximum_pitch_deg=12.0,
                )
                report = survey.validate_route(
                    sample_count=8192,
                    minimum_clearance_m=required_clearance,
                    require_measured_backscatter=True,
                )
            except ValueError as error:
                failures.append(
                    f"threshold={threshold:.1f}, smooth={smoothing_iterations}: {error}"
                )
                continue
            chosen = (
                threshold,
                smoothing_iterations,
                waypoints,
                report,
                int(component_sizes[component_id]),
            )
            break
        if chosen is not None:
            break
    if chosen is None:
        detail = "; ".join(failures[-6:])
        raise RuntimeError(f"{dataset_id}: no production-safe route: {detail}")

    threshold, smoothing_iterations, waypoints, report, component_cells = chosen
    mask_hash = _sha256(output / "bathymetry_valid_mask.npy")
    backscatter_hash = _sha256(output / "backscatter_valid_mask.npy")
    route = {
        "schema": "oceansim.measured-terrain-route/v1",
        "name": f"{dataset_id} joint measured-coverage traversal",
        "description": (
            "Closed contour traversal inside the largest connected region "
            "having both measured bathymetry and measured acoustic intensity."
        ),
        "map": {
            "directory": dataset_id + "/prepared",
            "source_product": world.metadata.get("title", dataset_id),
            "resolution_m": float(world.resolution_m),
            "scene_extent_m": [float(world.width), float(world.length)],
            "bathymetry_valid_mask_sha256": mask_hash,
            "backscatter_valid_mask_sha256": backscatter_hash,
        },
        "route_definition": {
            "purpose": "cross_dataset_production_algorithm_validation",
            "coverage_claim": "largest_joint_measured_connected_region_contour",
            "joint_valid_mask": "bathymetry_valid_mask AND backscatter_valid_mask",
            "planning_clearance_m": float(threshold),
            "required_validation_clearance_m": required_clearance,
            "planner": "contour of the largest eroded eight-connected joint-valid component",
            "polyline_simplification": "OpenCV Ramer-Douglas-Peucker, nominal 20 m tolerance",
            "smoothing": "periodic Chaikin plus periodic Catmull-Rom with full route revalidation",
            "smoothing_iterations": int(smoothing_iterations),
            "closed": True,
            "joint_component_cells": component_cells,
            "joint_maximum_clearance_m": maximum_clearance,
            "validated_3d_path_length_m": float(report.path_length_m),
            "validated_minimum_bathymetry_clearance_m": float(
                report.bathymetry_min_clearance_m
            ),
            "validated_minimum_backscatter_clearance_m": float(
                report.backscatter_min_clearance_m
            ),
            "recommended_flight_profile": {
                "speed_mps": 1.5,
                "minimum_navigation_surface_altitude_m": 40.0,
                "maximum_pitch_deg": 12.0,
                "validated_period_s": float(report.period_s),
                "pitch_range_deg": list(report.pitch_range_deg),
            },
        },
        "control_points_xy_m": waypoints.tolist(),
    }
    return route, asdict(report)


def _write_prepared(
    dataset_id: str,
    arrays: tuple[np.ndarray, ...],
    metadata: dict,
    *,
    force: bool,
) -> Path:
    output = DATA_ROOT / dataset_id / "prepared"
    if output.exists():
        complete = all((output / name).is_file() for name in REQUIRED_OUTPUTS)
        if complete and not force:
            print(f"[OceanSim] already prepared: {output}", flush=True)
            return output
        if not force:
            raise RuntimeError(
                f"incomplete prepared directory exists: {output}; use --force"
            )
        for path in output.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                raise RuntimeError(
                    f"refusing to remove unexpected nested directory: {path}"
                )
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "elevation_m.npy",
        "navigation_elevation_m.npy",
        "bathymetry_valid_mask.npy",
        "backscatter.npy",
        "backscatter_valid_mask.npy",
        "backscatter_source_mask.npy",
    )
    for name, array in zip(names, arrays):
        np.save(output / name, array, allow_pickle=False)

    bathymetry_valid = arrays[2]
    backscatter_valid = arrays[4]
    metadata.update(
        {
            "bathymetry_valid_samples": int(np.count_nonzero(bathymetry_valid)),
            "bathymetry_valid_fraction": float(np.mean(bathymetry_valid)),
            "backscatter_registered_samples": int(np.count_nonzero(backscatter_valid)),
            "backscatter_registered_fraction_of_bathymetry": float(
                np.count_nonzero(backscatter_valid)
                / max(1, np.count_nonzero(bathymetry_valid))
            ),
            "raw_elevation_range_m": [
                float(np.nanmin(arrays[0])),
                float(np.nanmax(arrays[0])),
            ],
        }
    )
    metadata_path = output / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    world = MeasuredBathymetryGrid.from_path(output)
    route, route_report = _route_from_joint_coverage(
        world, dataset_id, output
    )
    route_path = output / "full_map_traversal_route.json"
    with route_path.open("w", encoding="utf-8") as stream:
        json.dump(route, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    output_arrays = {}
    for name in names:
        path = output / name
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        output_arrays[name] = {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    metadata["output_arrays"] = output_arrays
    metadata["survey_route"] = {
        "file": route_path.name,
        "sha256": _sha256(route_path),
        "validation": route_report,
    }
    metadata["checksum_manifest"] = "sha256sums.txt"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    manifest_names = (*names, "metadata.json", route_path.name)
    with (output / "sha256sums.txt").open(
        "w", encoding="ascii", newline="\n"
    ) as stream:
        for name in manifest_names:
            stream.write(f"{_sha256(output / name)}  {name}\n")
    print(
        f"[OceanSim] prepared {dataset_id}: "
        f"{world.width / 1000.0:.2f} x {world.length / 1000.0:.2f} km, "
        f"route {route_report['path_length_m'] / 1000.0:.2f} km, "
        f"joint clearance {route_report['backscatter_min_clearance_m']:.1f} m",
        flush=True,
    )
    return output


PREPARERS = {
    "chelan": _prepare_chelan,
    "jenkinson": _prepare_jenkinson,
    "wallops": _prepare_wallops,
    "sheboygan": _prepare_sheboygan,
}


def main() -> int:
    args = parse_args()
    selected = list(PREPARERS) if args.dataset == "all" else [args.dataset]
    for alias in selected:
        dataset_id = DATASET_ALIASES[alias]
        output = DATA_ROOT / dataset_id / "prepared"
        if output.is_dir() and all(
            (output / name).is_file() for name in REQUIRED_OUTPUTS
        ) and not args.force:
            print(f"[OceanSim] already prepared: {output}", flush=True)
            continue
        print(f"[OceanSim] preparing {alias}...", flush=True)
        arrays, metadata = PREPARERS[alias]()
        _write_prepared(
            dataset_id, arrays, metadata, force=bool(args.force)
        )
        del arrays
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
