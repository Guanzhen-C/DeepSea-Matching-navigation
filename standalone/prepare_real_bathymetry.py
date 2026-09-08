#!/usr/bin/env python3
"""Prepare the complete public USGS Monterey Canyon acoustic map.

The output keeps the complete 5 m bathymetry extent and resamples the public
USGS backscatter products onto the same grid.  No geographic crop or
horizontal/vertical exaggeration is applied.  Invalid source cells remain
invalid and are not converted into invented seabed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile


ROOT = Path(__file__).resolve().parents[1]
BATHYMETRY_SOURCE_URL = (
    "https://www.sciencebase.gov/catalog/file/get/556f8b8ae4b0d9246a9fd0b4"
)
BACKSCATTER_BASE_URL = (
    "https://cmgds.marine.usgs.gov/data/csmp/MontereyCanyon/data/"
)
SOURCE_DOI = "https://doi.org/10.5066/F7XD0ZQ4"
BATHYMETRY_MEMBER = (
    "BathymetryB_5m_MontereyCanyon/"
    "BathymetryB_5m_MontereyCanyon.tif"
)
BACKSCATTER_LAYERS = (
    ("A_USGS_SWATH", "BackscatterA_USGS_Swath_MontereyCanyon.zip"),
    ("B_EM300", "BackscatterB_EM300_MontereyCanyon.zip"),
    ("C_7125", "BackscatterC_7125_MontereyCanyon.zip"),
    ("D_CSUMB_SWATH", "BackscatterD_CSUMB_MontereyCanyon.zip"),
)
BACKSCATTER_ENCODINGS = {
    "A_USGS_SWATH": {
        "native_encoding": "gain-normalized 16-bit amplitude stored as float32",
        "native_units": "normalized amplitude counts",
    },
    "B_EM300": {
        "native_encoding": "EM-300 amplitude stored as negative float32 dB",
        "native_units": "dB",
    },
    "C_7125": {
        "native_encoding": "radiometrically corrected 8-bit grayscale",
        "native_units": "grayscale intensity counts",
    },
    "D_CSUMB_SWATH": {
        "native_encoding": "gain-normalized SWATHplus 8-bit grayscale",
        "native_units": "grayscale intensity counts",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the complete 29.2 x 17.7 km USGS Monterey Canyon "
            "bathymetry and measured acoustic-backscatter map"
        )
    )
    parser.add_argument(
        "--bathymetry-archive",
        "--archive",
        dest="bathymetry_archive",
        type=Path,
        default=ROOT / "downloads/BathymetryB_5m_MontereyCanyon.zip",
    )
    for index, (label, filename) in enumerate(BACKSCATTER_LAYERS):
        option = "--backscatter-" + label[0].lower() + "-archive"
        aliases = [option]
        if index == 0:
            aliases.append("--backscatter-archive")
        parser.add_argument(
            *aliases,
            dest="backscatter_" + label[0].lower() + "_archive",
            type=Path,
            default=ROOT / "downloads" / filename,
        )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/monterey_canyon_full_5m",
        help="output directory containing memory-mappable NPY arrays",
    )
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[OceanSim] downloading {url} -> {destination}", flush=True)
    urllib.request.urlretrieve(url, destination)


def _ensure_archive(path: Path, url: str, download: bool) -> None:
    if path.is_file():
        return
    if not download:
        raise FileNotFoundError(
            f"{path} is missing; pass --download or provide the archive"
        )
    _download(url, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_member(archive: zipfile.ZipFile, stem: str, suffix: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if Path(name).stem.lower() == stem.lower()
        and name.lower().endswith(suffix.lower())
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {stem}{suffix} member, found {matches}"
        )
    return matches[0]


def _read_tiff_and_world_file(
    archive_path: Path, member: str | None, stem: str
) -> tuple[
    np.ndarray,
    tuple[float, float, float, float, float, float],
    float | None,
]:
    with zipfile.ZipFile(archive_path) as archive:
        if member is not None:
            tiff_member = member
        else:
            tiff_candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".tif")
            ]
            exact = [
                name
                for name in tiff_candidates
                if Path(name).stem.lower() == stem.lower()
            ]
            candidates = exact or tiff_candidates
            if len(candidates) != 1:
                raise RuntimeError(
                    f"expected one raster TIFF in {archive_path}, found "
                    f"{candidates}"
                )
            tiff_member = candidates[0]
        with tifffile.TiffFile(io.BytesIO(archive.read(tiff_member))) as tiff:
            page = tiff.pages[0]
            nodata_tag = page.tags.get("GDAL_NODATA")
            try:
                nodata = (
                    None if nodata_tag is None else float(nodata_tag.value)
                )
            except (TypeError, ValueError):
                nodata = None
            raster = page.asarray()
        world_member = _find_member(archive, Path(tiff_member).stem, ".tfw")
        world_values = tuple(
            float(value)
            for value in archive.read(world_member).decode("ascii").splitlines()
        )
        if len(world_values) != 6:
            raise RuntimeError(f"invalid world file: {world_member}")
    if raster.ndim != 2:
        raise RuntimeError(f"expected one-band raster, got {raster.shape}")
    return np.asarray(raster), world_values, nodata


def _valid_mask(values: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return source validity without imposing an intensity sign convention.

    The four USGS products do not share one numeric encoding: notably the
    EM-300 product contains valid *negative* dB values and the Reson 7125
    product uses 0 as a valid dark pixel while reserving 255 for nodata.
    Consequently the GeoTIFF nodata value, rather than ``values > 0``, is the
    authoritative discriminator.
    """

    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    if np.issubdtype(values.dtype, np.floating):
        valid &= values > -1.0e20
    return valid


def _robust_unit_normalize(
    values: np.ndarray,
    valid: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[np.ndarray, float, float]:
    """Map one native sonar product monotonically to [0, 1].

    The source products were acquired with different frequencies and
    processing chains, so their native numbers are not directly calibrated
    against one another.  Independent robust linear scaling preserves each
    product's measured spatial texture without pretending that amplitude
    counts, dB, and display-gray values are one physical unit.
    """

    if not np.any(valid):
        raise ValueError("cannot normalize an empty backscatter layer")
    low, high = np.percentile(
        values[valid].astype(np.float64),
        (float(lower_percentile), float(upper_percentile)),
    )
    if not high > low:
        raise ValueError("backscatter layer has no usable dynamic range")
    normalized = np.full(values.shape, np.nan, dtype=np.float32)
    normalized[valid] = np.clip(
        (values[valid].astype(np.float32) - float(low)) / float(high - low),
        0.0,
        1.0,
    )
    return normalized, float(low), float(high)


def _nearest_registered_resample(
    source: np.ndarray,
    source_valid: np.ndarray,
    source_world: tuple[float, float, float, float, float, float],
    target_shape: tuple[int, int],
    target_world: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Resample two north-up UTM rasters without materializing XY grids."""

    source_dx, source_rx, source_ry, source_dy, source_x0, source_y0 = source_world
    target_dx, target_rx, target_ry, target_dy, target_x0, target_y0 = target_world
    if max(abs(source_rx), abs(source_ry), abs(target_rx), abs(target_ry)) > 1.0e-9:
        raise RuntimeError("rotated world files are not supported")
    if source_dx <= 0.0 or target_dx <= 0.0 or source_dy >= 0.0 or target_dy >= 0.0:
        raise RuntimeError("expected north-up UTM rasters")

    rows, columns = target_shape
    target_x = target_x0 + np.arange(columns, dtype=np.float64) * target_dx
    target_y = target_y0 + np.arange(rows, dtype=np.float64) * target_dy
    source_columns = np.rint((target_x - source_x0) / source_dx).astype(np.int64)
    source_rows = np.rint((target_y - source_y0) / source_dy).astype(np.int64)
    columns_inside = (source_columns >= 0) & (source_columns < source.shape[1])
    rows_inside = (source_rows >= 0) & (source_rows < source.shape[0])
    clipped_columns = np.clip(source_columns, 0, source.shape[1] - 1)
    clipped_rows = np.clip(source_rows, 0, source.shape[0] - 1)
    sampled = source[np.ix_(clipped_rows, clipped_columns)]
    sampled_valid = source_valid[np.ix_(clipped_rows, clipped_columns)]
    sampled_valid &= rows_inside[:, None] & columns_inside[None, :]
    return np.asarray(sampled), sampled_valid


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


def main() -> int:
    args = parse_args()
    _ensure_archive(
        args.bathymetry_archive, BATHYMETRY_SOURCE_URL, args.download
    )
    backscatter_archives: list[tuple[str, Path, str]] = []
    for label, filename in BACKSCATTER_LAYERS:
        path = getattr(args, "backscatter_" + label[0].lower() + "_archive")
        url = BACKSCATTER_BASE_URL + filename
        _ensure_archive(path, url, args.download)
        backscatter_archives.append((label, path, url))

    bathymetry, bathymetry_world, bathymetry_nodata = _read_tiff_and_world_file(
        args.bathymetry_archive, BATHYMETRY_MEMBER, ""
    )
    bathymetry_valid = _valid_mask(bathymetry, bathymetry_nodata)

    backscatter_sum = np.zeros(bathymetry.shape, dtype=np.float32)
    backscatter_count = np.zeros(bathymetry.shape, dtype=np.uint8)
    backscatter_source_mask = np.zeros(bathymetry.shape, dtype=np.uint8)
    layer_metadata: list[dict] = []
    for layer_index, (label, archive_path, source_url) in enumerate(
        backscatter_archives
    ):
        expected_stem = Path(archive_path).stem
        backscatter, backscatter_world, backscatter_nodata = (
            _read_tiff_and_world_file(
                archive_path, None, expected_stem
            )
        )
        source_valid = _valid_mask(backscatter, backscatter_nodata)
        registered_raw, layer_valid = _nearest_registered_resample(
            backscatter,
            source_valid,
            backscatter_world,
            bathymetry.shape,
            bathymetry_world,
        )
        layer_valid &= bathymetry_valid
        if not np.any(layer_valid):
            raise RuntimeError(f"backscatter layer {label} has no overlap")
        try:
            unit, low, high = _robust_unit_normalize(
                registered_raw, layer_valid
            )
        except ValueError as error:
            raise RuntimeError(f"backscatter layer {label}: {error}") from error
        backscatter_sum[layer_valid] += unit[layer_valid]
        backscatter_count[layer_valid] += 1
        backscatter_source_mask[layer_valid] |= np.uint8(1 << layer_index)
        encoding = BACKSCATTER_ENCODINGS[label]
        layer_metadata.append(
            {
                "label": label,
                "archive_file": archive_path.name,
                "source_download": source_url,
                "archive_sha256": _sha256(archive_path),
                "source_shape_pixels": list(backscatter.shape),
                "source_resolution_m": float(backscatter_world[0]),
                "source_nodata": (
                    None
                    if backscatter_nodata is None
                    else float(backscatter_nodata)
                ),
                "source_valid_samples": int(np.count_nonzero(source_valid)),
                "registered_samples": int(np.count_nonzero(layer_valid)),
                "raw_percentile_1_99": [float(low), float(high)],
                "mosaic_normalization": (
                    "linear clip of native values between registered valid "
                    "1st and 99th percentiles"
                ),
                **encoding,
            }
        )
        del backscatter, source_valid, registered_raw, unit

    registered_valid = backscatter_count > 0
    normalized_backscatter = np.full(bathymetry.shape, np.nan, dtype=np.float32)
    normalized_backscatter[registered_valid] = (
        backscatter_sum[registered_valid]
        / backscatter_count[registered_valid].astype(np.float32)
    )

    elevation = bathymetry.astype(np.float32)
    elevation[~bathymetry_valid] = np.nan
    navigation = _masked_navigation_surface(elevation, bathymetry_valid)

    # GeoTIFF rows are north-to-south; local OceanSim +Y is south-to-north.
    elevation = np.flipud(elevation).copy()
    navigation = np.flipud(navigation).copy()
    normalized_backscatter = np.flipud(normalized_backscatter).copy()
    bathymetry_valid = np.flipud(bathymetry_valid).copy()
    registered_valid = np.flipud(registered_valid).copy()
    backscatter_source_mask = np.flipud(backscatter_source_mask).copy()

    resolution = float(bathymetry_world[0])
    rows, columns = elevation.shape
    west = float(bathymetry_world[4] - 0.5 * resolution)
    east = west + columns * resolution
    north = float(bathymetry_world[5] + 0.5 * resolution)
    south = north - rows * resolution
    overlap_histogram = {
        str(layer_count): int(np.count_nonzero(backscatter_count == layer_count))
        for layer_count in range(1, len(BACKSCATTER_LAYERS) + 1)
    }
    metadata = {
        "title": "Complete Monterey Canyon measured acoustic map",
        "source_agency": "U.S. Geological Survey",
        "source_product": (
            "BathymetryB [5m] and Backscatter A/B/C/D measured mosaic"
        ),
        "source_doi": SOURCE_DOI,
        "bathymetry_source_download": BATHYMETRY_SOURCE_URL,
        "bathymetry_archive_sha256": _sha256(args.bathymetry_archive),
        "backscatter_layers": layer_metadata,
        "survey_years": "1998-2014",
        "sensors": [
            "Simrad EM-300 multibeam echosounder",
            "Reson 7125 multibeam echosounder",
            "SEA SWATHplus bathymetric side-scan sonar",
        ],
        "bathymetry_crs": "NAD83 / UTM zone 10N",
        "backscatter_crs": "WGS84 / UTM zone 10N",
        "bounds_utm_m": {
            "west": west,
            "east": east,
            "south": south,
            "north": north,
        },
        "source_shape_pixels": [rows, columns],
        "resolution_m": resolution,
        "raster_footprint_m": [resolution * columns, resolution * rows],
        "scene_extent_m": [
            resolution * (columns - 1),
            resolution * (rows - 1),
        ],
        "is_geographically_cropped": False,
        "bathymetry_valid_samples": int(np.count_nonzero(bathymetry_valid)),
        "bathymetry_valid_fraction": float(np.mean(bathymetry_valid)),
        "backscatter_registered_samples": int(
            np.count_nonzero(registered_valid)
        ),
        "backscatter_registered_fraction_of_bathymetry": float(
            np.count_nonzero(registered_valid)
            / max(1, np.count_nonzero(bathymetry_valid))
        ),
        "backscatter_layer_overlap_histogram": overlap_histogram,
        "backscatter_mosaic_units": "unitless normalized intensity [0, 1]",
        "backscatter_mosaic_note": (
            "Every valid registered source contributes with equal weight in "
            "overlap cells. Native values are heterogeneous and this mosaic "
            "is suitable as measured spatial texture, not as calibrated "
            "absolute target strength. The source bitmask preserves layer "
            "lineage for every mosaic cell."
        ),
        "raw_elevation_range_m": [
            float(np.nanmin(elevation)),
            float(np.nanmax(elevation)),
        ],
        "vertical_offset_m": 0.0,
        "vertical_scale": 1.0,
        "processing": (
            "complete source extent; no geographic crop or exaggeration; "
            "north-up row reversal; invalid cells retained as NaN; four "
            "measured backscatter layers independently linearly normalized "
            "between their 1st and 99th percentiles, nearest-registered to "
            "the 5 m bathymetry grid, and averaged only where measured layers "
            "overlap"
        ),
        "datum_note": (
            "The source publishes bathymetry in NAD83 and backscatter in "
            "WGS84. Their UTM grids are overlaid directly; this sub-cell "
            "difference is below the published >=2 m backscatter horizontal "
            "uncertainty and the 5 m output cell size."
        ),
        "access": (
            "Public-domain USGS scientific data with attribution; not for "
            "navigation"
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "elevation_m.npy", elevation, allow_pickle=False)
    np.save(
        args.output / "navigation_elevation_m.npy",
        navigation,
        allow_pickle=False,
    )
    np.save(
        args.output / "bathymetry_valid_mask.npy",
        bathymetry_valid,
        allow_pickle=False,
    )
    np.save(
        args.output / "backscatter.npy",
        normalized_backscatter,
        allow_pickle=False,
    )
    np.save(
        args.output / "backscatter_valid_mask.npy",
        registered_valid,
        allow_pickle=False,
    )
    np.save(
        args.output / "backscatter_source_mask.npy",
        backscatter_source_mask,
        allow_pickle=False,
    )
    output_array_names = (
        "elevation_m.npy",
        "navigation_elevation_m.npy",
        "bathymetry_valid_mask.npy",
        "backscatter.npy",
        "backscatter_valid_mask.npy",
        "backscatter_source_mask.npy",
    )
    output_arrays = {}
    for filename in output_array_names:
        path = args.output / filename
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        output_arrays[filename] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    metadata["output_arrays"] = output_arrays
    metadata["checksum_manifest"] = "sha256sums.txt"
    with (args.output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    manifest_names = (*output_array_names, "metadata.json")
    with (args.output / "sha256sums.txt").open(
        "w", encoding="ascii", newline="\n"
    ) as stream:
        for filename in manifest_names:
            stream.write(f"{_sha256(args.output / filename)}  {filename}\n")
    print(
        f"[OceanSim] wrote complete map {args.output}: {columns} x {rows}, "
        f"{metadata['scene_extent_m'][0] / 1000.0:.2f} x "
        f"{metadata['scene_extent_m'][1] / 1000.0:.2f} km, "
        f"{metadata['bathymetry_valid_samples']:,} measured depth cells, "
        f"{metadata['backscatter_registered_samples']:,} registered "
        "backscatter cells",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
