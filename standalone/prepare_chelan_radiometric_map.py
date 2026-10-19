#!/usr/bin/env python3
"""Create a non-destructive, dB-decoded Chelan map on the existing 6 m grid.

Geometry, coverage, route, and sensor configuration remain byte-for-byte those
of the archived validation map. Only the acoustic encoding changes. The source
metadata declares intensity in dB; display-normalized [0,1] pixels must not be
used as a linear scattering coefficient. This product preserves within-section
dB differences but cannot establish absolute calibration between surveys.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from standalone.prepare_public_terrain_datasets import (  # noqa: E402
    GridSpec, _mosaic_backscatter, _raster_info,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite map: {output}")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_id") != "lake_chelan_usgs_2023_3m":
        raise ValueError("this conversion requires the documented Lake Chelan dB product")
    grid = GridSpec(
        resolution_m=float(metadata["resolution_m"]),
        rows=int(metadata["source_shape_pixels"][0]),
        columns=int(metadata["source_shape_pixels"][1]),
        x0=float(metadata["bounds_projected_m"]["west"]) + 0.5 * metadata["resolution_m"],
        y0=float(metadata["bounds_projected_m"]["north"]) - 0.5 * metadata["resolution_m"],
    )
    bath_valid = np.load(source / "bathymetry_valid_mask.npy", mmap_mode="r")
    layers = metadata["backscatter_layers"]
    sources = [(layer["label"], _raster_info(ROOT / layer["file"])) for layer in layers]
    print("[OceanSim] decoding native dB layers on the archived grid", flush=True)
    backscatter, valid, source_mask, layer_metadata = _mosaic_backscatter(
        sources, grid, np.flipud(bath_valid), documented_db=True,
    )
    archived_valid = np.load(source / "backscatter_valid_mask.npy", mmap_mode="r")
    if not np.array_equal(np.flipud(valid), archived_valid):
        raise RuntimeError("radiometric conversion changed measured coverage")
    archived_sources = np.load(source / "backscatter_source_mask.npy", mmap_mode="r")
    if not np.array_equal(np.flipud(source_mask), archived_sources):
        raise RuntimeError("radiometric conversion changed source registration")
    del valid, source_mask
    output.mkdir(parents=True)
    np.save(output / "backscatter.npy", np.flipud(backscatter), allow_pickle=False)
    del backscatter
    reused = (
        "elevation_m.npy", "navigation_elevation_m.npy", "bathymetry_valid_mask.npy",
        "backscatter_valid_mask.npy", "backscatter_source_mask.npy",
    )
    for name in reused:
        (output / name).symlink_to(source / name)
    route = json.loads((source / "full_map_traversal_route.json").read_text(encoding="utf-8"))
    route["map"]["directory"] = str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)
    route_path = output / "full_map_traversal_route.json"
    route_path.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = deepcopy(metadata)
    metadata["backscatter_layers"] = layer_metadata
    metadata["backscatter_mosaic_units"] = "relative linear acoustic power; per-section median power = 1 before overlap averaging"
    metadata["processing"] = (
        "archived geometry/coverage/route unchanged; documented native intensity dB "
        "converted to relative linear power before interpolation; no percentile "
        "clipping or display contrast stretch; equal power averaging at overlaps"
    )
    metadata["radiometry"] = {
        "encoding": "relative_linear_power_from_documented_db",
        "source_metadata": str((source.parent / "raw/Chelan_3m_backscatter_NAD83_metadata.txt").relative_to(ROOT)),
        "absolute_calibration": False,
        "unknown_inter_section_gain": True,
        "reference": "one constant median dB offset per source section",
        "display_contrast_is_not_measurement": True,
        "parent_map": str(source),
    }
    path = output / "backscatter.npy"
    array = np.load(path, mmap_mode="r")
    metadata["output_arrays"]["backscatter.npy"] = {
        "sha256": sha256(path), "bytes": path.stat().st_size,
        "shape": list(array.shape), "dtype": str(array.dtype),
    }
    metadata["survey_route"]["sha256"] = sha256(route_path)
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    names = (*reused, "backscatter.npy", "metadata.json", route_path.name)
    (output / "sha256sums.txt").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in names), encoding="ascii",
    )
    print(f"[OceanSim] prepared {output}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "data/underwater_terrain_datasets/lake_chelan_usgs_2023_3m"
    parser.add_argument("--source", type=Path, default=base / "prepared")
    parser.add_argument("--output", type=Path, default=base / "prepared_db_power")
    args = parser.parse_args()
    prepare(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
