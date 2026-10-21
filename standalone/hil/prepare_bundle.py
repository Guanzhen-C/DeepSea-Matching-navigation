#!/usr/bin/env python3
"""Export pinned lake/sea estimator sources and checksummed measured maps.

No checkout/reset is performed and unrelated project data never enters the
bundle. Run one bundle per process to avoid mixing baseline Python modules.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from standalone.hil.frozen_engine import BASELINES, build_module

ROOT = Path(__file__).resolve().parents[2]
MAPS = {
    "lake": ROOT / "data/underwater_terrain_datasets/lake_chelan_usgs_2023_3m/prepared_db_power",
    "sea": ROOT / "data/monterey_canyon_full_5m",
}
PATHS = ["isaacsim/oceansim/__init__.py", "isaacsim/oceansim/slam",
         "isaacsim/oceansim/utils/acoustic_sonar.py",
         "isaacsim/oceansim/utils/acoustic_visualization.py", "isaacsim/oceansim/sensors",
         "standalone/large_scale_slam_isaac_demo.py", "standalone/run_v2_full_map_slam_isaac.sh"]
MAP_NAMES = {"metadata.json", "elevation_m.npy", "navigation_elevation_m.npy", "bathymetry_valid_mask.npy",
             "backscatter.npy", "backscatter_valid_mask.npy", "backscatter_source_mask.npy",
             "full_map_traversal_route.json"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True, choices=BASELINES)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists():
        p.error("fresh output directory required")
    sha = BASELINES[args.profile][0]
    archive = subprocess.check_output(["git", "archive", sha, *PATHS], cwd=ROOT)
    output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        # Fixed git archive of trusted selected paths. Reject links/escape.
        for member in stream:
            destination = (output / member.name).resolve()
            if not destination.is_relative_to(output) or not (member.isdir() or member.isfile()):
                raise ValueError("unsafe archive member")
            stream.extract(member, output)
    _, audit = build_module(output / "standalone/large_scale_slam_isaac_demo.py", args.profile)
    (output / "map").mkdir()
    for name in sorted(MAP_NAMES):
        source = MAPS[args.profile] / name
        if source.exists():
            # Follow only explicitly selected known map files; no symlinks on Jetson.
            shutil.copyfile(source, output / "map" / name)
    manifest = {"profile": args.profile, "source_commit": sha, "extraction": audit,
                "map_source": str(MAPS[args.profile]), "files": {}}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(output))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), files=len(manifest["files"]),
                          bytes=sum(x["bytes"] for x in manifest["files"].values()), audit=audit), indent=2))


if __name__ == "__main__":
    main()
