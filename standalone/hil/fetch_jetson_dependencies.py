#!/usr/bin/env python3
"""Download pinned ARM build inputs on the PC; never install into system Python."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl": {
        "url": "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl",
        "sha256": "6f75fd2d2ef840ede1a90dbcf40a5458214bee26cc803fa510cda2e8978d972a",
        "status": "validated on L4T 36.4.7: CUDA float64 solve/SVD and feature CPU/CUDA consensus parity",
    },
    "torch-2.8.0-cp310-cp310-linux_aarch64.whl": {
        "url": "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl",
        "sha256": "62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc",
        "status": "rejected on this board: undefined cusolverDnXsyevBatched_bufferSize in CUDA linalg",
    },
    "gtsam-835f791f72dc7710dac5a453e2d60ec60d705659.tar.gz": {
        "url": "https://codeload.github.com/borglab/gtsam/tar.gz/835f791f72dc7710dac5a453e2d60ec60d705659",
        "git_commit": "835f791f72dc7710dac5a453e2d60ec60d705659",
        "sha256": "7df5b28d6c0e4a434f336d41e5860389e23896c27a901e15fdd9b59315f003ff",
        "version": "4.3a2",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=ROOT / "downloads/hil_wheels")
    p.add_argument("--only", choices=("torch", "gtsam"))
    p.add_argument("--official-torch-only", action="store_true")
    p.add_argument("--include-rejected", action="store_true", help="diagnostics only, never install rejected wheels")
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, metadata in INPUTS.items():
        if metadata.get("status", "").startswith("rejected") and not args.include_rejected:
            continue
        if args.official_torch_only and not name.startswith("torch-2.5.0a0"):
            continue
        if args.only and not name.startswith(args.only + "-"):
            continue
        target = args.directory / name
        if not target.exists():
            partial = target.with_suffix(target.suffix + f".part.{os.getpid()}")
            print("Downloading", name, flush=True)
            with urllib.request.urlopen(metadata["url"], timeout=60) as source, partial.open("wb") as dest:
                while block := source.read(4 * 1024 * 1024):
                    dest.write(block)
            if metadata.get("sha256") and sha256(partial) != metadata["sha256"]:
                raise ValueError(f"Checksum mismatch: {partial}")
            partial.rename(target)
        digest = sha256(target)
        if metadata.get("sha256") and digest != metadata["sha256"]:
            raise ValueError(f"Existing file checksum mismatch: {target}")
        record = dict(metadata, sha256=digest, bytes=target.stat().st_size)
        if target.suffix == ".whl":
            with zipfile.ZipFile(target) as wheel:
                filename = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
                record["requires_dist"] = [line[15:] for line in wheel.read(filename).decode().splitlines()
                                           if line.startswith("Requires-Dist: ")]
        manifest[name] = record
        print(json.dumps({name: record}, ensure_ascii=False), flush=True)
        (args.directory / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
