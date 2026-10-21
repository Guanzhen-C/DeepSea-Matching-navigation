#!/usr/bin/env python3
"""Write a non-secret deployment inventory; distinguish missing backends explicitly."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-full-backends", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("fresh inventory path required")
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), architecture=platform.machine(),
                  kernel=platform.release(), python=sys.version, executable=sys.executable,
                  packages={}, backends={}, storage={})
    for name in ("numpy", "scipy", "opencv-python", "torch", "gtsam", "pybind11"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    for name in ("gtsam", "gtsam_unstable", "torch", "cv2"):
        try:
            module = importlib.import_module(name)
            record = dict(available=True, path=module.__file__)
            if name in ("gtsam", "gtsam_unstable"):
                record["IncrementalFixedLagSmoother"] = hasattr(module, "IncrementalFixedLagSmoother")
            if name == "torch":
                record["cuda_available"] = module.cuda.is_available()
                record["cuda_runtime"] = module.version.cuda
                if module.cuda.is_available():
                    record["device"] = module.cuda.get_device_name(0)
                    record["capability"] = list(module.cuda.get_device_capability(0))
                    tensor = module.eye(3, dtype=module.float64, device="cuda")
                    record["float64_solve"] = module.linalg.solve_ex(tensor, tensor).result.cpu().numpy().tolist()
            result["backends"][name] = record
        except Exception as exc:
            result["backends"][name] = dict(available=False, error=f"{type(exc).__name__}: {exc}")
    for path in (Path("/"), Path("/media/cgz/data")):
        if path.exists():
            space = shutil.disk_usage(path)
            result["storage"][str(path)] = dict(total_bytes=space.total, free_bytes=space.free)
    if Path("/etc/nv_tegra_release").exists():
        result["L4T"] = Path("/etc/nv_tegra_release").read_text().splitlines()[0]
    result["full_native_runtime_ready"] = (
        result["backends"]["gtsam"].get("available", False)
        and (result["backends"]["gtsam"].get("IncrementalFixedLagSmoother", False)
             or result["backends"]["gtsam_unstable"].get("IncrementalFixedLagSmoother", False))
        and result["backends"]["torch"].get("cuda_available", False)
        and "float64_solve" in result["backends"]["torch"]
        and result["backends"]["cv2"].get("available", False))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.require_full_backends and not result["full_native_runtime_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
