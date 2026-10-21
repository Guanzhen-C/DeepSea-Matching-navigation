#!/usr/bin/env python3
"""Admit and start one persistent full-estimator Jetson raw-replay test.

Refuses running builds, reused outputs, dirty implementation, or another HIL
estimator. Does not stop jobs, change power mode, or alter either baseline.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/media/cgz/data/oceansim_hil_20261021"
SSH = ["ssh", "-i", "/home/cgz/.ssh/oceansim_jetson_hil_20261021", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=10", "cgz@192.168.77.2"]


def remote(args):
    return subprocess.check_output([*SSH, shlex.join(args)], text=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("lake", "sea"), required=True)
    p.add_argument("--mode", choices=("lockstep", "realtime"), required=True)
    p.add_argument("--recording", type=Path, required=True, help="completed native capture directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.output = args.output.resolve()
    args.recording = args.recording.resolve()
    if args.output.exists():
        p.error("fresh output directory required")
    record = json.loads((args.recording / "recording_summary.json").read_text())
    if record["status"] != "completed" or record["profile"] != args.profile:
        p.error("wrong or incomplete native sensor recording")
    if subprocess.check_output(["git", "diff", "HEAD", "--stat"], cwd=ROOT, text=True).strip():
        p.error("commit tracked implementation before a reproducible board test")
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:") and int(line.split()[1]) < 6 * 1024**2:
            p.error("need >=6 GiB available host RAM for ROS transport test")
    running = remote(["systemctl", "--user", "list-units", "--no-legend", "--state=running", "oceansim-hil-*"])
    if running.strip():
        p.error("another board HIL/build unit is running; it was not stopped:\n" + running)
    # Verify the exact Python environment and real native backends before launch.
    code_sha256 = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted((ROOT / "standalone/hil").glob("*")) if f.is_file()}
    code_check = remote([REMOTE + "/venv/bin/python", "-c",
        "import hashlib,json,pathlib; r=pathlib.Path(" + repr(REMOTE + "/code") + "); "
        "names=" + repr(list(code_sha256)) + "; "
        "print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names}))"])
    if json.loads(code_check) != code_sha256:
        p.error("board HIL interface code differs from this committed implementation; synchronize before testing")
    preflight = remote([REMOTE + "/venv/bin/python", "-c",
        "import gtsam,gtsam_unstable,torch; "
        "assert hasattr(gtsam_unstable,'IncrementalFixedLagSmoother'); "
        "assert torch.cuda.is_available(); "
        "a=torch.eye(3,dtype=torch.float64,device='cuda'); "
        "torch.linalg.solve_ex(a,a); print('native GTSAM fixed-lag + CUDA solve ready')"])
    label = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    unit = f"oceansim-hil-{args.profile}-{args.mode}-{label}"
    board_output = f"{REMOTE}/results/{unit}"
    board = ["systemd-run", "--user", "--unit=" + unit,
             "--working-directory=" + REMOTE + "/code", "--property=MemoryHigh=10G",
             "--property=MemoryMax=10G", "--property=MemorySwapMax=0", "--property=CPUQuota=600%",
             "--property=Nice=5", "--property=StandardOutput=append:" + board_output + ".log",
             "--property=StandardError=append:" + board_output + ".log", "/bin/bash",
             REMOTE + "/code/standalone/hil/run_ros.sh", "--side", "jetson", "--role", "receiver",
             "--profile", args.profile, "--bundle", REMOTE + "/bundles/" + args.profile,
             "--output", board_output]
    host = ["systemd-run", "--user", "--unit=" + unit + "-sender", "--working-directory=" + str(ROOT),
            "--property=MemoryHigh=3G", "--property=MemoryMax=3G", "--property=MemorySwapMax=0",
            "--property=CPUQuota=200%", "--property=Nice=5",
            "--property=StandardOutput=append:" + str(args.output / "sender.log"),
            "--property=StandardError=append:" + str(args.output / "sender.log"),
            "/bin/bash", str(ROOT / "standalone/hil/run_ros.sh"), "--side", "pc", "--role", "sender",
            "--mode", args.mode, "--events", str(args.recording / "sensor_events.osh.gz"),
            "--output", str(args.output / "sender")]
    manifest = dict(profile=args.profile, mode=args.mode, board_unit=unit + ".service",
                    host_unit=unit + "-sender.service", board_output=board_output,
                    recording=str(args.recording), host_output=str(args.output),
                    implementation_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    board_preflight=preflight.strip(), source_truth_is_host_only=True,
                    code_sha256=code_sha256,
                    board_argv=board, host_argv=host)
    if not args.dry_run:
        args.output.mkdir(parents=True)
        (args.output / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        remote(["mkdir", "-p", REMOTE + "/results"])
        print(remote(board), flush=True)
        subprocess.run(host, check=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
