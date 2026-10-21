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
import shutil
import subprocess
import time

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
    input_mode = p.add_mutually_exclusive_group(required=True)
    input_mode.add_argument("--recording", type=Path, help="completed native capture directory")
    input_mode.add_argument("--live-bundle", type=Path, help="live Isaac source, original frozen bundle on the PC")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--distance-m", type=float)
    scope.add_argument("--full-route", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.recording:
        args.recording = args.recording.resolve()
    if args.live_bundle:
        args.live_bundle = args.live_bundle.resolve()
    if args.output.exists():
        p.error("fresh output directory required")
    if args.recording:
        record = json.loads((args.recording / "recording_summary.json").read_text())
        metrics = json.loads((args.recording / "native_metrics.json").read_text())
        if record["status"] != "completed" or metrics["run_status"] != "completed" or record["profile"] != args.profile:
            p.error("wrong or incomplete native sensor recording")
    else:
        from standalone.hil.replay import verify_bundle
        verify_bundle(args.live_bundle, args.profile)
        if not args.full_route and not (args.distance_m is not None and 0 < args.distance_m <= 5000):
            p.error("live source requires explicit --distance-m (<=5000) or --full-route")
        if subprocess.run(["pgrep", "-f", "[p]ython.*(standalone.hil.capture_native|/large_scale_slam_isaac_demo.py)"],
                          capture_output=True).returncode == 0:
            p.error("another OceanSim source/reference task is running; it was not stopped")
        gpu_free = int(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                                               "--format=csv,noheader,nounits"], text=True).strip())
        if gpu_free < 2048:
            p.error("live source requires >=2048 MiB free GPU memory")
        if shutil.disk_usage(ROOT).free < (150 if args.full_route else 10) * 1024**3:
            p.error("insufficient free host disk for lossless sensor recording")
    if subprocess.check_output(["git", "diff", "HEAD", "--stat"], cwd=ROOT, text=True).strip():
        p.error("commit tracked implementation before a reproducible board test")
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:") and int(line.split()[1]) < (12 if args.live_bundle else 6) * 1024**2:
            p.error("insufficient available host RAM for bounded HIL admission")
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
        "assert hasattr(gtsam,'IncrementalFixedLagSmoother') or hasattr(gtsam_unstable,'IncrementalFixedLagSmoother'); "
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
            "--mode", args.mode, "--output", str(args.output / "sender")]
    producer = None
    endpoint = None
    if args.live_bundle:
        endpoint = Path(f"/run/user/{os.getuid()}/oceansim-hil-{label}.sock")
        if endpoint.exists():
            p.error("live IPC endpoint unexpectedly exists")
        host += ["--live-socket", str(endpoint)]
        producer = ["systemd-run", "--user", "--unit=" + unit + "-producer", "--working-directory=" + str(ROOT),
            "--property=MemoryHigh=7G", "--property=MemoryMax=7G", "--property=MemorySwapMax=0",
            "--property=CPUQuota=200%", "--property=Nice=10",
            "--property=StandardOutput=append:" + str(args.output / "native_reference.log"),
            "--property=StandardError=append:" + str(args.output / "native_reference.log"),
            "/usr/bin/flock", "--nonblock", f"/run/user/{os.getuid()}/oceansim-chelan-slam.lock",
            "/usr/bin/env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "OCEANSIM_HIL_SOCKET=" + str(endpoint),
            "OCEANSIM_BBS3D_LIBRARY=" + str(ROOT / "native/bbs3d_bridge/build/liboceansim_bbs3d.so"),
            "PYTHONPATH=" + str(ROOT), "/home/cgz/.conda/envs/isaacsim-6.0/bin/python",
            "-m", "standalone.hil.capture_native", "--profile", args.profile,
            "--bundle", str(args.live_bundle), "--output", str(args.output / "native_reference")]
        producer += ["--full-route"] if args.full_route else ["--distance-m", str(args.distance_m)]
    else:
        host += ["--events", str(args.recording / "sensor_events.osh.gz")]
    manifest = dict(profile=args.profile, mode=args.mode, board_unit=unit + ".service",
                    host_unit=unit + "-sender.service", board_output=board_output,
                    recording=str(args.recording) if args.recording else None, host_output=str(args.output),
                    live_Isaac=bool(args.live_bundle), producer_unit=unit + "-producer.service" if producer else None,
                    implementation_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    board_preflight=preflight.strip(), source_truth_is_host_only=True,
                    code_sha256=code_sha256,
                    board_argv=board, host_argv=host, producer_argv=producer)
    if not args.dry_run:
        args.output.mkdir(parents=True)
        (args.output / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        remote(["mkdir", "-p", REMOTE + "/results"])
        print(remote(board), flush=True)
        subprocess.run(host, check=True)
        if producer:
            deadline = time.monotonic() + 30.
            while not endpoint.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("ROS peer/IPC not ready; producer was not launched; inspect existing units")
                time.sleep(.1)
            subprocess.run(producer, check=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
