#!/usr/bin/env python3
"""Read-only host/cgroup sampling; never stop, restart or alter a HIL task.

Board resources already have the receiver-owned tegrastats recording. This
observer records the host source/sender, including kernel lifetime peaks where
available. Resource observation is not navigation/lifecycle acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def memory_info(path=Path("/proc/meminfo")):
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[key + "_bytes"] = int(value.split()[0]) * 1024
    return values


def cgroup_memory(directory):
    result = {}
    for name in ("current", "peak", "max", "swap.current"):
        path = directory / ("memory." + name)
        if path.exists():
            value = path.read_text().strip()
            result[name + "_bytes"] = None if value == "max" else int(value)
    for name in ("events", "stat"):
        path = directory / ("memory." + name)
        if path.exists():
            result[name] = {key: int(value) for key, value in
                            (line.split() for line in path.read_text().splitlines())
                            if name == "events" or key in {"anon", "file", "shmem"}}
    return result


def unit_sample(unit):
    command = ["systemctl", "--user", "show", unit, "-p", "LoadState", "-p", "ActiveState",
               "-p", "MainPID", "-p", "ExecMainStatus", "-p", "Result", "-p", "ControlGroup"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=5.)
    values = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    group = values.get("ControlGroup")
    if group:
        root = Path("/sys/fs/cgroup")
        path = (root / group.lstrip("/")).resolve()
        if not path.is_relative_to(root):
            raise ValueError("invalid control group path")
        try:
            values["memory"] = cgroup_memory(path)
        except OSError:
            values["memory_unavailable_after_unit_exit"] = True
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--interval-s", type=float, default=5.)
    p.add_argument("--max-seconds", type=float, default=30.*3600.)
    args = p.parse_args()
    if not 1. <= args.interval_s <= 60. or args.max_seconds <= 0.:
        p.error("finite duration and sampling interval between 1 and 60 s required")
    manifest = json.loads((args.run / "launch_manifest.json").read_text())
    units = {role: manifest[key] for role, key in (("source", "producer_unit"), ("sender", "host_unit"))
             if manifest.get(key)}
    start = time.monotonic()
    summaries = {name: dict(maximum_observed_current_bytes=0, maximum_kernel_lifetime_peak_bytes=0,
                           maximum_observed_oom_kills=0) for name in units}
    minimum_available = None
    samples = 0
    reason = "observation_duration_exhausted_jobs_untouched"
    with (args.run / "host_resources.jsonl").open("x", buffering=1) as stream:
        while time.monotonic() - start < args.max_seconds:
            host = memory_info()
            record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                          observation_elapsed_s=time.monotonic()-start, host=host,
                          units={name: unit_sample(unit) for name, unit in units.items()})
            try:
                gpu = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3., check=True)
                record["gpu0_used_MiB_total_MiB_utilization_percent"] = gpu.stdout.strip()
            except (OSError, subprocess.SubprocessError) as exc:
                record["gpu_read_error"] = type(exc).__name__
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            minimum_available = min(minimum_available or host["MemAvailable_bytes"], host["MemAvailable_bytes"])
            for name, values in record["units"].items():
                memory = values.get("memory", {})
                summary = summaries[name]
                summary["maximum_observed_current_bytes"] = max(summary["maximum_observed_current_bytes"], memory.get("current_bytes") or 0)
                summary["maximum_kernel_lifetime_peak_bytes"] = max(summary["maximum_kernel_lifetime_peak_bytes"], memory.get("peak_bytes") or 0)
                summary["maximum_observed_oom_kills"] = max(summary["maximum_observed_oom_kills"], memory.get("events", {}).get("oom_kill", 0))
            samples += 1
            if all(values.get("ActiveState") not in {"active", "activating", "deactivating"}
                   for values in record["units"].values()):
                reason = "host_participants_terminal_no_task_was_stopped"
                break
            time.sleep(args.interval_s)
    report = dict(kind="read_only_resource_observation_not_navigation_acceptance", samples=samples,
                  stop_reason=reason, units=units, memory=summaries,
                  minimum_observed_host_available_bytes=minimum_available,
                  earlier_GPU_timeline_not_reconstructed=True, task_state_modified=False)
    (args.run / "host_resource_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
