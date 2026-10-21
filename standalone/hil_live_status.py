#!/usr/bin/env python3
"""Read real board progress and score its EXACT sensor epoch on the PC.

Do not confuse the original PC reference's progress log with Jetson results.
This is one timestamp, not cumulative RMSE/P95 or mission qualification. Host
truth is never transmitted to the board, interpolated or spatially aligned.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from standalone.hil.launch_validation import remote
from standalone.hil.recording import read_packets


def evaluate_instant(reference_records, progress):
    target = float(progress["timestamp"])
    if not np.isfinite(target):
        raise ValueError("nonfinite board sensor timestamp")
    distance = 0.
    previous = None
    found = None
    try:
        for record in reference_records:
            truth = np.asarray(record["truth_position"], dtype=np.float64)
            if truth.shape != (3,) or not np.isfinite(truth).all():
                raise ValueError("invalid host truth position")
            if previous is not None:
                distance += float(np.linalg.norm(truth - previous))
            previous = truth
            if abs(record["timestamp"] - target) <= 1e-9:
                found = truth
                break
            if record["timestamp"] > target + 1e-9:
                break
    except EOFError:
        pass  # Expected while the live gzip member has not been finalized.
    if found is None:
        return dict(awaiting_exact_truth_timestamp=True, estimate_sensor_time_s=target,
                    instantaneous_errors_3d_m=None, full_route_acceptance=False)
    errors = {}
    for name in ("dr", "local", "global"):
        position = np.asarray(progress[name + "_position"], dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("invalid board position")
        errors[name] = float(np.linalg.norm(position - found))
    return dict(estimate_sensor_time_s=target, processed_sonar_pings=progress["ping_count"],
                estimated_truth_arc_distance_m=distance, instantaneous_errors_3d_m=errors,
                awaiting_exact_truth_timestamp=False, full_route_acceptance=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, help="optional mutable live snapshot, not final experiment results")
    args = p.parse_args()
    manifest = json.loads((args.run / "launch_manifest.json").read_text())
    board_state = dict(line.split("=", 1) for line in remote(["systemctl", "--user", "show",
        manifest["board_unit"], "-p", "ActiveState", "-p", "MainPID", "-p", "ExecMainStatus"]).splitlines() if "=" in line)
    progress = json.loads(remote(["cat", manifest["board_output"] + "/progress.json"]))
    source = Path(manifest["recording"]) if manifest.get("recording") else args.run / "native_reference"
    report = evaluate_instant(read_packets(source / "host_reference.osh.gz"), progress)
    local_states = {}
    for key in ("host_unit", "producer_unit"):
        if manifest.get(key):
            result = subprocess.check_output(["systemctl", "--user", "show", manifest[key],
                "-p", "ActiveState", "-p", "MainPID", "-p", "ExecMainStatus"], text=True)
            local_states[key] = dict(line.split("=", 1) for line in result.splitlines() if "=" in line)
    report.update(kind="instantaneous_board_epoch_health_not_final_statistics", profile=manifest["profile"],
                  recorded_utc=datetime.now(timezone.utc).isoformat(), board_unit=manifest["board_unit"],
                  actual_board_state=board_state, actual_host_states=local_states,
                  truth_evaluated_only_on_PC=True, interpolation_or_spatial_alignment=False,
                  progress_snapshot_can_be_older_than_current_simulation=True)
    if args.output:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
