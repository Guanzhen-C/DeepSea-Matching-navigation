#!/usr/bin/env python3
"""Finite HIL qualification pipeline; wait for the EXISTING build, never restart it.

Only short, recorded-input tests are queued here. Full missions and live Isaac
qualification remain separate steps, after examining these measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from standalone.hil.launch_validation import REMOTE, ROOT, SSH, remote
from standalone.hil.score_board_results import score

BUILD_UNIT = "oceansim-hil-build-gtsam-serial-20261021.service"
RECORDINGS = {"lake": ROOT / "demo/hil/jetson_20261021/lake_native_600m_v3",
              "sea": ROOT / "demo/hil/jetson_20261021/sea_native_600m"}


def state(unit, on_board=False):
    argv = ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "Result", "-p", "ExecMainStatus"]
    text = remote(argv) if on_board else subprocess.check_output(argv, text=True)
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def wait_finished(unit, on_board, deadline):
    while True:
        result = state(unit, on_board)
        if result.get("ActiveState") not in ("active", "activating", "deactivating"):
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"still running: {unit}; not stopped/restarted; inspect this SAME unit")
        time.sleep(5.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists():
        p.error("fresh pipeline output directory required")
    output.mkdir(parents=True)
    status = dict(status="waiting_for_existing_native_build", build_unit=BUILD_UNIT, stages=[],
                  full_route_HIL_complete=False, live_Isaac_HIL_complete=False,
                  implementation_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())

    def save():
        status["updated_at"] = datetime.now().astimezone().isoformat()
        temporary = output / "pipeline_status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "pipeline_status.json")

    try:
        save()
        build_state = wait_finished(BUILD_UNIT, True, time.monotonic() + 4 * 3600)
        status["build_unit_terminal_state"] = build_state
        # The original active script has an obsolete final unstable-only
        # assertion. Test the REAL installed stable-or-unstable API instead of
        # launching a needless rebuild if installation otherwise succeeded.
        preflight = remote(["env", "LD_LIBRARY_PATH=" + REMOTE + "/gtsam/lib",
            REMOTE + "/venv/bin/python", "-c",
            "import gtsam,gtsam_unstable,torch; "
            "assert hasattr(gtsam,'IncrementalFixedLagSmoother') or hasattr(gtsam_unstable,'IncrementalFixedLagSmoother'); "
            "assert torch.cuda.is_available(); print('native fixed-lag and CUDA ready')"])
        status["installed_native_preflight"] = preflight.strip()
        status["status"] = "testing_original_GTSAM_backend"
        save()
        test_command = ["env", "PYTHONPATH=" + REMOTE + "/bundles/lake/isaacsim:" + REMOTE + "/code",
            "LD_LIBRARY_PATH=" + REMOTE + "/gtsam/lib", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
            REMOTE + "/venv/bin/python", "-m", "pytest",
            REMOTE + "/code/tests/test_gtsam_pose_graph_backend.py", "-q",
            "--junitxml=" + REMOTE + "/gtsam_backend_tests.xml"]
        with (output / "board_GTSAM_tests.log").open("x") as log:
            subprocess.run([*SSH, shlex.join(test_command)], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["scp", "-i", "/home/cgz/.ssh/oceansim_jetson_hil_20261021",
                        "cgz@192.168.77.2:" + REMOTE + "/gtsam_backend_tests.xml", str(output)], check=True)
        import xml.etree.ElementTree as ET
        junit = ET.parse(output / "gtsam_backend_tests.xml").getroot()
        cases = junit.findall(".//testcase")
        if not cases or any(c.find("skipped") is not None or c.find("failure") is not None or c.find("error") is not None for c in cases):
            raise RuntimeError("GTSAM qualification must execute actual tests without skips/failures")
        status["original_GTSAM_tests_passed"] = len(cases)
        for mode in ("lockstep", "realtime"):
            for profile in ("lake", "sea"):
                name = profile + "_600m_" + mode
                directory = output / name
                status["status"] = "running_" + name
                save()
                subprocess.run([sys.executable, "-m", "standalone.hil.launch_validation",
                    "--profile", profile, "--mode", mode, "--recording", str(RECORDINGS[profile]),
                    "--output", str(directory)], cwd=ROOT, check=True)
                manifest = json.loads((directory / "launch_manifest.json").read_text())
                deadline = time.monotonic() + 2 * 3600
                host_state = wait_finished(manifest["host_unit"], False, deadline)
                board_state = wait_finished(manifest["board_unit"], True, deadline)
                if host_state.get("Result") != "success" or board_state.get("Result") != "success":
                    raise RuntimeError(f"short HIL failed: host={host_state}, board={board_state}; do not retry silently")
                sender = json.loads((directory / "sender/sender_summary.json").read_text())
                reference_summary = json.loads((RECORDINGS[profile] / "recording_summary.json").read_text())
                if sender["events"] != reference_summary["events"] or sender["samples"] != reference_summary["samples"] or sender["skipped_events"] != 0:
                    raise RuntimeError("HIL event/sample loss")
                board_local = directory / "board_artifacts"
                board_local.mkdir()
                subprocess.run(["rsync", "-rt", "-e", shlex.join(SSH[:-1]),
                    "cgz@192.168.77.2:" + manifest["board_output"] + "/", str(board_local) + "/"], check=True)
                measured = score(RECORDINGS[profile] / "host_reference.osh.gz",
                                 directory / "sender/returned_estimator_outputs.osh.gz")
                (directory / "cross_architecture_metrics.json").write_text(json.dumps(measured, indent=2) + "\n")
                stage = dict(name=name, sender=sender, metrics=measured,
                             board_unit=manifest["board_unit"], host_unit=manifest["host_unit"])
                status["stages"].append(stage)
                # Qualification rule preregistered before board measurements;
                # never change navigation thresholds to make this gate pass.
                bad = [(label, metric, change) for label, values in measured["RMSE_P95_relative_changes"].items()
                       for metric, change in values.items() if change > .10]
                if bad:
                    status["regression_requires_review"] = bad
                    raise RuntimeError("cross-architecture RMSE/P95 regression exceeds the proposed 10% allowance; inspect, do not tune silently")
                save()
        status["status"] = "short_recorded_input_qualification_completed"
        save()
    except Exception as exc:
        status["status"] = "needs_review"
        status["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
