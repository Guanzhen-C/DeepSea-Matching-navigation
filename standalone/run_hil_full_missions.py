#!/usr/bin/env python3
"""Finite, fail-closed HIL evidence collection and sequential full missions.

Observe an EXISTING pilot first. Never stop/restart any participant, retry a
failed run, relax a gate or modify a baseline. New missions require the explicit
--run-full-missions flag and the ordinary resource/source admission checks.
All truth stays on the PC; this coordinator is never deployed to the Jetson.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

from standalone.analyze_hil_relative_motion import analyze, file_sha256
from standalone.hil.after_build_validation import state
from standalone.hil.frozen_engine import BASELINES
from standalone.hil.launch_validation import ROOT, SSH
from standalone.hil.recording import read_packets
from standalone.hil.score_board_results import score
from standalone.plot_hil_results import load_run

ACTIVE = frozenset({"active", "activating", "deactivating", "reloading"})
CHANNELS = ("dr", "local", "global")


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_returned_poses(original, returned):
    """Network results must equal the ACTUAL board log, not a PC estimator."""
    count = 0
    for board, received in itertools.zip_longest(read_packets(original), read_packets(returned)):
        if board is None or received is None or board["timestamp"] != received["timestamp"]:
            raise ValueError("board/returned output count or sensor timestamp mismatch")
        for name in CHANNELS:
            for suffix in ("position", "rotation"):
                field = name + "_" + suffix
                if not np.array_equal(board[field], received[field]):
                    raise ValueError("returned pose differs from actual board output: " + field)
        if board["input_sequence"] != received["input_sequence"]:
            raise ValueError("returned output refers to a different input event")
        count += 1
    if not count:
        raise ValueError("empty board output")
    return dict(samples=count, six_pose_channels_exactly_equal=True, comparison_to_PC_estimator=False)


def accuracy_regressions(measured):
    # Same proposed review rule as after_build_validation, not an estimator
    # threshold. Do not replace the original PC with the new canonical variant.
    return [dict(channel=channel, metric=metric, relative_change=float(change))
            for channel, values in measured["RMSE_P95_relative_changes"].items()
            for metric, change in values.items() if not np.isfinite(change) or change > .10]


def verify_declaration(manifest, receiver, extraction, *, full, native):
    profile = manifest["profile"]
    if (not manifest["live_Isaac"] or not manifest["source_truth_is_host_only"]
            or manifest["deployment_layout"] != "packed-and-feature-batch"
            or not manifest["batch_intensity_jacobians"] or not manifest["canonical_correspondences"]):
        raise ValueError("not the explicitly qualified live deployment variant")
    if receiver["input_truth_after_seed"]:
        raise ValueError("ongoing truth must not enter board navigation")
    for key in ("deployment_layout", "batch_intensity_jacobians", "canonical_correspondences"):
        if receiver[key] != manifest[key]:
            raise ValueError("board declaration differs from launch manifest")
    if extraction["profile"] != profile or extraction["baseline_sha"] != BASELINES[profile][0]:
        raise ValueError("wrong board frozen baseline")
    if full and (not native["full_map_completed"] or native["requested_survey_segment_distance_m"] != 0.):
        raise ValueError("segment completion cannot replace a full requested route")


def launch_command(profile, bundle, output, *, distance_m=None):
    return [sys.executable, "-m", "standalone.hil.launch_validation", "--profile", profile,
            "--mode", "realtime", "--live-bundle", str(bundle),
            *(["--full-route"] if distance_m is None else ["--distance-m", str(distance_m)]),
            "--deployment-layout", "packed-and-feature-batch", "--batch-intensity-jacobians",
            "--canonical-correspondences", "--output", str(output)]


def verify_qualified_implementation(manifest, root=ROOT):
    if not manifest.get("code_sha256"):
        raise ValueError("qualified implementation hashes are missing")
    for name, expected in manifest["code_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != expected:
            raise ValueError("qualified deployment implementation changed: " + name)


def wait_existing(directory, timeout_s, save):
    manifest = json.loads((directory / "launch_manifest.json").read_text())
    units = {"board": (manifest["board_unit"], True), "sender": (manifest["host_unit"], False)}
    for role, key in (("source", "producer_unit"), ("resources", "host_resource_unit")):
        if manifest.get(key):
            units[role] = (manifest[key], False)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            states = {name: state(unit, board) for name, (unit, board) in units.items()}
        except (OSError, subprocess.SubprocessError) as exc:
            save(dict(observation_error=f"{type(exc).__name__}: {exc}",
                      existing_tasks_not_known_terminal=True))
            if time.monotonic() >= deadline:
                raise TimeoutError("observation budget expired during polling; jobs remain untouched") from exc
            time.sleep(10.)
            continue  # Re-poll the SAME handles, never restart after an SSH failure.
        save(states)
        if any(s.get("ActiveState") is None for s in states.values()):
            if time.monotonic() >= deadline:
                raise TimeoutError("authoritative state unavailable; jobs remain untouched")
            time.sleep(10.)
            continue
        if all(s["ActiveState"] not in ACTIVE for s in states.values()):
            if any(s.get("Result") not in (None, "success") or s.get("ExecMainStatus") not in (None, "0")
                   or s["ActiveState"] == "failed" for s in states.values()):
                raise RuntimeError("a participant failed; queued runs were not started")
            # A collected transient unit may be gone. Its apparent default
            # Result=success is NOT independent proof of an observed exit code.
            return states
        if time.monotonic() >= deadline:
            raise TimeoutError("observation budget expired; existing jobs remain untouched")
        time.sleep(10.)


def finalize(directory, output, terrain, *, full=False, terminal_states=None):
    """Only completed files; no process state changes or navigation changes."""
    output.mkdir()
    manifest = json.loads((directory / "launch_manifest.json").read_text())
    artifacts = directory / "board_artifacts"
    if not artifacts.exists():
        artifacts.mkdir()
        subprocess.run(["rsync", "-rt", "-e", shlex.join(SSH[:-1]),
                        SSH[-1] + ":" + manifest["board_output"] + "/", str(artifacts) + "/"], check=True)
        write_new(output / "collection_origin.json", dict(board_output=manifest["board_output"],
            launch_manifest_sha256=file_sha256(directory / "launch_manifest.json")))
    run = load_run(directory, require_full_route=full, include_rotations=True)
    native = json.loads((run["source"] / "native_metrics.json").read_text())
    receiver = json.loads((artifacts / "receiver_summary.json").read_text())
    extraction = json.loads((artifacts / "extraction_audit.json").read_text())
    verify_declaration(manifest, receiver, extraction, full=full, native=native)
    returned = check_returned_poses(artifacts / "estimator_outputs.osh.gz", run["outputs"])
    if returned["samples"] != len(run["timestamps"]):
        raise ValueError("board output count differs from completed input trajectory")
    resource = json.loads((directory / "host_resource_summary.json").read_text())
    if resource["stop_reason"] != "host_participants_terminal_no_task_was_stopped":
        raise ValueError("host resource observation did not cover completion")
    if any(v["maximum_observed_oom_kills"] for v in resource["memory"].values()):
        raise ValueError("observed host OOM kill requires review")
    measured = score(run["reference"], run["outputs"])
    write_new(output / "common_truth_metrics.json", measured)
    relative = analyze(run)
    relative.update(profile=manifest["profile"], full_route=run["full_route"],
                    samples=len(run["timestamps"]), distance_m=float(run["distance"][-1]))
    write_new(output / "relative_motion_audit.json", relative)
    bad = accuracy_regressions(measured)
    evidence = dict(status="needs_review" if bad else "completed_integrity_and_accuracy_review_passed",
        profile=manifest["profile"], full_route=run["full_route"],
        live_Isaac=True, distance_m=float(run["distance"][-1]), samples=len(run["timestamps"]),
        original_PC_reference_retained=True, returned_board_equality=returned,
        input_counts=receiver["events"], latency_and_duration=run["sender"],
        host_resources=resource, terminal_state_snapshot=terminal_states,
        terminal_snapshot_alone_does_not_prove_collected_transient_unit_exit_code=True,
        accuracy_review_limit_relative_change=.10, accuracy_regressions=bad,
        truth_after_initial_seed_sent_to_board=False, physical_sensor_trial=False,
        controller_closed_loop=False, hard_realtime=False, final_acceptance=False)
    write_new(output / "completion_evidence.json", evidence)
    # Analysis is CPU-only, and the previous producer has already terminated.
    subprocess.run([sys.executable, "-m", "standalone.plot_hil_results", "--run", str(directory),
                    "--output", str(output / "figures"), "--terrain", str(terrain),
                    *(["--require-full-route"] if full else [])], cwd=ROOT, check=True)
    if bad:
        raise RuntimeError("original RMSE/P95 review limit exceeded; see completion_evidence.json; no threshold was changed")
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True, help="existing sea live 5 km run; NEVER restarted")
    p.add_argument("--lake-short", type=Path, default=ROOT / "demo/hil/jetson_20261021/lake_live_600m_canonical_v3")
    p.add_argument("--lake-bundle", type=Path, default=ROOT / "downloads/hil_bundles/lake-v2")
    p.add_argument("--sea-bundle", type=Path, default=ROOT / "downloads/hil_bundles/sea-v2")
    p.add_argument("--output", type=Path, required=True, help="fresh pipeline directory")
    p.add_argument("--run-full-missions", action="store_true", help="explicitly authorize the two sequential full runs")
    args = p.parse_args()
    for name in ("pilot", "lake_short", "lake_bundle", "sea_bundle", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        p.error("fresh pipeline directory required; never resume by re-launching a previous stage")
    pilot_manifest = json.loads((args.pilot / "launch_manifest.json").read_text())
    if pilot_manifest["profile"] != "sea" or not pilot_manifest["live_Isaac"]:
        p.error("existing sea live pilot required")
    verify_qualified_implementation(pilot_manifest)
    args.output.mkdir(parents=True)
    status = dict(status="observing_existing_sea_pilot", pilot=str(args.pilot), stages=[],
        full_missions_explicitly_requested=args.run_full_missions,
        implementation_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        coordinator_source_sha256=file_sha256(Path(__file__).resolve()), final_acceptance=False)

    def save(states=None):
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        if states is not None:
            status["observed_participant_states"] = states
        temporary = args.output / "pipeline_status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "pipeline_status.json")

    try:
        save()
        states = wait_existing(args.pilot, 3.*3600., save)
        status["status"] = "reviewing_completed_sea_pilot"
        save()
        evidence = finalize(args.pilot, args.output / "pilot_review", args.sea_bundle / "map", terminal_states=states)
        if not 4990. <= evidence["distance_m"] <= 5010. or evidence["full_route"]:
            raise ValueError("the pilot must be a completed 5 km segment, not a different scope")
        status["stages"].append(dict(name="sea_5km_live_pilot", evidence=evidence))
        status["status"] = "reviewing_existing_lake_short"
        save()
        # Reuse completed short evidence; this never launches a second pilot.
        lake_states = wait_existing(args.lake_short, 60., save)
        evidence = finalize(args.lake_short, args.output / "lake_short_review", args.lake_bundle / "map", terminal_states=lake_states)
        status["stages"].append(dict(name="lake_600m_live_pilot", evidence=evidence))
        save()
        if args.run_full_missions:
            # Lake has a substantially larger prepared map. Qualify its own
            # 5 km source/resource chain rather than extrapolating sea memory
            # behavior or relabelling the first 5 km of a full run as a pilot.
            stages = [("lake", args.lake_bundle, 5000.),
                      ("lake", args.lake_bundle, None), ("sea", args.sea_bundle, None)]
            for profile, bundle, distance_m in stages:
                full = distance_m is None
                name = profile + ("_full_route_live" if full else "_5km_live_pilot")
                directory = args.output / name
                status["status"] = "admitting_" + name
                command = launch_command(profile, bundle, directory, distance_m=distance_m)
                status["next_launch_argv"] = command
                save()
                verify_qualified_implementation(pilot_manifest)
                # Existing launcher rechecks clean Git, exact board code, no
                # running estimator, original source/map hashes, RAM/disk/GPU.
                subprocess.run(command, cwd=ROOT, check=True)
                status["status"] = "running_" + name
                status["active_run"] = str(directory)
                save()
                states = wait_existing(directory, (24. if full else 3.)*3600., save)
                status["status"] = "reviewing_" + name
                save()
                evidence = finalize(directory, args.output / (name + "_review"), bundle / "map", full=full, terminal_states=states)
                if not full and not 4990. <= evidence["distance_m"] <= 5010.:
                    raise ValueError("lake pilot did not complete its requested 5 km")
                status["stages"].append(dict(name=name, evidence=evidence))
                save()
            status["status"] = "both_full_missions_completed_pending_final_review"
        else:
            status["status"] = "pilot_reviews_completed_no_full_mission_requested"
        status.pop("active_run", None)
        save()
    except Exception as exc:
        status["status"] = "needs_review_existing_tasks_untouched"
        status["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
