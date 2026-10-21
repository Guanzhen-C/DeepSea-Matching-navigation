#!/usr/bin/env python3
"""Run every recorded raw sensor event through a frozen FULL estimator.

This is deterministic input replay, NOT a real-time network HIL acceptance.
Never open a host_reference file on the Jetson estimator side.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

from standalone.hil.frozen_engine import BASELINES, load_engine
from standalone.hil.protocol import EventOrder, pack
from standalone.hil.recording import LENGTH, read_packets


def verify_bundle(bundle, profile):
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    if manifest["source_commit"] != BASELINES[profile][0] or manifest["profile"] != profile:
        raise ValueError("wrong frozen baseline bundle")
    for relative, record in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or path.stat().st_size != record["bytes"]:
            raise ValueError("invalid bundle entry")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != record["sha256"]:
            raise ValueError(f"changed bundle file: {relative}")
    return manifest


def initialize(bundle, profile, boot):
    if boot["payload"]["baseline_sha"] != BASELINES[profile][0]:
        raise ValueError("sensor recording baseline mismatch")
    # Must be called in a fresh process: never reuse modules from another map profile.
    if "oceansim" in sys.modules:
        raise RuntimeError("fresh process required before loading frozen baseline")
    sys.path.insert(0, str(bundle / "isaacsim"))
    ns, audit = load_engine(bundle / "standalone/large_scale_slam_isaac_demo.py", profile, bundle)
    config = {k: Path(v["path"]) if isinstance(v, dict) and set(v) == {"path"} else v
              for k, v in boot["payload"]["configuration"].items()}
    config.update(bathymetry=bundle / "map", survey_route=bundle / "map/full_map_traversal_route.json",
                  terrain_replay_directory=None)
    args = argparse.Namespace(**config)
    initial = ns["NavigationInitialization"](**boot["payload"]["initial"])
    world = ns["MeasuredBathymetryGrid"].from_path(bundle / "map")
    engine = ns["create_engine"](args, world, initial, boot["payload"]["startup_specific_force"])
    next(engine)
    return engine, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True, choices=BASELINES)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--events", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.bundle = args.bundle.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        p.error("fresh output directory required")
    manifest = verify_bundle(args.bundle, args.profile)
    order = EventOrder()
    events = iter(read_packets(args.events))
    boot = next(events)
    order.accept(boot)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    engine, audit = initialize(args.bundle, args.profile, boot)
    from oceansim.slam import sanitize_for_strict_json
    latencies = []
    counts = {}
    snapshots = 0
    last_stamp = 0.0
    final_diagnostics = None
    # The output has no truth and is safe to copy back to the PC for scoring.
    with gzip.open(args.output / "estimator_outputs.osh.gz", "xb", compresslevel=1) as outputs:
        for event in events:
            order.accept(event)
            begin = time.perf_counter()
            if event["kind"] == "finish":
                try:
                    engine.send(event)
                except StopIteration as stop:
                    final_diagnostics = stop.value
                break
            result = engine.send(event)
            elapsed = time.perf_counter() - begin
            counts[event["kind"]] = counts.get(event["kind"], 0) + 1
            last_stamp = event["timestamp"]
            if result is not None:
                result["input_sequence"] = event["sequence"]
                result["processing_wall_s"] = elapsed
                packet = pack(result)
                outputs.write(LENGTH.pack(len(packet)))
                outputs.write(packet)
                latencies.append(elapsed)
                snapshots += 1
                if snapshots % 250 == 0:
                    print(f"[HIL replay] {last_stamp:.1f} simulated s, {snapshots} full-estimator pings", flush=True)
    if not order.finished:
        raise ValueError("incomplete sensor recording: no finish event")
    # Also ensure there are no trailing duplicate events hidden after finish.
    try:
        next(events)
    except StopIteration:
        pass
    else:
        raise ValueError("trailing sensor events after finish")
    import numpy as np
    elapsed = time.monotonic() - started
    summary = dict(status="completed", mode="raw_sensor_replay_not_network_realtime_HIL",
                   extraction=audit, profile=args.profile, event_counts=counts,
                   simulated_duration_s=last_stamp, wall_duration_s=elapsed,
                   achieved_time_scale=last_stamp / elapsed, output_samples=snapshots,
                   maximum_sonar_processing_s=max(latencies, default=0.),
                   p95_sonar_processing_s=float(np.percentile(latencies, 95)) if latencies else 0.,
                   no_truth_input_after_initial_seed=True)
    diagnostics, _ = sanitize_for_strict_json(final_diagnostics)
    (args.output / "estimator_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    summary, _ = sanitize_for_strict_json(summary)
    (args.output / "replay_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
