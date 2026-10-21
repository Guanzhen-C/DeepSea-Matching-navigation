#!/usr/bin/env python3
"""Profile a finite raw-sensor prefix without changing the frozen estimator.

An intentionally truncated diagnostic is NEVER a completed HIL/navigation run.
Input contains boot/raw sensors only; host truth must not be copied to the board.
"""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time

from standalone.hil.protocol import EventOrder
from standalone.hil.recording import read_packets
from standalone.hil.replay import initialize, verify_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("lake", "sea"), required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=30.)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or not 0 < args.seconds <= 60.:
        p.error("fresh diagnostic output and prefix of at most 60 simulated seconds required")
    verify_bundle(args.bundle.resolve(), args.profile)
    order = EventOrder()
    events = iter(read_packets(args.events))
    boot = next(events)
    order.accept(boot)
    engine, extraction = initialize(args.bundle.resolve(), args.profile, boot)
    args.output.mkdir(parents=True)
    profiler = cProfile.Profile()
    counts = {}
    durations = []
    stamp = 0.
    started = time.perf_counter()
    profiler.enable()
    for event in events:
        if event["kind"] == "finish" or event["timestamp"] > args.seconds + 1e-9:
            break
        order.accept(event)
        begin = time.perf_counter()
        result = engine.send(event)
        counts[event["kind"]] = counts.get(event["kind"], 0) + 1
        if result is not None:
            durations.append(time.perf_counter() - begin)
        stamp = event["timestamp"]
    profiler.disable()
    elapsed = time.perf_counter() - started
    profiler.dump_stats(str(args.output / "engine.pstats"))
    with (args.output / "profile.txt").open("x") as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(65)
    import numpy as np
    report = dict(kind="finite_raw_prefix_profiling_NOT_HIL_acceptance", event_counts=counts,
                  extraction=extraction,
                  last_sensor_timestamp_s=stamp, wall_duration_s=elapsed,
                  sonar_processing_p50_p95_max_s=np.percentile(durations, [50, 95, 100]).tolist(),
                  completed_navigation_run=False, algorithm_parameters_modified=False,
                  no_truth_after_initial_seed=True)
    (args.output / "diagnostic_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    engine.close()


if __name__ == "__main__":
    main()
