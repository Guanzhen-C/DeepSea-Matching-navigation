#!/usr/bin/env python3
"""Record original Isaac-generated sensors AND native reference, without baseline edits.

Run via the existing Isaac Python under the normal 7 GiB host guard. This is a
correctness capture, not yet a network HIL or real-time performance result.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import shlex
import sys

from standalone.hil.frozen_engine import BASELINES, _assigned, source_tree
from standalone.hil.recording import Recorder

ROOT = Path(__file__).resolve().parents[2]


def launcher_argv(bundle, profile, output, distance):
    text = (bundle / "standalone/run_v2_full_map_slam_isaac.sh").read_text()
    text = text[text.index('exec "${SCRIPT_DIR}/run_large_scale_slam_isaac.sh"'):]
    parts = shlex.split(text.replace("\\\n", " "))
    argv = [p.replace("${SCRIPT_DIR}", str(ROOT / "standalone")) for p in parts[2:] if p != "$@"]
    argv += ["--bathymetry", str(bundle / "map"), "--survey-route", str(bundle / "map/full_map_traversal_route.json"),
             "--headless", "--duration", "0", "--metrics", str(output / "native_metrics.json"),
             "--survey-segment-distance-m", str(distance)]
    if profile == "lake":
        argv += ["--survey-laps", "1.0", "--survey-start-distance-m", "0",
                 "--terrain-merge-converged-cohorts", "--terrain-raw-mode-recovery"]
    return argv


class Instrument(ast.NodeTransformer):
    def __init__(self, bundle):
        self.bundle = bundle
        self.hooks = []
        self.in_main = False
        self.record_events = False

    def visit_FunctionDef(self, node):
        previous = self.in_main
        self.in_main = node.name == "main"
        node = self.generic_visit(node)
        self.in_main = previous
        return node

    def visit_Assign(self, node):
        if not self.in_main:
            if _assigned(node, "ROOT"):
                node.value = ast.parse(f"Path({str(ROOT)!r})", mode="eval").body
            if _assigned(node, "local_package"):
                node.value = ast.Constant(str(self.bundle / "isaacsim"))
            return node
        if _assigned(node, "imu_bias_limit"):
            self.hooks.append("boot")
            self.record_events = True
            return ast.parse("hil_capture.boot(ARGS, initial, startup_specific_force)").body + [node]
        # The event-loop samples are distinguished from the startup window.
        if self.record_events and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            call = node.value.func
            if isinstance(call.value, ast.Name) and call.value.id == "navigation" and call.attr in {"imu", "dvl", "depth"}:
                self.hooks.append(call.attr)
                payload = {"imu": "{'gyro': gyro, 'acceleration': acceleration}",
                           "dvl": "{'velocity_body': dvl}", "depth": "{'depth_m': depth}"}[call.attr]
                return [node] + ast.parse(f"hil_capture.emit({call.attr!r}, sim_time, {payload})").body
        if _assigned(node, "sonar_ping"):
            self.hooks.append("sonar")
            return [node] + ast.parse("hil_capture.sonar(sim_time, sonar_ping)").body
        if _assigned(node, "ping_samples"):
            self.hooks.append("reference")
            return ast.parse("hil_capture.reference_sample(sim_time, truth, slam, dead_reckoning, global_rotation, global_position)").body + [node]
        return self.generic_visit(node)

    def visit_Try(self, node):
        node = self.generic_visit(node)
        if self.in_main and any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                                and ast.unparse(n.value.func) == "slam.pose_graph.optimize"
                                for n in node.finalbody):
            self.hooks.append("finish")
            # End the live sensor stream BEFORE host-only posterior reporting.
            # A long final smoothing/report must not be charged as sensor lag.
            node.finalbody = ast.parse("hil_capture.finish(run_status, termination_reason)").body + node.finalbody
        return node


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True, choices=BASELINES)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--distance-m", type=float)
    scope.add_argument("--full-route", action="store_true")
    args = p.parse_args()
    args.bundle = args.bundle.resolve()
    args.output = args.output.resolve()
    if args.output.exists() or (not args.full_route and not 0 < args.distance_m <= 5000):
        p.error("fresh output and positive short capture (<=5 km), or explicit --full-route required")
    source = args.bundle / "standalone/large_scale_slam_isaac_demo.py"
    tree, digest = source_tree(source, args.profile)
    transform = Instrument(args.bundle)
    tree = transform.visit(copy.deepcopy(tree))
    expected = ["boot", "imu", "dvl", "depth", "sonar", "reference", "finish"]
    if sorted(transform.hooks) != sorted(expected):
        raise ValueError(f"unexpected native instrumentation sites: {transform.hooks}")
    ast.fix_missing_locations(tree)
    args.output.mkdir(parents=True)
    recorder = Recorder(args.output, args.profile, BASELINES[args.profile][0])
    argv = launcher_argv(args.bundle, args.profile, args.output, args.distance_m or 0.)
    (args.output / "capture_manifest.json").write_text(json.dumps(dict(
        source_sha256=digest, profile=args.profile, hooks=transform.hooks, argv=argv,
        live_Isaac_sensor_delivery=bool(os.environ.get("OCEANSIM_HIL_SOCKET")),
        purpose=("live Isaac simulated raw-sensor producer with independent frozen PC control; processor-in-loop, not physical sensors"
                 if os.environ.get("OCEANSIM_HIL_SOCKET") else
                 "native sensor capture and parity reference, not network HIL")), indent=2) + "\n")
    namespace = dict(__name__="__main__", __file__=str(source), hil_capture=recorder)
    sys.argv = [str(source), *argv]
    try:
        exec(compile(tree, str(source) + "::hil-capture", "exec"), namespace)
    finally:
        if not recorder.closed:
            recorder.finish("incomplete", "native_script_did_not_finish_normally")


if __name__ == "__main__":
    main()
