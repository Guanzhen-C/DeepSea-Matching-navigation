"""CPU-only frozen DR-channel controls; never a live/full HIL acceptance tool.

The diagnostic cache retains every original event sequence, replacing acoustic
arrays with clock markers. It deliberately cannot be replayed as a normal HIL
recording. Compare the original-PC replay with native DR BEFORE interpreting
an alternate NumPy result. No truth is consumed by the diagnostic filter after
the controlled initial seed; truth/reference files are used only by compare.
"""
from __future__ import annotations

import argparse
import ast
import copy
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np

from standalone.hil.frozen_engine import BASELINES, _assigned, source_tree
from standalone.hil.protocol import EventOrder, pack
from standalone.hil.recording import LENGTH, read_packets

FORMAT = "oceansim.inertial_channel_diagnostic.v1"
BOOT = "inertial_diagnostic_boot"
CLOCK = "inertial_diagnostic_sonar_clock"
END = "inertial_diagnostic_end"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_record(stream, value):
    packet = pack(value)
    stream.write(LENGTH.pack(len(packet)))
    stream.write(packet)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(events, output, profile, cutoff_s):
    if not math.isfinite(cutoff_s) or cutoff_s <= 0:
        raise ValueError("positive finite cutoff required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    order, counts, reached = EventOrder(), Counter(), False
    started = time.monotonic()
    cache = output / "inertial_events.navdiag.gz"
    with gzip.open(cache, "xb", compresslevel=1) as stream:
        for event in read_packets(events):
            if event["timestamp"] > cutoff_s + 1e-9:
                break
            order.accept(event)
            kind = event["kind"]
            if kind == "finish":
                break
            counts[kind] += 1
            cached = dict(event)
            if kind == "boot":
                if event["payload"]["baseline_sha"] != BASELINES[profile][0]:
                    raise ValueError("wrong frozen baseline")
                cached.update(kind=BOOT, payload=dict(format=FORMAT, source_boot=event,
                                                      cutoff_s=cutoff_s, profile=profile))
            elif kind == "sonar":
                cached.update(kind=CLOCK, payload={})
            elif kind not in ("imu", "dvl", "depth"):
                raise ValueError("unsupported physical event")
            write_record(stream, cached)
            if order.next_sequence % 10000 == 0:
                print(f"[DR cache] {order.next_sequence} source events, {event['timestamp']:.2f} s", flush=True)
            if kind == "sonar" and abs(event["timestamp"] - cutoff_s) <= 1e-9:
                reached = True
                break
        if not reached:
            raise ValueError("requested prefix must end at an available sonar epoch")
        write_record(stream, dict(schema=1, sequence=order.next_sequence, kind=END,
                                  timestamp=cutoff_s, payload=dict(source_counts=dict(counts))))
    summary = dict(format=FORMAT, profile=profile, cutoff_s=cutoff_s,
                   baseline_sha=BASELINES[profile][0], source_events=str(Path(events).resolve()),
                   source_counts=dict(counts), cache_sha256=sha256(cache),
                   complete_requested_prefix=True, whole_source_stream_audited=False,
                   acoustic_arrays_excluded=True, sonar_clocks_retained=True,
                   full_route_acceptance=False, elapsed_wall_s=time.monotonic() - started)
    write_json(output / "cache_manifest.json", summary)
    return summary


def initialization_nodes(tree):
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main.body) if _assigned(n, "imu_bias_limit"))
    stop = next(i for i, n in enumerate(main.body) if _assigned(n, "slam"))
    prefix = main.body[start:stop]
    if len(prefix) != 2 or not _assigned(prefix[1], "initial_covariance"):
        raise ValueError("unrecognized frozen DR initialization")
    constructors = [n for n in main.body if _assigned(n, "dead_reckoning")]
    initializers = [n for n in main.body if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                    and isinstance(n.value.func.value, ast.Name)
                    and n.value.func.value.id == "dead_reckoning"
                    and n.value.func.attr == "initialize"]
    if len(constructors) != 1 or len(initializers) != 1:
        raise ValueError("ambiguous frozen DR initialization")
    return copy.deepcopy(prefix + constructors + initializers)


def create_filter(bundle, profile, boot):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    if manifest["profile"] != profile or manifest["source_commit"] != BASELINES[profile][0]:
        raise ValueError("wrong frozen bundle")
    # Do not read/hash the large map for a navigation-channel-only control.
    for name, record in manifest["files"].items():
        if name.startswith("isaacsim/oceansim/") and name.endswith(".py"):
            path = (bundle / name).resolve()
            if not path.is_relative_to(bundle) or sha256(path) != record["sha256"]:
                raise ValueError("modified frozen numerical module")
    tree, digest = source_tree(bundle / "standalone/large_scale_slam_isaac_demo.py", profile)
    if "oceansim" in sys.modules:
        raise RuntimeError("a fresh process is required")
    sys.path.insert(0, str(bundle / "isaacsim"))
    from oceansim.slam.esikf import AcousticESIKF, ESIKFConfig
    from oceansim.slam.initialization import gauge_anchored_initial_covariance
    namespace = dict(np=np, initial=SimpleNamespace(**boot["payload"]["initial"]),
                     AcousticESIKF=AcousticESIKF, ESIKFConfig=ESIKFConfig,
                     gauge_anchored_initial_covariance=gauge_anchored_initial_covariance)
    module = ast.fix_missing_locations(ast.Module(body=initialization_nodes(tree), type_ignores=[]))
    exec(compile(module, str(bundle) + "::DR-only", "exec"), namespace)
    return namespace["dead_reckoning"], digest


def replay(cache_directory, bundle, output, profile, filter_factory=create_filter):
    cache_directory, output = Path(cache_directory), Path(output)
    manifest = json.loads((cache_directory / "cache_manifest.json").read_text())
    cache = cache_directory / "inertial_events.navdiag.gz"
    if (manifest["format"] != FORMAT or manifest["profile"] != profile
            or not manifest["complete_requested_prefix"] or sha256(cache) != manifest["cache_sha256"]):
        raise ValueError("invalid diagnostic cache")
    records = iter(read_packets(cache))
    header = next(records)
    if (header["kind"] != BOOT or header["sequence"] != 0 or header["timestamp"] != 0
            or header["payload"]["format"] != FORMAT
            or header["payload"]["cutoff_s"] != manifest["cutoff_s"]):
        raise ValueError("invalid diagnostic header")
    boot = header["payload"]["source_boot"]
    if boot["payload"]["baseline_sha"] != BASELINES[profile][0]:
        raise ValueError("wrong initial seed baseline")
    dr, digest = filter_factory(bundle, profile, boot)
    output.mkdir(parents=True, exist_ok=False)
    latest_gyro = np.zeros(3)
    latest_acceleration = np.mean(boot["payload"]["startup_specific_force"], axis=0)
    counts, accepted, samples = Counter(boot=1), Counter(), 0
    last_key, expected_sequence, ended = (0.0, -1), 1, False
    last_sonar_stamp = None
    started = time.monotonic()
    outputs = output / "dr_outputs.osh.gz"
    with gzip.open(outputs, "xb", compresslevel=1) as stream:
        for event in records:
            if event["schema"] != 1 or event["sequence"] != expected_sequence:
                raise ValueError("diagnostic sequence mismatch")
            expected_sequence += 1
            kind, stamp, payload = event["kind"], float(event["timestamp"]), event["payload"]
            if kind == END:
                if (abs(stamp - manifest["cutoff_s"]) > 1e-9
                        or dict(counts) != manifest["source_counts"]
                        or payload["source_counts"] != manifest["source_counts"]):
                    raise ValueError("incomplete diagnostic prefix")
                ended = True
                if next(records, None) is not None:
                    raise ValueError("trailing diagnostic records")
                break
            original_kind = "sonar" if kind == CLOCK else kind
            if original_kind not in ("imu", "dvl", "depth", "sonar"):
                raise ValueError("invalid diagnostic event kind")
            key = (stamp, EventOrder.priority[original_kind])
            if not math.isfinite(stamp) or key <= last_key:
                raise ValueError("nonmonotonic diagnostic events")
            last_key = key
            counts[original_kind] += 1
            if kind == "imu":
                latest_gyro, latest_acceleration = payload["gyro"], payload["acceleration"]
                dr.propagate(stamp, latest_gyro, latest_acceleration)
            elif kind == "dvl":
                accepted["dvl_accepted" if dr.update_dvl(payload["velocity_body"]) else "dvl_rejected"] += 1
            elif kind == "depth":
                accepted["depth_accepted" if dr.update_depth(payload["depth_m"]) else "depth_rejected"] += 1
            else:
                # This assumption is verified against native PC DR before any
                # cross-version result may be interpreted.
                if dr.state.timestamp < stamp - 1e-12:
                    dr.propagate(stamp, latest_gyro, latest_acceleration)
                samples += 1
                last_sonar_stamp = stamp
                write_record(stream, dict(timestamp=stamp, dr_position=dr.state.position.copy(),
                                          dr_rotation=dr.state.rotation.copy(),
                                          velocity=dr.state.velocity.copy(), gyro_bias=dr.state.gyro_bias.copy(),
                                          accel_bias=dr.state.accel_bias.copy(), covariance=dr.state.covariance.copy()))
                if samples % 1000 == 0:
                    print(f"[DR-only] {samples} samples, {stamp:.2f} s", flush=True)
    if (not ended or samples != manifest["source_counts"].get("sonar")
            or last_sonar_stamp is None or abs(last_sonar_stamp - manifest["cutoff_s"]) > 1e-9):
        raise ValueError("diagnostic end marker or final sonar epoch missing")
    summary = dict(format=FORMAT, mode="CPU-only inertial-channel control, not full HIL",
                   profile=profile, source_sha256=digest, cache_sha256=manifest["cache_sha256"],
                   cutoff_s=manifest["cutoff_s"], samples=samples, source_counts=dict(counts),
                   accepted_updates=dict(accepted), numpy_version=np.__version__, python_version=platform.python_version(),
                   machine=platform.machine(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   outputs_sha256=sha256(outputs), elapsed_wall_s=time.monotonic() - started,
                   truth_after_seed_consumed=False, navigation_parameters_changed=False,
                   native_PC_parity_must_be_verified_before_alternate_version=True, full_route_acceptance=False)
    write_json(output / "replay_summary.json", summary)
    return summary


def compare(reference, outputs, cutoff_s):
    maximum = {"position_m": 0.0, "rotation_component": 0.0}
    samples, last, first_nonzero = 0, None, None
    for a, b in zip(read_packets(reference), read_packets(outputs)):
        if abs(a["timestamp"] - b["timestamp"]) > 1e-9:
            raise ValueError("unaligned DR control")
        last = float(a["timestamp"])
        for field, key in (("dr_position", "position_m"), ("dr_rotation", "rotation_component")):
            av, bv = np.asarray(a[field]), np.asarray(b[field])
            if av.shape != bv.shape or not np.isfinite(av).all() or not np.isfinite(bv).all():
                raise ValueError("invalid DR output shape/value")
            difference = float(np.max(np.abs(av - bv)))
            maximum[key] = max(maximum[key], difference)
            if difference != 0.0 and first_nonzero is None:
                first_nonzero = dict(timestamp=last, field=field, maximum_component_difference=difference)
        samples += 1
        if abs(last - cutoff_s) <= 1e-9:
            break
        if last > cutoff_s:
            raise ValueError("missing requested DR epoch")
    if last is None or abs(last - cutoff_s) > 1e-9:
        raise ValueError("incomplete DR control prefix")
    return dict(samples=samples, cutoff_s=cutoff_s, maximum_component_differences=maximum,
                first_nonzero_difference=first_nonzero, exactly_equal=all(x == 0 for x in maximum.values()),
                native_parity_within_tolerance=maximum["position_m"] <= 1e-7 and maximum["rotation_component"] <= 1e-9,
                truth_used=False, full_route_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--events", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--profile", choices=BASELINES, required=True)
    prepare_parser.add_argument("--cutoff-s", type=float, required=True)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--cache", type=Path, required=True)
    replay_parser.add_argument("--bundle", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.add_argument("--profile", choices=BASELINES, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--outputs", type=Path, required=True)
    compare_parser.add_argument("--cutoff-s", type=float, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "prepare":
        result = prepare(args.events, args.output, args.profile, args.cutoff_s)
    elif args.command == "replay":
        result = replay(args.cache, args.bundle, args.output, args.profile)
    else:
        result = compare(args.reference, args.outputs, args.cutoff_s)
        write_json(args.report, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
