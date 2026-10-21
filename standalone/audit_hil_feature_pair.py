#!/usr/bin/env python3
"""Isolate one real acoustic feature pair across platforms, without truth input.

Capture executes the frozen estimator up to a requested physical ping and wraps
one pair call without changing its result. Evaluate reuses IDENTICAL images,
XYZ, descriptors, prediction and config. This is not navigation acceptance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from standalone.hil.protocol import EventOrder
from standalone.hil.recording import read_packets
from standalone.hil.replay import initialize, verify_bundle


def save_pair(directory, feature, previous, current, prediction, result):
    arrays = {"body_from_sensor": feature._body_from_sensor}
    optional = []
    for prefix, frame in (("previous", previous), ("current", current)):
        for field in fields(frame):
            key = prefix + "_" + field.name
            value = getattr(frame, field.name)
            if value is None:
                optional.append(key)
            else:
                arrays[key] = np.asarray(value)
    if prediction is None:
        optional.append("prediction")
    else:
        arrays["prediction"] = prediction
    np.savez_compressed(directory / "pair_inputs.npz", **arrays)
    metadata = dict(config=asdict(feature.config), optional_none_fields=optional,
                    architecture=platform.machine(), numpy_version=np.__version__,
                    input_sha256=hashlib.sha256((directory / "pair_inputs.npz").read_bytes()).hexdigest(),
                    no_truth_input=True, kind="captured_real_pair_diagnostic_not_navigation_acceptance")
    (directory / "pair_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_result(directory / "captured_result.json", result, feature)


def write_result(path, result, feature, **extra):
    import cv2
    import torch
    record = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(result).items()}
    record.update(architecture=platform.machine(), numpy_version=np.__version__,
                  opencv_version=cv2.__version__, torch_version=torch.__version__,
                  feature_backend=feature.active_compute_backend,
                  no_truth_input=True, navigation_acceptance=False, **extra)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in {
        "transform_previous_sensor_current_sensor", "covariance", "klt_pairs"}}, indent=2))


def load_pair(directory, frame_class):
    metadata = json.loads((directory / "pair_metadata.json").read_text())
    path = directory / "pair_inputs.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["input_sha256"]:
        raise ValueError("changed pair data")
    arrays = np.load(path, allow_pickle=False)
    frames = []
    for prefix in ("previous", "current"):
        values = {}
        for field in fields(frame_class):
            key = prefix + "_" + field.name
            value = None if key in metadata["optional_none_fields"] else arrays[key]
            values[field.name] = value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value
        frames.append(frame_class(**values))
    prediction = None if "prediction" in metadata["optional_none_fields"] else arrays["prediction"]
    return metadata, frames, prediction, arrays["body_from_sensor"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("capture", "evaluate"))
    p.add_argument("--profile", choices=("lake", "sea"), required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--events", type=Path)
    p.add_argument("--timestamp", type=float, default=3.6)
    p.add_argument("--pair-directory", type=Path)
    p.add_argument("--forced-order-result", type=Path,
                   help="isolated diagnostic only: reuse KLT ordering from an evaluated identical pair")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("fresh audit output directory required")
    args.bundle = args.bundle.resolve()
    verify_bundle(args.bundle, args.profile)
    args.output.mkdir(parents=True)
    if args.action == "capture":
        if not args.events or not 0. < args.timestamp <= 100.:
            p.error("capture requires raw events and a finite short timestamp")
        events = iter(read_packets(args.events))
        order = EventOrder()
        boot = next(events)
        order.accept(boot)
        engine, _ = initialize(args.bundle, args.profile, boot)
        feature = engine.gi_frame.f_locals["slam"].feature_odometry
        original = feature._estimate_pair
        captured = False

        def observe(previous, current, predicted_transform=None):
            nonlocal captured
            result = original(previous, current, predicted_transform)
            if abs(current.timestamp - args.timestamp) <= 1e-8:
                save_pair(args.output, feature, previous, current, predicted_transform, result)
                captured = True
            return result

        feature._estimate_pair = observe
        for event in events:
            if event["kind"] == "finish" or event["timestamp"] > args.timestamp + 1e-8:
                break
            order.accept(event)
            engine.send(event)
        engine.close()
        if not captured:
            raise ValueError("requested acoustic pair was not encountered")
    else:
        if not args.pair_directory:
            p.error("evaluate requires captured pair directory")
        sys.path.insert(0, str(args.bundle / "isaacsim"))
        from oceansim.slam.feature_sonar_odometry import FeatureSonarOdometry, FeatureSonarOdometryConfig, _FeatureFrame
        metadata, (previous, current), prediction, body = load_pair(args.pair_directory, _FeatureFrame)
        feature = FeatureSonarOdometry(FeatureSonarOdometryConfig(**metadata["config"]),
                                      sensor_rotation_body=body[:3, :3], sensor_translation_body=body[:3, 3])
        forced = None
        if args.forced_order_result:
            saved = json.loads(args.forced_order_result.read_text())
            if saved["input_sha256"] != metadata["input_sha256"]:
                raise ValueError("forced order belongs to another pair")
            forced = [tuple(int(i) for i in pair) for pair in saved["klt_pairs"]]
            natural = feature._klt_pairs(previous, current, prediction)
            if len(forced) != len(natural) or set(forced) != set(natural):
                raise ValueError("forced order may only permute the SAME correspondence set")
            feature._klt_pairs = lambda *values: list(forced)
        result = feature._estimate_pair(previous, current, prediction)
        pairs = feature._klt_pairs(previous, current, prediction) if prediction is not None else []
        write_result(args.output / "evaluated_result.json", result, feature,
                     input_sha256=metadata["input_sha256"], klt_pairs=pairs,
                     forced_correspondence_order_diagnostic=forced is not None,
                     kind="identical_pair_portability_diagnostic")


if __name__ == "__main__":
    main()
