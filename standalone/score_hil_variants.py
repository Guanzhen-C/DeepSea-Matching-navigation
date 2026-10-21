#!/usr/bin/env python3
"""PC-only, common-truth position scores; never hide the original PC control.

These are aligned numerical measurements, NOT lifecycle/realtime/full-mission
qualification. Validate the input/receiver manifests separately. Never copy the
host truth recording onto the estimator board.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from standalone.hil.recording import read_packets
from standalone.hil.score_board_results import stats


def compare(recording, variants, control="original_PC"):
    if "original_PC" in variants or control not in {"original_PC", *variants}:
        raise ValueError("original_PC is reserved; comparison control must exist")
    source = json.loads((recording / "recording_summary.json").read_text())
    native = json.loads((recording / "native_metrics.json").read_text())
    if source["status"] != "completed" or native["run_status"] != "completed":
        raise ValueError("original native sensor/reference capture is incomplete")
    count = source["samples"]
    if count != native["pings"] or not count:
        raise ValueError("original native sample counters disagree")
    names = ["original_PC", *variants]
    inputs = {"original_PC": recording / "host_reference.osh.gz", **variants}
    streams = [read_packets(path) for path in inputs.values()]
    errors = {name: {track: [] for track in ("dr", "local", "global")} for name in names}
    differences = {name: {track: 0. for track in ("dr", "local", "global")} for name in names}
    last = None
    seen = 0
    for records in itertools.zip_longest(*streams):
        if any(record is None for record in records):
            raise ValueError("missing/excess estimator samples")
        stamp = records[0]["timestamp"]
        if not np.isfinite(stamp) or any(not np.isfinite(record["timestamp"])
                or abs(record["timestamp"] - stamp) > 1e-9 for record in records):
            raise ValueError("nonfinite/misaligned sensor timestamps")
        if last is not None and abs(stamp - last - 1. / native["ping_rate_hz"]) > 1e-8:
            raise ValueError("sensor clock gap")
        truth = np.asarray(records[0]["truth_position"], dtype=np.float64)
        positions = {}
        for name, record in zip(names, records):
            positions[name] = {}
            for track in errors[name]:
                position = np.asarray(record[track + "_position"], dtype=np.float64)
                if truth.shape != (3,) or position.shape != (3,) or not np.isfinite(truth).all() or not np.isfinite(position).all():
                    raise ValueError("invalid/nonfinite position")
                positions[name][track] = position
                errors[name][track].append(float(np.linalg.norm(position - truth)))
        for name in names:
            for track in differences[name]:
                differences[name][track] = max(differences[name][track],
                    float(np.linalg.norm(positions[name][track] - positions[control][track])))
        seen += 1
        last = stamp
    if seen != count:
        raise ValueError("native expected sample count mismatch")
    metrics = {name: {track: stats(values) for track, values in tracks.items()} for name, tracks in errors.items()}
    hashes = {}
    for name, path in inputs.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        hashes[name] = dict(path=str(path), sha256=digest.hexdigest())
    return dict(kind="PC_common_truth_position_metrics_only", samples=seen, end_timestamp_s=last,
        original_baseline_sha=source["baseline_sha"], control_role=control,
        metrics=metrics, maximum_position_difference_from_control_m=differences,
        relative_RMSE_P95_changes_from_original_PC={name: {
            track: {metric: values[metric] / max(metrics["original_PC"][track][metric], 1e-12) - 1.
                    for metric in ("rmse_m", "p95_m")} for track, values in tracks.items()}
            for name, tracks in metrics.items()}, input_files=hashes,
        trajectory_alignment_applied=False, original_PC_reference_always_retained=True,
        candidate_lifecycle_or_realtime_acceptance=False, full_route_acceptance=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True, type=Path)
    p.add_argument("--variant", action="append", required=True, help="explicit_role=estimator_outputs.osh.gz")
    p.add_argument("--control", default="original_PC")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("fresh comparison output required")
    variants = {}
    for item in args.variant:
        name, path = item.split("=", 1)
        if not name or name in variants:
            p.error("variant roles must be nonempty and unique")
        variants[name] = Path(path)
    report = compare(args.recording, variants, args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: values["global"] for name, values in report["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
