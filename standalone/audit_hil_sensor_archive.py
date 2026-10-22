#!/usr/bin/env python3
"""Read a completed sensor archive once, without maps, estimators, or CUDA.

This verifies the saved input file, not navigation accuracy, hard real time,
physical sonar fidelity, or equality with an independently recorded board input.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from standalone.hil.protocol import EventOrder, FIELDS, MAX_PACKET, unpack
from standalone.hil.recording import LENGTH

RATES = {"imu": 50., "dvl": 5., "depth": 10., "sonar": 5.}


class DigestReader:
    """Hash compressed bytes as gzip reads them; never seek or buffer the file."""
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.bytes_read += len(block)
        return block


def audit(directory, *, rates=None, progress=None):
    rates = dict(RATES if rates is None else rates)
    if set(rates) != set(RATES) or any(not np.isfinite(v) or v <= 0 for v in rates.values()):
        raise ValueError("positive explicit rates for all four sensor kinds required")
    directory = Path(directory)
    summary = json.loads((directory / "recording_summary.json").read_text())
    if summary["status"] != "completed":
        raise ValueError("only a completed archive may be audited")
    path = directory / "sensor_events.osh.gz"
    before = path.stat()
    order, counts = EventOrder(), Counter()
    first, last, max_cadence_error = {}, {}, dict.fromkeys(rates, 0.)
    nonfinite = Counter()
    started = time.monotonic()
    baseline = None
    with path.open("rb") as raw:
        reader = DigestReader(raw)
        with gzip.GzipFile(fileobj=reader, mode="rb") as stream:
            while header := stream.read(LENGTH.size):
                if len(header) != LENGTH.size:
                    raise ValueError("truncated recording header")
                size, = LENGTH.unpack(header)
                if not 0 < size <= MAX_PACKET:
                    raise ValueError("invalid recording extent")
                packet = stream.read(size)
                if len(packet) != size:
                    raise ValueError("truncated recording packet")
                event = unpack(packet)  # Every envelope/body SHA and array extent.
                order.accept(event)  # Lifecycle, exact sequence, strict outer keys.
                kind, stamp, payload = event["kind"], float(event["timestamp"]), event["payload"]
                counts[kind] += 1
                if kind == "boot":
                    baseline = payload["baseline_sha"]
                    if baseline != summary["baseline_sha"]:
                        raise ValueError("initial seed baseline differs from completed source")
                if kind in rates:
                    if kind in last:
                        error = abs(stamp - last[kind] - 1. / rates[kind])
                        max_cadence_error[kind] = max(max_cadence_error[kind], error)
                        if error > 1e-8:
                            raise ValueError("sensor cadence gap: " + kind)
                    else:
                        first[kind] = stamp
                        phase = 0. if kind == "sonar" else 1. / rates[kind]
                        if abs(stamp - phase) > 1e-8:
                            raise ValueError("unexpected first sensor phase: " + kind)
                    last[kind] = stamp
                if kind == "sonar":
                    if set(payload) != set(FIELDS):
                        raise ValueError("invalid acoustic sections")
                    for section, fields in FIELDS.items():
                        if set(payload[section]) != set(fields):
                            raise ValueError("unapproved acoustic fields: " + section)
                    if payload["ping"]["products_share_raw_returns"] is not True:
                        raise ValueError("shared-return declaration missing")
                    for section in ("bathymetry", "side_scan", "imaging"):
                        if abs(float(payload[section]["timestamp"]) - stamp) > 1e-8:
                            raise ValueError("acoustic product epoch differs from event")
                elif kind in ("imu", "dvl", "depth"):
                    # Invalid samples remain invalid: count, never repair/drop.
                    for field, value in payload.items():
                        array = np.asarray(value)
                        shape = () if kind == "depth" else (3,)
                        if array.shape != shape:
                            raise ValueError("invalid sensor measurement shape")
                        if not np.isfinite(array).all():
                            nonfinite[kind + "." + field] += 1
                if progress and order.next_sequence % 100000 == 0:
                    progress(dict(events=order.next_sequence, sensor_time_s=stamp,
                                  compressed_bytes_read=reader.bytes_read,
                                  elapsed_s=time.monotonic() - started))
        # Reaching gzip EOF validates all members/footers, not just the finish packet.
        compressed_sha256 = reader.digest.hexdigest()
        bytes_read = reader.bytes_read
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("archive changed during audit")
    if not order.finished or dict(counts) != summary["events"]:
        raise ValueError("incomplete lifecycle or source/archive count mismatch")
    if counts["sonar"] != summary["samples"] or set(first) != set(rates):
        raise ValueError("source sample count or sensor kind mismatch")
    if bytes_read != after.st_size:
        raise ValueError("compressed archive was not entirely consumed")
    return dict(kind="completed_sensor_archive_integrity_not_navigation_acceptance",
                status="completed", profile=summary["profile"], baseline_sha=baseline,
                input=str(path), input_bytes=bytes_read, input_sha256=compressed_sha256,
                events=dict(counts), samples=counts["sonar"], last_sequence=order.next_sequence - 1,
                expected_rates_hz=rates, first_timestamps_s=first, last_timestamps_s=last,
                maximum_cadence_roundoff_s=max_cadence_error,
                nonfinite_inertial_measurement_counts=dict(nonfinite),
                gzip_EOF_and_all_packet_checksums_verified=True,
                only_initial_seed_has_navigation_truth_fields=True,
                acoustic_field_whitelist_and_shared_return_declarations_checked=True,
                physical_echo_lineage_or_sensor_fidelity_validation=False,
                independent_board_input_byte_comparison=False,
                new_estimator_or_board_workload=False, navigation_changed=False,
                full_route_acceptance=False, elapsed_s=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output file required")
    report = audit(args.recording_dir, progress=lambda value: print(json.dumps(value), flush=True))
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["host_process_peak_RSS_bytes_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
