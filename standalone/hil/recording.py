"""Lossless, bounded-memory event recording; truth has a separate host-only file."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import struct
import time

from standalone.hil.protocol import EventOrder, MAX_PACKET, acoustic_payload, pack, unpack

LENGTH = struct.Struct("!I")


def read_packets(path):
    with gzip.open(path, "rb") as stream:
        while header := stream.read(LENGTH.size):
            if len(header) != LENGTH.size:
                raise ValueError("truncated recording header")
            size, = LENGTH.unpack(header)
            if not 0 < size <= MAX_PACKET:
                raise ValueError("invalid recording extent")
            packet = stream.read(size)
            if len(packet) != size:
                raise ValueError("truncated recording packet")
            yield unpack(packet)


class Recorder:
    def __init__(self, directory: Path, profile: str, baseline_sha: str):
        self.directory = directory
        self.profile = profile
        self.baseline_sha = baseline_sha
        self.order = EventOrder()
        self.raw = (directory / "sensor_events.osh.gz").open("xb")
        self.events = gzip.GzipFile(fileobj=self.raw, mode="wb", compresslevel=1, mtime=0)
        self.reference_raw = (directory / "host_reference.osh.gz").open("xb")
        self.reference = gzip.GzipFile(fileobj=self.reference_raw, mode="wb", compresslevel=1, mtime=0)
        self.last_flush = time.monotonic()
        self.last_stamp = 0.0
        self.samples = 0
        self.counts = {}
        self.closed = False

    def emit(self, kind, stamp, payload):
        event = dict(schema=1, sequence=self.order.next_sequence, kind=kind, timestamp=float(stamp), payload=payload)
        self.order.accept(event)
        packet = pack(event)
        self.events.write(LENGTH.pack(len(packet)))
        self.events.write(packet)
        self.last_stamp = float(stamp)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if time.monotonic() - self.last_flush >= 5.:
            self.flush()

    def boot(self, args, initial, startup_specific_force):
        configuration = {k: {"path": str(v)} if isinstance(v, Path) else v for k, v in vars(args).items()}
        self.emit("boot", 0., dict(configuration=configuration, baseline_sha=self.baseline_sha,
                                 initial={k: getattr(initial, k) for k in ("rotation", "position", "velocity")},
                                 startup_specific_force=startup_specific_force))

    def sonar(self, stamp, ping):
        self.emit("sonar", stamp, acoustic_payload(ping))

    def reference_sample(self, stamp, truth, slam, dead_reckoning, global_rotation, global_position):
        value = dict(timestamp=float(stamp), truth_position=truth.position, truth_rotation=truth.rotation,
                     dr_position=dead_reckoning.state.position, dr_rotation=dead_reckoning.state.rotation,
                     local_position=slam.state.position, local_rotation=slam.state.rotation,
                     global_position=global_position, global_rotation=global_rotation)
        packet = pack(value)
        self.reference.write(LENGTH.pack(len(packet)))
        self.reference.write(packet)
        self.samples += 1

    def flush(self):
        self.events.flush()
        self.reference.flush()
        for stream in (self.raw, self.reference_raw):
            stream.flush()
            os.fsync(stream.fileno())
        self.last_flush = time.monotonic()

    def finish(self, status, reason):
        if self.closed:
            return
        if self.order.next_sequence:
            self.emit("finish", self.last_stamp, {})
        self.flush()
        self.events.close()
        self.reference.close()
        for stream in (self.raw, self.reference_raw):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self.closed = True
        summary = dict(status=status, termination_reason=reason, profile=self.profile,
                       baseline_sha=self.baseline_sha, samples=self.samples, events=self.counts,
                       sensor_truth_fields_transmitted=False,
                       reference_file_host_only="host_reference.osh.gz")
        (self.directory / "recording_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
