#!/usr/bin/env python3
"""ROS 2 Humble raw-sensor replay bridge, separate from Isaac's Python runtime.

Use domain 177 and the per-host XML interface whitelist. Acknowledgements are
issued AFTER estimator consumption, not receipt. No frequency reduction or
silent skipping is allowed. Lockstep is a correctness test; realtime replay
reports lateness/backpressure rather than claiming success when it runs slow.
"""
from __future__ import annotations

import argparse
from array import array
from collections import OrderedDict
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt8MultiArray

from standalone.hil.protocol import EventOrder, pack, unpack
from standalone.hil.recording import LENGTH, read_packets

QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=128,
                 reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
TOPIC = "/oceansim_hil/raw_events_v1"
ACK_TOPIC = "/oceansim_hil/consumed_events_v1"


def message(packet):
    return UInt8MultiArray(data=array("B", packet))


class Receiver(Node):
    def __init__(self, args):
        super().__init__("oceansim_jetson_full_estimator")
        from standalone.hil.replay import verify_bundle
        verify_bundle(args.bundle.resolve(), args.profile)
        self.args = args
        self.order = EventOrder()
        self.engine = None
        self.cache = OrderedDict()
        self.error = None
        self.finished = False
        self.publisher = self.create_publisher(UInt8MultiArray, ACK_TOPIC, QOS)
        self.subscription = self.create_subscription(UInt8MultiArray, TOPIC, self.consume, QOS)
        self.output = gzip.open(args.output / "estimator_outputs.osh.gz", "xb", compresslevel=1)
        self.started = time.monotonic()
        self.counts = {}
        self.resource_stream = None
        self.resource_process = None
        if Path("/usr/bin/tegrastats").is_file():
            self.resource_stream = (args.output / "tegrastats.log").open("x", buffering=1)
            self.resource_process = subprocess.Popen(["/usr/bin/tegrastats", "--interval", "1000"],
                stdout=self.resource_stream, stderr=subprocess.STDOUT)

    def close_resource_monitor(self):
        if self.resource_process is not None:
            # Terminate only the monitor this receiver created, never --stop.
            self.resource_process.terminate()
            try:
                self.resource_process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                self.resource_process.kill()
                self.resource_process.wait(timeout=3.)
            self.resource_stream.close()

    def consume(self, msg):
        packet = bytes(msg.data)
        try:
            event = unpack(packet)
            sequence = event["sequence"]
            digest = hashlib.sha256(packet).digest()
            if sequence in self.cache:
                old_digest, old_response = self.cache[sequence]
                if old_digest != digest:
                    raise ValueError("conflicting duplicate sensor event")
                self.publisher.publish(message(old_response))
                return
            if self.finished or self.error:
                raise ValueError("receiver no longer consuming")
            self.order.accept(event)
            begin = time.perf_counter()
            result = None
            if event["kind"] == "boot":
                from standalone.hil.replay import initialize
                self.engine, audit = initialize(self.args.bundle.resolve(), self.args.profile, event)
                (self.args.output / "extraction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
            elif event["kind"] == "finish":
                try:
                    self.engine.send(event)
                except StopIteration as stop:
                    from oceansim.slam import sanitize_for_strict_json
                    diagnostics, _ = sanitize_for_strict_json(stop.value)
                    (self.args.output / "estimator_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
                self.finished = True
                self.output.close()
            else:
                result = self.engine.send(event)
            elapsed = time.perf_counter() - begin
            self.counts[event["kind"]] = self.counts.get(event["kind"], 0) + 1
            if result is not None:
                result.update(input_sequence=sequence, processing_wall_s=elapsed)
                encoded = pack(result)
                self.output.write(LENGTH.pack(len(encoded)))
                self.output.write(encoded)
                if result["ping_count"] % 50 == 0:
                    progress = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in result.items()}
                    temporary = self.args.output / "progress.json.tmp"
                    temporary.write_text(json.dumps(progress, indent=2) + "\n")
                    temporary.replace(self.args.output / "progress.json")
                    self.output.flush()
                    os.fsync(self.output.fileobj.fileno())
            response = pack(dict(sequence=sequence, status="consumed", result=result, processing_wall_s=elapsed))
            self.cache[sequence] = (digest, response)
            while len(self.cache) > 128:
                self.cache.popitem(last=False)
            self.publisher.publish(message(response))
            if self.finished:
                (self.args.output / "receiver_summary.json").write_text(json.dumps(dict(
                    status="completed", profile=self.args.profile, events=self.counts,
                    wall_duration_s=time.monotonic() - self.started, input_truth_after_seed=False,
                    mode="full_estimator_ROS2_raw_replay_not_controller_HIL"), indent=2) + "\n")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(self.error)
            self.publisher.publish(message(pack(dict(sequence=event.get("sequence", -1) if 'event' in locals() else -1,
                                                     status="error", error=self.error))))
            (self.args.output / "receiver_error.json").write_text(json.dumps(dict(error=self.error)) + "\n")


class Sender(Node):
    def __init__(self, args):
        super().__init__("oceansim_pc_sensor_injector")
        self.args = args
        self.publisher = self.create_publisher(UInt8MultiArray, TOPIC, QOS)
        self.subscription = self.create_subscription(UInt8MultiArray, ACK_TOPIC, self.acknowledge, QOS)
        self.pending = OrderedDict()
        self.last_ack = -1
        self.error = None
        self.maximum_lateness_s = 0.
        self.maximum_pending = 0
        self.source_clock_start = None
        self.result_lateness_s = []
        self.output_samples = 0
        self.output = gzip.open(args.output / "returned_estimator_outputs.osh.gz", "xb", compresslevel=1)
        self.live_connection = None

    def acknowledge(self, msg):
        response = unpack(bytes(msg.data))
        if response["status"] != "consumed":
            self.error = response.get("error", "invalid acknowledgement")
            return
        sequence = response["sequence"]
        if sequence <= self.last_ack:
            return
        if sequence != self.last_ack + 1 or sequence not in self.pending:
            self.error = "missing or reordered consumption acknowledgement"
            return
        published_at = self.pending.pop(sequence)
        self.last_ack = sequence
        if self.live_connection is not None:
            from standalone.hil.live_ipc import send_packet
            send_packet(self.live_connection, pack(response))
        if response["result"] is not None:
            response["result"]["transport_and_processing_roundtrip_s"] = time.monotonic() - published_at
            if self.args.mode == "realtime" and self.source_clock_start is not None:
                lateness = time.monotonic() - self.source_clock_start - response["result"]["timestamp"]
                response["result"]["arrival_lateness_s"] = lateness
                self.result_lateness_s.append(lateness)
            packet = pack(response["result"])
            self.output.write(LENGTH.pack(len(packet)))
            self.output.write(packet)
            self.output_samples += 1

    def spin_until(self, predicate, timeout_s=120.):
        deadline = time.monotonic() + timeout_s
        if self.error:
            raise RuntimeError(self.error)
        while not predicate():
            if self.error:
                raise RuntimeError(self.error)
            if time.monotonic() >= deadline:
                raise TimeoutError("HIL acknowledgement/discovery timeout; no automatic restart or event skipping")
            rclpy.spin_once(self, timeout_sec=.02)

    def run(self):
        self.spin_until(lambda: self.publisher.get_subscription_count() > 0
                        and self.count_publishers(ACK_TOPIC) > 0)
        order = EventOrder()
        start = None
        last_stamp = 0.
        counts = {}
        live_server = None
        if self.args.live_socket:
            live_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            live_server.bind(str(self.args.live_socket))
            os.chmod(self.args.live_socket, 0o600)
            live_server.listen(1)
            live_server.settimeout(120.)
            self.live_connection, _ = live_server.accept()
            self.live_connection.settimeout(120.)

        def live_events():
            from standalone.hil.live_ipc import read_packet
            while True:
                item = unpack(read_packet(self.live_connection))
                yield item
                if item["kind"] == "finish":
                    return

        source = live_events() if self.live_connection is not None else read_packets(self.args.events)
        for event in source:
            order.accept(event)
            if self.args.mode == "lockstep":
                self.spin_until(lambda: not self.pending)
            else:
                self.spin_until(lambda: len(self.pending) < 128)
            if event["kind"] == "boot":
                pass
            else:
                # Boot warmup is not charged as sensor-period processing.
                if start is None:
                    self.spin_until(lambda: not self.pending)
                    start = time.monotonic()
                    self.source_clock_start = start
                if self.args.mode == "realtime" and event["kind"] != "finish":
                    target = start + event["timestamp"]
                    while time.monotonic() < target:
                        rclpy.spin_once(self, timeout_sec=max(0., min(.02, target - time.monotonic())))
                    self.maximum_lateness_s = max(self.maximum_lateness_s, time.monotonic() - target)
            packet = pack(event)
            self.pending[event["sequence"]] = time.monotonic()
            self.publisher.publish(message(packet))
            self.maximum_pending = max(self.maximum_pending, len(self.pending))
            counts[event["kind"]] = counts.get(event["kind"], 0) + 1
            last_stamp = event["timestamp"]
            rclpy.spin_once(self, timeout_sec=0.)
        if not order.finished:
            raise ValueError("recording has no finish event")
        self.spin_until(lambda: not self.pending)
        self.output.close()
        if self.live_connection is not None:
            self.live_connection.close()
            live_server.close()
        summary = dict(status="completed", mode=self.args.mode, simulated_duration_s=last_stamp,
                       wall_sensor_duration_s=time.monotonic() - (start or time.monotonic()),
                       maximum_source_lateness_s=self.maximum_lateness_s,
                       maximum_in_flight_events=self.maximum_pending, samples=self.output_samples,
                       events=counts, last_consumed_sequence=self.last_ack,
                       transport="ROS2 Humble reliable UDP direct-NIC whitelist",
                       input_source="live_Isaac_sensor_stream" if self.live_connection else "completed_native_recording",
                       skipped_events=0, full_route_test=False)
        if self.result_lateness_s:
            import numpy as np
            summary["result_arrival_lateness_s"] = dict(maximum=max(self.result_lateness_s),
                p95=float(np.percentile(self.result_lateness_s, 95)), mean=float(np.mean(self.result_lateness_s)))
        (self.args.output / "sender_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("sender", "receiver"), required=True)
    p.add_argument("--profile", choices=("lake", "sea"))
    p.add_argument("--bundle", type=Path)
    p.add_argument("--events", type=Path)
    p.add_argument("--live-socket", type=Path, help="PC-only local Unix socket for a live Isaac producer")
    p.add_argument("--mode", choices=("lockstep", "realtime"), default="lockstep")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "177" or not os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"):
        p.error("require isolated ROS_DOMAIN_ID=177 and explicit per-host FASTRTPS_DEFAULT_PROFILES_FILE")
    if (args.output.exists() or (args.role == "sender" and bool(args.events) == bool(args.live_socket))
            or (args.role == "receiver" and not (args.bundle and args.profile))):
        p.error("fresh output and role-specific inputs required")
    args.output.mkdir(parents=True)
    rclpy.init()
    node = None
    try:
        node = Sender(args) if args.role == "sender" else Receiver(args)
        if args.role == "sender":
            node.run()
        else:
            terminal_since = None
            while rclpy.ok():
                # Remain alive to service reliable ACKs/retransmits after finish.
                rclpy.spin_once(node, timeout_sec=.1)
                if node.error or node.finished:
                    terminal_since = terminal_since or time.monotonic()
                    if time.monotonic() - terminal_since > 5.:
                        if node.error:
                            raise RuntimeError(node.error)
                        break
    finally:
        if node:
            node.output.close()
            if isinstance(node, Receiver):
                node.close_resource_monitor()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
