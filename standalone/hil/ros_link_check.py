#!/usr/bin/env python3
"""Direct-wire DDS integrity/throughput diagnostic; deliberately NOT SLAM."""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt8MultiArray

from standalone.hil.protocol import pack, unpack


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("pc", "jetson"), required=True)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--bytes", type=int, default=1_500_000)
    args = p.parse_args()
    if args.report.exists() or not 1 <= args.count <= 1000 or not 1 <= args.bytes <= 4_000_000:
        p.error("fresh report and bounded diagnostic parameters required")
    if os.environ.get("ROS_DOMAIN_ID") != "177" or not os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"):
        p.error("isolated DDS domain and interface configuration required")
    rclpy.init()
    node = Node("oceansim_direct_link_" + args.role)
    received = []
    begin = time.monotonic()
    qos = QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE)
    input_topic, output_topic = "/oceansim_hil/link_check_request", "/oceansim_hil/link_check_response"
    if args.role == "pc":
        input_topic, output_topic = output_topic, input_topic
    publisher = node.create_publisher(UInt8MultiArray, output_topic, qos)

    def receive(message):
        data = bytes(message.data)
        decoded = unpack(data)
        if args.role == "jetson":
            if decoded["sequence"] != len(received):
                raise ValueError("DDS link sequence discontinuity")
            received.append(decoded["sequence"])
            publisher.publish(UInt8MultiArray(data=array("B", pack(dict(
                sequence=decoded["sequence"], sha256=hashlib.sha256(decoded["data"].tobytes()).hexdigest())))))
        else:
            received.append(decoded)

    subscriber = node.create_subscription(UInt8MultiArray, input_topic, receive, qos)

    def wait(predicate):
        deadline = time.monotonic() + 30.
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError("direct DDS diagnostic timed out")
            rclpy.spin_once(node, timeout_sec=.01)

    try:
        latencies = []
        if args.role == "pc":
            wait(lambda: publisher.get_subscription_count() > 0 and node.count_publishers(input_topic) > 0)
            values = np.random.default_rng(31).integers(0, 256, args.bytes, dtype=np.uint8)
            expected = hashlib.sha256(values.tobytes()).hexdigest()
            for sequence in range(args.count):
                started = time.monotonic()
                publisher.publish(UInt8MultiArray(data=array("B", pack(dict(sequence=sequence, data=values)))))
                wait(lambda: len(received) > sequence)
                if received[sequence] != dict(sequence=sequence, sha256=expected):
                    raise ValueError("DDS roundtrip integrity failure")
                latencies.append(time.monotonic() - started)
        else:
            wait(lambda: len(received) == args.count)
            # Allow reliable writer ACK flushing after the final reply.
            until = time.monotonic() + 1.
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=.05)
        summary = dict(status="completed", role=args.role, count=len(received), payload_bytes=args.bytes,
                       elapsed_s=time.monotonic() - begin, test_is_navigation=False,
                       mean_round_trip_s=float(np.mean(latencies)) if latencies else None,
                       p95_round_trip_s=float(np.percentile(latencies, 95)) if latencies else None)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
