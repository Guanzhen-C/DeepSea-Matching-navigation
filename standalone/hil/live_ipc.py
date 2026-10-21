"""Local Python-3.12/3.10 boundary. Only the ROS bridge opens the network.

The Isaac producer uses kernel backpressure, never a lossy latest-frame queue.
Consumption acknowledgements flow back asynchronously, so the PC reference
and Jetson estimator can execute concurrently. Truth never enters this socket.
"""
from __future__ import annotations

import select
import socket
import struct
import time

from standalone.hil.protocol import MAX_PACKET, unpack

LENGTH = struct.Struct("!I")


def read_exact(connection, count):
    result = bytearray()
    while len(result) < count:
        block = connection.recv(count - len(result))
        if not block:
            raise EOFError("HIL local bridge disconnected")
        result.extend(block)
    return bytes(result)


def read_packet(connection):
    size, = LENGTH.unpack(read_exact(connection, LENGTH.size))
    if not 0 < size <= MAX_PACKET:
        raise ValueError("invalid IPC packet size")
    return read_exact(connection, size)


def send_packet(connection, packet):
    if not 0 < len(packet) <= MAX_PACKET:
        raise ValueError("invalid IPC packet size")
    connection.sendall(LENGTH.pack(len(packet)) + packet)


class LiveProducer:
    def __init__(self, path):
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(120.)
        self.connection.connect(str(path))
        self.buffer = bytearray()
        self.last_sent = -1
        self.last_ack = -1
        self.last_result = None

    def send(self, packet, sequence):
        if sequence != self.last_sent + 1:
            raise ValueError("nonsequential producer packet")
        self.drain()
        send_packet(self.connection, packet)
        self.last_sent = sequence
        self.drain()

    def drain(self):
        disconnected = False
        while select.select([self.connection], [], [], 0.)[0]:
            block = self.connection.recv(1024 * 1024)
            if not block:
                disconnected = True
                break
            self.buffer.extend(block)
        while len(self.buffer) >= LENGTH.size:
            size, = LENGTH.unpack_from(self.buffer)
            if not 0 < size <= MAX_PACKET:
                raise ValueError("invalid IPC acknowledgement extent")
            if len(self.buffer) < LENGTH.size + size:
                break
            response = unpack(bytes(self.buffer[LENGTH.size:LENGTH.size + size]))
            del self.buffer[:LENGTH.size + size]
            if response["status"] != "consumed":
                raise RuntimeError(response.get("error", "Jetson rejected a sensor event"))
            if response["sequence"] != self.last_ack + 1 or response["sequence"] > self.last_sent:
                raise ValueError("missing or reordered board consumption acknowledgement")
            self.last_ack = response["sequence"]
            if response.get("result") is not None:
                self.last_result = response["result"]
        if disconnected and (self.last_ack != self.last_sent or self.buffer):
            raise EOFError("bridge disconnected with unconsumed sensors")

    def finish(self):
        deadline = time.monotonic() + 120.
        while self.last_ack < self.last_sent:
            self.drain()
            if time.monotonic() >= deadline:
                raise TimeoutError("Jetson did not consume the complete live sensor stream")
            if self.last_ack < self.last_sent:
                select.select([self.connection], [], [], .02)
        self.connection.close()
