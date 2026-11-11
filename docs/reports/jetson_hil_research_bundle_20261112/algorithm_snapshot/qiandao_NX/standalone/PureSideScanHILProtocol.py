"""Sensor-only, acknowledged HIL transport over an authenticated SSH pipe.

The estimator sees one controlled startup seed, scalar SSS range/amplitude,
IMU, body DVL velocity and pressure. World positions/DEM/clean IQ are absent.
"""
from __future__ import annotations

import struct
import numpy as np

from standalone.hil.protocol import MAX_PACKET, pack, unpack
from PureSideScanFeatures import SideScanFeaturePacket

LENGTH = struct.Struct('!I')
PACKET_FIELDS = frozenset(('timestamp', 'ranges_m', 'amplitudes', 'rows', 'sides',
                           'row_centres_m', 'sample_sequence', 'max_range_m'))
KINDS = {'imu': (0, 6), 'dvl': (1, 3), 'depth': (2, 1), 'sonar': (3, None)}


def read_exact(stream, count):
    blocks = bytearray()
    while len(blocks) < count:
        block = stream.read(count-len(blocks))
        if not block:
            raise EOFError('HIL pipe disconnected before complete acknowledgement.')
        blocks.extend(block)
    return bytes(blocks)


def read_frame(stream):
    length, = LENGTH.unpack(read_exact(stream, LENGTH.size))
    if not 0 < length <= MAX_PACKET:
        raise ValueError('Invalid HIL frame length.')
    return read_exact(stream, length)


def write_frame(stream, payload):
    encoded = pack(payload)
    for block in (LENGTH.pack(len(encoded)), encoded):
        remaining = memoryview(block)
        while remaining:
            count = stream.write(remaining)
            if count is None or count <= 0:
                raise EOFError('HIL pipe failed while sending a complete sensor packet.')
            remaining = remaining[count:]
    stream.flush()
    return encoded


def packet_payload(packet):
    return {name: getattr(packet, name) for name in PACKET_FIELDS}


def restore_packet(value):
    if set(value) != PACKET_FIELDS:
        raise ValueError('Only actual scalar side-scan observations and calibration are permitted.')
    if not np.isfinite(value['timestamp']) or value['timestamp'] < 0:
        raise ValueError('Finite nonnegative acquisition time required.')
    if not np.isfinite(value['max_range_m']) or value['max_range_m'] <= 0:
        raise ValueError('Finite positive range scale required.')
    return SideScanFeaturePacket(**value)


def validate_boot(value):
    if set(value) != {'kind', 'initial', 'startup_specific_force', 'expected_events', 'prior_files_SHA256'} or value['kind'] != 'boot':
        raise ValueError('Unexpected startup payload; truth/configuration fields are forbidden.')
    initial = value['initial']
    if set(initial) != {'position', 'rotation', 'velocity'}:
        raise ValueError('Only the controlled t=0 state is allowed.')
    for key, shape in [('position', (3,)), ('velocity', (3,)), ('rotation', (3, 3))]:
        array = np.asarray(initial[key])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('Invalid initial state.')
    rotation = np.asarray(initial['rotation'])
    if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-10) or np.linalg.det(rotation) <= 0:
        raise ValueError('Initial attitude must be a proper rotation.')
    startup = np.asarray(value['startup_specific_force'])
    if startup.shape != (32, 3) or not np.isfinite(startup).all():
        raise ValueError('Actual startup IMU samples required.')
    expected = value['expected_events']
    if set(expected) != set(KINDS) or any(type(v) is not int or v <= 0 for v in expected.values()):
        raise ValueError('Complete finite sensor event counts required.')
    hashes = value['prior_files_SHA256']
    if set(hashes) != {'backscatter.npy', 'backscatter_valid_mask.npy', 'metadata.json'}:
        raise ValueError('Only the intensity prior is permitted on the board.')


class SensorOrder:
    def __init__(self, expected):
        self.expected = dict(expected)
        self.counts = dict.fromkeys(expected, 0)
        self.previous = (-1., -1)

    def accept(self, event):
        if set(event) != {'kind', 'timestamp', 'values'} or event['kind'] not in KINDS:
            raise ValueError('Unexpected event fields/kind; no geometry or truth accepted.')
        kind, stamp = event['kind'], event['timestamp']
        rank, size = KINDS[kind]
        key = (stamp, rank)
        if not np.isfinite(stamp) or stamp < 0 or key <= self.previous:
            raise ValueError('Missing, duplicate or reordered sensor acquisition.')
        if kind == 'sonar':
            packet = restore_packet(event['values'])
            if packet.timestamp != stamp:
                raise ValueError('Side-scan acquisition timestamp changed.')
            values = packet
        else:
            values = np.asarray(event['values'])
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError('Actual finite sensor values required.')
        if self.counts[kind] >= self.expected[kind]:
            raise ValueError('Too many sensor samples.')
        self.previous = key
        self.counts[kind] += 1
        return kind, values

    def finish(self):
        if self.counts != self.expected:
            raise ValueError('HIL ended with missing sensor events.')
