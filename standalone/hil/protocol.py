"""Versioned lossless HIL packets. No pickle, display image, or live ground truth.

The binary envelope is transport-independent so ROS and disk replay consume
exactly the same samples. SHA256 detects corruption, not an adversarial sender;
run the ROS bridge only on the isolated direct interface.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

import numpy as np

HEADER = struct.Struct("!8sII32s")
MAGIC = b"OSIMHIL1"
MAX_PACKET = 32 * 1024 * 1024
MAX_METADATA = 128 * 1024
WORLD_FIELDS = frozenset({"origin_world", "points_world", "port_points_world",
                          "starboard_points_world", "ray_origins_world",
                          "shared_ray_origins_world", "prim_paths"})
FIELDS = {
    "bathymetry": ("timestamp", "frame_id", "sound_speed", "frequency", "beam_angles",
                   "beam_directions", "ranges", "two_way_travel_times", "intensities",
                   "hit_mask", "points_sensor"),
    "side_scan": ("timestamp", "frame_id", "sound_speed", "frequency", "range_bin_centres",
                  "port", "starboard", "port_beams", "starboard_beams"),
    "imaging": ("timestamp", "frame_id", "beam_angles", "beam_directions", "ranges",
                "intensities", "hit_mask", "points_sensor", "source_ray_indices",
                "ray_origins_sensor", "along_track_row_indices", "sample_time_offsets_s"),
    "ping": ("port_source_beam_indices", "starboard_source_beam_indices",
             "shared_ray_directions", "bathymetry_source_ray_indices",
             "port_source_imaging_indices", "starboard_source_imaging_indices",
             "shared_ray_origins_sensor", "products_share_raw_returns", "along_track_coverage_m"),
}
PAYLOAD_KEYS = {
    "imu": frozenset({"gyro", "acceleration"}),
    "dvl": frozenset({"velocity_body"}),
    "depth": frozenset({"depth_m"}),
    "sonar": frozenset({"bathymetry", "side_scan", "imaging", "ping"}),
    # This seed is sent ONCE, at t=0, matching the declared controlled trial.
    "boot": frozenset({"configuration", "initial", "startup_specific_force", "baseline_sha"}),
    "finish": frozenset(),
}


def _json_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def pack(value: dict) -> bytes:
    """Preserve every array dtype/shape/value, including no-return NaNs."""
    blocks = []
    offset = 0

    def encode(item):
        nonlocal offset
        if isinstance(item, np.ndarray):
            if item.dtype.kind not in "bifu" or item.ndim > 4:
                raise ValueError("only bounded numeric arrays are allowed")
            array = np.ascontiguousarray(item)
            block = array.tobytes()
            marker = {"$array": [array.dtype.str, list(item.shape), offset, len(block)]}
            blocks.append(block)
            offset += len(block)
            if offset > MAX_PACKET:
                raise ValueError("packet too large")
            return marker
        if isinstance(item, np.generic):
            return encode(item.item())
        if isinstance(item, dict):
            if any(not isinstance(k, str) or k.startswith("$") for k in item):
                raise ValueError("invalid mapping key")
            return {k: encode(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [encode(v) for v in item]
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float) and math.isfinite(item):
            return item
        raise ValueError(f"unsupported packet value {type(item).__name__}")

    metadata = json.dumps(encode(value), separators=(",", ":"), allow_nan=False).encode()
    raw = b"".join(blocks)
    if len(metadata) > MAX_METADATA or HEADER.size + len(metadata) + len(raw) > MAX_PACKET:
        raise ValueError("packet too large")
    body = metadata + raw
    return HEADER.pack(MAGIC, len(metadata), len(raw), hashlib.sha256(body).digest()) + body


def unpack(packet: bytes) -> dict:
    if len(packet) < HEADER.size or len(packet) > MAX_PACKET:
        raise ValueError("invalid packet size")
    magic, meta_size, raw_size, digest = HEADER.unpack_from(packet)
    if magic != MAGIC or meta_size > MAX_METADATA or len(packet) != HEADER.size + meta_size + raw_size:
        raise ValueError("invalid envelope")
    body = packet[HEADER.size:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("checksum mismatch")
    root = json.loads(body[:meta_size], object_pairs_hook=_json_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    raw = memoryview(body)[meta_size:]
    next_offset = 0

    def decode(item, depth=0):
        nonlocal next_offset
        if depth > 24:
            raise ValueError("metadata nesting limit")
        if isinstance(item, dict):
            if "$array" in item:
                if set(item) != {"$array"}:
                    raise ValueError("invalid array marker")
                dtype_str, shape, offset, size = item["$array"]
                dtype = np.dtype(dtype_str)
                if dtype.kind not in "bifu" or dtype.itemsize not in (1, 2, 4, 8):
                    raise ValueError("invalid dtype")
                if not isinstance(shape, list) or len(shape) > 4 or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError("invalid shape")
                count = math.prod(shape)
                if type(offset) is not int or type(size) is not int or offset != next_offset or size != count * dtype.itemsize:
                    raise ValueError("invalid array extent")
                if size < 0 or offset + size > len(raw):
                    raise ValueError("truncated array")
                next_offset += size
                return np.frombuffer(raw[offset:offset + size], dtype=dtype).reshape(shape).copy()
            if any(k.startswith("$") for k in item):
                raise ValueError("unknown marker")
            return {k: decode(v, depth + 1) for k, v in item.items()}
        if isinstance(item, list):
            return [decode(v, depth + 1) for v in item]
        return item

    result = decode(root)
    if next_offset != len(raw) or not isinstance(result, dict):
        raise ValueError("unused raw data or non-mapping root")
    return result


def acoustic_payload(ping) -> dict:
    if not ping.products_share_raw_returns or ping.imaging is None:
        raise ValueError("HIL requires physically shared raw returns")
    return {section: {name: getattr(ping if section == "ping" else getattr(ping, section), name)
                      for name in names} for section, names in FIELDS.items()}


class _SensorOnly:
    def __getattribute__(self, name):
        if name in WORLD_FIELDS:
            raise RuntimeError(f"live truth/world-frame sensor field is unavailable: {name}")
        return super().__getattribute__(name)


def restore_acoustic(payload: dict):
    """Reconstruct the original pure numerical types, then deny world access.

    Constructors validate same-echo lineage using an arbitrary zero-origin
    identity frame. These dummy world fields are never transmitted and become
    inaccessible before the estimator receives the object.
    """
    from oceansim.utils.acoustic_sonar import (
        BathymetricSideScanPing, MultibeamPing, RegisteredSideScanReturns, SideScanPing,
    )
    if set(payload) != set(FIELDS):
        raise ValueError("invalid acoustic sections")
    for section, names in FIELDS.items():
        if set(payload[section]) != set(names):
            raise ValueError(f"invalid acoustic fields: {section}")
    b, s, i, p = (dict(payload[k]) for k in ("bathymetry", "side_scan", "imaging", "ping"))
    bathymetry = MultibeamPing(**b, points_world=b["points_sensor"].copy())
    side = SideScanPing(**s, display_line=np.empty(0, dtype=np.uint8),
                        port_points_world=np.empty((0, 3)), starboard_points_world=np.empty((0, 3)))
    imaging = RegisteredSideScanReturns(**i, origin_world=np.zeros(3),
                                       points_world=i["points_sensor"].copy(),
                                       ray_origins_world=i["ray_origins_sensor"].copy())
    ping = BathymetricSideScanPing(**p, bathymetry=bathymetry, side_scan=side, imaging=imaging,
                                  origin_world=np.zeros(3),
                                  shared_ray_origins_world=p["shared_ray_origins_sensor"].copy())
    for obj in (bathymetry, side, imaging, ping):
        obj.__class__ = type("SensorOnly" + type(obj).__name__, (_SensorOnly, type(obj)), {})
    return ping


class EventOrder:
    """Fail on missing, duplicate, reordered, or nonmonotonic events."""
    priority = {"boot": -1, "imu": 0, "dvl": 1, "depth": 2, "sonar": 3, "finish": 4}

    def __init__(self):
        self.next_sequence = 0
        self.last_key = (-1.0, -2)
        self.finished = False

    def accept(self, event):
        if set(event) != {"schema", "sequence", "kind", "timestamp", "payload"}:
            raise ValueError("invalid event fields")
        if event["schema"] != 1 or type(event["sequence"]) is not int or event["sequence"] != self.next_sequence:
            raise ValueError("missing, duplicate, or reordered event")
        kind, stamp, payload = event["kind"], event["timestamp"], event["payload"]
        if kind not in PAYLOAD_KEYS or self.finished or (self.next_sequence == 0) != (kind == "boot"):
            raise ValueError("invalid lifecycle")
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < 0:
            raise ValueError("invalid timestamp")
        if kind == "boot" and stamp != 0:
            raise ValueError("initial seed only allowed at t=0")
        key = (stamp, self.priority[kind])
        # Identical sensor/timestamp is also a duplicate, even with a new seq.
        if key <= self.last_key or set(payload) != PAYLOAD_KEYS[kind]:
            raise ValueError("invalid event order or payload")
        self.last_key = key
        self.next_sequence += 1
        self.finished = kind == "finish"
