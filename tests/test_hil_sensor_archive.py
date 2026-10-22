from __future__ import annotations

import copy
import gzip
import hashlib
import json

import numpy as np
import pytest

from standalone.audit_hil_sensor_archive import audit
from standalone.hil.protocol import FIELDS, EventOrder, pack
from standalone.hil.recording import LENGTH


def events():
    def sonar(t):
        p = {s: {f: np.empty(0) for f in names} for s, names in FIELDS.items()}
        for s in ("bathymetry", "side_scan", "imaging"):
            p[s]["timestamp"] = t
        p["ping"]["products_share_raw_returns"] = True
        return p
    values = [(0., "boot", dict(configuration={}, initial={}, startup_specific_force=np.zeros(3), baseline_sha="test")),
              (0., "sonar", sonar(0.))]
    for step in range(1, 11):
        t = step / 50.
        values.append((t, "imu", dict(gyro=np.zeros(3), acceleration=np.zeros(3))))
        if step % 10 == 0:
            values.append((t, "dvl", dict(velocity_body=np.zeros(3))))
        if step % 5 == 0:
            values.append((t, "depth", dict(depth_m=3.)))
        if step % 10 == 0:
            values.append((t, "sonar", sonar(t)))
    values.append((.2, "finish", {}))
    values.sort(key=lambda x: (x[0], EventOrder.priority[x[1]]))
    return [dict(schema=1, sequence=i, kind=k, timestamp=t, payload=p) for i, (t, k, p) in enumerate(values)]


def save(directory, values, *, damage_packet=False):
    directory.mkdir()
    counts = {}
    with gzip.open(directory / "sensor_events.osh.gz", "wb") as f:
        for i, event in enumerate(values):
            counts[event["kind"]] = counts.get(event["kind"], 0) + 1
            packet = pack(event)
            if damage_packet and i == 1:
                packet = packet[:-1] + bytes([packet[-1] ^ 1])
            f.write(LENGTH.pack(len(packet)))
            f.write(packet)
    summary = dict(status="completed", profile="lake", baseline_sha="test", events=counts, samples=counts.get("sonar", 0))
    (directory / "recording_summary.json").write_text(json.dumps(summary))


def test_complete_archive_hash_and_cadence(tmp_path):
    root = tmp_path / "run"
    save(root, events())
    report = audit(root)
    assert report["input_sha256"] == hashlib.sha256((root / "sensor_events.osh.gz").read_bytes()).hexdigest()
    assert report["events"] == dict(boot=1, sonar=2, imu=10, depth=2, dvl=1, finish=1)
    assert report["gzip_EOF_and_all_packet_checksums_verified"]
    assert not report["full_route_acceptance"]
    assert not report["physical_echo_lineage_or_sensor_fidelity_validation"]


@pytest.mark.parametrize("damage", ["sequence", "cadence", "world", "shared", "epoch", "shape", "baseline", "missing_finish", "count"])
def test_semantic_corruption_fails_closed(tmp_path, damage):
    root = tmp_path / "run"
    values = copy.deepcopy(events())
    if damage == "sequence": values[2]["sequence"] += 1
    elif damage == "cadence": values[2]["timestamp"] += .001
    elif damage == "world": values[1]["payload"]["imaging"]["points_world"] = np.zeros((1, 3))
    elif damage == "shared": values[1]["payload"]["ping"]["products_share_raw_returns"] = False
    elif damage == "epoch": values[1]["payload"]["side_scan"]["timestamp"] += .01
    elif damage == "shape": values[2]["payload"]["gyro"] = np.zeros(2)
    elif damage == "baseline": values[0]["payload"]["baseline_sha"] = "different"
    elif damage == "missing_finish": values.pop()
    save(root, values)
    if damage == "count":
        p = root / "recording_summary.json"
        s = json.loads(p.read_text()); s["events"]["imu"] += 1; p.write_text(json.dumps(s))
    with pytest.raises(ValueError): audit(root)


def test_packet_checksum_and_gzip_footer(tmp_path):
    corrupt = tmp_path / "corrupt"
    save(corrupt, events(), damage_packet=True)
    with pytest.raises(ValueError, match="checksum"): audit(corrupt)
    truncated = tmp_path / "truncated"
    save(truncated, events())
    p = truncated / "sensor_events.osh.gz"
    p.write_bytes(p.read_bytes()[:-4])
    with pytest.raises((EOFError, gzip.BadGzipFile)): audit(truncated)


def test_nonfinite_measurement_preserved_not_repaired(tmp_path):
    root = tmp_path / "run"
    values = events()
    next(v for v in values if v["kind"] == "dvl")["payload"]["velocity_body"][0] = np.nan
    save(root, values)
    assert audit(root)["nonfinite_inertial_measurement_counts"] == {"dvl.velocity_body": 1}


def test_reject_live_archive(tmp_path):
    root = tmp_path / "run"
    save(root, events())
    p = root / "recording_summary.json"
    s = json.loads(p.read_text()); s["status"] = "running"; p.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="completed"): audit(root)
