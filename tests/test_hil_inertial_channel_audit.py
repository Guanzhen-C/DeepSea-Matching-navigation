import ast
import gzip
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from standalone.audit_hil_inertial_channel import (
    BOOT, BASELINES, compare, create_filter, initialization_nodes,
    initial_rotation_control, prepare, read_packets, replay, write_record,
)
from standalone.hil.protocol import EventOrder, PAYLOAD_KEYS


def write_events(path, rows):
    with gzip.open(path, "xb") as stream:
        for row in rows:
            write_record(stream, row)


def source_rows():
    boot = dict(configuration={}, baseline_sha=BASELINES["lake"][0],
                initial=dict(rotation=np.eye(3), position=np.array([1., 2., -3.]), velocity=np.zeros(3)),
                startup_specific_force=np.tile([0., 0., 9.81], (5, 1)))
    physical = [
        ("boot", 0., boot),
        ("sonar", 0., {k: None for k in PAYLOAD_KEYS["sonar"]}),
        ("imu", .02, dict(gyro=np.array([.1, .2, .3]), acceleration=np.array([0., 0., 9.81]))),
        ("dvl", .2, dict(velocity_body=np.array([1., 2., 3.]))),
        ("depth", .2, dict(depth_m=10.)),
        ("sonar", .2, {k: None for k in PAYLOAD_KEYS["sonar"]}),
    ]
    return [dict(schema=1, sequence=i, kind=k, timestamp=t, payload=p)
            for i, (k, t, p) in enumerate(physical)]


class FakeDR:
    def __init__(self, initial):
        self.state = SimpleNamespace(timestamp=0., position=initial["position"].copy(),
                                     rotation=initial["rotation"].copy(), velocity=np.zeros(3),
                                     gyro_bias=np.zeros(3), accel_bias=np.zeros(3), covariance=np.eye(15))
        self.propagations = []

    def propagate(self, stamp, gyro, acceleration):
        self.propagations.append((stamp, gyro.copy(), acceleration.copy()))
        self.state.position += (stamp - self.state.timestamp) * self.state.velocity
        self.state.timestamp = stamp

    def update_dvl(self, velocity):
        self.state.velocity = velocity.copy()
        return True

    def update_depth(self, depth):
        self.state.position[2] = -depth
        return True


def cached(tmp_path):
    source = tmp_path / "raw.osh.gz"
    write_events(source, source_rows())
    cache = tmp_path / "cache"
    summary = prepare(source, cache, "lake", .2)
    return cache, summary


def test_cache_preserves_source_sequences_and_numeric_payloads(tmp_path):
    cache, summary = cached(tmp_path)
    rows = list(read_packets(cache / "inertial_events.navdiag.gz"))
    assert [r["sequence"] for r in rows] == list(range(7))
    assert rows[0]["kind"] == BOOT
    assert rows[1]["payload"] == {} and rows[-2]["payload"] == {}
    np.testing.assert_array_equal(rows[2]["payload"]["gyro"], source_rows()[2]["payload"]["gyro"])
    assert summary["source_counts"] == dict(boot=1, sonar=2, imu=1, dvl=1, depth=1)
    assert summary["sonar_clocks_retained"] and not summary["full_route_acceptance"]
    with pytest.raises(ValueError, match="lifecycle"):
        EventOrder().accept(rows[0])


def test_replay_keeps_sonar_zero_order_hold_without_acoustic_updates(tmp_path):
    cache, _ = cached(tmp_path)
    dr = FakeDR(source_rows()[0]["payload"]["initial"])
    result = replay(cache, tmp_path, tmp_path / "outputs", "lake",
                    filter_factory=lambda *args: (dr, "frozen-test"))
    assert result["samples"] == 2
    assert [row[0] for row in dr.propagations] == [.02, .2]
    np.testing.assert_array_equal(dr.propagations[-1][1], [.1, .2, .3])
    assert not result["truth_after_seed_consumed"] and not result["full_route_acceptance"]


def test_prepare_refuses_missing_requested_epoch_and_existing_output(tmp_path):
    cache, _ = cached(tmp_path)
    source = tmp_path / "raw.osh.gz"
    with pytest.raises(FileExistsError):
        prepare(source, cache, "lake", .2)
    with pytest.raises(ValueError, match="available sonar epoch"):
        prepare(source, tmp_path / "incomplete", "lake", .3)
    assert not (tmp_path / "incomplete/cache_manifest.json").exists()


def test_prepare_rejects_reordered_physical_input(tmp_path):
    rows = source_rows()
    rows[2]["sequence"] += 1
    source = tmp_path / "bad.osh.gz"
    write_events(source, rows)
    with pytest.raises(ValueError, match="reordered"):
        prepare(source, tmp_path / "bad-cache", "lake", .2)


def test_replay_checks_immutable_cache_digest_before_initializing_filter(tmp_path):
    cache, _ = cached(tmp_path)
    with (cache / "inertial_events.navdiag.gz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="invalid diagnostic cache"):
        replay(cache, tmp_path, tmp_path / "out", "lake", filter_factory=lambda *args: pytest.fail())


def test_compare_needs_entire_requested_prefix_and_reports_exact_zero(tmp_path):
    cache, _ = cached(tmp_path)
    replay(cache, tmp_path, tmp_path / "out", "lake",
           filter_factory=lambda b, p, boot: (FakeDR(boot["payload"]["initial"]), "test"))
    outputs = tmp_path / "out/dr_outputs.osh.gz"
    assert compare(outputs, outputs, .2)["exactly_equal"]
    with pytest.raises(ValueError, match="incomplete"):
        compare(outputs, outputs, .4)


def test_initializer_selects_only_frozen_dr_nodes():
    tree = ast.parse("""def main():
    imu_bias_limit = np.deg2rad(1.0) / 3600.0
    initial_covariance = gauge_anchored_initial_covariance(gyro_bias_std_rad_s=imu_bias_limit / 2)
    slam = unwanted_gpu_initialization()
    dead_reckoning = AcousticESIKF(ESIKFConfig(gyro_bias_max_norm=imu_bias_limit))
    dead_reckoning.initialize(0., rotation=initial.rotation, covariance=initial_covariance)
""")
    nodes = initialization_nodes(tree)
    assert len(nodes) == 4
    assert "unwanted_gpu" not in ast.unparse(ast.Module(body=nodes, type_ignores=[]))


def test_unknown_frozen_source_is_rejected_before_import(tmp_path):
    (tmp_path / "standalone").mkdir()
    (tmp_path / "standalone/large_scale_slam_isaac_demo.py").write_text("not the frozen source")
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(dict(
        profile="lake", source_commit=BASELINES["lake"][0], files={})))
    with pytest.raises(ValueError, match="unrecognized baseline"):
        create_filter(tmp_path, "lake", {})


def rotation_pair(tmp_path):
    matrices = {"pc": np.eye(3), "arm": np.eye(3)}
    matrices["arm"][2, 1] = 1e-19
    value = dict(source_epoch_s=0., dtype="<f8", shape=[3, 3],
                 matrix_hex={k: r.astype("<f8").tobytes().hex() for k, r in matrices.items()},
                 matrix_sha256={k: hashlib.sha256(r.astype("<f8").tobytes()).hexdigest()
                                for k, r in matrices.items()})
    path = tmp_path / "rotation_pair.json"
    path.write_text(json.dumps(value))
    return path


def test_counterfactual_is_explicit_and_retains_other_initial_state(tmp_path, monkeypatch):
    cache, _ = cached(tmp_path)
    monkeypatch.setattr("standalone.audit_hil_inertial_channel.create_filter",
                        lambda b, p, boot: (FakeDR(boot["payload"]["initial"]), "test"))
    result = initial_rotation_control(cache, tmp_path, tmp_path / "out", "lake",
                                      rotation_pair(tmp_path), "arm")
    assert result["counterfactual_initialization"]["role"] == "arm"
    assert result["counterfactual_initialization"]["other_nominal_states_and_covariance_unchanged"]
    assert result["counterfactual_initialization"]["actual_rotation_delta_from_original_PC_initialization"] == 1e-19
    first = next(read_packets(tmp_path / "out/dr_outputs.osh.gz"))
    assert first["dr_rotation"][2, 1] == 1e-19
    assert not result["full_route_acceptance"]


def test_counterfactual_rejects_nonzero_epoch_and_corrupt_pair(tmp_path):
    pair = rotation_pair(tmp_path)
    value = json.loads(pair.read_text())
    value["source_epoch_s"] = 1.
    pair.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="epoch-zero"):
        initial_rotation_control(tmp_path, tmp_path, tmp_path / "out", "lake", pair, "pc")
    value["source_epoch_s"] = 0.
    value["matrix_sha256"]["arm"] = "bad"
    pair.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        initial_rotation_control(tmp_path, tmp_path, tmp_path / "out", "lake", pair, "pc")
