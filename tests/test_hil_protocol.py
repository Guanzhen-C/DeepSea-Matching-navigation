from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from standalone.hil.frozen_engine import build_module
from standalone.hil.capture_native import Instrument, launcher_argv
from standalone.hil.recording import Recorder, read_packets
from standalone.hil.protocol import (
    EventOrder, FIELDS, acoustic_payload, pack, restore_acoustic, unpack,
)
from oceansim.slam.large_scale_sim import FigureEightSurvey, LargeScaleSensorSimulator, LargeUnderwaterWorld


def test_packet_preserves_bits_dtype_shape_and_invalid_returns():
    values = {
        "float64": np.array([1.2345678901234567, np.nan, -0., np.inf]),
        "float32": np.arange(30, dtype=np.float32).reshape(5, 6)[:, ::2],
        "indices": np.array([0, 2**45], dtype=np.int64),
        "bool": np.array([True, False]),
        "empty": np.empty((0, 3)),
        "big_endian": np.array([1., 2.], dtype=">f8"),
        "scalar_array": np.array(3.0),
    }
    decoded = unpack(pack(values))
    for name, expected in values.items():
        assert decoded[name].dtype == expected.dtype
        assert decoded[name].shape == expected.shape
        assert decoded[name].tobytes() == expected.tobytes()


@pytest.mark.parametrize("value", [np.array([object()]), np.array(["s"]), {"$array": []}, float("nan")])
def test_packet_rejects_executable_or_ambiguous_values(value):
    with pytest.raises(ValueError):
        pack({"bad": value})


def test_packet_detects_corruption_truncation_and_trailing_data():
    packet = pack({"a": np.arange(8)})
    for altered in (packet[:-1], packet + b"x", packet[:60] + bytes([packet[60] ^ 1]) + packet[61:]):
        with pytest.raises(ValueError):
            unpack(altered)


def test_full_eight_row_ping_is_lossless_and_truth_inaccessible():
    survey = FigureEightSurvey()
    simulator = LargeScaleSensorSimulator(
        LargeUnderwaterWorld(), survey, num_beams=32, imaging_rays_per_side=64,
        sidescan_range_bins=200, along_track_rows_per_side=8,
        along_track_sample_spacing_m=0.15, seed=31,
    )
    original = simulator.bathymetric_sidescan_ping(survey.state(12.))
    payload = acoustic_payload(original)
    wire = pack(payload)
    assert b"points_world" not in wire
    assert b"origin_world" not in wire
    assert b"display_line" not in wire
    restored = restore_acoustic(unpack(wire))
    for section, names in FIELDS.items():
        for name in names:
            a = getattr(original if section == "ping" else getattr(original, section), name)
            b = getattr(restored if section == "ping" else getattr(restored, section), name)
            if isinstance(a, np.ndarray):
                assert a.dtype == b.dtype and a.shape == b.shape
                assert a.tobytes() == b.tobytes()
            else:
                assert a == b
    for original_array, restored_array in zip(original.valid_acoustic_samples(), restored.valid_acoustic_samples()):
        np.testing.assert_array_equal(original_array, restored_array)
    assert len(np.unique(restored.valid_acoustic_row_indices())) == 8
    with pytest.raises(RuntimeError, match="truth"):
        restored.imaging.points_world
    with pytest.raises(RuntimeError, match="truth"):
        restored.valid_acoustic_samples(frame="world")


def event(sequence, kind, stamp, payload):
    return dict(schema=1, sequence=sequence, kind=kind, timestamp=stamp, payload=payload)


def boot():
    return event(0, "boot", 0., dict(configuration={}, initial={}, startup_specific_force=[], baseline_sha="test"))


def test_order_enforces_complete_lifecycle_and_no_sensor_loss():
    order = EventOrder()
    order.accept(boot())
    order.accept(event(1, "imu", .02, dict(gyro=np.zeros(3), acceleration=np.ones(3))))
    with pytest.raises(ValueError):
        order.accept(event(3, "depth", .1, dict(depth_m=1.)))
    order.accept(event(2, "depth", .1, dict(depth_m=1.)))
    order.accept(event(3, "sonar", .1, {k: {} for k in FIELDS}))
    with pytest.raises(ValueError):
        order.accept(event(4, "imu", .1, dict(gyro=np.zeros(3), acceleration=np.ones(3))))
    order.accept(event(4, "finish", .1, {}))
    with pytest.raises(ValueError):
        order.accept(event(5, "depth", .2, dict(depth_m=1.)))


@pytest.mark.parametrize("kind,payload", [("depth", {"depth_m": 3., "truth": [1, 2, 3]}),
                                         ("imu", {"gyro": [0]*3}), ("boot", {})])
def test_no_unknown_truth_fields_or_second_seed(kind, payload):
    order = EventOrder()
    order.accept(boot())
    with pytest.raises(ValueError):
        order.accept(event(1, kind, .2, payload))


def test_engine_source_is_frozen_and_contains_full_backend_without_kit():
    root = Path(__file__).resolve().parents[1]
    module, audit = build_module(root / "standalone/large_scale_slam_isaac_demo.py", "lake")
    source = ast.unparse(module)
    compile(module, "extracted", "exec")
    assert "update_terrain_backend_particle_filter" in source
    assert "try_add_current_raw_particle_prior" in source
    assert "BBS3DCandidateVerifier" in source
    assert "gtsam-fixed-lag-isam2" in source
    assert "SimulationApp" not in source
    assert "LargeScaleSensorSimulator" not in source
    assert len(audit["diagnostic_truth_replacements"]) == 3
    assert not audit["native_parity_validated"]
    # The sole publication call must not accidentally execute map->odom twice.
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "create_engine")
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "published_online_global_pose"]
    assert len(calls) == 1


def test_engine_rejects_a_changed_baseline(tmp_path):
    source = tmp_path / "changed.py"
    source.write_text("raise RuntimeError('not allowed')\n")
    with pytest.raises(ValueError, match="unrecognized baseline"):
        build_module(source, "lake")


def test_native_hooks_do_not_record_startup_as_timed_imu_events():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "standalone/large_scale_slam_isaac_demo.py").read_text())
    instrument = Instrument(root)
    result = instrument.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(result)
    compile(result, "instrumented", "exec")
    assert sorted(instrument.hooks) == sorted(["boot", "imu", "dvl", "depth", "sonar", "reference", "finish", "final_log"])
    main = next(n for n in result.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    startup = next(n for n in main.body if isinstance(n, ast.For) and "range(32)" in ast.unparse(n.iter))
    assert "hil_capture" not in ast.unparse(startup)


def test_recording_round_trip_and_terminal_event(tmp_path):
    recorder = Recorder(tmp_path, "lake", "sha")
    # No need to import any simulator dependency to record the wire payload.
    from types import SimpleNamespace
    args = SimpleNamespace(map=Path("/map"), scalar=1., flag=True)
    initial = SimpleNamespace(rotation=np.eye(3), position=np.zeros(3), velocity=np.ones(3))
    recorder.boot(args, initial, np.tile([0, 0, 9.8], (32, 1)))
    recorder.emit("imu", .02, dict(gyro=np.zeros(3), acceleration=np.ones(3)))
    recorder.finish("completed", "test")
    events = list(read_packets(tmp_path / "sensor_events.osh.gz"))
    order = EventOrder()
    for item in events:
        order.accept(item)
    assert order.finished and order.next_sequence == 3
    assert events[0]["payload"]["configuration"]["map"] == {"path": "/map"}
    np.testing.assert_array_equal(events[0]["payload"]["initial"]["rotation"], np.eye(3))


def test_launcher_does_not_leak_shell_executable_to_kit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    argv = launcher_argv(root, "lake", tmp_path, 600.)
    assert argv[0].startswith("--")
    assert "$@" not in argv
    assert "--terrain-raw-mode-recovery" in argv
