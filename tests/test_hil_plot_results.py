from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from standalone.hil.protocol import pack
from standalone.hil.recording import LENGTH
from standalone.plot_hil_results import load_run


def write_packets(path, records):
    with gzip.open(path, "wb") as stream:
        for record in records:
            packet = pack(record)
            stream.write(LENGTH.pack(len(packet)))
            stream.write(packet)


def fixture_run(tmp_path):
    root = tmp_path / "synthetic_unit_fixture"
    source, sender, board = (root / name for name in ("native_reference", "sender", "board_artifacts"))
    for directory in (source, sender, board):
        directory.mkdir(parents=True)
    (root / "launch_manifest.json").write_text(json.dumps(dict(
        recording=str(source), profile="lake", mode="lockstep", live_Isaac=False)))
    events = dict(imu=20, dvl=2, depth=4, sonar=3, boot=1, finish=1)
    (source / "recording_summary.json").write_text(json.dumps(dict(status="completed", samples=3, events=events)))
    (source / "native_metrics.json").write_text(json.dumps(dict(run_status="completed", completed_requested_route=False,
        ping_rate_hz=5., imu_samples=20, dvl_samples=2, depth_samples=4, pings=3, survey_travelled_distance_m=.6)))
    (sender / "sender_summary.json").write_text(json.dumps(dict(status="completed", samples=3, skipped_events=0, events=events)))
    (board / "receiver_summary.json").write_text(json.dumps(dict(status="completed", events=events)))
    (board / "estimator_diagnostics.json").write_text(json.dumps(dict(
        graph_diagnostics=dict(optimizer_backend="gtsam_isam2_incremental_fixed_lag"),
        feature_diagnostics=dict(active_compute_backend="cuda", cuda_fallbacks=0))))
    reference, estimates = [], []
    for t in (0., .2, .4):
        truth = np.array([1.5*t, 0., -40.])
        r = dict(timestamp=t, truth_position=truth)
        b = dict(timestamp=t, transport_and_processing_roundtrip_s=.03)
        for name, error in (("dr", .5), ("local", .25), ("global", .1)):
            r[name + "_position"] = truth.copy()
            b[name + "_position"] = truth + [error, 0., 0.]
        reference.append(r)
        estimates.append(b)
    write_packets(source / "host_reference.osh.gz", reference)
    write_packets(sender / "returned_estimator_outputs.osh.gz", estimates)
    return root


def test_plot_loader_uses_real_board_positions_without_alignment(tmp_path):
    run = load_run(fixture_run(tmp_path))
    np.testing.assert_allclose(run["errors"], np.tile([.5, .25, .1], (3, 1)))
    np.testing.assert_allclose(run["distance"], [0., .3, .6])
    np.testing.assert_array_equal(run["native_errors"], np.zeros((3, 3)))
    assert not run["full_route"]


def test_short_run_cannot_be_labelled_full_hil(tmp_path):
    with pytest.raises(ValueError, match="full-route"):
        load_run(fixture_run(tmp_path), require_full_route=True)


def test_plot_loader_rejects_dropped_input_even_if_host_board_counts_agree(tmp_path):
    root = fixture_run(tmp_path)
    native = root / "native_reference/native_metrics.json"
    values = json.loads(native.read_text())
    values["imu_samples"] += 1
    native.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="independent native sensor"):
        load_run(root)


def test_plot_loader_rejects_mislabeled_live_delivery(tmp_path):
    root = fixture_run(tmp_path)
    path = root / "launch_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["live_Isaac"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="live delivery"):
        load_run(root)
