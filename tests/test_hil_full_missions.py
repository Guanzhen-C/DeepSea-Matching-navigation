from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from standalone.hil.frozen_engine import BASELINES
from standalone.hil.protocol import pack
from standalone.hil.recording import LENGTH
import standalone.run_hil_full_missions as pipeline


def write_packets(path, records):
    with gzip.open(path, "wb") as stream:
        for record in records:
            value = pack(record)
            stream.write(LENGTH.pack(len(value)))
            stream.write(value)


@pytest.mark.parametrize("fault", [None, "position", "rotation", "time", "sequence", "missing"])
def test_network_equality_is_against_actual_board_not_PC(tmp_path, fault):
    records = []
    for index in range(3):
        record = dict(timestamp=.2*index, input_sequence=index*14)
        for channel in pipeline.CHANNELS:
            record[channel + "_position"] = np.array([index, 2., 3.])
            record[channel + "_rotation"] = np.eye(3)
        records.append(record)
    a, b = tmp_path / "board.gz", tmp_path / "returned.gz"
    write_packets(a, records)
    if fault in ("position", "rotation"):
        records[1]["local_" + fault].flat[0] += .1
    elif fault == "time":
        records[1]["timestamp"] += .1
    elif fault == "sequence":
        records[1]["input_sequence"] += 1
    elif fault == "missing":
        records.pop()
    write_packets(b, records)
    if fault:
        with pytest.raises(ValueError):
            pipeline.check_returned_poses(a, b)
    else:
        result = pipeline.check_returned_poses(a, b)
        assert result["samples"] == 3
        assert result["six_pose_channels_exactly_equal"]
        assert not result["comparison_to_PC_estimator"]


def declarations():
    manifest = dict(profile="sea", live_Isaac=True, source_truth_is_host_only=True,
        deployment_layout="packed-and-feature-batch", batch_intensity_jacobians=True,
        canonical_correspondences=True)
    receiver = dict(input_truth_after_seed=False, deployment_layout="packed-and-feature-batch",
        batch_intensity_jacobians=True, canonical_correspondences=True)
    extraction = dict(profile="sea", baseline_sha=BASELINES["sea"][0], numerical_portability_variant=dict(
        name="spatially_canonical_completed_KLT_correspondence_order"))
    native = dict(full_map_completed=True, requested_survey_segment_distance_m=0.)
    return manifest, receiver, extraction, native


@pytest.mark.parametrize("fault", [None, "truth", "variant", "baseline", "segment"])
def test_full_evidence_requires_original_baseline_live_variant_and_no_truth(fault):
    manifest, receiver, extraction, native = declarations()
    if fault == "truth":
        receiver["input_truth_after_seed"] = True
    elif fault == "variant":
        receiver["canonical_correspondences"] = False
    elif fault == "baseline":
        extraction["baseline_sha"] = "unqualified"
    elif fault == "segment":
        native["requested_survey_segment_distance_m"] = 5000.
    if fault:
        with pytest.raises(ValueError):
            pipeline.verify_declaration(manifest, receiver, extraction, full=True, native=native)
    else:
        pipeline.verify_declaration(manifest, receiver, extraction, full=True, native=native)


def test_accuracy_review_rule_is_not_relaxed_or_a_navigation_gate():
    measured = dict(RMSE_P95_relative_changes=dict(
        dr=dict(rmse_m=0., p95_m=.1), local=dict(rmse_m=.1000001, p95_m=-.5),
        **{"global": dict(rmse_m=float("nan"), p95_m=0.)}))
    bad = pipeline.accuracy_regressions(measured)
    assert {(r["channel"], r["metric"]) for r in bad} == {("local", "rmse_m"), ("global", "rmse_m")}


def test_full_launch_keeps_all_sensors_and_qualified_flags():
    command = pipeline.launch_command("lake", Path("bundle"), Path("output"))
    assert "--full-route" in command
    assert "--distance-m" not in command
    assert "--batch-intensity-jacobians" in command
    assert "--canonical-correspondences" in command
    assert not any("skip" in word or "stop" in word for word in command)


def test_lake_pilot_remains_explicitly_separate_from_full_mission():
    command = pipeline.launch_command("lake", Path("bundle"), Path("output"), distance_m=5000.)
    assert "--full-route" not in command
    assert command[command.index("--distance-m") + 1] == "5000.0"


def test_original_order_requires_explicit_plan_and_actual_matching_audit():
    manifest, receiver, extraction, native = declarations()
    manifest["canonical_correspondences"] = False
    receiver["canonical_correspondences"] = False
    extraction.pop("numerical_portability_variant")
    with pytest.raises(ValueError, match="qualified"):
        pipeline.verify_declaration(manifest, receiver, extraction, full=True, native=native)
    pipeline.verify_declaration(manifest, receiver, extraction, full=True, native=native, canonical=False)
    extraction["numerical_portability_variant"] = dict(name="unexpected_patch")
    with pytest.raises(ValueError, match="conflicts"):
        pipeline.verify_declaration(manifest, receiver, extraction, full=True, native=native, canonical=False)


def test_canonical_manifest_cannot_replace_actual_installation_evidence():
    manifest, receiver, extraction, native = declarations()
    extraction.pop("numerical_portability_variant")
    with pytest.raises(ValueError, match="extraction evidence"):
        pipeline.verify_declaration(manifest, receiver, extraction, full=True, native=native)


def test_original_order_launch_does_not_change_other_sensor_or_navigation_flags():
    default = pipeline.launch_command("lake", Path("bundle"), Path("output"))
    original = pipeline.launch_command("lake", Path("bundle"), Path("output"), canonical=False)
    assert original == [arg for arg in default if arg != "--canonical-correspondences"]


def test_qualified_code_cannot_be_changed_between_pilot_and_full_mission(tmp_path):
    source = tmp_path / "qualified.py"
    source.write_text("qualified implementation\n")
    manifest = dict(code_sha256={source.name: pipeline.file_sha256(source)})
    pipeline.verify_qualified_implementation(manifest, tmp_path)
    source.write_text("different implementation\n")
    with pytest.raises(ValueError, match="changed"):
        pipeline.verify_qualified_implementation(manifest, tmp_path)


def test_transient_polling_error_repolls_same_handles_without_restarting(tmp_path, monkeypatch):
    (tmp_path / "launch_manifest.json").write_text(json.dumps(dict(board_unit="existing-board", host_unit="existing-sender")))
    calls, saved = [], []
    sequence = [subprocess.CalledProcessError(255, "ssh"),
                dict(ActiveState="active"), dict(ActiveState="active"),
                dict(ActiveState="inactive", Result="success", ExecMainStatus="0"),
                dict(ActiveState="inactive", Result="success", ExecMainStatus="0")]

    def poll(unit, board):
        calls.append((unit, board))
        result = sequence.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(pipeline, "state", poll)
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)
    final = pipeline.wait_existing(tmp_path, 10., saved.append)
    assert final["board"]["ActiveState"] == "inactive"
    assert saved[0]["existing_tasks_not_known_terminal"]
    assert {name for name, _ in calls} == {"existing-board", "existing-sender"}
    assert len(calls) == 5


def test_failed_participant_stops_queue_not_existing_jobs(tmp_path, monkeypatch):
    (tmp_path / "launch_manifest.json").write_text(json.dumps(dict(board_unit="existing-board", host_unit="existing-sender")))
    monkeypatch.setattr(pipeline, "state", lambda unit, board: dict(
        ActiveState="failed" if board else "inactive", Result="signal" if board else "success", ExecMainStatus="9" if board else "0"))
    with pytest.raises(RuntimeError, match="queued runs were not started"):
        pipeline.wait_existing(tmp_path, 10., lambda states: None)


@pytest.mark.parametrize("failure", [None, "pilot", "lake_full"])
@pytest.mark.parametrize("lake_order", ["original", "spatial-canonical"])
def test_finite_pipeline_orders_stages_and_never_launches_after_failure(tmp_path, monkeypatch, failure, lake_order):
    pilot, output = tmp_path / "existing_pilot", tmp_path / "pipeline"
    pilot.mkdir()
    (pilot / "launch_manifest.json").write_text(json.dumps(dict(profile="sea", live_Isaac=True)))
    launched = []
    monkeypatch.setattr(pipeline, "verify_qualified_implementation", lambda manifest: None)
    monkeypatch.setattr(pipeline, "wait_existing", lambda *args: dict(board=dict(ActiveState="inactive")))
    monkeypatch.setattr(pipeline.subprocess, "check_output", lambda *args, **kwargs: "synthetic-unit-test-commit\n")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kwargs: launched.append(command))

    def finish(directory, review, terrain, *, full=False, terminal_states=None, canonical=True):
        if directory == pilot:
            assert canonical is True
        else:
            assert canonical == (lake_order == "spatial-canonical" or directory.name == "sea_full_route_live")
        if ((failure == "pilot" and directory == pilot)
                or (failure == "lake_full" and directory.name == "lake_full_route_live")):
            raise RuntimeError("synthetic review failure; never modify navigation thresholds")
        return dict(full_route=full, distance_m=47000. if full else 5000.)

    monkeypatch.setattr(pipeline, "finalize", finish)
    monkeypatch.setattr(pipeline.sys, "argv", ["pipeline", "--pilot", str(pilot),
        "--output", str(output), "--lake-short", str(tmp_path / "existing_lake_short"),
        "--lake-correspondence-order", lake_order, "--run-full-missions"])
    if failure:
        with pytest.raises(RuntimeError, match="synthetic review failure"):
            pipeline.main()
    else:
        pipeline.main()
    report = json.loads((output / "pipeline_status.json").read_text())
    profiles = [c[c.index("--profile") + 1] for c in launched]
    assert profiles == ([] if failure == "pilot" else ["lake", "lake"] if failure else ["lake", "lake", "sea"])
    if not failure:
        assert "--distance-m" in launched[0]
        assert "--full-route" in launched[1] and "--full-route" in launched[2]
        assert ("--canonical-correspondences" in launched[0]) == (lake_order == "spatial-canonical")
        assert ("--canonical-correspondences" in launched[1]) == (lake_order == "spatial-canonical")
        assert "--canonical-correspondences" in launched[2]
        assert report["status"] == "both_full_missions_completed_pending_final_review"
        assert not report["final_acceptance"]
    else:
        assert report["status"] == "needs_review_existing_tasks_untouched"
