from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from ValidateProjectedSideScanFullRoute import (VARIANTS, COMPARISONS, verify_target, validate_full_source,
    verify_runtime, replay_command, verify_legacy_control)
from FinalizePureSideScanExperiment import sha
from AnalyzeSideScanLoopReplay import validate_comparison


def complete_report():
    return dict(status='completed_pure_side_scan_FEATURE_route_test', full_closed_loop_requested=True,
        full_source_snapshots_verified=True, recorded_navigation_inputs=True,
        historical_side_scan_loops_requested=True, scan_plane_registration='rank2_baseline',
        secondary_denoising_used=False, event_counts={'sonar': 100}, expected_events={'sonar': 100},
        processed_pings=100, observation_source='complex_IQ', requested_feature_matcher='LoFTR',
        requested_distance_m=16488., complete_planned_loop_length_m=16488.)


def test_only_exact_live_handle_can_start_wait_and_failure_never_becomes_completion():
    assert not verify_target({'MainPID': '42', 'SubState': 'running'}, 42, initially_live=True)
    assert verify_target({'MainPID': '0', 'ExecMainStatus': '0'}, 42)
    for state in [{'MainPID': '43'}, {}, {'MainPID': '0', 'ExecMainStatus': '1'}]:
        with pytest.raises(RuntimeError): verify_target(state, 42)
    with pytest.raises(RuntimeError): verify_target({'MainPID': '0', 'ExecMainStatus': '0'}, 42, initially_live=True)


def test_full_matrix_rejects_partial_stale_short_or_incompletely_recorded_source():
    validate_full_source(complete_report())
    for changed in [dict(status='running'), dict(full_closed_loop_requested=False),
                    dict(recorded_navigation_inputs=False), dict(processed_pings=99),
                    dict(event_counts={'sonar': 99}), dict(requested_distance_m=2000.),
                    dict(secondary_denoising_used=True), dict(submap_representation='adaptive_projected')]:
        with pytest.raises(ValueError): validate_full_source(dict(complete_report(), **changed))


def test_matrix_is_the_complete_two_factor_design_and_commands_keep_exact_input_source():
    assert {(representation, loops) for _, representation, loops in VARIANTS} == {
        ('legacy_strip', False), ('legacy_strip', True), ('adaptive_projected', False), ('adaptive_projected', True)}
    assert len(COMPARISONS) == 4
    reports = {name: dict(submap_representation=rep, loop_closure=loops) for name, rep, loops in VARIANTS}
    for _, old, new, kind in COMPARISONS: validate_comparison(reports[old], reports[new], kind)
    for _, rep, loops in VARIANTS:
        cmd = replay_command('python', Path('/runtime'), Path('/experiment'), Path('/recorded_run'), Path('/new_output'), rep, loops)
        assert cmd[1] == '/runtime/standalone/ReplayPureSideScanNavigation.py'
        assert cmd[cmd.index('--run')+1] == '/recorded_run'
        assert ('--loop-closure' in cmd) == loops
        assert cmd[cmd.index('--submap-representation')+1] == rep
        assert '--seed' not in cmd and '--distance-m' not in cmd


def test_staged_runtime_rehashed_and_rejects_changed_code_or_manifest(tmp_path):
    source = tmp_path/'standalone'; source.mkdir()
    names = ['ReplayPureSideScanNavigation.py', 'AnalyzeSideScanLoopReplay.py', 'SideScanHistoricalMatching.py']
    for name in names: (source/name).write_text('original')
    path = tmp_path/'runtime_manifest.json'
    path.write_text(json.dumps(dict(files_SHA256={n: sha(source/n) for n in names})))
    digest = verify_runtime(tmp_path)
    (source/names[0]).write_text('modified')
    with pytest.raises(RuntimeError, match='source changed'): verify_runtime(tmp_path, digest)
    path.write_text('{}')
    with pytest.raises(RuntimeError, match='manifest changed'): verify_runtime(tmp_path, digest)


def test_legacy_control_checks_every_frame_and_all_outputs_not_just_endpoint(tmp_path):
    source, replay = tmp_path/'old', tmp_path/'new'; source.mkdir(); replay.mkdir()
    a = np.arange(60.).reshape(5, 4, 3); yaw = np.arange(20.).reshape(5, 4)
    np.savez(source/'navigation_states.npz', positions=a, timestamps=np.arange(5.), distance_m=np.arange(5.), yaw_rad=yaw)
    fields = dict(positions=a[:, 1:].copy(), timestamps=np.arange(5.), distance_m=np.arange(5.), yaw_rad=yaw[:, 1:])
    np.savez(replay/'estimated_states.npz', **fields)
    assert verify_legacy_control(source, replay)['frames'] == 5
    fields['positions'][2, 2, 0] += 1.
    np.savez(replay/'estimated_states.npz', **fields)
    with pytest.raises(AssertionError): verify_legacy_control(source, replay)


def test_observer_inserts_frozen_bundle_before_first_report_and_preserves_wait_failure(tmp_path, monkeypatch):
    import ValidateProjectedSideScanFullRoute as observer
    out = tmp_path/'fresh'; experiment = tmp_path/'experiment'
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', ['observer', '--unit', 'exact.service', '--pid', '42',
        '--run', str(tmp_path/'source'), '--experiment', str(experiment),
        '--runtime', str(tmp_path/'runtime'), '--out', str(out)])
    states = iter([{'MainPID': '42', 'SubState': 'running'}, {'MainPID': '43', 'SubState': 'running'}])
    monkeypatch.setattr(observer, 'service_state', lambda unit: next(states))
    monkeypatch.setattr(observer, 'verify_runtime', lambda *args: 'verified-manifest')
    def writer(path, report):
        assert sys.path[0] == str(experiment/'estimator_bundle/isaacsim')
        path.write_text(json.dumps(report, allow_nan=False))
    monkeypatch.setattr(observer, 'dump', writer)
    with pytest.raises(RuntimeError, match='PID changed'): observer.main()
    result = json.loads((out/'validation_status.json').read_text())
    assert result['status'] == 'failed_preserved_full_matrix_no_automatic_restart'
    assert result['completed_replays'] == [] and not result['estimator_restart_allowed']
