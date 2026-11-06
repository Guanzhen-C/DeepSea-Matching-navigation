from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from RunValidatedSideScanNavigation import validate_complete_matrix_header, build_command
from ValidateProjectedSideScanFullRoute import VARIANTS, COMPARISONS


def verified(tmp_path):
    return dict(runtime=tmp_path/'runtime', experiment=tmp_path/'experiment', source=tmp_path/'executed_source',
        validated_source_seed=31, source_report=dict(requested_distance_m=16488.743))


def test_entry_refuses_partial_matrix_or_missing_full_control():
    complete = dict(status='completed_independent_full_route_2x2_ablation',
        completed_replays=[n for n, _, _ in VARIANTS], completed_comparisons=[n for n, _, _, _ in COMPARISONS],
        legacy_control=dict(positions_timestamps_yaw_distance_bit_exact=True))
    validate_complete_matrix_header(complete)
    for change in [dict(status='replaying_complete_identical_inputs'), dict(completed_replays=['projected_loops']),
                   dict(completed_comparisons=[]), dict(legacy_control={})]:
        with pytest.raises(ValueError): validate_complete_matrix_header(dict(complete, **change))


def test_default_is_complete_iq_feature_simulation_with_the_actual_validated_profile(tmp_path):
    v = verified(tmp_path); command = build_command(v, tmp_path/'fresh', python='python')
    assert command[1] == str(v['runtime']/'standalone/RunPureSideScanFeatureShort.py')
    for option, value in [('--observations', 'complex_IQ'), ('--feature-matcher', 'LoFTR'),
                          ('--submap-representation', 'adaptive_projected'), ('--seed', '31')]:
        assert command[command.index(option)+1] == value
    assert '--full-route' in command and '--loop-closure' in command
    assert '--distance-m' not in command


def test_replay_cannot_silently_change_seed_truncate_inputs_or_pick_a_different_runtime(tmp_path):
    v = verified(tmp_path); command = build_command(v, tmp_path/'fresh', mode='replay', python='python')
    assert command[1] == str(v['runtime']/'standalone/ReplayPureSideScanNavigation.py')
    assert command[command.index('--run')+1] == str(v['source'])
    assert '--loop-closure' in command and '--full-route' not in command
    for extra in [dict(seed=32), dict(distance_m=500.)]:
        with pytest.raises(ValueError): build_command(v, tmp_path/'fresh', mode='replay', **extra)


def test_prefix_is_only_explicit_and_existing_outputs_are_preserved(tmp_path):
    v = verified(tmp_path)
    command = build_command(v, tmp_path/'fresh', distance_m=10., seed=32)
    assert '--full-route' not in command and command[command.index('--distance-m')+1] == '10.0'
    for value in [0., -1., float('nan'), 20000.]:
        with pytest.raises(ValueError): build_command(v, tmp_path/'fresh', distance_m=value)
    with pytest.raises(FileExistsError): build_command(v, tmp_path)


def test_dry_run_checks_evidence_but_never_launches_or_creates_output(tmp_path, monkeypatch):
    import RunValidatedSideScanNavigation as entry
    v = verified(tmp_path); v.update(backend_provenance={}, global_RMSE_m={'projected_loops': 1.})
    monkeypatch.setattr(entry, 'validate_matrix', lambda path: v)
    monkeypatch.setattr(entry, 'verify_feature_environment', lambda provenance: None)
    monkeypatch.setattr(entry.subprocess, 'run', lambda *args, **kw: pytest.fail('Dry run cannot start an estimator.'))
    out = tmp_path/'fresh'
    monkeypatch.setattr(sys, 'argv', ['entry', '--validation', str(tmp_path/'evidence'), '--out', str(out), '--dry-run'])
    entry.main(); assert not out.exists()


def matrix_fixture(tmp_path, monkeypatch, *, selected_rmse=1.):
    """Small protocol fixture only; this does not stand in for a full route."""
    import RunValidatedSideScanNavigation as entry
    experiment = tmp_path/'experiment'; matrix = experiment/'matrix'; matrix.mkdir(parents=True)
    source = experiment/'executed'; source.mkdir(); runtime = experiment/'runtime'
    code = runtime/'standalone'; code.mkdir(parents=True)
    for name in ['sensor.py', 'ReplayPureSideScanNavigation.py', 'WriteSideScanReport.py']:
        (code/name).write_text('frozen source')
    inputs = source/'recorded_inputs'; inputs.mkdir(); (inputs/'input_manifest.json').write_text('{}')
    scene = experiment/'scene'; scene.mkdir(); (scene/'scene_report.json').write_text('{}')
    dependency = experiment/'estimator_bundle/isaacsim/oceansim/slam/kernel.py'
    dependency.parent.mkdir(parents=True); dependency.write_bytes(b'unchanged')
    np.savez(source/'navigation_states.npz', positions=np.zeros((5, 4, 3)))
    source_report = dict(seed=31, expected_events={'sonar': 5}, navigation_states_SHA256=entry.sha(source/'navigation_states.npz'),
        simulator_scene_report_SHA256=entry.sha(scene/'scene_report.json'),
        original_core_source_SHA256={'kernel.py': entry.sha(dependency)}, prior_intensity_inputs_SHA256={})
    (source/'run_summary.json').write_text(json.dumps(source_report))
    status = dict(status='completed_independent_full_route_2x2_ablation',
        completed_replays=[n for n, _, _ in VARIANTS], completed_comparisons=[n for n, _, _, _ in COMPARISONS],
        legacy_control=dict(positions_timestamps_yaw_distance_bit_exact=True), source_run=str(source), runtime=str(runtime),
        runtime_manifest_SHA256='expected-runtime-manifest', source_run_summary_SHA256=entry.sha(source/'run_summary.json'))
    (matrix/'validation_status.json').write_text(json.dumps(status))
    records, audits = {}, {}
    for k, (name, rep, loops) in enumerate(VARIANTS):
        metrics = {'Global': {'RMSE_m': selected_rmse if k == 0 else float(k+1)}, 'DR': {'RMSE_m': 5.}}
        record = dict(submap_representation=rep, loop_closure=loops, events={'sonar': 5},
            input_manifest_SHA256=entry.sha(inputs/'input_manifest.json'), secondary_denoising_used=False,
            source_snapshots_verified=True, scan_plane_registration='rank2_baseline', canonical_submaps=False,
            all_navigation_observations_reused_losslessly=True, source_SHA256={'sensor.py': entry.sha(code/'sensor.py')},
            replay_driver_SHA256=entry.sha(code/'ReplayPureSideScanNavigation.py'),
            strict_report_writer_SHA256=entry.sha(code/'WriteSideScanReport.py'),
            frozen_core_and_prior_dependency_SHA256={str(dependency): entry.sha(dependency)},
            diagnostics=dict(backend_feature_matcher={}, DVL_altitude_observations=0, MBES_points_consumed=0,
                measured_height_likelihood_terms=0, measured_height_graph_factors=0, optical_flow_calls=0,
                raw_intensity_correlation_calls=0, secondary_submap_denoising_used=False,
                optimizer_backend='gtsam_isam2_incremental_fixed_lag'), metrics=metrics)
        records[name] = record
        audits[name] = dict(metrics=metrics, output_SHA256={'three_outputs': 'verified-'+name})
    for name, old, new, kind in COMPARISONS:
        out = matrix/('review_'+name); out.mkdir()
        review = dict(status='completed_independent_identical_input_'+kind+'_comparison', comparison_kind=kind,
            all_events_and_input_SHA256_verified=True, frontend_and_DR_bit_exact=True, source_run=str(source),
            reference=audits[old], candidate=audits[new])
        (out/'independent_comparison.json').write_text(json.dumps(review))
    calls = []
    monkeypatch.setattr(entry, 'verify_runtime', lambda *args: calls.append(('runtime', args)))
    monkeypatch.setattr(entry, 'validate_full_source', lambda *args: None)
    monkeypatch.setattr(entry, 'verify_executed_source_scope', lambda *args: calls.append(('full_scope', args)))
    monkeypatch.setattr(entry, 'verify_bundle', lambda *args: calls.append(('bundle', args)))
    monkeypatch.setattr(entry, 'verify_legacy_control', lambda *args: calls.append(('control', args)))
    monkeypatch.setattr(entry, 'RecordedSideScanInputs', lambda *args: SimpleNamespace(manifest={'events': {'sonar': 5}}))
    monkeypatch.setattr(entry, 'read_replay', lambda path, truth: (records[path.name], audits[path.name],
        {'positions': np.zeros((5, 3, 3))}, None))
    return matrix, calls, dependency


def test_complete_gate_checks_actual_four_group_bindings_and_all_original_bundle_entries(tmp_path, monkeypatch):
    import RunValidatedSideScanNavigation as entry
    matrix, calls, _ = matrix_fixture(tmp_path, monkeypatch)
    result = entry.validate_matrix(matrix)
    assert result['global_RMSE_m']['projected_loops'] == 1.
    assert result['validated_source_seed'] == 31
    assert any(kind == 'bundle' and args[1] == 'lake' for kind, args in calls)
    assert any(kind == 'control' for kind, _ in calls)
    assert any(kind == 'full_scope' for kind, _ in calls)
    path = matrix/'review_projection_with_loops/independent_comparison.json'
    d = json.loads(path.read_text()); d['candidate']['metrics']['Global']['RMSE_m'] = .001
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='Saved comparison'): entry.validate_matrix(matrix)


def test_valid_comparison_labels_cannot_admit_an_inferior_proposed_profile(tmp_path, monkeypatch):
    import RunValidatedSideScanNavigation as entry
    matrix, _, _ = matrix_fixture(tmp_path, monkeypatch, selected_rmse=6.)
    with pytest.raises(ValueError, match='did not outperform'): entry.validate_matrix(matrix)


def test_frozen_external_dependency_tampering_is_refused(tmp_path, monkeypatch):
    import RunValidatedSideScanNavigation as entry
    matrix, _, dependency = matrix_fixture(tmp_path, monkeypatch)
    dependency.write_bytes(b'changed')
    with pytest.raises(ValueError, match='dependency changed'): entry.validate_matrix(matrix)
