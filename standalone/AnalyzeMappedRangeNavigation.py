"""Full identical-input mapped-range navigation comparison, evaluation ONLY.

Rejects running/prefix results. No estimator parameters are selected here.
Negative results are exported as faithfully as improvements. Source raw
inputs, native I/Q, caches and GPS are never copied or uploaded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from AnalyzeSideScanLoopReplay import read_replay, metrics, sha
from AnalyzePureSideScanShort import planar_relative_metrics
from AuditMappedRangeInputs import audit_mapped_range_inputs
from ReplayPureSideScanNavigation import same_executed_configuration


def verify_frozen_dependencies(old, replay):
    """Independently rehash the exact seven executed core/prior dependencies.

    A matching statement in a summary is not sufficient: inspect the actual
    files, require the complete original dependency set, and reject duplicate
    basenames or a different dependency role. Evaluation only, no estimator
    selection or changes to any running source snapshot.
    """
    core = old.get('original_core_source_SHA256', {})
    prior = old.get('prior_intensity_inputs_SHA256', {})
    if (set(core) != {'esikf.py', 'global_tracking.py', 'pose_graph.py', 'map_odom.py'}
            or set(prior) != {'backscatter.npy', 'backscatter_valid_mask.npy', 'metadata.json'}):
        raise ValueError('Complete original core and prior dependency hashes required')
    expected = {**core, **prior}
    dependencies = replay.get('frozen_core_and_prior_dependency_SHA256', {})
    names = [Path(path).name for path in dependencies]
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError('Executed core/prior dependencies differ or contain duplicate names')
    for name, recorded_sha in dependencies.items():
        path = Path(name)
        suffix = ('estimator_bundle', 'isaacsim', 'oceansim', 'slam', path.name) if path.name in core else (
            'scene', 'reference_baseline', path.name)
        if (path.parts[-len(suffix):] != suffix or recorded_sha != expected[path.name]
                or sha(path) != recorded_sha):
            raise ValueError('Executed frozen dependency changed: '+str(path))
    return dict(status='independently_rehashed_all_executed_original_core_and_prior_files',
                core_files=len(core), prior_files=len(prior))


def verify_preserve_prefix(old, replay, source, state, trace):
    """Recompute the causal no-change prefix from saved states, not its label.

    The interval ends at the first actual new range-factor timestamp. No
    ground-truth pose or error is used; this compares two estimated Global
    trajectories after both runs are complete. Does not reject a worse final
    metric or choose any observation by its true localization error.
    """
    if replay.get('mapped_range_policy', 'tempered_legacy') != 'preserve_legacy':
        return dict(status='not_applicable_to_tempered_legacy_experiment')
    if not same_executed_configuration(old,
            loop_closure=replay.get('loop_closure', False),
            scan_plane_registration=replay.get('scan_plane_registration', 'rank2_baseline'),
            canonical_submaps=replay.get('canonical_submaps', False),
            submap_representation=replay.get('submap_representation', 'legacy_strip')):
        raise ValueError('Preserve-legacy attribution requires the same original navigation protocol')
    stamps = np.asarray(source['timestamps'])
    times = np.array([entry['timestamp_s'] for entry in trace], dtype=float)
    if times.size and (not np.isfinite(times).all() or np.any(np.diff(times) < 0)
                       or times[0] < stamps[0] or times[-1] > stamps[-1]):
        raise ValueError('Actual new range-factor timestamps must be causal and within the recording')
    first = float(times.min()) if times.size else None
    prefix = stamps < first if first is not None else np.ones(len(stamps), dtype=bool)
    difference = (float(np.max(np.abs(state['positions'][prefix, 2]-source['positions'][prefix, 3])))
                  if prefix.any() else 0.)
    reported = replay.get('maximum_Global_difference_before_first_range_m')
    if reported is None or not np.isfinite(reported) or difference != reported or difference > 1e-7:
        raise ValueError('Independent pre-range Global prefix differs from the original baseline or run claim')
    return dict(status='independently_verified_original_Global_before_any_new_range_factor',
                samples=int(prefix.sum()), first_new_range_timestamp_s=first,
                maximum_coordinate_difference_m=difference,
                compared_estimates_not_ground_truth=True)


def direction_metrics(positions, source):
    error = positions-source['positions'][:, 0]
    yaw = source['yaw_rad'][:, 0]
    along = error[:, 0]*np.cos(yaw)+error[:, 1]*np.sin(yaw)
    cross = -error[:, 0]*np.sin(yaw)+error[:, 1]*np.cos(yaw)
    return dict(along_track_RMSE_m=float(np.sqrt(np.mean(along**2))),
                cross_track_RMSE_m=float(np.sqrt(np.mean(cross**2))),
                vertical_RMSE_m=float(np.sqrt(np.mean(error[:, 2]**2))))


def require_execution_scope(old, replay, *, heldout_recording=False):
    expected = old.get('expected_events', {})
    n = old.get('processed_pings')
    scope_ok = (not old.get('full_closed_loop_requested', True) if heldout_recording else
                old.get('full_closed_loop_requested') is True and n == 62458)
    if (old.get('status') != 'completed_pure_side_scan_FEATURE_route_test' or not scope_ok
            or n != expected.get('sonar') or old.get('event_counts') != expected
            or replay.get('status') != 'completed_lossless_navigation_input_replay'
            or replay.get('mapped_range_changes') is not True or replay.get('estimated_frames') != n
            or replay.get('evaluation_truth_read_only_after_estimation') is not True
            or replay.get('events') != expected):
        raise ValueError('Complete identical-input executed scope required; no running/prefix substitution')
    return ('completed_heldout_source_recording_NOT_full16km' if heldout_recording else
            'completed_full16km_planned_loop')


def analyze(source_run, candidate, out, *, heldout_recording=False):
    source_run, candidate, out = map(Path, (source_run, candidate, out))
    if out.exists(): raise FileExistsError('Preserve every previous comparison')
    old = json.loads((source_run/'run_summary.json').read_text())
    replay = json.loads((candidate/'run_summary.json').read_text())
    executed_scope = require_execution_scope(old, replay, heldout_recording=heldout_recording)
    dependencies = verify_frozen_dependencies(old, replay)
    if sha(source_run/'navigation_states.npz') != old['navigation_states_SHA256']:
        raise ValueError('Original baseline states changed')
    manifest_path = source_run/'recorded_inputs/input_manifest.json'
    if sha(manifest_path) != replay['input_manifest_SHA256']:
        raise ValueError('Input manifests differ')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files_SHA256'].items():
        if sha(manifest_path.parent/name) != expected: raise ValueError('Input archive changed: '+name)
    source = np.load(source_run/'navigation_states.npz')
    report, review, state, error = read_replay(candidate, source)
    graph = json.loads((candidate/'graph_factors.json').read_text())
    trace = json.loads((candidate/'mapped_range_change_trace.json').read_text())
    preserved_prefix = verify_preserve_prefix(old, report, source, state, trace)
    amplitudes = np.load(source_run/'recorded_inputs/received_side_scan_amplitudes.npy', mmap_mode='r')
    inputs = audit_mapped_range_inputs(graph, trace, source, manifest['calibration'], amplitudes)
    if inputs['factors'] != report['diagnostics']['mapped_range_changes']['factors']:
        raise ValueError('Claimed new constraints differ from actual graph and raw inputs')
    baseline_error = np.linalg.norm(source['positions'][:, 3]-source['positions'][:, 0], axis=1)
    baseline = metrics(baseline_error); after = metrics(error[:, 2])
    for k in baseline:
        np.testing.assert_allclose(baseline[k], old['metrics']['Global'][k], atol=1e-11, rtol=0)
    status = ('completed_independent_heldout_recording_mapped_range_comparison_NOT_full16km'
              if heldout_recording else 'completed_independent_full_identical_input_mapped_range_comparison')
    precision_policy = report.get('mapped_range_policy', 'tempered_legacy')
    result = dict(status=status, executed_scope=executed_scope, source_seed=old.get('seed'),
        mapped_range_precision_policy=precision_policy,
        samples=len(error), distance_m=float(source['distance_m'][-1]),
        all_received_events_verified=True, DR_and_Local_bit_exact=True,
        full_original_input_file_SHA256_reverified=True,
        executed_dependency_audit=dependencies, preserved_prefix_audit=preserved_prefix,
        baseline_Global=baseline, candidate_Global=after,
        baseline_directional=direction_metrics(source['positions'][:, 3], source),
        candidate_directional=direction_metrics(state['positions'][:, 2], source),
        baseline_planar_relative=planar_relative_metrics(source['positions'], source['yaw_rad'], source['distance_m']),
        candidate_planar_relative=planar_relative_metrics(
            np.concatenate([source['positions'][:, :1], state['positions']], axis=1),
            np.column_stack([source['yaw_rad'][:, 0], state['yaw_rad']]), source['distance_m']),
        Global_RMSE_change_percent=float(100*(after['RMSE_m']/baseline['RMSE_m']-1)),
        actual_range_input_audit=inputs, actual_historical_loop_edges=review['actual_loop_edges'],
        legacy_pre_budget_loop_trace_reconciliations=review['legacy_pre_budget_loop_trace_reconciliations'],
        actual_new_map_priors=len(graph['priors']),
        source_run=str(source_run.resolve()), candidate_run=str(candidate.resolve()),
        source_states_SHA256=old['navigation_states_SHA256'], replay_outputs_SHA256=review['output_SHA256'],
        raw_data_uploaded=False, external_upload_performed=False,
        limitations=['One same-voyage nominally geocoded simulated route and seed, not field navigation accuracy.',
            ('Legacy map and loop precisions preserved; new range precision budget .5; no full correlation proof.'
             if precision_policy == 'preserve_legacy' else
             'Opt-in combination includes conservative correlation tempering, not isolated range-information ablation.'),
            'Unknown landmark height remains an interval; relative geometry does not create measured bathymetry.',
            'Map/loop correlations are conservatively tempered but not fully jointly modelled.'])
    out.mkdir(parents=True)
    (out/'independent_comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (out/'full_errors.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_s', 'distance_m', 'DR_error_m', 'Local_error_m',
                         'baseline_Global_error_m', 'candidate_Global_error_m'])
        writer.writerows(np.column_stack([source['timestamps'], source['distance_m'], error[:, :2],
                                          baseline_error, error[:, 2]]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists(): plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    x = source['distance_m']/1000
    ax.plot(x, baseline_error, color='#7654a3', lw=.8, label='已保留基线 Global')
    ax.plot(x, error[:, 2], color='#009789', lw=.8, label='跨航线斜距变化候选 Global')
    scope_label = f'补充独立记录{x[-1]:.2f}km（非16.49km全程）' if heldout_recording else '完整16.49km'
    ax.set(xlabel='累计航程 / km', ylabel='在线三维位置误差 / m',
        title=scope_label+'同输入对照：不重生成声呐，不修改DR或Local，不删除峰值')
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'完整同输入Global误差对照.png', dpi=200); plt.close(fig)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--heldout-recording', action='store_true',
                   help='Explicit additional completed short recording; never label it full16km validation')
    a = p.parse_args(); analyze(a.source_run, a.candidate, a.out, heldout_recording=a.heldout_recording)
