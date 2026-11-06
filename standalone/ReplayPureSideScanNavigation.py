"""Replay exact received navigation inputs; no sonar/world resimulation.

Evaluation truth is opened ONLY after estimation terminates. A file-access
guard rejects terrain heights/GPS and premature truth access. Default replay
uses the executed runtime snapshot for a strict baseline-reproduction check.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from WriteSideScanReport import dump as dump_report


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--runtime', choices=('recorded', 'current'), default='recorded')
    p.add_argument('--loop-closure', action='store_true')
    p.add_argument('--scan-plane-registration', choices=('rank2_baseline', 'observable_subspace'), default='rank2_baseline')
    p.add_argument('--canonical-submaps', action='store_true')
    p.add_argument('--submap-representation', choices=('legacy_strip', 'adaptive_projected'), default='legacy_strip')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.out.exists() or ((a.loop_closure or a.canonical_submaps or a.submap_representation != 'legacy_strip'
                          or a.scan_plane_registration != 'rank2_baseline') and a.runtime != 'current'):
        raise ValueError('New output required; new loop module is only available in the current runtime.')
    original = json.loads((a.run/'run_summary.json').read_text())
    if not original['status'].startswith('completed_') or not original.get('recorded_navigation_inputs'):
        raise ValueError('Complete executed input archive required; never replay a partial stale running file.')
    dependencies = {a.experiment/'estimator_bundle/isaacsim/oceansim/slam'/name: expected
        for name, expected in original['original_core_source_SHA256'].items()}
    dependencies.update({a.experiment/'scene/reference_baseline'/name: expected
        for name, expected in original['prior_intensity_inputs_SHA256'].items()})
    for path, expected in dependencies.items():
        if sha(path) != expected: raise ValueError('Frozen estimator/prior input changed: '+str(path))
    runtime = a.run/'source_snapshot' if a.runtime == 'recorded' else ROOT/'standalone'
    sys.path[:0] = [str(runtime), str(a.experiment/'estimator_bundle/isaacsim'), str(ROOT/'standalone'), str(ROOT)]
    estimation_finished = False
    evaluation_file = (a.run/'navigation_states.npz').resolve()
    def guard(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)): return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        if ('gps' in path.name.lower() or path.name in ('elevation.npy', 'navigation_elevation.npy')
                or (not estimation_finished and path.resolve() == evaluation_file)):
            raise RuntimeError('Forbidden estimator access to GPS/terrain height/evaluation truth: '+str(path))
    sys.addaudithook(guard)
    from SideScanRecordedInputs import RecordedSideScanInputs
    from SideScanFeatureMap import IntensityFeatureMap
    from PureSideScanFeatureNavigation import PureSideScanFeatureNavigator
    from SideScanLearnedFeatures import LoFTRSideScanMatcher
    from oceansim.slam import sanitize_for_strict_json
    import torch
    torch.set_num_threads(1)
    inputs = RecordedSideScanInputs(a.run/'recorded_inputs')
    init = inputs.manifest['initialization']
    initial = SimpleNamespace(position=np.array(init['position']), rotation=np.array(init['rotation']),
                              velocity=np.array(init['velocity']))
    options = dict(loop_closure=True) if a.loop_closure else {}
    if a.scan_plane_registration != 'rank2_baseline': options['scan_plane_registration'] = a.scan_plane_registration
    if a.canonical_submaps: options['canonical_submaps'] = True
    if a.submap_representation != 'legacy_strip': options['submap_representation'] = a.submap_representation
    nav = PureSideScanFeatureNavigator(IntensityFeatureMap.from_path(a.experiment/'scene/reference_baseline'),
        initial, np.array(init['startup_specific_force']), image_matcher=LoFTRSideScanMatcher(a.device), **options)
    a.out.mkdir()
    names = ['PureSideScanFeatures.py', 'SideScanFeatureMap.py', 'PureSideScanFeatureNavigation.py',
             'SideScanLearnedFeatures.py', 'SideScanRecordedInputs.py']
    if a.loop_closure: names += ['SideScanLoopGeometry.py', 'SideScanLoopClosure.py']
    if a.scan_plane_registration != 'rank2_baseline': names += ['SideScanPlaneRegistration.py']
    if a.submap_representation != 'legacy_strip':
        names += ['SideScanProjectedSubmap.py', 'SideScanProjectedMatching.py']
        if 'SideScanPlaneRegistration.py' not in names: names += ['SideScanPlaneRegistration.py']
    snapshot = a.out/'source_snapshot'; snapshot.mkdir()
    for name in names: shutil.copy2(runtime/name, snapshot/name)
    shutil.copy2(Path(__file__), snapshot/Path(__file__).name)
    writer_path = Path(__file__).with_name('WriteSideScanReport.py')
    shutil.copy2(writer_path, snapshot/writer_path.name)
    report = dict(status='running_lossless_navigation_input_replay', runtime=a.runtime,
        loop_closure=a.loop_closure, secondary_denoising_used=False, sonar_resimulated=False,
        scan_plane_registration=a.scan_plane_registration,
        canonical_submaps=a.canonical_submaps,
        submap_representation=a.submap_representation,
        all_navigation_observations_reused_losslessly=True, estimated_frames=0,
        input_manifest_SHA256=sha(a.run/'recorded_inputs/input_manifest.json'),
        executed_original_report_SHA256=sha(a.run/'run_summary.json'),
        source_SHA256={n: sha(snapshot/n) for n in names}, events={})
    report['frozen_core_and_prior_dependency_SHA256'] = {str(k): v for k, v in dependencies.items()}
    report['replay_driver_SHA256'] = sha(Path(__file__))
    report['strict_report_writer_SHA256'] = sha(writer_path)
    def save_report():
        dump_report(a.out/'run_summary.json', report)
    save_report()
    started = time.monotonic(); positions, stamps, yaws = [], [], []
    try:
        with (a.out/'estimated_positions.csv').open('x', newline='') as f, (a.out/'backend_trace.jsonl').open('x') as trace:
            writer = csv.writer(f)
            writer.writerow(['timestamp_s', *[f'{n}_{c}' for n in ['DR', 'Local', 'Global'] for c in 'xyz']])
            for stamp, kind, values in inputs.events():
                if kind == 'imu': nav.imu(stamp, values[:3], values[3:])
                elif kind == 'dvl': nav.dvl(values)
                elif kind == 'depth': nav.depth(values[0])
                else:
                    result = nav.side_scan(values)
                    position = np.array([result[n] for n in ['DR', 'Local', 'Global']])
                    positions.append(position); stamps.append(stamp)
                    yaws.append([float(np.arctan2(nav.dr.state.rotation[1, 0], nav.dr.state.rotation[0, 0])),
                                 float(result['raw_pose2'][2]), float(result['global_pose2'][2])])
                    writer.writerow([stamp, *position.ravel()])
                    clean, _ = sanitize_for_strict_json(nav.diagnostics())
                    if nav.observation_id % 150 == 0: trace.write(json.dumps(clean, allow_nan=False)+'\n')
                    report['estimated_frames'] += 1
                    if nav.observation_id % 250 == 0:
                        report.update(last_timestamp_s=stamp, elapsed_s=time.monotonic()-started, diagnostics=clean)
                        save_report(); f.flush(); trace.flush()
                        print(json.dumps(dict(frames=nav.observation_id, timestamp_s=stamp,
                            map_priors=nav.graph_feature_priors, loops=clean.get('loop_closure', {}))), flush=True)
                report['events'][kind] = report['events'].get(kind, 0)+1
        estimation_finished = True
        if report['events'] != inputs.manifest['events']: raise RuntimeError('Replay dropped sensor events.')
        # Truth appears for the first time only in this evaluation section.
        source = np.load(evaluation_file)
        array = np.asarray(positions)
        np.testing.assert_array_equal(stamps, source['timestamps'])
        np.testing.assert_array_equal(array[:, :2], source['positions'][:, 1:3])
        difference = float(np.max(abs(array-source['positions'][:, 1:])))
        if (not a.loop_closure and not a.canonical_submaps and a.submap_representation == 'legacy_strip'
                and a.scan_plane_registration == 'rank2_baseline' and difference > 1e-7):
            raise RuntimeError(f'Baseline replay failed reproducibility check: maximum state difference {difference}')
        error = np.linalg.norm(array-source['positions'][:, :1], axis=2)
        report['metrics'] = {n: dict(RMSE_m=float(np.sqrt(np.mean(error[:, k]**2))),
            P95_m=float(np.percentile(error[:, k], 95)), max_m=float(error[:, k].max()), final_m=float(error[-1, k]))
            for k, n in enumerate(['DR', 'Local', 'Global'])}
        for name, expected in report['source_SHA256'].items():
            if sha(runtime/name) != expected: raise RuntimeError('Replay source changed while running: '+name)
        if sha(Path(__file__)) != report['replay_driver_SHA256']:
            raise RuntimeError('Replay driver changed while running.')
        if sha(writer_path) != report['strict_report_writer_SHA256']:
            raise RuntimeError('Strict report writer changed while running.')
        for path, expected in dependencies.items():
            if sha(path) != expected: raise RuntimeError('Frozen dependency changed during replay: '+str(path))
        np.savez_compressed(a.out/'estimated_states.npz', positions=array, timestamps=stamps,
                            yaw_rad=np.asarray(yaws), distance_m=source['distance_m'])
        (a.out/'graph_factors.json').write_text(json.dumps(nav.graph.factor_stream_payload(), indent=2)+'\n')
        if a.loop_closure:
            loop_trace, _ = sanitize_for_strict_json(nav.historical_loops.trace)
            (a.out/'historical_loop_trace.json').write_text(json.dumps(loop_trace, indent=2, allow_nan=False)+'\n')
            for k, example in enumerate(nav.historical_loops.examples):
                np.savez_compressed(a.out/f'actual_historical_loop_{k}.npz', **example)
        diagnostics, _ = sanitize_for_strict_json(nav.diagnostics())
        report.update(status='completed_lossless_navigation_input_replay', elapsed_s=time.monotonic()-started,
            diagnostics=diagnostics, DR_and_Local_bit_exact=True, maximum_difference_from_original_m=difference,
            evaluation_truth_read_only_after_estimation=True,
            full_native_navigation_inputs_verified=True, source_snapshots_verified=True)
        save_report(); print(json.dumps(dict(status=report['status'], metrics=report['metrics'],
            maximum_difference_m=difference, diagnostics=diagnostics), ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(status='failed_preserved_replay_no_automatic_restart', error=f'{type(exc).__name__}: {exc}',
                      elapsed_s=time.monotonic()-started)
        save_report(); raise


if __name__ == '__main__': main()
