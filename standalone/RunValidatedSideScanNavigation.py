"""Run the tested PC side-scan profile only after the FULL 2x2 audit passes.

This is an explicit entry point, not a change to the lake/sea launchers or
the estimator. Its evidence checks happen offline before a fresh estimator
process; no validation truth or error labels are passed into that process.
Private scenes/recordings stay in place, and existing outputs are never reused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from AnalyzeSideScanLoopReplay import read_replay, sha
from SideScanRecordedInputs import RecordedSideScanInputs
from ValidateProjectedSideScanFullRoute import (VARIANTS, COMPARISONS, validate_full_source,
    verify_runtime, verify_legacy_control)
from standalone.hil.replay import verify_bundle


def validate_complete_matrix_header(status):
    if status.get('status') != 'completed_independent_full_route_2x2_ablation':
        raise ValueError('Full four-group audit is not complete; this entry point will not start an unverified profile.')
    names = [name for name, _, _ in VARIANTS]
    comparisons = [name for name, _, _, _ in COMPARISONS]
    if (status.get('completed_replays') != names or status.get('completed_comparisons') != comparisons
            or not status.get('legacy_control', {}).get('positions_timestamps_yaw_distance_bit_exact')):
        raise ValueError('All four complete groups, four independent comparisons and exact legacy control are required.')


def verify_executed_source_scope(report, truth, experiment, runtime):
    """Recheck actual programmed full-route geometry, not just green metadata."""
    model = runpy.run_path(str(runtime/'standalone/QiandaoCombTrajectory.py'))
    parameters = json.loads((experiment/'AUV全幅梳形轨迹.json').read_text())['parameters']
    route = model['CombSurveyTrajectory'](model['CombSurveyParameters'](**parameters))
    if not np.isclose(report['requested_distance_m'], route.length_m, rtol=0, atol=1e-8):
        raise ValueError('Executed source is not the complete declared route.')
    duration = route.length_m/route.speed_mps
    period = float(np.median(np.diff(np.load(experiment/'scene/physical_record_timestamps_s.npy'))))
    if not np.isfinite(period) or period <= 0: raise ValueError('Valid measured acquisition timing is required.')
    expected = {kind: int(np.floor(duration/step))+(kind == 'sonar')
        for kind, step in [('imu', .02), ('dvl', .2), ('depth', .1), ('sonar', period)]}
    if report['expected_events'] != expected or len(truth['timestamps']) != expected['sonar']:
        raise ValueError('Actual full-route sample coverage does not match programmed acquisition.')
    np.testing.assert_allclose(truth['distance_m'], truth['timestamps']*route.speed_mps, rtol=0, atol=1e-8)
    expected_positions = np.array([route.position(float(t)) for t in truth['timestamps']])
    np.testing.assert_allclose(truth['positions'][:, 0], expected_positions, rtol=0, atol=1e-10)


def validate_matrix(matrix):
    matrix = Path(matrix).resolve()
    status = json.loads((matrix/'validation_status.json').read_text())
    validate_complete_matrix_header(status)  # BEFORE loading reference truth or creating output
    source = Path(status['source_run']).resolve(); experiment = source.parent
    runtime = Path(status['runtime']).resolve()
    if not runtime.is_relative_to(experiment): raise ValueError('The recorded local experiment runtime is required.')
    verify_runtime(runtime, status['runtime_manifest_SHA256'])
    source_report = json.loads((source/'run_summary.json').read_text())
    validate_full_source(source_report)
    if sha(source/'run_summary.json') != status['source_run_summary_SHA256']:
        raise ValueError('Executed source report changed.')
    if sha(source/'navigation_states.npz') != source_report['navigation_states_SHA256']:
        raise ValueError('Executed evaluation states changed.')
    # This verifies all 53 original bundle entries, not just four core files.
    verify_bundle((experiment/'estimator_bundle').resolve(), 'lake')
    archive = RecordedSideScanInputs(source/'recorded_inputs')
    expected_events = source_report['expected_events']
    if archive.manifest['events'] != expected_events: raise ValueError('Source input coverage changed.')
    input_digest = sha(source/'recorded_inputs/input_manifest.json')
    dependencies = {str(experiment/'estimator_bundle/isaacsim/oceansim/slam'/name): digest
        for name, digest in source_report['original_core_source_SHA256'].items()}
    dependencies.update({str(experiment/'scene/reference_baseline'/name): digest
        for name, digest in source_report['prior_intensity_inputs_SHA256'].items()})
    with np.load(source/'navigation_states.npz') as truth:
        verify_executed_source_scope(source_report, truth, experiment, runtime)
        records = {}; audits = {}
        for name, representation, loops in VARIANTS:
            report, audit, state, _ = read_replay(matrix/name, truth)
            if (report['submap_representation'] != representation or report['loop_closure'] != loops
                    or report['events'] != expected_events or report['input_manifest_SHA256'] != input_digest
                    or report['secondary_denoising_used'] or not report['source_snapshots_verified']
                    or report.get('scan_plane_registration') != 'rank2_baseline'
                    or report.get('canonical_submaps', False)
                    or not report.get('all_navigation_observations_reused_losslessly')
                    or not np.isfinite(state['positions']).all()):
                raise ValueError('Recorded-input protocol or complete finite output changed: '+name)
            diagnostics = report['diagnostics']
            forbidden = ['DVL_altitude_observations', 'MBES_points_consumed', 'measured_height_likelihood_terms',
                'measured_height_graph_factors', 'optical_flow_calls', 'raw_intensity_correlation_calls']
            if (any(diagnostics.get(k) != 0 for k in forbidden)
                    or diagnostics.get('secondary_submap_denoising_used') is not False
                    or diagnostics.get('optimizer_backend') != 'gtsam_isam2_incremental_fixed_lag'):
                raise ValueError('Estimator sensor/feature/backend contract changed: '+name)
            for filename, digest in report['source_SHA256'].items():
                if sha(runtime/'standalone'/filename) != digest:
                    raise ValueError('Runtime no longer matches the executed estimator: '+filename)
            if sha(runtime/'standalone/ReplayPureSideScanNavigation.py') != report['replay_driver_SHA256']:
                raise ValueError('Replay driver changed.')
            if sha(runtime/'standalone/WriteSideScanReport.py') != report['strict_report_writer_SHA256']:
                raise ValueError('Executed strict report writer changed.')
            if report['frozen_core_and_prior_dependency_SHA256'] != dependencies:
                raise ValueError('Replay dependencies do not match the executed source core and reference map.')
            records[name], audits[name] = report, audit
    for filename, digest in dependencies.items():
        if sha(filename) != digest: raise ValueError('Frozen estimator/map dependency changed: '+filename)
    if sha(experiment/'scene/scene_report.json') != source_report['simulator_scene_report_SHA256']:
        raise ValueError('Declared measured-derived scene report changed.')
    verify_legacy_control(source, matrix/'legacy_loops_control')
    for name, old, new, kind in COMPARISONS:
        review = json.loads((matrix/('review_'+name)/'independent_comparison.json').read_text())
        if (review['status'] != 'completed_independent_identical_input_'+kind+'_comparison'
                or review['comparison_kind'] != kind or not review['all_events_and_input_SHA256_verified']
                or not review['frontend_and_DR_bit_exact'] or Path(review['source_run']).resolve() != source):
            raise ValueError('Independent comparison does not cover the completed source: '+name)
        for label, run_name in [('reference', old), ('candidate', new)]:
            if (review[label]['metrics'] != audits[run_name]['metrics']
                    or review[label]['output_SHA256'] != audits[run_name]['output_SHA256']):
                raise ValueError('Saved comparison no longer matches actual full outputs: '+name)
    selected = records['projected_loops']
    rmse = {n: audits[n]['metrics']['Global']['RMSE_m'] for n, _, _ in VARIANTS}
    if not (rmse['projected_loops'] < selected['metrics']['DR']['RMSE_m']
            and all(rmse['projected_loops'] <= v for n, v in rmse.items() if n != 'projected_loops')):
        raise ValueError('Proposed fixed profile did not outperform the completed controls; review before adopting it.')
    return dict(source=source, experiment=experiment, runtime=runtime, source_report=source_report,
        backend_provenance=selected['diagnostics']['backend_feature_matcher'],
        global_RMSE_m=rmse, validated_source_seed=source_report['seed'])


def build_command(verified, out, *, mode='simulate', seed=None, distance_m=None, python=sys.executable):
    out = Path(out).resolve()
    if out.exists(): raise FileExistsError('A fresh output directory is required; never overwrite validation or earlier runs.')
    runtime, experiment, source = verified['runtime'], verified['experiment'], verified['source']
    if mode == 'replay':
        if seed is not None or distance_m is not None:
            raise ValueError('A replay reuses the COMPLETE recorded input stream; seed/prefix overrides are not allowed.')
        command = [str(python), str(runtime/'standalone/ReplayPureSideScanNavigation.py'),
            '--experiment', str(experiment), '--run', str(source), '--out', str(out), '--runtime', 'current']
    elif mode == 'simulate':
        actual_seed = verified['validated_source_seed'] if seed is None else seed
        if actual_seed < 0: raise ValueError('A nonnegative simulation seed is required.')
        if distance_m is not None and (not np.isfinite(distance_m) or distance_m <= 0
                or distance_m > verified['source_report']['requested_distance_m']):
            raise ValueError('A prefix must be finite, positive and within the complete declared route.')
        command = [str(python), str(runtime/'standalone/RunPureSideScanFeatureShort.py'),
            '--experiment', str(experiment), '--out', str(out), '--seed', str(actual_seed),
            '--observations', 'complex_IQ', '--feature-matcher', 'LoFTR']
        command += ['--full-route'] if distance_m is None else ['--distance-m', str(distance_m)]
    else: raise ValueError('Choose an explicit simulation or recorded-input replay.')
    return command+['--submap-representation', 'adaptive_projected', '--loop-closure']


def verify_feature_environment(provenance):
    import torch
    import kornia
    if not torch.cuda.is_available(): raise RuntimeError('The validated profile requires CUDA; no silent CPU fallback.')
    if str(torch.__version__) != provenance['torch_version'] or kornia.__version__ != provenance['kornia_version']:
        raise RuntimeError('Feature library versions differ from the executed validation; revalidate explicitly.')
    weight = Path(torch.hub.get_dir())/'checkpoints/loftr_outdoor.ckpt'
    if not weight.is_file() or sha(weight) != provenance['checkpoint_SHA256']:
        raise RuntimeError('Validated local LoFTR weights are missing/changed; no implicit model replacement.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation', required=True, type=Path, help='Completed full 2x2 matrix directory.')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--mode', choices=('simulate', 'replay'), default='simulate')
    p.add_argument('--seed', type=int, help='Simulation only. New seeds are experiments, not automatically validated.')
    p.add_argument('--distance-m', type=float, help='Explicit simulation prefix; default is the COMPLETE route.')
    p.add_argument('--dry-run', action='store_true', help='Verify completed evidence and print command without launching.')
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError('Preserve earlier output directories.')
    verified = validate_matrix(a.validation)
    command = build_command(verified, a.out, mode=a.mode, seed=a.seed, distance_m=a.distance_m)
    verify_feature_environment(verified['backend_provenance'])
    print(json.dumps(dict(status='verified_dry_run' if a.dry_run else 'starting_verified_profile', command=command,
        validated_Global_RMSE_m=verified['global_RMSE_m'],
        scope='Measured-derived simulated scene, controlled initialization; not independent field-navigation validation.'),
        ensure_ascii=False), flush=True)
    if a.dry_run: return
    a.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run(command, env=environment, cwd=ROOT, check=True)


if __name__ == '__main__': main()
