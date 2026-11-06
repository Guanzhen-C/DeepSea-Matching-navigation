"""Run one raw/baseline/edge PC experiment through the FULL frozen estimator.

The estimator is the unchanged lake baseline extracted by the existing HIL
adapter. Only the map, simulated truth/sensors and explicit small-scene scale
configuration change. Never reads GPS or sends live truth/world-frame points.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.hil.frozen_engine import load_engine
from standalone.hil.replay import verify_bundle
from standalone.hil.protocol import acoustic_payload, EventOrder, pack
from QiandaoLocalScene import reject_gps_file
from QiandaoVirtualNavigation import (VirtualAUVTrajectory, measured_field_sensor_class,
                                      deterministic_schedule)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, data):
    p = Path(path)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def setup_arguments(namespace, bundle, *, seed, ping_hz):
    text = (bundle/'standalone/run_v2_full_map_slam_isaac.sh').read_text()
    command = shlex.split(text.split('\nexec ', 1)[1].replace('\\\n', ' '))
    flags = [x for x in command[1:] if x != '$@']
    overrides = ['--headless', '--bathymetry', str(bundle/'map'),
        '--survey-route', str(bundle/'map/full_map_traversal_route.json'),
        '--survey-start-distance-m', '0', '--survey-speed-mps', '.8',
        '--seed', str(seed), '--ping-hz', str(ping_hz),
        '--slam-voxel-size-m', '.6', '--intensity-voxel-size-m', '.6',
        '--feature-compute-backend', 'cuda', '--subgrid-reflectivity-amplitude', '0']
    previous = sys.argv
    try:
        sys.argv = ['frozen-estimator-configuration', *flags, *overrides]
        args = namespace['parse_args']()
    finally:
        sys.argv = previous
    args.terrain_replay_directory = None
    return args, dict(original_launcher_flags=flags, explicit_scene_overrides=overrides,
                      estimator_numerical_source_changed=False)


def summarize(rows, latencies):
    a = np.asarray(rows, float)
    result = {}
    for name, col in [('DR', 4), ('Local', 7), ('Global', 10)]:
        error = np.linalg.norm(a[:, col:col+3]-a[:, 1:4], axis=1)
        result[name] = dict(RMSE_m=float(np.sqrt(np.mean(error**2))),
            median_m=float(np.median(error)), P95_m=float(np.percentile(error, 95)),
            max_m=float(error.max()), final_m=float(error[-1]))
    result['sonar_processing_s'] = dict(mean=float(np.mean(latencies)),
        P95=float(np.percentile(latencies, 95)), max=float(max(latencies)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--variant', choices=('raw', 'baseline', 'edge'), required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--seed', type=int, default=31)
    p.add_argument('--distance-m', type=float, default=250.)
    p.add_argument('--initial-offset-m', type=float, default=0.)
    p.add_argument('--initial-yaw-deg', type=float, default=0.)
    a = p.parse_args()
    sys.addaudithook(reject_gps_file)
    experiment, out = a.experiment.resolve(), a.out.resolve()
    if out.exists() or not 0 < a.distance_m <= 250:
        raise ValueError('New output and an explicit <=250m small-region run required.')
    if 'oceansim' in sys.modules:
        raise RuntimeError('Fresh process required for the exact frozen module versions.')
    bundle = experiment/'estimator_bundle'
    native_bbs = ROOT/'native/bbs3d_bridge/build/liboceansim_bbs3d.so'
    if not native_bbs.is_file():
        raise FileNotFoundError('Existing native 3D-BBS library is required; no algorithm fallback.')
    os.environ.setdefault('OCEANSIM_BBS3D_LIBRARY', str(native_bbs))
    manifest = verify_bundle(bundle, 'lake')
    sys.path.insert(0, str(bundle/'isaacsim'))
    namespace, extraction = load_engine(bundle/'standalone/large_scale_slam_isaac_demo.py', 'lake', bundle)
    from oceansim.slam import MeasuredBathymetryGrid, NavigationInitialization, sanitize_for_strict_json
    from oceansim.slam.math3d import rotation_from_rpy, rpy_from_rotation
    scene_report = json.loads((experiment/'scene/scene_report.json').read_text())
    if scene_report['GPS_file_opened'] or scene_report['shared_record_IDs']:
        raise ValueError('Forbidden GPS or reused map/query records.')
    ping_period = float(np.median(np.diff(np.load(experiment/'scene/physical_record_timestamps_s.npy'))))
    args, configuration = setup_arguments(namespace, bundle, seed=a.seed, ping_hz=1/ping_period)
    # Same geometry AND valid observation support for all three input variants.
    folder = experiment/'scene'/('query_'+a.variant)
    q = MeasuredBathymetryGrid.from_path(folder)
    query = MeasuredBathymetryGrid(q.elevation_m, resolution_m=q.resolution_m,
        valid_mask=q.backscatter_valid_mask, backscatter=q.backscatter,
        backscatter_valid_mask=q.backscatter_valid_mask, metadata=q.metadata)
    reference = MeasuredBathymetryGrid.from_path(bundle/'map')
    trajectory = VirtualAUVTrajectory()
    duration = a.distance_m/trajectory.speed_mps
    for stamp in np.linspace(0, duration, 101):
        state = trajectory.state(stamp)
        bottom = query.seabed_height(*state.position[:2])
        if not np.isfinite(bottom) or state.position[2]-bottom < 5:
            raise ValueError('Programmed virtual route leaves observed support or bottom clearance.')
    Sensor = measured_field_sensor_class()
    sensor = Sensor(query, trajectory, num_beams=96, swath_width_deg=120,
        min_range=0., max_range=250., sidescan_range_bins=4096,
        imaging_rays_per_side=256, along_track_rows_per_side=8,
        along_track_sample_spacing_m=.15, subgrid_reflectivity_amplitude=0., seed=a.seed)
    first = trajectory.state(0.)
    startup = np.array([sensor.imu(first)[1] for _ in range(32)])
    initial = NavigationInitialization(rotation=rotation_from_rpy(0., 0., np.deg2rad(a.initial_yaw_deg))@first.rotation,
        position=first.position+np.array([a.initial_offset_m, 0., 0.]), velocity=first.velocity.copy())
    # Synthetic launch gauge is a declared controlled seed, not GPS; only this
    # t=0 seed is passed. All later estimator events contain measurements only.
    engine = namespace['create_engine'](args, reference, initial, startup)
    next(engine)
    out.mkdir(parents=True)
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    source_names = ('RunQiandaoNoGPSNavigation.py', 'QiandaoVirtualNavigation.py', 'QiandaoLocalScene.py')
    for n in source_names:
        shutil.copy2(Path(__file__).parent/n, snapshot/n)
    started = time.monotonic()
    report = dict(status='running', variant=a.variant, seed=a.seed, requested_distance_m=a.distance_m,
        source_commit=manifest['source_commit'], full_frozen_estimator=True, extraction=extraction,
        native_bbs_SHA256=sha(native_bbs),
        configuration=configuration, reference_map_same_across_groups=True,
        reference_source='even physical records, baseline variant',
        query_source='odd physical records, selected scalar variant',
        GPS_file_opened=False, GPS_input_or_truth=False,
        real_vessel_ground_truth=False, evaluation_truth='programmed virtual AUV trajectory',
        after_initialization_truth_accessible_to_estimator=False,
        reconstructed_scene_not_independent_field_navigation=True,
        source_code_SHA256={n: sha(snapshot/n) for n in source_names},
        scene_report_SHA256=sha(experiment/'scene/scene_report.json'),
        assumed_virtual_sensor_rates_Hz=dict(IMU=50, DVL=5, pressure_depth=10, sonar=1/ping_period),
        simulated_sensor_biases=dict(gyro_rad_s=sensor.gyro_bias.tolist(),
            accel_mps2=sensor.accel_bias.tolist(), DVL_mps=sensor.dvl_bias.tolist()),
        initial_offset_m=a.initial_offset_m, initial_yaw_deg=a.initial_yaw_deg,
        processed_pings=0, event_counts={}, algorithm_intensity_input='float native-derived proxy, never PNG/PDF',
        added_random_texture=False)
    digest = {n: hashlib.sha256() for n in ('navigation_measurements', 'sonar_geometry', 'sonar_appearance')}
    rows, latencies = [], []
    history = []
    order = EventOrder()
    order.accept(dict(schema=1, sequence=0, kind='boot', timestamp=0., payload=dict(
        configuration={}, initial={}, startup_specific_force=startup, baseline_sha=manifest['source_commit'])))
    header = ['time_s', 'truth_x', 'truth_y', 'truth_z', 'DR_x', 'DR_y', 'DR_z',
        'Local_x', 'Local_y', 'Local_z', 'Global_x', 'Global_y', 'Global_z',
        'DR_error_m', 'Local_error_m', 'Global_error_m', 'terrain_accepts', 'terrain_attempts',
        'feature_filter_updates', 'graph_keyframes']
    dump(out/'run_summary.json', report)
    try:
        with (out/'trajectory.csv').open('x', newline='') as f, (out/'ping_diagnostics.jsonl').open('x') as detail:
            writer = csv.writer(f)
            writer.writerow(header)
            for sequence, (stamp, kind) in enumerate(deterministic_schedule(duration, ping_period), 1):
                truth = trajectory.state(stamp)
                if kind == 'imu':
                    gyro, acceleration = sensor.imu(truth)
                    payload = dict(gyro=gyro, acceleration=acceleration)
                elif kind == 'dvl':
                    payload = dict(velocity_body=sensor.dvl(truth))
                elif kind == 'depth':
                    payload = dict(depth_m=sensor.depth(truth))
                else:
                    ping = sensor.bathymetric_sidescan_ping(truth)
                    payload = acoustic_payload(ping)
                    # Hash sensor-frame geometry, not simulator world positions.
                    for n in ('points_sensor', 'ranges', 'hit_mask'):
                        digest['sonar_geometry'].update(np.ascontiguousarray(payload['bathymetry'][n]).tobytes())
                    digest['sonar_appearance'].update(np.ascontiguousarray(payload['imaging']['intensities']).tobytes())
                event = dict(schema=1, sequence=sequence, kind=kind, timestamp=stamp, payload=payload)
                order.accept(event)
                if kind != 'sonar':
                    digest['navigation_measurements'].update(pack(event))
                before = time.perf_counter()
                result = engine.send(event)
                report['event_counts'][kind] = report['event_counts'].get(kind, 0)+1
                if result is None:
                    continue
                elapsed = time.perf_counter()-before
                latencies.append(elapsed)
                state = engine.gi_frame.f_locals
                slam = state['slam']
                diagnostic = slam.diagnostics()
                row = [stamp, *truth.position, *result['dr_position'],
                       *result['local_position'], *result['global_position']]
                error = [float(np.linalg.norm(result[n]-truth.position)) for n in
                         ('dr_position', 'local_position', 'global_position')]
                writer.writerow([*row, *error, result['terrain_prior_accepts'], result['terrain_match_attempts'],
                                 diagnostic.feature_filter_updates, result['keyframes']])
                rows.append(row)
                entry = dict(timestamp=stamp, error_m=error, processing_s=elapsed,
                    terrain_accepts=result['terrain_prior_accepts'], terrain_attempts=result['terrain_match_attempts'],
                    last_feature=asdict(diagnostic.last_feature_odometry) if diagnostic.last_feature_odometry is not None else None,
                    feature_diagnostics=asdict(slam.feature_odometry.diagnostics()),
                    terrain_output_statistics=state.get('terrain_navigation_output_stats', {}),
                    particle_statistics=state.get('terrain_backend_particle_stats', {}))
                clean, _ = sanitize_for_strict_json(entry)
                detail.write(json.dumps(clean, allow_nan=False)+'\n')
                report['processed_pings'] += 1
                if report['processed_pings'] % 50 == 0:
                    f.flush()
                    detail.flush()
                    report.update(elapsed_wall_s=time.monotonic()-started, last_timestamp_s=stamp,
                                  last_errors_m=error, last_terrain_accepts=result['terrain_prior_accepts'])
                    dump(out/'run_summary.json', report)
                    print(json.dumps(dict(variant=a.variant, distance_m=stamp*.8, requested_m=a.distance_m,
                        pings=report['processed_pings'], DR_local_global_m=error,
                        terrain=result['terrain_prior_accepts'], wall_s=report['elapsed_wall_s'])), flush=True)
            finish = dict(schema=1, sequence=sequence+1, kind='finish', timestamp=duration+.001, payload={})
            order.accept(finish)
            try:
                engine.send(finish)
            except StopIteration as stop:
                final_diagnostics = stop.value
            else:
                raise ValueError('Frozen estimator did not finish.')
        for n in source_names:
            if sha(Path(__file__).parent/n) != report['source_code_SHA256'][n]:
                raise ValueError('Executed source changed during experiment.')
        diagnostics, _ = sanitize_for_strict_json(final_diagnostics)
        dump(out/'estimator_diagnostics.json', diagnostics)
        report.update(status='completed_full_frozen_estimator_GPS_free_virtual_AUV_experiment',
            metrics=summarize(rows, latencies), elapsed_wall_s=time.monotonic()-started,
            simulated_duration_s=duration, input_hashes={n: h.hexdigest() for n, h in digest.items()},
            trajectory_SHA256=sha(out/'trajectory.csv'), original_sources_and_algorithm_unchanged=True)
        dump(out/'run_summary.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(status='failed_preserved_no_automatic_restart', error=type(exc).__name__+': '+str(exc),
                      elapsed_wall_s=time.monotonic()-started)
        dump(out/'run_summary.json', report)
        raise


if __name__ == '__main__':
    main()
