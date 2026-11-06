"""Run the original FULL joint estimator on the ALL2020-derived comb scene.

Only scene/virtual motion/event transport differ. Feature odometry, cohort
particles, bathymetry+backscatter matching and GTSAM stay source-hash frozen.
No GPS or simulator world coordinates are passed to the estimator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from standalone.hil.frozen_engine import load_engine
from standalone.hil.replay import verify_bundle
from standalone.hil.protocol import acoustic_payload, EventOrder, pack
from RunQiandaoNoGPSNavigation import sha, dump, summarize, setup_arguments
from QiandaoLocalScene import reject_gps_file
from QiandaoLocalRaycaster import intersect_local_grid
from QiandaoVirtualNavigation import measured_field_sensor_class
from QiandaoCombTrajectory import CombSurveyTrajectory, CombSurveyParameters


def iter_sensor_schedule(duration_s, ping_period_s):
    """Same event times/order as the pilot, O(4) instead of O(million) RAM."""
    if not np.isfinite([duration_s, ping_period_s]).all() or duration_s <= 0 or ping_period_s <= 0:
        raise ValueError('Finite positive duration and sonar period required.')
    queue = []
    for kind, period, priority in [('imu', .02, 0), ('dvl', .2, 1), ('depth', .1, 2), ('sonar', ping_period_s, 3)]:
        first = 0 if kind == 'sonar' else 1
        last = int(math.floor(duration_s/period))
        if first <= last:
            heapq.heappush(queue, (round(first*period, 9), priority, kind, first, last, period))
    while queue:
        stamp, priority, kind, index, last, period = heapq.heappop(queue)
        yield stamp, kind
        if index < last:
            heapq.heappush(queue, (round((index+1)*period, 9), priority, kind, index+1, last, period))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--seed', type=int, default=31)
    p.add_argument('--distance-m', type=float, help='Only for separately labelled pilot; omitted runs the FULL closed loop.')
    a = p.parse_args()
    sys.addaudithook(reject_gps_file)
    if a.out.exists():
        raise FileExistsError('No overwrite/restart of a prior full-route result.')
    if 'oceansim' in sys.modules:
        raise RuntimeError('Fresh process required for the exact frozen estimator modules.')
    experiment, out = a.experiment.resolve(), a.out.resolve()
    scene = json.loads((experiment/'scene/scene_report.json').read_text())
    if scene['source_records'] != 2020 or scene['shared_record_IDs'] or scene['GPS_file_opened']:
        raise ValueError('ALL2020 GPS-free disjoint physical input records required.')
    route_spec = json.loads((experiment/'AUV全幅梳形轨迹.json').read_text())
    trajectory = CombSurveyTrajectory(CombSurveyParameters(**route_spec['parameters']))
    distance = trajectory.length_m if a.distance_m is None else a.distance_m
    if not 0 < distance <= trajectory.length_m:
        raise ValueError('One complete programmed loop, or a labelled shorter pilot, only.')
    duration = distance/trajectory.speed_mps
    bundle = experiment/'estimator_bundle'
    manifest = verify_bundle(bundle, 'lake')
    native_bbs = ROOT/'native/bbs3d_bridge/build/liboceansim_bbs3d.so'
    if not native_bbs.is_file():
        raise FileNotFoundError('Original native3D-BBS verification library required.')
    os.environ.setdefault('OCEANSIM_BBS3D_LIBRARY', str(native_bbs))
    sys.path.insert(0, str(bundle/'isaacsim'))
    namespace, extraction = load_engine(bundle/'standalone/large_scale_slam_isaac_demo.py', 'lake', bundle)
    from oceansim.slam import MeasuredBathymetryGrid, NavigationInitialization, sanitize_for_strict_json
    from oceansim.slam.math3d import rpy_from_rotation
    from scipy.ndimage import binary_erosion
    ping_period = float(np.median(np.diff(np.load(experiment/'scene/physical_record_timestamps_s.npy'))))
    args, configuration = setup_arguments(namespace, bundle, seed=a.seed, ping_hz=1/ping_period)
    if not args.enable_terrain_map_matching or not args.enable_backscatter_map_matching:
        raise ValueError('Both original joint global matching branches must be enabled.')
    q = MeasuredBathymetryGrid.from_path(experiment/'scene/query_baseline')
    class QueryGrid(MeasuredBathymetryGrid):
        def intersect_rays(self, origin_world, directions_world, *, min_range, max_range, iterations=24):
            return intersect_local_grid(self, origin_world, directions_world,
                                       min_range=min_range, max_range=max_range, iterations=iterations)
    query = QueryGrid(q.elevation_m, resolution_m=q.resolution_m, valid_mask=q.backscatter_valid_mask,
        backscatter=q.backscatter, backscatter_valid_mask=q.backscatter_valid_mask, metadata=q.metadata)
    query.ray_support_mask = binary_erosion(q.backscatter_valid_mask, iterations=2)
    reference = MeasuredBathymetryGrid.from_path(bundle/'map')
    sensor = measured_field_sensor_class()(query, trajectory, num_beams=96, swath_width_deg=160.,
        min_range=0., max_range=250., sidescan_range_bins=4096, imaging_rays_per_side=256,
        along_track_rows_per_side=8, along_track_sample_spacing_m=.15,
        subgrid_reflectivity_amplitude=0., seed=a.seed)
    first = trajectory.state(0.)
    startup = np.array([sensor.imu(first)[1] for _ in range(32)])
    initial = NavigationInitialization(rotation=first.rotation.copy(), position=first.position.copy(),
                                       velocity=first.velocity.copy())
    engine = namespace['create_engine'](args, reference, initial, startup)
    next(engine)
    out.mkdir()
    snapshot = out/'source_snapshot'
    source_names = ('RunQiandaoFullFrameCombNavigation.py', 'RunQiandaoNoGPSNavigation.py',
        'QiandaoCombTrajectory.py', 'QiandaoLocalRaycaster.py', 'QiandaoVirtualNavigation.py',
        'QiandaoLocalScene.py', 'hil/frozen_engine.py', 'hil/protocol.py')
    for name in source_names:
        dest = snapshot/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent/name, dest)
    report = dict(status='running', experiment='ALL2020_joint_measured_depth_denoised_side_scan_comb',
        source_physical_records=2020, original_estimator_source_commit=manifest['source_commit'],
        full_frozen_estimator=True, extraction=extraction, configuration=configuration,
        both_depth_and_side_scan_global_matching_enabled=True,
        native_bbs_SHA256=sha(native_bbs), seed=a.seed,
        requested_distance_m=distance, full_closed_loop_requested=a.distance_m is None,
        complete_planned_loop_length_m=trajectory.length_m, simulated_duration_s=duration,
        virtual_rate_Hz=dict(IMU=50, DVL=5, depth=10, sonar=1/ping_period),
        reference_records='even1010', query_records='odd1010', reference_query_depth_shared=False,
        full2020_display_map_is_NOT_estimator_reference=True,
        GPS_used=False, after_initialization_truth_accessible_to_estimator=False,
        initial_condition='controlled exact virtual initial state, not GPS',
        evaluation_truth='programmed virtual AUV, NOT original vessel field truth',
        new_comb_measurements_are_simulated_from_measured_scene_not_real_comb_acquisitions=True,
        calibrated_real_world_bathymetry_or_reflectivity_claimed=False,
        algorithm_intensity_input='frozen native floating amplitude, never PNG/PDF',
        added_random_texture=False, processed_pings=0, event_counts={},
        source_code_SHA256={name: sha(snapshot/name) for name in source_names},
        scene_report_SHA256=sha(experiment/'scene/scene_report.json'),
        expected_events={kind: int(math.floor(duration/period))+(kind=='sonar')
                         for kind, period in [('imu', .02), ('dvl', .2), ('depth', .1), ('sonar', ping_period)]})
    started = time.monotonic()
    rows, rotations_history, latencies = [], [], []
    hashes = {n: hashlib.sha256() for n in ('navigation', 'sonar_geometry', 'sonar_intensity')}
    order = EventOrder()
    order.accept(dict(schema=1, sequence=0, kind='boot', timestamp=0., payload=dict(
        configuration={}, initial={}, startup_specific_force=startup, baseline_sha=manifest['source_commit'])))
    dump(out/'run_summary.json', report)
    progress = -250.
    cached_stamp, cached_truth = None, None
    header = ['time_s', 'truth_x', 'truth_y', 'truth_z', 'DR_x', 'DR_y', 'DR_z',
        'Local_x', 'Local_y', 'Local_z', 'Global_x', 'Global_y', 'Global_z',
        'DR_error_m', 'Local_error_m', 'Global_error_m', 'terrain_accepts', 'terrain_attempts',
        'feature_filter_updates', 'graph_keyframes', 'distance_m', 'segment_index',
        'truth_roll', 'truth_pitch', 'truth_yaw', 'DR_roll', 'DR_pitch', 'DR_yaw',
        'Local_roll', 'Local_pitch', 'Local_yaw', 'Global_roll', 'Global_pitch', 'Global_yaw']
    try:
        with (out/'trajectory.csv').open('x', newline='') as f, (out/'ping_diagnostics.jsonl').open('x') as detail:
            writer = csv.writer(f)
            writer.writerow(header)
            for sequence, (stamp, kind) in enumerate(iter_sensor_schedule(duration, ping_period), 1):
                if stamp != cached_stamp:
                    cached_stamp, cached_truth = stamp, trajectory.state(stamp)
                truth = cached_truth
                if kind == 'imu':
                    gyro, acceleration = sensor.imu(truth)
                    payload = dict(gyro=gyro, acceleration=acceleration)
                elif kind == 'dvl':
                    payload = dict(velocity_body=sensor.dvl(truth))
                elif kind == 'depth':
                    payload = dict(depth_m=sensor.depth(truth))
                else:
                    payload = acoustic_payload(sensor.bathymetric_sidescan_ping(truth))
                    for name in ('points_sensor', 'ranges', 'hit_mask'):
                        hashes['sonar_geometry'].update(np.ascontiguousarray(payload['bathymetry'][name]).tobytes())
                    hashes['sonar_intensity'].update(np.ascontiguousarray(payload['imaging']['intensities']).tobytes())
                event = dict(schema=1, sequence=sequence, kind=kind, timestamp=stamp, payload=payload)
                order.accept(event)
                if kind != 'sonar':
                    hashes['navigation'].update(pack(event))
                before = time.perf_counter()
                result = engine.send(event)
                report['event_counts'][kind] = report['event_counts'].get(kind, 0)+1
                if result is None:
                    continue
                latencies.append(time.perf_counter()-before)
                frame = engine.gi_frame.f_locals
                slam = frame['slam']
                diagnostic = slam.diagnostics()
                row = [stamp, *truth.position, *result['dr_position'], *result['local_position'], *result['global_position']]
                error = [float(np.linalg.norm(result[name]-truth.position)) for name in
                         ('dr_position', 'local_position', 'global_position')]
                rotations = np.array([truth.rotation, result['dr_rotation'], result['local_rotation'], result['global_rotation']])
                _, _, _, segment = trajectory.evaluate_distance(stamp*trajectory.speed_mps)
                writer.writerow([*row, *error, result['terrain_prior_accepts'], result['terrain_match_attempts'],
                    diagnostic.feature_filter_updates, result['keyframes'], stamp*trajectory.speed_mps,
                    int(segment), *np.concatenate([rpy_from_rotation(r) for r in rotations])])
                rows.append(row)
                rotations_history.append(rotations)
                entry = dict(timestamp=stamp, distance_m=stamp*trajectory.speed_mps, segment=int(segment), error_m=error,
                    hits=int(np.count_nonzero(payload['imaging']['hit_mask'])),
                    terrain_accepts=result['terrain_prior_accepts'], terrain_attempts=result['terrain_match_attempts'],
                    feature_diagnostics=asdict(slam.feature_odometry.diagnostics()),
                    backscatter_match_stats=frame.get('backscatter_match_stats', {}),
                    terrain_output_statistics=frame.get('terrain_navigation_output_stats', {}),
                    particle_statistics=frame.get('terrain_backend_particle_stats', {}))
                clean, _ = sanitize_for_strict_json(entry)
                detail.write(json.dumps(clean, allow_nan=False)+'\n')
                report['processed_pings'] += 1
                if report['processed_pings'] % 50 == 0:
                    f.flush(); detail.flush()
                    report.update(elapsed_wall_s=time.monotonic()-started, last_timestamp_s=stamp,
                        last_distance_m=stamp*trajectory.speed_mps, last_errors_m=error,
                        terrain_accepts=result['terrain_prior_accepts'], terrain_attempts=result['terrain_match_attempts'],
                        last_backscatter_match_stats=entry['backscatter_match_stats'])
                    dump(out/'run_summary.json', report)
                    if shutil.disk_usage(out).free < 512*1024**2:
                        raise RuntimeError('Disk safety reserve reached; preserve all recorded progress.')
                if stamp*trajectory.speed_mps-progress >= 250:
                    progress = stamp*trajectory.speed_mps
                    print(json.dumps(dict(distance_km=progress/1000, total_km=distance/1000,
                        DR_Local_Global_m=error, pings=report['processed_pings'], segment=int(segment),
                        terrain=result['terrain_prior_accepts'], wall_s=time.monotonic()-started)), flush=True)
            frame = engine.gi_frame.f_locals
            trace = {name: frame[name] for name in ('terrain_backend_particle_records',
                'terrain_recovery_invocation_records', 'terrain_consistency_records', 'backscatter_match_stats',
                'terrain_gauge_state') if name in frame}
            clean, _ = sanitize_for_strict_json(trace)
            dump(out/'frozen_backend_trace.json', clean)
            finish = dict(schema=1, sequence=sequence+1, kind='finish', timestamp=duration+.001, payload={})
            order.accept(finish)
            try:
                engine.send(finish)
            except StopIteration as stop:
                diagnostics = stop.value
            else:
                raise RuntimeError('Frozen estimator did not terminate at the full scheduled duration.')
        for name in source_names:
            if sha(Path(__file__).parent/name) != report['source_code_SHA256'][name]:
                raise RuntimeError('Executed adapter source changed while live.')
        if report['expected_events'] != report['event_counts']:
            raise RuntimeError('Not all scheduled sensor events were processed.')
        clean, _ = sanitize_for_strict_json(diagnostics)
        dump(out/'estimator_diagnostics.json', clean)
        states = np.asarray(rows)
        np.savez_compressed(out/'navigation_states.npz', timestamps=states[:, 0],
            positions=states[:, 1:].reshape(-1, 4, 3), rotations=np.asarray(rotations_history))
        report.update(status='completed_ALL_scheduled_events_full_frozen_joint_estimator',
            elapsed_wall_s=time.monotonic()-started, metrics=summarize(rows, latencies),
            trajectory_SHA256=sha(out/'trajectory.csv'), navigation_states_SHA256=sha(out/'navigation_states.npz'),
            input_hashes={name: h.hexdigest() for name, h in hashes.items()},
            old_algorithm_source_unchanged=True, full_schedule_event_counts_verified=True)
        dump(out/'run_summary.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(status='failed_preserved_no_automatic_restart', error=type(exc).__name__+': '+str(exc),
                      elapsed_wall_s=time.monotonic()-started)
        dump(out/'run_summary.json', report)
        raise


if __name__ == '__main__':
    main()
