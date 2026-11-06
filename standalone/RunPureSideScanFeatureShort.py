"""Short, paired side-scan FEATURE-only navigation test. Keeps old run alive.

Same simulated physical world and sensor seed as the joint baseline. World
geometry is private to the simulator; navigation receives SSS range/amplitude,
IMU, DVL velocity and pressure depth only. No DVL-altitude/MBES/flow input.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from RunQiandaoNoGPSNavigation import dump, sha
from RunQiandaoFullFrameCombNavigation import iter_sensor_schedule
from QiandaoCombTrajectory import CombSurveyTrajectory, CombSurveyParameters
from QiandaoVirtualNavigation import measured_field_sensor_class
from QiandaoLocalRaycaster import intersect_local_grid
from QiandaoLocalScene import reject_gps_file
from PureSideScanFeatures import packet_from_simulated_side_scan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--distance-m', type=float, default=400.)
    a = p.parse_args()
    sys.addaudithook(reject_gps_file)
    if a.out.exists() or not 0 < a.distance_m <= 1300:
        raise ValueError('New separately named short-test output required.')
    sys.path.insert(0, str(a.experiment/'estimator_bundle/isaacsim'))
    from oceansim.slam import MeasuredBathymetryGrid, sanitize_for_strict_json
    from SideScanFeatureMap import IntensityFeatureMap
    from PureSideScanFeatureNavigation import PureSideScanFeatureNavigator
    from scipy.ndimage import binary_erosion
    query_source = a.experiment/'scene/query_baseline'
    # Simulator private world, not a navigation input or measurement.
    q = MeasuredBathymetryGrid.from_path(query_source)
    class SimulationWorld(MeasuredBathymetryGrid):
        def intersect_rays(self, origin_world, directions_world, *, min_range, max_range, iterations=24):
            return intersect_local_grid(self, origin_world, directions_world,
                min_range=min_range, max_range=max_range, iterations=iterations)
    world = SimulationWorld(q.elevation_m, resolution_m=q.resolution_m, valid_mask=q.backscatter_valid_mask,
        backscatter=q.backscatter, backscatter_valid_mask=q.backscatter_valid_mask, metadata=q.metadata)
    world.ray_support_mask = binary_erosion(q.backscatter_valid_mask, iterations=2)
    route = CombSurveyTrajectory(CombSurveyParameters(**json.loads((a.experiment/'AUV全幅梳形轨迹.json').read_text())['parameters']))
    duration = a.distance_m/route.speed_mps
    period = float(np.median(np.diff(np.load(a.experiment/'scene/physical_record_timestamps_s.npy'))))
    sensor = measured_field_sensor_class()(world, route, num_beams=96, swath_width_deg=160.,
        min_range=0., max_range=250., sidescan_range_bins=4096, imaging_rays_per_side=256,
        along_track_rows_per_side=8, along_track_sample_spacing_m=.15,
        subgrid_reflectivity_amplitude=0., seed=31)
    first = route.state(0.)
    startup = np.array([sensor.imu(first)[1] for _ in range(32)])
    # Only scalar image/mask paths are loaded by this estimator map.
    prior = IntensityFeatureMap.from_path(a.experiment/'scene/reference_baseline')
    nav = PureSideScanFeatureNavigator(prior, first, startup)
    a.out.mkdir()
    sources = ['RunPureSideScanFeatureShort.py', 'PureSideScanFeatures.py', 'SideScanFeatureMap.py',
               'PureSideScanFeatureNavigation.py', 'SideScanSubmapDenoising.py']
    snapshot = a.out/'source_snapshot'; snapshot.mkdir()
    for source in sources: shutil.copy2(Path(__file__).parent/source, snapshot/source)
    report = dict(status='running', requested_distance_m=a.distance_m,
        mode='pure_side_scan_descriptor_FEATURE_navigation', original_joint_run_changed=False,
        simulator_world_bathymetry_retained_only_for_fair_same_world_observations=True,
        estimator_map_inputs=['backscatter.npy', 'backscatter_valid_mask.npy', 'resolution_m'],
        observation_pipeline=['denoised_measured_intensity_world', 'pose_dependent_side_scan_returns',
            'causal_historical_received_ping_alignment_using_Local', 'robust_overlap_fusion_denoising',
            'local_intensity_submap', 'descriptor_features'],
        submap_is_NOT_reference_map_crop=True, submap_frozen_complex_IQ_neural_model_used=False,
        frontend='GFTT + subpixel + SIFT descriptors + mutual matches + robust scanline model',
        backend='feature-landmark scan-plane candidates + original cohort tracker + original GTSAM fixed lag',
        optical_flow_used=False, DVL_altitude_used=False, MBES_XYZ_given_to_estimator=False,
        measured_height_likelihood_used=False, direct_intensity_correlation_used=False,
        controlled_exact_initial_state=True, GPS_used=False,
        prior_geocoding_still_uses_existing_nominal_survey_frame=True,
        this_is_new_sensor_adapter_NOT_unchanged_full_joint_engine=True,
        short_test_NOT_full_unknown_initial_pose_global_relocalization_validation=True,
        source_code_SHA256={name: sha(snapshot/name) for name in sources},
        original_core_source_SHA256={name: sha(a.experiment/'estimator_bundle/isaacsim/oceansim/slam'/name)
            for name in ('esikf.py', 'global_tracking.py', 'pose_graph.py', 'map_odom.py')},
        prior_intensity_inputs_SHA256={name: sha(a.experiment/'scene/reference_baseline'/name)
            for name in ('backscatter.npy', 'backscatter_valid_mask.npy', 'metadata.json')},
        simulator_scene_report_SHA256=sha(a.experiment/'scene/scene_report.json'),
        expected_events={kind: int(np.floor(duration/step))+(kind == 'sonar')
            for kind, step in [('imu', .02), ('dvl', .2), ('depth', .1), ('sonar', period)]},
        processed_pings=0, event_counts={})
    dump(a.out/'run_summary.json', report)
    started = time.monotonic()
    errors, rows, yaw_history = [], [], []
    try:
        with (a.out/'trajectory.csv').open('x', newline='') as f, (a.out/'feature_diagnostics.jsonl').open('x') as detail:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'distance_m', *[f'{n}_{c}' for n in ('truth', 'DR', 'Local', 'Global')
                             for c in ('x', 'y', 'z')], 'DR_error_m', 'Local_error_m', 'Global_error_m',
                             'truth_yaw', 'DR_yaw', 'Local_yaw', 'Global_yaw'])
            cached_stamp, truth = None, None
            for stamp, kind in iter_sensor_schedule(duration, period):
                if stamp != cached_stamp:
                    cached_stamp, truth = stamp, route.state(stamp)
                if kind == 'imu':
                    gyro, acc = sensor.imu(truth); nav.imu(stamp, gyro, acc)
                elif kind == 'dvl': nav.dvl(sensor.dvl(truth))
                elif kind == 'depth': nav.depth(sensor.depth(truth))
                else:
                    packet = packet_from_simulated_side_scan(sensor.bathymetric_sidescan_ping(truth))
                    result = nav.side_scan(packet)
                    e = [float(np.linalg.norm(result[n]-truth.position)) for n in ('DR', 'Local', 'Global')]
                    row = [stamp, stamp*route.speed_mps, *truth.position,
                           *result['DR'], *result['Local'], *result['Global'], *e]
                    yaws = [float(np.arctan2(truth.rotation[1, 0], truth.rotation[0, 0])),
                            float(np.arctan2(nav.dr.state.rotation[1, 0], nav.dr.state.rotation[0, 0])),
                            float(result['raw_pose2'][2]), float(result['global_pose2'][2])]
                    writer.writerow([*row, *yaws]); rows.append(row); errors.append(e); yaw_history.append(yaws)
                    entry = dict(timestamp=stamp, error_m=e, frontend=nav.last_feature, backend=nav.last_backend)
                    clean, _ = sanitize_for_strict_json(entry)
                    detail.write(json.dumps(clean, allow_nan=False)+'\n')
                    report['processed_pings'] += 1
                    if report['processed_pings'] % 50 == 0:
                        f.flush(); detail.flush()
                        report.update(last_distance_m=stamp*.8, last_errors_m=e, elapsed_wall_s=time.monotonic()-started,
                                      diagnostics=nav.diagnostics())
                        dump(a.out/'run_summary.json', report)
                    if report['processed_pings'] % 250 == 0:
                        print(json.dumps(dict(distance_m=stamp*.8, errors_m=e, frontend=nav.front_updates,
                            feature_map_accepts=nav.map_accepts, priors=nav.graph_feature_priors)), flush=True)
                report['event_counts'][kind] = report['event_counts'].get(kind, 0)+1
        error = np.asarray(errors)
        if report['expected_events'] != report['event_counts']:
            raise RuntimeError('Missing side-scan/IMU/DVL/pressure events.')
        report['metrics'] = {n: dict(RMSE_m=float(np.sqrt(np.mean(error[:, k]**2))),
            P95_m=float(np.percentile(error[:, k], 95)), max_m=float(error[:, k].max()), final_m=float(error[-1, k]))
            for k, n in enumerate(('DR', 'Local', 'Global'))}
        for source in sources:
            if sha(Path(__file__).parent/source) != report['source_code_SHA256'][source]:
                raise RuntimeError('Executed side-scan adapter source changed during the test.')
        diagnostic = nav.diagnostics()
        clean, _ = sanitize_for_strict_json(diagnostic)
        dump(a.out/'final_diagnostics.json', clean)
        dump(a.out/'feature_graph_factors.json', nav.graph.factor_stream_payload())
        for k, example in enumerate(nav.correspondence_examples):
            np.savez_compressed(a.out/f'actual_feature_matches_{k}.npz', **example)
        for k, example in enumerate(nav.map_correspondence_examples):
            np.savez_compressed(a.out/f'actual_map_feature_matches_{k}.npz', **example)
        for k, example in enumerate(nav.map_failure_examples):
            np.savez_compressed(a.out/f'actual_map_feature_failures_{k}.npz', **example)
        for k, example in enumerate(nav.backend_query_examples):
            np.savez_compressed(a.out/f'backend_query_{k}.npz', **example)
        state_array = np.asarray(rows)
        np.savez_compressed(a.out/'navigation_states.npz', timestamps=state_array[:, 0],
            distance_m=state_array[:, 1], positions=state_array[:, 2:14].reshape(-1, 4, 3),
            yaw_rad=np.asarray(yaw_history))
        report.update(status='completed_pure_side_scan_FEATURE_short_test', elapsed_wall_s=time.monotonic()-started,
            diagnostics=clean, trajectory_SHA256=sha(a.out/'trajectory.csv'),
            navigation_states_SHA256=sha(a.out/'navigation_states.npz'),
            full_source_snapshots_verified=True)
        dump(a.out/'run_summary.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(status='failed_preserved_no_automatic_restart', error=type(exc).__name__+': '+str(exc),
                      elapsed_wall_s=time.monotonic()-started)
        dump(a.out/'run_summary.json', report)
        raise


if __name__ == '__main__':
    main()
