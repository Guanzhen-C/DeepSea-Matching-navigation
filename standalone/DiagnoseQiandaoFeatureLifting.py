"""OFFLINE diagnostic only: original feature code with perfect motion seeds.

No online estimator is attached/changed. Same scene/sensor stochastic schedule
prefix, original CUDA consensus, and an output-preserving optical-flow observer
distinguish continuous KLT flow from the snapped 3-D correspondence positions.
Perfect seeds are an offline controlled test, NOT navigation observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import binary_erosion

from QiandaoCombTrajectory import CombSurveyTrajectory, CombSurveyParameters
from QiandaoLocalRaycaster import intersect_local_grid
from QiandaoVirtualNavigation import measured_field_sensor_class
from RunQiandaoFullFrameCombNavigation import iter_sensor_schedule
from RunQiandaoNoGPSNavigation import sha


def stats(a):
    a = np.asarray(a)
    return dict(count=len(a), mean=float(np.mean(a)), median=float(np.median(a)),
                P05=float(np.percentile(a, 5)), P95=float(np.percentile(a, 95)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--frames', type=int, default=160)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Do not overwrite diagnostic evidence.')
    bundle = a.experiment/'estimator_bundle'
    sys.path.insert(0, str(bundle/'isaacsim'))
    from oceansim.slam import MeasuredBathymetryGrid
    from oceansim.slam.feature_sonar_odometry import FeatureSonarOdometry, FeatureSonarOdometryConfig
    q = MeasuredBathymetryGrid.from_path(a.experiment/'scene/query_baseline')
    class Query(MeasuredBathymetryGrid):
        def intersect_rays(self, origin_world, directions_world, *, min_range, max_range, iterations=24):
            return intersect_local_grid(self, origin_world, directions_world,
                min_range=min_range, max_range=max_range, iterations=iterations)
    world = Query(q.elevation_m, resolution_m=q.resolution_m, valid_mask=q.backscatter_valid_mask,
        backscatter=q.backscatter, backscatter_valid_mask=q.backscatter_valid_mask, metadata=q.metadata)
    world.ray_support_mask = binary_erosion(q.backscatter_valid_mask, iterations=2)
    route = CombSurveyTrajectory(CombSurveyParameters(**json.loads((a.experiment/'AUV全幅梳形轨迹.json').read_text())['parameters']))
    period = float(np.median(np.diff(np.load(a.experiment/'scene/physical_record_timestamps_s.npy'))))
    sensor = measured_field_sensor_class()(world, route, num_beams=96, swath_width_deg=160.,
        min_range=0., max_range=250., sidescan_range_bins=4096, imaging_rays_per_side=256,
        along_track_rows_per_side=8, along_track_sample_spacing_m=.15,
        subgrid_reflectivity_amplitude=0., seed=31)
    for _ in range(32):
        sensor.imu(route.state(0.))
    feature = FeatureSonarOdometry(FeatureSonarOdometryConfig(compute_backend='cuda'))
    cv2 = feature._opencv()
    original_flow = cv2.calcOpticalFlowPyrLK
    previous, previous_pose = None, None
    records, snapped_all, continuous_all, corrections = [], [], [], []
    started = time.monotonic()
    for stamp, kind in iter_sensor_schedule((a.frames-1)*period, period):
        truth = route.state(stamp)
        if kind == 'imu': sensor.imu(truth); continue
        if kind == 'dvl': sensor.dvl(truth); continue
        if kind == 'depth': sensor.depth(truth); continue
        frame = feature._canonical_frame(sensor.bathymetric_sidescan_ping(truth))
        frame.rotation_world_sensor = truth.rotation.copy()
        pose = np.eye(4); pose[:3, :3] = truth.rotation; pose[:3, 3] = truth.position
        if previous is not None:
            predicted = np.linalg.inv(previous_pose)@pose
            captured = []
            def observe(*args, **kwargs):
                result = original_flow(*args, **kwargs)
                captured.append((args[2].copy(), result[0].copy() if result[0] is not None else None))
                return result
            cv2.calcOpticalFlowPyrLK = observe
            try:
                pairs = feature._klt_pairs(previous, frame, predicted)
            finally:
                cv2.calcOpticalFlowPyrLK = original_flow
            snapped, continuous = [], []
            if captured and captured[0][1] is not None:
                source, target = captured[0]
                flow = {tuple(s[0].tolist()): t[0] for s, t in zip(source, target)}
                for ci, pi in pairs:
                    c = frame.keypoint_pixels[ci].astype(np.float32)
                    t = flow[tuple(c.tolist())]
                    continuous.append(float(t[1]-c[1])*.15/feature.config.image_row_scale)
                    snapped.append(float(previous.keypoint_points_sensor[pi, 0]-frame.keypoint_points_sensor[ci, 0]))
            pair = feature._estimate_pair(previous, frame, predicted)
            applied = None
            if pair.accepted:
                fitted = pair.transform_previous_sensor_current_sensor@np.linalg.inv(predicted)
                applied = float(fitted[0, 3]); corrections.append(applied)
            gt = float(predicted[0, 3])
            snapped_all.extend(np.asarray(snapped)-gt)
            continuous_all.extend(np.asarray(continuous)-gt)
            records.append(dict(timestamp_s=stamp, true_step_x_m=gt, matches=len(pairs),
                pair_accepted=pair.accepted, pair_rmse_m=float(pair.rmse_m),
                mean_snapped_step_m=float(np.mean(snapped)) if snapped else None,
                mean_continuous_KLT_step_m=float(np.mean(continuous)) if continuous else None,
                original_applied_along_correction_m=applied))
        previous, previous_pose = frame, pose
    if not records or not snapped_all or not corrections:
        raise RuntimeError('No diagnostic correspondences; do not infer quantization from code alone.')
    values, counts = np.unique(np.rint((np.asarray(snapped_all)+period*.8)/.15).astype(int), return_counts=True)
    report = dict(status='completed_offline_original_feature_lifting_diagnostic_NOT_a_navigation_rerun',
        perfect_motion_seed_only_for_offline_isolation=True, main_estimator_or_source_changed=False,
        recorded_sensor_schedule_prefix_replayed=True, actual_compute_backend=feature._active_compute_backend,
        frames=len(records)+1, true_step_m=period*.8, physical_row_spacing_m=.15,
        continuous_KLT_step_bias_m=stats(continuous_all), snapped_3D_step_bias_m=stats(snapped_all),
        accepted_original_along_corrections_m=stats(corrections),
        integer_row_difference_histogram={str(k): int(v) for k, v in zip(values, counts)},
        records=records, diagnostic_code_SHA256=sha(Path(__file__)),
        estimator_feature_code_SHA256=sha(bundle/'isaacsim/oceansim/slam/feature_sonar_odometry.py'),
        elapsed_wall_s=time.monotonic()-started)
    a.out.mkdir()
    (a.out/'原特征前端_连续光流与三维回填诊断.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
