"""Independent source-conservation and AFTER-estimation pose review.

Does not call the projection builder/matcher. Reconstructs source calibration
from saved estimator poses and raw received amplitudes. Evaluation truth is
read only after a completed, immutable diagnostic has been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def matrix(p):
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array([[c, -s, p[0]], [s, c, p[1]], [0., 0., 1.]])


def verify_received_projection(query, received, estimate, calibration):
    ids = query['physical_ping_observation_ids'].astype(int)
    poses = query['contributing_local_pose2']
    np.testing.assert_array_equal(poses[:, :2], estimate['positions'][ids-1, 1, :2])
    np.testing.assert_array_equal(poses[:, 2], estimate['yaw_rad'][ids-1, 1])
    anchor = query['anchor_pose2'] if 'anchor_pose2' in query else poses[-1]
    inverse = np.linalg.inv(matrix(anchor))
    origins, normals, laterals = [], [], []
    for identifier, pose in zip(ids, poses):
        rel = inverse@matrix(pose); forward = rel[:2, 0]
        xy = rel[:2, 2]+np.asarray(calibration['row_centres_m'])[:, None]*forward[None]
        z = estimate['positions'][identifier-1, 1, 2]-query['anchor_vertical_z_m']
        origins.append(np.column_stack([xy, np.full(8, z)])); normals.append(np.tile(forward, (8, 1)))
        laterals.append(np.tile(rel[:2, 1], (8, 1)))
    origins, normals, laterals = np.concatenate(origins), np.concatenate(normals), np.concatenate(laterals)
    np.testing.assert_allclose(origins, query['phase_origins'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(normals, query['phase_normals']*query['phase_side_sign'][:, None], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(query['phase_packet_ids'], np.repeat(ids, 8))
    np.testing.assert_array_equal(query['phase_beam_rows'], np.tile(np.arange(8), len(ids)))
    np.testing.assert_array_equal(query['phase_timestamps_s'], np.repeat(estimate['timestamps'][ids-1], 8))
    bins, max_range = int(query['bins']), float(query['max_range_m'])
    if bins != calibration['bins']: raise ValueError('Native-grid review does not silently resample.')
    block = received[ids-1]
    native_count = int(np.count_nonzero(block > 0))
    # Reproduce the DECLARED shared range rasterizer independently: an exact
    # observed bin currently requires an adjacent next acquisition. This
    # excludes isolated/last bins even on a native grid. Report that loss
    # explicitly; do not pretend the projector used every raw received bin.
    next_present = np.concatenate([block[..., 1:] > 0, np.zeros(block.shape[:-1]+(1,), bool)], axis=-1)
    sampled = np.where((block > 0)&next_present, block, 0.)
    raster = np.concatenate([sampled[:, 0, :, ::-1], sampled[:, 1]], axis=-1).reshape(-1, 2*bins)
    signed = np.r_[(bins-.5-np.arange(bins))*max_range/bins, -(np.arange(bins)+.5)*max_range/bins]
    # Use independently verified source phase centres, retaining their exact
    # stored floating representation at quantization boundaries.
    point = query['phase_origins'][:, None, :2]+signed[None, :, None]*laterals[:, None]
    row = np.rint((point[:, :, 0]-query['x0'])/query['step']).astype(int)
    col = np.rint((query['grid_y_max_m']-point[:, :, 1])/query['grid_step_y_m']).astype(int)
    mask = raster > 0; h, w = query['support'].shape
    flat = row[mask]*w+col[mask]; value = raster[mask]
    count = np.bincount(flat, minlength=h*w).reshape(h, w)
    np.testing.assert_array_equal(count, query['observation_count'])
    np.testing.assert_array_equal(count > 0, query['support'])
    sums = np.bincount(flat, weights=value, minlength=h*w).reshape(h, w)
    expected = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    np.testing.assert_allclose(expected, query['assembled_amplitude'], rtol=0, atol=1e-12)
    occupied = query['support']; phase = query['source_phase_centre'][occupied]
    source_column = query['source_range_column'][occupied].astype(int)
    if np.any(phase < 0): raise ValueError('Supported pixel has no real source.')
    selected = raster[phase, source_column]
    maximum = np.zeros(h*w); np.maximum.at(maximum, flat, value)
    np.testing.assert_array_equal(selected, maximum.reshape(h, w)[occupied])
    occupied_flat = np.flatnonzero(occupied)
    source_flat = row[phase, source_column]*w+col[phase, source_column]
    np.testing.assert_array_equal(source_flat, occupied_flat)
    return dict(independent_raw_projection_counts_verified=True, amplitude_blends_verified=True,
        every_labeled_feature_source_is_a_real_maximum_contribution=True,
        source_scan_geometry_checked_against_estimator_poses_and_calibration=True,
        physical_pings=len(ids), received_contributions=int(mask.sum()), occupied_pixels=int(occupied.sum()),
        native_observed_bin_count=native_count,
        native_bins_excluded_by_existing_shared_resampler=native_count-int(mask.sum()),
        projection_preserves_all_declared_raster_contributions=True,
        no_unknown_pixels_filled=bool(np.all(query['assembled_amplitude'][~occupied] == 0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic', required=True, type=Path)
    p.add_argument('--source-run', required=True, type=Path)
    p.add_argument('--estimator-replay', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError('Preserve earlier independent reviews.')
    report = json.loads((a.diagnostic/'diagnostic.json').read_text())
    if report['status'] != 'completed_single_window_received_feature_diagnostic':
        raise ValueError('Truth evaluation requires a completed diagnostic, never an active fit.')
    for name, expected in report['source_SHA256'].items():
        if sha(a.diagnostic/'source_snapshot'/name) != expected: raise ValueError('Executed source snapshot changed.')
    if sha(a.estimator_replay/'estimated_states.npz') != report['estimator_only_pose_archive_SHA256']:
        raise ValueError('Estimator-only source poses changed.')
    manifest_path = a.source_run/'recorded_inputs/input_manifest.json'
    if sha(manifest_path) != report['inputs_manifest_SHA256']: raise ValueError('Recorded input manifest changed.')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files_SHA256'].items():
        if sha(manifest_path.parent/name) != expected: raise ValueError('Received input archive changed.')
    received = np.load(manifest_path.parent/'received_side_scan_amplitudes.npy', mmap_mode='r')
    estimate = np.load(a.estimator_replay/'estimated_states.npz')
    projected = np.load(a.diagnostic/'projected_received_query.npz')
    source_review = verify_received_projection(projected, received, estimate, manifest['calibration'])
    # The estimator has terminated and provenance has passed. Truth first
    # appears ONLY here, never in image creation, matching or candidate selection.
    source_report = json.loads((a.source_run/'run_summary.json').read_text())
    truth_path = a.source_run/'navigation_states.npz'
    if sha(truth_path) != source_report['navigation_states_SHA256']: raise ValueError('Evaluation truth changed.')
    truth = np.load(truth_path)
    index = int(np.searchsorted(truth['timestamps'], report['timestamp_s']))
    if truth['timestamps'][index] != report['timestamp_s']: raise ValueError('Exact shared observation time required.')
    true_pose = np.r_[truth['positions'][index, 0, :2], truth['yaw_rad'][index, 0]]
    original_path = next(path for path in a.source_run.glob('backend_query_*.npz') if sha(path) == report['selected_saved_query_SHA256'])
    original = np.load(original_path); seed = original['seed_pose2']
    review = dict(status='completed_independent_received_projection_and_POST_estimation_pose_review',
        diagnostic_report_SHA256=sha(a.diagnostic/'diagnostic.json'), source_review=source_review,
        timestamp_s=report['timestamp_s'], seed_XY_error_m=float(np.linalg.norm(seed[:2]-true_pose[:2])),
        evaluation_truth_accessed_only_after_completed_estimation=True, per_variant={},
        navigation_trajectory_accuracy_not_yet_validated=True)
    for variant, record in report['variants'].items():
        result = dict(record)
        if record['accepted']:
            match = np.load(a.diagnostic/f'{variant}_matches.npz'); pose = match['matched_pose2']
            result.update(XY_error_m=float(np.linalg.norm(pose[:2]-true_pose[:2])),
                yaw_error_deg=float(np.rad2deg(np.arctan2(np.sin(pose[2]-true_pose[2]), np.cos(pose[2]-true_pose[2])))))
        review['per_variant'][variant] = result
    a.out.mkdir()
    (a.out/'independent_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = [f.name for f in font_manager.fontManager.ttflist if 'Noto Sans CJK' in f.name]
    if fonts: plt.rcParams['font.family'] = fonts[0]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1., 1., 5.]))
    for row, label in enumerate(['legacy', 'canonical', 'projected']):
        data = np.load(a.diagnostic/f'{label}_postmask_only_matches.npz')
        accepted = bool(data['accepted']); info = review['per_variant'][label+'_postmask_only']
        for col, name in enumerate(['query_image', 'map_image']):
            axes[row, col].imshow(data[name], cmap='gray', vmin=0, vmax=255, aspect='equal')
            key = ('inlier_query_pixels' if col == 0 else 'inlier_map_pixels') if label == 'projected' else (
                'query_pixels' if col == 0 else 'map_pixels')
            point = data[key]
            axes[row, col].scatter(point[:, 0], point[:, 1], s=3, c='#00d4a4' if accepted else '#ff695e')
            axes[row, col].set_title(('收到回波子图' if col == 0 else '先验强度候选')+f'：{label}，匹配{info["matches"]}，'+('几何通过' if accepted else '拒绝'))
            axes[row, col].set(xlabel='显示列 / px', ylabel='显示行 / px')
    fig.suptitle('同一转弯窗口：旧条带与保留观测来源的外观投影\n投影不是海底水平真值图；绿色为通过扫描平面验证的匹配，尚不是整程定位验收')
    fig.savefig(a.out/'转弯侧扫_原条带与测量来源投影_实际特征对照.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(status=review['status'], source_review=source_review,
        seed_XY_error_m=review['seed_XY_error_m'], pose_errors={k: v.get('XY_error_m') for k, v in review['per_variant'].items()}), ensure_ascii=False))


if __name__ == '__main__': main()
