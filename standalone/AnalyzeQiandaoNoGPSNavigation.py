"""Independently verify paired runs and plot error curves/3D terrain/matches.

Reports errors ONLY against explicitly programmed virtual truth. These plots
must never be presented as original-vessel GPS/ground-truth field accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LABELS = {'raw': '原始含噪', 'baseline': '固定去噪', 'edge': '去噪＋边缘修复'}
COLOURS = {'raw': '#d95319', 'baseline': '#0072bd', 'edge': '#2e8b57'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024**2), b''):
            h.update(data)
    return h.hexdigest()


def relative_errors(positions, rotations, distance, interval_m):
    end = np.searchsorted(distance, distance+interval_m)
    start = np.flatnonzero(end < len(distance))
    end = end[start]
    output = {}
    truth_delta = np.einsum('nij,nj->ni', rotations[start, 0].transpose(0, 2, 1),
                             positions[end, 0]-positions[start, 0])
    truth_rotation = rotations[start, 0].transpose(0, 2, 1)@rotations[end, 0]
    for channel, name in enumerate(('DR', 'Local', 'Global'), 1):
        delta = np.einsum('nij,nj->ni', rotations[start, channel].transpose(0, 2, 1),
                         positions[end, channel]-positions[start, channel])
        dr = rotations[start, channel].transpose(0, 2, 1)@rotations[end, channel]
        re = truth_rotation.transpose(0, 2, 1)@dr
        angle = np.rad2deg(np.arccos(np.clip((np.trace(re, axis1=1, axis2=2)-1)/2, -1, 1)))
        err = np.linalg.norm(delta-truth_delta, axis=1)
        output[name] = dict(samples=len(start), translation_RMSE_m=float(np.sqrt(np.mean(err**2))),
            translation_P95_m=float(np.percentile(err, 95)), rotation_RMSE_deg=float(np.sqrt(np.mean(angle**2))))
    return output


def verify_run(folder):
    report = json.loads((folder/'run_summary.json').read_text())
    if (report['status'] != 'completed_full_frozen_estimator_GPS_free_virtual_AUV_experiment'
            or report.get('swath_width_deg') != 160 or not report.get('small_map_raycaster')
            or report['GPS_file_opened'] or report['GPS_input_or_truth']):
        raise ValueError('Only completed wide-swath, GPS-free experiments are eligible.')
    if sha(folder/'navigation_states.npz') != report['navigation_states_SHA256']:
        raise ValueError('Saved native pose arrays changed.')
    with np.load(folder/'navigation_states.npz') as data:
        positions = data['positions'].copy()
        rotations = data['rotations'].copy()
        timestamps = data['timestamps'].copy()
    if positions.shape != (947, 4, 3) or rotations.shape != (947, 4, 3, 3):
        raise ValueError('Complete 250m native-ping pose stream required.')
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Finite ordered full pose history required.')
    errors = np.linalg.norm(positions[:, 1:]-positions[:, 0, None], axis=2)
    independently_measured = {}
    for k, name in enumerate(('DR', 'Local', 'Global')):
        e = errors[:, k]
        values = dict(RMSE_m=float(np.sqrt(np.mean(e**2))), median_m=float(np.median(e)),
                      P95_m=float(np.percentile(e, 95)), max_m=float(e.max()), final_m=float(e[-1]))
        for metric, value in values.items():
            if not np.isclose(value, report['metrics'][name][metric], rtol=1e-12, atol=1e-12):
                raise ValueError('Independent full-stream error calculation differs.')
        independently_measured[name] = values
    diagnostic = json.loads((folder/'estimator_diagnostics.json').read_text())
    if diagnostic['graph_diagnostics']['optimizer_backend'] != 'gtsam_isam2_incremental_fixed_lag':
        raise ValueError('Not the required GTSAM backend.')
    if diagnostic['feature_diagnostics']['active_compute_backend'] != 'cuda':
        raise ValueError('Not the requested existing CUDA feature consensus.')
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(positions[:, 0], axis=0), axis=1))]
    rpe = {str(d): relative_errors(positions, rotations, distance, d) for d in (10., 30.)}
    return dict(folder=str(folder), report=report, metrics=independently_measured, rpe=rpe,
        terrain_accepts=diagnostic['terrain_prior_accepts'], terrain_attempts=diagnostic['terrain_match_attempts'],
        feature=diagnostic['feature_diagnostics'], actual_distance_m=float(distance[-1]),
        report_SHA256=sha(folder/'run_summary.json')), (positions, rotations, timestamps, distance, errors)


def plot_first_case(runs, arrays, experiment, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LightSource
    fonts = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if fonts.exists():
        font_manager.fontManager.addfont(fonts)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fonts).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    selected = {r['report']['variant']: r for r in runs
                if r['report']['seed'] == 31 and r['report']['initial_offset_m'] == 0}
    if set(selected) != set(LABELS):
        raise ValueError('All three completed seed31 exact-initialisation inputs required for the representative plots.')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for variant, r in selected.items():
        positions, rotation, timestamps, distance, errors = arrays[r['folder']]
        for k, name in enumerate(('DR', 'Local', 'Global')):
            if name == 'DR' and variant != 'raw':
                continue
            axes[k].plot(distance, errors[:, k], color='#657382' if name == 'DR' else COLOURS[variant],
                         label='同输入纯航位推算' if name == 'DR' else LABELS[variant], linewidth=1.4)
            axes[k].set_ylabel(name+' 误差 / m')
            axes[k].grid(alpha=.25)
            axes[k].legend(loc='upper left', fontsize=9)
    axes[-1].set_xlabel('虚拟AUV累计航程 / m')
    fig.suptitle('实测声呐构景：PC三组导航对照（GPS未使用）\n误差真值为设定的虚拟运动，不是采集船真实航迹', fontsize=14)
    fig.savefig(out/'250m_三组导航误差曲线.png', dpi=220)
    fig.savefig(out/'250m_三组导航误差曲线.pdf')
    plt.close(fig)
    folder = experiment/'scene/reference_baseline'
    elevation = np.load(folder/'elevation_m.npy', mmap_mode='r')
    valid = np.load(folder/'bathymetry_valid_mask.npy', mmap_mode='r')
    resolution = json.loads((folder/'metadata.json').read_text())['resolution_m']
    row = np.arange(0, elevation.shape[0], 10)
    col = np.arange(0, elevation.shape[1], 8)
    x = (col-(elevation.shape[1]-1)/2)*resolution
    y = (row-(elevation.shape[0]-1)/2)*resolution
    xx, yy = np.meshgrid(x, y)
    z = np.array(elevation[np.ix_(row, col)], float)
    z[~valid[np.ix_(row, col)]] = np.nan
    for variant, r in selected.items():
        positions = arrays[r['folder']][0]
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, z, cmap='terrain', alpha=.62, linewidth=0,
                        rcount=len(row), ccount=len(col), antialiased=False)
        for index, label, colour, style, width in ((0, '虚拟真值', '#202020', '-', 2.2),
                (2, 'Local', '#e86a10', '-', 2.5), (3, '在线Global', '#0072bd', '--', 1.7)):
            xyz = positions[:, index]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], style, color=colour, linewidth=width, label=label)
        ax.set(xlabel='局部沿航向 / m', ylabel='局部横航向 / m', zlabel='Z / m')
        ax.set_title(LABELS[variant]+'：重建局部地形与三条轨迹\n地形布局采用名义直线航速；无GPS，不是地理定位海图', fontsize=13)
        ax.view_init(elev=32, azim=-70)
        ax.set_box_aspect((1., 1.35, .42))
        ax.legend(loc='upper left')
        fig.savefig(out/(LABELS[variant]+'_三维地形轨迹.png'), dpi=220)
        fig.savefig(out/(LABELS[variant]+'_三维地形轨迹.pdf'))
        plt.close(fig)
        records = r['report']['recorded_actual_feature_correspondences']
        for rec in records:
            if rec['matches'] == 0:
                continue
            file = Path(rec['file'])
            if sha(file) != rec['SHA256']:
                raise ValueError('Feature capture changed.')
            with np.load(file) as data:
                previous, current = data['previous_image'], data['current_image']
                pk, ck, pairs = data['previous_keypoints'], data['current_keypoints'], data['pairs_current_previous']
            h = max(previous.shape[0], current.shape[0])
            canvas = np.zeros((2*h+8, max(previous.shape[1], current.shape[1])), np.uint8)
            canvas[:previous.shape[0], :previous.shape[1]] = previous
            canvas[h+8:h+8+current.shape[0], :current.shape[1]] = current
            fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
            ax.imshow(canvas, cmap='gray', vmin=0, vmax=255, interpolation='nearest', aspect='auto')
            for k, (ci, pi) in enumerate(pairs[::max(1, len(pairs)//45)]):
                p0, p1 = pk[pi], ck[ci]
                ax.plot([p0[0], p1[0]], [p0[1], h+8+p1[1]], linewidth=.65, alpha=.7)
            ax.set_title(LABELS[variant]+f'：前端实际KLT对应（{rec["matches"]}对，显示其中至多约45对）\n上：上一帧；下：当前帧。对应关系不等于全部RANSAC内点。')
            ax.set_xlabel('前端规范化图像像素（仅展示，不是原生实测分辨率）')
            fig.savefig(out/(LABELS[variant]+'_实际特征对应.png'), dpi=200)
            plt.close(fig)
            break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--first-case-only', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Preserve previous reviews.')
    state = json.loads((a.experiment/'wide_v3_suite/suite_state.json').read_text())
    if not a.first_case_only and state['status'] != 'completed_all18_paired_GPS_free_PC_experiments':
        raise ValueError('Full suite is not yet completed.')
    tasks = state['tasks'][:3] if a.first_case_only else state['tasks']
    runs, arrays = [], {}
    for task in tasks:
        r, data = verify_run(Path(task['directory']))
        runs.append(r)
        arrays[r['folder']] = data
    for case in ('exact',) if a.first_case_only else ('exact', 'offset20m_yaw5deg'):
        for seed in (31,) if a.first_case_only else (31, 32, 33):
            group = [r for r in runs if r['report']['seed'] == seed
                     and (r['report']['initial_offset_m'] == 0) == (case == 'exact')]
            if len(group) != 3:
                raise ValueError('Incomplete paired input group.')
            for key in ('navigation_measurements', 'sonar_geometry'):
                if len({r['report']['input_hashes'][key] for r in group}) != 1:
                    raise ValueError('Unfair navigation/geometry inputs.')
            if len({r['report']['input_hashes']['sonar_appearance'] for r in group}) != 3:
                raise ValueError('All three processing variants must actually be observed.')
            first = arrays[group[0]['folder']][0]
            for r in group[1:]:
                np.testing.assert_array_equal(first[:, :2], arrays[r['folder']][0][:, :2])
    a.out.mkdir(parents=True)
    summary = dict(status='completed_independent_first_case_review' if a.first_case_only else 'completed_independent_all18_review',
        GPS_used=False, truth='synthetic programmed AUV trajectory, NOT original vessel truth',
        source_scene_layout_assumed=True, independent_cross_survey_validation=False,
        reference_map='same even-record baseline-processed field for all query groups',
        this_reference_choice_is_NOT_proof_of_universal_denoiser_superiority=True,
        records=[{k: v for k, v in r.items() if k != 'report'}|dict(variant=r['report']['variant'],
            seed=r['report']['seed'], initial_offset_m=r['report']['initial_offset_m'],
            initial_yaw_deg=r['report']['initial_yaw_deg']) for r in runs],
        fairness_and_full_stream_metrics_independently_verified=True, analysis_code_SHA256=sha(Path(__file__)))
    (a.out/'独立核验与定位指标.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    plot_first_case(runs, arrays, a.experiment, a.out)
    print(json.dumps(dict(status=summary['status'], completed_runs=len(runs), output=str(a.out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
