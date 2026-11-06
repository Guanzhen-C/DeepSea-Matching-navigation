"""Independently verify paired runs and plot error curves/3D terrain/matches.

Reports errors ONLY against explicitly programmed virtual truth. These plots
must never be presented as original-vessel GPS/ground-truth field accuracy.
"""
from __future__ import annotations

import argparse
import csv
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
    for name, digest in report['source_code_SHA256'].items():
        if sha(folder/'source_snapshot'/name) != digest:
            raise ValueError('Executed source snapshot differs from the captured source hash.')
    if report['source_commit'] != '204d3e24048a8b515616d9305b7516221bcf8fed':
        raise ValueError('Not the unchanged lake estimator source baseline.')
    if report['event_counts'] != dict(sonar=947, imu=15625, depth=3125, dvl=1562):
        raise ValueError('Missing or extra sensor events in the 250m experiment.')
    with np.load(folder/'navigation_states.npz') as data:
        positions = data['positions'].copy()
        rotations = data['rotations'].copy()
        timestamps = data['timestamps'].copy()
    if positions.shape != (947, 4, 3) or rotations.shape != (947, 4, 3, 3):
        raise ValueError('Complete 250m native-ping pose stream required.')
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Finite ordered full pose history required.')
    errors = np.linalg.norm(positions[:, 1:]-positions[:, 0, None], axis=2)
    if sha(folder/'trajectory.csv') != report['trajectory_SHA256']:
        raise ValueError('CSV trajectory changed.')
    data = np.genfromtxt(folder/'trajectory.csv', delimiter=',', names=True)
    csv_positions = np.stack([np.column_stack([data[f'{name}_{axis}'] for axis in ('x', 'y', 'z')])
                              for name in ('truth', 'DR', 'Local', 'Global')], axis=1)
    np.testing.assert_array_equal(csv_positions, positions)
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
    yaw = np.arctan2(rotations[..., 1, 0], rotations[..., 0, 0])
    yaw_error = np.rad2deg((yaw[:, 1:]-yaw[:, :1]+np.pi)%(2*np.pi)-np.pi)
    yaw_metrics = {name: dict(RMSE_deg=float(np.sqrt(np.mean(yaw_error[:, k]**2))),
        P95_abs_deg=float(np.percentile(abs(yaw_error[:, k]), 95)), final_deg=float(yaw_error[-1, k]))
        for k, name in enumerate(('DR', 'Local', 'Global'))}
    accepted = np.flatnonzero(data['terrain_accepts'] > 0)
    first_fix = (dict(timestamp_s=float(timestamps[accepted[0]]),
                      distance_m=float(distance[accepted[0]]),
                      Global_error_m=float(errors[accepted[0], 2])) if len(accepted) else None)
    return dict(folder=str(folder), report=report, metrics=independently_measured, rpe=rpe,
        yaw=yaw_metrics, first_accepted_terrain_factor=first_fix,
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
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        ax.plot_surface(xx, yy, z, cmap='terrain', alpha=.62, linewidth=0,
                        rcount=len(row), ccount=len(col), antialiased=False, zorder=1)
        for index, label, colour, style, width in ((0, '虚拟真值', '#202020', '-', 2.2),
                (2, 'Local', '#e86a10', '-', 2.5), (3, '在线Global', '#0072bd', '--', 1.7)):
            xyz = positions[:, index]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], style, color=colour, linewidth=width, label=label, zorder=10+index)
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
    # Initialisation errors must not change sensor streams either.
    if not a.first_case_only:
        for seed in (31, 32, 33):
            group = [r for r in runs if r['report']['seed'] == seed]
            for key in ('navigation_measurements', 'sonar_geometry'):
                if len({r['report']['input_hashes'][key] for r in group}) != 1:
                    raise ValueError('Initialisation cases did not receive the same sensor streams.')
    a.out.mkdir(parents=True)
    aggregates = []
    for offset in (0.,) if a.first_case_only else (0., 20.):
        for variant in LABELS:
            group = [r for r in runs if r['report']['variant'] == variant
                     and r['report']['initial_offset_m'] == offset]
            record = dict(variant=variant, initial_offset_m=offset, independent_noise_seeds=len(group), metrics={})
            for channel in ('DR', 'Local', 'Global'):
                rmse = np.array([r['metrics'][channel]['RMSE_m'] for r in group])
                record['metrics'][channel] = dict(mean_run_RMSE_m=float(rmse.mean()),
                    sample_std_run_RMSE_m=float(rmse.std(ddof=1)) if len(group) > 1 else None,
                    pooled_RMSE_m=float(np.sqrt(np.mean(rmse**2))),
                    mean_run_P95_m=float(np.mean([r['metrics'][channel]['P95_m'] for r in group])),
                    mean_final_m=float(np.mean([r['metrics'][channel]['final_m'] for r in group])))
            record['runs_with_accepted_terrain_factor'] = sum(r['terrain_accepts'] > 0 for r in group)
            aggregates.append(record)
    summary = dict(status='completed_independent_first_case_review' if a.first_case_only else 'completed_independent_all18_review',
        GPS_used=False, truth='synthetic programmed AUV trajectory, NOT original vessel truth',
        source_scene_layout_assumed=True, independent_cross_survey_validation=False,
        reference_map='same even-record baseline-processed field for all query groups',
        this_reference_choice_is_NOT_proof_of_universal_denoiser_superiority=True,
        aggregate_mean_std_and_pooled_RMSE_are_distinct=True, aggregates=aggregates,
        records=[{k: v for k, v in r.items() if k != 'report'}|dict(variant=r['report']['variant'],
            seed=r['report']['seed'], initial_offset_m=r['report']['initial_offset_m'],
            initial_yaw_deg=r['report']['initial_yaw_deg']) for r in runs],
        fairness_and_full_stream_metrics_independently_verified=True, analysis_code_SHA256=sha(Path(__file__)))
    (a.out/'独立核验与定位指标.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    table_columns = ['variant', 'seed', 'initial_offset_m', 'initial_yaw_deg', 'actual_distance_m',
        'terrain_attempts', 'terrain_accepts', 'accepted_feature_pairs', 'feature_pair_attempts']
    table_columns += [f'{c}_{m}' for c in ('DR', 'Local', 'Global') for m in ('RMSE_m', 'P95_m', 'max_m', 'final_m')]
    with (a.out/'各次实验定位指标.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table_columns)
        writer.writeheader()
        for r in runs:
            record = {n: r['report'][n] for n in ('variant', 'seed', 'initial_offset_m', 'initial_yaw_deg')}
            record.update(actual_distance_m=r['actual_distance_m'], terrain_attempts=r['terrain_attempts'],
                          terrain_accepts=r['terrain_accepts'], accepted_feature_pairs=r['feature']['accepted_pairs'],
                          feature_pair_attempts=r['feature']['pair_attempts'])
            record.update({f'{c}_{m}': r['metrics'][c][m] for c in ('DR', 'Local', 'Global')
                           for m in ('RMSE_m', 'P95_m', 'max_m', 'final_m')})
            writer.writerow(record)
    lines = ['PC局部导航仿真结果与适用边界', '',
        '真值：设定的虚拟AUV运动；没有使用或读取GPS。不是原采集船的真实轨迹定位精度。',
        '场景：实测角度/时延/幅度与冻结增益构景；名义直线航速布局，不是标定海图。',
        '参考/查询使用不同物理记录，但同一测线邻近观测不构成独立航次或独立场景验证。',
        '各组共用偶记录固定去噪参考图，不代表任意去噪都能普遍改善导航。',
        '定位器保持原湖试特征前端、cohort粒子地形匹配、GTSAM固定滞后及map→odom。',
        '严格核对同种子传感器/声呐几何全流一致、三个强度输入均不同，未放宽导航门限。',
        '', '以下为各次全段RMSE的均值（不是将所有误差直接算术平均）：']
    for r in aggregates:
        values = r['metrics']
        lines.append(f"初值偏差{r['initial_offset_m']:g}m · {LABELS[r['variant']]} · n={r['independent_noise_seeds']}："
            f"DR/Local/Global平均RMSE {values['DR']['mean_run_RMSE_m']:.3f}/"
            f"{values['Local']['mean_run_RMSE_m']:.3f}/{values['Global']['mean_run_RMSE_m']:.3f}m；"
            f"产生地形确认的实验{r['runs_with_accepted_terrain_factor']}次。")
    lines += ['', '完整样本标准差、合并RMSE、各次P95/末值、10m/30m SE3相对误差、航向误差，',
        '以及首个接受地形因子的时刻/里程保存在JSON；逐次简表见CSV。',
        '初值偏差对照包含20m位置和5度航向误差，不能把初始误差主导的RMSE等同于稳态精度。',
        'Local如果不如DR应保留负面结论；边缘视觉改善也不能自动当作导航改善。',
        '三维图的地形LOD和图像规范化只用于显示，不改变数值输入、轨迹或场景数组。',
        '首次120度扇面和宽扇面NoData失败结果另存，未混入本正式宽扇面汇总。',
        '本次去噪算法未继续调参，也未改动既有72km海试或湖试稳定基线。']
    (a.out/'实验结果与适用边界.txt').write_text('\n'.join(lines)+'\n')
    plot_first_case(runs, arrays, a.experiment, a.out)
    print(json.dumps(dict(status=summary['status'], completed_runs=len(runs), output=str(a.out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
