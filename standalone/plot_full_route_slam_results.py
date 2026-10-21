#!/usr/bin/env python3
"""Plot completed native SLAM runs using every synchronized error sample.

Distance comes from the recorded trajectory, not speed multiplied by time.
Validate the online errors against saved poses before drawing. Posthoc graph
curves are explicitly diagnostic and are never substituted for online output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import numpy as np


def load_verified_run(directory):
    metrics_path = directory / 'metrics.json'
    error_path = directory / 'metrics_position_errors.csv'
    pose_path = directory / 'terrain_replay/online_poses_5hz.csv'
    metrics = json.loads(metrics_path.read_text())
    if metrics['run_status'] != 'completed' or not metrics['completed_requested_route']:
        raise ValueError('a completed full-route run is required')
    errors = np.loadtxt(error_path, delimiter=',', skiprows=1)
    poses = np.loadtxt(pose_path, delimiter=',', skiprows=1)
    if errors.shape[1] != 6 or poses.shape != (len(errors), 14):
        raise ValueError('incomplete or unexpected error/pose columns')
    if not (np.isfinite(errors).all() and np.isfinite(poses).all()):
        raise ValueError('nonfinite trajectory/error samples')
    np.testing.assert_allclose(errors[:, 0], poses[:, 0], atol=1.e-8, rtol=0.)
    np.testing.assert_allclose(np.diff(poses[:, 0]), 1./metrics['ping_rate_hz'], atol=1.e-7, rtol=0.)
    if np.any(np.diff(poses[:, 1]) < -1.e-8):
        raise ValueError('travelled distance must be monotonic')
    for column, pose_start, name in ((1, 5, 'dead_reckoning_imu_dvl_depth'),
                                   (2, 8, 'local_acoustic_esikf'), (3, 11, 'online_global_slam')):
        actual = np.linalg.norm(poses[:, pose_start:pose_start+3]-poses[:, 2:5], axis=1)
        np.testing.assert_allclose(actual, errors[:, column], atol=1.e-8, rtol=0.)
        stats = metrics['position_error_statistics'][name]
        computed = dict(rmse_3d_m=np.sqrt(np.mean(actual**2)), p95_3d_m=np.percentile(actual, 95),
                        max_3d_m=actual.max(), final_3d_m=actual[-1])
        for key, value in computed.items():
            np.testing.assert_allclose(value, stats[key], atol=1.e-8, rtol=0.)
        if stats['samples'] != len(errors):
            raise ValueError('reported sample count differs from recorded poses')
    return metrics, errors, poses


def plot_run(directory, output_prefix, title):
    metrics, errors, poses = load_verified_run(directory)
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(), 'font.size':12,
                         'axes.unicode_minus':False, 'path.simplify':False,
                         'pdf.fonttype':3, 'axes.edgecolor':'#45515e'})
    distance = poses[:, 1] / 1000.
    stat = metrics['position_error_statistics']
    online = stat['online_global_slam']
    fig, (overview, detail) = plt.subplots(2, 1, figsize=(18, 10.5), sharex=True,
        gridspec_kw={'height_ratios':[1., 1.15]})
    fig.subplots_adjust(left=.065, right=.98, top=.90, bottom=.09, hspace=.20)
    fig.suptitle(f'{title}：{distance[-1]:.2f} km 全航程定位误差曲线'
                 f'（{metrics["ping_rate_hz"]:g} Hz，{len(errors):,} 个同步样本）',
                 fontsize=21, fontweight='bold', y=.975)
    colors = dict(dr='#748292', local='#f07425', online='#007fba', optimized='#1a9d83', history='#bd6ba1')
    overview.plot(distance, errors[:, 1], color=colors['dr'], lw=1.0, label='纯航位推算 DR', rasterized=True)
    overview.plot(distance, errors[:, 2], color=colors['local'], lw=1.0, label='局部声呐里程计 Local', rasterized=True)
    overview.plot(distance, errors[:, 3], color=colors['online'], lw=1.0, label='在线 Global（实际发布）', zorder=4, rasterized=True)
    overview.set_title('(a) 全尺度三维位置误差', fontsize=14, pad=10)
    overview.set_ylim(0, max(errors[:, 1].max(), errors[:, 2].max()) * 1.12)
    overview.legend(loc='upper left', ncol=3, fontsize=11, framealpha=.96)
    overview.text(.99, .97,
        f'DR RMSE {stat["dead_reckoning_imu_dvl_depth"]["rmse_3d_m"]:.2f} m    '
        f'Local RMSE {stat["local_acoustic_esikf"]["rmse_3d_m"]:.2f} m\n'
        f'在线 Global：中位数 {online["median_3d_m"]:.2f} m    P95 {online["p95_3d_m"]:.2f} m    '
        f'RMSE {online["rmse_3d_m"]:.2f} m    最终 {online["final_3d_m"]:.2f} m',
        ha='right', va='top', transform=overview.transAxes, fontsize=11,
        bbox=dict(boxstyle='round,pad=.45', facecolor='white', edgecolor='#a6b4c4', alpha=.97))
    detail.plot(distance, errors[:, 4], color=colors['optimized'], lw=.65, alpha=.65,
                label='最终固定滞后优化（事后）', zorder=2, rasterized=True)
    detail.plot(distance, errors[:, 5], color=colors['history'], lw=.65, alpha=.60,
                label='全历史 iSAM2（诊断）', zorder=2, rasterized=True)
    detail.plot(distance, errors[:, 3], color=colors['online'], lw=.9,
                label='在线 Global（实际发布）', zorder=4, rasterized=True)
    detail.axhline(online['p95_3d_m'], color='#58aee4', ls='--', lw=1.1,
                  label=f'在线 Global P95 = {online["p95_3d_m"]:.2f} m', zorder=3)
    detail.set_yscale('symlog', linthresh=1.)
    detail.set_ylim(0, max(100., 10.**np.ceil(np.log10(errors[:, 3:].max()))))
    detail.set_yticks([0, .5, 1, 2, 5, 10, 20, 50, 100])
    detail.yaxis.set_major_formatter(ScalarFormatter())
    detail.set_title('(b) Global 误差细节（0–1 m 线性，1 m 以上对数）', fontsize=14, pad=10)
    detail.set_xlabel('累计航程 / km', fontsize=14)
    handles, labels = detail.get_legend_handles_labels()
    order = [2, 0, 1, 3]
    detail.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc='upper left', ncol=2, fontsize=10.5, framealpha=.96)
    peak_index = int(np.argmax(errors[:, 3]))
    peak_x, peak_y = float(distance[peak_index]), float(errors[peak_index, 3])
    detail.scatter([peak_x], [peak_y], s=35, color='#c52d43', zorder=6)
    detail.annotate(f'全程最大在线误差\n{peak_y:.2f} m @ {peak_x:.3f} km\n终点：{online["final_3d_m"]:.2f} m',
        xy=(peak_x, peak_y), xycoords='data', xytext=(.72, .80), textcoords='axes fraction',
        color='#b4233a', fontsize=11, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=.4', fc='white', ec='#c52d43', alpha=.98),
        arrowprops=dict(arrowstyle='->', color='#c52d43', lw=1.3), zorder=7)
    for axis in (overview, detail):
        axis.set_ylabel('三维位置误差 / m', fontsize=13)
        axis.set_xlim(0, distance[-1])
        axis.xaxis.set_major_locator(MultipleLocator(5))
        axis.xaxis.set_minor_locator(MultipleLocator(1))
        axis.grid(which='major', color='#d5dfe8', lw=.8)
        axis.grid(which='minor', axis='x', color='#edf1f5', lw=.45)
        axis.set_axisbelow(True)
    fig.text(.065, .025, 'Chelan 真实湖底地形上的传感器仿真，非实船湖试；误差未作轨迹对齐。'
             '绿色、紫色为事后/诊断结果，不代表在线发布精度。', fontsize=10, color='#596575')
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for extension in ('.png', '.pdf'):
        target = output_prefix.with_suffix(extension)
        fig.savefig(target, dpi=180 if extension == '.png' else 150, facecolor='white')
        files.append(str(target))
    plt.close(fig)
    summary = dict(run_directory=str(directory.resolve()), title=title,
        source_commit=json.loads((directory/'run_manifest.json').read_text())['source_commit'],
        completed_requested_route=True, distance_m=float(poses[-1, 1]), samples=len(errors),
        gap_count=0, all_online_errors_recomputed_from_saved_poses=True,
        plotted_every_sample=True, distance_source='recorded travelled_distance_m',
        online_global=online, peak_distance_m=float(poses[peak_index, 1]),
        posthoc_curves_are_not_online_results=True, files=files,
        input_sha256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                      for name in ('metrics.json', 'metrics_position_errors.csv',
                                   'terrain_replay/online_poses_5hz.csv')})
    output_prefix.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--title', default='湖底仿真基线 · Chelan')
    parser.add_argument('--desktop', type=Path)
    args = parser.parse_args()
    summary = plot_run(args.run_directory, args.output_prefix, args.title)
    if args.desktop:
        args.desktop.parent.mkdir(parents=True, exist_ok=True)
        if args.desktop.exists():
            raise FileExistsError('desktop target already exists; choose a new name')
        shutil.copy2(args.output_prefix.with_suffix('.png'), args.desktop)
        summary['desktop_png'] = str(args.desktop)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
