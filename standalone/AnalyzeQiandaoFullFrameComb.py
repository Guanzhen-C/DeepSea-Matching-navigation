"""Independently verify the finished full-loop joint experiment, then plot.

Does not accept a running/short pilot result or turn an unsuccessful accuracy
result into a positive field-navigation claim. No estimation or retraining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from RunQiandaoNoGPSNavigation import sha
from QiandaoCombTrajectory import CombSurveyTrajectory, CombSurveyParameters
from AnalyzeQiandaoNoGPSNavigation import relative_errors


def metrics(error):
    return dict(samples=len(error), RMSE_m=float(np.sqrt(np.mean(error**2))),
        median_m=float(np.median(error)), P95_m=float(np.percentile(error, 95)),
        max_m=float(np.max(error)), final_m=float(error[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    a = parser.parse_args()
    if a.out.exists():
        raise FileExistsError('Preserve earlier independent reviews.')
    report = json.loads((a.run/'run_summary.json').read_text())
    if (report['status'] != 'completed_ALL_scheduled_events_full_frozen_joint_estimator'
            or not report['full_closed_loop_requested'] or report['source_physical_records'] != 2020
            or report['GPS_used'] or report['after_initialization_truth_accessible_to_estimator']
            or report['reference_query_depth_shared'] or not report['both_depth_and_side_scan_global_matching_enabled']):
        raise ValueError('Only completed all2020/full-loop/GPS-free/frozen-joint runs eligible.')
    if report['original_estimator_source_commit'] != '204d3e24048a8b515616d9305b7516221bcf8fed':
        raise ValueError('Different estimator, do not silently combine with this experiment.')
    if report['expected_events'] != report['event_counts']:
        raise ValueError('Incomplete multi-sensor event stream.')
    for name, expected in report['source_code_SHA256'].items():
        if sha(a.run/'source_snapshot'/name) != expected:
            raise ValueError('Executed source snapshot hash mismatch.')
    for file, key in [('trajectory.csv', 'trajectory_SHA256'), ('navigation_states.npz', 'navigation_states_SHA256')]:
        if sha(a.run/file) != report[key]:
            raise ValueError('Captured full trajectory hash mismatch.')
    scene = json.loads((a.experiment/'scene/scene_report.json').read_text())
    if sha(a.experiment/'scene/scene_report.json') != report['scene_report_SHA256']:
        raise ValueError('Scene report changed after the full run.')
    reference_ids = np.load(a.experiment/'scene/reference_record_ids.npy')
    query_ids = np.load(a.experiment/'scene/query_record_ids.npy')
    np.testing.assert_array_equal(np.sort(np.r_[reference_ids, query_ids]), np.arange(2020))
    assert not len(np.intersect1d(reference_ids, query_ids))
    with np.load(a.run/'navigation_states.npz') as data:
        positions, rotations, stamps = data['positions'], data['rotations'], data['timestamps']
    csv = np.genfromtxt(a.run/'trajectory.csv', delimiter=',', names=True)
    csv_positions = np.stack([np.column_stack([csv[f'{name}_{axis}'] for axis in ('x', 'y', 'z')])
                              for name in ('truth', 'DR', 'Local', 'Global')], axis=1)
    np.testing.assert_array_equal(csv_positions, positions)
    np.testing.assert_array_equal(csv['time_s'], stamps)
    if positions.shape != (report['expected_events']['sonar'], 4, 3) or not np.isfinite(positions).all():
        raise ValueError('Wrong/incomplete/invalid full sonar pose history.')
    if rotations.shape != (len(stamps), 4, 3, 3) or not np.isfinite(rotations).all() or np.any(np.diff(stamps) <= 0):
        raise ValueError('Ordered complete finite rotation history required.')
    parameters = json.loads((a.experiment/'AUV全幅梳形轨迹.json').read_text())['parameters']
    route = CombSurveyTrajectory(CombSurveyParameters(**parameters))
    expected_xy, tangent, curvature, segment = route.evaluate_distance(csv['distance_m'])
    np.testing.assert_allclose(positions[:, 0, :2], expected_xy, atol=1e-10)
    np.testing.assert_allclose(positions[:, 0, 2], route.parameters.vehicle_z_m, atol=1e-12)
    np.testing.assert_array_equal(csv['segment_index'].astype(int), segment)
    if (abs(report['requested_distance_m']-route.length_m) > 1e-8
            or route.length_m-csv['distance_m'][-1] > route.speed_mps/report['virtual_rate_Hz']['sonar']+1e-8):
        raise ValueError('Not the full planned loop.')
    error = np.linalg.norm(positions[:, 1:]-positions[:, 0, None], axis=2)
    totals = {name: metrics(error[:, k]) for k, name in enumerate(('DR', 'Local', 'Global'))}
    for name in totals:
        for key in ('RMSE_m', 'median_m', 'P95_m', 'max_m', 'final_m'):
            if not np.isclose(totals[name][key], report['metrics'][name][key], rtol=1e-12, atol=1e-12):
                raise ValueError('Independent full error calculation differs from recorded result.')
    diagnostic = json.loads((a.run/'estimator_diagnostics.json').read_text())
    if diagnostic['graph_diagnostics']['optimizer_backend'] != 'gtsam_isam2_incremental_fixed_lag':
        raise ValueError('Not actually GTSAM incremental fixed lag.')
    if diagnostic['feature_diagnostics']['active_compute_backend'] != 'cuda':
        raise ValueError('Feature consensus did not actually use the requested CUDA backend.')
    yaw = np.arctan2(rotations[..., 1, 0], rotations[..., 0, 0])
    dyaw = np.rad2deg((yaw[:, 1:]-yaw[:, :1]+np.pi)%(2*np.pi)-np.pi)
    phase = np.array([route.segments[int(k)].kind for k in segment])
    phase_metrics = {kind: {name: metrics(error[phase == kind, k])
                     for k, name in enumerate(('DR', 'Local', 'Global'))}
                     for kind in np.unique(phase)}
    trace = json.loads((a.run/'frozen_backend_trace.json').read_text())
    result = dict(status='completed_independent_ALL2020_full_comb_review', metrics=totals,
        phase_metrics=phase_metrics, GPS_used=False, source_records=2020,
        complete_planned_distance_m=route.length_m, actual_last_logged_distance_m=float(csv['distance_m'][-1]),
        geometry_and_intensity_from_disjoint_records=True, all2020_union_verified=True,
        truth='programmed virtual AUV, NOT real vessel field truth',
        yaw={name: dict(RMSE_deg=float(np.sqrt(np.mean(dyaw[:, k]**2))),
            P95_abs_deg=float(np.percentile(abs(dyaw[:, k]), 95)), final_deg=float(dyaw[-1, k]))
            for k, name in enumerate(('DR', 'Local', 'Global'))},
        rpe={str(length): relative_errors(positions, rotations, csv['distance_m'], length) for length in (10., 30.)},
        actual_backend=diagnostic['graph_diagnostics']['optimizer_backend'],
        actual_feature_backend=diagnostic['feature_diagnostics']['active_compute_backend'],
        terrain_attempts=diagnostic['terrain_match_attempts'], terrain_accepts=diagnostic['terrain_prior_accepts'],
        backscatter_match_stats=trace.get('backscatter_match_stats', {}),
        source_report_SHA256=sha(a.run/'run_summary.json'), analysis_source_SHA256=sha(Path(__file__)),
        not_independent_field_precision_or_generalization_validation=True,
        source_attitude_sound_speed_and_radiometric_calibration_limitations_remain=True)
    a.out.mkdir()
    (a.out/'全航程独立核验与指标.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    for font in font_manager.fontManager.ttflist:
        if 'Noto Sans CJK' in font.name:
            plt.rcParams['font.family'] = font.name
            break
    plt.rcParams['axes.unicode_minus'] = False
    distance = csv['distance_m']/1000
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
    for k, (name, colour) in enumerate([('DR', '#778899'), ('Local', '#ed741e'), ('Global', '#0072bd')]):
        axes[0].plot(distance, error[:, k], color=colour, linewidth=1, label=f'{name} RMSE {totals[name]["RMSE_m"]:.3f}m')
    axes[1].plot(distance, error[:, 2], color='#0072bd', linewidth=1., label='在线Global')
    axes[1].axhline(totals['Global']['P95_m'], color='#444444', linestyle='--', linewidth=.8, label='全程Global P95')
    starts = np.r_[0, route.ends_m[:-1]]
    for start, end, part in zip(starts, route.ends_m, route.segments):
        if 'turn' in part.kind:
            for ax in axes:
                ax.axvspan(start/1000, end/1000, color='#f1b570', alpha=.17)
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(); ax.set_ylabel('三维位置误差 / m')
    axes[1].set_xlabel('累计虚拟航程 / km（浅色背景为转弯）')
    axes[0].set_title('整幅2020ping测深＋去噪侧扫构景：原联合算法完整梳形实验\n无GPS；误差针对虚拟AUV，不是原采集船实测定位精度')
    fig.savefig(a.out/'全16km梳形_DR_Local_Global误差曲线.png', dpi=180)
    fig.savefig(a.out/'全16km梳形_DR_Local_Global误差曲线.pdf')
    plt.close(fig)
    field = a.experiment/'scene/full_baseline_display_only'
    z = np.load(field/'elevation_m.npy', mmap_mode='r')
    mask = np.load(field/'bathymetry_valid_mask.npy', mmap_mode='r')
    row, col = np.arange(0, z.shape[0], 10), np.arange(0, z.shape[1], 14)
    xx, yy = np.meshgrid((col-(z.shape[1]-1)/2)*.3, (row-(z.shape[0]-1)/2)*.3)
    zz = z[np.ix_(row, col)].copy(); zz[~mask[np.ix_(row, col)]] = np.nan
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.plot_surface(xx, yy, zz, cmap='terrain', alpha=.55, linewidth=0,
                    rcount=len(row), ccount=len(col), zorder=1)
    for index, label, colour, style, width in [(0, '虚拟真值', '#171717', '-', 1.8),
            (2, 'Local', '#f07818', '-', 1.3), (3, '在线Global', '#0072bd', '--', 1.2)]:
        xyz = positions[:, index]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], style, color=colour,
                linewidth=width, zorder=10+index, label=label)
    ax.set(xlabel='名义局部X / m', ylabel='名义局部Y / m', zlabel='Z / m',
           title='全部实测测深构景上的三条实际仿真输出轨迹\n无GPS；源轨迹／姿态未标定，地形不是已知高精度海图')
    ax.view_init(elev=45, azim=-65); ax.legend()
    fig.savefig(a.out/'全16km梳形_实测构景与三条输出轨迹.png', dpi=170)
    fig.savefig(a.out/'全16km梳形_实测构景与三条输出轨迹.pdf')
    plt.close(fig)
    lines = ['全部2020ping联合构景：原算法完整闭合梳形结果', '',
        '完整循环已执行，CSV/NPZ/源快照/三维误差/全部事件计数独立核验。',
        '误差以虚拟AUV真值为参考，不是实船定位精度。没有使用GPS。',
        '用实测测深和冻结去噪原生幅度构景；源名义运动和姿态并未标定。',
        '偶/奇源记录分离，但仍同航次，不是独立泛化验证。',
        '保留转弯、缺测、拒绝和失锁数据；高误差属于结果，不因跑完而当通过。']
    for name in totals:
        q = totals[name]
        lines.append(f'{name}：RMSE {q["RMSE_m"]:.3f}m，P95 {q["P95_m"]:.3f}m，最大{q["max_m"]:.3f}m，末值{q["final_m"]:.3f}m。')
    (a.out/'结果与科学边界.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'], metrics=totals, output=str(a.out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
