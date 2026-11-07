"""Full-route enhanced-map comparison, evaluation only and local outputs only.

Refuses running/prefix results. Never changes the estimator, uses truth to
select matches, or uploads raw data. The original complete control is read
unchanged. New/old sonar observations are NOT falsely called identical inputs.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from AnalyzePureSideScanShort import planar_relative_metrics
from AnalyzeSideScanLoopReplay import read_replay, sha
from BuildSeabedSimulationReportEvidence import summarize, segmented, write_csv


ROOT = Path(__file__).resolve().parents[1]


def require_full_execution(report, review):
    if (report.get('status') != 'completed_pure_side_scan_FEATURE_route_test'
            or not report.get('full_closed_loop_requested')
            or not report.get('full_source_snapshots_verified')
            or review.get('status') != 'independently_verified_complete_executed_scope'
            or not review.get('complete_planned_route_verified')
            or not review.get('recorded_navigation_inputs_verified')
            or not review.get('paired_legacy_DR_bit_exact')
            or report.get('event_counts') != report.get('expected_events')
            or report.get('processed_pings') != report.get('expected_events', {}).get('sonar')
            or report.get('submap_representation') != 'adaptive_projected'
            or report.get('observation_source') != 'complex_IQ'
            or report.get('requested_feature_matcher') != 'LoFTR'
            or not report.get('historical_side_scan_loops_requested')
            or report.get('secondary_denoising_used') is not False
            or report.get('MBES_XYZ_given_to_estimator') is not False
            or report.get('measured_height_likelihood_used') is not False
            or report.get('GPS_used') is not False):
        raise ValueError('Need the complete audited enhanced-map route, all inputs and the same validated pure-SSS profile')
    if not np.isclose(report['requested_distance_m'], report['complete_planned_loop_length_m'], rtol=0, atol=1e-8):
        raise ValueError('A prefix cannot substitute for the complete route')


def load_json(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def compare(experiment, run, review, original_source, baseline_replay, out):
    experiment, run, review, original_source, baseline_replay, out = map(
        Path, (experiment, run, review, original_source, baseline_replay, out))
    report = load_json(run / 'run_summary.json')
    audit = load_json(review / 'independent_review.json')
    require_full_execution(report, audit)
    if sha(run / 'run_summary.json') != audit['source_run_SHA256']:
        raise ValueError('Review no longer matches executed new result')
    if out.exists():
        raise FileExistsError('Keep all previous reports and comparisons')
    with np.load(run / 'navigation_states.npz') as archive:
        new = {k: archive[k] for k in archive.files}
    if sha(run / 'navigation_states.npz') != report['navigation_states_SHA256']:
        raise ValueError('Executed online states changed')
    n = len(new['timestamps'])
    if n != report['processed_pings'] or new['positions'].shape != (n, 4, 3):
        raise ValueError('Full saved online state coverage differs')
    # Independent full CSV readback; never use terminal progress samples.
    with (run / 'trajectory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    actual = np.array([[float(row[f'{name}_{c}']) for name in ('truth', 'DR', 'Local', 'Global')
                       for c in 'xyz'] for row in rows]).reshape(-1, 4, 3)
    np.testing.assert_array_equal(actual, new['positions'])
    np.testing.assert_array_equal([float(row['time_s']) for row in rows], new['timestamps'])
    with np.load(original_source / 'navigation_states.npz') as archive:
        old_truth = {k: archive[k] for k in archive.files}
    np.testing.assert_array_equal(new['timestamps'], old_truth['timestamps'])
    np.testing.assert_array_equal(new['distance_m'], old_truth['distance_m'])
    np.testing.assert_array_equal(new['positions'][:, :2], old_truth['positions'][:, :2])
    baseline_report, baseline_audit, baseline_state, _ = read_replay(baseline_replay, old_truth)
    old = np.concatenate([old_truth['positions'][:, :1], baseline_state['positions']], axis=1)
    expected_scene = load_json(experiment / 'scene/scene_report.json')
    if (expected_scene.get('appearance_variant') != 'edge_shotwell_HSV_V_appearance'
            or sha(experiment / 'scene/scene_report.json') != report['simulator_scene_report_SHA256']):
        raise ValueError('Actual new run did not use the approved enhanced appearance scene')
    for name, expected in report['prior_intensity_inputs_SHA256'].items():
        if sha(experiment / 'scene/reference_baseline' / name) != expected:
            raise ValueError('Executed enhanced reference map changed')
    for name, expected in report['original_core_source_SHA256'].items():
        if sha(experiment / 'estimator_bundle/isaacsim/oceansim/slam' / name) != expected:
            raise ValueError('Estimator core changed')
    errors = {'DR': np.linalg.norm(new['positions'][:, 1]-new['positions'][:, 0], axis=1)}
    for prefix, positions in [('old', old), ('new', new['positions'])]:
        for index, name in ((2, 'Local'), (3, 'Global')):
            errors[prefix+'_'+name] = np.linalg.norm(positions[:, index]-positions[:, 0], axis=1)
    metrics = {name: summarize(value) for name, value in errors.items()}
    for name in ('Local', 'Global'):
        for lower, upper in [('rmse_m', 'RMSE_m'), ('p95_m', 'P95_m'), ('max_m', 'max_m'), ('final_m', 'final_m')]:
            np.testing.assert_allclose(metrics['new_'+name][lower], report['metrics'][name][upper], rtol=0, atol=1e-10)
    out.mkdir(parents=True)
    for directory in ('data', 'figures', 'evidence'):
        (out / directory).mkdir()
    write_csv(out / 'data/full_errors_new_vs_old.csv',
              ['timestamp_s', 'distance_m', *[name+'_error_m' for name in errors]],
              np.column_stack([new['timestamps'], new['distance_m'], *errors.values()]))
    names = ['truth', 'DR', 'old_Local', 'old_Global', 'new_Local', 'new_Global']
    coordinates = [new['positions'][:, 0], new['positions'][:, 1], old[:, 2], old[:, 3],
                   new['positions'][:, 2], new['positions'][:, 3]]
    write_csv(out / 'data/full_trajectories_new_vs_old.csv.gz',
              ['timestamp_s', 'distance_m', *[name+'_'+c+'_m' for name in names for c in 'xyz']],
              np.column_stack([new['timestamps'], new['distance_m'], *coordinates]))
    write_csv(out / 'data/segments_100m_new_vs_old.csv',
              ['start_m', 'end_m', 'samples', 'channel', 'rmse_m', 'median_m', 'p95_m', 'max_m', 'final_m'],
              segmented(new['distance_m'], errors))
    worst = int(np.argmax(errors['new_Global']))
    result = dict(status='completed_full_new_map_vs_unchanged_old_map_comparison',
                  samples=n, distance_m=float(new['distance_m'][-1]), metrics=metrics,
                  same_truth_timestamps_and_DR_bit_exact=True,
                  same_sonar_observations=False,
                  enhanced_field_is_uncalibrated_display_appearance_not_physical_echo_amplitude=True,
                  new_Global_reduction_vs_DR=1-metrics['new_Global']['rmse_m']/metrics['DR']['rmse_m'],
                  new_Global_reduction_vs_old_Global=1-metrics['new_Global']['rmse_m']/metrics['old_Global']['rmse_m'],
                  new_worst_Global=dict(index=worst, distance_m=float(new['distance_m'][worst]),
                                        error_m=float(errors['new_Global'][worst])),
                  new_planar_relative=planar_relative_metrics(new['positions'], new['yaw_rad'], new['distance_m']),
                  new_graph=dict(map_priors=audit['side_scan_absolute_feature_priors'],
                                 loop_factors=audit['actual_historical_loop_factors']),
                  baseline_graph=dict(loop_factors=baseline_audit['actual_loop_edges']),
                  baseline_sources=dict(
                      original_states_SHA256=sha(original_source / 'navigation_states.npz'),
                      replay_outputs_SHA256=baseline_audit['output_SHA256']),
                  new_sources=dict(executed_run_SHA256=sha(run / 'run_summary.json'),
                                   executed_states_SHA256=sha(run / 'navigation_states.npz'),
                                   review_SHA256=sha(review / 'independent_review.json'),
                                   scene_report_SHA256=sha(experiment / 'scene/scene_report.json')),
                  source_run=str(run.resolve()), source_review=str(review.resolve()),
                  original_source=str(original_source.resolve()), original_replay=str(baseline_replay.resolve()),
                  source_experiment=str(experiment.resolve()),
                  external_upload_performed=False)
    save(out / 'data/verified_new_map_metrics.json', result)
    shutil.copy2(review / 'independent_review.json', out / 'evidence/new_full_route_review.json')
    shutil.copy2(experiment / 'scene/scene_report.json', out / 'evidence/new_scene_report.json')
    plot_results(out, new, errors, experiment)
    print(json.dumps({k: result[k] for k in ('status', 'samples', 'metrics')}, ensure_ascii=False), flush=True)
    return result


def plot_results(out, state, errors, experiment):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.colors import LinearSegmentedColormap
    from PlotPureSideScanTerrain import terrain_mesh, select_three_trajectories
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams['path.simplify'] = False
    distance = state['distance_m']/1000
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    styles = [('DR', 'DR', '#9099a3'), ('old_Local', '旧底图 Local', '#e3ad79'),
              ('new_Local', '新底图 Local', '#ce7115'), ('old_Global', '旧底图 Global', '#7654a3'),
              ('new_Global', '新底图 Global', '#009789')]
    for name, label, color in styles:
        axes[0].plot(distance, errors[name], label=label, color=color, linewidth=.65)
    for name, label, color in styles[-2:]:
        axes[1].plot(distance, errors[name], label=label, color=color, linewidth=.7)
    for ax in axes:
        ax.set_ylabel('三维位置误差 / m'); ax.grid(alpha=.2); ax.legend(ncol=3)
    axes[1].set_xlabel('累计航程 / km')
    axes[0].set_title('新底图完整梳形航线：实际在线误差与旧底图对照')
    fig.savefig(out / 'figures/new_vs_old_full_errors.png', dpi=190)
    plt.close(fig)
    field = experiment / 'scene/query_baseline'
    meta = load_json(field / 'metadata.json')
    elevation = np.load(field / 'elevation_m.npy', mmap_mode='r')
    valid = np.load(field / 'bathymetry_valid_mask.npy', mmap_mode='r')
    x, y, z = terrain_mesh(elevation, valid, meta['resolution_m'], 12)
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    cmap = LinearSegmentedColormap.from_list('pale_relief', ['#ebeff2', '#c7d3dc', '#a5b9c7'])
    ax.plot_surface(x, y, z, cmap=cmap, alpha=.6, linewidth=0, rcount=len(x), ccount=x.shape[1],
                    antialiased=False, rasterized=True, zorder=1)
    lines = select_three_trajectories(state['positions'])
    for key, label, color, style, width in [
            ('Truth', '真值（仅评价）', '#222222', '--', .8),
            ('Local', 'Local', '#ef8b23', '-', .85), ('Global', '在线 Global', '#006de5', '-', 1.05)]:
        q = lines[key]
        ax.plot(q[:, 0], q[:, 1], q[:, 2], label=label, color=color, linestyle=style,
                linewidth=width, zorder=10)
    all_z = np.r_[z[np.isfinite(z)], state['positions'][:, :, 2].ravel()]
    ax.set_box_aspect((float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(all_z))*3), zoom=1.1)
    ax.view_init(elev=48, azim=-67)
    ax.set(xlabel='名义 X / m', ylabel='名义 Y / m', zlabel='Z / m',
           title='新底图完整航程：真实轨迹、Local 与在线 Global\n地形只用于仿真和展示；Z显示比例3倍，轨迹不偏移不贴地')
    ax.legend(loc='upper left')
    fig.savefig(out / 'figures/new_map_3d_trajectories.png', dpi=210)
    fig.savefig(out / 'figures/new_map_3d_trajectories.pdf', dpi=210)
    plt.close(fig)
    full = experiment / 'scene/full_baseline_display_only'
    appearance = np.load(full / 'backscatter.npy', mmap_mode='r')
    mask = np.load(full / 'backscatter_valid_mask.npy', mmap_mode='r')
    h, w = appearance.shape
    r = meta['resolution_m']
    shown = np.ma.masked_where(~mask[::2, ::2], appearance[::2, ::2])
    fig, ax = plt.subplots(figsize=(14, 5.2), constrained_layout=True)
    cmap = plt.get_cmap('gray').copy(); cmap.set_bad('#dce2e5')
    ax.imshow(shown, origin='lower', extent=[-(w-1)*r/2, (w-1)*r/2, -(h-1)*r/2, (h-1)*r/2],
              cmap=cmap, vmin=0, vmax=1, interpolation='nearest', aspect='equal')
    q = state['positions'][:, 0]
    ax.plot(q[:, 0], q[:, 1], color='#00a5ee', linewidth=.7, label='实际执行的真值轨迹（仅评价）')
    ax.set(xlabel='名义 X / m', ylabel='名义 Y / m',
           title='新底图数值外观场及完整梳形航线\n背景为增强HSV-V外观强度，非标定背散射系数；灰色为无数据')
    ax.legend(loc='upper right', fontsize=9)
    fig.savefig(out / 'figures/new_field_comb_trajectory.png', dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('experiment', 'run', 'review', 'original-source', 'baseline-replay', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    compare(a.experiment, a.run, a.review, a.original_source, a.baseline_replay, a.out)
