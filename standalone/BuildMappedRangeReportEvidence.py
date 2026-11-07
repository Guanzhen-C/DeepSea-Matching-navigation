"""Local, full-sample report evidence for the completed mapped-range trials.

Recomputes online errors and provenance. Never runs navigation, changes a
parameter, copies raw sonar/cache payloads, or uploads any artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from AnalyzeMappedRangeNavigation import (
    direction_metrics, require_execution_scope, verify_frozen_dependencies, verify_preserve_prefix,
)
from AnalyzeSideScanLoopReplay import read_replay, sha
from AuditMappedRangeInputs import audit_mapped_range_inputs
from BuildSeabedSimulationReportEvidence import segmented, summarize, write_csv

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'output/real_sonar/qiandao_range_change_navigation_20261107_v1'
PREVIOUS = ROOT/'output/reports/seabed_matching_enhanced_scene_20261107_v1'
EXPERIMENT = ROOT/'output/real_sonar/qiandao_edge_shotwell_navigation_20261107_v1/enhanced_scene'


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def require_report_scope(d, *, full):
    status = ('completed_independent_full_identical_input_mapped_range_comparison' if full else
              'completed_independent_heldout_recording_mapped_range_comparison_NOT_full16km')
    if (d.get('status') != status or d.get('samples') != (62458 if full else 15152)
            or d.get('source_seed') != (31 if full else 33)
            or d.get('mapped_range_precision_policy') != 'preserve_legacy'
            or not all(d.get(k) for k in ['all_received_events_verified', 'DR_and_Local_bit_exact',
                                         'full_original_input_file_SHA256_reverified'])
            or d.get('executed_dependency_audit', {}).get('core_files') != 4
            or d.get('executed_dependency_audit', {}).get('prior_files') != 3
            or d.get('preserved_prefix_audit', {}).get('maximum_coordinate_difference_m') != 0.):
        raise ValueError('Complete independent full/fresh preserve-legacy evidence required; no prefix substitution')


def read_trial(comparison, *, full):
    comparison = Path(comparison); d = load(comparison)
    require_report_scope(d, full=full)
    source, candidate = Path(d['source_run']), Path(d['candidate_run'])
    old, new = load(source/'run_summary.json'), load(candidate/'run_summary.json')
    require_execution_scope(old, new, heldout_recording=not full)
    verify_frozen_dependencies(old, new)
    if sha(source/'navigation_states.npz') != d['source_states_SHA256']:
        raise ValueError('Original complete states changed')
    for name, expected in d['replay_outputs_SHA256'].items():
        if sha(candidate/name) != expected: raise ValueError('Candidate output changed: '+name)
    manifest_path = source/'recorded_inputs/input_manifest.json'
    if sha(manifest_path) != new['input_manifest_SHA256']:
        raise ValueError('Source input manifest changed')
    manifest = load(manifest_path)
    for name, expected in manifest['files_SHA256'].items():
        if sha(manifest_path.parent/name) != expected: raise ValueError('Actual navigation input changed')
    with np.load(source/'navigation_states.npz') as data:
        truth = {k: data[k] for k in data.files}
    _, _, state, _ = read_replay(candidate, truth)
    graph, trace = load(candidate/'graph_factors.json'), load(candidate/'mapped_range_change_trace.json')
    prefix = verify_preserve_prefix(old, new, truth, state, trace)
    amplitudes = np.load(source/'recorded_inputs/received_side_scan_amplitudes.npy', mmap_mode='r')
    physical = audit_mapped_range_inputs(graph, trace, truth, manifest['calibration'], amplitudes)
    if physical != d['actual_range_input_audit']:
        raise ValueError('Physical-source audit differs from the independent comparison')
    positions = dict(truth=truth['positions'][:, 0], DR=state['positions'][:, 0],
        Local=state['positions'][:, 1], baseline_Global=truth['positions'][:, 3],
        improved_Global=state['positions'][:, 2])
    errors = {k: np.linalg.norm(v-positions['truth'], axis=1) for k,v in positions.items() if k != 'truth'}
    statistics = {k: summarize(v) for k,v in errors.items()}
    for name, key in [('baseline_Global', 'baseline_Global'), ('improved_Global', 'candidate_Global')]:
        for low, high in [('rmse_m','RMSE_m'), ('p95_m','P95_m'), ('max_m','max_m'), ('final_m','final_m')]:
            np.testing.assert_allclose(statistics[name][low], d[key][high], rtol=0, atol=1e-10)
    first = prefix['first_new_range_timestamp_s']
    pre = truth['timestamps'] < first
    extra = dict(first_new_range_distance_m=float(np.interp(first, truth['timestamps'], truth['distance_m'])),
        worst={k:dict(distance_m=float(truth['distance_m'][np.argmax(v)]), error_m=float(v.max()))
               for k,v in errors.items()},
        before_first_range={k:summarize(v[pre]) for k,v in errors.items()},
        after_first_range={k:summarize(v[~pre]) for k,v in errors.items()},
        source_events=old['event_counts'], candidate_events=new['events'],
        elapsed_replay_s=new['elapsed_s'], original_core=old['original_core_source_SHA256'],
        prior_inputs=old['prior_intensity_inputs_SHA256'],
        online_states_not_final_history_smoothing=True)
    return dict(comparison=d, comparison_path=str(comparison.resolve()), comparison_SHA256=sha(comparison),
                statistics=statistics, extra=extra), truth, positions, errors, trace


def copy_local(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha(source) != sha(target): raise ValueError('Copy changed report evidence')


def export_trial(out, label, trial):
    summary, truth, positions, errors, _ = trial
    t, distance = truth['timestamps'], truth['distance_m']
    write_csv(out/f'data/{label}_errors_full.csv',
        ['timestamp_s','distance_m',*[name+'_error_m' for name in errors]],
        np.column_stack([t, distance, *errors.values()]))
    write_csv(out/f'data/{label}_trajectories_full.csv.gz',
        ['timestamp_s','distance_m',*[name+'_'+c+'_m' for name in positions for c in 'xyz']],
        np.column_stack([t, distance, *positions.values()]))
    write_csv(out/f'data/{label}_segments_100m.csv',
        ['start_m','end_m','samples','channel','rmse_m','median_m','p95_m','max_m','final_m'],
        segmented(distance, errors))
    copy_local(summary['comparison_path'], out/f'evidence/{label}_independent_comparison.json')


def test_totals(path):
    result = {k:0 for k in ['tests','failures','errors','skipped']}
    for suite in ET.parse(path).getroot().iter('testsuite'):
        for key in result: result[key] += int(suite.get(key, 0))
    if result['failures'] or result['errors'] or result['skipped']:
        raise ValueError('Report regression scope contains a failure or skip')
    return result


def prepare(out):
    out = Path(out).resolve()
    if out.exists(): raise FileExistsError('Preserve all previous reports')
    main = read_trial(BASE/'review_full_preserve_legacy_seed31_v5/independent_comparison.json', full=True)
    fresh = read_trial(BASE/'blind_seed33_4km_v1/review_v5/independent_comparison.json', full=False)
    driver = load(BASE/'blind_seed33_4km_status.json')
    if (driver['status'] != 'fresh_seed_source_replay_and_independent_review_complete_pending_result_review'
            or driver.get('completed_steps') != 3 or not driver['parameters_frozen_before_fresh_recording']
            or not driver['source_and_candidate_run_serially']):
        raise ValueError('Completed predeclared fresh-noise protocol required')
    for path, expected in driver['frozen_source_SHA256'].items():
        if sha(path) != expected: raise ValueError('Predeclared source changed')
    for key in ['original_core','prior_inputs']:
        if main[0]['extra'][key] != fresh[0]['extra'][key]: raise ValueError('Main and fresh dependencies differ')
    for name in ['data','figures','evidence']: (out/name).mkdir(parents=True)
    export_trial(out, 'main', main); export_trial(out, 'fresh', fresh)
    extras = {}
    for name, path in [('tempered_full', BASE/'review_full_mapped_ranges_seed31_v3/independent_comparison.json'),
                       ('tempered_development', BASE/'review_heldout_seed32_mapped_ranges_v3/independent_comparison.json'),
                       ('preserved_development', BASE/'review_validation_seed32_preserve_v4/independent_comparison.json')]:
        d = load(path)
        if not d['status'].startswith('completed_independent_'): raise ValueError('Incomplete development evidence')
        for file, expected in d['replay_outputs_SHA256'].items():
            if sha(Path(d['candidate_run'])/file) != expected: raise ValueError('Development output changed')
        copy_local(path, out/f'evidence/{name}.json'); extras[name] = d
    regression = {}
    for name, count in [('regression_candidate_and_independent_audit_v5.xml',117),
                         ('regression_original_sea_lake_core_v1.xml',161)]:
        regression[name] = test_totals(BASE/name)
        if regression[name]['tests'] != count: raise ValueError('Unexpected declared regression count')
        copy_local(BASE/name, out/'evidence'/name)
    save(out/'data/verified_report_results.json', dict(status='completed_full_and_predeclared_fresh_report_evidence',
        main=main[0], fresh=fresh[0], development=extras, regression=regression,
        fresh_sources_reverified=len(driver['frozen_source_SHA256']),
        same_main_fresh_core_and_prior=True, raw_payloads_copied=False, external_upload_performed=False))
    copy_local(EXPERIMENT/'scene/scene_report.json', out/'evidence/scene_report.json')
    copy_local(PREVIOUS/'海底地形地貌匹配算法仿真报告.md', out/'evidence/previous_report_source.md')
    for relative in ['new_field_comb_trajectory.png','new_received_feature_match.png','approved_enhanced_portrait.png',
                     'lake_hil_board_position_errors.png','lake_hil_board_3d_trajectories.png',
                     'sea_hil_board_position_errors.png','sea_hil_board_3d_trajectories.png']:
        copy_local(PREVIOUS/'figures'/relative, out/'figures'/relative)
    # Keep historical numerical claims tied to their original reviewed archive.
    historical = ROOT/'docs/reports/seabed_matching_simulation_20261107'
    for name in ['hil_cross_scene_scope.json','hil_log_memory_followup.json']:
        copy_local(historical/'evidence'/name, out/'evidence/historical'/name)
    copy_local(historical/'data/verified_metrics.json', out/'evidence/historical/verified_metrics.json')
    plot(out, main, fresh)
    return out


def plot(out, main, fresh):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.font_manager import FontProperties
    from PlotPureSideScanTerrain import terrain_mesh
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists(): plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams['path.simplify'] = False
    for label, trial in [('main',main),('fresh',fresh)]:
        summary, truth, positions, errors, trace = trial
        x = truth['distance_m']/1000
        fig, axes = plt.subplots(2,1,figsize=(13,7),sharex=True,constrained_layout=True)
        for name, title, color in [('DR','航位推算 DR','#8d96a2'),('Local','局部 Local','#dc8122'),
                ('baseline_Global','原版 Global','#7851a9'),('improved_Global','改进 Global','#008d7f')]:
            axes[0].plot(x,errors[name],color=color,lw=.7,label=title)
            if 'Global' in name: axes[1].plot(x,errors[name],color=color,lw=.8,label=title)
        for ax in axes: ax.grid(alpha=.2); ax.set_ylabel('三维位置误差 / m'); ax.legend(ncol=2)
        axes[1].set_xlabel('累计航程 / km')
        title = '16.49 km 全程同输入对照' if label=='main' else '4.00 km 新噪声种子补充验证'
        axes[0].set_title(title+'：全部在线样本，未删除峰值')
        fig.savefig(out/f'figures/{label}_full_errors.png',dpi=220); plt.close(fig)
    summary, truth, positions, errors, trace = main
    x = truth['distance_m']/1000
    yaw = truth['yaw_rad'][:,0]
    fig, axes = plt.subplots(2,1,figsize=(13,6.5),sharex=True,constrained_layout=True)
    for name,title,color in [('baseline_Global','原版 Global','#7851a9'),('improved_Global','改进 Global','#008d7f')]:
        e=positions[name]-positions['truth']
        along=e[:,0]*np.cos(yaw)+e[:,1]*np.sin(yaw)
        cross=-e[:,0]*np.sin(yaw)+e[:,1]*np.cos(yaw)
        axes[0].plot(x,along,lw=.7,label=title,color=color)
        axes[1].plot(x,cross,lw=.7,label=title,color=color)
    for ax,title in zip(axes,['沿航向有符号误差 / m','横航向有符号误差 / m']):
        ax.set_ylabel(title); ax.grid(alpha=.2); ax.legend()
    axes[0].set_title('主试验误差方向分解：采用评价真值航向，仅用于事后统计')
    axes[1].set_xlabel('累计航程 / km')
    fig.savefig(out/'figures/main_directional_errors.png',dpi=220); plt.close(fig)
    times=np.array([r['timestamp_s'] for r in trace]); distances=np.interp(times,truth['timestamps'],x)
    fig, axes=plt.subplots(2,1,figsize=(13,5.8),sharex=True,constrained_layout=True)
    axes[0].step(distances,np.arange(1,len(times)+1),where='post',color='#008d7f',lw=1.1)
    axes[1].scatter(distances,[r['features'] for r in trace],s=10,color='#3678b3')
    axes[0].set_ylabel('实际累计新增因子数'); axes[1].set_ylabel('每条因子配对特征数')
    axes[0].set_title('实际GTSAM新增斜距变化约束：独立来源、全局一次性距离token')
    axes[1].set_xlabel('累计航程 / km')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out/'figures/actual_range_constraints.png',dpi=220); plt.close(fig)
    field=EXPERIMENT/'scene/query_baseline'; meta=load(field/'metadata.json')
    xx,yy,zz=terrain_mesh(np.load(field/'elevation_m.npy',mmap_mode='r'),
        np.load(field/'bathymetry_valid_mask.npy',mmap_mode='r'),meta['resolution_m'],12)
    fig=plt.figure(figsize=(13,7),constrained_layout=True)
    ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    cmap=LinearSegmentedColormap.from_list('pale_relief',['#edf1f4','#cad6de','#aebfcb'])
    ax.plot_surface(xx,yy,zz,cmap=cmap,alpha=.6,linewidth=0,rcount=len(xx),ccount=xx.shape[1],
        rasterized=True,antialiased=False,zorder=1)
    for i,(name,label,color,style,width) in enumerate([
            ('truth','真值（仅评价）','#242424','--',.9),('Local','Local','#e8871e','-',1.),
            ('improved_Global','改进后在线 Global','#006adf','-',1.25)]):
        q=positions[name]; ax.plot(q[:,0],q[:,1],q[:,2],label=label,color=color,ls=style,lw=width,zorder=10+i)
    allz=np.r_[zz[np.isfinite(zz)],*[p[:,2] for p in positions.values()]]
    ax.set_box_aspect((float(np.ptp(xx)),float(np.ptp(yy)),3*float(np.ptp(allz))),zoom=1.1)
    ax.view_init(elev=48,azim=-67); ax.legend(loc='upper left')
    ax.set(xlabel='名义 X / m',ylabel='名义 Y / m',zlabel='Z / m',
        title='改进算法完整16.49 km：地形与三条实际轨迹\n仅显示Z比例为3倍，轨迹未偏移、未贴地')
    fig.savefig(out/'figures/improved_3d_trajectories.png',dpi=230); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); print(prepare(args.out))
