"""Independent full-short-run metrics, paired DR check and actual match plots."""
import argparse
import csv
import json
from pathlib import Path
import hashlib
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from QiandaoCombTrajectory import CombSurveyTrajectory, CombSurveyParameters


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def summarize(positions):
    error = np.linalg.norm(positions[:, 1:]-positions[:, :1], axis=2)
    return {n: dict(RMSE_m=float(np.sqrt(np.mean(error[:, k]**2))),
        P95_m=float(np.percentile(error[:, k], 95)), max_m=float(error[:, k].max()),
        final_m=float(error[-1, k])) for k, n in enumerate(['DR', 'Local', 'Global'])}


def planar_relative_metrics(positions, yaw, distances, *, segment_index=None, selected_segment=None):
    """Yaw-aligned planar RPE; fixed world gauge offsets do not count as drift."""
    result = {}
    for length in (10., 30.):
        end = np.searchsorted(distances, distances+length)
        start = np.flatnonzero(end < len(distances)); end = end[start]
        if selected_segment is not None:
            keep = (segment_index[start] == selected_segment)&(segment_index[end] == selected_segment)
            start, end = start[keep], end[keep]
        if not len(start):
            result[str(int(length))+'m'] = dict(pairs=0, RMSE_m=None)
            continue
        delta = positions[end, :, :2]-positions[start, :, :2]
        c, s = np.cos(yaw[start]), np.sin(yaw[start])
        relative = np.stack([c*delta[:, :, 0]+s*delta[:, :, 1], -s*delta[:, :, 0]+c*delta[:, :, 1]], axis=-1)
        error = np.linalg.norm(relative[:, 1:]-relative[:, :1], axis=-1)
        result[str(int(length))+'m'] = dict(pairs=len(start), RMSE_m={n: float(np.sqrt(np.mean(error[:, k]**2)))
            for k, n in enumerate(['DR', 'Local', 'Global'])})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--joint-reference', type=Path, help='Optional identical-seed legacy prefix for a bit-exact DR check.')
    p.add_argument('--experiment', type=Path, help='Defaults to the executed output parent directory.')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    r = json.loads((a.run/'run_summary.json').read_text())
    if r['status'] not in ('completed_pure_side_scan_FEATURE_short_test', 'completed_pure_side_scan_FEATURE_route_test') or not r['full_source_snapshots_verified']:
        raise ValueError('Completed, source-verified run required; no partial-run acceptance.')
    if r['event_counts'] != r['expected_events'] or r['secondary_denoising_used']:
        raise ValueError('All sensor events and no secondary denoising required.')
    if a.out.exists(): raise FileExistsError('Preserve earlier independent reviews.')
    csv_path, state_path = a.run/'trajectory.csv', a.run/'navigation_states.npz'
    if digest(csv_path) != r['trajectory_SHA256'] or digest(state_path) != r['navigation_states_SHA256']:
        raise ValueError('Executed numeric artifacts changed.')
    for name, expected in r['source_code_SHA256'].items():
        if digest(a.run/'source_snapshot'/name) != expected: raise ValueError('Executed source snapshot changed: '+name)
    rows = read_rows(csv_path); state = np.load(state_path)
    if len(rows) != r['processed_pings'] or len(rows) != r['expected_events']['sonar']:
        raise ValueError('Full ping coverage required.')
    position = np.array([[float(x[f'{n}_{c}']) for n in ['truth', 'DR', 'Local', 'Global'] for c in 'xyz']
                         for x in rows]).reshape(-1, 4, 3)
    np.testing.assert_array_equal(position, state['positions'])
    metrics = summarize(position)
    for n in metrics:
        for key in metrics[n]:
            if not np.isclose(metrics[n][key], r['metrics'][n][key], atol=1e-12):
                raise ValueError('Independent metrics differ from the executed report.')
    paired_legacy_DR = None
    if a.joint_reference:
        paired = []
        with (a.joint_reference/'trajectory.csv').open() as f:
            for row in csv.DictReader(f):
                if len(paired) == len(rows): break
                paired.append(row)
        if len(paired) != len(rows): raise ValueError('Actual paired reference prefix missing.')
        t = np.array([float(x['time_s']) for x in paired])
        np.testing.assert_array_equal(t, state['timestamps'])
        joint_dr = np.array([[float(x['DR_'+c]) for c in 'xyz'] for x in paired])
        np.testing.assert_array_equal(joint_dr, position[:, 1]); paired_legacy_DR = True
    experiment = a.experiment or a.run.parent
    spec = json.loads((experiment/'AUV全幅梳形轨迹.json').read_text())['parameters']
    route = CombSurveyTrajectory(CombSurveyParameters(**spec))
    analytic_positions = np.array([route.position(float(t)) for t in state['timestamps']])
    np.testing.assert_allclose(position[:, 0], analytic_positions, atol=1e-10, rtol=0)
    segment_index = np.minimum(np.searchsorted(route.ends_m, state['distance_m'], side='right'), len(route.segments)-1)
    if 'segment_index' in state:
        np.testing.assert_array_equal(segment_index, state['segment_index'])
    if r.get('full_closed_loop_requested') and not np.isclose(r['requested_distance_m'], route.length_m, atol=1e-9):
        raise ValueError('A prefix cannot be relabeled a complete route.')
    phase_metrics = {}
    for index in np.unique(segment_index):
        keep = segment_index == index
        phase_metrics[f'{index}_{route.segments[index].kind}'] = dict(samples=int(keep.sum()),
            start_m=float(state['distance_m'][keep][0]), end_m=float(state['distance_m'][keep][-1]),
            metrics=summarize(position[keep]), planar_relative=planar_relative_metrics(position,
                state['yaw_rad'], state['distance_m'], segment_index=segment_index, selected_segment=index))
    factors = json.loads((a.run/'feature_graph_factors.json').read_text())
    kinds = [x['kind'] for x in factors['priors']]
    if kinds and set(kinds) != {'side_scan_feature_prior'}:
        raise ValueError('Unexpected absolute observation source in pure side-scan test.')
    diagnostics = r['diagnostics']
    for field in ['MBES_points_consumed', 'measured_height_likelihood_terms',
                  'measured_height_graph_factors', 'optical_flow_calls', 'raw_intensity_correlation_calls']:
        if diagnostics[field] != 0: raise ValueError('Forbidden source: '+field)
    a.out.mkdir()
    recorded_inputs_verified = None
    if r.get('recorded_navigation_inputs'):
        from SideScanRecordedInputs import RecordedSideScanInputs
        recorded = RecordedSideScanInputs(a.run/'recorded_inputs')
        if recorded.manifest['events'] != r['event_counts']: raise ValueError('Recorded input coverage differs.')
        np.testing.assert_array_equal(recorded.timestamps, state['timestamps'])
        recorded_inputs_verified = True
    review = dict(status='independently_verified_complete_executed_scope', distance_m=float(state['distance_m'][-1]),
        source_run_SHA256=digest(a.run/'run_summary.json'), metrics=metrics,
        paired_legacy_DR_bit_exact=paired_legacy_DR, side_scan_absolute_feature_priors=len(kinds),
        source_snapshots_independently_rehashed=True, recorded_navigation_inputs_verified=recorded_inputs_verified,
        analytic_trajectory_checked=True, full_closed_loop_requested=bool(r.get('full_closed_loop_requested', False)),
        phase_metrics=phase_metrics, planar_relative=planar_relative_metrics(position, state['yaw_rad'], state['distance_m']),
        graph_optimizer=diagnostics['optimizer_backend'], secondary_denoising_used=False,
        Local_RMSE_reduction_vs_DR=1-metrics['Local']['RMSE_m']/metrics['DR']['RMSE_m'],
        Global_RMSE_reduction_vs_DR=1-metrics['Global']['RMSE_m']/metrics['DR']['RMSE_m'],
        coverage=diagnostics['feature_map'],
        does_NOT_prove_full_comb_or_unseen_map_generalization=True,
        same_voyage_nominal_map_NOT_independent_real_field_navigation_truth=True)
    (a.out/'independent_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists(): plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    error = np.linalg.norm(position[:, 1:]-position[:, :1], axis=2)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for k, (name, color) in enumerate(zip(['DR', 'Local', 'Global'], ['#858b93', '#ed8a2e', '#087fc2'])):
        ax.plot(state['distance_m'], error[:, k], color=color, lw=1.4,
                label=f"{name}：RMSE {metrics[name]['RMSE_m']:.3f} m")
    for index in np.unique(segment_index):
        if 'turn' in route.segments[index].kind:
            keep = segment_index == index
            ax.axvspan(state['distance_m'][keep][0], state['distance_m'][keep][-1], color='#b6bcc4', alpha=.22)
    ax.set(xlabel='累计航程 / m', ylabel='三维位置误差 / m',
        title='纯侧扫特征导航短程对照：无二次去噪、无测深约束\n同一传感器流，准确初值；不是全程或独立实测精度验收')
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(a.out/'纯侧扫_无二次去噪_完整短程误差.png', dpi=160); plt.close(fig)
    example_files = sorted(a.run.glob('actual_map_feature_matches_*.npz'))
    for index, path in enumerate(example_files):
        example = np.load(path); q, m = example['query_image'], example['map_image']
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), constrained_layout=True)
        axes[0].imshow(q, cmap='gray', vmin=0, vmax=255, aspect='equal')
        axes[1].imshow(m, cmap='gray', vmin=0, vmax=255, aspect='equal')
        for k, (qp, mp) in enumerate(zip(example['query_pixels'][:60], example['map_pixels'][:60])):
            color = plt.cm.tab20((k%20)/20)
            axes[0].plot(qp[0], qp[1], '.', color=color, ms=4)
            axes[1].plot(mp[0], mp[1], '.', color=color, ms=4)
        axes[0].set_title('实际当前＋历史回波拼接的查询子图（不是底图裁图）')
        axes[1].set_title('参考强度场候选区域；同色点为扫描平面验证内点，不等同人工标注真值')
        fig.savefig(a.out/f'实际侧扫地图特征_{index}.png', dpi=150); plt.close(fig)
    print(json.dumps(review, ensure_ascii=False))


if __name__ == '__main__': main()
