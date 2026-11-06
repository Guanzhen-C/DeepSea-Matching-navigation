"""Independently compare complete, identical-input SSS loop-off/on replays.

Evaluation only: truth, phase labels and errors NEVER enter the estimator.
Reports zero loops or worse errors honestly, without an acceptance shortcut.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from AnalyzePureSideScanShort import planar_relative_metrics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def metrics(error):
    return dict(RMSE_m=float(np.sqrt(np.mean(error**2))), P95_m=float(np.percentile(error, 95)),
                max_m=float(np.max(error)), final_m=float(error[-1]))


def validate_comparison(reference, candidate, kind):
    keys = ['loop_closure', 'scan_plane_registration', 'canonical_submaps']
    defaults = [False, 'rank2_baseline', False]
    old = [reference.get(k, d) for k, d in zip(keys, defaults)]
    new = [candidate.get(k, d) for k, d in zip(keys, defaults)]
    changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    expected = {'loop': 0, 'registration': 1, 'submap_orientation': 2}[kind]
    if changed != [expected]: raise ValueError('Comparison must change exactly the declared estimator component.')
    if kind == 'loop' and (old[0] or not new[0]): raise ValueError('Explicit loop off/on required.')
    if kind != 'loop' and (old[0] or new[0]): raise ValueError('Representation/registration comparison requires loops OFF.')


def read_replay(path, truth):
    report = json.loads((path/'run_summary.json').read_text())
    if report['status'] != 'completed_lossless_navigation_input_replay':
        raise ValueError('Only terminal complete replays can be compared.')
    for name, expected in report['source_SHA256'].items():
        if sha(path/'source_snapshot'/name) != expected: raise ValueError('Snapshot changed: '+name)
    if sha(path/'source_snapshot/ReplayPureSideScanNavigation.py') != report['replay_driver_SHA256']:
        raise ValueError('Replay driver snapshot changed.')
    if report.get('strict_report_writer_SHA256') and sha(path/'source_snapshot/WriteSideScanReport.py') != report['strict_report_writer_SHA256']:
        raise ValueError('Strict report writer snapshot changed.')
    state = np.load(path/'estimated_states.npz')
    np.testing.assert_array_equal(state['timestamps'], truth['timestamps'])
    np.testing.assert_array_equal(state['positions'][:, :2], truth['positions'][:, 1:3])
    np.testing.assert_array_equal(state['distance_m'], truth['distance_m'])
    if len(state['timestamps']) != report['estimated_frames']:
        raise ValueError('Missing saved estimates.')
    with (path/'estimated_positions.csv').open() as f:
        rows = list(csv.DictReader(f))
    np.testing.assert_array_equal([float(row['timestamp_s']) for row in rows], state['timestamps'])
    csv_positions = np.array([[float(row[f'{n}_{c}']) for n in ['DR', 'Local', 'Global'] for c in 'xyz']
                              for row in rows]).reshape(-1, 3, 3)
    np.testing.assert_array_equal(csv_positions, state['positions'])
    error = np.linalg.norm(state['positions']-truth['positions'][:, :1], axis=2)
    calculated = {name: metrics(error[:, k]) for k, name in enumerate(['DR', 'Local', 'Global'])}
    for n in calculated:
        for k in calculated[n]:
            np.testing.assert_allclose(calculated[n][k], report['metrics'][n][k], rtol=0, atol=1e-11)
    graph = json.loads((path/'graph_factors.json').read_text())
    loops = [e for e in graph['edges'] if e['kind'] == 'loop']
    claimed = report['diagnostics']['loop_closure']['actual_loop_edges']
    if len(loops) != claimed or len({(e['source'], e['target']) for e in loops}) != len(loops):
        raise ValueError('Missing or duplicated real loop factors.')
    loop_checks = []
    for edge in loops:
        separation = graph['pose_timestamps_s'][edge['target']]-graph['pose_timestamps_s'][edge['source']]
        info = np.asarray(edge['information']); values = np.linalg.eigvalsh(info)
        if separation < 180. or not np.isfinite(info).all() or values.min() < -1e-8 or values.max() <= 0:
            raise ValueError('Loop is not long-baseline or has invalid information.')
        loop_checks.append(dict(source=edge['source'], target=edge['target'], separation_s=separation,
            rank=int(np.linalg.matrix_rank(info, tol=1e-8)), information_eigenvalues=values.tolist()))
    if report['loop_closure']:
        trace = json.loads((path/'historical_loop_trace.json').read_text())
        if trace and 'actual_loop_edge_added' in trace[0]:
            actual = [row['factor'] for row in trace if row['actual_loop_edge_added']]
            if len(actual) != len(loops): raise ValueError('Loop diagnostics do not match actual graph factors.')
            for a, b in zip(actual, loops):
                if (a['source'], a['target']) != (b['source'], b['target']):
                    raise ValueError('Loop endpoint mismatch.')
                np.testing.assert_array_equal(a['measurement'], b['measurement'])
                np.testing.assert_allclose(a['information'], b['information'], rtol=0, atol=1e-9)
    positions = np.concatenate([truth['positions'][:, :1], state['positions']], axis=1)
    yaw = np.column_stack([truth['yaw_rad'][:, 0], state['yaw_rad']])
    phases = {}
    for segment in np.unique(truth['segment_index']):
        selected = truth['segment_index'] == segment
        phases[str(segment)] = dict(samples=int(selected.sum()),
            global_error=metrics(error[selected, 2]),
            planar_RPE=planar_relative_metrics(positions, yaw, truth['distance_m'],
                segment_index=truth['segment_index'], selected_segment=segment))
    return report, dict(metrics=calculated, actual_loop_edges=len(loops), loop_factors=loop_checks,
        global_anchor_bridge=graph['global_anchor_bridge'], phases=phases,
        output_SHA256={n: sha(path/n) for n in ['estimated_positions.csv', 'estimated_states.npz', 'graph_factors.json']}), state, error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True, type=Path)
    p.add_argument('--without-loops', '--reference', dest='without_loops', required=True, type=Path)
    p.add_argument('--with-loops', '--candidate', dest='with_loops', required=True, type=Path)
    p.add_argument('--comparison', choices=('loop', 'registration', 'submap_orientation'), default='loop')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError('Preserve earlier reviews.')
    truth = np.load(a.source_run/'navigation_states.npz')
    source_report = json.loads((a.source_run/'run_summary.json').read_text())
    if sha(a.source_run/'navigation_states.npz') != source_report['navigation_states_SHA256']:
        raise ValueError('Original evaluation artifact changed.')
    off, off_review, off_state, off_error = read_replay(a.without_loops, truth)
    on, on_review, on_state, on_error = read_replay(a.with_loops, truth)
    validate_comparison(off, on, a.comparison)
    if off['input_manifest_SHA256'] != on['input_manifest_SHA256']:
        raise ValueError('A/B input streams differ.')
    manifest_path = a.source_run/'recorded_inputs/input_manifest.json'
    if sha(manifest_path) != on['input_manifest_SHA256']: raise ValueError('Input manifest changed.')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files_SHA256'].items():
        if sha(manifest_path.parent/name) != expected: raise ValueError('Input archive changed.')
    if off['events'] != manifest['events'] or on['events'] != manifest['events']:
        raise ValueError('Not every received sensor event was processed.')
    np.testing.assert_array_equal(off_state['positions'][:, :2], on_state['positions'][:, :2])
    review = dict(status='completed_independent_identical_input_'+a.comparison+'_comparison', comparison_kind=a.comparison,
        all_events_and_input_SHA256_verified=True, frontend_and_DR_bit_exact=True,
        source_run=str(a.source_run.resolve()), reference=off_review, candidate=on_review,
        Global_RMSE_change_percent=float(100*(on_review['metrics']['Global']['RMSE_m']/off_review['metrics']['Global']['RMSE_m']-1)),
        limitations=['One controlled simulated route and seed, not independent real-field validation.',
            'Side-scan unknown-height ambiguities retained; no DEM/MBES/altimeter supplied.',
            'Historical/map observation correlations approximate, not a full joint landmark graph.',
            'No secondary denoising; source scene is built from previously denoised measured amplitudes.'])
    if a.comparison == 'loop': review.update(without_loops=off_review, with_loops=on_review)
    a.out.mkdir()
    (a.out/'independent_comparison.json').write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = [f.name for f in font_manager.fontManager.ttflist if 'Noto Sans CJK' in f.name]
    if fonts: plt.rcParams['font.family'] = fonts[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    before_label = {'loop': '无回环', 'registration': '固定两自由度', 'submap_orientation': '原始拼接方向'}[a.comparison]
    after_label = {'loop': '历史子图回环', 'registration': '可观子空间', 'submap_orientation': '舷侧与平面方向对齐'}[a.comparison]
    d = truth['distance_m']/1000
    axes[0].plot(d, off_error[:, 0], color='#888888', lw=1., label='DR')
    axes[0].plot(d, off_error[:, 1], color='#e67e22', lw=1., label='Local（两组逐值相同）')
    axes[0].plot(d, off_error[:, 2], color='#0072b2', lw=1.3, label='Global：'+before_label)
    axes[0].plot(d, on_error[:, 2], color='#009e73', lw=1.3, label='Global：'+after_label)
    axes[0].set(xlabel='累计航程 / km', ylabel='三维位置误差 / m')
    axes[0].legend(); axes[0].grid(alpha=.2)
    gt = truth['positions'][:, 0]
    axes[1].plot(gt[:, 0], gt[:, 1], color='#222222', lw=1.5, label='真值（仅评价）')
    for state, color, label in [(off_state, '#0072b2', 'Global：'+before_label), (on_state, '#009e73', 'Global：'+after_label)]:
        axes[1].plot(state['positions'][:, 2, 0], state['positions'][:, 2, 1], color=color, lw=1.2, label=label)
    axes[1].set(xlabel='场景 X / m', ylabel='场景 Y / m'); axes[1].set_aspect('equal', adjustable='datalim')
    axes[1].legend(); axes[1].grid(alpha=.2)
    experiment_name = {'loop': '历史子图回环', 'registration': '扫描平面可观性', 'submap_orientation': '反向扫描拼接'}[a.comparison]
    fig.suptitle(f'同输入纯侧扫{experiment_name}对照：{d[-1]:.2f} km，实际回环因子 {on_review["actual_loop_edges"]} 条')
    fig.savefig(a.out/f'纯侧扫_{experiment_name}_同输入对照.png', dpi=200)
    plt.close(fig)
    print(json.dumps(review, ensure_ascii=False))


if __name__ == '__main__': main()
