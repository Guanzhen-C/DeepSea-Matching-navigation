"""Post-estimation error directions and actual graph information, not tuning.

Only complete, source-verified runs are accepted. Reference truth is opened
after that check, never supplied to an estimator. Information diagonals are
conditional curvature, NOT marginal accuracy or calibrated confidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from AnalyzeSideScanLoopReplay import read_replay, sha


def directional_errors(positions):
    p = np.asarray(positions, float)
    if p.ndim != 3 or p.shape[1:] != (4, 3) or not len(p) or not np.isfinite(p).all():
        raise ValueError('Finite complete truth/DR/Local/Global position arrays required.')
    error = p[:, 1:]-p[:, :1]
    mse = np.mean(error**2, axis=0); totals = mse.sum(axis=1)
    fraction = np.divide(mse, totals[:, None], out=np.zeros_like(mse), where=totals[:, None] > 0)
    summary = {name: dict(axis_RMSE_xyz_m=np.sqrt(mse[k]).tolist(),
        RMSE_3d_m=float(np.sqrt(totals[k])), axis_MSE_fractions_xyz=fraction[k].tolist(),
        final_error_xyz_m=error[-1, k].tolist()) for k, name in enumerate(['DR', 'Local', 'Global'])}
    return error, summary


def absolute_information_summary(priors):
    if not priors: return dict(count=0, rank_counts={}, Y_over_X_diagonal_ratio=None)
    info = np.array([p['information'] for p in priors], float)
    if info.shape[1:] != (3, 3) or not np.isfinite(info).all():
        raise ValueError('Finite 3x3 graph information matrices required.')
    np.testing.assert_allclose(info, info.transpose(0, 2, 1), rtol=0, atol=1e-8)
    values = np.linalg.eigvalsh(info)
    if np.min(values) < -1e-8: raise ValueError('Information must be positive semidefinite.')
    ranks, count = np.unique(np.sum(values > 1e-8, axis=1), return_counts=True)
    positive_x = info[:, 0, 0] > 0
    ratio = info[positive_x, 1, 1]/info[positive_x, 0, 0]
    def describe(a):
        return None if not len(a) else dict(min=float(np.min(a)), median=float(np.median(a)), max=float(np.max(a)))
    return dict(count=len(info), rank_counts={str(k): int(n) for k, n in zip(ranks, count)},
        X_diagonal=describe(info[:, 0, 0]), Y_diagonal=describe(info[:, 1, 1]),
        Y_over_X_diagonal_ratio=describe(ratio), nonpositive_X_excluded_from_ratio=int((~positive_x).sum()),
        information_diagonals_are_not_marginal_covariance=True)


def relative_pose(source, target):
    c, s = np.cos(source[2]), np.sin(source[2]); delta = np.asarray(target)-source
    return np.array([c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1],
                     np.arctan2(np.sin(delta[2]), np.cos(delta[2]))])


def evaluate_loop_information(factors, truth_pose2, keyframe_distance):
    rows = []
    for edge in factors['edges']:
        if edge['kind'] != 'loop': continue
        a, b = edge['source'], edge['target']
        error = relative_pose(truth_pose2[a], truth_pose2[b])-np.asarray(edge['measurement'])
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        info = np.asarray(edge['information'], float)
        q = float(error@info@error)
        if not np.isfinite(q) or q < -1e-8: raise ValueError('Invalid weighted loop residual.')
        rows.append(dict(source=a, target=b, source_distance_m=float(keyframe_distance[a]),
            target_distance_m=float(keyframe_distance[b]), error_xy_yaw_rad=error.tolist(),
            squared_information_weighted_residual=max(q, 0.),
            information_rank=int(np.linalg.matrix_rank(info, tol=1e-8))))
    q = np.array([r['squared_information_weighted_residual'] for r in rows])
    return dict(count=len(rows), squared_residual_min_median_P95_max=None if not len(q) else
        np.percentile(q, [0, 50, 95, 100]).tolist(),
        largest_weighted_residuals=sorted(rows, key=lambda x: x['squared_information_weighted_residual'], reverse=True)[:8],
        truth_used_for_evaluation_only=True, not_a_calibrated_chi_square_test=True,
        large_unobservable_error_is_not_a_bad_observable_constraint=True)


def require_completed_inputs(original, review, report_digest, replay_report=None):
    if (not original.get('status', '').startswith('completed_pure_side_scan_FEATURE_')
            or review.get('status') != 'independently_verified_complete_executed_scope'
            or review.get('source_run_SHA256') != report_digest):
        raise ValueError('A completed, independently verified source scope is required.')
    if replay_report is not None and replay_report.get('status') != 'completed_lossless_navigation_input_replay':
        raise ValueError('Do not evaluate a running/partial replay.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True, type=Path)
    p.add_argument('--source-review', required=True, type=Path, help='Completed independent_review.json for the source.')
    p.add_argument('--replay-run', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError('Keep earlier diagnostic outputs.')
    source_report_path = a.source_run/'run_summary.json'
    original = json.loads(source_report_path.read_text())
    review = json.loads(a.source_review.read_text())
    candidate = None if not a.replay_run else json.loads((a.replay_run/'run_summary.json').read_text())
    require_completed_inputs(original, review, sha(source_report_path), candidate)
    path = a.source_run/'navigation_states.npz'
    if sha(path) != original['navigation_states_SHA256']: raise ValueError('Source states changed.')
    with np.load(path) as archive: truth = {k: archive[k] for k in archive.files}
    if a.replay_run:
        _, _, state, _ = read_replay(a.replay_run, truth)
        positions = np.concatenate([truth['positions'][:, :1], state['positions']], axis=1)
        run, factors_path = a.replay_run, a.replay_run/'graph_factors.json'
    else: positions, run, factors_path = truth['positions'], a.source_run, a.source_run/'feature_graph_factors.json'
    error, components = directional_errors(positions)
    factors = json.loads(factors_path.read_text())
    stamps = truth['timestamps']; distance = truth['distance_m']; key_stamps = np.array(factors['pose_timestamps_s'])
    idx = np.searchsorted(stamps, key_stamps)
    np.testing.assert_allclose(stamps[idx], key_stamps, rtol=0, atol=1e-9)
    key_truth = np.column_stack([truth['positions'][idx, 0, :2], truth['yaw_rad'][idx, 0]])
    intervals = {}
    for lo in np.arange(0., distance[-1], 2000.):
        selected = (distance >= lo)&(distance < lo+2000.)
        if not selected.any(): continue
        _, item = directional_errors(positions[selected])
        intervals[f'{lo:g}-{min(lo+2000., distance[-1]):g}m'] = dict(samples=int(selected.sum()), metrics=item)
    result = dict(status='completed_post_estimation_directional_information_diagnostic',
        run=str(run.resolve()), source_run=str(a.source_run.resolve()), samples=len(stamps),
        distance_m=float(distance[-1]), source_report_SHA256=sha(source_report_path),
        source_review_SHA256=sha(a.source_review), navigation_states_SHA256=sha(path),
        run_report_SHA256=sha(run/'run_summary.json'),
        evaluated_states_SHA256=sha(a.replay_run/'estimated_states.npz') if a.replay_run else sha(path),
        graph_factors_SHA256=sha(factors_path), axis_order=['world_X', 'world_Y', 'world_Z'],
        errors=components, intervals=intervals, absolute_priors=absolute_information_summary(factors['priors']),
        loops=evaluate_loop_information(factors, key_truth, distance[idx]),
        estimator_changed=False, truth_not_used_for_online_rejection_or_parameter_selection=True,
        limitation='World Y is cross-track on the X-aligned survey lines only; information rank/diagonals do not prove calibrated uncertainty.')
    a.out.mkdir()
    snapshot = a.out/'source_snapshot'; snapshot.mkdir()
    result['analysis_source_SHA256'] = {}
    for name in ['AnalyzeSideScanDirectionalError.py', 'AnalyzeSideScanLoopReplay.py', 'AnalyzePureSideScanShort.py']:
        source = Path(__file__).with_name(name)
        expected = sha(source); shutil.copy2(source, snapshot/name)
        if sha(snapshot/name) != expected: raise RuntimeError('Analysis source snapshot changed during copy.')
        result['analysis_source_SHA256'][name] = expected
    (a.out/'directional_information_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for k, (label, color) in enumerate(zip(['Global X', 'Global Y', 'Global Z'], ['#0072b2', '#d55e00', '#999999'])):
        axes[0].plot(distance/1000, error[:, 2, k], color=color, lw=1., label=label)
    axes[0].axhline(0, color='black', lw=.5); axes[0].set(ylabel='Signed position error / m')
    priors = factors['priors']
    if priors:
        information = np.array([q['information'] for q in priors]); targets = np.array([q['target'] for q in priors])
        positive = information[:, 0, 0] > 0
        ratio = information[positive, 1, 1]/information[positive, 0, 0]
        axes[1].semilogy(distance[idx[targets[positive]]]/1000, np.maximum(ratio, 1e-16), '.', ms=3,
            label='Map prior Iyy/Ixx (display floor 1e-16 only)')
    axes[1].set(xlabel='Cumulative distance / km', ylabel='Conditional information ratio')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Post-estimation directional audit; no truth feedback or estimator changes')
    fig.savefig(a.out/'directional_error_and_information.png', dpi=180); plt.close(fig)
    print(json.dumps(dict(status=result['status'], samples=len(stamps), errors=components,
        absolute_priors=result['absolute_priors']), ensure_ascii=False))


if __name__ == '__main__': main()
