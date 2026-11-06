"""Plot a COMPLETE SSS replay over simulator terrain, for evaluation only.

Exactly three native-sample trajectories: truth, Local, online Global. Terrain
is read only AFTER the estimator's complete outputs have been verified. It is
never an observation for the pure-side-scan navigator. No trajectory offsets,
terrain draping or invented gap filling are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from AnalyzeSideScanDirectionalError import require_completed_inputs
from AnalyzeSideScanLoopReplay import read_replay, sha


def terrain_mesh(elevation, valid, resolution, step=12):
    elevation, valid = np.asarray(elevation), np.asarray(valid, bool)
    if (elevation.ndim != 2 or min(elevation.shape) < 2 or valid.shape != elevation.shape or not np.isfinite(resolution)
            or resolution <= 0 or not isinstance(step, int) or step < 1):
        raise ValueError('A calibrated terrain grid and positive display-only mesh stride are required.')
    rows = np.unique(np.r_[np.arange(0, elevation.shape[0], step), elevation.shape[0]-1])
    cols = np.unique(np.r_[np.arange(0, elevation.shape[1], step), elevation.shape[1]-1])
    x, y = np.meshgrid((cols-(elevation.shape[1]-1)/2)*resolution,
                       (rows-(elevation.shape[0]-1)/2)*resolution)
    z = elevation[np.ix_(rows, cols)].astype(float).copy()
    z[~valid[np.ix_(rows, cols)]] = np.nan
    if not np.isfinite(z).any(): raise ValueError('No known terrain available for display.')
    return x, y, z


def select_three_trajectories(positions):
    p = np.asarray(positions)
    if p.ndim != 3 or p.shape[1:] != (4, 3) or not len(p) or not np.isfinite(p).all():
        raise ValueError('Complete finite truth/DR/Local/Global positions are required.')
    return {'Truth': p[:, 0].copy(), 'Local': p[:, 2].copy(), 'Global': p[:, 3].copy()}


def require_full_plot_scope(original, review):
    if not original.get('full_closed_loop_requested') or not review.get('complete_planned_route_verified'):
        raise ValueError('This full-route figure cannot be generated from a prefix experiment.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--source-run', required=True, type=Path)
    p.add_argument('--source-review', required=True, type=Path)
    p.add_argument('--replay-run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--mesh-step', type=int, default=12)
    p.add_argument('--vertical-exaggeration', type=float, default=3.)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError('Preserve existing figures.')
    if not np.isfinite(a.vertical_exaggeration) or a.vertical_exaggeration <= 0:
        raise ValueError('Finite positive display aspect is required.')
    original = json.loads((a.source_run/'run_summary.json').read_text())
    candidate = json.loads((a.replay_run/'run_summary.json').read_text())
    source_review = json.loads(a.source_review.read_text())
    require_completed_inputs(original, source_review, sha(a.source_run/'run_summary.json'), candidate)
    require_full_plot_scope(original, source_review)
    if sha(a.experiment/'scene/scene_report.json') != original['simulator_scene_report_SHA256']:
        raise ValueError('Display experiment does not match the executed simulator scene report.')
    state_path = a.source_run/'navigation_states.npz'
    if sha(state_path) != original['navigation_states_SHA256']: raise ValueError('Executed source states changed.')
    with np.load(state_path) as archive: truth = {k: archive[k] for k in archive.files}
    _, audit, state, _ = read_replay(a.replay_run, truth)
    positions = np.concatenate([truth['positions'][:, :1], state['positions']], axis=1)
    lines = select_three_trajectories(positions)
    # Private simulator world. It is NOT read by the estimator or used in the
    # feature observations/likelihood. Unknown display cells remain unknown.
    field = a.experiment/'scene/query_baseline'
    metadata = json.loads((field/'metadata.json').read_text())
    elevation = np.load(field/'elevation_m.npy', mmap_mode='r')
    valid = np.load(field/'bathymetry_valid_mask.npy', mmap_mode='r')
    xx, yy, zz = terrain_mesh(elevation, valid, metadata['resolution_m'], a.mesh_step)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.font_manager import FontProperties
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists(): plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams['path.simplify'] = False
    figure = plt.figure(figsize=(15, 7.2), constrained_layout=True)
    ax = figure.add_subplot(111, projection='3d', computed_zorder=False)
    cmap = LinearSegmentedColormap.from_list('pale_measured_relief', ['#e8edf1', '#cbd7de', '#aebfc9'])
    ax.plot_surface(xx, yy, zz, cmap=cmap, alpha=.65, linewidth=0., rcount=len(xx), ccount=xx.shape[1],
                    antialiased=False, rasterized=True, zorder=1)
    styles = [('Truth', '仿真真值（仅评价）', '#20252b', '--', .9),
              ('Local', 'Local 声呐里程计', '#e68418', '-', 1.),
              ('Global', '在线 Global', '#008e63', '-', 1.15)]
    for i, (key, label, color, style, width) in enumerate(styles):
        xyz = lines[key]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linestyle=style,
                linewidth=width, label=label, zorder=10+i)
    z_values = np.r_[zz[np.isfinite(zz)], positions[:, :, 2].ravel()]
    ax.set_box_aspect((float(np.ptp(xx)), float(np.ptp(yy)), float(np.ptp(z_values))*a.vertical_exaggeration), zoom=1.15)
    ax.view_init(elev=48, azim=-67)
    distance = float(truth['distance_m'][-1])
    ax.set(xlabel='名义局部 X / m', ylabel='名义局部 Y / m', zlabel='Z / m',
        title=f'纯侧扫特征导航：{distance/1000:.2f} km 全程地形与三条轨迹\n'
              f'地形仅供回波仿真/展示，未作为导航测深观测；Z显示比例 {a.vertical_exaggeration:g}×')
    ax.legend(loc='upper left', framealpha=.95)
    a.out.mkdir()
    png = a.out/'纯侧扫_16km_三维地形与三条轨迹.png'
    pdf = a.out/'纯侧扫_16km_三维地形与三条轨迹.pdf'
    figure.savefig(png, dpi=220); figure.savefig(pdf, dpi=220); plt.close(figure)
    snapshot = a.out/'source_snapshot'; snapshot.mkdir()
    sources = {}
    for name in ['PlotPureSideScanTerrain.py', 'AnalyzeSideScanDirectionalError.py',
                 'AnalyzeSideScanLoopReplay.py', 'AnalyzePureSideScanShort.py']:
        source = Path(__file__).with_name(name); expected = sha(source)
        shutil.copy2(source, snapshot/name)
        if sha(snapshot/name) != expected: raise RuntimeError('Plot source snapshot changed.')
        sources[name] = expected
    report = dict(status='completed_post_estimation_terrain_trajectory_plot',
        replay_run=str(a.replay_run.resolve()), distance_m=distance, trajectories=list(lines),
        trajectory_samples_each=len(positions), trajectory_coordinates_offset_or_draped=False,
        actual_replay_outputs_SHA256=audit['output_SHA256'],
        source_review_SHA256=sha(a.source_review), source_states_SHA256=sha(state_path),
        terrain_field='query_baseline_private_simulator_world', display_only_terrain_stride=a.mesh_step,
        terrain_files_SHA256={n: sha(field/n) for n in ['elevation_m.npy', 'bathymetry_valid_mask.npy', 'metadata.json']},
        vertical_display_exaggeration=a.vertical_exaggeration, trajectories_drawn_in_foreground=True,
        estimator_read_terrain_heights=False, actual_vessel_geographic_precision_verified=False,
        analysis_source_SHA256=sources, figures_SHA256={p.name: sha(p) for p in [png, pdf]})
    (a.out/'terrain_trajectory_plot_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], samples=report['trajectory_samples_each'], figures=[str(png), str(pdf)]), ensure_ascii=False))


if __name__ == '__main__': main()
