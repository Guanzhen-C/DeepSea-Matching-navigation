#!/usr/bin/env python3
"""Measured lakebed and verified online trajectories, without position offsets.

Read rasters with mmap; construct only a cropped plotting LOD. Reject every
coarse quad whose full native footprint contains NoData, rather than bridging
survey gaps with artificial terrain. This does not alter simulation inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_full_route_slam_results import load_verified_run


def measured_surface(elevation, valid, resolution_m, bounds, *, stride=8):
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError('stride must be a positive integer')
    rows, columns = elevation.shape
    if valid.shape != elevation.shape:
        raise ValueError('mask and elevation shapes differ')
    xmin, xmax, ymin, ymax = bounds
    c0 = max(0, int(np.floor(xmin/resolution_m + (columns-1)/2)))
    c1 = min(columns-1, int(np.ceil(xmax/resolution_m + (columns-1)/2)))
    r0 = max(0, int(np.floor(ymin/resolution_m + (rows-1)/2)))
    r1 = min(rows-1, int(np.ceil(ymax/resolution_m + (rows-1)/2)))
    if c1 <= c0 or r1 <= r0:
        raise ValueError('plot bounds do not overlap the measured raster')
    ri = np.unique(np.r_[np.arange(r0, r1+1, stride), r1])
    ci = np.unique(np.r_[np.arange(c0, c1+1, stride), c1])
    cells_valid = np.empty((len(ri)-1, len(ci)-1), dtype=bool)
    # Bounded row-band reduction: no full 68M-cell integral-image allocation.
    for i, (start, stop) in enumerate(zip(ri[:-1], ri[1:])):
        bad_columns = np.any(~valid[start:stop+1, c0:c1+1], axis=0)
        prefix = np.r_[0, np.cumsum(bad_columns, dtype=np.int64)]
        cells_valid[i] = (prefix[ci[1:]-c0+1] - prefix[ci[:-1]-c0]) == 0
    top_left = (np.arange(len(ri)-1)[:, None]*len(ci)
                + np.arange(len(ci)-1)[None, :])[cells_valid]
    if not len(top_left):
        raise ValueError('no fully measured plotting cells in this crop')
    triangles = np.concatenate((np.column_stack((top_left, top_left+1, top_left+len(ci)+1)),
        np.column_stack((top_left, top_left+len(ci)+1, top_left+len(ci)))))
    used, inverse = np.unique(triangles, return_inverse=True)
    sr, sc = ri[used//len(ci)], ci[used % len(ci)]
    points = np.column_stack(((sc-(columns-1)/2)*resolution_m,
                             (sr-(rows-1)/2)*resolution_m, elevation[sr, sc]))
    if not np.isfinite(points).all():
        raise ValueError('nonfinite elevation in a supposedly measured cell')
    return points, inverse.reshape(-1, 3)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--terrain', type=Path, default=root/'data/underwater_terrain_datasets/lake_chelan_usgs_2023_3m/prepared_db_power')
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--desktop', type=Path)
    args = parser.parse_args()
    metrics, _, poses = load_verified_run(args.run_directory)
    metadata = json.loads((args.terrain/'metadata.json').read_text())
    resolution = float(metadata['resolution_m'])
    elevation = np.load(args.terrain/'elevation_m.npy', mmap_mode='r')
    valid = np.load(args.terrain/'bathymetry_valid_mask.npy', mmap_mode='r')
    paths = [('真值轨迹', poses[:, 2:5], '#25313b', 2.4, (0, (6, 4))),
             ('Local 局部里程计', poses[:, 8:11], '#ed761b', 1.7, '-'),
             ('Global 在线定位', poses[:, 11:14], '#007cc4', 1.25, '-')]
    all_xy = np.concatenate([p[:, :2] for _, p, *_ in paths])
    low, high = all_xy.min(axis=0)-1000., all_xy.max(axis=0)+1000.
    bounds = (low[0], high[0], low[1], high[1])
    points, triangles = measured_surface(elevation, valid, resolution, bounds)
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(), 'font.size':11,
                         'axes.unicode_minus':False, 'pdf.fonttype':3,
                         'path.simplify':False})
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    fig.subplots_adjust(left=.01, right=.96, bottom=.09, top=.88)
    ax.plot_trisurf(points[:, 0]/1000., points[:, 1]/1000., points[:, 2], triangles=triangles,
                    color='#bccbd0', alpha=.67, linewidth=0, antialiased=False,
                    shade=True, zorder=1, rasterized=True)
    for order, (name, trajectory, color, width, style) in enumerate(paths, start=3):
        ax.plot(trajectory[:, 0]/1000., trajectory[:, 1]/1000., trajectory[:, 2],
                color=color, lw=width, ls=style, label=name, zorder=order, rasterized=True)
    ax.set_xlim(low[0]/1000., high[0]/1000.)
    ax.set_ylim(low[1]/1000., high[1]/1000.)
    zmin = min(float(points[:, 2].min()), *(float(p[:, 2].min()) for _, p, *_ in paths))
    zmax = max(float(points[:, 2].max()), *(float(p[:, 2].max()) for _, p, *_ in paths))
    ax.set_zlim(np.floor(zmin/50.)*50., np.ceil(zmax/50.)*50.)
    ax.set_box_aspect(((high-low)[0]/1000., (high-low)[1]/1000.,
                       (ax.get_zlim()[1]-ax.get_zlim()[0])/1000.*12.))
    ax.view_init(elev=32, azim=-55)
    ax.set_xlabel('局部东向 x / km', labelpad=14)
    ax.set_ylabel('局部北向 y / km', labelpad=14)
    ax.set_zlabel('局部高程 z / m', labelpad=10)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(5))
        axis.set_pane_color((1., 1., 1., 0.))
        axis._axinfo['grid']['color'] = '#e0e6eb'
    legend = ax.legend(loc='upper left', bbox_to_anchor=(.0, 1.045), ncol=3,
                       fontsize=12, frameon=False, handlelength=3.)
    for line in legend.get_lines():
        line.set_linewidth(2.5)
    fig.suptitle(f'湖底三维地形与定位轨迹 · Chelan {poses[-1, 1]/1000.:.2f} km',
                 fontsize=21, fontweight='bold', y=.975)
    online = metrics['position_error_statistics']['online_global_slam']
    fig.text(.5, .925, f'在线 Global RMSE {online["rmse_3d_m"]:.2f} m · P95 {online["p95_3d_m"]:.2f} m · '
             f'最终 {online["final_3d_m"]:.2f} m', ha='center', fontsize=12, color='#52616c')
    fig.text(.07, .045, '实测湖底输入栅格 6 m，绘图地形 LOD 48 m；NoData 留空。采用仿真局部高程基准，垂向显示夸张 12×。',
             fontsize=10, color='#53626f')
    fig.text(.07, .02, '三条轨迹均使用原始坐标，不人为平移或抬高；Global 与真值接近时会自然重合。真实地形上的传感器仿真，非实船湖试。',
             fontsize=10, color='#53626f')
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.png', '.pdf'):
        fig.savefig(args.output_prefix.with_suffix(extension), dpi=180, facecolor='white')
    plt.close(fig)
    audit = dict(run_directory=str(args.run_directory.resolve()),
        terrain_directory=str(args.terrain.resolve()), samples_per_trajectory=len(poses),
        trajectories=['truth', 'local', 'online_global'], source_commit=json.loads(
            (args.run_directory/'run_manifest.json').read_text())['source_commit'],
        input_resolution_m=resolution, drawing_lod_m=resolution*8,
        terrain_vertices=len(points), terrain_triangles=len(triangles),
        crop_bounds_xy_m=list(map(float, bounds)), vertical_display_exaggeration=12,
        trajectory_coordinates_offset=False, interpolation_across_nodata=False,
        all_trajectory_samples_plotted=True, posthoc_optimized_poses_used=False)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    if args.desktop:
        if args.desktop.exists():
            raise FileExistsError('desktop target exists; choose a new name')
        shutil.copy2(args.output_prefix.with_suffix('.png'), args.desktop)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
