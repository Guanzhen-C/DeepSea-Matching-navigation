"""Whole2020 joint depth/intensity map + long comb, no localization shortcut."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import binary_erosion

from QiandaoCombTrajectory import CombSurveyTrajectory, CombSurveyParameters
from QiandaoFlatIntensity import sample_mask
from QiandaoLocalScene import reject_gps_file
from QiandaoLocalRaycaster import intersect_local_grid
from PlanQiandaoCombSurvey import export_route, sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    a = p.parse_args()
    sys.addaudithook(reject_gps_file)
    base = a.experiment
    out = base/'route_review'
    if out.exists():
        raise FileExistsError('Preserve previous whole-frame plans/preflights.')
    out.mkdir()
    started = time.monotonic()
    scene = json.loads((base/'scene/scene_report.json').read_text())
    if scene['source_records'] != 2020 or scene['GPS_file_opened'] or scene['reference_query_elevation_shared']:
        raise ValueError('All records, no GPS, independently constructed even/odd surfaces required.')
    parameters = json.loads((base/'AUV全幅梳形轨迹.json').read_text())['parameters']
    route = CombSurveyTrajectory(CombSurveyParameters(**parameters))
    export_route(route, out)
    bundle = base/'estimator_bundle'
    # Before any estimator runs, replace ONLY generated map-route metadata;
    # all archived algorithm sources remain byte-identical to the baseline.
    meta_path = bundle/'map/full_map_traversal_route.json'
    manifest_path = bundle/'bundle_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(meta_path.read_text())
    metadata.update(purpose='full2020 comb route; simulator evaluates analytic constant-speed Bezier path',
        control_points_xy_m=[s.control[0].tolist() for s in route.segments]+[route.segments[-1].control[-1].tolist()],
        analytic_route_manifest=str((base/'AUV全幅梳形轨迹.json').resolve()),
        complete_route_distance_m=route.length_m)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    manifest['files']['map/full_map_traversal_route.json'] = dict(bytes=meta_path.stat().st_size, sha256=sha(meta_path))
    manifest['map_source'] = 'ALL2020 Qiandao even-record measured bathymetry + frozen native intensity'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    sys.path.insert(0, str(bundle/'isaacsim'))
    from oceansim.slam import MeasuredBathymetryGrid
    from oceansim.utils.acoustic_sonar import parallel_bathymetric_sidescan_rays
    query = MeasuredBathymetryGrid.from_path(base/'scene/query_baseline')
    reference = MeasuredBathymetryGrid.from_path(base/'scene/reference_baseline')
    query.ray_support_mask = binary_erosion(query.backscatter_valid_mask, iterations=2)
    d = np.r_[np.arange(0, route.length_m, .5), route.length_m]
    xy, _, _, segment = route.evaluate_distance(d)
    centre = sample_mask(query.ray_support_mask, xy[:, 0], xy[:, 1], .3, all_four=True)
    bottom = query.seabed_height(xy[:, 0], xy[:, 1])
    clearance = route.parameters.vehicle_z_m-bottom
    known = np.isfinite(clearance)
    if known.any() and np.min(clearance[known]) <= 0:
        raise ValueError('Programmed vehicle intersects measured supported bottom.')
    # Representative actual measured-geometry ray audit: every20m on lines,
    # every2m on turns, extreme synchronous row centres,64cross-track rays.
    sparse = []
    starts = np.r_[0, route.ends_m[:-1]]
    for start, segment_object in zip(starts, route.segments):
        step = 20. if 'line' in segment_object.kind else 2.
        sparse.extend((start+np.arange(0, segment_object.length_m, step)).tolist())
    sparse = np.r_[sparse, route.length_m]
    _, _, _, origins, directions, _, _ = parallel_bathymetric_sidescan_rays(
        num_beams=24, swath_width_deg=160., imaging_rays_per_side=32,
        along_track_rows_per_side=8, along_track_sample_spacing_m=.15)
    selected = np.r_[np.arange(0, 32), np.arange(7*32, 8*32),
                     np.arange(8*32, 9*32), np.arange(15*32, 16*32)]
    row_fraction = []
    for i, distance in enumerate(sparse):
        state = route.state(distance/route.speed_mps)
        hit_parts = []
        for offset in (-.525, .525):
            selected_dirs = directions[selected[:32]]
            # Both sides, same extreme physical phase centre.
            selected_dirs = np.r_[selected_dirs, directions[selected[64:96]]]
            hit, _, _, _ = intersect_local_grid(query,
                state.position+state.rotation@np.array([offset, 0., 0.]),
                selected_dirs@state.rotation.T, min_range=0., max_range=250.)
            hit_parts.append(hit)
        row_fraction.append(float(np.concatenate(hit_parts).mean()))
        if (i+1) % 200 == 0:
            print(json.dumps(dict(stage='actual_joint_map_ray_preflight', checked=i+1, total=len(sparse))), flush=True)
    np.savez_compressed(out/'全程覆盖与实际地形射线预检.npz', centre_distances_m=d,
        centre_valid=centre, measured_clearance_m=clearance, segment_index=segment,
        representative_ray_distances_m=sparse, representative_hit_fraction=np.asarray(row_fraction))
    common = query.backscatter_valid_mask & reference.backscatter_valid_mask
    delta = query.elevation_m[common]-reference.elevation_m[common]
    support = dict(centre_test_spacing_m=.5, centre_test_points=len(d),
        centre_supported_fraction=float(centre.mean()), nadir_or_missing_centres_not_filled=bool(not centre.all()),
        minimum_known_centre_clearance_m=float(np.min(clearance[known])),
        representative_ray_poses=len(sparse), rays_per_representative_pose=128,
        representative_hit_fraction_min=float(np.min(row_fraction)),
        representative_hit_fraction_median=float(np.median(row_fraction)),
        note='Representative geometric preflight only; actual runtime emits all8x256 rays per side every ping.',
        reference_query_height_abs_difference_P50_P95_m=np.percentile(abs(delta), [50, 95]).tolist(),
        height_differences_are_not_absolute_field_accuracy=True)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.colors import ListedColormap
    from caris_orangecopper import caris_orangecopper
    for f in font_manager.fontManager.ttflist:
        if 'Noto Sans CJK' in f.name:
            plt.rcParams['font.family'] = f.name
            break
    plt.rcParams['axes.unicode_minus'] = False
    full = base/'scene/full_baseline_display_only'
    intensity = np.load(full/'backscatter.npy', mmap_mode='r')
    valid = np.load(full/'backscatter_valid_mask.npy', mmap_mode='r')
    elevation = np.load(full/'elevation_m.npy', mmap_mode='r')
    ny, nx = intensity.shape
    extent = [(ny-1)*.15, -(ny-1)*.15, -(nx-1)*.15, (nx-1)*.15]
    dist = np.linspace(0, route.length_m, 40000)
    positions = route.evaluate_distance(dist)[0]
    fig, ax = plt.subplots(figsize=(10, 19), constrained_layout=True)
    ax.imshow(np.where(valid, intensity, np.nan).T[:, ::-1], origin='lower', extent=extent,
              cmap=ListedColormap(caris_orangecopper(256)), vmin=0, vmax=1,
              aspect='equal', interpolation='nearest')
    ax.plot(positions[:, 1], positions[:, 0], color='white', linewidth=3.)
    ax.plot(positions[:, 1], positions[:, 0], color='#00bfff', linewidth=1.6,
            label='单圈梳形规划（不是定位结果）')
    for k, y in enumerate(route.lane_y_m):
        direction = 1 if k % 2 == 0 else -1
        ax.annotate('', xy=(y, direction*80), xytext=(y, -direction*80),
                    arrowprops=dict(arrowstyle='-|>', color='white', linewidth=1.5))
        ax.text(y-3, -600, str(k+1), color='white', fontsize=8)
    initial = route.position(0.)
    ax.scatter(initial[1], initial[0], color='#55ff55', edgecolor='black', s=90,
               label='起终点', zorder=10)
    ax.set(xlabel='名义横航向Y / m（左舷在左）', ylabel='名义沿航向X / m',
        title=f'整幅去噪数据＋测深几何上的闭合梳形航线\n全部2020物理ping；12条×1300m，间距28m\n'
              f'总长{route.length_m/1000:.3f}km；虚拟航时{route.duration_s/60:.1f}min\n'
              '无GPS；源航速／姿态为仿真假设，不是标定地理海图')
    ax.legend(loc='upper left', fontsize=9)
    fig.savefig(out/'全2020ping_整幅去噪图_梳形航线.png', dpi=170)
    fig.savefig(out/'全2020ping_整幅去噪图_梳形航线.pdf')
    plt.close(fig)
    rows, columns = np.arange(0, ny, 9), np.arange(0, nx, 12)
    xx, yy = np.meshgrid((columns-(nx-1)/2)*.3, (rows-(ny-1)/2)*.3)
    zz = elevation[np.ix_(rows, columns)].copy()
    zz[~valid[np.ix_(rows, columns)]] = np.nan
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.plot_surface(xx, yy, zz, cmap='terrain', alpha=.65, linewidth=0,
                    rcount=len(rows), ccount=len(columns), zorder=1)
    ax.plot(positions[:, 0], positions[:, 1], np.full(len(positions), route.parameters.vehicle_z_m),
            color='#0072bd', linewidth=1.5, zorder=10)
    ax.view_init(elev=45, azim=-65)
    ax.set(xlabel='名义X / m', ylabel='名义Y / m', zlabel='Z / m',
           title='全部2020ping实测测深重建＋规划航线（源姿态／航速未标定，无GPS）')
    fig.savefig(out/'全2020ping_测深地形与梳形规划_三维.png', dpi=150)
    plt.close(fig)
    report = dict(status='completed_full2020_joint_comb_route_and_actual_map_preflight',
        source_records=2020, full_data_parts=['Data01', 'Data02'], route_length_m=route.length_m,
        support=support, both_measured_bathymetry_and_denoised_native_intensity_used=True,
        scene_report_SHA256=sha(base/'scene/scene_report.json'), bundle_manifest_SHA256=sha(manifest_path),
        GPS_used=False, nominal_source_pose_not_field_truth=True,
        full_display_field_NOT_used_as_estimator_reference=True,
        native_2020pings_are_actual_original_route_NOT_real_comb_observations=True,
        new_route_navigation_still_requires_full_estimator_execution=True,
        source_code_SHA256={n: sha(Path(__file__).parent/n) for n in
            ('PlanQiandaoFullJointComb.py', 'QiandaoCombTrajectory.py', 'QiandaoLocalRaycaster.py')},
        elapsed_s=time.monotonic()-started)
    write_json(out/'全幅联合梳形规划核验.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
