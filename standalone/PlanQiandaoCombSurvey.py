"""Prepare a measured-intensity-only comb survey; do NOT launch localization.

Reprojects stopped native denoised amplitudes without measured bottom depths,
then verifies a continuous single-loop lawnmower against their actual support.
The flat plane/nominal acquisition motion are explicitly synthetic assumptions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import shutil
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import binary_erosion

from QiandaoLocalScene import bounded_interpolation, physical_partition, reject_gps_file
from QiandaoFlatIntensity import project_native_intensity, sample_mask, flat_fan_support
from QiandaoCombTrajectory import CombSurveyTrajectory


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def measured_depth_quality_audit(experiment):
    source = experiment/'scene/reference_baseline/elevation_m.npy'
    z = np.load(source, mmap_mode='r')
    steps = [np.abs(np.diff(z, axis=i)) for i in (0, 1)]
    return dict(assessment='absolute measurement accuracy unverified, not a measured-error estimate',
        source_metadata_reports_missing_recovered_vessel_attitude=True,
        resolution_m=.3, elevation_quantiles_1_50_99_m=np.nanpercentile(z, [1, 50, 99]).tolist(),
        adjacent_grid_abs_delta_P95_m=[float(np.nanpercentile(d, 95)) for d in steps],
        roughness_alone_does_not_prove_noise_or_accuracy=True,
        independent_depth_accuracy_reference_available=False,
        decision='exclude measured bathymetry from this new scene geometry AND planned navigation constraints',
        original_elevation_SHA256=sha(source), original_elevation_changed=False)


def prepare_flat_intensity(experiment, raw_cache, out, route):
    old = json.loads((experiment/'scene/scene_report.json').read_text())
    stopped = json.loads((experiment/'stopped_input_provenance_audit.json').read_text())
    if not stopped['all_2020_physical_records_and_16_beams'] or stopped['GPS_used']:
        raise ValueError('Previously audited stopped native amplitude/gain fields required.')
    gain_path = experiment/'inputs/fusion_wide_front_supported_log_delta.npy'
    if sha(gain_path) != stopped['baseline_gain_file_SHA256']:
        raise ValueError('The stopped default gain field changed.')
    magnitude_path = raw_cache/'原始MAT逐束包络_float64.npy'
    ledger = json.loads((raw_cache/'run_summary.json').read_text())
    if not ledger['original_all_native_samples_reproduced_exactly']:
        raise ValueError('Verified original native amplitude cache required.')
    for path, stamp in ledger['source_guards'].items():
        s = Path(path).stat()
        if stamp != dict(size=s.st_size, mtime_ns=s.st_mtime_ns):
            raise ValueError('The guarded original source changed.')
    raw = np.load(magnitude_path, mmap_mode='r')
    gain = np.load(gain_path, mmap_mode='r')
    if raw.shape != (2, 8, 2020, 12189) or gain.shape != raw.shape:
        raise ValueError('Exact source beam/record/range layout required.')
    definition = old['definition']
    ids = np.arange(definition['physical_record_start'],
                    definition['physical_record_start']+definition['physical_record_count'])
    ref_ids, query_ids = physical_partition(ids)
    stamps = np.load(experiment/'scene/physical_record_timestamps_s.npy')
    if len(stamps) != len(ids) or not np.all(np.diff(stamps) > 0):
        raise ValueError('Ordered native physical-record time layout required.')
    res = float(definition['map_resolution_m'])
    speed = float(definition['scene_layout_speed_mps_assumed'])
    extent = speed*(stamps[-1]-stamps[0])
    # Exactly the prior selected area, but independently projected without
    # reading measured TargetAngleInfo, TargetDelayInfo or old height pixels.
    shape = (int(np.ceil(490/res))+1, int(np.ceil((extent+6)/res))+1)
    ys = (np.arange(shape[0])-(shape[0]-1)/2)*res
    totals = [np.zeros(shape, np.float64), np.zeros(shape, np.float64)]
    counts = [np.zeros(shape, np.uint32), np.zeros(shape, np.uint32)]
    source_z = -float(definition['source_platform_depth_m_assumed'])
    altitude = source_z-route.parameters.virtual_flat_bottom_z_m
    parameters = old['parameters']
    start = time.monotonic()
    for number, (index, stamp) in enumerate(zip(ids, stamps)):
        partition = int(index % 2)
        x = speed*(stamp-stamps[0])-extent/2+(np.arange(8)-3.5)*parameters['BeamSpacingAlongTrackDisRes']
        ix = np.rint(x/res+(shape[1]-1)/2).astype(int)
        for side in (0, 1):
            rows = np.flatnonzero(ys >= 0) if side == 0 else np.flatnonzero(ys < 0)
            values, valid = project_native_intensity(np.asarray(raw[side, :, index]),
                np.asarray(gain[side, :, index]), ys[rows], source_altitude_m=altitude,
                sound_speed=parameters['c'], sampling_frequency=parameters['BeamSigFs'])
            row = np.broadcast_to(rows[None, :], values.shape)
            col = np.broadcast_to(ix[:, None], values.shape)
            valid &= (col >= 0) & (col < shape[1])
            np.add.at(totals[partition], (row[valid], col[valid]), values[valid])
            np.add.at(counts[partition], (row[valid], col[valid]), 1)
        if number % 16 == 0:
            raw._mmap.madvise(mmap.MADV_DONTNEED)
            gain._mmap.madvise(mmap.MADV_DONTNEED)
        if (number+1) % 80 == 0:
            print(json.dumps(dict(stage='native_flat_projection', records=number+1,
                                 total=len(ids), elapsed_s=time.monotonic()-start)), flush=True)
    fields, masks = [], []
    for total, count in zip(totals, counts):
        field = np.divide(total, count, out=np.full(shape, np.nan), where=count > 0)
        field, mask = bounded_interpolation(field, count, resolution=res,
            max_distance_m=definition['maximum_local_interpolation_distance_m'])
        fields.append(field)
        masks.append(mask)
    common = masks[0] & masks[1]
    flat = np.full(shape, route.parameters.virtual_flat_bottom_z_m, np.float32)
    files = {}
    for k, name in enumerate(('reference_fixed', 'query_fixed')):
        folder = out/name
        folder.mkdir()
        # Missing exterior is not extrapolated, rendered as terrain or used
        # as synthetic side-scan texture in later tests.
        fields[k][~common] = np.nan
        np.save(folder/'backscatter.npy', fields[k])
        np.save(folder/'backscatter_valid_mask.npy', common)
        np.save(folder/'elevation_m.npy', flat)
        np.save(folder/'bathymetry_valid_mask.npy', common)
        np.save(folder/'observed_cell_counts.npy', counts[k])
        metadata = dict(resolution_m=res, coordinate_frame='nominal non-geodetic local simulation frame',
            source_product='frozen fusion_wide_front native amplitude on ASSUMED flat plane',
            backscatter_units='uncalibrated linear envelope amplitude proxy, NOT calibrated reflectivity',
            measured_bathymetry_used=False, elevation_is_virtual_constant_plane=True,
            virtual_flat_bottom_z_m=route.parameters.virtual_flat_bottom_z_m,
            source_altitude_m_ASSUMED=altitude, source_speed_mps_ASSUMED=speed,
            original_acquisition_attitude_ASSUMED='level, straight', GPS_used=False,
            same_voyage_even_odd_partition=True, not_independent_field_ground_truth=True,
            no_view_angle_or_acoustic_shadow_model=True, variant=name)
        write_json(folder/'metadata.json', metadata)
        files[name] = {p.name: sha(p) for p in sorted(folder.iterdir())}
    report = dict(status='completed_flat_native_intensity_reprojection_without_measured_bathymetry',
        reference_physical_records=ref_ids.tolist(), query_physical_records=query_ids.tolist(),
        shared_record_IDs=0, shape=list(shape), resolution_m=res,
        reference_observed_cells=int((counts[0] > 0).sum()), query_observed_cells=int((counts[1] > 0).sum()),
        common_valid_cells=int(common.sum()), common_valid_area_m2=float(common.sum()*res**2),
        maximum_interpolation_distance_m=definition['maximum_local_interpolation_distance_m'],
        source_report_SHA256=sha(experiment/'scene/scene_report.json'),
        stopped_provenance_audit_SHA256=sha(experiment/'stopped_input_provenance_audit.json'),
        frozen_gain_file_SHA256=stopped['baseline_gain_file_SHA256'],
        raw_native_array=dict(path=str(magnitude_path.resolve()), bytes=magnitude_path.stat().st_size,
                              mtime_ns=magnitude_path.stat().st_mtime_ns, prior_exact_cache_verification=True),
        measured_depth_angle_delay_used=False, original_MAT_or_gain_modified=False,
        flat_plane_not_real_bathymetry=True, GPS_used=False, files=files)
    write_json(out/'intensity_scene_report.json', report)
    return fields[0], common, res, report


def audit_route_support(route, common, resolution, frozen_bundle, out):
    sys.path.insert(0, str(frozen_bundle/'isaacsim'))
    from oceansim.utils.acoustic_sonar import parallel_bathymetric_sidescan_rays
    origins, directions, _, _, _, _, _ = parallel_bathymetric_sidescan_rays(
        num_beams=96, swath_width_deg=160., imaging_rays_per_side=256,
        along_track_rows_per_side=8, along_track_sample_spacing_m=.15)
    safe = binary_erosion(common, iterations=2)
    distances = np.r_[np.arange(0, route.length_m, .5), route.length_m]
    xy, tangent, curvature, segment = route.evaluate_distance(distances)
    centre = sample_mask(safe, xy[:, 0], xy[:, 1], resolution, all_four=True)
    fractions, sides = [], []
    port = directions[:, 1] >= 0
    for start in range(0, len(xy), 64):
        valid, _, _ = flat_fan_support(xy[start:start+64], tangent[start:start+64],
            origins, directions, mask=safe, resolution=resolution,
            altitude_m=route.parameters.vehicle_z_m-route.parameters.virtual_flat_bottom_z_m,
            max_range_m=250.)
        fractions.extend(valid.mean(axis=1).tolist())
        sides.extend(np.column_stack([valid[:, port].mean(axis=1), valid[:, ~port].mean(axis=1)]).tolist())
    fractions, sides = np.asarray(fractions), np.asarray(sides)
    np.savez_compressed(out/'全程声呐有效覆盖检查.npz', distance_m=distances,
        xy_m=xy, segment_index=segment, centre_valid=centre,
        nominal_ray_valid_fraction=fractions, port_starboard_valid_fraction=sides)
    if not centre.all() or fractions.min() < .5 or sides.min() < .2:
        raise ValueError('Route leaves measured scalar support or has insufficient nominal side-scan returns.')
    return dict(status='passed_dense_nominal_path_and_simultaneous_sonar_support_check',
        check_spacing_m=.5, checked_poses=len(xy), all_vehicle_centres_inside_safe_mask=True,
        support_boundary_erosion_m=2*resolution, full_bilinear_neighbours_required=True,
        rays_per_pose=int(len(directions)), simultaneous_rows_per_side=8,
        nominal_valid_ray_fraction_min=float(fractions.min()),
        nominal_valid_ray_fraction_median=float(np.median(fractions)),
        port_starboard_min_fraction=sides.min(axis=0).tolist(),
        missing_returns_remain_invalid_not_invented=True,
        first_last_xy_distance_m=float(np.linalg.norm(xy[-1]-xy[0])),
        maximum_yaw_rate_deg_s=float(np.rad2deg(abs(curvature).max()*route.speed_mps)),
        minimum_turn_radius_m=float(1/abs(curvature).max()),
        maximum_horizontal_acceleration_mps2=float(abs(curvature).max()*route.speed_mps**2),
        vehicle_dynamics_are_design_assumptions_not_real_AUV_certification=True,
        nominal_flat_geometry_only_not_localization_performance=True)


def export_route(route, out):
    distance = np.r_[np.arange(0, route.length_m, .25), route.length_m]
    xy, tangent, curvature, segment = route.evaluate_distance(distance)
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    rows = np.column_stack([distance/route.speed_mps, distance, xy,
        np.full(len(xy), route.parameters.vehicle_z_m), yaw, route.speed_mps*tangent,
        np.zeros(len(xy)), route.speed_mps*curvature, segment])
    with (out/'AUV梳形轨迹_全程.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_s', 'distance_m', 'x_m', 'y_m', 'z_m', 'yaw_rad',
                         'vx_mps', 'vy_mps', 'vz_mps', 'yaw_rate_radps', 'segment_index'])
        writer.writerows(rows)
    manifest = route.manifest()
    manifest['trajectory_sample_interval_m'] = .25
    manifest['csv_SHA256'] = sha(out/'AUV梳形轨迹_全程.csv')
    write_json(out/'AUV梳形轨迹.json', manifest)
    return xy, distance


def plot_route(route, field, mask, resolution, out):
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
    ny, nx = mask.shape
    extent = [-(nx-1)*resolution/2, (nx-1)*resolution/2,
              -(ny-1)*resolution/2, (ny-1)*resolution/2]
    shown = np.where(mask, field, np.nan)
    distance = np.linspace(0, route.length_m, 16000)
    xy, tangent, _, _ = route.evaluate_distance(distance)
    fig, ax = plt.subplots(figsize=(12, 13), constrained_layout=True)
    ax.imshow(shown, extent=extent, origin='lower', cmap=ListedColormap(caris_orangecopper(256)),
              vmin=0, vmax=1, interpolation='nearest', aspect='equal')
    # Narrow white casing separates the route from bright sonar texture;
    # neither the underlying map nor the trajectory is shifted or enhanced.
    ax.plot(xy[:, 0], xy[:, 1], color='white', linewidth=3.3, alpha=.95)
    ax.plot(xy[:, 0], xy[:, 1], color='#00bfff', linewidth=1.8, label='规划航线（非定位结果）')
    for k, y in enumerate(route.lane_y_m):
        dx = 18 if k % 2 == 0 else -18
        ax.annotate('', xy=(dx/2, y), xytext=(-dx/2, y),
                    arrowprops=dict(arrowstyle='-|>', color='white', linewidth=1.8))
        ax.text(-117, y+3, f'{k+1:02d}', color='white', fontsize=9,
                bbox=dict(facecolor='black', alpha=.4, edgecolor='none', pad=1))
    initial = route.position(0.)
    ax.scatter(*initial[:2], c='#55ff55', edgecolor='black', s=100, zorder=20, label='起点＝终点')
    ax.text(initial[0]+7, initial[1]-10, '起终点', color='white', fontsize=11,
            bbox=dict(facecolor='black', alpha=.6, edgecolor='none', pad=2))
    ax.set(xlabel='名义局部沿航向 X / m', ylabel='名义局部横航向 Y / m',
        title=f'固定去噪侧扫强度底图上的 AUV 闭合梳形规划\n'
              f'{route.parameters.lanes}条×{route.parameters.line_length_m:g} m，间距{route.parameters.line_spacing_m:g} m；'
              f'总长{route.length_m/1000:.3f} km，虚拟航时{route.duration_s/60:.1f} min\n'
              '平底／源航速为显式仿真假设；未使用实测测深，不是地理海图或定位结果')
    ax.legend(loc='upper left', facecolor='white', framealpha=.95)
    ax.grid(alpha=.15)
    fig.savefig(out/'AUV梳形航线_去噪强度底图.png', dpi=180)
    fig.savefig(out/'AUV梳形航线_去噪强度底图.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    _, _, curvature, _ = route.evaluate_distance(distance)
    ax.plot(distance/1000, np.rad2deg(route.speed_mps*curvature), linewidth=1.)
    ax.set(xlabel='累计规划航程 / km', ylabel='解析航向角速度 / (°/s)',
           title='转弯运动连续性：直线段角速度为零，转弯段平滑过渡，无原地瞬时掉头')
    ax.grid(alpha=.3)
    fig.savefig(out/'全程转弯角速度.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--raw-cache', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    a = parser.parse_args()
    sys.addaudithook(reject_gps_file)
    forbidden_depth_opens = []
    def forbid_measured_depth_inputs(event, arguments):
        if event == 'open' and arguments and isinstance(arguments[0], (str, bytes)):
            value = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
            if Path(value).name.endswith('DepthTotal.mat'):
                forbidden_depth_opens.append(value)
                raise PermissionError('Measured angle/delay/depth files are excluded from the new scene.')
    sys.addaudithook(forbid_measured_depth_inputs)
    if a.out.exists():
        raise FileExistsError('Preserve previous planned routes/scenes and failed attempts.')
    if shutil.disk_usage(a.out.parent).free < 256*1024**2:
        raise RuntimeError('At least256MiB free disk required, no deletion of other projects.')
    a.out.mkdir()
    started = time.monotonic()
    route = CombSurveyTrajectory()
    quality = measured_depth_quality_audit(a.experiment)
    field, mask, resolution, scene_report = prepare_flat_intensity(a.experiment, a.raw_cache, a.out, route)
    support = audit_route_support(route, mask, resolution, a.experiment/'estimator_bundle', a.out)
    export_route(route, a.out)
    plot_route(route, field, mask, resolution, a.out)
    source_names = ('QiandaoCombTrajectory.py', 'QiandaoFlatIntensity.py', 'PlanQiandaoCombSurvey.py')
    snapshot = a.out/'source_snapshot'
    snapshot.mkdir()
    for name in source_names:
        shutil.copy2(Path(__file__).parent/name, snapshot/name)
    report = dict(status='completed_planned_and_supported_3p9km_comb_survey_NOT_a_localization_run',
        measured_depth_quality_assessment=quality, route=route.manifest(), support=support,
        input_scene_report_SHA256=sha(a.out/'intensity_scene_report.json'),
        navigation_policy=dict(status='proposed_intensity_only_config_NOT_yet_executed',
            prior_and_query_intensity='frozen fusion_wide_front, physically disjoint even/odd source records',
            measured_bathymetry_geometry_used=False, measured_bathymetry_factor_enabled=False,
            bathymetry_global_likelihood_enabled=False, side_scan_map_matching_enabled=True,
            synthetic_IMU_DVL_pressure_depth_enabled=True, GPS_enabled=False,
            sensor_range_geometry='virtual flat-plane returns, NOT real measured MBES',
            existing_joint_launcher_must_NOT_be_called_intensity_only_without_adapter=True),
        original_frozen_denoiser_or_lake_sea_algorithms_changed=False,
        no_original_MAT_or_previous_experiment_overwritten=True,
        measured_depth_files_opened=forbidden_depth_opens, GPS_used=False,
        localization_simulation_launched=False, independent_real_world_precision_validated=False,
        larger_distance_is_revisit_in_same_328m_region_not_3p9km_independent_seabed=True,
        aspect_dependence_source_attitude_and_shadow_uncertainty_not_resolved=True,
        source_code_SHA256={name: sha(Path(__file__).parent/name) for name in source_names},
        elapsed_s=time.monotonic()-started)
    write_json(a.out/'规划与数据准备核验.json', report)
    lines = ['AUV闭合梳形轨迹与去噪强度场：已规划，未启动新定位实验', '',
        f'12条测线，每条250m，间距28m，闭合返回；总长{route.length_m:.3f}m。',
        f'航速0.8m/s，虚拟航时{route.duration_s/60:.2f}分钟。不是计算耗时。',
        f'转弯零端点曲率，最小半径{support["minimum_turn_radius_m"]:.2f}m，',
        f'最大角速度{support["maximum_yaw_rate_deg_s"]:.3f}度/s；非真实AUV动力学认证。',
        '使用固定wide去噪的原浮点幅度和增益，未使用RGB图片，也未改动去噪。',
        '未验证测深真实精度；粗糙统计不是测深误差。新强度场不读取实测角度/时延，',
        '以源名义直线4节、源深2.3m、虚拟底面Z=-50m作平底斜距投影。',
        '实测测深不用于新场景几何或拟议的定位约束；深度计仍保留，两者不是一回事。',
        '原实测高度数组只为质量审计读取，未输入新强度重投影或轨迹规划。',
        '这仍是假设几何上的实测纹理仿真，不是真实高精度测绘地图；平底不是重建成果。',
        '往返观测的视角散射、阴影以及源运动/姿态误差未由现有数据解决。',
        '航程增加来自同一约328m长、490m宽区域的梳形覆盖与重访，不冒充新海域长航程。',
        '每0.5m核对路线及4096条同步扇面射线，完整双线性支持和0.6m边缘留量。',
        f'全程最小有效射线比例{support["nominal_valid_ray_fraction_min"]:.3f}；',
        f'左/右侧各自最小{support["port_starboard_min_fraction"]}；无数据回波保持无效。',
        '参考图偶240记录、查询图奇240记录，互斥但非独立航次；仅适合受控实验。',
        '本轮只完成航线、强度场和几何可行性检查，尚无新的定位误差指标。',
        '旧运行器是测深＋强度联合入口，不能原样称为纯侧扫定位；启动前须接入强度专用适配。',
        '对后续定位需额外检验假设高度±5m敏感性、初值偏差、去程/回程、转弯段和回环。',
        '', '查看AUV梳形航线_去噪强度底图.png；轨迹JSON保留解析控制点，CSV为0.25m抽样。']
    (a.out/'查看说明.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], length_m=route.length_m, output=str(a.out),
                         support=support, elapsed_s=report['elapsed_s']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
