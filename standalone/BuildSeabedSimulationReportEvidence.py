"""Export audited, report-sized results without copying private raw sonar.

This is post-processing only. Every online error is recomputed from retained
positions, all full-route samples are exported, and rounded/binned views are
separate artifacts. Original experiments and estimator code are never edited.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HIL = Path('demo/hil/jetson_20261021/full_missions_lake_original_20261021_v2')
QI = Path('output/real_sonar/qiandao_fullframe_joint_comb_20261106_v2')
DENOISE = Path('output/real_sonar/matched_full_measured_172_20261105_v1/full_frozen_inference')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): result.update(chunk)
    return result.hexdigest()


def load(path): return json.loads(Path(path).read_text())


def summarize(error):
    error = np.asarray(error, dtype=np.float64)
    if error.ndim != 1 or not len(error) or not np.isfinite(error).all():
        raise ValueError('Nonempty finite complete errors are required.')
    return dict(rmse_m=float(np.sqrt(np.mean(error**2))), median_m=float(np.median(error)),
        p95_m=float(np.percentile(error, 95)), p99_m=float(np.percentile(error, 99)),
        max_m=float(error.max()), final_m=float(error[-1]))


def write_csv(path, names, values):
    if path.exists(): raise FileExistsError(path)
    compressed = path.suffix == '.gz'
    opener = gzip.open if compressed else Path.open
    arguments = dict(mode='wt' if compressed else 'w', newline='', encoding='utf-8')
    with opener(path, **arguments) as stream:
        writer = csv.writer(stream); writer.writerow(names)
        writer.writerows(values)


def segmented(distance, errors, width=100.):
    distance = np.asarray(distance)
    if (distance.ndim != 1 or len(distance) == 0 or not np.isfinite(distance).all()
            or np.any(np.diff(distance) < 0) or width <= 0): raise ValueError('Ordered full-route distances required.')
    bins = np.floor(distance/width).astype(int); result = []
    for index in np.unique(bins):
        mask = bins == index
        for name, value in errors.items():
            if len(value) != len(distance): raise ValueError('Mismatched error coverage.')
            stats = summarize(np.asarray(value)[mask])
            result.append([float(index*width), float((index+1)*width), int(mask.sum()), name,
                *[stats[k] for k in ('rmse_m', 'median_m', 'p95_m', 'max_m', 'final_m')]])
    return result


def assert_metrics(error, expected):
    stats = summarize(error)
    for name, value in expected.items():
        if name in stats: np.testing.assert_allclose(stats[name], value, rtol=0, atol=1e-8)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'docs/reports/seabed_matching_simulation_20261107')
    args = p.parse_args(); out = args.out.resolve()
    for name in ('data', 'figures', 'evidence'):
        if (out/name).exists(): raise FileExistsError('Preserve earlier report evidence: '+str(out/name))
    for name in ('data', 'figures', 'evidence'): (out/name).mkdir(parents=True)
    sources = {}; products = {}; summary = {}; figures = {}

    def record_source(path):
        path = ROOT/path if not path.is_absolute() else path
        sources[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, SHA256=sha(path))
        return path

    def copy(source, relative):
        source = record_source(Path(source)); target = out/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != sha(source): raise ValueError('Copy changed: '+str(relative))
        if str(relative).startswith('figures/'): figures[str(relative)] = str(source.relative_to(ROOT))

    scope_path = HIL/'sea_full_route_failed_resource_review_20261023/cross_scene_hil_scope_review_20261023.json'
    scope = load(record_source(scope_path))
    for profile, directory, frames in [('lake', 'lake_full_route_live_review', 157056),
                                      ('sea', 'sea_full_route_failed_resource_review_20261023', 243569)]:
        folder = HIL/directory
        path = record_source(folder/'figures/aligned_online_trajectories.csv')
        with path.open() as stream:
            reader = csv.reader(stream); names = next(reader)
            values = np.array([[float(x) for x in row] for row in reader])
        if len(values) != frames or not np.isfinite(values).all(): raise ValueError('Incomplete HIL data.')
        columns = {name: values[:, i] for i, name in enumerate(names)}
        time = columns['timestamp_s']; distance = columns['distance_m']
        if np.any(np.diff(time) <= 0): raise ValueError('HIL timestamps not unique/ordered.')
        truth = np.column_stack([columns['truth_'+c+'_m'] for c in 'xyz'])
        error = {}; stats = {}
        for name in ('dr', 'local', 'global'):
            position = np.column_stack([columns[name+'_'+c+'_m'] for c in 'xyz'])
            calculated = np.linalg.norm(position-truth, axis=1)
            np.testing.assert_allclose(calculated, columns[name+'_error_m'], rtol=0, atol=1e-8)
            error['board_'+name] = calculated
            error['pc_'+name] = columns['native_'+name+'_error_m']
            stats['board_'+name] = assert_metrics(calculated, scope[profile]['online_board'][name])
            stats['pc_'+name] = assert_metrics(error['pc_'+name], scope[profile]['original_PC'][name])
        all_names = ['timestamp_s', 'distance_m', *[n+'_error_m' for n in error]]
        write_csv(out/f'data/{profile}_hil_errors_full.csv', all_names,
                  np.column_stack([time, distance, *error.values()]))
        write_csv(out/f'data/{profile}_hil_trajectories_full.csv.gz', names, values)
        write_csv(out/f'data/{profile}_hil_segments_100m.csv',
            ['start_m', 'end_m', 'samples', 'channel', 'rmse_m', 'median_m', 'p95_m', 'max_m', 'final_m'],
            segmented(distance, error))
        summary[profile+'_hil'] = dict(samples=frames, distance_m=float(distance[-1]), metrics=stats,
            original_position_errors_recomputed=True, all_error_rows_retained=True,
            interval_statistics_are_not_independent_experiments=True)
        for filename in ('common_truth_metrics.json', 'relative_motion_audit.json'):
            copy(folder/filename, f'evidence/{profile}_hil/{filename}')
        copy(folder/('completion_evidence.json' if profile=='lake' else 'failed_run_integrity_evidence.json'),
             f'evidence/{profile}_hil/completion_and_integrity.json')
        for filename in ('board_position_errors.png', 'board_3d_trajectories.png'):
            copy(folder/'figures'/filename, f'figures/{profile}_hil_{filename}')
        print(json.dumps(dict(stage='exported_complete_HIL', profile=profile, samples=frames)), flush=True)
    copy(scope_path, 'evidence/hil_cross_scene_scope.json')
    copy(HIL/'sea_full_route_failed_resource_review_20261023/completed_online_motion_timing_and_risk_audit.json',
         'evidence/sea_hil/timing_and_risk.json')
    copy(Path('demo/hil/jetson_20261021/final_log_memory_review_20261023/final_log_memory_comparison.json'),
         'evidence/hil_log_memory_followup.json')

    matrix = ROOT/QI/'validation_projected_full_comb_seed31_v3'
    from RunValidatedSideScanNavigation import validate_matrix
    verified = validate_matrix(matrix)  # Actual complete inputs/source/graph/metric audit, not just a label.
    source = verified['source']
    state = np.load(record_source(source/'navigation_states.npz'))
    truth = state['positions'][:, 0]; time = state['timestamps']; distance = state['distance_m']
    errors = {'dr': np.linalg.norm(state['positions'][:, 1]-truth, axis=1),
              'local': np.linalg.norm(state['positions'][:, 2]-truth, axis=1)}
    variants = ['legacy_no_loops', 'legacy_loops_control', 'projected_no_loops', 'projected_loops']
    for variant in variants:
        path = matrix/variant/'estimated_states.npz'; values = np.load(record_source(path))
        np.testing.assert_array_equal(values['timestamps'], time)
        errors[variant] = np.linalg.norm(values['positions'][:, 2]-truth, axis=1)
        expected = load(matrix/variant/'run_summary.json')['metrics']['Global']
        assert_metrics(errors[variant], {k.replace('RMSE', 'rmse').replace('P95','p95'):v for k,v in expected.items()})
        copy(matrix/variant/'run_summary.json', f'evidence/qiandao/{variant}_run_summary.json')
    newest = np.load(matrix/'projected_loops/estimated_states.npz')
    xyz = np.concatenate([truth[:, None], newest['positions']], axis=1).reshape(len(time), -1)
    write_csv(out/'data/qiandao_side_scan_trajectories_full.csv.gz',
              ['timestamp_s', 'distance_m', *[f'{n}_{c}_m' for n in ('truth','dr','local','global') for c in 'xyz']],
              np.column_stack([time, distance, xyz]))
    write_csv(out/'data/qiandao_side_scan_errors_full.csv',
              ['timestamp_s', 'distance_m', *[n+'_error_m' for n in errors]],
              np.column_stack([time, distance, *errors.values()]))
    write_csv(out/'data/qiandao_side_scan_segments_100m.csv',
              ['start_m', 'end_m', 'samples', 'channel', 'rmse_m', 'median_m', 'p95_m', 'max_m', 'final_m'],
              segmented(distance, errors))
    summary['qiandao_side_scan'] = dict(samples=len(time), distance_m=float(distance[-1]),
        metrics={name:summarize(value) for name,value in errors.items()}, original_errors_recomputed=True,
        all_error_rows_retained=True, full_matrix_actual_audit_passed=True,
        local_and_DR_identical_in_all_four_variants=True,
        same_voyage_nominal_geocoding_not_independent_field_truth=True)
    copy(matrix/'validation_status.json', 'evidence/qiandao/full_matrix_validation.json')
    for name in ('projection_with_loops','projection_without_loops','loops_with_projection','loops_without_projection'):
        copy(matrix/('review_'+name)/'independent_comparison.json', f'evidence/qiandao/{name}.json')
    copy(matrix/'projected_loops/graph_factors.json', 'evidence/qiandao/actual_graph_factors.json')
    copy(matrix/'projected_loops/historical_loop_trace.json', 'evidence/qiandao/actual_loop_trace.json')
    copy(QI/'scene/scene_report.json', 'evidence/qiandao/measured_scene_report.json')
    copy(QI/'AUV全幅梳形轨迹.json', 'evidence/qiandao/route.json')
    copy(QI/'pure_sss_iq_loftr_full_comb_shared_loop_v2/recorded_inputs/input_manifest.json',
         'evidence/qiandao/recorded_input_manifest.json')
    copy(matrix.relative_to(ROOT)/'review_projection_with_loops/纯侧扫_转弯来源投影_同输入对照.png',
         'figures/qiandao_projection_full_route.png')
    copy(matrix.relative_to(ROOT)/'review_loops_with_projection/纯侧扫_历史子图回环_同输入对照.png',
         'figures/qiandao_loops_full_route.png')
    copy(QI/'review_projected_full_terrain_v2/纯侧扫_16km_三维地形与三条轨迹.png',
         'figures/qiandao_terrain_trajectories.png')
    copy(QI/'route_review/全2020ping_整幅去噪图_梳形航线.png', 'figures/qiandao_full_field_route.png')
    copy(QI/'review_current_submaps_20261107_v1/当前侧扫子图_直航与转弯.png', 'figures/qiandao_received_submaps.png')
    copy(QI/'review_current_submaps_20261107_v1/submap_display_review.json', 'evidence/qiandao/submap_display_review.json')
    for name in ('release_manifest.json', 'protocol_supplement.json'):
        copy(Path('releases/sonar_denoise_20261106')/name, f'evidence/denoising/{name}')
    for name in ('run_summary.json', 'render_summary.json', 'full_measured_quality_review.json'):
        copy(DENOISE/name, f'evidence/denoising/{name}')
    copy(DENOISE/'完整航段_原始_冻结旧版_新模型_琥珀色预览_未验收.png', 'figures/denoising_full_voyage_comparison.png')
    copy(DENOISE/'fusion_wide_front_完整航段处理结果_仅屏幕预览.png', 'figures/denoising_selected_portrait_preview.png')

    def dump(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    dump('data/verified_metrics.json', summary)
    dump('evidence/figure_sources.json', figures)
    for path in sorted(out.rglob('*')):
        if path.is_file(): products[str(path.relative_to(out))] = dict(bytes=path.stat().st_size, SHA256=sha(path))
    if any(item['bytes'] >= 100*1024**2 for item in products.values()):
        raise ValueError('An exported artifact exceeds the Git hosting single-file limit.')
    result = dict(schema='oceansim.simulation-report-evidence/v1', source_root_is_local_only=True,
        source_files=sources, exported_files=products, complete_hil_and_qiandao_error_samples_exported=True,
        trajectory_csv_gzip_is_lossless_not_downsampled=True, raw_private_sonar_archive_uploaded=False,
        one_hundred_percent_valid_pixel_coverage_not_proven=True, estimator_code_or_data_changed=False,
        source_script_SHA256=sha(Path(__file__)), status='completed_full_scope_report_export')
    dump('evidence/export_manifest.json', result)
    print(json.dumps(dict(status=result['status'], output=str(out), exported_files=len(products),
        exported_bytes=sum(x['bytes'] for x in products.values())), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
