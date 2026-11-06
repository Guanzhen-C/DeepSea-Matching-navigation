"""Audit the complete scan/history/IQ-denoise/submap pipeline, not nav success.

Build new physical pings from the actual query appearance/geometry. Recorded
DR poses align only past/current observations; no reference-map crop creates
the query. Save all native complex inputs/outputs for independent inspection.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'standalone')]
from RunQiandaoNoGPSNavigation import dump, sha
from QiandaoCombTrajectory import CombSurveyTrajectory, CombSurveyParameters
from QiandaoLocalScene import reject_gps_file
from SideScanComplexEcho import ComplexEchoParameters, PointScattererIQSimulator
from SideScanIQHistory import FrozenComplexWindowDenoiser, packet_from_compressed_IQ
from SideScanFeatureMap import SideScanFeatureWindow, IntensityFeatureMap, match_feature_prior


def render_review(out, queries, native_envelopes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.font_manager import FontProperties
    from caris_orangecopper import caris_orangecopper
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists(): plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    cmap = ListedColormap(caris_orangecopper(256))
    amplitude = np.concatenate([q['assembled_amplitude'][q['support']] for q in queries.values()])
    hi = float(np.percentile(amplitude, 99.5)); norm = Normalize(vmin=0, vmax=hi)
    fig, axes = plt.subplots(1, len(queries), figsize=(6*len(queries), 8), constrained_layout=True)
    for ax, (name, q) in zip(axes, queries.items()):
        im = q['assembled_amplitude']
        ax.imshow(im, cmap=cmap, norm=norm, origin='lower', aspect='auto', interpolation='nearest')
        ax.set_title(name); ax.set_xlabel('左舷斜距 ← 中央 → 右舷斜距（统一色限）')
        ax.set_ylabel('历史与当前 ping，按 DR 短时位姿累计')
    fig.suptitle('真实构景上的复数回波仿真：脉冲压缩 → 历史扫描拼接 → 特征子图\n默认不二次去噪；仅处理链验证，不是定位精度验收')
    fig.savefig(out/'复数回波_历史累积_无二次去噪_子图对照.png', dpi=150); plt.close(fig)
    # Portrait waterfall preview keeps identical dimensions and one shared
    # mapping across conditions. Full native numeric arrays remain separate.
    fig, axes = plt.subplots(1, len(native_envelopes), figsize=(5*len(native_envelopes), 10), constrained_layout=True)
    values = [np.concatenate([a[:, 0, 0, ::-1], a[:, 1, 0]], axis=1) for a in native_envelopes.values()]
    vmax = float(np.percentile(np.concatenate([a.ravel() for a in values]), 99.5))
    for ax, name, a in zip(axes, native_envelopes, values):
        ax.imshow(a, cmap=cmap, vmin=0, vmax=vmax, aspect='auto', origin='lower')
        ax.set_title(name+'（第 1 同步波束）'); ax.set_xlabel('两侧原生距离样本'); ax.set_ylabel('物理 ping')
    fig.savefig(out/'原生距离瀑布_同尺度对照.png', dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--pings', type=int, default=150)
    p.add_argument('--device', default='cuda')
    p.add_argument('--evaluate-frozen-denoiser', action='store_true',
        help='Historical negative-control only; OFF by default per user instruction.')
    a = p.parse_args()
    if a.out.exists() or not 8 <= a.pings <= 455:
        raise ValueError('New output and an actual available DR prefix required.')
    sys.addaudithook(reject_gps_file)
    sys.path.insert(0, str(a.experiment/'estimator_bundle/isaacsim'))
    from oceansim.slam import MeasuredBathymetryGrid, sanitize_for_strict_json
    import torch
    torch.set_num_threads(1)
    a.out.mkdir()
    sources = ['RunSideScanComplexPipelinePilot.py', 'SideScanComplexEcho.py', 'SideScanIQHistory.py',
        'SideScanFeatureMap.py', 'SideScanLearnedFeatures.py', 'PureSideScanFeatures.py']
    snapshot = a.out/'source_snapshot'; snapshot.mkdir()
    for name in sources: shutil.copy2(Path(__file__).with_name(name), snapshot/name)
    report = dict(status='running_complex_side_scan_pipeline_pilot', navigation_trial=False,
        actual_real_comb_field_validation=False, source_SHA256={n: sha(snapshot/n) for n in sources},
        world='query_baseline: denoised measured appearance and private simulator geometry',
        submap_source='received_current_and_historical_complex_observations_NOT_map_crop',
        alignment_poses='recorded_DR_NOT_truth', GPS_used=False, requested_physical_pings=a.pings)
    dump(a.out/'run_summary.json', report)
    started = time.monotonic()
    try:
        world = MeasuredBathymetryGrid.from_path(a.experiment/'scene/query_baseline')
        params = ComplexEchoParameters(thermal_noise_rms=.004, interference_rms=.008)
        simulator = PointScattererIQSimulator(world, params)
        spec = json.loads((a.experiment/'AUV全幅梳形轨迹.json').read_text())['parameters']
        route = CombSurveyTrajectory(CombSurveyParameters(**spec))
        rows = []
        with (a.experiment/'run_full_seed31/trajectory.csv').open() as f:
            for row in csv.DictReader(f):
                rows.append(row)
                if len(rows) == a.pings: break
        if len(rows) != a.pings: raise ValueError('Requested DR records not yet available.')
        observed = np.empty((a.pings, 2, 8, params.range_bins), np.complex64)
        clean = np.empty_like(observed)
        times, poses = [], []
        for index, row in enumerate(rows):
            stamp = float(row['time_s']); truth = route.state(stamp)
            echo = simulator.ping(stamp, truth.position, truth.rotation)
            observed[index], clean[index] = echo['observed_IQ'], echo['clean_simulated_IQ']
            times.append(stamp); poses.append([float(row['DR_x']), float(row['DR_y']), float(row['DR_yaw'])])
            if (index+1)%25 == 0: print(json.dumps(dict(stage='actual_IQ_generated', pings=index+1)), flush=True)
        np.save(a.out/'received_native_complex64.npy', observed)
        np.save(a.out/'clean_SIMULATION_CONTROL_ONLY_complex64.npy', clean)
        np.save(a.out/'physical_ping_timestamps_s.npy', times)
        np.save(a.out/'stitching_DR_pose2.npy', poses)
        provenance = dict(secondary_denoising_used=False, user_cancelled_additional_denoiser=True)
        conditions = {'含噪原始回波_无二次去噪': observed}
        if a.evaluate_frozen_denoiser:
            denoiser = FrozenComplexWindowDenoiser(ROOT, a.device)
            restored, provenance = denoiser(observed)
            np.save(a.out/'fusion_wide_front_native_complex64.npy', restored)
            conditions['额外固定网络_仅负面对照'] = restored
            del denoiser
            if a.device.startswith('cuda'): torch.cuda.empty_cache()
        conditions['无附加噪声仿真对照_非实测真值'] = clean
        queries, envelopes, matches = {}, {}, {}
        prior = IntensityFeatureMap.from_path(a.experiment/'scene/reference_baseline')
        from SideScanLearnedFeatures import LoFTRSideScanMatcher
        matcher = LoFTRSideScanMatcher(a.device)
        for name, iq in conditions.items():
            compressed = simulator.pulse_compress(iq)
            envelopes[name] = abs(compressed)
            window = SideScanFeatureWindow(a.pings)
            for index, (stamp, pose) in enumerate(zip(times, poses)):
                packet = packet_from_compressed_IQ(stamp, compressed[index], params)
                window.add(packet, pose, index+1)
            query = window.build(poses[-1]); queries[name] = query
            matched = match_feature_prior(prior, query, poses[-1], image_matcher=matcher)
            matches[name] = {k: v for k, v in matched.items() if not isinstance(v, np.ndarray)}
            np.savez_compressed(a.out/(name+'_actual_query_submap.npz'), **query)
        clean_amp = envelopes['无附加噪声仿真对照_非实测真值']
        metrics = {name: dict(envelope_RMSE_vs_simulation_clean=float(np.sqrt(np.mean((amp-clean_amp)**2))))
                   for name, amp in envelopes.items()}
        render_review(a.out, queries, envelopes)
        for n in sources:
            if sha(Path(__file__).with_name(n)) != report['source_SHA256'][n]:
                raise RuntimeError('Executed pipeline changed during this run.')
        report.update(status='completed_IQ_history_denoising_submap_pilot_NOT_navigation_acceptance',
            parameters=asdict(params), native_IQ_shape=list(observed.shape),
            native_IQ_all_samples_preserved=True, original_measured_phase_recovered=False,
            pose_dependent_fixed_world_scatterers=True, pipeline_neural_denoiser=provenance,
            envelope_metrics=metrics, actual_feature_matches=matches,
            actual_IQ_source_SHA256=sha(a.out/'received_native_complex64.npy'),
            restored_IQ_SHA256=sha(a.out/'fusion_wide_front_native_complex64.npy'),
            elapsed_s=time.monotonic()-started)
        report, _ = sanitize_for_strict_json(report)
        dump(a.out/'run_summary.json', report)
        print(json.dumps(dict(status=report['status'], metrics=metrics, matches=matches), ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(status='failed_preserved_no_automatic_restart', error=f'{type(exc).__name__}: {exc}',
                      elapsed_s=time.monotonic()-started)
        dump(a.out/'run_summary.json', report)
        raise


if __name__ == '__main__': main()
