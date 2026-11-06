"""Compare received-only representations on the first estimated turning window.

Selection uses estimated heading history, not route labels or errors. The
original truth NPZ, GPS and height rasters are explicitly blocked. This is a
single-window feature/geometry diagnostic, NOT a navigation-accuracy test.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--source-run', required=True, type=Path)
    p.add_argument('--estimator-replay', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError('Preserve earlier diagnostic results.')
    sys.path[:0] = [str(a.experiment/'estimator_bundle/isaacsim'), str(Path(__file__).parent)]
    def guard(event, args):
        if event == 'open' and args and isinstance(args[0], (str, Path)):
            name = Path(args[0]).name
            if 'gps' in name.lower() or name in ('navigation_states.npz', 'elevation.npy', 'navigation_elevation.npy'):
                raise RuntimeError('Truth/GPS/terrain-height access forbidden in feature diagnostic: '+str(args[0]))
    sys.addaudithook(guard)
    from SideScanRecordedInputs import RecordedSideScanInputs
    from SideScanFeatureMap import SideScanFeatureWindow, IntensityFeatureMap, match_feature_prior
    from SideScanProjectedSubmap import build_projected_submap
    from SideScanProjectedMatching import match_projected_prior
    from SideScanLearnedFeatures import LoFTRSideScanMatcher
    from WriteSideScanReport import dump
    import torch
    torch.set_num_threads(1)
    candidates = []
    for path in a.source_run.glob('backend_query_*.npz'):
        q = np.load(path)
        angles = np.unwrap(q['contributing_local_pose2'][:, 2])
        if np.ptp(angles) > np.deg2rad(15.): candidates.append((float(q['timestamp']), path))
    if not candidates: raise ValueError('No observed turning query available.')
    stamp, original_query_path = min(candidates)
    original = np.load(original_query_path)
    report_source = json.loads((a.source_run/'run_summary.json').read_text())
    if not report_source['status'].startswith('completed_'): raise ValueError('Complete observed source run required.')
    replay_report = json.loads((a.estimator_replay/'run_summary.json').read_text())
    if replay_report['status'] != 'completed_lossless_navigation_input_replay':
        raise ValueError('Complete estimator-only pose record required.')
    for name, expected in report_source['prior_intensity_inputs_SHA256'].items():
        if sha(a.experiment/'scene/reference_baseline'/name) != expected: raise ValueError('Prior intensity input changed.')
    inputs = RecordedSideScanInputs(a.source_run/'recorded_inputs')
    estimate = np.load(a.estimator_replay/'estimated_states.npz')
    if estimate['positions'].shape[1:] != (3, 3): raise ValueError('Estimator-only archive expected, without truth positions.')
    np.testing.assert_array_equal(estimate['timestamps'], inputs.timestamps)
    end = int(np.searchsorted(inputs.timestamps, stamp))
    if inputs.timestamps[end] != stamp: raise ValueError('Exact measured source window required.')
    old, canonical = SideScanFeatureWindow(), SideScanFeatureWindow(canonical_orientation=True)
    for i in range(max(0, end-149), end+1):
        packet = inputs.packet(i)
        raw_pose = np.r_[estimate['positions'][i, 1, :2], estimate['yaw_rad'][i, 1]]
        for window in (old, canonical):
            window.add(packet, raw_pose, i+1, vertical_position_m=estimate['positions'][i, 1, 2])
    old_q, canonical_q = old.build(raw_pose), canonical.build(raw_pose)
    np.testing.assert_array_equal(old_q['assembled_amplitude'], original['assembled_amplitude'])
    np.testing.assert_array_equal(old_q['row_normals'], original['row_normals'])
    projected_q = build_projected_submap(old, raw_pose)
    a.out.mkdir(); snapshot = a.out/'source_snapshot'; snapshot.mkdir()
    names = ['DiagnoseProjectedSideScanTurn.py', 'SideScanProjectedSubmap.py', 'SideScanProjectedMatching.py',
        'SideScanLearnedFeatures.py', 'SideScanPlaneRegistration.py', 'SideScanFeatureMap.py',
        'SideScanRecordedInputs.py', 'PureSideScanFeatures.py', 'WriteSideScanReport.py']
    for name in names: shutil.copy2(Path(__file__).parent/name, snapshot/name)
    report = dict(status='running_single_window_received_feature_diagnostic', timestamp_s=stamp,
        window_selected_from_estimated_heading_not_truth=True, original_query_bit_exact_reproduced=True,
        selected_saved_query_SHA256=sha(original_query_path), inputs_manifest_SHA256=sha(inputs.root/'input_manifest.json'),
        estimator_only_pose_archive_SHA256=sha(a.estimator_replay/'estimated_states.npz'),
        source_SHA256={n: sha(snapshot/n) for n in names}, evaluation_truth_or_height_read=False,
        extra_denoising_used=False, navigation_accuracy_validation=False, variants={})
    dump(a.out/'diagnostic.json', report)
    started = time.monotonic()
    try:
        base = LoFTRSideScanMatcher(a.device)
        masked = copy.copy(base); masked.context_masks = True
        masked.provenance = dict(base.provenance, received_support_context_masks=True)
        feature_map = IntensityFeatureMap.from_path(a.experiment/'scene/reference_baseline')
        seed = original['seed_pose2']  # exact estimated pre-match seed, NOT true pose
        for name, query in [('legacy', old_q), ('canonical', canonical_q), ('projected', projected_q)]:
            np.savez_compressed(a.out/f'{name}_received_query.npz', **query)
            for suffix, matcher in [('postmask_only', base), ('context_and_postmask', masked)]:
                when = time.monotonic()
                if name == 'projected': matched = match_projected_prior(feature_map, query, seed, matcher)
                else: matched = match_feature_prior(feature_map, query, seed, image_matcher=matcher)
                variant = name+'_'+suffix
                np.savez_compressed(a.out/f'{variant}_matches.npz', **matched)
                summary = {k: v for k, v in matched.items() if not isinstance(v, np.ndarray)}
                summary.update(elapsed_s=time.monotonic()-when, image_shape=list(query['image'].shape),
                    support_fraction=float(query['support'].mean()), detector_features=len(query['pixels']),
                    physical_pings_included=len(query['physical_ping_observation_ids']),
                    map_feature_matcher=matcher.provenance)
                if 'information' in matched: summary['information_eigenvalues'] = np.linalg.eigvalsh(matched['information']).tolist()
                report['variants'][variant] = summary
                dump(a.out/'diagnostic.json', report)
                print(json.dumps(dict(variant=variant, accepted=matched['accepted'], matches=matched['matches'],
                    inliers=matched.get('inliers'), reason=matched.get('reason'), shape=list(query['image'].shape))), flush=True)
        for name, expected in report['source_SHA256'].items():
            if sha(Path(__file__).parent/name) != expected: raise RuntimeError('Diagnostic source changed during execution.')
        report.update(status='completed_single_window_received_feature_diagnostic', elapsed_s=time.monotonic()-started,
            all_observation_projection_contributions_preserved=projected_q['received_contributions']==projected_q['projected_contributions'])
        dump(a.out/'diagnostic.json', report)
    except BaseException as exc:
        report.update(status='failed_preserved_diagnostic_no_automatic_restart', error=f'{type(exc).__name__}: {exc}')
        dump(a.out/'diagnostic.json', report); raise


if __name__ == '__main__': main()
