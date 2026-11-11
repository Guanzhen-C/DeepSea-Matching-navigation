"""Jetson estimator endpoint. Binary stdin/stdout; diagnostics use stderr.

Only an intensity prior is deployed. No simulator/query map or evaluation
truth file is required or allowed. The original estimator math is unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PureSideScanHILProtocol import read_frame, write_frame, validate_boot, SensorOrder
from standalone.hil.protocol import unpack
from QiandaoLargeModelField import sha_file


def deny_simulator_files(event, arguments):
    if event == 'open' and arguments and isinstance(arguments[0], (str, bytes, Path)):
        path = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0])
        lower = path.name.lower()
        if (lower in {'elevation_m.npy', 'navigation_states.npz', 'scene_report.json'}
                or lower.startswith(('gps', 'gnss')) or 'query_baseline' in path.parts):
            raise PermissionError('Estimator cannot read simulator geometry/evaluation truth: '+str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--core-root', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    a = parser.parse_args()
    if a.out.exists():
        raise FileExistsError('Preserve earlier board runs.')
    a.out.mkdir(parents=True)
    sys.addaudithook(deny_simulator_files)
    input_stream, output_stream = sys.stdin.buffer, sys.stdout.buffer
    boot_bytes = read_frame(input_stream)
    boot = unpack(boot_bytes)
    validate_boot(boot)
    for name, expected in boot['prior_files_SHA256'].items():
        if sha_file(a.prior/name) != expected:
            raise ValueError('PC/board intensity prior differs: '+name)
    metadata = json.loads((a.prior/'metadata.json').read_text())
    provenance = metadata.get('large_model_provenance', {})
    if (provenance.get('model') != 'random_adapted_56M'
            or any(provenance.get(k) is not False for k in
                   ('edge_repair_used', 'FIR_used', 'brightness_or_contrast_enhancement_used', 'RGB_or_PDF_input_used'))):
        raise ValueError('This HIL requires direct large-model output, no edge repair/enhancement.')
    sys.path.insert(0, str(a.core_root))
    with redirect_stdout(sys.stderr):
        import torch
        import cv2
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('No silent CPU LoFTR fallback.')
        from SideScanFeatureMap import IntensityFeatureMap
        from PureSideScanFeatureNavigation import PureSideScanFeatureNavigator
        from SideScanLearnedFeatures import LoFTRSideScanMatcher
        from oceansim.slam import sanitize_for_strict_json
        matcher = LoFTRSideScanMatcher('cuda', checkpoint=a.weights)
        initial = SimpleNamespace(**{k: np.asarray(v) for k, v in boot['initial'].items()})
        nav = PureSideScanFeatureNavigator(IntensityFeatureMap.from_path(a.prior), initial,
            boot['startup_specific_force'], image_matcher=matcher, loop_closure=True,
            submap_representation='adaptive_projected', mapped_range_changes=True,
            mapped_range_policy='preserve_legacy')
    order = SensorOrder(boot['expected_events'])
    digest = hashlib.sha256(boot_bytes)
    write_frame(output_stream, dict(status='ready', feature_matcher=matcher.provenance,
                                   board=Path('/proc/device-tree/model').read_bytes().rstrip(b'\0').decode(),
                                   estimator='pure_SSS_features_LoFTR_cohort_GTSAM_mapped_range_preserve_legacy',
                                   torch=str(torch.__version__), OpenCV=cv2.__version__))
    sequence = 0
    started = time.monotonic()
    with (a.out/'estimated_positions.jsonl').open('x') as states:
        while True:
            raw = read_frame(input_stream)
            message = unpack(raw)
            if set(message) != {'kind', 'sequence', 'events'} or message['kind'] not in ('batch', 'finish'):
                raise ValueError('Only ordered sensor batches/finish are allowed.')
            if message['sequence'] != sequence:
                raise ValueError('HIL transaction was repeated or dropped.')
            digest.update(raw)
            result = None
            begin = time.monotonic()
            with redirect_stdout(sys.stderr):
                for event in message['events']:
                    kind, values = order.accept(event)
                    if kind == 'imu': nav.imu(event['timestamp'], values[:3], values[3:])
                    elif kind == 'dvl': nav.dvl(values)
                    elif kind == 'depth': nav.depth(float(values[0]))
                    else:
                        if result is not None: raise ValueError('One physical sonar acquisition per transaction.')
                        estimated = nav.side_scan(values)
                        position = np.asarray([estimated[k] for k in ('DR', 'Local', 'Global')])
                        if not np.isfinite(position).all():
                            raise FloatingPointError('Non-finite board navigation output.')
                        result = dict(timestamp=event['timestamp'], positions=position,
                            yaws=np.array([np.arctan2(nav.dr.state.rotation[1, 0], nav.dr.state.rotation[0, 0]),
                                           estimated['raw_pose2'][2], estimated['global_pose2'][2]]))
            final = message['kind'] == 'finish'
            if final:
                order.finish()
            diagnostics, _ = sanitize_for_strict_json(nav.diagnostics()) if final or order.counts['sonar'] % 50 == 0 else ({}, {})
            response = dict(status='complete' if final else 'consumed', sequence=sequence,
                events=order.counts, input_stream_SHA256=digest.hexdigest(), result=result,
                estimator_wall_s=time.monotonic()-begin, elapsed_s=time.monotonic()-started,
                diagnostics=diagnostics)
            if result is not None:
                serial = dict(timestamp=result['timestamp'], positions=result['positions'].tolist(), yaws=result['yaws'].tolist())
                states.write(json.dumps(serial, allow_nan=False)+'\n')
            if final or order.counts['sonar'] % 50 == 0:
                states.flush()
                record = {k: v for k, v in response.items() if k != 'result'}
                record.update(feature_matcher=matcher.provenance, no_query_map_or_evaluation_truth_deployed=True,
                              large_model_provenance=provenance, pure_side_scan=True)
                (a.out/'run_summary.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
            write_frame(output_stream, response)
            sequence += 1
            if final:
                break


if __name__ == '__main__':
    main()
