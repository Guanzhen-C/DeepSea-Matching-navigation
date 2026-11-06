"""Independently bind navigation fields to the stopped denoising outputs.

Checks all original baseline beam digests and every last-512 native mosaic
pixel for BOTH stopped outputs. No GPS, inference, training or RGB editing.
"""
from pathlib import Path
import argparse
import hashlib
import json
import mmap
import sys

import numpy as np

from NativeSonarPaletteStream import decoded_index_rows
from caris_orangecopper import caris_orangecopper
from QiandaoLocalScene import reject_gps_file


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--project', required=True, type=Path)
    a = p.parse_args()
    sys.addaudithook(reject_gps_file)
    output = a.experiment/'stopped_input_provenance_audit.json'
    if output.exists():
        raise FileExistsError('Keep previous provenance audits.')
    frozen = a.project/'output/real_sonar/matched_full_measured_172_20261105_v1/full_frozen_inference'
    magnitude = np.load(a.project/'output/real_sonar/raw_mat_preprocessing_20261104/pure_wavelet_tuning/原始MAT逐束包络_float64.npy', mmap_mode='r')
    baseline = np.load(a.experiment/'inputs/fusion_wide_front_supported_log_delta.npy', mmap_mode='r')
    edge = np.load(a.experiment/'inputs/gain_and_commonmode_last512.npy', mmap_mode='r')
    expected = {}
    for i in range(4):
        worker = json.loads((frozen/f'worker_{i}.json').read_text())
        for item in worker['beams']:
            expected[item['side'], item['beam_0based']] = item['models']['fusion_wide_front']['applied_total_delta_SHA256']
    checked = []
    for side in range(2):
        for beam in range(8):
            values = np.asarray(baseline[side, beam])
            digest = hashlib.sha256(memoryview(values).cast('B')).hexdigest()
            if digest != expected[side, beam]:
                raise ValueError('A downloaded full baseline field differs from the original producer.')
            checked.append(dict(side=side, beam=beam, SHA256=digest, values=values.size))
            baseline._mmap.madvise(mmap.MADV_DONTNEED)
    images = {'baseline': (frozen/'native_full_fusion_wide_front.png',
                          '510621757a2323767974f19b6a1188169ff11df1f4044febb8ad415c3233d2fc'),
              'edge': (a.project/'output/real_sonar/range_gain_continuity_172_20261106_v1/native_gain_and_commonmode.png',
                       'b48ba90c5656e51279c34591d7fba719fe25a22e28a000d1b5cfa3d5009d97be')}
    palette = (caris_orangecopper(256)*255).astype(np.uint8)
    results = {}
    for variant, (image, expected_sha) in images.items():
        if sha(image) != expected_sha:
            raise ValueError('Stopped native image changed.')
        previous = np.zeros((2, 4, 512), np.float64)
        rows, pixels = 0, 0
        decoder = iter(decoded_index_rows(image, palette))

        def check(values):
            nonlocal rows, pixels
            # Independent original left-reversal/right-forward four-row layout.
            sides = np.concatenate((values[0, :, ::-1], values[1]), axis=1)
            indices = np.minimum((np.clip(sides, 0, 1)*256).astype(np.uint16), 255).astype(np.uint8)
            for row in indices:
                full = next(decoder, None)
                if full is None or not np.array_equal(np.r_[full[:512], full[-512:]], row):
                    raise ValueError('Stopped floating field does not reproduce actual native edge pixels.')
                rows += 1
                pixels += len(row)

        for t in range(2020):
            m = np.asarray(magnitude[:, :, t, -512:])
            d = np.asarray(edge[:, :, t] if variant == 'edge' else baseline[:, :, t, -512:], np.float64)
            restored = m*np.exp(d)
            check(restored[:, :4]+previous)
            previous = restored[:, 4:].copy()
            if t % 100 == 0:
                magnitude._mmap.madvise(mmap.MADV_DONTNEED)
                baseline._mmap.madvise(mmap.MADV_DONTNEED)
                edge._mmap.madvise(mmap.MADV_DONTNEED)
        check(previous)
        if rows != 8084 or next(decoder, None) is not None:
            raise ValueError('Full physical-record/tail coverage incomplete.')
        results[variant] = dict(native_rows=rows, exact_mosaic_edge_pixels_checked=pixels,
                               original_native_image_SHA256=expected_sha)
    report = dict(status='passed_original_full_beam_hashes_and_all_stopped_native_edge_pixels',
        full_baseline_beams=checked, native_mosaic_checks=results,
        all_2020_physical_records_and_16_beams=True, GPS_used=False,
        new_training_or_neural_inference=False, original_MAT_or_stopped_outputs_changed=False,
        baseline_gain_file_SHA256=sha(a.experiment/'inputs/fusion_wide_front_supported_log_delta.npy'),
        edge_gain_file_SHA256=sha(a.experiment/'inputs/gain_and_commonmode_last512.npy'),
        auditor_SHA256=sha(Path(__file__)))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], beams=len(checked), pixel_checks=results), ensure_ascii=False))


if __name__ == '__main__':
    main()
