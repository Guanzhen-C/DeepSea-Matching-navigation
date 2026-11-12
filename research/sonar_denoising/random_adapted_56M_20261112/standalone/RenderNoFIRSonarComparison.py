"""Same-display raw/FIR/previous25M/new56M no-FIR whole-voyage comparison."""
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from NoFIRSonarData import read, save, sha
from FullMeasuredSonarProtocol import overlap_block, overlap_tail
from NativeSonarPaletteStream import NativePaletteWriter, amplitude_indices, decoded_index_rows
from ReviewFullMeasured172 import STYLE_AREAS, projection
from caris_orangecopper import caris_orangecopper


def render(root, sonar):
    root, sonar = Path(root), Path(sonar)
    out = root/'evaluation'
    voyage = sonar/'matched_processing_recovery_20261105_v1/measured_full_voyage_20261105_v1'
    amplitude = np.load(voyage/'native_source/original_amplitude_float64.npy', mmap_mode='r')
    manifest = read(voyage/'metadata/measured_protocol_manifest.json')
    height, width = manifest['native_mosaic_shape']
    locations = {
        'FIR_only': voyage/'front_only_ablation_20261105_v1/same_frontend_zero_neural_supported_log_delta.npy',
        'previous_25M_FIR': sonar/'large_wavelet_capacity_20261108_v1/experiment/measured_full/large_mwcnn_supported_log_delta.npy',
        'public_56M_NO_FIR': out/'pretrain_supported_log_delta.npy',
        'adapted_56M_NO_FIR': out/'adapt_supported_log_delta.npy'}
    fields = {name: np.load(path, mmap_mode='r') for name, path in locations.items()}
    names = ('original', *fields)
    palette = (caris_orangecopper(256)*255).astype(np.uint8)
    writers = {name: NativePaletteWriter(out/f'{name}_native.png', width, height, palette,
                reserve_bytes=32*1024**3) for name in names}
    tails = {name: None for name in names}
    sums = {name: 0. for name in names}
    report = dict(status='rendering_all_native_rows_same_fixed_display', images={}, real_clean_truth=False,
        same_palette_and_display_bounds=[0., 1.], original_samples_resized=False,
        independent_enhancement_and_edge_repair=False, quality_accepted=False)
    save(out/'render_summary.json', report)
    try:
        for start in range(0, amplitude.shape[2], 16):
            stop = min(start+16, amplitude.shape[2])
            original = np.asarray(amplitude[:, :, start:stop]).transpose(0, 2, 1, 3)
            for name in names:
                value = original if name == 'original' else original*np.exp(
                    np.asarray(fields[name][:, :, start:stop], np.float64).transpose(0, 2, 1, 3))
                rows, tails[name] = overlap_block(value, tails[name])
                sums[name] += float(rows.sum(dtype=np.float64))
                writers[name].append(amplitude_indices(rows))
        for name in names:
            rows = overlap_tail(tails[name])
            sums[name] += float(rows.sum(dtype=np.float64))
            writers[name].append(amplitude_indices(rows))
            report['images'][name] = dict(file=f'{name}_native.png', encoding=writers[name].finish())
    finally:
        for writer in writers.values():
            writer.close()
    Image.MAX_IMAGE_PIXELS = 400_000_000
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    overview = Image.new('RGB', (2100, 1380), '#161616')
    draw = ImageDraw.Draw(overview)
    detail = Image.new('RGB', (5*1040, len(manifest['regions'])*1080+70), '#161616')
    dd = ImageDraw.Draw(detail)
    for index, name in enumerate(names):
        path = out/report['images'][name]['file']
        digest, count = hashlib.sha256(), 0
        for row in decoded_index_rows(path, palette):
            digest.update(row.tobytes())
            count += 1
        if count != height or digest.hexdigest() != report['images'][name]['encoding']['index_SHA256']:
            raise ValueError('complete PNG pixel readback differs from numeric output')
        report['images'][name]['file_SHA256'] = sha(path)
        with Image.open(path) as image:
            portrait = image.transpose(Image.Transpose.ROTATE_90)
            portrait.save(out/f'{name}_native_portrait.png')
            portrait.thumbnail((400, 1250), resample=Image.Resampling.NEAREST)
            overview.paste(portrait.convert('RGB'), (index*420+10, 92))
            for ordinal, region in enumerate(manifest['regions']):
                y, x, h, w = region['coordinates']
                detail.paste(image.crop((x, y, x+w, y+h)).convert('RGB'), (index*1040, ordinal*1080+70))
        draw.text((index*420+10, 22), name, font=font, fill='white')
        draw.text((index*420+10, 54), 'Same fixed display / full field', font=font, fill='white')
        dd.text((index*1040+8, 15), name, font=font, fill='white')
    overview.save(out/'full_voyage_NO_FIR_comparison.png')
    detail.save(out/'native_detail_NO_FIR_comparison.png')
    components = []
    for label, side, row, col, fy, fx in STYLE_AREAS:
        before = np.asarray(amplitude[side, :, row:row+256, col:col+1024])
        initial = projection(before, fy, fx)
        changes = {}
        for name, field in fields.items():
            changed = before*np.exp(np.asarray(field[side, :, row:row+256, col:col+1024], np.float64))
            after = projection(changed, fy, fx)
            changes[name] = float(np.median(20*np.log10(np.maximum(after, 1e-30)/np.maximum(initial, 1e-30))))
        components.append(dict(label=label, fixed_component_changes_dB=changes, component_NOT_pure_noise_truth=True))
    report.update(status='completed_all_native_pixels_and_same_display_NO_FIR_comparison_NOT_quality_acceptance',
        mean_amplitude_vs_original={n: sums[n]/sums['original'] for n in names[1:]},
        brightness_change_NOT_noise_removal=True, fixed_components=components,
        overview='full_voyage_NO_FIR_comparison.png', detail='native_detail_NO_FIR_comparison.png',
        lossless_native_portrait_rotation_not_resizing=True)
    save(out/'render_summary.json', report)
    return dict(report=str(out/'render_summary.json'), report_SHA256=sha(out/'render_summary.json'),
                overview=str(out/report['overview']), detail=str(out/report['detail']))
