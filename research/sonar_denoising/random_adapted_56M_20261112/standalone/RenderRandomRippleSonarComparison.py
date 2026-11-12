"""Original native pixels, fixed amber scale, actual vertical waterfall layout."""
from pathlib import Path
import hashlib
import subprocess
import sys

import numpy as np

from NoFIRSonarData import read, save, sha
from NativeSonarPaletteStream import NativePaletteWriter, amplitude_indices, decoded_index_rows
from FullMeasuredSonarProtocol import overlap_block, overlap_tail
from caris_orangecopper import caris_orangecopper


def render(root, sonar):
    from PIL import Image, ImageDraw, ImageFont
    root, sonar = Path(root), Path(sonar)
    out = root / 'evaluation'
    voyage = sonar / 'matched_processing_recovery_20261105_v1/measured_full_voyage_20261105_v1'
    amplitude = np.load(voyage / 'native_source/original_amplitude_float64.npy', mmap_mode='r')
    shape = (amplitude.shape[-1] * 2, 4 * (amplitude.shape[-2] + 1))
    palette = (caris_orangecopper(256) * 255).astype(np.uint8)
    fields = dict(FIR_only=np.load(voyage / 'front_only_ablation_20261105_v1/same_frontend_zero_neural_supported_log_delta.npy', mmap_mode='r'))
    for label, key in (('previous_56M', 'previous_56M'), ('random_public_56M', 'pretrain'), ('random_adapted_56M', 'adapt')):
        fields[label] = np.load(out / f'{key}_supported_log_delta.npy', mmap_mode='r')
    names = ('original', *fields)
    writers = {n: NativePaletteWriter(out / f'{n}_native.png', *shape, palette, reserve_bytes=32*1024**3) for n in names}
    tails = {n: None for n in names}
    images = {}
    try:
        for start in range(0, amplitude.shape[2], 16):
            original = np.asarray(amplitude[:, :, start:start + 16]).transpose(0, 2, 1, 3)
            for name in names:
                value = original if name == 'original' else original * np.exp(
                    np.asarray(fields[name][:, :, start:start + 16], np.float64).transpose(0, 2, 1, 3))
                rows, tails[name] = overlap_block(value, tails[name])
                writers[name].append(amplitude_indices(rows))
        for name in names:
            writers[name].append(amplitude_indices(overlap_tail(tails[name])))
            images[name] = dict(file=f'{name}_native.png', encoding=writers[name].finish())
    finally:
        for writer in writers.values():
            writer.close()
    Image.MAX_IMAGE_PIXELS = 400_000_000
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    overview = Image.new('RGB', (5 * 460, 1400), '#161616')
    detail = Image.new('RGB', (5 * 1040, 2 * 540 + 90), '#161616')
    draw, dd = ImageDraw.Draw(overview), ImageDraw.Draw(detail)
    draw.text((12, 10), 'Full real voyage / fixed CARIS amber 0..1 / no individual enhancement / no clean truth', font=font, fill='white')
    for column, name in enumerate(names):
        path = out / images[name]['file']
        digest = hashlib.sha256()
        count = 0
        for row in decoded_index_rows(path, palette):
            digest.update(row.tobytes())
            count += 1
        if count != shape[1] or digest.hexdigest() != images[name]['encoding']['index_SHA256']:
            raise ValueError('Full native image readback differs from restored data.')
        images[name]['SHA256'] = sha(path)
        with Image.open(path) as image:
            preview = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((450, 1280), Image.Resampling.NEAREST)
            overview.paste(preview.convert('RGB'), (column * 460 + 5, 100))
            if name == 'random_adapted_56M':
                image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((1080, 1920), Image.Resampling.NEAREST).save(out / 'random_adapted_waterfall_preview.png')
            for row, (y, x) in enumerate(((1024, 1024), (4096, 15360))):
                crop = image.crop((x, y, x + 1024, y + 512)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                detail.paste(crop.convert('RGB'), (column * 1040, row * 540 + 90))
        draw.text((column * 460 + 8, 60), name, font=font, fill='white')
        dd.text((column * 1040 + 8, 15), name, font=font, fill='white')
    overview.save(out / 'full_voyage_random_ripple_comparison.png')
    detail.save(out / 'native_detail_random_ripple_comparison.png')
    source = out / 'random_adapted_56M_native.png'
    pdf = out / '随机多源涟漪_56M模型_完整原生高清竖版.pdf'
    subprocess.run([sys.executable, str(Path(__file__).with_name('PackageAspectPreservingSonarPDF.py')),
        '--source', str(source), '--output', str(pdf), '--title', 'Random multi-source ripple trained 56M model - full native waterfall',
        '--layout', 'waterfall', '--page-width', '756', '--page-height', '1344', '--flip-vertical'], check=True)
    report = dict(status='completed_native_images_and_unrotated_waterfall_PDF_NOT_quality_acceptance', images=images,
        same_palette_and_display_limits=[0., 1.], measured_samples_resized=False,
        independent_enhancement=False, image_rotation=False, real_clean_truth=False, quality_accepted=False,
        full_real_voyage_all_16_beams=True, PDF=pdf.name, PDF_SHA256=sha(pdf),
        overview='full_voyage_random_ripple_comparison.png', detail='native_detail_random_ripple_comparison.png')
    report['known_added_noise_comparison'] = render_known(out, palette)
    save(out / 'render_summary.json', report)
    return report


def render_known(out, palette, *, reserve_bytes=32*1024**3):
    from PIL import Image, ImageDraw, ImageFont
    names = ('reference', 'noisy', 'previous_56M', 'pretrain', 'adapt')
    fields = {name: np.load(out / f'known_{name}_amplitude.npy', mmap_mode='r') for name in names}
    shape = fields['reference'].shape
    canvas = Image.new('RGB', (5 * 460, 1400), '#161616')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    draw.text((12, 12), 'Whole reserved native part / same random ripple inputs / NOT certified clean truth / same fixed CARIS 0..1', fill='white', font=font)
    images = {}
    for column, name in enumerate(names):
        path = out / f'known_{name}_native.png'
        writer = NativePaletteWriter(path, 2 * shape[-1], 4 * (shape[-2] + 1), palette, reserve_bytes=reserve_bytes)
        tail = None
        try:
            for row in range(0, shape[-2], 16):
                block = np.asarray(fields[name][:, :, row:row + 16]).transpose(0, 2, 1, 3)
                rows, tail = overlap_block(block, tail)
                writer.append(amplitude_indices(rows))
            writer.append(amplitude_indices(overlap_tail(tail)))
            encoding = writer.finish()
        finally:
            writer.close()
        digest, count = hashlib.sha256(), 0
        for row in decoded_index_rows(path, palette):
            digest.update(row.tobytes())
            count += 1
        if count != 4 * (shape[-2] + 1) or digest.hexdigest() != encoding['index_SHA256']:
            raise ValueError('Known-input native PNG failed full readback.')
        images[name] = dict(file=path.name, SHA256=sha(path), encoding=encoding)
        with Image.open(path) as image:
            canvas.paste(image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((450, 1280), Image.Resampling.NEAREST).convert('RGB'),
                         (column * 460 + 5, 100))
        draw.text((column * 460 + 8, 60), name, fill='white', font=font)
    canvas.save(out / 'known_random_ripple_full_comparison.png')
    return dict(images=images, full_part_preview='known_random_ripple_full_comparison.png',
        native_dimensions=[2 * shape[-1], 4 * (shape[-2] + 1)],
        waterfall_axes_not_rotated=True, complete_all16_beams=True, quality_accepted=False)
