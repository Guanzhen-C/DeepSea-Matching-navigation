"""Lossless portrait PDF/PNG for the user-pinned denoised native result.

No model inference, signal processing, invented detail, contrast or cropping.
PDF reuses the exact indexed PNG compressed stream. PNG retains every native
column/row and repeats rows for portrait layout only. Never decode the complete
billion-pixel portrait PNG into RAM or pass it to a full-image viewer here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import struct
import tempfile
import zlib

import numpy as np
from NativeSonarPaletteStream import NativePaletteWriter, decoded_index_rows
from ExportNativeSonarPortraitPNG import row_repetitions
from PackageSonarNativePortraitPDF import read_exact, hash_range


BLOCK = 1024*1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(BLOCK), b''):
            h.update(b)
    return h.hexdigest()


def indexed_info(path):
    spans = []
    full = hashlib.sha256()
    pixels = hashlib.sha256()
    palette = None
    width = height = None
    with Path(path).open('rb') as f:
        signature = read_exact(f, 8)
        if signature != b'\x89PNG\r\n\x1a\n':
            raise ValueError('Native PNG required.')
        full.update(signature)
        seen_idat = ended_idat = False
        while True:
            header = read_exact(f, 8)
            full.update(header)
            length, tag = struct.unpack('>I4s', header)
            if tag in (b'IHDR', b'PLTE') and length not in ((13,) if tag == b'IHDR' else (768,)):
                raise ValueError('Original 8-bit 256-colour native palette required.')
            if tag in (b'tRNS', b'iCCP', b'acTL', b'fcTL', b'fdAT'):
                raise ValueError('Unexpected transparency/profile/animation.')
            if tag == b'IDAT':
                if ended_idat:
                    raise ValueError('Nonconsecutive compressed pixels.')
                seen_idat = True
                spans.append((f.tell(), length))
            elif seen_idat:
                ended_idat = True
            crc = zlib.crc32(tag)
            remaining = length
            capture = bytearray()
            while remaining:
                data = read_exact(f, min(BLOCK, remaining))
                full.update(data)
                crc = zlib.crc32(data, crc)
                if tag == b'IDAT':
                    pixels.update(data)
                elif tag in (b'IHDR', b'PLTE'):
                    capture.extend(data)
                remaining -= len(data)
            trailer = read_exact(f, 4)
            full.update(trailer)
            if struct.unpack('>I', trailer)[0] != (crc & 0xffffffff):
                raise ValueError('Native PNG checksum failed.')
            if tag == b'IHDR':
                if width is not None:
                    raise ValueError('Duplicate native header.')
                width, height, bits, kind, comp, filt, interlace = struct.unpack('>IIBBBBB', capture)
                if min(width, height) < 1 or (bits, kind, comp, filt, interlace) != (8, 3, 0, 0, 0):
                    raise ValueError('Noninterlaced indexed native PNG required.')
            if tag == b'PLTE':
                palette = bytes(capture)
            if tag == b'IEND':
                if length or not spans or palette is None or width is None or f.read(1):
                    raise ValueError('Incomplete native PNG.')
                break
    return dict(width=width, height=height, palette=palette, idat_spans=spans,
                idat_bytes=sum(n for _, n in spans), file_SHA256=full.hexdigest(),
                idat_SHA256=pixels.hexdigest())


def portrait_pdf(source, output, info):
    if output.exists():
        raise FileExistsError('Keep previous PDF.')
    incomplete = output.with_suffix('.pdf.incomplete')
    if incomplete.exists():
        raise FileExistsError('Keep incomplete evidence.')
    offsets = [0]*7
    pw, ph = 756, 1344
    with incomplete.open('xb') as f, source.open('rb') as src:
        f.write(b'%PDF-1.5\n%\xe2\xe3\xcf\xd3\n')

        def obj(i, data):
            offsets[i] = f.tell()
            f.write(f'{i} 0 obj\n'.encode()+data+b'\nendobj\n')

        obj(1, b'<< /Type /Catalog /Pages 2 0 R >>')
        obj(2, b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>')
        obj(3, (f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw} {ph}] '
                '/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>').encode())
        offsets[4] = f.tell()
        palette_hex = info['palette'].hex()
        f.write((f"4 0 obj\n<< /Type /XObject /Subtype /Image /Width {info['width']} /Height {info['height']} "
                 f'/ColorSpace [/Indexed /DeviceRGB 255 <{palette_hex}>] /BitsPerComponent 8 '
                 '/Interpolate false /Filter /FlateDecode '
                 f"/DecodeParms << /Predictor 15 /Colors 1 /BitsPerComponent 8 /Columns {info['width']} >> "
                 f"/Length {info['idat_bytes']} >>\nstream\n").encode())
        pixel_start = f.tell()
        for position, size in info['idat_spans']:
            src.seek(position)
            while size:
                data = read_exact(src, min(BLOCK, size))
                f.write(data)
                size -= len(data)
        f.write(b'\nendstream\nendobj\n')
        # Flip the display rectangle, not the original pixel stream, to match
        # the existing portrait preview's newest-record-at-top convention.
        content = f'q\n{pw} 0 0 {-ph} 0 {ph} cm\n/Im0 Do\nQ\n'.encode()
        obj(5, f'<< /Length {len(content)} >>\nstream\n'.encode()+content+b'endstream')
        title = (b'\xfe\xff'+'固定去噪基线：融合／宽带／前级，全原生像素竖版'.encode('utf-16-be')).hex()
        obj(6, (f'<< /Title <{title}> /Creator (OceanSim lossless indexed native portrait layout) '
                '/Subject (Pinned fusion_wide_front denoised result. All native samples retained; layout is not georectification or super-resolution.) >>').encode())
        xref = f.tell()
        f.write(b'xref\n0 7\n0000000000 65535 f \n')
        for position in offsets[1:]:
            f.write(f'{position:010d} 00000 n \n'.encode())
        f.write(f'trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    embedded = hash_range(incomplete, pixel_start, info['idat_bytes'])
    if embedded != info['idat_SHA256']:
        raise ValueError('PDF compressed samples differ from source.')
    incomplete.rename(output)
    return dict(file=str(output), SHA256=sha(output), bytes=output.stat().st_size,
                original_compressed_pixels_exactly_preserved=True, embedded_idat_SHA256=embedded,
                embedded_dimensions=[info['width'], info['height']], page_points=[pw, ph],
                vertical_flip_for_waterfall_is_display_matrix_only=True,
                interpolation=False, spatial_resampling=False)


def export(source, out, expected_sha=None):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Keep earlier portrait exports.')
    before = (source.stat().st_size, source.stat().st_mtime_ns)
    info = indexed_info(source)
    if expected_sha is not None and info['file_SHA256'] != expected_sha:
        raise ValueError('Not the exact pinned native denoised image.')
    if shutil.disk_usage(out.parent).free < 2*1024**3:
        raise OSError('Keep at least two GiB free before this bounded export.')
    out.mkdir()
    width, height = info['width'], info['height']
    display_height = max(height, (width*16+8)//9)
    counts = row_repetitions(height, display_height)
    palette = np.frombuffer(info['palette'], np.uint8).reshape(256, 3)
    pdf = portrait_pdf(source, out/'固定方案_完整原生像素_竖版.pdf', info)
    print(json.dumps(dict(stage='lossless_portrait_PDF_done', embedded=[width, height])), flush=True)
    png_file = out/f'固定方案_全采样竖版_{width}x{display_height}.png'
    with tempfile.TemporaryDirectory(prefix='portrait-row-cache-', dir=out) as temporary:
        cache = np.lib.format.open_memmap(Path(temporary)/'original_indices.npy', mode='w+', dtype=np.uint8,
                                          shape=(height, width))
        count = 0
        indices_hash = hashlib.sha256()
        for count, row in enumerate(decoded_index_rows(source, palette), 1):
            cache[count-1] = row
            indices_hash.update(row.tobytes())
        if count != height:
            raise ValueError('Source rows incomplete.')
        cache.flush()
        writer = NativePaletteWriter(png_file, width, display_height, palette, reserve_bytes=1024**3)
        writer.chunk(b'tEXt', b'DisplayLayout\x00All original indices preserved. Rows flipped and repeated for portrait layout only; no new measured resolution, inference or image generation.')
        try:
            for i, repeat in enumerate(counts):
                row = np.asarray(cache[height-1-i])
                writer.append(np.repeat(row[None], repeat, axis=0))
                if i % 1024 == 0:
                    print(json.dumps(dict(stage='portrait_row_replication', native_rows=i, total=height)), flush=True)
            encoding = writer.finish()
        finally:
            writer.close()
        output_rows = iter(decoded_index_rows(png_file, palette))
        verified = 0
        for i, repeat in enumerate(counts):
            expected = np.asarray(cache[height-1-i])
            for _ in range(repeat):
                actual = next(output_rows, None)
                if actual is None or not np.array_equal(actual, expected):
                    raise ValueError('Portrait output changed a measured palette index.')
                verified += 1
        if verified != display_height or next(output_rows, None) is not None:
            raise ValueError('Portrait output coverage differs.')
        cache._mmap.close()
    if before != (source.stat().st_size, source.stat().st_mtime_ns) or sha(source) != info['file_SHA256']:
        raise ValueError('Pinned native source changed.')
    result = dict(status='completed_lossless_pinned_result_portrait_PDF_and_full_sample_PNG',
        model='fusion_wide_front', source=str(source), source_SHA256=info['file_SHA256'],
        source_palette_index_SHA256=indices_hash.hexdigest(), original_dimensions=[width, height],
        real_native_sample_pixels=width*height, pdf=pdf,
        png=dict(file=str(png_file), SHA256=sha(png_file), bytes=png_file.stat().st_size,
                 width_height=[width, display_height], encoding=encoding,
                 every_output_row_index_exactly_verified=verified),
        row_repetitions_min_max=[min(counts), max(counts)], all_source_rows_and_columns_retained=True,
        horizontal_downsampling=False, source_unchanged=True, display_vertical_flip=True,
        vertical_repeated_rows_are_layout_only=True, measured_resolution_increased=False,
        image_content_interpolated=False, extra_denoising_or_neural_inference=False,
        AI_generation=False, brightness_or_colour_limits_changed=False,
        decoded_portrait_RGBA_bytes=width*display_height*4,
        do_not_open_full_billion_pixel_PNG_in_memory_limited_viewer=True)
    (out/'全采样竖版校验.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    baseline = json.loads(a.baseline.read_text())
    if baseline['default_model'] != 'fusion_wide_front':
        raise ValueError('Only the user-pinned model, not a later boundary candidate.')
    item = baseline['verified_existing_result']
    result = export(item['local_native_image'], a.out, item['native_image_SHA256'])
    print(json.dumps(dict(status=result['status'], pdf=result['pdf']['file'], png=result['png']['file'],
                          png_dimensions=result['png']['width_height'], verified_rows=result['png']['every_output_row_index_exactly_verified']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
