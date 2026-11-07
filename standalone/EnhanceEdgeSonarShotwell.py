"""Enhance the reviewed edge candidate with the real Shotwell 0.30.14 engine.

No generative editing, signal reprocessing, resizing of native pixels, photo
database edits or navigation-map changes. Only the PNG's 256 RGB palette entries
are transformed. The original compressed spatial index stream is copied exactly.
Upstream LGPL engine source is required locally and hash-pinned, not reimplemented.
Run with /usr/bin/python3 (system GdkPixbuf GI, numpy and Pillow).
"""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ExportPinnedFullSamplePortrait import BLOCK, indexed_info, sha
from PackageSonarNativePortraitPDF import hash_range, read_exact


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = 'b48ba90c5656e51279c34591d7fba719fe25a22e28a000d1b5cfa3d5009d97be'
SOURCE_HASHES = {
    'ColorTransformation.vala': '4ff57860d364b839dc63f4875fc0b0b58d7384f4517bfea001609d8e54ac0f99',
    'KeyValueMap.vala': 'bc54cf23af0a382edc581cfc11cd705dd7c6cd14f3fda0fb70b066914ef40fe1',
    '_transformation.c': '45ba1c70639c3764aa6634b81109fb54ca8f049337a5826cc7ef341b1d9eade1',
}
DEFAULT_REFERENCE = ROOT / 'third_party/shotwell_0_30_14'
LOCAL_TOOLCHAIN = Path(os.environ.get('SONAR_VALA_TOOLCHAIN',
    str(ROOT / 'downloads/shotwell_0.30.14_reference_20261107/toolchain/usr')))


def clean_environment():
    env = os.environ.copy()
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env['LC_ALL'] = 'C.UTF-8'
    return env


def build_engine(reference):
    """Build unchanged upstream source using privately extracted compiler debs.

    Prerequisites: gcc + system glib/gdk-pixbuf dev headers; extract Ubuntu valac,
    valac-bin, valac-0.56-vapi, libvala-0.56-0, libvalacodegen-0.56-0 and
    libgee-0.8-dev debs into reference/toolchain. No system installation needed.
    """
    reference = Path(reference).resolve()
    for name, digest in SOURCE_HASHES.items():
        if sha(reference / name) != digest:
            raise ValueError(f'Unreviewed upstream source: {name}')
    build = reference / 'build_palette_adapter'
    build.mkdir(exist_ok=True)
    adapter = Path(__file__).parent / 'shotwell_reference/EnhancePalette.vala'
    inputs = [reference / name for name in SOURCE_HASHES] + [adapter]
    for source in inputs:
        destination = build / source.name
        if destination.exists() and sha(destination) != sha(source):
            raise ValueError(f'Preserve existing different build source: {destination}')
        shutil.copy2(source, destination)
    tool = reference / 'toolchain/usr'
    if not tool.is_dir():
        tool = LOCAL_TOOLCHAIN
    env = clean_environment()
    system_compiler = shutil.which('valac')
    compiler_options = []
    if system_compiler:
        compiler = system_compiler
    else:
        compiler = str(tool / 'bin/valac-0.56')
        if not Path(compiler).exists():
            raise RuntimeError('Install valac and libgee-0.8-dev; see the pipeline README')
        env['LD_LIBRARY_PATH'] = ':'.join(str(tool / x) for x in (
            'lib/x86_64-linux-gnu', 'lib/x86_64-linux-gnu/vala-0.56'))
        compiler_options = ['--vapidir=' + str(tool / 'share/vala-0.56/vapi'),
                            '--vapidir=' + str(tool / 'share/vala/vapi')]
    command = [compiler, *compiler_options,
               '--pkg=gdk-pixbuf-2.0', '--pkg=gio-2.0', '--pkg=gee-0.8',
               '-C', '-H', 'shotwell-graphics-processor.h',
               'ColorTransformation.vala', 'KeyValueMap.vala', 'EnhancePalette.vala']
    subprocess.run(command, cwd=build, env=env, check=True)
    flags = shlex.split(subprocess.check_output(
        ['pkg-config', '--cflags', '--libs', 'gdk-pixbuf-2.0', 'gio-2.0'],
        env=clean_environment(), text=True))
    binary = build / 'shotwell-enhance-palette'
    gee_flags = (shlex.split(subprocess.check_output(
        ['pkg-config', '--cflags', '--libs', 'gee-0.8'],
        env=clean_environment(), text=True)) if system_compiler else
        ['-I' + str(tool / 'include/gee-0.8'), '/usr/lib/x86_64-linux-gnu/libgee-0.8.so.2'])
    gcc = ['gcc', '-O2', '-ffp-contract=off', '-I.',
           'ColorTransformation.c', 'KeyValueMap.c', 'EnhancePalette.c',
           '_transformation.c', '-o', str(binary), *flags,
           *gee_flags, '-lm']
    with (build / 'gcc.log').open('w') as log:
        subprocess.run(gcc, cwd=build, env=clean_environment(), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    provenance = dict(source_tag='shotwell-0.30.14', source_hashes=SOURCE_HASHES,
                      source_repository='https://github.com/GNOME/shotwell',
                      adapter_SHA256=sha(adapter), binary_SHA256=sha(binary),
                      valac_command=command, gcc_command=gcc,
                      sources_are_unmodified=True, gtk_red_eye_Point_shim_unused=True)
    (build / 'build_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return binary, provenance


def replace_palette(source, output, palette):
    """Copy all PNG chunks exactly except PLTE and explanatory tEXt metadata."""
    palette = bytes(palette)
    if len(palette) != 768:
        raise ValueError('Require 256 RGB colours')
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    indexed_info(source)  # validates PNG, palette, profiles and compressed chunks
    partial = output.with_suffix('.png.incomplete')
    with source.open('rb') as src, partial.open('xb') as dest:
        dest.write(read_exact(src, 8))
        while True:
            header = read_exact(src, 8)
            length, tag = struct.unpack('>I4s', header)
            if tag == b'PLTE':
                read_exact(src, length + 4)
                dest.write(struct.pack('>I', len(palette)) + tag + palette)
                dest.write(struct.pack('>I', zlib.crc32(tag + palette) & 0xffffffff))
            elif tag == b'tEXt':
                read_exact(src, length + 4)
            else:
                dest.write(header)
                remaining = length + 4
                while remaining:
                    data = read_exact(src, min(BLOCK, remaining))
                    dest.write(data)
                    remaining -= len(data)
            if tag == b'IEND':
                break
    before, after = indexed_info(source), indexed_info(partial)
    if (before['width'], before['height'], before['idat_SHA256']) != (
            after['width'], after['height'], after['idat_SHA256']):
        raise ValueError('Native index stream or dimensions changed')
    if after['palette'] != palette:
        raise ValueError('Palette write differs')
    partial.rename(output)
    return after


def portrait_pdf(source, output, info):
    """Native pixels embedded without resampling; established portrait layout."""
    source, output = Path(source), Path(output)
    partial = output.with_suffix('.pdf.incomplete')
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    offsets = [0] * 7
    with source.open('rb') as src, partial.open('xb') as dest:
        dest.write(b'%PDF-1.5\n%\xe2\xe3\xcf\xd3\n')

        def obj(index, data):
            offsets[index] = dest.tell()
            dest.write(f'{index} 0 obj\n'.encode() + data + b'\nendobj\n')

        obj(1, b'<< /Type /Catalog /Pages 2 0 R >>')
        obj(2, b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>')
        obj(3, b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 756 1344] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>')
        offsets[4] = dest.tell()
        dest.write((f"4 0 obj\n<< /Type /XObject /Subtype /Image /Width {info['width']} /Height {info['height']} "
                    f"/ColorSpace [/Indexed /DeviceRGB 255 <{info['palette'].hex()}>] /BitsPerComponent 8 "
                    '/Interpolate false /Filter /FlateDecode '
                    f"/DecodeParms << /Predictor 15 /Colors 1 /BitsPerComponent 8 /Columns {info['width']} >> "
                    f"/Length {info['idat_bytes']} >>\nstream\n").encode())
        pixel_start = dest.tell()
        for position, count in info['idat_spans']:
            src.seek(position)
            while count:
                data = read_exact(src, min(BLOCK, count))
                dest.write(data)
                count -= len(data)
        dest.write(b'\nendstream\nendobj\n')
        content = b'q\n756 0 0 -1344 0 1344 cm\n/Im0 Do\nQ\n'
        obj(5, f'<< /Length {len(content)} >>\nstream\n'.encode() + content + b'endstream')
        title = (b'\xfe\xff' + '边缘修复候选＋Shotwell自动增强：完整原生像素高清竖版'.encode('utf-16-be')).hex()
        obj(6, (f'<< /Title <{title}> /Creator (Shotwell 0.30.14 engine and OceanSim lossless packaging) '
                '/Subject (Display-only enhancement. Native pixels retained. Portrait aspect is presentation, not physical georectification or added resolution.) >>').encode())
        xref = dest.tell()
        dest.write(b'xref\n0 7\n0000000000 65535 f \n')
        for offset in offsets[1:]:
            dest.write(f'{offset:010d} 00000 n \n'.encode())
        dest.write(f'trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    if hash_range(partial, pixel_start, info['idat_bytes']) != info['idat_SHA256']:
        raise ValueError('PDF native pixel embedding changed')
    partial.rename(output)
    return dict(embedded_IDAT_SHA256=info['idat_SHA256'],
                embedded_dimensions=[info['width'], info['height']],
                native_samples_not_resampled=True,
                display_vertical_flip=True, portrait_aspect_is_display_only=True)


def make_previews(source, out, palette):
    Image.MAX_IMAGE_PIXELS = 250_000_000
    resampling = getattr(Image, 'Resampling', Image)
    transpose = getattr(Image, 'Transpose', Image)
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 30)
    with Image.open(source) as original:
        original.load()  # ~197 MB indexed pixels, not a billion-pixel display PNG
        small = original.resize((1100, 1956), resampling.NEAREST)
        before = small.convert('RGB').transpose(transpose.FLIP_TOP_BOTTOM)
        small.putpalette(palette)
        after = small.convert('RGB').transpose(transpose.FLIP_TOP_BOTTOM)
        before.save(out / '边缘修复候选_竖版预览.png')
        after.save(out / 'Shotwell增强_竖版预览.png')
        canvas = Image.new('RGB', (2224, 2042), '#eeeeee')
        canvas.paste(before, (8, 76))
        canvas.paste(after, (1116, 76))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 17), '增强前：边缘修复候选', font=font, fill='black')
        draw.text((1128, 17), '增强后：Shotwell 0.30.14', font=font, fill='black')
        canvas.save(out / '增强前后_整幅竖版对比.png')
        # All native crop pixels retained at 1:1. Coordinates are in native PNG,
        # not physical metres. Avoid the centre no-return gap and border padding.
        crops = [(3200, 1700, 4736, 2468), (16200, 4600, 17736, 5368),
                 (21000, 5700, 22536, 6468)]
        if original.size != (24378, 8084):
            w, h = original.size
            cw, ch = min(w//4, 1536), min(h//4, 768)
            crops = [(int(w*x), int(h*y), int(w*x)+cw, int(h*y)+ch)
                     for x, y in ((.1, .1), (.65, .4), (.7, .7))]
        cropped = Image.new('RGB', (3088, 3 * 822 + 64), '#eeeeee')
        d = ImageDraw.Draw(cropped)
        d.text((16, 6), '左：增强前  |  右：增强后  |  原生像素 1:1 局部，不做锐化', font=font, fill='black')
        for index, box in enumerate(crops):
            tile = original.crop(box)
            a = tile.convert('RGB')
            tile.putpalette(palette)
            b = tile.convert('RGB')
            y = 64 + index * 822
            d.text((12, y), f'原图像素窗口 {box}', font=font, fill='black')
            cropped.paste(a, (0, y + 48))
            cropped.paste(b, (1552, y + 48))
        cropped.save(out / '增强前后_原生像素局部对比.png')
    return dict(preview_is_not_native_resolution=True,
                preview_dimensions=[1100, 1956],
                native_crop_rectangles=crops, native_crop_scale=1.0,
                portrait_layout_matches_previous_exports=True)


def enhance(source, out, reference=DEFAULT_REFERENCE, expected_sha=SOURCE_SHA):
    import gi
    gi.require_version('GdkPixbuf', '2.0')
    from gi.repository import GdkPixbuf

    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Keep all previous exports; choose a fresh output directory')
    before = indexed_info(source)
    if before['file_SHA256'] != expected_sha:
        raise ValueError('Not the reviewed gain_and_commonmode edge candidate')
    initial = (source.stat().st_size, source.stat().st_mtime_ns)
    version = 'not_installed_engine_source_is_pinned_0.30.14'
    if shutil.which('dpkg-query'):
        installed = subprocess.run(['dpkg-query', '-W', '-f=${Version}', 'shotwell'],
                                   text=True, capture_output=True)
        if installed.returncode == 0:
            version = installed.stdout
    binary, build = build_engine(reference)
    out.mkdir(parents=True)
    print('Native source verified; exact upstream engine built.', flush=True)
    width, height = before['width'], before['height']
    ratio = min(1.0, 360.0 / max(width, height))
    sw, sh = [max(1, math.floor(v * ratio + 0.5)) for v in (width, height)]
    analysis = GdkPixbuf.Pixbuf.new_from_file_at_scale(str(source), sw, sh, False)
    # Photo.vala performs this final bilinear fit as well, even at equal size.
    analysis = analysis.scale_simple(sw, sh, GdkPixbuf.InterpType.BILINEAR)
    analysis.savev(str(out / 'Shotwell分析缩略图_360.png'), 'png', [], [])
    Image.frombytes('RGB', (16, 16), before['palette']).save(out / '输入调色板.png')
    subprocess.run([str(binary), str(out / 'Shotwell分析缩略图_360.png'),
                    str(out / '输入调色板.png'), str(out / '增强调色板.png'),
                    str(out / 'Shotwell实际参数.json')], env=clean_environment(), check=True)
    with Image.open(out / '增强调色板.png') as p:
        palette = p.convert('RGB').tobytes()
    parameters = json.loads((out / 'Shotwell实际参数.json').read_text())
    print(json.dumps(parameters, ensure_ascii=False), flush=True)
    native = out / f'边缘修复_Shotwell增强_完整原生{width}x{height}.png'
    info = replace_palette(source, native, palette)
    pdf = out / '边缘修复_Shotwell增强_完整原生像素_高清竖版.pdf'
    pdf_report = portrait_pdf(native, pdf, info)
    preview_report = make_previews(source, out, palette)
    if initial != (source.stat().st_size, source.stat().st_mtime_ns) or sha(source) != expected_sha:
        raise ValueError('Input changed while exporting')
    new = np.frombuffer(palette, np.uint8).reshape(256, 3)
    old = np.frombuffer(before['palette'], np.uint8).reshape(256, 3)
    outputs = {p.name: dict(bytes=p.stat().st_size, SHA256=sha(p))
               for p in out.iterdir() if p.is_file()}
    report = dict(status='completed_display_only_native_pixels_preserved',
                  source=str(source), source_SHA256=expected_sha,
                  edge_candidate='gain_and_commonmode', installed_shotwell_version=version,
                  build=build, parameters=parameters, native_dimensions=[width, height],
                  original_IDAT_SHA256=before['idat_SHA256'],
                  output_IDAT_SHA256=info['idat_SHA256'],
                  spatial_indices_byte_exact=True, source_unchanged=True,
                  black_palette_entries_stay_black=bool(np.all(new[np.all(old == 0, axis=1)] == 0)),
                  no_new_denoising=True, no_generative_AI=True, navigation_inputs_unchanged=True,
                  new_display_not_radiometrically_calibrated_amplitudes=True,
                  pdf=pdf_report, previews=preview_report, outputs=outputs)
    (out / '处理与原生像素校验.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Complete: ' + str(pdf), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'output/real_sonar/range_gain_continuity_172_20261106_v1/native_gain_and_commonmode.png')
    parser.add_argument('--out', type=Path, default=ROOT / 'output/real_sonar/edge_shotwell_enhance_20261107_v2')
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--expected-source-sha', default=SOURCE_SHA,
                        help='Explicit preceding stage output SHA; default is the reviewed full candidate')
    args = parser.parse_args()
    enhance(args.source, args.out, args.reference, args.expected_source_sha)
