"""Lossless full-native indexed PNG portrait PDF, labelled as a NEW candidate.

Document packaging only. Reuses the exact compressed original input PNG IDAT
and palette. Does not relabel a candidate as the user's fixed approved result.
"""
from pathlib import Path
import argparse
import json

from ExportPinnedFullSamplePortrait import indexed_info, sha, BLOCK
from PackageSonarNativePortraitPDF import read_exact, hash_range


def package(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix('.pdf.incomplete').exists():
        raise FileExistsError('Preserve previous complete and incomplete exports.')
    info = indexed_info(source)
    initial = (source.stat().st_size, source.stat().st_mtime_ns)
    partial = output.with_suffix('.pdf.incomplete')
    offsets = [0]*7
    with partial.open('xb') as f, source.open('rb') as src:
        f.write(b'%PDF-1.5\n%\xe2\xe3\xcf\xd3\n')

        def obj(i, data):
            offsets[i] = f.tell()
            f.write(f'{i} 0 obj\n'.encode()+data+b'\nendobj\n')

        obj(1, b'<< /Type /Catalog /Pages 2 0 R >>')
        obj(2, b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>')
        obj(3, b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 756 1344] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>')
        offsets[4] = f.tell()
        f.write((f"4 0 obj\n<< /Type /XObject /Subtype /Image /Width {info['width']} /Height {info['height']} "
                 f"/ColorSpace [/Indexed /DeviceRGB 255 <{info['palette'].hex()}>] /BitsPerComponent 8 "
                 '/Interpolate false /Filter /FlateDecode '
                 f"/DecodeParms << /Predictor 15 /Colors 1 /BitsPerComponent 8 /Columns {info['width']} >> "
                 f"/Length {info['idat_bytes']} >>\nstream\n").encode())
        start = f.tell()
        for position, n in info['idat_spans']:
            src.seek(position)
            while n:
                data = read_exact(src, min(BLOCK, n))
                f.write(data)
                n -= len(data)
        f.write(b'\nendstream\nendobj\n')
        content = b'q\n756 0 0 -1344 0 1344 cm\n/Im0 Do\nQ\n'
        obj(5, f'<< /Length {len(content)} >>\nstream\n'.encode()+content+b'endstream')
        title = (b'\xfe\xff'+'边缘修复候选：全航段完整原生像素，高清竖版'.encode('utf-16-be')).hex()
        obj(6, (f'<< /Title <{title}> /Creator (OceanSim lossless candidate portrait) '
                '/Subject (Experimental edge correction. Not the pinned baseline. No clean ground truth or automatic quality acceptance. All native pixels retained.) >>').encode())
        xref = f.tell()
        f.write(b'xref\n0 7\n0000000000 65535 f \n')
        for position in offsets[1:]:
            f.write(f'{position:010d} 00000 n \n'.encode())
        f.write(f'trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    if hash_range(partial, start, info['idat_bytes']) != info['idat_SHA256']:
        raise ValueError('Embedded native compressed pixels differ.')
    if initial != (source.stat().st_size, source.stat().st_mtime_ns) or sha(source) != info['file_SHA256']:
        raise ValueError('Native image changed during packaging.')
    partial.rename(output)
    report = dict(source=str(source), source_SHA256=info['file_SHA256'],
                  output=str(output), SHA256=sha(output), bytes=output.stat().st_size,
                  embedded_original_dimensions=[info['width'], info['height']],
                  compressed_pixels_and_palette_exact=True, embedded_IDAT_SHA256=info['idat_SHA256'],
                  display_vertical_flip=True, interpolation=False, source_unchanged=True,
                  quality_accepted=False, adopted_as_default=False)
    output.with_suffix('.校验.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(package(a.source, a.output), ensure_ascii=False))
