"""Place a complete native RGB PNG into a portrait PDF without re-encoding it.

This is document layout, NOT image editing or image generation. PNG IDAT bytes
are copied unchanged to a PDF FlateDecode image with PNG prediction. Every
source pixel remains embedded; only its display rectangle changes to the
user-requested portrait waterfall layout. Neither axis is a georectified metre
axis. No rotation, cropping, spatial resampling, denoising or upscaling occurs.

Reference: Adobe PDF Reference 1.5, section 3.3, FlateDecode PNG predictors.
https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.5_v6.pdf
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import zlib


BLOCK = 1024 * 1024


def read_exact(stream, length):
    value = stream.read(length)
    if len(value) != length:
        raise ValueError("truncated PNG")
    return value


def scan_png(path):
    """Validate CRCs and record IDAT spans with bounded memory, no decoding."""
    spans = []
    digest = hashlib.sha256()
    image_digest = hashlib.sha256()
    width = height = None
    seen_idat = closed_idat = False
    with Path(path).open("rb") as stream:
        signature = read_exact(stream, 8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError("not a PNG")
        digest.update(signature)
        count = 0
        while True:
            header = read_exact(stream, 8)
            digest.update(header)
            length, tag = struct.unpack(">I4s", header)
            if count == 0 and (tag != b"IHDR" or length != 13):
                raise ValueError("IHDR must be first")
            count += 1
            if tag == b"IDAT":
                if closed_idat:
                    raise ValueError("nonconsecutive PNG IDAT chunks")
                seen_idat = True
                spans.append((stream.tell(), length))
            elif seen_idat:
                closed_idat = True
            if tag in (b"tRNS", b"iCCP", b"acTL", b"fcTL", b"fdAT"):
                raise ValueError("unsupported transparency, ICC profile or animation")
            crc = zlib.crc32(tag)
            remaining = length
            captured = bytearray()
            while remaining:
                block = read_exact(stream, min(BLOCK, remaining))
                crc = zlib.crc32(block, crc)
                digest.update(block)
                if tag == b"IDAT":
                    image_digest.update(block)
                elif tag == b"IHDR":
                    captured.extend(block)
                remaining -= len(block)
            trailer = read_exact(stream, 4)
            digest.update(trailer)
            if (crc & 0xffffffff) != struct.unpack(">I", trailer)[0]:
                raise ValueError("PNG CRC mismatch")
            if tag == b"IHDR":
                if width is not None or length != 13:
                    raise ValueError("invalid duplicate IHDR")
                width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", captured)
                if min(width, height) < 1 or (bits, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                    raise ValueError("requires noninterlaced 8-bit RGB PNG")
            if tag == b"IEND":
                if length or not spans or stream.read(1):
                    raise ValueError("invalid PNG end or missing image data")
                break
    return dict(width=width, height=height, idat_spans=spans,
                idat_bytes=sum(length for _, length in spans),
                png_sha256=digest.hexdigest(), idat_sha256=image_digest.hexdigest(),
                all_png_chunk_crcs_verified=True)


def hash_range(path, start, length):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        stream.seek(start)
        while length:
            block = stream.read(min(BLOCK, length))
            if not block:
                raise ValueError("truncated embedded PDF image")
            digest.update(block)
            length -= len(block)
    return digest.hexdigest()


def package(source, output, page_width=756, page_height=1344):
    source, output = Path(source).resolve(), Path(output).resolve()
    incomplete = output.with_suffix(output.suffix + ".incomplete")
    if source == output or output.exists() or incomplete.exists():
        raise FileExistsError("refusing to replace existing input/output")
    if not 0 < page_width < page_height:
        raise ValueError("portrait page dimensions required")
    before = source.stat()
    png = scan_png(source)
    offsets = [0] * 7
    with incomplete.open("xb") as stream, source.open("rb") as image:
        stream.write(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")

        def obj(index, content):
            offsets[index] = stream.tell()
            stream.write(f"{index} 0 obj\n".encode() + content + b"\nendobj\n")

        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        obj(3, (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
                "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode())
        offsets[4] = stream.tell()
        stream.write((f"4 0 obj\n<< /Type /XObject /Subtype /Image /Width {png['width']} "
                      f"/Height {png['height']} /ColorSpace /DeviceRGB /BitsPerComponent 8 "
                      "/Interpolate false /Filter /FlateDecode "
                      f"/DecodeParms << /Predictor 15 /Colors 3 /BitsPerComponent 8 /Columns {png['width']} >> "
                      f"/Length {png['idat_bytes']} >>\nstream\n").encode())
        image_start = stream.tell()
        for offset, length in png["idat_spans"]:
            image.seek(offset)
            while length:
                block = read_exact(image, min(BLOCK, length))
                stream.write(block)
                length -= len(block)
        stream.write(b"\nendstream\nendobj\n")
        content = f"q\n{page_width} 0 0 {page_height} 0 0 cm\n/Im0 Do\nQ\n".encode()
        obj(5, f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream")
        title = (b"\xfe\xff" + "原始实测完整航次：全部原生像素，竖向瀑布图".encode("utf-16-be")).hex().encode()
        obj(6, b"<< /Title <" + title + b"> /Creator (OceanSim lossless native PNG portrait layout) "
               b"/Subject (Original measured image. Portrait presentation only, not equal-metre georectification or AI restoration.) >>")
        xref = stream.tell()
        stream.write(b"xref\n0 7\n0000000000 65535 f \n")
        for position in offsets[1:]:
            stream.write(f"{position:010d} 00000 n \n".encode())
        stream.write(f"trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    embedded_hash = hash_range(incomplete, image_start, png["idat_bytes"])
    if embedded_hash != png["idat_sha256"]:
        raise ValueError("embedded image bytes differ from original PNG")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed during packaging")
    incomplete.rename(output)
    return dict(status="completed_portrait_PDF_original_pixel_stream_preserved",
                source=str(source), source_png_sha256=png["png_sha256"],
                output=str(output), output_bytes=output.stat().st_size,
                embedded_image_width=png["width"], embedded_image_height=png["height"],
                embedded_original_pixels=png["width"] * png["height"],
                page_size_points=[page_width, page_height], portrait_waterfall_presentation=True,
                embedded_image_stream_offset=image_start, embedded_image_stream_bytes=png["idat_bytes"],
                original_png_IDAT_sha256=png["idat_sha256"], embedded_pdf_image_sha256=embedded_hash,
                original_compressed_pixels_exactly_preserved=True,
                all_source_png_chunk_CRCs_verified=True, source_size_and_mtime_unchanged=True,
                rotated=False, cropped=False, spatially_resampled=False, upscaled_image_created=False,
                AI_denoising_or_generation_performed=False, equal_metre_geometry_claimed=False,
                note="Layout-only portrait rendering of the original measured PNG, not an AI-restored image.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = package(args.source, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
