"""Rasterize the native sonar waterfall in portrait layout, streaming rows.

This is deterministic document/image-format conversion, NOT AI image editing.
All source RGB columns and rows are retained. Only rows are repeated to match
the portrait PDF display rectangle; that adds no measured resolution. Never
allocate the complete billion-pixel output in RAM or open it in an image viewer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import zlib

import numpy as np

from PackageSonarNativePortraitPDF import scan_png, read_exact


def png_chunk(stream, tag, payload):
    stream.write(struct.pack(">I", len(payload)))
    stream.write(tag)
    stream.write(payload)
    stream.write(struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff))


def rgb_rows(path, info):
    """Decode only native writer filters 0/2 with at most a few rows in RAM."""
    stride = 3 * info["width"] + 1
    decoder = zlib.decompressobj()
    pending = bytearray()
    previous = np.zeros(stride - 1, np.uint8)
    produced = 0
    with Path(path).open("rb") as stream:
        for offset, size in info["idat_spans"]:
            stream.seek(offset)
            while size:
                data = read_exact(stream, min(64 * 1024, size))
                size -= len(data)
                while True:
                    capacity = stride - len(pending)
                    decoded = decoder.decompress(data, capacity)
                    data = decoder.unconsumed_tail
                    pending.extend(decoded)
                    if len(pending) == stride:
                        method = pending[0]
                        current = np.frombuffer(pending, np.uint8, offset=1).copy()
                        if method == 2:
                            current += previous
                        elif method != 0:
                            raise ValueError("unsupported PNG row predictor")
                        previous = current
                        pending.clear()
                        produced += 1
                        if produced > info["height"]:
                            raise ValueError("too many source rows")
                        yield current.tobytes()
                    if not data and len(decoded) < capacity:
                        break
    if pending or produced != info["height"] or not decoder.eof or decoder.unused_data:
        raise ValueError("incomplete or excess PNG pixels")


def row_repetitions(source_height, output_height):
    if output_height < source_height or source_height < 1:
        raise ValueError("source rows must not be discarded")
    return [((i+1)*output_height + source_height-1)//source_height
            - (i*output_height + source_height-1)//source_height
            for i in range(source_height)]


def export(source, output, pdf_receipt=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    incomplete = output.with_suffix(".png.incomplete")
    if source == output or output.exists() or incomplete.exists():
        raise FileExistsError("refusing to overwrite an existing file")
    before = source.stat()
    original = scan_png(source)
    if pdf_receipt is not None:
        receipt = json.loads(Path(pdf_receipt).read_text())
        if receipt["source_png_sha256"] != original["png_sha256"]:
            raise ValueError("not the original image embedded in the specified PDF")
    width, height = original["width"], original["height"]
    output_height = max(height, (width*16 + 8)//9)
    counts = row_repetitions(height, output_height)
    compressor = zlib.compressobj(level=3)
    previous = np.zeros(3*width, np.uint8)
    repeated_line = b"\x02" + bytes(3*width)
    source_rgb_hash = hashlib.sha256()
    with incomplete.open("xb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        png_chunk(stream, b"IHDR", struct.pack(">IIBBBBB", width, output_height, 8, 2, 0, 0, 0))
        png_chunk(stream, b"tEXt", (f"Description\x00Portrait waterfall layout of original {width}x{height} measured image. "
                                   "All original RGB pixels retained. Rows repeated for layout only, not new resolution or AI restoration.").encode("ascii"))
        for index, (rgb, repeat) in enumerate(zip(rgb_rows(source, original), counts)):
            source_rgb_hash.update(rgb)
            current = np.frombuffer(rgb, np.uint8)
            difference = np.subtract(current, previous, dtype=np.uint8)
            filtered = b"\x02" + difference.tobytes()
            previous = current
            for line in (filtered, *([repeated_line] * (repeat-1))):
                encoded = compressor.compress(line)
                if encoded:
                    png_chunk(stream, b"IDAT", encoded)
            if index % 1024 == 0:
                print(json.dumps(dict(stage="rendering_native_rows", original_rows=index,
                                      total_original_rows=height)), flush=True)
        png_chunk(stream, b"IDAT", compressor.flush())
        png_chunk(stream, b"IEND", b"")

    # Independent complete scan: compare EVERY output RGB row to the source
    # row it must represent, including every repeated row. No full-frame decode.
    rendered = scan_png(incomplete)
    target_rows = rgb_rows(incomplete, rendered)
    checked = 0
    for rgb, repeat in zip(rgb_rows(source, original), counts):
        for _ in range(repeat):
            if next(target_rows) != rgb:
                raise ValueError("portrait pixels do not match original")
            checked += 1
    if next(target_rows, None) is not None or checked != output_height:
        raise ValueError("output coverage mismatch")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed")
    incomplete.rename(output)
    return dict(status="completed_all_source_RGB_samples_preserved_in_portrait_PNG_layout",
                source=str(source), source_png_sha256=original["png_sha256"],
                corresponding_PDF_receipt=str(pdf_receipt) if pdf_receipt else None,
                original_grid_width_height=[width, height], original_samples=width*height,
                output_width_height=[width, output_height], output=str(output), output_bytes=output.stat().st_size,
                output_pixels=width*output_height, approximate_RGBA_decode_bytes=width*output_height*4,
                source_row_repetitions_min_max=[min(counts), max(counts)],
                all_output_rows_pixel_exact_checked=checked, all_source_rows_and_columns_preserved=True,
                downsampled=False, rotated=False, source_unchanged=True,
                vertical_row_replication_is_layout_expansion_only=True, measured_resolution_increased=False,
                AI_denoising_or_generation_performed=False, full_image_display_not_attempted=True,
                original_RGB_sha256=source_rgb_hash.hexdigest(), output_png_sha256=rendered["png_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pdf-receipt", type=Path)
    args = parser.parse_args()
    report = export(args.source, args.output, args.pdf_receipt)
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
