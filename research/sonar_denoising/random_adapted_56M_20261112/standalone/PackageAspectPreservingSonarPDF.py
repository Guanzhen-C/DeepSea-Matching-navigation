"""Wrap a native indexed PNG in a PDF without resampling its stored pixels.

Document conversion only: copy the PNG's compressed pixels and palette exactly.
Default layout preserves a supplied portrait's aspect ratio. Explicit waterfall
layout instead displays range columns horizontally and ping rows vertically on
the established portrait page, without rotating those two acquisition axes.
Waterfall axis display scaling is NOT an equal-metre georectification.
"""
from pathlib import Path
import argparse
import json
import math

from ExportPinnedFullSamplePortrait import indexed_info, sha, BLOCK
from PackageSonarNativePortraitPDF import read_exact, hash_range


def package(source, output, *, title="声呐处理结果：原生高清竖版", page_width=595.275590551,
            layout="source", page_height=None, flip_vertical=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    partial = output.with_suffix(output.suffix + ".incomplete")
    report_path = output.with_suffix(".校验.json")
    if source == output or output.exists() or partial.exists() or report_path.exists():
        raise FileExistsError("Preserve existing source, output and verification files.")
    if not math.isfinite(page_width) or page_width <= 0:
        raise ValueError("Finite positive page width required.")
    if layout not in ("source", "waterfall"):
        raise ValueError("Choose source or waterfall layout explicitly.")
    if layout == "source" and (page_height is not None or flip_vertical):
        raise ValueError("Source layout preserves aspect ratio and supplied orientation.")
    initial = (source.stat().st_size, source.stat().st_mtime_ns)
    info = indexed_info(source)
    if layout == "source" and info["height"] <= info["width"]:
        raise ValueError("Supply the already-oriented portrait PNG; no automatic rotation.")
    if layout == "source":
        page_height = page_width * info["height"] / info["width"]
    elif page_height is None:
        page_height = page_width * 16 / 9
    if not math.isfinite(page_height) or page_height <= page_width:
        raise ValueError("Finite portrait page height larger than page width required.")
    if max(page_width, page_height) > 14400:
        raise ValueError("Page exceeds PDF limits; use a smaller display page width.")
    pw, ph = f"{page_width:.9f}", f"{page_height:.9f}"
    offsets = [0] * 7
    with partial.open("xb") as stream, source.open("rb") as image:
        stream.write(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")

        def obj(index, content):
            offsets[index] = stream.tell()
            stream.write(f"{index} 0 obj\n".encode() + content + b"\nendobj\n")

        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        obj(3, (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw} {ph}] "
                "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode())
        offsets[4] = stream.tell()
        stream.write((f"4 0 obj\n<< /Type /XObject /Subtype /Image /Width {info['width']} "
                      f"/Height {info['height']} "
                      f"/ColorSpace [/Indexed /DeviceRGB 255 <{info['palette'].hex()}>] "
                      "/BitsPerComponent 8 /Interpolate false /Filter /FlateDecode "
                      f"/DecodeParms << /Predictor 15 /Colors 1 /BitsPerComponent 8 /Columns {info['width']} >> "
                      f"/Length {info['idat_bytes']} >>\nstream\n").encode())
        image_start = stream.tell()
        for position, length in info["idat_spans"]:
            image.seek(position)
            while length:
                block = read_exact(image, min(BLOCK, length))
                stream.write(block)
                length -= len(block)
        stream.write(b"\nendstream\nendobj\n")
        matrix = f"{pw} 0 0 -{ph} 0 {ph}" if flip_vertical else f"{pw} 0 0 {ph} 0 0"
        content = f"q\n{matrix} cm\n/Im0 Do\nQ\n".encode()
        obj(5, f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream")
        encoded_title = (b"\xfe\xff" + title.encode("utf-16-be")).hex()
        subject = (
            "Exact supplied PNG pixels and palette. Waterfall layout: horizontal range columns, "
            "vertical ping rows. Axis display scaling and optional vertical flip only; no "
            "resampling, rotation, equal-metre calibration or quality acceptance."
            if layout == "waterfall" else
            "Exact supplied PNG pixels and palette. No resizing, flipping, denoising "
            "or quality acceptance is performed by this document conversion."
        )
        obj(6, (f"<< /Title <{encoded_title}> /Creator (OceanSim lossless native PDF) "
                f"/Subject ({subject}) >>").encode())
        xref = stream.tell()
        stream.write(b"xref\n0 7\n0000000000 65535 f \n")
        for offset in offsets[1:]:
            stream.write(f"{offset:010d} 00000 n \n".encode())
        stream.write(f"trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    embedded_hash = hash_range(partial, image_start, info["idat_bytes"])
    if embedded_hash != info["idat_SHA256"]:
        raise ValueError("Embedded PNG compressed pixels differ.")
    if initial != (source.stat().st_size, source.stat().st_mtime_ns) or sha(source) != info["file_SHA256"]:
        raise ValueError("Source changed during document conversion.")
    partial.rename(output)
    report = dict(
        status=("completed_lossless_unrotated_waterfall_PDF" if layout == "waterfall" else
                "completed_lossless_aspect_preserving_portrait_PDF"),
        layout=layout,
        source=str(source), source_SHA256=info["file_SHA256"],
        output=str(output), output_SHA256=sha(output), output_bytes=output.stat().st_size,
        embedded_dimensions=[info["width"], info["height"]],
        page_size_points=[float(pw), float(ph)],
        source_aspect_ratio=info["width"] / info["height"],
        page_aspect_ratio=float(pw) / float(ph),
        image_stream_offset=image_start, image_stream_bytes=info["idat_bytes"],
        source_IDAT_SHA256=info["idat_SHA256"], embedded_IDAT_SHA256=embedded_hash,
        original_compressed_pixels_and_palette_exact=True,
        source_unchanged=True, rotated=False, flipped=bool(flip_vertical), cropped=False,
        resampled=False, interpolation=False, extra_denoising=False,
        axis_display_scaling_changed_for_waterfall_layout=layout == "waterfall",
        range_columns_horizontal=layout == "waterfall",
        ping_rows_vertical=layout == "waterfall",
        equal_metre_geometry_claimed=False, additional_measured_resolution=False,
        quality_accepted=False,
    )
    with report_path.open("x") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
        report_file.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="声呐处理结果：原生高清竖版")
    parser.add_argument("--layout", choices=("source", "waterfall"), default="source")
    parser.add_argument("--page-width", type=float, default=595.275590551)
    parser.add_argument("--page-height", type=float)
    parser.add_argument("--flip-vertical", action="store_true")
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.output, title=args.title, layout=args.layout,
                             page_width=args.page_width, page_height=args.page_height,
                             flip_vertical=args.flip_vertical), ensure_ascii=False))
