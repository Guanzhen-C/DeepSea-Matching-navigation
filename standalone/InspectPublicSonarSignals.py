"""Fetch three bounded public survey members and inspect native JSF records.

This is data qualification, not denoising or clean-target generation. Format
offsets follow EdgeTech's public JSF file/message specification, message 80.
All metadata records are retained, including unknown message-type counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time
import zipfile

from InspectPublicSonarDatasets import ARCHIVES, HTTPRangeReader

SPEC = "https://www.edgetech.com/wp-content/uploads/2023/04/0023492_Rev_R.pdf"
MEMBERS = ("BoxA/20130328061200.jsf", "BoxA/20130328061200L.xtf", "BoxA/20130328061200H.xtf")


def inspect_jsf(payload):
    position, counts, traces = 0, Counter(), []
    while position < len(payload):
        if len(payload)-position < 16:
            raise ValueError("incomplete JSF message header")
        marker, protocol, session, kind, command, subsystem, channel, sequence, reserved, size = struct.unpack_from("<HBBHBBBBHI", payload, position)
        if marker != 0x1601 or size > len(payload)-position-16:
            raise ValueError("bad JSF sync marker or incomplete declared payload")
        counts[str(kind)] += 1
        if kind == 80:
            if size < 240:
                raise ValueError("message 80 requires a complete 240-byte trace header")
            body = memoryview(payload)[position+16:position+16+size]
            fmt = struct.unpack_from("<h", body, 34)[0]
            samples = struct.unpack_from("<H", body, 114)[0]
            msb = struct.unpack_from("<H", body, 16)[0]
            if protocol >= 0xA:
                samples |= ((msb >> 8) & 15) << 16
            bytes_per_sample = {0:2, 1:4, 2:2, 9:4}.get(fmt)
            if bytes_per_sample is not None and size != 240+bytes_per_sample*samples:
                raise ValueError("trace sample count and data format disagree with payload length")
            item = dict(file_offset=position, protocol=protocol, subsystem=subsystem,
                channel=channel, message_bytes=size,
                ping_number=struct.unpack_from("<I", body, 8)[0], data_format=fmt,
                samples=samples, stored_acoustic_bytes=size-240,
                weighting_factor=struct.unpack_from("<h", body, 168)[0],
                sample_interval_ns_integer=struct.unpack_from("<I", body, 116)[0],
                sample_interval_fraction_bits=struct.unpack_from("<H", body, 18)[0] & 255,
                start_sample=struct.unpack_from("<I", body, 4)[0],
                unix_seconds=struct.unpack_from("<i", body, 0)[0] if protocol >= 8 else None,
                milliseconds_today=struct.unpack_from("<I", body, 200)[0],
                validity_flags=struct.unpack_from("<H", body, 30)[0])
            traces.append(item)
        position += 16+size
    return dict(complete_file_consumed=position == len(payload), file_bytes=len(payload),
        message_counts=dict(counts), trace_count=len(traces),
        data_formats=dict(Counter(str(t["data_format"]) for t in traces)),
        subsystem_channel_counts=dict(Counter(f"{t['subsystem']}:{t['channel']}" for t in traces)),
        weighting_factor_values=sorted({t["weighting_factor"] for t in traces}),
        complex_IQ_present_in_sample=any(t["data_format"] in (1, 9) for t in traces),
        every_trace_is_envelope=bool(traces) and all(t["data_format"] == 0 for t in traces),
        all_sample_sizes_validated_for_public_formats=True,
        clean_reference_claimed=False, signal_reconstruction_performed=False,
        specification=SPEC, traces=traces)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_dir = root/"source_snapshot"
    source_dir.mkdir()
    for source in (Path(__file__), Path(__file__).with_name("InspectPublicSonarDatasets.py")):
        shutil.copy2(source, source_dir/source.name)
    meta = ARCHIVES["bassurelle_raw"]
    report = dict(status="reading_bounded_public_raw_samples", archive=meta,
        members=[], entire_archive_downloaded=False, private_data_uploaded=False,
        no_denoising_or_training=True, source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source_dir.iterdir()})
    reader = HTTPRangeReader(meta["url"], meta["size"])
    start = time.monotonic()
    try:
        with zipfile.ZipFile(reader) as archive:
            infos = [archive.getinfo(name) for name in MEMBERS]
            if any(not 0 < f.file_size <= 8*1024**2 or f.compress_size > 3*1024**2 for f in infos) or sum(f.file_size for f in infos) > 16*1024**2:
                raise ValueError("public signal sample download exceeds fixed size limits")
            for member in infos:
                with archive.open(member) as stream:
                    data = stream.read(member.file_size+1)
                if len(data) != member.file_size:
                    raise ValueError("ZIP member length mismatch")
                destination = root/Path(member.filename).name
                with destination.open("xb") as stream:
                    stream.write(data)
                row = dict(archive_member=member.filename, local_file=str(destination),
                    bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), crc32_checked=True,
                    first_32_bytes_hex=data[:32].hex())
                if destination.suffix == ".jsf":
                    row["JSF"] = inspect_jsf(data)
                report["members"].append(row)
                print(json.dumps({k:v for k,v in row.items() if k != "JSF"}), flush=True)
        report["status"] = "completed_public_raw_sample_qualification_not_denoising"
    except Exception as exc:
        report.update(status="failed_no_bulk_fallback", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        reader.session.close()
        reader.close()
        report.update(validated_HTTP_range_bytes_read=reader.bytes_read, requests=reader.request_count,
                      elapsed_s=time.monotonic()-start)
        (root/"run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    brief = {k:v for k,v in report.items() if k != "members"}
    brief["JSF_summaries"] = [{k:v for k,v in m["JSF"].items() if k != "traces"} for m in report["members"] if "JSF" in m]
    print(json.dumps(brief, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
