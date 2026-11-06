"""Inspect public sonar ZIP inventories without downloading entire archives.

By default only HTTP byte ranges and ZIP directory metadata are read. Optional
bounded public image samples can be saved at native dimensions for inspection.
No private survey data is transmitted and no model is trained.
An HTTP 206 status alone is insufficient: the returned interval must match
the requested interval exactly before any response body is consumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import zipfile

import requests


ARCHIVES = {
    "bassurelle_raw": {
        "landing_page": "https://ckan.publishing.service.gov.uk/dataset/raw-side-scan-sonar-data-from-bassurelle-sandbanks-sci",
        "url": "https://s3-eu-west-1.amazonaws.com/data.defra.gov.uk/Marine/JNCC2013_03BassSandRawAcousticssraw_data.zip",
        "size": 51884907924,
        "license": "UK Open Government Licence; see publisher record",
        "modality": "side-scan sonar; original survey archive",
        "clean_noisy_pairs_verified": False,
    },
    "subpipe_mini2": {
        "landing_page": "https://zenodo.org/records/12666132",
        "url": "https://zenodo.org/records/12666132/files/SubPipeMini2.zip?download=1",
        "size": 4945761374,
        "license": "CC-BY-4.0; retain Oceanscan-MST and REMARO attribution",
        "modality": "real AUV side-scan images and other navigation/sensor channels",
        "clean_noisy_pairs_verified": False,
    },
    "gavia_2021": {
        "landing_page": "https://doi.org/10.6084/m9.figshare.24574879.v2",
        "url": "https://ndownloader.figshare.com/files/43168999",
        "size": 1611502,
        "license": "CC-BY-4.0",
        "modality": "real AUV side-scan image subset; object annotations are not clean targets",
        "clean_noisy_pairs_verified": False,
    },
    "chena_raw": {
        "landing_page": "https://catalog.data.gov/dataset/raw-side-scan-sonar-data-relating-to-chena-river-habitat-mapping-in-2022",
        "url": "https://iris.fws.gov/APPS/ServCat/DownloadFile/267350?Reference=178459",
        "size": 3097992165,
        "license": "public access; consult publisher metadata for reuse terms",
        "modality": "8-bit Humminbird SON records, IDX ping indices and DAT settings; not complex IQ",
        "clean_noisy_pairs_verified": False,
    },
}


def validate_interval(status, headers, start, end, total):
    expected = f"bytes {start}-{end}/{total}"
    actual = headers.get("Content-Range", "")
    if status != 206 or actual != expected:
        raise ValueError(f"server did not honor exact byte range: status={status}, interval={actual!r}, expected={expected!r}")
    length = headers.get("Content-Length")
    if length is not None and int(length) != end-start+1:
        raise ValueError("HTTP Content-Length disagrees with requested interval")


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url, size, *, session=None, budget=16*1024**2):
        super().__init__()
        if not url.startswith("https://") or size <= 0:
            raise ValueError("a known positive size and HTTPS archive URL are required")
        self.url, self.size, self.position = url, size, 0
        self.session = session or requests.Session()
        self.budget = budget
        self.bytes_read = self.request_count = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        if whence not in origin or origin[whence]+offset < 0:
            raise ValueError("invalid archive seek")
        self.position = origin[whence]+offset
        return self.position

    def read(self, n=-1):
        if self.closed:
            raise ValueError("read on closed range reader")
        available = max(0, self.size-self.position)
        n = available if n is None or n < 0 else min(n, available)
        if not n:
            return b""
        if n > 4*1024**2 or self.bytes_read+n > self.budget or self.request_count >= 64:
            raise ValueError("ZIP inspection exceeded bounded metadata-only read budget")
        start, end = self.position, self.position+n-1
        self.request_count += 1
        with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                              stream=True, timeout=(15, 35)) as response:
            validate_interval(response.status_code, response.headers, start, end, self.size)
            payload = response.raw.read(n+1)
            if len(payload) != n:
                raise ValueError("truncated or oversized HTTP byte-range body")
        self.position += n
        self.bytes_read += n
        return payload


def inspect(name, *, sample_images=0, sample_directory=None):
    metadata = ARCHIVES[name]
    report = dict(name=name, **metadata, entire_archive_downloaded=False,
                  image_or_signal_member_downloaded=False, private_data_uploaded=False,
                  denoising_ground_truth_not_inferred_from_labels=True,
                  inspector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    started = time.monotonic()
    reader = HTTPRangeReader(metadata["url"], metadata["size"])
    try:
        with zipfile.ZipFile(reader) as archive:
            files = [member for member in archive.infolist() if not member.is_dir()]
            counts = Counter(PurePosixPath(member.filename).suffix.lower() or "[no suffix]" for member in files)
            report.update(status="public_zip_directory_verified_without_member_download",
                          file_count=len(files), extensions=dict(counts),
                          uncompressed_bytes=sum(member.file_size for member in files),
                          members=[dict(path=m.filename, bytes=m.file_size,
                                        compressed_bytes=m.compress_size, crc32=f"{m.CRC:08x}",
                                        header_offset=m.header_offset, compression=m.compress_type) for m in files])
            if sample_images:
                if sample_directory is None or not 1 <= sample_images <= 4:
                    raise ValueError("one to four bounded samples and a destination are required")
                eligible = [m for m in files if PurePosixPath(m.filename).suffix.lower() in
                            (".pbm", ".bpm", ".png", ".jpg") and
                            0 < m.file_size <= 8*1024**2 and m.compress_size <= 3*1024**2]
                # Prefer one LF and one HF sonar frame over unrelated camera JPGs.
                preferred = [m for m in eligible if PurePosixPath(m.filename).suffix.lower() in (".pbm", ".bpm")]
                if preferred:
                    eligible = preferred
                selected, groups = [], sorted({str(PurePosixPath(m.filename).parent) for m in eligible})
                for group in groups:
                    candidates = sorted([m for m in eligible if str(PurePosixPath(m.filename).parent) == group], key=lambda m:m.filename)
                    if candidates:
                        selected.append(candidates[len(candidates)//2])
                    if len(selected) >= sample_images:
                        break
                if sum(m.file_size for m in selected) > 16*1024**2:
                    raise ValueError("sample uncompressed size exceeds 16 MiB")
                if selected:
                    sample_directory.mkdir(parents=True, exist_ok=False)
                report["samples"] = []
                for index, member in enumerate(selected):
                    with archive.open(member) as stream:
                        payload = stream.read(member.file_size+1)
                    if len(payload) != member.file_size:
                        raise ValueError("sample length disagrees with ZIP metadata")
                    destination = sample_directory/(str(index)+"_"+PurePosixPath(member.filename).name)
                    with destination.open("xb") as stream:
                        stream.write(payload)
                    report["samples"].append(dict(archive_path=member.filename,
                        local_path=str(destination.resolve()), bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest(),
                        zip_CRC_checked=True, native_file_not_resized=True,
                        first_32_bytes_hex=payload[:32].hex()))
                    report["image_or_signal_member_downloaded"] = True
                if selected:
                    report["status"] = "public_zip_directory_and_bounded_native_samples_verified"
                else:
                    report["sample_status"] = "no_eligible_image_members_no_samples_downloaded"
    except (ValueError, OSError, requests.RequestException, zipfile.BadZipFile, NotImplementedError) as exc:
        report.update(status="inspection_failed_no_bulk_fallback", error=f"{type(exc).__name__}: {exc}")
    finally:
        reader.session.close()
        reader.close()
    report.update(validated_HTTP_range_bytes_read=reader.bytes_read, requests=reader.request_count,
                  elapsed_s=time.monotonic()-started)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", choices=tuple(ARCHIVES), action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-images", type=int, choices=range(0, 5), default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = []
    for name in dict.fromkeys(args.archive):
        report = inspect(name, sample_images=args.sample_images,
                         sample_directory=args.output/(name+"_samples"))
        (args.output/(name+".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        concise = {k:v for k,v in report.items() if k != "members"}
        inventory.append(concise)
        print(json.dumps(concise, ensure_ascii=False), flush=True)
    (args.output/"inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
