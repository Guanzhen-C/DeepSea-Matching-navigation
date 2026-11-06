"""Bounded lossless original-MAT stream to the user's authorized 172 host.

Every native complex two-ULA sum is checked against the established envelope
cache and every original mosaic value against the previous numeric mosaic.
No persistent local array, resampling, filtering, training or MAT edits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import mmap
import struct
import sys

import numpy as np

from RecordedSubbandNativeStream import original_native_blocks, overlap_block, overlap_tail
from NativeSonarPaletteStream import amplitude_indices


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):digest.update(block)
    return digest.hexdigest()


def guard(path):
    s = Path(path).stat();return dict(size=s.st_size, mtime_ns=s.st_mtime_ns)


def frame(value):
    encoded = json.dumps(value, ensure_ascii=False).encode()
    sys.stdout.buffer.write(struct.pack('>I', len(encoded)))
    sys.stdout.buffer.write(encoded)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    args = p.parse_args();root = args.source_run.absolute()
    report = json.loads((root/'run_summary.json').read_text())
    audit = json.loads((root/'independent_native_replay_audit.json').read_text())
    if (report['status'] != 'completed_full_native_recorded_subband_candidate_not_quality_acceptance'
            or audit['status'] != 'passed_complete_original_MAT_replay_and_native_PNG_not_quality_acceptance'
            or audit['source_run_summary_SHA256'] != sha(root/'run_summary.json')):
        raise ValueError('completed fully replayed original source required')
    for name in ('RecordedSubbandNativeStream.py', 'NativeSonarPaletteStream.py', 'BathymetrySideScanDrawing.py'):
        if sha(Path(__file__).with_name(name)) != report['source_code_sha256'][name]:
            raise ValueError('previous original reconstruction implementation changed: '+name)
    guards = report['original_MAT_guards']
    if any(guard(path) != value for path,value in guards.items()):raise ValueError('original MAT guard changed')
    reference = np.load(report['reference_mosaic'], mmap_mode='r')
    if list(reference.shape) != report['native_mosaic_shape']:raise ValueError('complete original mosaic required')
    frame(dict(schema='authorized_172_full_measured_complex128_v1', destination='172.18.80.26',
               native_shape=report['native_shape'], dtype='<c16', native_mosaic_shape=list(reference.shape),
               source_run_summary_SHA256=sha(root/'run_summary.json'), source_audit_SHA256=sha(root/'independent_native_replay_audit.json'),
               source_code_SHA256=sha(Path(__file__)), original_MAT_guards=guards,
               source_layout='two sides, eight synchronous beams, physical records, range samples',
               block_layout='side, record, beam, range', maximum_block_records=16,
               native_source_values=int(np.prod(report['native_shape'])), original_clean_truth=False,
               transported_original_amplitude_dtype='<f8',
               export_runtime=dict(python=sys.version.split()[0],numpy=np.__version__)))
    wire = hashlib.sha256();indices = hashlib.sha256();mosaic = hashlib.sha256()
    blocks = records = written = total = 0;previous = None
    def consume(raw):
        nonlocal written
        expected = reference[written:written+len(raw)]
        if not np.array_equal(raw, expected):raise ValueError('every original full native mosaic value must match')
        indices.update(memoryview(amplitude_indices(raw)).cast('B'))
        mosaic.update(memoryview(np.ascontiguousarray(raw)).cast('B'))
        written += len(raw);reference._mmap.madvise(mmap.MADV_DONTNEED)
    for position,z in original_native_blocks(Path(report['envelope_cache']),Path(report['replica_root']),block_records=16):
        if position != records or z.dtype != np.complex128:raise ValueError('contiguous exact original records required')
        amplitude = abs(z)
        payload = memoryview(z).cast('B');digest = hashlib.sha256(payload).hexdigest()
        amplitude_bytes = memoryview(amplitude).cast('B')
        frame(dict(kind='native_block', start=position, count=z.shape[1], shape=list(z.shape),
                   payload_bytes=len(payload), payload_SHA256=digest,
                   amplitude_bytes=len(amplitude_bytes),amplitude_SHA256=hashlib.sha256(amplitude_bytes).hexdigest()))
        for offset in range(0,len(payload),4*1024**2):sys.stdout.buffer.write(payload[offset:offset+4*1024**2])
        wire.update(payload)
        for offset in range(0,len(amplitude_bytes),4*1024**2):sys.stdout.buffer.write(amplitude_bytes[offset:offset+4*1024**2])
        wire.update(amplitude_bytes);blocks += 1;records += z.shape[1];total += z.size
        rows,previous = overlap_block(amplitude,previous);consume(rows)
        if blocks % 16 == 0:print(f'original native export: {records}/{report["native_shape"][2]} records', file=sys.stderr, flush=True)
    consume(overlap_tail(previous))
    if total != int(np.prod(report['native_shape'])) or written != len(reference):raise ValueError('complete native coverage required')
    if any(guard(path) != value for path,value in guards.items()):raise ValueError('MAT changed during export')
    frame(dict(kind='completed_native_stream', blocks=blocks, records=records, native_values=total,
               payload_bytes=total*24, complex_payload_bytes=total*16, amplitude_payload_bytes=total*8,
               wire_payload_SHA256=wire.hexdigest(),
               original_forward_mosaic_index_SHA256=indices.hexdigest(),
               original_forward_float64_mosaic_SHA256=mosaic.hexdigest(),
               every_original_native_amplitude_and_mosaic_value_checked=True, original_MAT_guards_unchanged=True))
    sys.stdout.buffer.flush()


if __name__ == '__main__':main()
