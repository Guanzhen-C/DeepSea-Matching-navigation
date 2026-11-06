"""Checksum-guarded original processed MAT to LOCAL complex128 binary pipe.

No new filtering, hidden PingNum drops, resize, persistent arrays or uploads.
The optional single-beam mode still reads the original complex two-ULA sum.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import sys

import numpy as np

from AuditSonarTrainingDomains import verified_source_indices, guard, sha
from BathymetrySideScanDrawing import PingFile


def extract(manifest, source_index, context, *, beam=None):
    root = Path(manifest)
    report = json.loads(root.read_text())
    if report['status'] != 'prepared_native_processed_observation_teacher_manifest_not_training':
        raise ValueError('qualified local processed-observation manifest required')
    sources = report['sources']
    if not 0 <= source_index < len(sources) or (beam is not None and not 0 <= beam < 8):
        raise ValueError('actual qualified source and original synchronous beam required')
    source = sources[source_index]; path = Path(source['source_MAT'])
    if guard(path) != source['source_guard'] or sha(source['parameter_file']) != source['parameter_SHA256']:
        raise ValueError('original source or parameter guard changed')
    identities_path = Path(source['selected_identity_file'])
    if sha(identities_path) != source['selected_identity_SHA256']:
        raise ValueError('original saved physical identities changed')
    identities = np.load(identities_path)
    top, left, bottom, right = map(int, context)
    if (not 0 <= top < bottom <= source['native_records']
            or not 0 <= left < right <= source['native_range_bins'] or top % 4 or left % 4):
        raise ValueError('factor-four native context and actual original bounds required')
    first = max(0, left-1)
    beams = list(range(8)) if beam is None else [beam]
    z = np.empty((len(beams), bottom-top, right-first), np.complex128)
    with PingFile(path) as reader:
        saved = source['original_saved_indices_file']
        index_path = Path(saved) if saved else identities_path.with_name('SourceCellIndices1Based.npy')
        indices, index_meta = verified_source_indices(reader, identities, index_path)
        if index_meta['zero_based_selected_indices_sha256'] != source['selected_zero_based_original_index_SHA256']:
            raise ValueError('qualified original source-record selection changed')
        for ordinal in range(top, bottom):
            index = int(indices[ordinal])
            if reader.identity(index) != tuple(identities[ordinal]):
                raise ValueError('native physical identity changed')
            data = reader.beam_sum(index, source['side'])
            if data.shape != (8, source['native_range_bins']):
                raise ValueError('original range/beam sampling grid changed')
            z[:, ordinal-top] = data[beams, first:right]
    if guard(path) != source['source_guard'] or not np.isfinite(z).all():
        raise ValueError('source changed during extraction or nonfinite original observations')
    metadata = dict(source_index=source_index, source_MAT=str(path), source_guard=source['source_guard'],
        physical_part=source['physical_part'], side=source['side'], split=source['split'], beams_0based=beams,
        native_context=[top,left,bottom,right], stored_columns_start=first,
        actual_left_predecessor_included=left > 0, native_shape=list(z.shape), dtype=z.dtype.str,
        array_bytes=z.nbytes, array_SHA256=hashlib.sha256(memoryview(z).cast('B')).hexdigest(),
        manifest_SHA256=sha(root), parameter_SHA256=source['parameter_SHA256'],
        selected_indices_SHA256=index_meta['zero_based_selected_indices_sha256'],
        context_identity_SHA256=hashlib.sha256(identities[top:bottom].tobytes()).hexdigest(),
        is_clean_truth=False, raw_MAT_modified=False, private_data_uploaded=False,
        runtime=dict(python=sys.version.split()[0],numpy=np.__version__,interpreter=sys.executable))
    return z, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--source-index', type=int, required=True)
    p.add_argument('--context', type=int, nargs=4, required=True)
    p.add_argument('--beam', type=int)
    a = p.parse_args()
    z, metadata = extract(a.manifest, a.source_index, a.context, beam=a.beam)
    header = json.dumps(metadata, ensure_ascii=False).encode()
    out = sys.stdout.buffer; out.write(struct.pack('>I', len(header))); out.write(header)
    data = memoryview(z).cast('B')
    for start in range(0, len(data), 4*1024**2):
        out.write(data[start:start+4*1024**2])
    out.flush()


if __name__ == '__main__':
    main()
