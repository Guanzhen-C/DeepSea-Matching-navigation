"""Lossless native complex128 LOCAL-to-authorized-172 stdin receiver.

Reads the extractor's exact binary protocol into a fresh .npy file, checks all
payload bytes/identities and preserves the original immutable source metadata.
No raw-data output on stdout, overwrite, target relabeling or input resizing.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np


def read_exact(stream,n):
    data=bytearray(n);view=memoryview(data);offset=0
    while offset<n:
        count=stream.readinto(view[offset:])
        if not count:raise EOFError('incomplete original native transfer header')
        offset+=count
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source-index',type=int,required=True)
    args=p.parse_args();os.umask(0o077);root=args.bundle.absolute()
    bundle=json.loads((root/'bundle_manifest.json').read_text())
    source=next(s for s in bundle['native_sources'] if s['source_index']==args.source_index)
    folder=root/'native_arrays';folder.mkdir(exist_ok=True)
    final=folder/f'source_{args.source_index:02d}_complex128.npy'
    temp=folder/f'source_{args.source_index:02d}_complex128.incomplete.npy'
    receipt=folder/f'source_{args.source_index:02d}_receipt.json'
    if any(path.exists() for path in (final,temp,receipt)):raise FileExistsError('preserve existing native transfer')
    incoming=sys.stdin.buffer;size=struct.unpack('>I',read_exact(incoming,4))[0]
    if not 0<size<128*1024:raise ValueError('bounded original native header required')
    meta=json.loads(read_exact(incoming,size))
    if (meta['source_index']!=args.source_index or meta['native_shape']!=source['shape']
            or meta['manifest_SHA256']!=bundle['source_manifest_SHA256'] or meta['dtype']!='<c16'
            or meta['beams_0based']!=list(range(8)) or meta['source_guard']!=source['source_guard']
            or meta['native_context']!=[0,0,source['shape'][1],source['shape'][2]]):
        raise ValueError('complete original qualified side/beam/range provenance differs')
    values=np.lib.format.open_memmap(temp,mode='w+',dtype=np.complex128,shape=tuple(meta['native_shape']))
    view=memoryview(values).cast('B');digest=hashlib.sha256();offset=0
    if len(view)!=meta['array_bytes']:raise ValueError('original native byte count differs')
    while offset<len(view):
        count=incoming.readinto(view[offset:offset+4*1024**2])
        if not count:raise EOFError('incomplete native payload preserved for diagnosis')
        digest.update(view[offset:offset+count]);offset+=count
    if digest.hexdigest()!=meta['array_SHA256'] or incoming.read(1):raise ValueError('complete transferred payload SHA/length differs')
    values.flush();del view,values;temp.rename(final)
    result=dict(status='completed_exact_authorized_native_complex128_transfer',
        destination=str(final),source_index=args.source_index,source_extractor_metadata=meta,
        source_payload_SHA256=digest.hexdigest(),payload_bytes=offset,
        original_MAT_moved_or_modified=False,user_authorized_private_destination='172.18.80.26 only',
        native_pixels_resized=False,source_is_clean_truth=False)
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','source_index','payload_bytes','source_payload_SHA256')}),flush=True)


if __name__=='__main__':main()
