"""Bounded all-eight-beam local pipe; no eight repeated reads of one MAT.

Reads straight into one native complex128 allocation instead of concatenating
gigabytes of bytes. Complete validation sides fit the explicit 1.5 GiB bound.
No private arrays are stored or transmitted off-machine. A single-beam view's
metadata describes exactly the same bytes as the old single-beam extractor.
"""
from pathlib import Path
import hashlib
import json
import struct
import subprocess

import numpy as np


def read_exact(stream,count):
    data=bytearray(count);view=memoryview(data);offset=0
    while offset<count:
        n=stream.readinto(view[offset:])
        if not n:raise EOFError('incomplete local native binary header')
        offset+=n
    return data


def stream_side(python,manifest,source_index,context):
    command=[str(Path(python).absolute()),str(Path(__file__).with_name('ExtractNativeProcessedSonarContext.py')),
        '--manifest',str(manifest),'--source-index',str(source_index),'--context',*map(str,context)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        size=struct.unpack('>I',read_exact(process.stdout,4))[0]
        if not 0<size<128*1024:raise ValueError('bounded native header required')
        meta=json.loads(read_exact(process.stdout,size));shape=meta['native_shape']
        if (meta['dtype']!='<c16' or len(shape)!=3 or shape[0]!=8 or min(shape)<1
                or meta['beams_0based']!=list(range(8)) or meta['native_context']!=list(context)
                or meta['source_index']!=source_index or np.prod(shape)*16>1536*1024**2):
            raise ValueError('bounded complete native eight-beam side required')
        z=np.empty(shape,np.complex128);buffer=memoryview(z).cast('B')
        if len(buffer)!=meta['array_bytes']:raise ValueError('native byte count changed')
        offset=0
        while offset<len(buffer):
            n=process.stdout.readinto(buffer[offset:offset+4*1024**2])
            if not n:raise EOFError('incomplete all-beam complex128 local payload')
            offset+=n
        if hashlib.sha256(buffer).hexdigest()!=meta['array_SHA256'] or process.stdout.read(1):
            raise ValueError('complete native all-beam payload integrity differs')
        error=process.stderr.read().decode(errors='replace');status=process.wait()
        if status:raise RuntimeError('original native extraction failed: '+error[-2000:])
        return z,meta
    except BaseException:
        process.stdout.close();error=process.stderr.read().decode(errors='replace');process.wait()
        if error:raise RuntimeError('local native stream extraction: '+error[-2000:])
        raise


def beam_view(z,metadata,beam):
    if (z.shape[0]!=8 or not 0<=beam<8 or metadata['beams_0based']!=list(range(8))
            or metadata['native_shape']!=list(z.shape)):
        raise ValueError('actual complete eight-beam observation buffer required')
    view=z[beam:beam+1]
    meta=dict(metadata)
    meta.update(beams_0based=[beam],native_shape=list(view.shape),array_bytes=view.nbytes,
        array_SHA256=hashlib.sha256(memoryview(view).cast('B')).hexdigest())
    return view,meta
