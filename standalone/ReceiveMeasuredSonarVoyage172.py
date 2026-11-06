"""Authorized-only bounded receiver for the COMPLETE measured native voyage.

Preserves all complex128 values, block receipts, and original source metadata.
No overwrite, deleted partial output, interpolation, training or fake phase.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np


def exact(stream,n):
    result = bytearray(n);view = memoryview(result);offset = 0
    while offset < n:
        count = stream.readinto(view[offset:])
        if not count:raise EOFError('incomplete measured native stream')
        offset += count
    return result


def frame(stream):
    n = struct.unpack('>I',exact(stream,4))[0]
    if not 0 < n <= 256*1024:raise ValueError('bounded JSON framing required')
    return json.loads(exact(stream,n))


def receive(root, stream, expected_source_sha):
    header = frame(stream)
    if (header['schema'] != 'authorized_172_full_measured_complex128_v1'
            or header['destination'] != '172.18.80.26' or header['dtype'] != '<c16'
            or header['source_run_summary_SHA256'] != expected_source_sha
            or header['block_layout'] != 'side, record, beam, range'
            or header['transported_original_amplitude_dtype'] != '<f8'):
        raise ValueError('explicitly authorized exact measured original protocol required')
    shape = header['native_shape']
    if (len(shape) != 4 or shape[:2] != [2,8] or min(shape[2:]) < 1
            or header['native_source_values'] != int(np.prod(shape))
            or header['native_mosaic_shape'] != [4*shape[2]+4, 2*shape[3]]):
        raise ValueError('complete native side/beam/record/range grid required')
    root.mkdir(exist_ok=False, parents=True);temp = root/'measured_complex128.incomplete.npy'
    array = np.lib.format.open_memmap(temp, mode='w+', dtype=np.complex128, shape=tuple(shape))
    amplitude_temp = root/'original_amplitude_float64.incomplete.npy'
    amplitudes = np.lib.format.open_memmap(amplitude_temp,mode='w+',dtype=np.float64,shape=tuple(shape))
    wire = hashlib.sha256();records = total = 0;blocks = [];start = time.monotonic()
    try:
        while True:
            meta = frame(stream)
            if meta['kind'] == 'completed_native_stream':break
            n = meta['count'];expected_shape = [2,n,8,shape[3]]
            if (meta['kind'] != 'native_block' or meta['start'] != records
                    or not 0 < n <= header['maximum_block_records'] <= 16 or records+n > shape[2]
                    or meta['shape'] != expected_shape or meta['payload_bytes'] != int(np.prod(expected_shape))*16
                    or meta['amplitude_bytes'] != int(np.prod(expected_shape))*8):
                raise ValueError('incomplete/duplicated/out-of-order original block')
            payload = exact(stream,meta['payload_bytes']);digest = hashlib.sha256(payload).hexdigest()
            if digest != meta['payload_SHA256']:raise ValueError('original complex block SHA differs')
            values = np.frombuffer(payload,np.complex128).reshape(expected_shape)
            if not np.isfinite(values).all():raise ValueError('finite actual complex values required')
            source_amplitudes = exact(stream,meta['amplitude_bytes'])
            if hashlib.sha256(source_amplitudes).hexdigest() != meta['amplitude_SHA256']:
                raise ValueError('original unrounded amplitude payload differs')
            observed = np.frombuffer(source_amplitudes,np.float64).reshape(expected_shape)
            np.testing.assert_allclose(abs(values),observed,rtol=2e-13,atol=1e-14)
            array[:,:,records:records+n] = values.transpose(0,2,1,3)
            amplitudes[:,:,records:records+n] = observed.transpose(0,2,1,3)
            wire.update(payload);wire.update(source_amplitudes);records += n;total += values.size;blocks.append(meta)
            if len(blocks)%16 == 0:print(json.dumps(dict(records=records,total=shape[2])),flush=True)
        if (records != shape[2] or total != int(np.prod(shape)) or meta['records'] != records
                or meta['native_values'] != total or meta['payload_bytes'] != total*24
                or meta['blocks'] != len(blocks) or meta['wire_payload_SHA256'] != wire.hexdigest()
                or not meta['every_original_native_amplitude_and_mosaic_value_checked']
                or not meta['original_MAT_guards_unchanged'] or stream.read(1)):
            raise ValueError('complete end-to-end original stream proof differs')
        array.flush();amplitudes.flush();del array,amplitudes
        final = root/'measured_complex128.npy';temp.rename(final)
        amplitude_temp.rename(root/'original_amplitude_float64.npy')
        result = dict(status='completed_authorized_full_measured_complex128_transfer', header=header,
                      trailer=meta, blocks=blocks, original_MAT_modified=False, native_grid_resampled=False,
                      authorized_private_destination='172.18.80.26 only', clean_truth=False,
                      file=str(final), payload_bytes=total*24, elapsed_s=time.monotonic()-start)
        with (root/'transfer_receipt.json').open('x') as output:output.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('status','payload_bytes','elapsed_s')}),flush=True)
        return result
    except BaseException as exc:
        with (root/'failed_transfer.json').open('x') as output:
            output.write(json.dumps(dict(error=f'{type(exc).__name__}: {exc}',complete_records=records,
                                         original_header=header,partial_outputs_preserved=True),ensure_ascii=False,indent=2)+'\n')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-source-sha',required=True)
    args = p.parse_args();os.umask(0o077)
    receive(args.output.absolute(),sys.stdin.buffer,args.expected_source_sha)


if __name__ == '__main__':main()
