"""Extract actual original/subband complex contexts through a LOCAL RAM pipe.

Runs in the existing h5py environment. No large temporary files, raw-data
uploads, invented phases, input resizing or change to completed full processing.
The ML environment receives lossless complex128 bytes, not screenshots.
"""
from pathlib import Path
import argparse
import hashlib
import json
import mmap
import struct
import sys

import numpy as np

from BathymetrySideScanDrawing import PingFile
from RecordedSubbandNativeStream import supported_complex_subband
from SonarRecordedSubbandHypothesis import complex_bandpass


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def guard(path):
    s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns)


def extract(source_run,side,context):
    root=Path(source_run);report=json.loads((root/'run_summary.json').read_text())
    audit=json.loads((root/'independent_native_replay_audit.json').read_text())
    if (report['status']!='completed_full_native_recorded_subband_candidate_not_quality_acceptance'
            or audit['status']!='passed_complete_original_MAT_replay_and_native_PNG_not_quality_acceptance'
            or audit['source_run_summary_SHA256']!=sha(root/'run_summary.json')):
        raise ValueError('completed and fully independently replayed source preprocessing required')
    for name in ('RecordedSubbandNativeStream.py','SonarRecordedSubbandHypothesis.py','BathymetrySideScanDrawing.py'):
        if sha(Path(__file__).with_name(name))!=report['source_code_sha256'][name]:
            raise ValueError('completed full-source processing implementation changed')
    top,left,bottom,right=map(int,context);count,bins=report['native_shape'][-2:]
    if side not in (0,1) or not 0<=top<bottom<=count or not 0<=left<right<=bins or top%4 or left%4:
        raise ValueError('aligned model context must lie within the original native grid')
    guards=report['original_MAT_guards']
    if any(guard(path)!=value for path,value in guards.items()):raise ValueError('original source MAT changed')
    cache=Path(report['envelope_cache']);replica=Path(report['replica_root'])
    ledger=json.loads((cache/'run_summary.json').read_text())
    envelope=np.load(cache/'原始MAT逐束包络_float64.npy',mmap_mode='r')
    bounds=np.load(cache/'bottom_first_ranges.npy')
    c=ledger['parts'][0]['parameters']['c'];fs=report['sampling_frequency_Hz']
    h=complex_bandpass(fs,report['recorded_comparison_subband_Hz'])
    first=left-1 if left else 0
    arrays=[np.empty((8,bottom-top,right-first),np.complex128) for _ in range(2)]
    seen=np.zeros(bottom-top,bool);side_name=('Port','Stbd')[side];identities=[]
    for part in ledger['parts']:
        begin=max(top,part['offset']);end=min(bottom,part['offset']+part['count'])
        if begin>=end:continue
        folder=replica/('下午返航_'+part['name'])
        metadata=json.loads((folder/'run_summary.json').read_text())
        indices=np.load(folder/'SourceCellIndices1Based.npy')-1
        ids=np.load(folder/'PingNum_PingS_PingUs.npy')
        path=Path(metadata['source'])/(side_name+'NormalModeBeamformingTotal.mat')
        with PingFile(path) as reader:
            for absolute in range(begin,end):
                local=absolute-part['offset'];index=int(indices[local])
                identity=reader.identity(index)
                if identity!=tuple(ids[local]):raise ValueError('original physical record identity differs')
                z=reader.beam_sum(index,side_name)
                if not np.array_equal(abs(z),envelope[side,:,absolute]):raise ValueError('original complex magnitude differs from full verified cache')
                filtered,_=supported_complex_subband(z[None],h,bounds[side,absolute:absolute+1],c,fs)
                arrays[0][:,absolute-top]=z[:,first:right]
                arrays[1][:,absolute-top]=filtered[0,:,first:right]
                seen[absolute-top]=True;identities.append(list(identity))
                if absolute%16==0:envelope._mmap.madvise(mmap.MADV_DONTNEED)
    if not seen.all() or any(guard(path)!=value for path,value in guards.items()):
        raise ValueError('context is incomplete or source changed during extraction')
    metadata=dict(source_run_summary_SHA256=sha(root/'run_summary.json'),side=side,
        model_context=[top,left,bottom,right],stored_columns_start=first,
        actual_left_predecessor_included=left>0,native_shape=list(arrays[0].shape),dtype=arrays[0].dtype.str,
        array_bytes=[a.nbytes for a in arrays],array_SHA256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
        identities_SHA256=hashlib.sha256(np.asarray(identities,np.int64).tobytes()).hexdigest(),
        all_context_record_identities_and_full_range_raw_amplitudes_checked=True,
        full_original_range_trace_filtered_before_context_crop=True,
        original_sources_unchanged=True,private_data_transferred_off_machine=False,
        extraction_runtime=dict(interpreter=sys.executable,python=sys.version.split()[0],numpy=np.__version__))
    return arrays,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--side',type=int,required=True)
    p.add_argument('--context',type=int,nargs=4,required=True);args=p.parse_args()
    arrays,metadata=extract(args.source_run,args.side,args.context)
    encoded=json.dumps(metadata,ensure_ascii=False).encode('utf-8')
    out=sys.stdout.buffer;out.write(struct.pack('>I',len(encoded)));out.write(encoded)
    for a in arrays:
        data=memoryview(a).cast('B')
        for start in range(0,len(data),4*1024**2):out.write(data[start:start+4*1024**2])
    out.flush()


if __name__=='__main__':main()
