"""Small-memory original-MAT iterator and documented supported subband control.

No private uploads, MAT modification, new downsampling, invented phases, neural
filtering or assertion that the recorded subband equals clean signal.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import mmap

import numpy as np

from BathymetrySideScanDrawing import PingFile
from SonarRecordedSubbandHypothesis import centered_filter


def supported_complex_subband(z,kernel,limits,sound_speed,sampling_frequency):
    a=np.abs(z);bounds=np.asarray(limits,np.float64)
    if z.ndim!=3 or z.shape[1]!=8 or bounds.shape!=(len(z),2) or not np.isfinite(bounds).all():
        raise ValueError('native record/eight-beam/range complex observations and supplied bounds required')
    if (bounds[:,0]<=0).any() or (bounds[:,1]<=bounds[:,0]).any():raise ValueError('reviewed supplied range limits required')
    n=z.shape[-1];radius=len(kernel)//2
    if n<=4*radius:raise ValueError('native waveform too short for finite FIR interior')
    ranges=(np.arange(n)+1)*sound_speed/(2*sampling_frequency)
    u=np.clip((ranges[None]-bounds[:,:1])/(bounds[:,1:]-bounds[:,:1]),0,1)
    depth=u*u*(3-2*u)
    x=np.arange(n);edge=np.minimum(np.clip((x-radius)/radius,0,1),np.clip((n-1-radius-x)/radius,0,1))
    edge=edge*edge*(3-2*edge)
    blend=depth[:,None,:]*edge[None,None,:]
    candidate=centered_filter(z,kernel)
    result=z+(candidate-z)*blend
    protected=(ranges[None,None,:]<=bounds[:,0,None,None])|(a<=1e-12)|(edge[None,None,:]==0)
    result=np.where(protected,z,result)
    if not np.isfinite(result).all():raise ValueError('nonfinite supported complex subband')
    return result,dict(water_protected_values=int(np.broadcast_to(ranges[None,None,:]<=bounds[:,0,None,None],z.shape).sum()),
                       near_zero_protected_values=int((a<=1e-12).sum()),
                       FIR_edge_protected_values=int(np.broadcast_to(edge[None,None,:]==0,z.shape).sum()))


def overlap_block(amplitudes,previous=None):
    a=np.asarray(amplitudes)
    if a.ndim!=4 or a.shape[:1]!=(2,) or a.shape[2]!=8 or len(a[0])<1:
        raise ValueError('dual-side native record/eight-beam/range block required')
    previous=np.zeros((2,4,a.shape[-1]),a.dtype) if previous is None else np.asarray(previous)
    if previous.shape!=(2,4,a.shape[-1]):raise ValueError('actual preceding four-beam tail required')
    rows=a[:,:,:4].copy();rows[:,0]+=previous
    rows[:,1:]+=a[:,:-1,4:]
    side=rows.reshape(2,4*a.shape[1],a.shape[-1])
    return np.concatenate((side[0,:,::-1],side[1]),axis=1),a[:,-1,4:].copy()


def overlap_tail(previous):
    a=np.asarray(previous)
    if a.ndim!=3 or a.shape[:2]!=(2,4):raise ValueError('final original four-beam tail required')
    return np.concatenate((a[0,:,::-1],a[1]),axis=1)


def original_native_blocks(cache,replica,*,block_records=16):
    ledger=json.loads((cache/'run_summary.json').read_text())
    original=np.load(cache/'原始MAT逐束包络_float64.npy',mmap_mode='r')
    if not ledger['original_all_native_samples_reproduced_exactly'] or block_records<1:
        raise ValueError('verified complete original cache and bounded record blocks required')
    for part in ledger['parts']:
        folder=replica/('下午返航_'+part['name'])
        metadata=json.loads((folder/'run_summary.json').read_text())
        indices=np.load(folder/'SourceCellIndices1Based.npy')-1
        ids=np.load(folder/'PingNum_PingS_PingUs.npy')
        if len(indices)!=part['count'] or ids.shape!=(part['count'],3):raise ValueError('saved native identity coverage differs')
        with ExitStack() as stack:
            readers=[stack.enter_context(PingFile(Path(metadata['source'])/(side+'NormalModeBeamformingTotal.mat')))
                     for side in ('Port','Stbd')]
            for start in range(0,part['count'],block_records):
                stop=min(start+block_records,part['count']);absolute=part['offset']+start
                values=np.empty((2,stop-start,8,original.shape[-1]),np.complex128)
                for local in range(start,stop):
                    index=int(indices[local])
                    for side,reader in enumerate(readers):
                        if reader.identity(index)!=tuple(ids[local]):raise ValueError('original selected physical identity differs')
                        values[side,local-start]=reader.beam_sum(index,('Port','Stbd')[side])
                expected=original[:,:,absolute:absolute+stop-start].transpose(0,2,1,3)
                if not np.array_equal(np.abs(values),expected):raise ValueError('original complex values do not reproduce every cached envelope')
                yield absolute,values
                original._mmap.madvise(mmap.MADV_DONTNEED)
