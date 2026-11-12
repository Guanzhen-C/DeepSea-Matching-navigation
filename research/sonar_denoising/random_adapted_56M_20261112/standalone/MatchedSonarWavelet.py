"""Explicit composed preprocessing + fixed-phase Haar restoration protocol.

The two-level neural graph is unchanged. A condition's front end is applied
identically in training, native validation and real diagnostic inference. Total
residuals include preprocessing; neural residuals are recorded separately so a
selected zero-head baseline cannot be misreported as learned denoising success.
"""
import hashlib
import numpy as np
import torch
from torch import nn

from BandAnchoredNativeSonarWavelet import BandAnchoredNativeSonarWavelet
from BandAnchoredNativeSonarFeatures import band_features,PROTOCOL
from NativeLocalSonarWavelet import context_box
from MatchedSonarSignal import HALO,negative_frontend,source_logs


class MatchedSonarWavelet(BandAnchoredNativeSonarWavelet):
    family='matched_native_acquisition_noise_and_recorded_front_wavelet_v1'
    native_halo=HALO

    def __init__(self,width=8,*,aggregation='learned_fusion'):
        super().__init__(width)
        if aggregation not in ('mandatory','learned_fusion'):raise ValueError('explicit residual aggregation required')
        self.aggregation=aggregation
        self.local_gate=nn.Sequential(nn.Conv2d(5,8,1),nn.GELU(),nn.Conv2d(8,1,1))
        nn.init.zeros_(self.local_gate[-1].weight);nn.init.zeros_(self.local_gate[-1].bias)

    def feature_residual(self,features,support,*,use_phase):
        if features.ndim!=4 or features.shape[1]!=3 or support.shape!=features[:,0:1].shape:
            raise ValueError('aligned three-channel native features and support required')
        if not isinstance(use_phase,bool) or not torch.isfinite(features).all():raise ValueError('finite explicit observed features required')
        if not use_phase:
            features=torch.cat((features[:,:1],torch.zeros_like(features[:,1:])),dim=1)
        local=self.network(features,1,base=torch.zeros_like(features))[:,:1]*support
        coherent=self.consensus(local,support)
        if self.aggregation=='mandatory':return coherent
        # The selector is learned from observations and both residuals, not
        # from true brightness, a gain floor or a hand-set intensity threshold.
        weight=torch.sigmoid(self.local_gate(torch.cat((features,local,coherent),dim=1)))
        return (weight*local+(1-weight)*coherent)*support


def context_input(values,*,first_model_column,full_range_bins,frontend):
    if not np.iscomplexobj(values):raise ValueError('actual complex source context required')
    z=np.asarray(values,np.complex64)
    first=max(0,first_model_column-1)
    prepared=negative_frontend(z,first_range_bin=first,full_range_bins=full_range_bins) if frontend else z
    f,logs,mask=band_features(prepared,first_column=first_model_column)
    raw=z[:,1:] if first_model_column else z
    raw_logs,raw_mask=source_logs(raw)
    if np.any(raw_mask&~mask):raise ValueError('positive original support vanished exactly in the front end')
    return f,logs,mask,raw_logs,raw_mask


def context_prediction(model,values,device,*,first_model_column,full_range_bins,frontend,core,zero_neural=False):
    f,logs,mask,raw_logs,raw_mask=context_input(values,first_model_column=first_model_column,
        full_range_bins=full_range_bins,frontend=frontend)
    cy,cx,h,w=core
    if zero_neural:
        neural=np.zeros(logs.shape,np.float32)
    else:
        neural=model.feature_residual(torch.from_numpy(f[None]).to(device),
            torch.from_numpy(mask[None,None]).to(device),use_phase=True)[0,0].detach().cpu().numpy()
    total=(logs-raw_logs)+neural
    total=np.where(raw_mask,total,0).astype(np.float32)
    return total[cy:cy+h,cx:cx+w],neural[cy:cy+h,cx:cx+w]


@torch.inference_mode()
def infer_matched(model,values,device,*,frontend,core_shape=(128,1024),zero_neural=False):
    if not np.iscomplexobj(values):raise ValueError('actual complete complex source required')
    z=np.asarray(values,np.complex64)
    if z.ndim!=2 or not np.isfinite(z).all():raise ValueError('finite complete native complex source required')
    if not zero_neural and model.feature_protocol!=PROTOCOL:raise ValueError('explicit fixed-frequency feature protocol required')
    h,w=z.shape;ch,cw=core_shape
    if min(ch,cw)<4 or ch%4 or cw%4:raise ValueError('factor-four inference cores required')
    total=np.empty(z.shape,np.float32);neural=np.empty_like(total)
    if model is not None:model.eval()
    for y in range(0,h,ch):
        for x in range(0,w,cw):
            box,core=context_box(z.shape,(y,x,min(ch,h-y),min(cw,w-x)),halo=HALO)
            top,left,bottom,right=box
            a,b=context_prediction(model,z[top:bottom,max(0,left-1):right],device,
                first_model_column=left,full_range_bins=w,frontend=frontend,core=core,zero_neural=zero_neural)
            height,width=a.shape;total[y:y+height,x:x+width]=a;neural[y:y+height,x:x+width]=b
    if not np.isfinite(total).all() or not np.isfinite(neural).all():raise ValueError('nonfinite matched native restoration')
    return total,neural


@torch.inference_mode()
def infer_group(models,values,device,*,core_shape=(256,2048),zero_neural_for=()):
    """Prepare each complete raw/front field once, then evaluate all models.

    Full-source finite features are computed before taking CNN windows. Thus
    their real neighbors and FIR are not recomputed at artificial tile edges.
    The smaller 80x160 halo is for the CNN only; raw composed inference above
    separately uses the proven 80x416 halo including noise/FIR/features.
    """
    if not np.iscomplexobj(values):raise ValueError('actual complete complex source required')
    z=np.asarray(values,np.complex64);qraw,original_mask=source_logs(z)
    if z.ndim!=2 or not np.isfinite(z).all():raise ValueError('complete finite native complex field required')
    h,w=z.shape;ch,cw=core_shape
    if min(ch,cw)<4 or ch%4 or cw%4:raise ValueError('factor-four native group cores required')
    output={name:np.empty(z.shape,np.float32) for name in models}
    metadata=dict(prepared_fields={},neural={})
    for front in sorted({flag for _,flag in models.values()}):
        selected={name:model for name,(model,flag) in models.items() if flag==front}
        prepared=negative_frontend(z) if front else z
        features,q,mask=band_features(prepared)
        if np.any(original_mask&~mask):raise ValueError('original positive support vanished')
        metadata['prepared_fields']['front' if front else 'raw']=dict(
            native_shape=list(z.shape),prepared_complex64_SHA256=hashlib.sha256(memoryview(prepared).cast('B')).hexdigest(),
            feature_SHA256=hashlib.sha256(memoryview(features).cast('B')).hexdigest(),
            log_base_SHA256=hashlib.sha256(memoryview(q).cast('B')).hexdigest())
        digests={name:hashlib.sha256() for name in selected}
        sums={name:[0.,0.] for name in selected}
        for model in selected.values():model.eval()
        for y in range(0,h,ch):
            for x in range(0,w,cw):
                box,core=context_box(z.shape,(y,x,min(ch,h-y),min(cw,w-x)),halo=(80,160))
                top,left,bottom,right=box;cy,cx,height,width=core
                f=torch.from_numpy(features[:,top:bottom,left:right].copy()[None]).to(device)
                support=torch.from_numpy(mask[top:bottom,left:right].copy()[None,None]).to(device)
                for name,model in selected.items():
                    if name in zero_neural_for:neural=np.zeros((height,width),np.float32)
                    else:
                        neural=model.feature_residual(f,support,use_phase=True)[0,0,
                            cy:cy+height,cx:cx+width].cpu().numpy().copy()
                    base=q[y:y+height,x:x+width]-qraw[y:y+height,x:x+width]
                    delta=np.where(original_mask[y:y+height,x:x+width],base+neural,0).astype(np.float32)
                    output[name][y:y+height,x:x+width]=delta
                    digests[name].update(memoryview(np.ascontiguousarray(neural)).cast('B'))
                    sums[name][0]+=float(np.abs(neural).sum(dtype=np.float64))
                    sums[name][1]+=float(np.square(neural,dtype=np.float64).sum())
        for name in selected:
            metadata['neural'][name]=dict(native_values=h*w,absolute_mean=sums[name][0]/(h*w),
                RMS=float(np.sqrt(sums[name][1]/(h*w))),
                tile_stream_SHA256=digests[name].hexdigest(),tile_core=list(core_shape))
    if any(not np.isfinite(a).all() for a in output.values()):raise ValueError('nonfinite complete native group output')
    return output,metadata
