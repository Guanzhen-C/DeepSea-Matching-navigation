"""Same native Haar/scalar-residual graph, explicitly NEW phase protocol.

Never load a trained local-relative checkpoint as these weights. The paired
experiment only copies a verified zero-residual step-zero initialization and
then actually trains with the different observed phase channels.
"""
import numpy as np
import torch

from NativeLocalSonarWavelet import NativeLocalSonarWavelet,context_box
from BandAnchoredNativeSonarFeatures import band_features,PROTOCOL


class BandAnchoredNativeSonarWavelet(NativeLocalSonarWavelet):
    family='native_processed_fixed_recorded_band_phase_wavelet_v1'
    feature_protocol=PROTOCOL


@torch.inference_mode()
def infer_band_native(model,values,device,*,use_phase=True,core_shape=(128,1024)):
    z=np.asarray(values)
    if (model.family!=BandAnchoredNativeSonarWavelet.family or z.ndim!=2
            or not np.iscomplexobj(z) or not np.isfinite(z).all()):
        raise ValueError('explicit band-anchored model and complete native complex field required')
    result=np.empty(z.shape,np.float32);ch,cw=core_shape
    if min(ch,cw)<4 or ch%4 or cw%4:raise ValueError('factor-four native inference core required')
    model.eval()
    for y in range(0,len(z),ch):
        for x in range(0,z.shape[1],cw):
            box,core=context_box(z.shape,(y,x,min(ch,len(z)-y),min(cw,z.shape[1]-x)))
            top,left,bottom,right=box
            f,_,mask=band_features(z[top:bottom,max(0,left-1):right],first_column=left)
            prediction=model.feature_residual(torch.from_numpy(f[None]).to(device),
                torch.from_numpy(mask[None,None]).to(device),use_phase=use_phase)[0,0].cpu().numpy()
            cy,cx,h,w=core;result[y:y+h,x:x+w]=prediction[cy:cy+h,cx:cx+w]
    if not np.isfinite(result).all():raise ValueError('nonfinite native band-anchored prediction')
    return result
