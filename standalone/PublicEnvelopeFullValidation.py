"""All-native held-out validation, same tiled context as deployment."""
import numpy as np
import torch
from PublicSonarNativeGrid import tiles


@torch.inference_mode()
def full_case_statistics(model,target,support,noisy,device,legacy=False,core=512,halo=96):
    if target.shape!=noisy.shape or target.shape!=support.shape or target.shape[:2]!=(1,1):
        raise ValueError('one complete native log observation and aligned positive support required')
    q=target[0,0].numpy();mask=support[0,0].numpy();damaged=noisy[0,0].numpy()
    count=int(mask.sum())
    if not count:raise ValueError('empty validation support')
    origins={name:float(values[mask].mean(dtype=np.float64)) for name,values in (('noisy',damaged),('identity',q))}
    result=dict(input_SSE=0.,output_SSE=0.,identity_SSE=0.,observed_values=0,native_values_processed=0)
    for y,x,ey,ex,top,left,bottom,right in tiles(q.shape,core=core,halo=halo):
        local_mask=torch.from_numpy(mask[top:bottom,left:right].copy())[None,None].to(device)
        cy,cx=y-top,x-left;h,w=ey-y,ex-x
        region_mask=local_mask[...,cy:cy+h,cx:cx+w]
        reference=torch.from_numpy(q[y:ey,x:ex].copy())[None,None].to(device)
        n=region_mask.sum().item()
        result['observed_values']+=int(n);result['native_values_processed']+=h*w
        input_core=torch.from_numpy(damaged[y:ey,x:ex].copy())[None,None].to(device)
        result['input_SSE']+=float(((input_core-reference).square()*region_mask).sum(dtype=torch.float64))
        for name,values in (('noisy',damaged),('identity',q)):
            tensor=torch.from_numpy(values[top:bottom,left:right].copy())[None,None].to(device)
            if legacy:
                output=model(tensor.repeat(1,3,1,1),1)[:,0:1]
            else:
                origin=tensor.new_tensor(origins[name]).reshape(1,1,1,1)
                output=model(tensor,local_mask,origin=origin)
            prediction=output[...,cy:cy+h,cx:cx+w]
            key='output_SSE' if name=='noisy' else 'identity_SSE'
            result[key]+=float(((prediction-reference).square()*region_mask).sum(dtype=torch.float64))
    if result['native_values_processed']!=q.size or result['observed_values']!=count:
        raise ValueError('full-native validation coverage is incomplete')
    if not all(np.isfinite(result[key]) for key in ('input_SSE','output_SSE','identity_SSE')):
        raise ValueError('nonfinite full-native validation error')
    return result
