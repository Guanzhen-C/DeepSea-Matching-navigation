"""Display-domain brightness for pseudo-colored public sonar observations.

Luminance is NOT recovered linear acoustic amplitude. Networks see brightness
only, so known injected gray stripes cannot be removed through RGB color-map
shortcuts. Original colors remain available for display reconstruction.
"""
import numpy as np


def display_luminance(sample):
    if sample.domain!="display_rgb":raise ValueError("native display RGB sample required")
    rgb=sample.values.astype(np.float64)/255.
    y=np.einsum("hwc,c->hw",rgb,np.array([.2126,.7152,.0722]))
    return np.repeat(y[None].astype(np.float32),3,axis=0),np.ones(y.shape,dtype=bool)


def restore_display(sample,prediction):
    q,_=display_luminance(sample)
    prediction=np.asarray(prediction,np.float64)
    if prediction.shape!=q.shape or not np.isfinite(prediction).all():
        raise ValueError("finite native 3CHW display brightness prediction required")
    original=sample.values.astype(np.float64)
    source_y=np.einsum("hwc,c->hw",original,np.array([.2126,.7152,.0722]))
    requested=np.clip(prediction.mean(0),0,1)*255
    ratio=np.divide(requested,source_y,out=np.ones_like(source_y),where=source_y>0)
    result=np.rint(np.clip(original*ratio[...,None],0,255)).astype(np.uint8)
    # No color or texture is fabricated for original black/no-data pixels.
    result[source_y==0]=sample.values[source_y==0]
    return result
