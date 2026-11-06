"""Render brightness changes along the source pseudocolor curve, not RGB gain.

This is a presentation codec, not recovery of original linear echo amplitude.
It prevents near-black JPEG chroma errors from becoming bright blue artifacts.
"""
import numpy as np
from PublicSonarDisplayLuminance import display_luminance


def display_lookup(sample):
    if sample.domain!="display_rgb":raise ValueError("source pseudocolor image required")
    pixels=sample.values.reshape(-1,3)
    # Only a bounded thumbnail of colors is used to identify whether the
    # renderer is discrete. The full measured image itself is never resized.
    probe=pixels[::max(1,len(pixels)//50000)]
    colors=np.unique(probe,axis=0)
    if len(colors)<=256:
        colors=np.unique(pixels,axis=0)
    if len(colors)<=256:
        levels=colors.astype(float)@np.array([.2126,.7152,.0722])/255
        order=np.argsort(levels)
        return dict(kind="discrete_observed_source_palette",levels=levels[order],colors=colors[order])
    levels=pixels.astype(float)@np.array([.2126,.7152,.0722])/255
    groups=np.rint(levels*511).astype(int)
    order=np.argsort(groups,kind="stable")
    boundaries=np.r_[0,np.flatnonzero(np.diff(groups[order]))+1,len(order)]
    colors=[];values=[]
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        indices=order[left:right]
        colors.append(np.median(pixels[indices],axis=0))
        values.append(float(np.median(levels[indices])))
    return dict(kind="robust_source_pseudocolor_curve",levels=np.asarray(values),colors=np.asarray(colors))


def render_pseudocolor(sample,prediction,lookup=None):
    q,_=display_luminance(sample)
    prediction=np.asarray(prediction,np.float64)
    if prediction.shape!=q.shape or not np.isfinite(prediction).all():raise ValueError("native finite display brightness required")
    lookup=display_lookup(sample) if lookup is None else lookup
    requested=prediction.mean(0)
    levels,colors=lookup["levels"],lookup["colors"]
    if lookup["kind"]=="discrete_observed_source_palette":
        right=np.searchsorted(levels,requested).clip(0,len(levels)-1)
        left=(right-1).clip(0,len(levels)-1)
        nearest=np.where(np.abs(levels[left]-requested)<=np.abs(levels[right]-requested),left,right)
        result=colors[nearest].astype(np.uint8)
    else:
        result=np.stack([np.interp(requested,levels,colors[:,c]) for c in range(3)],axis=-1)
        result=np.rint(result.clip(0,255)).astype(np.uint8)
    identity=np.all(prediction==q.astype(np.float64),axis=0)
    protected=~sample.values.any(-1)
    result[identity|protected]=sample.values[identity|protected]
    return result
