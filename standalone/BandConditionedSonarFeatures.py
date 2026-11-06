"""Observed features using fixed whole-source statistics, no local retuning.

An interior context includes the actual left complex predecessor. The same
frozen model feature definition is applied to original and subband observations.
Neither observed complex phase nor an unpaired image is called clean truth.
"""
import numpy as np

from CarrierRelativeSonarFeatures import carrier_relative_features


def context_features(values,origin,source_mean,*,first_column):
    z=np.asarray(values)
    if (z.ndim!=2 or not np.iscomplexobj(z) or not np.isfinite(z).all()
            or not np.isfinite(origin) or first_column<0 or z.shape[1]<2):
        raise ValueError('native complex context with actual predecessor and global statistics required')
    a=np.abs(z);positive=a>0;unit=z/np.where(positive,a,1.)
    g=unit[:,1:]*unit[:,:-1].conj();valid=positive[:,1:]&positive[:,:-1]
    if first_column:
        selected=a[:,1:];mask=positive[:,1:]
        gradient=np.where(valid,g,0)
    else:
        selected=a;mask=positive
        gradient=np.zeros(z.shape,np.complex128);gradient[:,1:]=np.where(valid,g,0)
    logged=np.zeros(selected.shape,np.float32)
    logged[mask]=np.log(selected[mask]).astype(np.float32)
    raw=np.stack(((logged-origin)*mask,gradient.real,gradient.imag)).astype(np.float32)
    features,metadata=carrier_relative_features(raw,mask[None],source_mean=source_mean)
    return features,mask,metadata


def mosaic_single_side(amplitudes,side,*,phase=0,height=None):
    a=np.asarray(amplitudes)
    if a.ndim!=3 or a.shape[0]!=8 or a.shape[1]<2 or side not in (0,1) or not 0<=phase<4:
        raise ValueError('all eight native beams, preceding/current records and known side required')
    paired=(a[:4,1:]+a[4:,:-1]).transpose(1,0,2)
    rows=paired.reshape(4*(a.shape[1]-1),a.shape[-1])
    height=len(rows)-phase if height is None else height
    if height<1 or phase+height>len(rows):raise ValueError('requested native mosaic coverage is missing')
    image=rows[phase:phase+height]
    return image[:,::-1] if side==0 else image
