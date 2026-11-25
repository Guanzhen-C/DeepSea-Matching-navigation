"""Statistical primitives for externally surveyed slant-range references.

The input is calibrated pick records, not an undocumented MAT or IQ format.
"""
import argparse
from pathlib import Path

import numpy as np

from research.sonar_range_factor_v3.replay import sha
from .replay import write_json


def scale_and_clock(reference_m,measured_m,sound_speed_nominal=1480.):
    x,y=np.asarray(reference_m,float),np.asarray(measured_m,float)
    if x.ndim!=1 or x.shape!=y.shape or x.size<4 or not np.isfinite(x).all() or (
            not np.isfinite(y).all() or (x<=0).any() or (y<=0).any() or
            not np.isfinite(sound_speed_nominal) or sound_speed_nominal<=0):
        raise ValueError("Positive distances, aligned independent target reference required")
    A=np.column_stack((np.ones(len(x)),x))
    if np.linalg.matrix_rank(A)!=2 or np.ptp(x)<1e-3:
        raise ValueError("Single-distance data cannot separate time zero and sound-speed scale")
    beta=np.linalg.lstsq(A,y-x,rcond=None)[0]
    b,k=beta
    return dict(offset_m=float(b),fractional_scale=float(k),
                time_zero_s=float(2*b/sound_speed_nominal),n=len(x),
                ref_span_m=float(np.ptp(x)))


def residuals(reference,measured,model):
    x=np.asarray(reference,float)
    y=np.asarray(measured,float)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Unaligned held-out picks")
    return y-x-model["offset_m"]-model["fractional_scale"]*x


def paired_covariance(ea,eb,block_ids=None,*,seed=64051,resamples=1000):
    xa,xb=np.asarray(ea,float),np.asarray(eb,float)
    if xa.ndim!=1 or xa.shape!=xb.shape or len(xa)<4 or not (
            np.isfinite(xa).all() and np.isfinite(xb).all()):
        raise ValueError("At least four correctly linked valid original receptions required")
    X=np.column_stack((xa,xb))
    sigma=np.cov(X,rowvar=False)
    if (sigma.diagonal()<=0).any():
        raise ValueError("Zero residual variance")
    rho=float(sigma[0,1]/np.sqrt(sigma[0,0]*sigma[1,1]))
    out=dict(n=len(X),bias_m=X.mean(axis=0).tolist(),covariance_m2=sigma.tolist(),rho=rho)
    if block_ids is not None:
        b=np.asarray(block_ids)
        if b.shape!=xa.shape:
            raise ValueError("Bad flight-block label")
        unique=np.unique(b)
        if len(unique)<3:
            raise ValueError("At least three independent blocks for bootstrap diagnostic")
        rows=[X[b==i] for i in unique]
        rng=np.random.default_rng(seed)
        reps=[]
        for _ in range(resamples):
            R=np.concatenate([rows[j] for j in rng.integers(len(rows),size=len(rows))])
            if np.all(R.std(axis=0)>0):
                reps.append(np.corrcoef(R.T)[0,1])
        out.update(blocks=len(rows),rho_ci95_block=np.quantile(reps,[.025,.975]).tolist(),bootstrap_seed=seed)
    return out


def map_semivariance(xy,error_xy,lag_edges):
    xy,E,edge=np.asarray(xy,float),np.asarray(error_xy,float),np.asarray(lag_edges,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or E.shape!=xy.shape or (
            not np.isfinite(xy).all() or not np.isfinite(E).all() or
            edge.ndim!=1 or len(edge)<2 or (np.diff(edge)<=0).any()):
        raise ValueError("Metric XY, independent map discrepancies and increasing lag bins required")
    if len(xy)>4000:
        raise ValueError("Pre-stratify map targets before quadratic all-pair calculation")
    i,j=np.triu_indices(len(xy),1)
    d=np.linalg.norm(xy[i]-xy[j],axis=1)
    g=.5*np.sum((E[i]-E[j])**2,axis=1)
    return [dict(lo_m=float(a),hi_m=float(b),pairs=int(mask.sum()),
                 semivariance_m2=float(g[mask].mean()) if mask.any() else None)
            for a,b in zip(edge[:-1],edge[1:]) for mask in [(d>=a)&(d<b)]]


def self_check():
    seed=64040
    rng=np.random.default_rng(seed)
    x=np.linspace(15.,150.,120)
    bias=.18
    scale=.0017
    fit=scale_and_clock(x,x+bias+scale*x)
    v0=np.array([[.24,-.12],[-.12,.36]])
    X=rng.uniform(15.,150.,(2048,2))
    noise=rng.multivariate_normal([0.,0.],v0,len(X))
    y=X+bias+scale*X+noise
    e=residuals(X,y,fit)
    return dict(schema="ocean_range_v5_calibration_selfcheck",seed=seed,scope="synthetic estimator validation only",
                fit=fit,reference_covariance_m2=v0.tolist(),known_expected_rho=float(v0[0,1]/np.sqrt(v0[0,0]*v0[1,1])),
                heldout=paired_covariance(*e.T,block_ids=np.arange(len(e))//64),
                code_sha256=sha(Path(__file__)))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error("Choose a new output path")
    write_json(a.output,self_check())
