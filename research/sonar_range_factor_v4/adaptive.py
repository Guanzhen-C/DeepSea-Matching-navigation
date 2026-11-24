"""Causal gate for frozen per-edge objective selection; not an M0/M1 equivalence."""
from dataclasses import asdict, dataclass
import json

import numpy as np

from research.sonar_range_factor_v2.models import Track, physical_interval
from research.sonar_range_factor_v3.likelihood import matched_huber
from .batch import Batch


@dataclass(frozen=True)
class Policy:
    tolerance_sigma: float
    radius_limit: float
    deadband_min_sigma: float
    min_curvature: float=.015625           # height std <= 8 m under local linear approximation
    min_bound_margin_m: float=.50
    accept_confidence: float=.80
    reject_confidence: float=.35
    max_boxes: int=128
    m1_info_known: float=1.
    m1_info_unknown: float=.5
    weak_m0_info: float=.5

    def __post_init__(self):
        positive=(self.tolerance_sigma,self.radius_limit,self.deadband_min_sigma,
                  self.min_curvature,self.min_bound_margin_m)
        if not np.isfinite(positive).all() or min(positive)<=0 or self.max_boxes<0 or not (
                0<=self.reject_confidence<=self.accept_confidence<=1) or not (
                0<self.m1_info_unknown<=self.m1_info_known<=1 and 0<self.weak_m0_info<=1):
            raise ValueError("Invalid gate configuration")


@dataclass(frozen=True)
class Gate:
    mode: np.ndarray
    reason: np.ndarray
    information: np.ndarray
    feasible: np.ndarray
    deadband_sigma: np.ndarray
    evaluated_boxes: np.ndarray


def physical_tube(batch,poses,eps):
    """Below-sensor intersection for each view's interval [r-eps,r+eps]."""
    d=batch.geom(np.asarray(poses,float))[2]
    eps=np.asarray(eps,float)
    if eps.shape!=d.shape or not np.isfinite(eps).all() or (eps<0).any():
        raise ValueError("A finite tolerance per original range is required")
    high=(batch.r+eps)**2-d
    low=np.maximum(0.,batch.r-eps)**2-d
    a=np.maximum(batch.lo,np.max(batch.z-np.sqrt(np.maximum(0.,high)),axis=1))
    b=np.minimum(batch.hi,np.min(batch.z-np.sqrt(np.maximum(0.,low)),axis=1))
    scale=32*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(a),abs(b)))
    return (high>=0).all(axis=1)&(a<=b+scale)


def decide(batch,poses,old_factor,confidence,policy,*,arrival_s,node_times_s):
    """Evaluate current snapshot only. Freeze the result before optimization."""
    confidence=np.asarray(confidence,float)
    n=len(batch.tracks)
    stamps=np.asarray(node_times_s,float)
    if stamps.shape!=(2,) or not np.isfinite(stamps).all() or not np.isfinite(arrival_s) or np.max(stamps)>arrival_s:
        raise ValueError("Future or invalid pose seed supplied for causal gate")
    if confidence.shape!=(n,) or np.isinf(confidence).any() or (
            (confidence[np.isfinite(confidence)]<0)|(confidence[np.isfinite(confidence)]>1)).any():
        raise ValueError("Confidence must be calibrated probability or NaN for unknown")
    if old_factor.features!=n:
        raise ValueError("Frozen M0 must reference the same candidate rows")
    std=np.sqrt(np.stack([np.diag(t.covariance) for t in batch.tracks]))
    tube=physical_tube(batch,poses,policy.tolerance_sigma*std)
    out=batch.evaluate(poses,max_boxes=policy.max_boxes,allow_reference=False)
    dz=abs(batch.z[:,1]-batch.z[:,0])
    width=old_factor.maximum_range_m+6*old_factor.pressure_std_m
    dead=2*dz*width/(batch.r.sum(axis=1)*old_factor.standard_deviation_m)
    minimum_margin=np.minimum(out.height-batch.lo,batch.hi-out.height)
    mode=np.full(n,"M0",dtype="<U6")
    reason=np.full(n,"m1_interior",dtype="<U40")
    information=np.ones(n)

    def mark(mask,tag):
        select=(reason=="m1_interior")&mask
        reason[select]=tag
        return select

    bad=np.isfinite(confidence)&(confidence<policy.reject_confidence)
    mode[mark(bad,"low_association")]="reject"
    weak=np.isfinite(confidence)&(confidence<policy.accept_confidence)
    information[mark(weak,"weak_association")]=policy.weak_m0_info
    mark(~tube,"outside_noise_tube")
    mark(dead<policy.deadband_min_sigma,"m0_narrow_deadband")
    mark(out.status!="ok","profile_"+ "uncertain")
    mark(~np.isfinite(out.cost)|np.isnan(out.curvature)|(out.curvature<policy.min_curvature),"weak_curvature")
    mark(minimum_margin<policy.min_bound_margin_m,"near_height_boundary")
    mark(2*out.cost>policy.radius_limit**2,"large_profile_innovation")
    good=reason=="m1_interior"
    mode[good]="M1"
    information[good]=np.where(np.isnan(confidence[good]),policy.m1_info_unknown,policy.m1_info_known)
    reason[good&np.isnan(confidence)]="m1_unknown_association_tempered"
    information[mode=="reject"]=0
    for a in (mode,reason,information,tube,dead):
        a.setflags(write=False)
    return Gate(mode,reason,information,tube,dead,out.work)


def calibrate():
    """Freeze numerical cutoffs with independent synthetic received pairs.

    No navigation error, archived truth, future time or test mission is used.
    Engineering curvature, p(confidence), and budget limits are predeclared.
    """
    seed=792043
    rng=np.random.default_rng(seed)
    tracks=[]
    sigmas=[]
    low_depth=[]
    poses=np.array([[0.,0.,.13],[7.,4.,.45]])
    n=240
    for i in range(n):
        L=rng.uniform([28.,20.],[65.,95.])
        z=np.array([0.,rng.uniform(0,1.) if i<n//2 else rng.uniform(2.,17.)])
        Z=rng.uniform(-32.,-13.)
        dx=L-poses[:,:2]
        ideal=np.sqrt(np.sum(dx*dx,axis=1)+(Z-z)**2)
        sigma=rng.uniform(.25,.8)
        cov=sigma**2*np.array([[1.,.25],[.25,1.]])
        r=ideal+rng.multivariate_normal(np.zeros(2),cov)
        tracks.append(Track(L,np.zeros((2,2)),z,r,cov,physical_interval(z,250.)))
        g=2*r*np.array([1.,-1.])/r.sum()
        sigmas.append(sigma)
        if i<n//2:
            low_depth.append(2*z[1]*(250.+.18)/(r.sum()*np.sqrt(g@cov@g)))
    b=Batch(tracks,matched_huber(2))
    result=b.evaluate(poses)
    ok=np.char.startswith(result.status,"ok")
    if ok.sum()<n*.8:
        raise RuntimeError("Independent calibration lacks usable profiles")
    coverage_k=[]
    for k in (2.,2.5,3.,3.5,4.,4.5):
        f=physical_tube(b,poses,np.repeat(k*np.array(sigmas)[:,None],2,axis=1))
        coverage_k.append((k,float(f.mean())))
    threshold=next((k for k,f in coverage_k if f>=.98),4.5)
    policy=Policy(threshold,max(2.,float(np.quantile(np.sqrt(2*result.cost[ok]),.99))),
                  max(1.,float(np.quantile(low_depth,.75))))
    return dict(schema="ocean_adaptive_range_policy_v1",seed=seed,pairs=n,profiles=int(ok.sum()),
                nominal_tube_coverage=coverage_k,policy=asdict(policy),
                confidence_note="engineering thresholds; historical archives supply no calibrated probability",
                no_navigation_truth_used=True)


def load_policy(path):
    doc=json.loads(path.read_text())
    if doc["schema"]!="ocean_adaptive_range_policy_v1" or doc["no_navigation_truth_used"] is not True:
        raise ValueError("Unfrozen or mismatched Adaptive configuration")
    return Policy(**doc["policy"])


if __name__=="__main__":
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(calibrate(),f,indent=2,allow_nan=False)
        f.write("\n")
