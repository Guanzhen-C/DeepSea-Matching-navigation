"""Equivalent cheap gate front-end and an independent labelled feasibility holdout."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from research.sonar_range_factor_v2.models import Track,physical_interval
from research.sonar_range_factor_v3.replay import sha
from research.sonar_range_factor_v4.adaptive import Gate,decide,load_policy,physical_tube
from research.sonar_range_factor_v4.batch import Batch
from .diagnostics import tube_prediction
from .replay import SCENES,Study,write_json


def early_gate(batch,pose,old,confidence,policy,*,arrival_s,node_times_s):
    """Short-circuit only when no row would reach the numerical profile test."""
    confidence=np.asarray(confidence,float)
    n=len(batch.tracks)
    stamps=np.asarray(node_times_s,float)
    if stamps.shape!=(2,) or not np.isfinite(stamps).all() or not np.isfinite(arrival_s) or stamps.max()>arrival_s:
        raise ValueError("Future or invalid pose seed supplied for causal gate")
    if confidence.shape!=(n,) or np.isinf(confidence).any() or (
        (confidence[np.isfinite(confidence)]<0)|(confidence[np.isfinite(confidence)]>1)).any():
        raise ValueError("Confidence must be calibrated probability or NaN for unknown")
    if old.features!=n:
        raise ValueError("Frozen M0 must reference the same candidate rows")
    std=np.sqrt(np.stack([np.diag(t.covariance) for t in batch.tracks]))
    tube=physical_tube(batch,pose,policy.tolerance_sigma*std)
    dz=abs(batch.z[:,1]-batch.z[:,0])
    width=old.maximum_range_m+6*old.pressure_std_m
    dead=2*dz*width/(batch.r.sum(axis=1)*old.standard_deviation_m)
    lost=np.isfinite(confidence)&(confidence<policy.reject_confidence)
    weak=np.isfinite(confidence)&(confidence<policy.accept_confidence)
    early=lost|weak|~tube|(dead<policy.deadband_min_sigma)
    if not early.all():
        return (decide(batch,pose,old,confidence,policy,arrival_s=arrival_s,
                       node_times_s=node_times_s),False)
    reason=np.full(n,"m0_narrow_deadband",dtype="<U40")
    reason[~tube]="outside_noise_tube"
    reason[weak]="weak_association"
    reason[lost]="low_association"
    mode=np.where(lost,"reject","M0")
    weights=np.where(lost,0,np.where(weak,policy.weak_m0_info,1.))
    boxes=np.zeros(n,int)
    for a in (mode,reason,weights,tube,dead,boxes):
        a.setflags(write=False)
    return Gate(mode,reason,weights,tube,dead,boxes),True


def independent_holdout(policy,n=128):
    """Association labels are generated; never use the route score to choose."""
    seed=272143
    rng=np.random.default_rng(seed)
    v=np.array([.5,.5])
    cov=.30**2*np.array([[1.,.25],[.25,1.]])
    p=np.array([[0.,0.,0.],[4.,6.,0.]])
    table=[]
    for drift in (0.,.5,2.,4.):
        stats=Counter()
        for i in range(n):
            L=rng.uniform([18.,28.],[72.,95.])
            z=rng.uniform(0.,5.,2)
            h=rng.uniform(-35.,-16.)
            r=np.sqrt(np.sum((L-p[:,:2])**2,axis=1)+(h-z)**2)+rng.multivariate_normal([0.,0.],cov)
            reported=p.copy()
            reported[:,:2]+=rng.normal(0.,drift,(2,2))
            for wrong in (False,True):
                ranges=r.copy()
                if wrong:
                    ranges[1]+=rng.choice([-1.,1.])*rng.uniform(2.,12.)
                t=Track(L,np.zeros((2,2)),z,ranges,cov,physical_interval(z,250.))
                b=Batch([t])
                o=b.evaluate(reported)
                std=np.sqrt(cov.diagonal())[None]
                plain=physical_tube(b,reported,policy.tolerance_sigma*std)[0]
                P=np.diag([drift**2,drift**2,0.])
                new=physical_tube(b,reported,policy.tolerance_sigma*tube_prediction(b,reported,o,[P,P]))[0]
                label="wrong" if wrong else "correct"
                stats[label+"_nominal_pass"]+=int(plain)
                stats[label+"_predicted_marginal_pass"]+=int(new)
        table.append(dict(pose_std_m=drift,n_per_label=n,passes=dict(stats)))
    return dict(seed=seed,scope="known synthetic associations; wrong arm creates one inconsistent range",
                table=table)


def run(core,policy_file):
    policy=load_policy(policy_file)
    results={}
    for scene in SCENES:
        s=Study(scene,core)
        elapsed0,elapsed1=[],[]
        skipped=0
        for i in s.ids:
            old,b=s.old[i],s.batch[i]
            p=s.initial[list(s.nodes[i])]
            kwargs=dict(arrival_s=s.arrival[i],node_times_s=s.times[list(s.nodes[i])])
            confidence=np.full(old.features,np.nan)
            b.last=None
            b.warm.fill(np.nan)
            begin=time.perf_counter()
            base=decide(b,p,old,confidence,policy,**kwargs)
            elapsed0.append((time.perf_counter()-begin)*1000)
            b.last=None
            b.warm.fill(np.nan)
            begin=time.perf_counter()
            a,skip=early_gate(b,p,old,confidence,policy,**kwargs)
            elapsed1.append((time.perf_counter()-begin)*1000)
            skipped+=int(skip)
            if any(not np.array_equal(getattr(a,q),getattr(base,q)) for q in (
                    "mode","reason","information","feasible","deadband_sigma")):
                raise ValueError("Short-circuit changed a navigation action")
        results[scene]=dict(batches=len(s.ids),profile_batches_avoided=skipped,
            nominal_median_ms=float(np.median(elapsed0)),early_median_ms=float(np.median(elapsed1)),
            archive_hashes=s.hashes)
    return dict(schema="ocean_range_v5_gate_shortcircuit",code_sha256=sha(Path(__file__)),
            policy_sha256=sha(policy_file),fast_gate=results,holdout=independent_holdout(policy))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    for x in ("core","policy","output"):
        p.add_argument("--"+x,required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():
        p.error("Choose a new output path")
    write_json(a.output,run(a.core,a.policy))
