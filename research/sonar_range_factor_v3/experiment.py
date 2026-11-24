"""Paired, fixed-association synthetic benchmark; never opens real navigation data."""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.stats import chi2

from standalone.SideScanMappedRangeChange import MappedRangeChange, profiled_range_residual_jacobians
from research.sonar_range_factor_v2.experiment import Config, RMAX, PRIOR_STD, synthesize
from research.sonar_range_factor_v2.models import Track, joint_residual_jacobian, profile_minimum
from .certified import FrozenHeight
from .likelihood import RadialLoss, matched_huber, profile_pair, propagate

ROOT=Path(__file__).resolve().parents[2]
METHODS=("M0","M0_plain","M1_Gaussian","M1_Robust","M2")


@dataclass(frozen=True)
class Case:
    dz: float=4.
    sigma: float=.35
    heading: float=35.
    map_std: float=.5
    false_fraction: float=0.
    views: int=4
    width: float=90.
    noise: str="gaussian"
    raw_correlation: float=.25
    receiver_xy_std: float=.15
    receiver_z_std: float=.04


def scale_calibration():
    """Separate data: nominal Gaussian residual blocks, no pose error or holdout labels."""
    shape=np.array([[1.,.25],[.25,1.]])
    C=np.linalg.inv(np.linalg.cholesky(shape))
    seeds=list(range(4030,4094))
    values=np.concatenate([np.random.default_rng(s).multivariate_normal(np.zeros(2),shape,32)
                           for s in seeds])
    t=np.einsum("ni,ni->n",values@C.T,values@C.T)
    return dict(seeds=seeds,blocks=len(t),scale=float(np.sqrt(np.median(t)/chi2.ppf(.5,2))),
                radial_huber=matched_huber(2).scale,nominal_probability=math.erf(3/math.sqrt(2)))


def observations(case,seed,calibrated_scale):
    # All view-count scans use prefixes of the SAME complete raw acquisition.
    full=6
    n=case.views
    cfg=Config(height_difference=case.dz,heading_difference_deg=case.heading,
               range_std=0.,map_std=case.map_std,repetitions=full,features=6)
    ref=synthesize(cfg,seed)
    rng=np.random.default_rng(seed+304199)
    shape=(1-case.raw_correlation)*np.eye(full)+case.raw_correlation*np.ones((full,full))
    eps=rng.multivariate_normal(np.zeros(full),shape,len(ref.tracks))
    tail=np.sqrt(2/rng.chisquare(4,len(ref.tracks)))
    if case.noise=="student4":
        eps*=tail[:,None]
    r=np.stack([t.ranges for t in ref.tracks])+case.sigma*eps
    u=np.stack([t.lever for t in ref.tracks])+case.receiver_xy_std*rng.normal(size=(len(r),full,2))
    z=np.stack([t.sensor_z for t in ref.tracks])+case.receiver_z_std*rng.normal(size=(len(r),full))
    n_bad=int(round(case.false_fraction*len(r)))
    if n_bad>=2:
        idx=rng.choice(len(r),n_bad,replace=False)
        src=np.roll(idx,1)
        r[idx,n-1]=r[src,n-1]
        u[idx,n-1]=u[src,n-1]
        z[idx,n-1]=z[src,n-1]
    else:
        n_bad=0
    raw_cov=shape[:n,:n]*(max(.025,case.sigma)*calibrated_scale)**2
    poses=ref.initial_poses[:n].copy()
    tracks=[]
    for i,t in enumerate(ref.tracks):
        lo=max(float(np.max(z[i,:n]-RMAX)),-17.-case.width/2)
        hi=min(float(np.min(z[i,:n])),-17.+case.width/2)
        track=Track(t.xy,u[i,:n],z[i,:n],r[i,:n],raw_cov,(lo,hi))
        var=np.r_[case.map_std**2*np.ones(2),case.receiver_xy_std**2*np.ones(2*n),
                  case.receiver_z_std**2*np.ones(n),0.,0.]
        cov=propagate(track,poses,np.clip(-17.,lo,hi),raw_cov,np.diag(var))
        tracks.append(Track(track.xy,track.lever,track.sensor_z,track.ranges,cov,track.bounds))
    return replace(ref,config=replace(cfg,repetitions=n),initial_poses=poses,tracks=tracks,n_swapped=n_bad)


def m0_pairs(scene,plain):
    """Freeze M0 residual, height envelope and kernel. Only supply matched sigma input."""
    ts=scene.tracks
    for i in range(0,scene.config.repetitions,2):
        idx=[i,i+1]
        ra,rb=(np.array([t.ranges[j] for t in ts]) for j in idx)
        gradient=np.column_stack((2*ra,-2*rb))/(ra+rb)[:,None]
        cov=np.array([t.covariance[np.ix_(idx,idx)] for t in ts])
        sigma=np.sqrt(np.einsum("ni,nij,nj->n",gradient,cov,gradient))
        yield i,MappedRangeChange(np.array([t.xy for t in ts]),
              *[np.array([t.lever[j] for t in ts]) for j in idx],
              *[np.array([t.sensor_z[j] for t in ts]) for j in idx],
              ra,rb,sigma,maximum_range_m=RMAX,information_budget=1. if plain else .5)


def fit(scene,method):
    """Retain failures; intention-to-treat uses the common prior estimate on failure."""
    t0=time.perf_counter()
    t_composition=np.nan
    try:
        B=scene.contractions()
        pairs=[]
        init=np.zeros(2)
        bounds=(-np.inf,np.inf)
        if method.startswith("M0"):
            pairs=list(m0_pairs(scene,method=="M0_plain"))
        elif method.startswith("M1"):
            pairs=[(i,t.subset([i,i+1])) for t in scene.tracks
                   for i in range(0,scene.config.repetitions,2)]
            loss=matched_huber(2) if method=="M1_Robust" else RadialLoss()
        elif method=="M2":
            lo,hi=np.array([t.bounds for t in scene.tracks]).T
            init=np.r_[init,np.clip(-17.,lo+1e-7,hi-1e-7)]
            bounds=np.r_[[-np.inf,-np.inf],lo],np.r_[[np.inf,np.inf],hi]
        else:
            raise ValueError("Unknown experiment method")
        t_composition=(time.perf_counter()-t0)*1000
        dim=len(init)
        prior=np.pad(np.eye(2)/PRIOR_STD,((0,0),(0,dim-2)))

        def evaluate(x):
            pose=scene.at(x[:2])
            r,J=[x[:2]/PRIOR_STD],[prior]
            if method.startswith("M0"):
                for i,t in pairs:
                    e,A,D=profiled_range_residual_jacobians(*pose[i:i+2],t,robust=method=="M0")
                    r.append(e)
                    J.append(A@B[3*i:3*i+3]+D@B[3*i+3:3*i+6])
            elif method.startswith("M1"):
                for i,t in pairs:
                    e,A,_=profile_pair(t,pose[i:i+2],loss)
                    r.append(e)
                    J.append(A@B[3*i:3*i+6])
            else:
                for k,t in enumerate(scene.tracks):
                    e,A=joint_residual_jacobian(t,pose,x[2+k])
                    Q=np.zeros((len(e),dim))
                    Q[:,:2]=A[:,:-1]@B
                    Q[:,2+k]=A[:,-1]
                    r.append(e)
                    J.append(Q)
            return np.concatenate(r),np.vstack(J)

        previous=None

        def cached(x):
            nonlocal previous
            if previous is None or not np.array_equal(previous[0],x):
                previous=x.copy(),evaluate(x)
            return previous[1]

        p=least_squares(lambda x:cached(x)[0],init,jac=lambda x:cached(x)[1],
                       bounds=bounds,max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        ok=bool(p.success and np.isfinite(p.x).all())
        return dict(ok=ok,reported_error_m=float(np.linalg.norm(
                        (p.x[:2] if ok else init[:2])-scene.true_shift)),
                    unfinished_error_m=None if ok else float(np.linalg.norm(p.x[:2]-scene.true_shift)),
                    elapsed_ms=(time.perf_counter()-t0)*1000,prep_ms=t_composition,
                    nfev=p.nfev,reason=None if ok else str(p.message))
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
        return dict(ok=False,reported_error_m=float(np.linalg.norm(scene.true_shift)),
                    unfinished_error_m=None,elapsed_ms=(time.perf_counter()-t0)*1000,
                    prep_ms=float(t_composition) if np.isfinite(t_composition) else None,
                    nfev=None,reason=f"{type(exc).__name__}: {exc}")


def study_design():
    fields=dict(dz=[0.,.001,4.,16.],sigma=[.15,.35,.7,1.4],heading=[0.,35.,90.,180.],
                map_std=[0.,1.,2.5],false_fraction=[0.,1/3,.5],views=[2,4,6],width=[8.,22.,90.])
    design=[(k,v,replace(Case(),**{k:v})) for k,values in fields.items() for v in values]
    design.append(("exact_measured_equal_z",0.,replace(Case(),dz=0.,receiver_z_std=0.)))
    design.extend((kind,s,replace(Case(),noise="student4",**args)) for kind,s,args in [
        ("student4_sigma",.7,dict(sigma=.7)),("student4_sigma",1.4,dict(sigma=1.4)),
        ("student4_swapped",.5,dict(sigma=.7,false_fraction=.5))])
    return design


def result_stats(trials):
    out={}
    for name in METHODS:
        rows=[r["models"][name] for r in trials]
        errors=np.array([r["reported_error_m"] for r in rows])
        successes=np.array([r["ok"] for r in rows],bool)
        ms=np.array([r["elapsed_ms"] for r in rows])
        out[name]=dict(rmse_m=float(np.sqrt(np.mean(errors**2))),
            n=len(rows),failed=int((~successes).sum()),
            success_only_rmse_m=float(np.sqrt(np.mean(errors[successes]**2))) if successes.any() else None,
            p95_error_m=float(np.quantile(errors,.95)),median_ms=float(np.median(ms)),p95_ms=float(np.quantile(ms,.95)))
    return out


def comparisons(rows):
    rng=np.random.default_rng(300924)
    draw=rng.integers(0,len(rows),(3000,len(rows)))
    result={}
    for a,b in (("M1_Gaussian","M0_plain"),("M1_Robust","M0"),
                ("M1_Robust","M1_Gaussian"),("M2","M1_Gaussian")):
        x=np.array([[r["models"][k]["reported_error_m"]**2 for k in [a,b]] for r in rows])
        f=lambda y:np.sqrt(y[...,0])-np.sqrt(y[...,1])
        result[f"{a}-{b}"]=dict(delta_m=float(f(x.mean(axis=0))),
                                ci95_m=np.quantile(f(x[draw].mean(axis=1)),[.025,.975]).tolist())
    return result


def certificate_suite(scale):
    p0=np.array([[0.,0.,0.],[1.,0.,0.]])
    tie=Track([1.,0.],np.zeros((2,2)),[0.,2.],[2.8125,1.8125],np.eye(2),(-4.,0.))
    pairs=[("tie",tie,p0),("nadir",Track([0.,0.],np.zeros((2,2)),[0.,2.],
            [1.,3.],np.eye(2),(-4.,0.)),np.zeros((2,3)))]
    for i,case in enumerate([Case(),Case(dz=0.),Case(dz=16.,sigma=1.4)]):
        s=observations(case,881+i,scale)
        pairs.append((f"cov_pair_{i}",s.tracks[0].subset([0,1]),s.initial_poses[:2]))
    log={}
    for label,t,p in pairs:
        problem=FrozenHeight(t,p)
        t0=time.perf_counter()
        opt=problem.minimum(timeout_s=None)
        profile_time=(time.perf_counter()-t0)*1000
        stationary=problem.stationary(timeout_s=None)
        try:
            fast=asdict(profile_minimum(t,p))
        except ValueError as exc:
            fast=dict(error=str(exc))
        log[label]=dict(certificate=opt,strict_ms=profile_time,stationary=stationary,fast=fast,
                        budget_zero=problem.minimum(max_boxes=0,timeout_s=None))
    return log


def multiview_graph(scale):
    """Measure Hessian structure for a block-shared synthetic pose/height graph."""
    logs=[]
    for m in (2,4,6):
        scene=observations(Case(views=m),76404,scale)
        for landmarks in (6,48,192):
            n=3*m
            J=np.zeros((m*landmarks,n+landmarks))
            r=np.empty(m*landmarks)
            for k in range(landmarks):
                t=scene.tracks[k%len(scene.tracks)]
                e,A=joint_residual_jacobian(t,scene.initial_poses,-17.)
                rows=slice(m*k,m*(k+1))
                r[rows]=e
                J[rows,:n]=A[:,:-1]
                J[rows,n+k]=A[:,-1]
            t0=time.perf_counter()
            H=J.T@J
            H[:n,:n]+=np.eye(n)*.01
            g=J.T@r
            solved=np.linalg.solve(H,-g)
            joint_ms=(time.perf_counter()-t0)*1000
            t0=time.perf_counter()
            # Independent height blocks (conditioned on this covariance in each track).
            A,B=H[:n,:n],H[:n,n:]
            zz=H[n:,n:].diagonal()
            S=A-(B/zz)@B.T
            ds=np.linalg.solve(S,-g[:n]+B@(g[n:]/zz))
            schur_ms=(time.perf_counter()-t0)*1000
            sizes=[]
            for mat in (H,S):
                sparse=csr_matrix(np.where(abs(mat)>1e-11,mat,0.))
                sizes.append(dict(dim=len(mat),nnz=int(sparse.nnz),
                    csr_bytes=sum(x.nbytes for x in (sparse.data,sparse.indices,sparse.indptr))))
            logs.append(dict(views=m,landmarks=landmarks,joint=sizes[0],reduced=sizes[1],
                             eliminated_given_H_ms=schur_ms,normal_plus_joint_ms=joint_ms,
                             max_pose_step_difference=float(abs(ds-solved[:n]).max())))
    return logs


def run(n):
    cal=scale_calibration()
    seeds=list(range(53000,53000+n))
    out=[]
    for i,(field,value,case) in enumerate(study_design()):
        trials=[]
        for seed in seeds:
            s=observations(case,seed,cal["scale"])
            k=seed%len(METHODS)
            trials.append(dict(seed=seed,swapped=s.n_swapped,
                               models={m:fit(s,m) for m in METHODS[k:]+METHODS[:k]}))
        out.append(dict(field=field,level=value,case=asdict(case),trials=trials,
                        scores=result_stats(trials),paired=comparisons(trials)))
        print(f"{i+1:02d} {field}={value}",flush=True)
    paths=["standalone/SideScanMappedRangeChange.py","research/sonar_range_factor_v2/models.py",
           "research/sonar_range_factor_v2/experiment.py"]
    paths.extend(str(p.relative_to(ROOT)) for p in sorted(Path(__file__).parent.glob("*.py")))
    return dict(schema="ocean_m1_research_v3",seeds=seeds,calibration=cal,
                reference_git=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                                 os=platform.platform()),
                cases=out,certificates=certificate_suite(cal["scale"]),
                graph=multiview_graph(cal["scale"]))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--trials",type=int,default=24)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.trials<=120 or args.output.exists():
        parser.error("Use 1 to 120 trials and a new file")
    result=run(args.trials)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")


if __name__=="__main__":
    main()
