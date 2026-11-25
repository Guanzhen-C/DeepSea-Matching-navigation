"""Controlled pair geometry sweep. A range-only problem, not acoustic ray-tracing."""
import argparse
from dataclasses import asdict,dataclass,replace
from pathlib import Path

import numpy as np

from research.sonar_range_factor_v2.experiment import Config,Scene
from research.sonar_range_factor_v2.models import Track,joint_residual_jacobian,physical_interval,projected_linearization
from research.sonar_range_factor_v3.experiment import fit,m0_pairs
from research.sonar_range_factor_v3.likelihood import matched_huber,propagate
from research.sonar_range_factor_v3.replay import sha
from research.sonar_range_factor_v4.batch import Batch
from research.sonar_range_factor_v4.benchmark import batch_fit
from standalone.SideScanMappedRangeChange import profiled_range_residual_jacobians
from .diagnostics import decompose,finite
from .replay import write_json


@dataclass(frozen=True)
class Design:
    dz: float=4.
    heading_deg: float=35.
    cross_m: float=15.
    nominal_range_m: float=65.
    bearing_deg: float=75.
    map_std_m: float=.5
    centre_std_m: float=.15
    range_std_m: float=.35
    false_fraction: float=0.
    views: int=4


def build(case,seed):
    nmax,n,nfeatures=6,case.views,8
    rng=np.random.default_rng(seed)
    i=np.arange(nmax)
    yaw=np.deg2rad(case.heading_deg*(i%2))
    sign=np.where(i//2%2==0,1.,-1.)
    centre=np.column_stack((9.*(i%2)+3.*(i//2),case.cross_m*(i%2)+16.*(i//2)*sign))
    z=case.dz*(i%2)+.13*case.dz*(i//2)
    R=np.stack((np.stack((np.cos(yaw),-np.sin(yaw)),axis=-1),
                np.stack((np.sin(yaw),np.cos(yaw)),axis=-1)),axis=1)
    alpha=np.deg2rad(case.bearing_deg+rng.uniform(-6.,6.,nfeatures))
    radial=case.nominal_range_m+rng.uniform(-6.,6.,nfeatures)
    L=radial[:,None]*np.column_stack((np.cos(alpha),np.sin(alpha)))
    height=rng.uniform(-24.,-12.,nfeatures)
    raw_lever=rng.uniform(-3.,3.,(nfeatures,nmax,2))
    O=centre[None]+np.einsum("ijk,mik->mij",R,raw_lever)
    d=L[:,None]-O
    r=np.sqrt(np.sum(d*d,axis=-1)+(height[:,None]-z)**2)
    common=(np.eye(nmax)*.75+.25*np.ones((nmax,nmax)))*case.range_std_m**2
    r+=rng.multivariate_normal(np.zeros(nmax),common,nfeatures)
    lever=raw_lever+case.centre_std_m*rng.normal(size=raw_lever.shape)
    chart=L+case.map_std_m*rng.normal(size=L.shape)
    # Keep nuisance draws and prefixes identical across view-count changes.
    shift=rng.normal(0.,4.,2)
    p=np.column_stack((centre,yaw))
    p[i%2==1,:2]-=shift
    nbad=int(round(nfeatures*case.false_fraction))
    if nbad>=2:
        grab=rng.choice(nfeatures,nbad,replace=False)
        r[grab,n-1]=r[np.roll(grab,1),n-1]
        lever[grab,n-1]=lever[np.roll(grab,1),n-1]
    bounds=physical_interval(z[:n],250.)
    cov=common[:n,:n]+np.eye(n)*.025**2
    var=np.r_[np.repeat(case.map_std_m**2,2),np.repeat(case.centre_std_m**2,2*n),np.zeros(n+2)]
    tracks=[]
    for j in range(nfeatures):
        t=Track(chart[j],lever[j,:n],z[:n],r[j,:n],cov,bounds)
        Q=propagate(t,p[:n],-17.,cov,np.diag(var))
        tracks.append(Track(t.xy,t.lever,t.sensor_z,t.ranges,Q,t.bounds))
    # The full-angle ray stress study is a synthetic mathematical test; the
    # real along-track antenna aperture and shadows are not emulated.
    return Scene(Config(repetitions=n,features=nfeatures),p[:n],shift,tracks,np.sign(d[:,:n,1]),height,nbad)


def spectrum(s):
    B=s.contractions()
    A0,A1=[],[]
    invalid=0
    for i,old in m0_pairs(s,False):
        _,J,K=profiled_range_residual_jacobians(s.initial_poses[i],s.initial_poses[i+1],old)
        C=B[3*i:3*i+6]
        A0.append(np.column_stack((J,K))@C)
        tracks=[t.subset([i,i+1]) for t in s.tracks]
        p=s.initial_poses[i:i+2]
        batch=Batch(tracks,matched_huber(2)).evaluate(p)
        for k,t in enumerate(tracks):
            if not batch.status[k].startswith("ok"):
                invalid+=1
                continue
            height=batch.height[k]
            e,j=joint_residual_jacobian(t,p,height)
            if t.bounds[0]<height<t.bounds[1]:
                _,a=projected_linearization(e,j)
            else:
                a=j[:,:-1]
            A1.append(a@C)
    def get(rows):
        mat=np.concatenate(rows) if len(rows) else np.zeros((0,2))
        return decompose(mat)|dict(information=mat.T@mat)
    return {"M0":get(A0),"M1_Robust":get(A1)},invalid


def design():
    fields=dict(dz=[0.,1e-6,4.,16.],heading_deg=[0.,35.,90.,180.],cross_m=[2.,15.,30.],
          nominal_range_m=[35.,65.,115.],bearing_deg=[30.,75.,90.],map_std_m=[0.,.5,2.5],
          centre_std_m=[0.,.15,1.5],range_std_m=[.1,.35,1.4],false_fraction=[0.,.25,.5],views=[2,4,6])
    return [(field,v,replace(Design(),**{field:v})) for field,values in fields.items() for v in values]


def summary(rows,m):
    v=np.array([r[m]["error_m"] for r in rows])
    good=[r[m]["geometry"] for r in rows if r[m]["geometry"]["rank"]==2]
    H=np.mean([r[m]["geometry"]["information"] for r in rows],axis=0)
    x,E=np.linalg.eigh(H)
    a=E[:,-1]
    return dict(rmse_m=float(np.sqrt(np.mean(v**2))),failures=sum(not r[m]["success"] for r in rows),
        rank_two=len(good),median_singular=np.median(
            [np.pad(r[m]["geometry"]["singular"],(0,2-len(r[m]["geometry"]["singular"]))) for r in rows],axis=0),
        median_condition_if_rank2=float(np.median([r["condition"] for r in good])) if good else None,
        pooled_effective_axis_deg_mod180=float(np.rad2deg(np.arctan2(a[1],a[0]))%180) if x[-1]>1e-15 else None)


def run(n):
    seeds=list(range(86020,86020+n))
    result=[]
    for name,value,case in design():
        rows=[]
        for seed in seeds:
            s=build(case,seed)
            info,bad=spectrum(s)
            r=fit(s,"M0")
            try:
                q=batch_fit(s)
                m1=dict(success=q["ok"],error_m=q["error"] if q["ok"] else np.linalg.norm(s.true_shift))
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                m1=dict(success=False,error_m=np.linalg.norm(s.true_shift),reason=str(exc))
            rows.append(dict(seed=seed,reference_prior_error_m=np.linalg.norm(s.true_shift),unresolved_rows=bad,
                             M0=dict(success=r["ok"],error_m=r["reported_error_m"],geometry=info["M0"]),
                             M1_Robust=m1|dict(geometry=info["M1_Robust"])))
        result.append(dict(axis=name,level=value,config=asdict(case),
             summaries={m:summary(rows,m) for m in ("M0","M1_Robust")},rows=rows))
        print(name,value,flush=True)
    return finite(dict(schema="ocean_range_v5_geometry_stress",scenarios=result,seeds=seeds,
             code_sha256=sha(Path(__file__)),scope="range-only synthetic stress; no image front end or aperture visibility",
             no_range_rmse_m=np.sqrt(np.mean([r["reference_prior_error_m"]**2 for r in result[0]["rows"]]))))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--trials",type=int,default=16)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.trials<=64:
        p.error("Use a new path and between 1..64 trials")
    write_json(a.output,run(a.trials))
