"""PC-only consistency and latency benchmark for original versus batched M1."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.optimize import least_squares

from research.sonar_range_factor_v2.models import _profiled,geometry
from research.sonar_range_factor_v3.certified import FrozenHeight
from research.sonar_range_factor_v3.experiment import Case,fit,observations,scale_calibration
from research.sonar_range_factor_v3.likelihood import matched_huber,profile_pair,transform
from research.sonar_range_factor_v3.replay import sha
from .batch import Batch


def batch_fit(scene):
    t0=time.perf_counter()
    m=scene.config.repetitions
    batches=[(i,Batch([t.subset([i,i+1]) for t in scene.tracks],matched_huber(2)))
             for i in range(0,m,2)]
    B=scene.contractions()
    prior=4.
    def evaluate(x):
        p=scene.at(x)
        e,J=[],[]
        for i,b in batches:
            out=b.evaluate(p[i:i+2])
            if not np.char.startswith(out.status,"ok").all():
                raise RuntimeError("Batch failed; no dropped input row")
            e.append(out.residual)
            J.append(out.jacobian@B[i*3:i*3+6])
        return np.r_[x/prior,np.stack(e,axis=1).ravel()],np.vstack(
                    [np.eye(2)/prior,np.stack(J,axis=1).reshape(-1,2)])
    s=least_squares(lambda x:evaluate(x)[0],np.zeros(2),jac=lambda x:evaluate(x)[1],
                    bounds=(-np.inf,np.inf),max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    return dict(error=float(np.linalg.norm(s.x-scene.true_shift)),ok=bool(s.success),
                nfev=s.nfev,ms=(time.perf_counter()-t0)*1000)


def error_report(n=24):
    loss=matched_huber(2)
    metrics={k:[] for k in ("height","cost","residual","jacobian","hessian")}
    delays={k:[] for k in ("original_48","batch_cold_48","batch_moved_48","batch_cache_hit_48",
                           "original_graph","batch_graph")}
    profile_counts=dict(valid=0,failed_match=0,disagreements=0,reference_rows=0,
                        strict_bad=0,strict_examined=0)
    differences=[]
    source=scale_calibration()["scale"]
    seeds=list(range(61500,61500+n))
    for k,seed in enumerate(seeds):
        scene=observations(Case(dz=(0.,.001,4.,16.)[k%4],
                  heading=(0.,35.,90.,180.)[(k//4)%4],views=2),seed,source)
        p=scene.initial_poses
        tracks=scene.tracks*8
        B=Batch(tracks,loss)
        t0=time.perf_counter()
        for tr in tracks:
            try:
                profile_pair(tr,p,loss)
            except (ValueError,RuntimeError):
                pass
        delays["original_48"].append((time.perf_counter()-t0)*1000)
        t0=time.perf_counter()
        out=B.evaluate(p,use_cache=False)
        delays["batch_cold_48"].append((time.perf_counter()-t0)*1000)
        p2=p+np.array([[.001,0,0],[.001,0,0]])
        t0=time.perf_counter()
        B.evaluate(p2)
        delays["batch_moved_48"].append((time.perf_counter()-t0)*1000)
        t0=time.perf_counter()
        B.evaluate(p2)
        delays["batch_cache_hit_48"].append((time.perf_counter()-t0)*1000)
        profile_counts["reference_rows"]+=out.reference_rows
        for i,tr in enumerate(scene.tracks):
            ok=out.status[i].startswith("ok")
            try:
                opt,r,J,H,zp=_profiled(tr,geometry(tr,p))
                oldok=True
            except (ValueError,RuntimeError):
                oldok=False
            if ok!=oldok:
                profile_counts["disagreements"]+=1
            elif not ok:
                profile_counts["failed_match"]+=1
            else:
                profile_counts["valid"]+=1
                e,j=transform(r,J[:,:-1]+np.outer(J[:,-1],zp),loss)
                tuples=[(out.height[i],opt.height),(out.cost[i],opt.cost),
                        (out.residual[i],e),(out.jacobian[i],j),
                        (out.hessian[i],H[:-1,:-1]+np.outer(H[:-1,-1],zp))]
                for kk,(a,b) in zip(metrics,tuples):
                    metrics[kk].append(float(np.max(abs(a-b))))
                if i==0:
                    cert=FrozenHeight(tr,p).minimum(gap="1e-8",timeout_s=None)
                    profile_counts["strict_examined"]+=1
                    if cert["status"]!="gap_certified" or not (
                            float(cert["lower"])-2e-10<=out.cost[i]<=float(cert["upper"])+2e-10):
                        profile_counts["strict_bad"]+=1
        scene=observations(Case(),seed,source)
        old=fit(scene,"M1_Robust")
        new=batch_fit(scene)
        if old["ok"]!=new["ok"]:
            raise ValueError("Optimizer failure behavior differs")
        delays["original_graph"].append(old["elapsed_ms"])
        delays["batch_graph"].append(new["ms"])
        differences.append(abs(old["reported_error_m"]-new["error"]))
    stats={k:dict(median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),samples=len(v))
           for k,v in delays.items()}
    return dict(schema="ocean_range_v4_benchmark",seeds=seeds,platform=platform.platform(),
                numpy=np.__version__,profiles=profile_counts,
                max_abs={k:max(v,default=0.) for k,v in metrics.items()},latency=stats,
                speedup_48=stats["original_48"]["median_ms"]/stats["batch_cold_48"]["median_ms"],
                speedup_graph=stats["original_graph"]["median_ms"]/stats["batch_graph"]["median_ms"],
                max_fit_error_difference=max(differences),
                code_sha256={x.name:sha(x) for x in [Path(__file__),Path(__file__).with_name("batch.py")]})


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Do not overwrite benchmark evidence")
    payload=error_report()
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(payload,f,indent=2,allow_nan=False)
        f.write("\n")


if __name__=="__main__":
    main()
