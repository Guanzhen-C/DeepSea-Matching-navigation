"""Exact fixed-graph ablations for archived range-factor *batches*."""
from collections import Counter
from dataclasses import fields
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from standalone.SideScanMappedRangeChange import MappedRangeChange,profiled_range_residual_jacobians
from research.sonar_range_factor_v2.models import Track,physical_interval
from research.sonar_range_factor_v3.adapter import _two_pose_factor,make_m0
from research.sonar_range_factor_v3.likelihood import matched_huber
from research.sonar_range_factor_v3.replay import sha
from research.sonar_range_factor_v4.batch import Batch
from research.sonar_range_factor_v4.replay import SCENES,PREVIOUS,load_scene

METHODS=("M0","M1_Robust")


def write_json(path,document):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(document,f,indent=2,allow_nan=False,ensure_ascii=False)
        f.write("\n")


def make_tracks(old,rho):
    shape=np.array([[1.,rho],[rho,1.]])
    tracks=[]
    for q in range(old.features):
        z=np.array([old.sensor_z_a[q],old.sensor_z_b[q]])
        r=np.array([old.range_a[q],old.range_b[q]])
        g=2*r*np.array([1.,-1.])/r.sum()
        cov=shape*(old.standard_deviation_m[q]**2/(g@shape@g))
        tracks.append(Track(old.landmark_xy[q],np.stack([old.origin_a[q],old.origin_b[q]]),
                            z,r,cov,physical_interval(z,old.maximum_range_m)))
    return tracks


class Study:
    def __init__(self,scene,frozen_core,rho=.25):
        import gtsam
        from oceansim.slam import pose_graph as core
        cp=Path(inspect.getfile(core)).resolve()
        if not cp.is_relative_to(Path(frozen_core).resolve()) or not -1<rho<1:
            raise ValueError("Frozen pose graph core with valid raw covariance assumption required")
        self.scene,self.rho,self.gtsam=scene,rho,gtsam
        self.core_hash=sha(cp)
        self.payload,self.trace,self.truth_path,self.hashes,self.source=load_scene(scene)
        self.times=np.asarray(self.payload["pose_timestamps_s"])
        self.initial=np.asarray(self.payload["pose_initializations"])
        if self.initial.shape!=(len(self.times),3) or (np.diff(self.times)<=0).any():
            raise ValueError("Invalid frozen initialization")
        self.base=core.GTSAMPlanarPoseGraph(core.PoseGraphConfig(**self.payload["objective_configuration"]))
        self.edges={}
        self.priors=[]
        self.old={}
        self.tracks={}
        self.batch={}
        self.range={}
        self.tokens={}
        self.arrival={}
        self.nodes={}
        self.gauge=self.base._gauge_factor(np.asarray(self.payload["gauge_pose"]))
        fields_old={f.name for f in fields(MappedRangeChange)}
        records=self.payload["mapped_range_changes"]
        if len(records)!=len(self.trace) or not records:
            raise ValueError("No authenticated range provenance")
        seen=set()
        for row,trace in zip(records,self.trace):
            i=row["edge_index"]
            a,b=row["source"],row["target"]
            edge=self.payload["edges"][i]
            if i in self.old or edge["kind"]!="mapped_range_change" or (
                    (a,b)!=(edge["source"],edge["target"]) or (a,b)!=(trace["source"],trace["target"])):
                raise ValueError("Misaligned edge and source trace")
            obs=MappedRangeChange(**{k:v for k,v in row["observations"].items() if k in fields_old})
            tokens=[tuple(t) for t in trace["received_tokens"]]
            if len(tokens)!=2*obs.features or len(set(tokens))!=len(tokens) or not seen.isdisjoint(tokens) or any(
                len(t)!=3 or not all(isinstance(v,int) and v>=0 for v in t) for t in tokens):
                raise ValueError("Duplicate or invalid received-range token")
            seen.update(tokens)
            if max(self.times[[a,b]])>trace["timestamp_s"]:
                raise ValueError("Noncausal pair endpoints")
            self.tokens[i]=tokens
            self.arrival[i]=float(trace["timestamp_s"])
            self.nodes[i]=(a,b)
            self.old[i]=obs
            self.tracks[i]=make_tracks(obs,rho)
            keys=self.base._key(a),self.base._key(b)
            self.batch[i]=Batch(self.tracks[i],matched_huber(2))
            self.range[i]={"M0":make_m0(obs,*keys),"M1_Robust":self._m1(i,keys)}
        for i,e in enumerate(self.payload["edges"]):
            if e["kind"]=="mapped_range_change":
                if i not in self.old:
                    raise ValueError("Missing saved original pair")
                continue
            e=dict(e)
            e["measurement"],e["information"]=map(np.asarray,(e["measurement"],e["information"]))
            self.edges[i]=self.base._edge_factor(core.PoseGraphEdge(**e))
        for q in self.payload["priors"]:
            q=dict(q)
            q["measurement"],q["information"]=map(np.asarray,(q["measurement"],q["information"]))
            self.priors.append(self.base._prior_factor(core.PoseGraphPrior(**q)))
        self.ids=sorted(self.range)

    def _m1(self,i,keys):
        batch=self.batch[i]
        def measurement(poses):
            result=batch.evaluate(poses)
            ok=np.char.startswith(result.status,"ok")
            if not ok.all():
                raise ValueError(f"batch {i} failed without dropping pair: "+",".join(sorted(set(result.status[~ok]))))
            return result.residual.ravel(),result.jacobian.reshape(-1,6)
        return _two_pose_factor(2*len(batch.tracks),*keys,measurement)

    def graph(self,method,remove):
        g=self.gtsam.NonlinearFactorGraph()
        g.add(self.gauge)
        i=0
        for record in self.payload["edges"]:
            if record["kind"]!="mapped_range_change":
                g.add(self.edges[i])
            elif i not in remove:
                if method=="M1_Robust":
                    self.batch[i].last=None
                    self.batch[i].warm.fill(np.nan)  # identical algorithm state for each arm
                g.add(self.range[i][method])
            i+=1
        for prior in self.priors:
            g.add(prior)
        return g

    def values(self,poses,start=0):
        v=self.gtsam.Values()
        for i,p in enumerate(poses,start):
            v.insert(self.base._key(i),self.base._pose2(p))
        return v

    def poses(self,v):
        return np.array([self.base._pose_array(v.atPose2(self.base._key(i))) for i in range(len(self.initial))])

    def solve(self,method,remove,first=False):
        """Common initial state, no warm starts inherited across arms, truth not read."""
        start=time.perf_counter()
        try:
            g=self.graph(method,set(remove))
            v=self.values(self.initial)
            before=g.error(v)
            cfg=self.gtsam.LevenbergMarquardtParams()
            cfg.setMaxIterations(60)
            s=self.gtsam.LevenbergMarquardtOptimizer(g,v,cfg)
            one=None
            if first:
                s.iterate()
                one=self.poses(s.values())
            out=s.optimize()
            p=self.poses(out)
            cost=g.error(out)
            if not np.isfinite(p).all() or not np.isfinite(cost):
                raise ValueError("Nonfinite graph output")
            return dict(status="returned",ms=(time.perf_counter()-start)*1000,pose=p,one=one,
                        objective_before=before,objective_after=cost,iterations=s.iterations(),factors=g.size())
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            return dict(status="failed",ms=(time.perf_counter()-start)*1000,error=str(exc))

    def truth(self):
        with np.load(self.truth_path,allow_pickle=False) as f:
            t=f["timestamps"]
            i=np.searchsorted(t,self.times)
            if (i>=len(t)).any() or not np.array_equal(t[i],self.times):
                raise ValueError("Graph nodes and evaluation timestamps differ")
            return f["positions"][i,0,:2]

    def receipt(self):
        return dict(scene=self.scene,seed=self.source["source_seed"],distance_m=self.source["distance_m"],
                    nodes=len(self.times),batches=len(self.ids),pairs=sum(x.features for x in self.old.values()),
                    archive_hashes=self.hashes,frozen_core_sha256=self.core_hash,
                    code_sha256=sha(Path(__file__)),rho_assumed=self.rho,
                    scope="archived simulation batch LM; geometry and associations fixed")


def score(result,truth,baseline=None,endpoints=None):
    row={k:v for k,v in result.items() if k not in ("pose","one")}
    if "pose" in result:
        errors=np.linalg.norm(result["pose"][:,:2]-truth,axis=1)
        mse=float(np.mean(errors**2))
        row.update(mse_m2=mse,rmse_m=float(np.sqrt(mse)),p95_m=float(np.quantile(errors,.95)),last_m=float(errors[-1]))
        if baseline is not None:
            ref=np.linalg.norm(baseline[:,:2]-truth,axis=1)
            row["delta_mse_m2"]=float(mse-np.mean(ref**2))
            row["delta_rmse_m"]=float(np.sqrt(mse)-np.sqrt(np.mean(ref**2)))
            motion=np.linalg.norm(result["pose"][:,:2]-baseline[:,:2],axis=1)
            row["pose_change_rmse_m"]=float(np.sqrt(np.mean(motion**2)))
            if endpoints is not None:
                row["endpoint_changes_m"]=[float(motion[i]) for i in endpoints]
    return row


_worker=None


def _init(scene,core,rho):
    global _worker
    _worker=Study(scene,core,rho)


def _solve(task):
    method,tag,removed=task
    return method,tag,removed,_worker.solve(method,removed,first=(tag=="A_full"))


def execute(study,tasks,workers):
    if workers==1:
        _init(study.scene,study._core_path,study.rho)
        yield from map(_solve,tasks)
        return
    ctx=mp.get_context("spawn")
    with ctx.Pool(workers,_init,(study.scene,study._core_path,study.rho)) as pool:
        yield from pool.imap_unordered(_solve,tasks,chunksize=1)


def loo(scene,core,out,workers):
    s=Study(scene,core)
    s._core_path=core
    jobs=[(m,"A_full",()) for m in METHODS]
    jobs.extend((m,"C_single",(i,)) for i in s.ids for m in METHODS)
    results=[]
    for n,x in enumerate(execute(s,jobs,workers),1):
        results.append(x)
        if n%30==0 or n==len(jobs):
            print(scene,n,"/",len(jobs),"optimizations",flush=True)
    # No evaluator was available to any ablation selection or solver.
    truth=s.truth()
    complete={m:r for m,t,_d,r in results if t=="A_full"}
    rows=[]
    for m,t,removed,r in sorted(results,key=lambda x:(x[0],x[1],x[2])):
        if t=="C_single":
            rows.append(dict(method=m,edge_id=removed[0],
                         result=score(r,truth,complete[m].get("pose"),s.nodes[removed[0]])))
    document=s.receipt()|dict(schema="ocean_range_v5_exact_loo",workers=workers,
        A={m:score(r,truth) for m,r in complete.items()},C=rows,
        positive_delta="dropping raises error; batch helped conditionally in this fixed-graph experiment")
    write_json(out,document)
    return document


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--scene",choices=SCENES,required=True)
    p.add_argument("--core",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=6 or a.output.exists():
        p.error("Use 1..6 workers and a new output file")
    loo(a.scene,a.core,a.output,a.workers)
