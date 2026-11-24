"""Multi-scene replay on immutable factor streams, no association or raw-data export."""
import argparse
from collections import Counter
from dataclasses import fields
import inspect
import json
from pathlib import Path
import time

import numpy as np

from standalone.SideScanMappedRangeChange import MappedRangeChange
from research.sonar_range_factor_v2.models import Track,physical_interval
from research.sonar_range_factor_v3.adapter import make_m0
from research.sonar_range_factor_v3.likelihood import RadialLoss,matched_huber
from research.sonar_range_factor_v3.replay import sha
from .adaptive import decide,load_policy
from .batch import Batch
from .graph import make_batch,make_hybrid

ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/"docs/reports/seabed_matching_improved_algorithm_20261107/evidence"
SCENES={"qiandao16":"main_independent_comparison","qiandao4":"fresh_independent_comparison",
        "qiandao2":"preserved_development"}
ARMS=tuple((m+("_loop" if loop else ""),m,loop)
           for loop in (False,True) for m in ("no_range","M0","M1_Gaussian","M1_Robust","Adaptive"))


def load_scene(scene):
    source=json.loads((PREVIOUS/(SCENES[scene]+".json")).read_text())
    candidate=Path(source["candidate_run"])
    truth=Path(source["source_run"])/"navigation_states.npz"
    graph=candidate/"graph_factors.json"
    trace=candidate/"mapped_range_change_trace.json"
    hashes={"graph":sha(graph),"trace":sha(trace),"truth":sha(truth)}
    if hashes["graph"]!=source["replay_outputs_SHA256"]["graph_factors.json"] or (
            hashes["truth"]!=source["source_states_SHA256"]):
        raise ValueError("Graph or truth differs from the archived evidence")
    return json.loads(graph.read_text()),json.loads(trace.read_text()),truth,hashes,source


def replay(scene,policy_file,frozen_core,rho):
    import gtsam
    from oceansim.slam import pose_graph as core
    cp=Path(inspect.getfile(core)).resolve()
    if not cp.is_relative_to(frozen_core.resolve()) or not np.isfinite(rho) or not -1<rho<1:
        raise ValueError("Frozen core and SPD correlation required")
    p,tr,truth,hashes,source=load_scene(scene)
    z=np.asarray(p["pose_initializations"])
    ts=np.asarray(p["pose_timestamps_s"])
    if p["schema"]!="oceansim.planar_pose_graph_factor_stream.v1" or (
            z.shape!=(len(ts),3) or (np.diff(ts)<=0).any()):
        raise ValueError("Inconsistent initial values or graph schema")
    base=core.GTSAMPlanarPoseGraph(core.PoseGraphConfig(**p["objective_configuration"]))
    oldfields={x.name for x in fields(MappedRangeChange)}
    corr=np.array([[1.,rho],[rho,1.]])
    policy=load_policy(policy_file)
    rows,decisions,modecount,alltokens={},Counter(),Counter(),set()
    robust=matched_huber(2)
    gate_work=[]
    mapped=p["mapped_range_changes"]
    if len(tr)!=len(mapped) or len(mapped)!=sum(x["kind"]=="mapped_range_change" for x in p["edges"]):
        raise ValueError("Archived pair stream and observing trace disagree")
    for rec,trace in zip(mapped,tr):
        i,a,b=rec["edge_index"],rec["source"],rec["target"]
        edge=p["edges"][i]
        if (a,b)!=(edge["source"],edge["target"]) or edge["kind"]!="mapped_range_change" or (
                a,b)!=(trace["source"],trace["target"]) or i in rows:
            raise ValueError("Archived range keys or provenance do not match")
        old=MappedRangeChange(**{k:v for k,v in rec["observations"].items() if k in oldfields})
        tokens=[tuple(t) for t in trace["received_tokens"]]
        if trace["features"]!=old.features or len(tokens)!=2*old.features or (
                any(len(t)!=3 or not all(isinstance(x,int) and x>=0 for x in t) for t in tokens)) or (
                len(set(tokens))!=len(tokens) or not alltokens.isdisjoint(tokens)):
            raise ValueError("Missing, malformed, or reused original range tokens")
        alltokens.update(tokens)
        tracks=[]
        for k in range(old.features):
            r=np.array([old.range_a[k],old.range_b[k]])
            elevations=np.array([old.sensor_z_a[k],old.sensor_z_b[k]])
            g=2*r*np.array([1.,-1.])/r.sum()
            cov=corr*(old.standard_deviation_m[k]**2/(g@corr@g))
            tracks.append(Track(old.landmark_xy[k],np.stack([old.origin_a[k],old.origin_b[k]]),
                                elevations,r,cov,physical_interval(elevations,old.maximum_range_m)))
        t0=time.perf_counter()
        gate=decide(Batch(tracks,robust),z[[a,b]],old,np.full(old.features,np.nan),
                    policy,arrival_s=trace["timestamp_s"],node_times_s=ts[[a,b]])
        gate_work.append((time.perf_counter()-t0)*1000)
        rows[i]=(old,tracks,gate)
        decisions.update(gate.reason.tolist())
        modecount.update(gate.mode.tolist())

    def build(method,loop,rebuild):
        value=gtsam.Values()
        graph=gtsam.NonlinearFactorGraph()
        graph.add(base._gauge_factor(np.asarray(p["gauge_pose"])))
        for i,pose in enumerate(z):
            value.insert(base._key(i),base._pose2(pose))
        for i,record in enumerate(p["edges"]):
            kind=record["kind"]
            if kind=="loop" and not loop:
                continue
            if kind!="mapped_range_change":
                q=dict(record)
                q["measurement"],q["information"]=map(np.asarray,(q["measurement"],q["information"]))
                graph.add(base._edge_factor(core.PoseGraphEdge(**q)))
            elif method!="no_range":
                old,t,gate=rows[i]
                keys=base._key(record["source"]),base._key(record["target"])
                if method=="M0":
                    f=make_m0(old,*keys)
                elif method=="Adaptive":
                    f=make_hybrid(t,old,keys,robust,gate,policy,rebuild_as_m0=rebuild)
                else:
                    f=make_batch(t,keys,robust if method=="M1_Robust" else RadialLoss())
                if f is not None:
                    graph.add(f)
        for record in p["priors"]:
            q=dict(record)
            q["measurement"],q["information"]=map(np.asarray,(q["measurement"],q["information"]))
            graph.add(base._prior_factor(core.PoseGraphPrior(**q)))
        return graph,value

    def optimize(method,loop,rebuild):
        g,v=build(method,loop,rebuild)
        before=g.error(v)
        cfg=gtsam.LevenbergMarquardtParams()
        cfg.setMaxIterations(60)
        s=gtsam.LevenbergMarquardtOptimizer(g,v,cfg)
        y=s.optimize()
        objective=g.error(y)
        est=np.array([base._pose_array(y.atPose2(base._key(i))) for i in range(len(z))])
        if not np.isfinite(est).all() or not np.isfinite(objective):
            raise ValueError("nonfinite optimizer result")
        return est,dict(factors=g.size(),iterations=s.iterations(),cost_before=before,cost_after=objective)

    out={}
    staged={}
    for name,method,loop in ARMS:
        t0=time.perf_counter()
        fallback=None
        try:
            try:
                staged[name],out[name]=optimize(method,loop,False)
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                if method!="Adaptive":
                    raise
                fallback=str(exc)
                staged[name],out[name]=optimize(method,loop,True)
            out[name].update(status="returned_rebuilt_M0" if fallback else "returned_estimate",fallback=fallback)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            out[name]=dict(status="failed",reason=str(exc),fallback=fallback)
        out[name]["wall_ms"]=(time.perf_counter()-t0)*1000
        print(scene,name,out[name]["status"],out[name]["wall_ms"],flush=True)
    # Score only after every arm is frozen. No pose samples leave this process.
    with np.load(truth,allow_pickle=False) as data:
        stamp=data["timestamps"]
        xy=data["positions"][:,0,:2]
    i=np.searchsorted(stamp,ts)
    if (i>=len(stamp)).any() or not np.array_equal(stamp[i],ts):
        raise ValueError("No exact node timestamps in historical truth")
    for name,est in staged.items():
        e=np.linalg.norm(est[:,:2]-xy[i],axis=1)
        out[name].update(rmse_m=float(np.sqrt(np.mean(e**2))),p95_m=float(np.quantile(e,.95)),
                         last_m=float(e[-1]),max_m=float(e.max()))
    tracked_files=[Path(__file__).with_name(f) for f in
                   ("batch.py","adaptive.py","graph.py","replay.py")]
    return dict(schema="ocean_range_v4_frozen_graph",scene=scene,seed=source["source_seed"],
                distance_m=source["distance_m"],hashes=hashes,
                policy_sha256=sha(policy_file),frozen_core_sha256=sha(cp),
                code_sha256={f.name:sha(f) for f in tracked_files},
                nodes=len(z),range_pairs=len(alltokens)//2,range_batches=len(rows),
                loop_edges=sum(x["kind"]=="loop" for x in p["edges"]),map_priors=len(p["priors"]),
                association_confidence="unknown; accepted historical pairs with unique token trace",
                archived_covariance_assumption=f"rho={rho}, equal raw marginals matching frozen M0 scalar difference variance",
                adaptive_initial_modes=dict(modecount),adaptive_reasons=dict(decisions),
                gate_median_ms=float(np.median(gate_work)),gate_p95_ms=float(np.quantile(gate_work,.95)),arms=out)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene",choices=SCENES,required=True)
    parser.add_argument("--frozen-core",type=Path,required=True)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rho",type=float,default=.25)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt path")
    r=replay(args.scene,args.policy,args.frozen_core,args.rho)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(r,f,indent=2,allow_nan=False)
        f.write("\n")


if __name__=="__main__":
    main()
