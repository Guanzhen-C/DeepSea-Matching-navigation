"""Batch replay of an archived simulation factor graph; no re-association."""
import argparse
from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from standalone.SideScanMappedRangeChange import MappedRangeChange
from research.sonar_range_factor_v2.models import Track, physical_interval
from .adapter import make_m0, make_m1
from .likelihood import RadialLoss, matched_huber


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(graph_file,truth_file,frozen_core):
    import gtsam
    from oceansim.slam import pose_graph as core
    core_path=Path(inspect.getfile(core)).resolve()
    if not core_path.is_relative_to(frozen_core.resolve()):
        raise ValueError("PYTHONPATH must select the requested frozen core")
    payload=json.loads(graph_file.read_text())
    p=np.asarray(payload["pose_initializations"])
    stamps=np.asarray(payload["pose_timestamps_s"])
    if p.shape!=(len(stamps),3) or np.any(np.diff(stamps)<=0):
        raise ValueError("Invalid initialization or timestamps")
    if payload["schema"]!="oceansim.planar_pose_graph_factor_stream.v1":
        raise ValueError("Unexpected factor schema")
    cfg=core.PoseGraphConfig(**payload["objective_configuration"])
    base=core.GTSAMPlanarPoseGraph(cfg)
    mapped={item["edge_index"]:item for item in payload["mapped_range_changes"]}
    observed=[i for i,e in enumerate(payload["edges"]) if e["kind"]=="mapped_range_change"]
    if sorted(mapped)!=observed or len(mapped)!=len(payload["mapped_range_changes"]) or not mapped:
        raise ValueError("Stored edges and nonlinear factors do not agree")
    kwargs={f.name for f in fields(MappedRangeChange)}
    counts={}
    for e in payload["edges"]:
        counts[e["kind"]]=counts.get(e["kind"],0)+1
    counts["priors"]=len(payload["priors"])
    counts["nodes"]=len(p)

    def graph_and_initial(method,corr):
        graph=gtsam.NonlinearFactorGraph()
        graph.add(base._gauge_factor(np.asarray(payload["gauge_pose"])))
        values=gtsam.Values()
        for i,pose in enumerate(p):
            values.insert(base._key(i),base._pose2(pose))
        for i,edge in enumerate(payload["edges"]):
            a,b=edge["source"],edge["target"]
            if i not in mapped:
                e=dict(edge)
                e["measurement"],e["information"]=map(np.asarray,(e["measurement"],e["information"]))
                graph.add(base._edge_factor(core.PoseGraphEdge(**e)))
                continue
            record=mapped[i]
            if (a,b)!=(record["source"],record["target"]):
                raise ValueError("Mismatched factor keys")
            obs={k:v for k,v in record["observations"].items() if k in kwargs}
            pair=MappedRangeChange(**obs)
            if method=="M0":
                graph.add(make_m0(pair,base._key(a),base._key(b)))
            elif method!="no_range":
                shape=np.array([[1.,corr],[corr,1.]])
                for k in range(pair.features):
                    r=np.array([pair.range_a[k],pair.range_b[k]])
                    z=np.array([pair.sensor_z_a[k],pair.sensor_z_b[k]])
                    gradient=2*r*np.array([1.,-1.])/r.sum()
                    # Same difference variance as archived M0. Raw covariance
                    # is NOT identifiable from this archived scalar.
                    sigma_sq=pair.standard_deviation_m[k]**2/(gradient@shape@gradient)
                    t=Track(pair.landmark_xy[k],np.stack([pair.origin_a[k],pair.origin_b[k]]),
                            z,r,sigma_sq*shape,physical_interval(z,pair.maximum_range_m))
                    loss=matched_huber(2) if method=="M1_Robust" else RadialLoss()
                    graph.add(make_m1(t,base._key(a),base._key(b),loss))
        for prior in payload["priors"]:
            e=dict(prior)
            e["measurement"],e["information"]=map(np.asarray,(e["measurement"],e["information"]))
            graph.add(base._prior_factor(core.PoseGraphPrior(**e)))
        return graph,values

    results,estimates={},{}
    for name,corr in [("no_range",0.),("M0",0.),("M1_Gaussian",.25),("M1_Robust",.25)]:
        start=time.perf_counter()
        try:
            graph,values=graph_and_initial(name,corr)
            before=graph.error(values)
            params=gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(60)
            solver=gtsam.LevenbergMarquardtOptimizer(graph,values,params)
            estimate=solver.optimize()
            final=graph.error(estimate)
            states=np.array([base._pose_array(estimate.atPose2(base._key(i))) for i in range(len(p))])
            if not np.isfinite(states).all() or not np.isfinite(final):
                raise ValueError("Graph produced nonfinite estimates")
            estimates[name]=states
            results[name]=dict(status="returned_estimate",raw_pair_rho_assumption=corr,
                               graph_factors=graph.size(),cost_initial=before,cost_final=final,
                               iterations=solver.iterations())
        except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            results[name]=dict(status="failed",error=str(exc),raw_pair_rho_assumption=corr)
        results[name]["wall_ms"]=(time.perf_counter()-start)*1000
        print(name,results[name]["status"],flush=True)
    # Open truth only after every branch is finalized. Do not emit coordinates.
    with np.load(truth_file,allow_pickle=False) as archive:
        expected=np.asarray(archive["timestamps"])
        xy=np.asarray(archive["positions"])[:,0,:2]
    idx=np.searchsorted(expected,stamps)
    if np.any(idx>=len(expected)) or not np.array_equal(expected[idx],stamps):
        raise ValueError("Graph timestamps do not bind exactly to source truth")
    for name,states in estimates.items():
        e=np.linalg.norm(states[:,:2]-xy[idx],axis=1)
        results[name].update(RMSE_xy_m=float(np.sqrt(np.mean(e*e))),P95_xy_m=float(np.quantile(e,.95)),
                             final_xy_m=float(e[-1]),max_xy_m=float(e.max()))
    return dict(schema="ocean_m1_fixed_graph_replay_v3",scope="archived_simulation_batch_graph_no_reassociation",
                factor_graph_sha256=sha(graph_file),source_truth_sha256=sha(truth_file),
                frozen_core_sha256=sha(core_path),replay_code_sha256=sha(Path(__file__)),
                counts=counts,range_pairs=sum(len(x["observations"]["range_a"]) for x in mapped.values()),
                batches=results,solver="GTSAM batch LM max_iterations=60, all modes same initialization",
                range_covariance="rho=0.25 assumed for M1; archived difference sigma matched algebraically, raw covariance unavailable")


def main():
    parser=argparse.ArgumentParser(__doc__)
    for x in ["graph","truth","frozen-core","output"]:
        parser.add_argument("--"+x,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():
        parser.error("Do not overwrite an existing receipt")
    result=replay(a.graph,a.truth,a.frozen_core)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")


if __name__=="__main__":
    main()
