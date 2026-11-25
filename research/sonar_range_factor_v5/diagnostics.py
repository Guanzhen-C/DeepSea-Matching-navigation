"""Pair-level diagnostics and truth-free decisions from archived graph prefixes."""
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from standalone.SideScanMappedRangeChange import profiled_range_residual_jacobians,world_origins
from research.sonar_range_factor_v4.adaptive import decide,load_policy,physical_tube
from research.sonar_range_factor_v3.replay import sha
from .replay import METHODS,Study,score,write_json

# Geometry thresholds are predeclared in SI on the conditional target XY block.
# They define an exploratory structural ablation, not production quality gates.
RANK_ABSOLUTE=.10
RANK_RATIO=.10
HIGH_QUALITY_TUBE_FRACTION=.60
YAW_LEVER_M=20.


def chart(p):
    c,s=np.cos(p[2]),np.sin(p[2])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def finite(x):
    """Portable JSON; never serialize NaN as if it were an observed value."""
    if isinstance(x,np.ndarray):
        return finite(x.tolist())
    if isinstance(x,(list,tuple)):
        return [finite(y) for y in x]
    if isinstance(x,dict):
        return {k:finite(y) for k,y in x.items()}
    if isinstance(x,(np.floating,float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):
        return int(x)
    if isinstance(x,np.bool_):
        return bool(x)
    return x


def decompose(J):
    J=np.asarray(J,float)
    if not len(J):
        return dict(singular=[],rank=0,condition=None,right_strongest=None)
    _,s,V=np.linalg.svd(J,full_matrices=False)
    tol=max(1e-12,(float(s[0])*1e-8) if len(s) else 1e-12)
    rank=int((s>tol).sum())
    rank_full=rank==J.shape[1]
    vector=V[0].copy() if rank else None
    if vector is not None and vector[np.argmax(abs(vector))]<0:
        vector=-vector
    return dict(singular=s,rank=rank,condition=float(s[0]/s[-1]) if rank_full else None,
                right_strongest=vector)


def prefix_predictions(study):
    """Pre-boundary full-history iSAM2 proxy using only committed archive prefixes.

    This is explicitly not a reconstruction of actual runtime fixed-lag
    smoothing, map/odom publication or the true predictor covariance.
    """
    g=study.gtsam
    solver=g.ISAM2(study.base._new_isam_params())
    nprev=eprev=pprev=0
    result={}
    for k,b in enumerate(study.payload["optimization_boundaries"]):
        n,e,p=(b["poses"],b["edges"],b["priors"])
        if n<nprev or e<eprev or p<pprev:
            raise ValueError("Nonmonotone frozen optimization prefix")
        edges=[i for i in range(eprev,e) if i in study.old]
        if edges:
            boundary_available=study.times[nprev-1] if nprev else -np.inf
            previous=solver.calculateEstimate()
            for i in edges:
                if boundary_available>study.arrival[i]:
                    raise ValueError("A future key would enter the causal diagnostic")
                arr=[]
                covs=[]
                state_source=[]
                for node in study.nodes[i]:
                    key=study.base._key(node)
                    known=node<nprev and previous.exists(key)
                    row=study.base._pose_array(previous.atPose2(key)) if known else study.initial[node]
                    cov=None
                    if known:
                        try:
                            trial=np.asarray(solver.marginalCovariance(key))
                            if trial.shape==(3,3) and np.isfinite(trial).all() and np.linalg.eigvalsh(
                                (trial+trial.T)*.5).min()>=-1e-8:
                                cov=trial
                        except (RuntimeError,ValueError,np.linalg.LinAlgError):
                            pass
                    arr.append(row)
                    covs.append(cov)
                    state_source.append("previous_prefix_global_isam2" if known else "archived_node_initial_only")
                result[i]=dict(poses=np.stack(arr),covs=covs,source=state_source,boundary_index=k,
                               n_previous_poses=nprev,n_previous_edges=eprev,n_previous_priors=pprev,
                               last_committed_node_time_s=float(boundary_available))
        graph=g.NonlinearFactorGraph()
        if nprev==0:
            graph.add(study.gauge)
        for j in range(eprev,e):
            graph.add(study.range[j]["M0"] if j in study.old else study.edges[j])
        for j in range(pprev,p):
            graph.add(study.priors[j])
        if n!=nprev or graph.size():
            solver.update(graph,study.values(study.initial[nprev:n],nprev))
        nprev,eprev,pprev=n,e,p
    if set(result)!=set(study.ids):
        raise ValueError("Some original factors have no causal optimization boundary")
    return result


def source_relationships(s):
    archive=Path(s.source["candidate_run"])
    path=archive/"historical_loop_trace.json"
    trace=json.loads(path.read_text())
    loop=[x for x in trace if x.get("actual_loop_edge_added")]
    # No verified range token lists are exported for either map priors or
    # loop patches. Same-window/endpoint is not exact repeated raw reception.
    by_time={(x["factor"]["target"],float(x["timestamp_s"])) for x in loop}
    endpoints={(x["factor"]["source"],x["factor"]["target"]) for x in loop}
    mapnodes={x["target"] for x in s.payload["priors"] if x.get("kind")=="side_scan_feature_prior"}
    return {i:dict(range_association_via_confirmed_map=True,
            map_prior_at_target_node=s.nodes[i][1] in mapnodes,
            identical_loop_current_window=(s.nodes[i][1],s.arrival[i]) in by_time,
            identical_loop_endpoints=s.nodes[i] in endpoints,
            exact_map_token_overlap=None,exact_loop_token_overlap=None)
            for i in s.ids},sha(path)


def snapshots(s,i,p):
    old,b=s.old[i],s.batch[i]
    oa,_=world_origins(p[0],old.origin_a)
    ob,_=world_origins(p[1],old.origin_b)
    za,zb=old.sensor_z_a,old.sensor_z_b
    L=old.landmark_xy
    da,db=np.sum((L-oa)**2,axis=1),np.sum((L-ob)**2,axis=1)
    dz=zb-za
    mid=(za+zb)*.5
    lo=np.minimum(za,zb)-old.maximum_range_m-3*old.pressure_std_m
    hi=np.minimum(za,zb)+3*old.pressure_std_m
    D=db-da+old.range_a**2-old.range_b**2
    edge0,edge1=2*dz*(lo-mid),2*dz*(hi-mid)
    div=old.range_a+old.range_b
    res=(D-np.clip(D,np.minimum(edge0,edge1),np.maximum(edge0,edge1)))/div
    m0h=np.full(len(dz),np.nan)
    ii=abs(dz)>0
    m0h[ii]=np.clip(mid[ii]+D[ii]/(2*dz[ii]),lo[ii],hi[ii])
    E,A,B=profiled_range_residual_jacobians(*p,old)
    M=np.block([[chart(p[0]),np.zeros((3,3))],[np.zeros((3,3)),chart(p[1])]])
    J0=np.column_stack((A,B))[:,None,:]@M
    d=b.evaluate(p)
    H=d.height
    d2=np.column_stack((da,db))
    rho=np.sqrt(d2+(H[:,None]-b.z)**2)
    raw=rho-b.r
    white=np.einsum("nij,nj->ni",b.C,raw)
    good=np.char.startswith(d.status,"ok")
    J1=d.jacobian@M
    J1=np.where(good[:,None,None],J1,np.nan)
    per=dict(centers_world=np.stack([np.column_stack((oa,za)),np.column_stack((ob,zb))],axis=1),
             signed_height_difference=dz,
             M0=dict(height_profile=m0h,unweighted_height_curvature=(2*dz/(div*old.standard_deviation_m))**2,
                     unweighted_profile_cost=.5*(res/old.standard_deviation_m)**2,
                     residual_m=res,white=res/old.standard_deviation_m,final_residual=E[:,None],
                     jacobian_local=J0,information=J0.transpose(0,2,1)@J0),
             M1=dict(height_profile=np.where(good,H,np.nan),unweighted_height_curvature=np.where(good,d.curvature,np.nan),
                     unweighted_profile_cost=np.where(good,d.cost,np.nan),
                     residual_m=np.where(good[:,None],raw,np.nan),white=np.where(good[:,None],white,np.nan),
                     final_residual=np.where(good[:,None],d.residual,np.nan),status=d.status.tolist(),
                     jacobian_local=J1,information=J1.transpose(0,2,1)@J1))
    # For local geometry, all headings use a fixed scaling for m/rad. The
    # conditional target XY slice is explicitly NOT a graph marginal.
    scale=np.array([1,1,1/YAW_LEVER_M,1,1,1/YAW_LEVER_M])
    col0=J0.reshape(-1,6)
    col1=J1[good].reshape(-1,6)
    obs={}
    for name,J in [("M0",col0),("M1_Robust",col1)]:
        obs[name]=dict(both_scaled=decompose(J*scale),conditional_target_xy=decompose(J[:,3:5]))
    return per,obs


def tube_prediction(batch,poses,opt,covs):
    """Approximate per-range inflation; cross-key covariance is NOT reconstructed."""
    v,u,d=batch.geom(poses)
    h=np.clip(opt.height,batch.lo,batch.hi)
    r=np.sqrt(d+(h[:,None]-batch.z)**2)
    var=np.stack([t.covariance.diagonal() for t in batch.tracks])
    for j,P in enumerate(covs):
        if P is None:
            continue
        dy=np.column_stack((-u[:,j,1],u[:,j,0]))
        A=np.column_stack((v[:,j],np.sum(v[:,j]*dy,axis=1)))
        with np.errstate(divide="ignore",invalid="ignore"):
            A=-(A/r[:,j,None])@chart(poses[j])
        variance=np.einsum("ni,ij,nj->n",A,P,A)
        var[:,j]+=np.maximum(0,np.where(np.isfinite(variance),variance,0.))
    return np.sqrt(var)


def gate_diagnosis(s,i,pred,policy):
    b=s.batch[i]
    ids=s.nodes[i]
    seed=s.initial[list(ids)]
    p=pred["poses"]
    kwargs=dict(arrival_s=s.arrival[i],node_times_s=s.times[list(ids)])
    confidence=np.full(len(b.tracks),np.nan)
    d0=decide(b,seed,s.old[i],confidence,policy,**kwargs)
    d=decide(b,p,s.old[i],confidence,policy,**kwargs)
    out=b.evaluate(p,max_boxes=policy.max_boxes,allow_reference=False)
    std=tube_prediction(b,p,out,pred["covs"])
    shadow=physical_tube(b,p,policy.tolerance_sigma*std)
    margin=np.minimum(out.height-b.lo,b.hi-out.height)
    tests=dict(narrow_deadband=d.deadband_sigma<policy.deadband_min_sigma,
        profile_not_stable=out.status!="ok",low_curvature=~np.isfinite(out.curvature)|(out.curvature<policy.min_curvature),
        near_boundary=~np.isfinite(margin)|(margin<policy.min_bound_margin_m),
        large_radius=~np.isfinite(out.cost)|(2*out.cost>policy.radius_limit**2),
        tube_fail=~d.feasible)
    ok=np.ones(len(d.mode),bool)
    for k,condition in tests.items():
        if k!="tube_fail":
            ok&=~condition
    return dict(initial_mode=d0.mode,initial_reason=d0.reason,causal_mode=d.mode,causal_reason=d.reason,
        independent_failures=tests,initial_tube=d0.feasible,causal_tube=d.feasible,
        proxy_cov_available=np.array([x is not None for x in pred["covs"]]),
        shadow_tube=shadow,shadow_geometry_only_m1=shadow&ok,
        standard_deviation_with_marginals=std,max_budget_boxes=out.work,deadband_sigma=d.deadband_sigma,
        raw_profile_status=out.status)


def select_masks(s,records,relationships):
    buckets={k:[] for k in ["D_q1","D_q2","D_q3","D_q4","E_overlap_risk","F_weak_geometry","G_only_high_quality"]}
    denom=s.times[-1]-s.times[0]
    for i in s.ids:
        fraction=(s.arrival[i]-s.times[0])/denom
        q=min(3,max(0,int(4*fraction)))
        buckets[f"D_q{q+1}"].append(i)
        risk=relationships[i]["identical_loop_current_window"] or relationships[i]["identical_loop_endpoints"]
        if risk:
            buckets["E_overlap_risk"].append(i)
        info=records[i]["observability_causal"]
        horizontal=[info[k]["conditional_target_xy"]["singular"] for k in METHODS]
        strong=[len(v)>=2 and v[1]>=RANK_ABSOLUTE and v[1]>=RANK_RATIO*v[0] for v in horizontal]
        if not any(strong):
            buckets["F_weak_geometry"].append(i)
        if risk or not any(strong) or records[i]["causal_tube_fraction"]<HIGH_QUALITY_TUBE_FRACTION:
            buckets["G_only_high_quality"].append(i)
    return buckets


def increments(s,p0,p1,ids):
    return [np.asarray(s.base._pose2(p0[j]).localCoordinates(s.base._pose2(p1[j]))) for j in ids]


def _row(x,j,n):
    if isinstance(x,dict):
        return {k:_row(v,j,n) for k,v in x.items()}
    if isinstance(x,(np.ndarray,list)) and len(x)==n:
        return x[j]
    return x


def generate(scene,core,policy_path,detail_path,out):
    s=Study(scene,core)
    pred=prefix_predictions(s)
    relation,loop_hash=source_relationships(s)
    policy=load_policy(policy_path)
    references={m:s.solve(m,[],first=True) for m in METHODS}
    records={}
    counters=Counter()
    native0,native1=Counter(),Counter()
    dx=[]
    dtheta=[]
    budget=[]
    detail_path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(detail_path,"xt",encoding="utf-8",compresslevel=4) as f:
        for i in s.ids:
            p=pred[i]
            ids=s.nodes[i]
            before,_=snapshots(s,i,s.initial[list(ids)])
            ca,obs=snapshots(s,i,p["poses"])
            after={}
            delta={}
            for m,result in references.items():
                if "pose" in result:
                    after[m],_=snapshots(s,i,result["pose"][list(ids)])
                    delta[m]=dict(init_to_first_GTSAM_LM=increments(s,s.initial,result["one"],ids),
                                  init_to_final_GTSAM_LM=increments(s,s.initial,result["pose"],ids))
            g=gate_diagnosis(s,i,p,policy)
            old=s.old[i]
            n=old.features
            pxy=np.linalg.norm(s.initial[list(ids),:2]-p["poses"][:,:2],axis=1)
            dx.extend(pxy.tolist())
            angles=s.initial[list(ids),2]-p["poses"][:,2]
            dtheta.extend(abs(np.arctan2(np.sin(angles),np.cos(angles))).tolist())
            budget.extend(g["max_budget_boxes"].tolist())
            records[i]=dict(edge_id=i,features=n,association=relation[i],observability_causal=obs,
                            causal_tube_fraction=float(g["causal_tube"].mean()),
                            batch_cov_endpoints=int(g["proxy_cov_available"].sum()),
                            initial_to_causal_shift_m=pxy)
            native0.update(g["initial_reason"].tolist())
            native1.update(g["causal_reason"].tolist())
            counters.update({k:int(x.sum()) for k,x in g["independent_failures"].items()})
            counters.update(initial_tube=int(g["initial_tube"].sum()),causal_tube=int(g["causal_tube"].sum()),
                            marginal_shadow_tube=int(g["shadow_tube"].sum()),
                            shadow_geometry_only_m1=int(g["shadow_geometry_only_m1"].sum()),
                            initial_m1=int(np.count_nonzero(g["initial_mode"]=="M1")),
                            causal_m1=int(np.count_nonzero(g["causal_mode"]=="M1")),
                            marginals_both_pairs=n if g["proxy_cov_available"].all() else 0)
            for j in range(n):
                row=dict(scene=scene,edge_id=i,pair_in_batch=j,endpoints=ids,
                         arrival_s=s.arrival[i],tokens=s.tokens[i][2*j:2*j+2],
                         map_L=old.landmark_xy[j],original_ranges_m=[old.range_a[j],old.range_b[j]],
                         phase_centres_in_node=[old.origin_a[j],old.origin_b[j]],raw_covariance_assumed=s.tracks[i][j].covariance,
                         observation_relation=relation[i],causal_prediction=p,
                         gate={k:(_row(v,j,n) if k!="proxy_cov_available" else v) for k,v in g.items()},
                         snapshots=dict(archived_initial=_row(before,j,n),causal=_row(ca,j,n),
                                        final_each_model={m:_row(x,j,n) for m,x in after.items()}),
                         optimizer_actual_endpoints=delta)
                f.write(json.dumps(finite(row),allow_nan=False,separators=(",",":"))+"\n")
    masks=select_masks(s,records,relation)
    truth=s.truth()  # scoring AFTER every diagnostic and selection
    report=s.receipt()|dict(schema="ocean_range_v5_pair_diagnostics",detail_local=str(detail_path),
        detail_sha256=sha(detail_path),code_sha256=sha(Path(__file__)),loop_trace_sha256=loop_hash,
        gate_policy_sha256=sha(policy_path),decisions_initial=dict(native0),decisions_prefix=dict(native1),
        counts=dict(counters),masks=masks,
        geometry_thresholds=dict(absolute=RANK_ABSOLUTE,ratio=RANK_RATIO,quality_tube=HIGH_QUALITY_TUBE_FRACTION,
                                 yaw_lever_m=YAW_LEVER_M),
        predictor_shift_m=dict(median=float(np.median(dx)),p95=float(np.quantile(dx,.95)),max=float(max(dx))),
        predictor_shift_heading_deg_p95=float(np.rad2deg(np.quantile(dtheta,.95))),
        budget=dict(median_boxes=float(np.median(budget)),max_boxes=int(max(budget))),
        solver_reference={m:score(x,truth) for m,x in references.items()},
        association_exact_token_overlap_identifiable=False,
        batches_detail=[records[i] for i in s.ids])
    write_json(out,finite(report))
    return report


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--scene",required=True)
    for x in ("core","policy","detail","output"):
        p.add_argument("--"+x,required=True,type=Path)
    args=p.parse_args()
    if args.output.exists() or args.detail.exists():
        p.error("Choose unused output paths")
    generate(args.scene,args.core,args.policy,args.detail,args.output)
