"""Read-only reference replay, archive inventory and held-out gate probes."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from standalone.SideScanMappedRangeChange import MappedRangeChange
from research.sonar_range_factor_v2.models import Track,physical_interval
from research.sonar_range_factor_v3.adapter import _two_pose_factor
from research.sonar_range_factor_v3.likelihood import matched_huber,profile_pair
from research.sonar_range_factor_v3.replay import sha
from . import replay as stream
from .adaptive import decide,load_policy
from .batch import Batch
from .online_probe import fixed_lag_probe


def scalar_factor(tracks,keys,loss):
    def f(p):
        e,j=[],[]
        for t in tracks:
            a,b,_=profile_pair(t,p,loss)
            e.append(a)
            j.append(b)
        return np.concatenate(e),np.concatenate(j)
    return _two_pose_factor(2*len(tracks),*keys,f)


def heldout(policy_file):
    # Fixed synthetic inputs independent from the calibration, never use a
    # navigation error to select actions or label archival associations.
    seed=724334
    rng=np.random.default_rng(seed)
    p=np.array([[0.,0.,.12],[7.,4.,.4]])
    policy=load_policy(policy_file)
    labels={}
    for kind in ("large_dz_unknown","small_dz_unknown","verified_low_association","gross_incompatibility"):
        rows=[]
        for _ in range(64):
            z=np.array([0.,rng.uniform(12,18) if kind!="small_dz_unknown" else rng.uniform(.03,.12)])
            q=rng.uniform([28.,32.],[55.,85.])
            h=rng.uniform(-26.,-14.)
            r=np.sqrt(np.sum((q-p[:,:2])**2,axis=1)+(h-z)**2)+rng.normal(0.,.3,2)
            if kind=="gross_incompatibility":
                r[0]-=20.
            rows.append(Track(q,np.zeros((2,2)),z,r,[[.35,.08],[.08,.35]],physical_interval(z,250.)))
        x=lambda a:np.stack([getattr(t,a) for t in rows])
        old=MappedRangeChange(x("xy"),x("lever")[:,0],x("lever")[:,1],x("sensor_z")[:,0],
              x("sensor_z")[:,1],x("ranges")[:,0],x("ranges")[:,1],np.ones(len(rows)))
        conf=np.full(len(rows),.1 if kind=="verified_low_association" else np.nan)
        g=decide(Batch(rows,matched_huber(2)),p,old,conf,policy,arrival_s=40.,node_times_s=[5.,39.])
        labels[kind]=dict(modes=dict(Counter(g.mode.tolist())),reasons=dict(Counter(g.reason.tolist())))
    return dict(seed=seed,n_per_stratum=64,result=labels)


def archives():
    files={
        "Monterey":"demo/hil/jetson_20261021/sea_live_5km_canonical_v3/native_reference/native_metrics_pose_graph_factors.json",
        "Chelan":"demo/generalization/final_improved_multiterrain_20261015/chelan_full_route_seed31_confirmed_output_fix_20261016_pose_graph_factors.json"}
    data={}
    for label,path in files.items():
        f=stream.ROOT/path
        x=json.loads(f.read_text())
        data[label]=dict(relative_archive=path,sha256=sha(f),nodes=len(x["pose_initializations"]),
                         edges=dict(Counter(e["kind"] for e in x["edges"])),
                         has_original_dual_ranges=bool(x.get("mapped_range_changes")),
                         source_trace_present=(f.parent/"mapped_range_change_trace.json").exists(),
                         paired_replay_ready=False)
    return data


def run(policy,frozen):
    arms=stream.ARMS
    make=stream.make_batch
    try:
        stream.ARMS=(("M1_Gaussian","M1_Gaussian",False),("M1_Robust_loop","M1_Robust",True))
        stream.make_batch=scalar_factor
        old={}
        for name in ("qiandao2","qiandao4"):
            r=stream.replay(name,policy,frozen,.25)
            old[name]=dict(arms=r["arms"],hashes=r["hashes"],nodes=r["nodes"],range_pairs=r["range_pairs"])
    finally:
        stream.ARMS,stream.make_batch=arms,make
    return dict(schema="ocean_range_v4_reference_validation",scalar_v3=old,coverage=heldout(policy),
                archive_inventory=archives(),pc_fixed_lag_probe=fixed_lag_probe(),
                policy_sha256=sha(policy),code_sha256=sha(Path(__file__)))


if __name__=="__main__":
    cli=argparse.ArgumentParser(__doc__)
    for name in ("policy","frozen-core","output"):
        cli.add_argument("--"+name,required=True,type=Path)
    a=cli.parse_args()
    if a.output.exists():
        cli.error("Choose new evidence output")
    r=run(a.policy,a.frozen_core)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(r,f,indent=2,allow_nan=False)
        f.write("\n")
