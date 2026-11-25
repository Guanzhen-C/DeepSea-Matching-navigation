"""Batch/segment/geometry ablations and covariance sensitivity; no data re-association."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.sonar_range_factor_v3.replay import sha
from .replay import METHODS,Study,execute,score,write_json


def interaction(group_delta, singles):
    """MSE nonadditivity on fixed graph, not independent causal effects."""
    if group_delta is None or any(x is None for x in singles):
        return None
    return float(group_delta-sum(singles))


def summary_single(rows):
    v=np.asarray([x["result"]["delta_rmse_m"] for x in rows if "delta_rmse_m" in x["result"]])
    s=dict(total=len(rows),failed=sum(x["result"]["status"]!="returned" for x in rows),measurable=len(v))
    if not len(v):
        return s
    s.update(gain_when_retained_ge_1mm=int((v>=.001).sum()),loss_when_retained_ge_1mm=int((v<=-.001).sum()),
              abs_lt_1mm=int((abs(v)<.001).sum()),median_delta_rmse_m=float(np.median(v)),
              max_improves_by_removal_m=float(max(0,-v.min())),
              max_worsens_by_removal_m=float(max(0,v.max())))
    for label,fn in [("largest_gain_edge",max),("largest_loss_edge",min)]:
        row=fn((x for x in rows if "delta_rmse_m" in x["result"]),key=lambda x:x["result"]["delta_rmse_m"])
        s[label]=row["edge_id"]
    return s


def run(scene,core,diagnostic_path,loo_path,out,workers):
    s=Study(scene,core)
    s._core_path=core
    g=json.loads(diagnostic_path.read_text())
    solo=json.loads(loo_path.read_text())
    for r in (g,solo):
        if r["scene"]!=scene or r["archive_hashes"]!=s.hashes:
            raise ValueError("Masks or leave-one receipts not from this historical graph")
    if len(solo["C"])!=2*len(s.ids):
        raise ValueError("Missing attempted single-batch replays")
    # Read only the truth-free masks from the independent diagnostic tool;
    # none of A/C score fields participates in selecting an edge.
    masks={k:tuple(v) for k,v in g["masks"].items()}
    if any(len(x)!=len(set(x)) or not set(x)<=set(s.ids) for x in masks.values()):
        raise ValueError("Invalid held-out mask")
    masks["A_full"]=()
    masks["B_no_range"]=tuple(s.ids)
    # Even masks with no removals are executed and reported.
    jobs=[(m,k,d) for m in METHODS for k,d in masks.items()]
    opt={}
    for m,k,_d,r in execute(s,jobs,workers):
        opt[m,k]=r
    # Test raw-coefficient sensitivity independently without retuning model.
    sens={}
    for rho in (-.50,0.,.50,.75):
        trial=Study(scene,core,rho=rho)
        sens[str(rho)]=trial.solve("M1_Robust",[])
    truth=s.truth()
    loo_by_method={m:{x["edge_id"]:x["result"] for x in solo["C"] if x["method"]==m} for m in METHODS}
    table={}
    for key,val in masks.items():
        r=dict(removed=len(val),removed_pairs=sum(s.old[i].features for i in val),
               kept=len(s.ids)-len(val),edge_ids=list(val),models={})
        for m in METHODS:
            a=score(opt[m,key],truth,opt[m,"A_full"].get("pose"))
            if key.startswith("D_"):
                singles=[loo_by_method[m][i].get("delta_mse_m2") for i in val]
                a.update(sum_single_delta_mse_m2=sum(singles) if all(x is not None for x in singles) else None,
                         group_minus_sum_single_mse_m2=interaction(a.get("delta_mse_m2"),singles),
                         missing_single_comparisons=sum(x is None for x in singles))
            r["models"][m]=a
        table[key]=r
    token_digest=hashlib.sha256(json.dumps([(i,s.tokens[i]) for i in s.ids],separators=(",",":")).encode()).hexdigest()
    report=s.receipt()|dict(schema="ocean_range_v5_ablations",mask_receipt_sha256=sha(diagnostic_path),
        loo_receipt_sha256=sha(loo_path),code_sha256=sha(Path(__file__)),arms=table,
        C_summary={m:summary_single([x for x in solo["C"] if x["method"]==m]) for m in METHODS},
        H=dict(same_ordered_tokens=True,token_sha256=token_digest,pairs=sum(len(s.tokens[i])//2 for i in s.ids),
               distinction="M1 assumed full covariance; identical pair identity does not equate the likelihoods"),
        rho_sensitivity_M1={k:score(x,truth,opt["M1_Robust","A_full"].get("pose")) for k,x in sens.items()})
    write_json(out,report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--scene",required=True)
    for x in ("core","diagnostic","loo","output"):
        p.add_argument("--"+x,required=True,type=Path)
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=6 or a.output.exists():
        p.error("Use 1..6 workers and a new output file")
    run(a.scene,a.core,a.diagnostic,a.loo,a.output,a.workers)
