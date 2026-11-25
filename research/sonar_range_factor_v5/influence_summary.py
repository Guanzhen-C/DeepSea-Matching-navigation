"""Descriptive comparison of residual/information with exact graph leave-one effects."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from research.sonar_range_factor_v3.replay import sha
from .replay import METHODS,write_json


def run(detail,loo_path):
    loo=json.loads(loo_path.read_text())
    data=defaultdict(lambda:defaultdict(list))
    rows=0
    with gzip.open(detail,"rt",encoding="utf-8") as f:
        for line in f:
            d=json.loads(line)
            if d["scene"]!=loo["scene"]:
                raise ValueError("Mismatched detail and graph ablation")
            i=d["edge_id"]
            for m in METHODS:
                p=d["snapshots"]["final_each_model"].get(m)
                if p is None:
                    continue
                r=p[m.split("_")[0]]
                e=r["final_residual"]
                if e[0] is None:
                    continue
                J=np.asarray(r["jacobian_local"])
                data[i,m]["residual_square"].append(np.mean(np.asarray(e)**2))
                data[i,m]["information_trace"].append(np.sum(J*J))
                data[i,m]["abs_dz"].append(abs(p["signed_height_difference"]))
            rows+=1
    summary={}
    for m in METHODS:
        rows_m=[]
        for x in loo["C"]:
            if x["method"]!=m or "delta_rmse_m" not in x["result"]:
                continue
            j=x["edge_id"]
            w=data[j,m]
            if not w["residual_square"]:
                continue
            rows_m.append(dict(edge=j,delta_rmse_m=x["result"]["delta_rmse_m"],
                full_graph_residual_rms=float(np.sqrt(np.mean(w["residual_square"]))),
                full_graph_information_trace=float(sum(w["information_trace"])),
                avg_abs_height_difference_m=float(np.mean(w["abs_dz"]))))
        def corr(field,signed):
            a=np.array([x[field] for x in rows_m])
            b=np.array([x["delta_rmse_m"] for x in rows_m])
            if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:
                return None
            return float(spearmanr(a,b if signed else abs(b)).statistic)
        columns=("full_graph_residual_rms","full_graph_information_trace","avg_abs_height_difference_m")
        summary[m]=dict(n=len(rows_m),stats={k:dict(signed=corr(k,True),magnitude=corr(k,False)) for k in columns},
              largest_residual=max(rows_m,key=lambda z:z["full_graph_residual_rms"]) if rows_m else None,
              most_helpful_batch=max(rows_m,key=lambda z:z["delta_rmse_m"]) if rows_m else None,
              most_harmful_batch=min(rows_m,key=lambda z:z["delta_rmse_m"]) if rows_m else None)
    return dict(schema="ocean_range_v5_factor_descriptive",scene=loo["scene"],pairs_read=rows,
                models=summary,raw_detail_sha256=sha(detail),loo_sha256=sha(loo_path),code_sha256=sha(Path(__file__)))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    for name in ("detail","loo","output"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():
        p.error("Choose new output path")
    write_json(a.output,run(a.detail,a.loo))
