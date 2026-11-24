"""Independent consistency audit of a linearized projected raw-range block."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from research.sonar_range_factor_v2.models import Track, joint_residual_jacobian


def consistency(seed=761123,n=12000):
    t=Track([14.,5.],[[1.,0.],[2.,.6]],[0.,4.],[17.,18.],np.eye(2),(-50.,0.))
    _,J=joint_residual_jacobian(t,np.array([[0.,0.,0.],[4.,-3.,.1]]),-12.)
    A=J[:,3:5]
    b=J[:,-1]
    w=np.array([b[1],-b[0]])/np.linalg.norm(b)
    h=w@A
    P=np.array([[1.2,.3],[.3,1.7]])
    rng=np.random.default_rng(seed)
    x=rng.multivariate_normal([0.,0.],P,n)
    perturb_z=rng.normal(0.,.5,n)
    records={}
    for rate in (-.72,.72):
        std=np.array([.55,.65])
        S=np.outer(std,std)*np.array([[1.,rate],[rate,1.]])
        raw=x@A.T+perturb_z[:,None]*b+rng.multivariate_normal(np.zeros(2),S,n)
        y=raw@w
        choices={}
        for name,cov in [("correct",S),("diagonal",np.diag(S.diagonal()))]:
            kept_variance=float(w@cov@w)
            S_y=float(h@P@h+kept_variance)
            K=P@h/S_y
            posterior=P-np.outer(K,h@P)
            state=x-y[:,None]*K
            nis=y*y/S_y
            nees=np.einsum("ni,ij,nj->n",state,np.linalg.inv(posterior),state)
            choices[name]=dict(NIS_mean=float(nis.mean()),NEES_mean=float(nees.mean()),
                               NIS_95=float(np.mean(nis<=chi2.ppf(.95,1))),
                               NEES_95=float(np.mean(nees<=chi2.ppf(.95,2))),
                               contrast_noise_variance=kept_variance)
        records[str(rate)]=choices
    return dict(schema="ocean_m1_linear_consistency_v3",n=n,seed=seed,
                NIS_dof=1,NEES_dof=2,results=records,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(consistency(),f,indent=2,allow_nan=False)
        f.write("\n")


if __name__=="__main__":
    main()
