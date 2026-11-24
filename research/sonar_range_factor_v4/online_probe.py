"""PC-only solver probes: iSAM2 transactions and fixed-lag key safety."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Epoch:
    # Each (logical ID, a builder producing a fresh nonlinear factor).
    factors: tuple
    poses: tuple=()
    remove: tuple=()


class Ledger:
    """In-memory reference recovery; an immutable event log is the checkpoint."""
    def __init__(self):
        self.events=[]
        self.restore()

    def restore(self):
        import gtsam
        cfg=gtsam.ISAM2Params()
        cfg.setRelinearizeThreshold(.01)
        cfg.relinearizeSkip=1
        self.solver=gtsam.ISAM2(cfg)
        self.ids={}
        for e in self.events:
            self._apply(e)

    def _apply(self,e):
        import gtsam
        ids={k:v for k,v in self.ids.items() if k not in e.remove}
        removed=[self.ids[q] for q in e.remove]
        if len(set(e.remove))!=len(e.remove) or len({x[0] for x in e.factors})!=len(e.factors) or (
                any(k in ids for k,_ in e.factors)):
            raise ValueError("Unknown or duplicated factor ledger operation")
        g,v=gtsam.NonlinearFactorGraph(),gtsam.Values()
        for name,f in e.factors:
            g.add(f())
        for key,p in e.poses:
            v.insert(key,gtsam.Pose2(*p))
        info=self.solver.update(g,v,removed)
        added=list(info.getNewFactorsIndices())
        if len(added)!=len(e.factors):
            raise RuntimeError("New factor index bookkeeping differs")
        self.ids=ids|dict(zip((name for name,_ in e.factors),added))
        result=self.solver.calculateEstimate()
        for key in result.keys():
            p=result.atPose2(key)
            if not np.isfinite([p.x(),p.y(),p.theta()]).all():
                raise ValueError("Nonfinite incremental estimate")
        return result,info

    def advance(self,e):
        try:
            r=self._apply(e)
        except (ValueError,RuntimeError,FloatingPointError,KeyError):
            self.restore()
            raise
        self.events.append(e)
        return r


def fixed_lag_probe():
    import gtsam
    api=gtsam
    if not hasattr(api,"IncrementalFixedLagSmoother"):
        import gtsam_unstable as api
    solver=api.IncrementalFixedLagSmoother(2.)
    noise=gtsam.noiseModel.Isotropic.Sigma(3,.5)
    keys=[gtsam.symbol("x",i) for i in range(4)]
    counts=[]
    for i,key in enumerate(keys):
        graph,v=gtsam.NonlinearFactorGraph(),gtsam.Values()
        v.insert(key,gtsam.Pose2(i,0,0))
        if i==0:
            graph.add(gtsam.PriorFactorPose2(key,gtsam.Pose2(),noise))
        else:
            graph.add(gtsam.BetweenFactorPose2(keys[i-1],key,gtsam.Pose2(1,0,0),noise))
        timestamps=api.FixedLagSmootherKeyTimestampMap()
        timestamps.insert((key,float(i)))
        solver.update(graph,v,timestamps)
        counts.append(solver.calculateEstimate().size())
    x=solver.calculateEstimate()
    # Explicit caller check: do not send a long-loop factor to marginalized endpoint.
    return dict(active_counts=counts,expired_first=not x.exists(keys[0]),
                latest_present=x.exists(keys[-1]),
                expired_edge_allowed=bool(x.exists(keys[0]) and x.exists(keys[-1])))
