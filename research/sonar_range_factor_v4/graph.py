"""Research-only batched two-Pose2 factor; fixed action after insertion."""
import numpy as np

from standalone.SideScanMappedRangeChange import profiled_range_residual_jacobians
from research.sonar_range_factor_v3.adapter import _two_pose_factor
from .batch import Batch


def make_batch(tracks,keys,loss):
    b=Batch(tracks,loss)

    def measurement(p):
        d=b.evaluate(p)
        ok=np.char.startswith(d.status,"ok")
        if not ok.all():
            raise ValueError("Pure M1 cannot silently drop a range row: "+",".join(sorted(set(d.status[~ok]))))
        return d.residual.ravel(),d.jacobian.reshape(-1,6)

    return _two_pose_factor(2*len(tracks),*keys,measurement)


def make_hybrid(tracks,old,keys,loss,gate,policy,*,rebuild_as_m0=False):
    mode=gate.mode
    valid=mode!="reject"
    m1=np.flatnonzero((mode=="M1")&~np.full(len(mode),rebuild_as_m0))
    m0=np.flatnonzero(valid&~np.isin(np.arange(len(mode)),m1))
    if rebuild_as_m0:
        # Replace previously accepted M1 rows with original M0 weights;
        # weak-confidence original M0 rows keep their recorded tempering.
        old_weight=np.where(mode[m0]=="M1",1.,gate.information[m0])
    else:
        old_weight=gate.information[m0]
    old_weight=np.sqrt(old_weight)
    m1_weight=np.sqrt(gate.information[m1]*min(old.features,old.independent_equivalent_features)/old.features)
    b=Batch([tracks[i] for i in m1],loss) if len(m1) else None
    n=len(m0)+2*len(m1)
    if not n:
        return None

    def measurement(p):
        e,j=[],[]
        if len(m0):
            E,A,D=profiled_range_residual_jacobians(*p,old)
            e.append(E[m0]*old_weight)
            j.append(np.column_stack((A[m0],D[m0]))*old_weight[:,None])
        if b is not None:
            d=b.evaluate(p,max_boxes=policy.max_boxes,allow_reference=False)
            if not (d.status=="ok").all():
                raise ValueError("Adaptive latched M1 left its valid chart: "+",".join(sorted(set(d.status))))
            e.append((d.residual*m1_weight[:,None]).ravel())
            j.append((d.jacobian*m1_weight[:,None,None]).reshape(-1,6))
        return np.concatenate(e),np.concatenate(j)

    return _two_pose_factor(n,*keys,measurement)
