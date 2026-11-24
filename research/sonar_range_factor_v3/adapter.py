"""Isolated GTSAM adapter and explicit whole-graph fallback policy."""
from dataclasses import dataclass
import math

import numpy as np

from standalone.SideScanMappedRangeChange import profiled_range_residual_jacobians
from .likelihood import profile_pair


def _two_pose_factor(size,key_a,key_b,measurement):
    import gtsam
    if key_a==key_b:
        raise ValueError("A pair needs two distinct Pose2 keys")

    def callback(_factor,values,jacobians):
        p=[values.atPose2(k) for k in (key_a,key_b)]
        world=np.array([[x.x(),x.y(),x.theta()] for x in p])
        e,J=measurement(world)
        if not np.isfinite(e).all() or not np.isfinite(J).all():
            raise ValueError("Invalid range factor output")
        if jacobians is not None:
            for i,pose in enumerate(p):
                c,s=math.cos(pose.theta()),math.sin(pose.theta())
                chart=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
                jacobians[i]=np.asfortranarray(J[:,i*3:i*3+3]@chart)
        return e

    return gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(size),[key_a,key_b],callback)


def make_m1(track,key_a,key_b,loss):
    if len(track.ranges)!=2:
        raise ValueError("M1 requires a paired track")

    def measurement(poses):
        r,J,_=profile_pair(track,poses,loss)
        return r,J

    return _two_pose_factor(2,key_a,key_b,measurement)


def make_m0(batch,key_a,key_b):
    def measurement(poses):
        r,A,B=profiled_range_residual_jacobians(*poses,batch)
        return r,np.column_stack((A,B))

    return _two_pose_factor(batch.features,key_a,key_b,measurement)


@dataclass
class GraphResult:
    values: object
    selected: str
    primary_failure: str|None


def solve_with_replay(builder, *, incremental=False):
    """Each retry starts with fresh factors, Values and optimizer.

    builder(mode) must deterministically replay the same observation epochs.
    Never switch one callback or mutate the objective mid-optimization.
    """
    import gtsam

    def solve(mode):
        epochs=list(builder(mode))
        if not epochs or (not incremental and len(epochs)!=1):
            raise ValueError("Expected replay epochs; batch expects one epoch")
        if not incremental:
            graph,initial=epochs[0]
            graph.error(initial)  # preflight the exact objective
            result=gtsam.LevenbergMarquardtOptimizer(graph,initial).optimize()
            if not math.isfinite(graph.error(result)):
                raise ValueError("Non-finite batch objective")
            return result
        solver=gtsam.ISAM2()
        combined=gtsam.NonlinearFactorGraph()
        for graph,initial in epochs:
            solver.update(graph,initial)
            for i in range(graph.size()):
                combined.add(graph.at(i))
        result=solver.calculateEstimate()
        if not math.isfinite(combined.error(result)):
            raise ValueError("Non-finite incremental objective")
        return result

    try:
        return GraphResult(solve("M1"),"M1",None)
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        # Discard any partially updated iSAM2. If M0 fails, do not mask it.
        return GraphResult(solve("M0"),"M0",f"{type(exc).__name__}: {exc}")
