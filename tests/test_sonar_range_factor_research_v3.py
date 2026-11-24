from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import chi2

from research.sonar_range_factor_v2.models import Track, profile_minimum
from research.sonar_range_factor_v3.adapter import make_m0, make_m1, solve_with_replay
from research.sonar_range_factor_v3.certified import Bounds, I, FrozenHeight
from research.sonar_range_factor_v3.experiment import Case, fit, observations, scale_calibration
from research.sonar_range_factor_v3.likelihood import (
    RadialLoss, matched_huber, profile_pair, propagate, transform,
)


def data(tie=False):
    if tie:
        t=Track([1.,0.],np.zeros((2,2)),[0.,2.],[2.8125,1.8125],np.eye(2),(-4.,0.))
        return t,np.array([[0.,0.,0.],[1.,0.,0.]])
    t=Track([15.,3.],[[.5,1.],[2.,.1]],[0.,4.],[17.,21.],[[.5,.2],[.2,.8]],(-50.,0.))
    return t,np.array([[0.,1.,.14],[1.,-2.,-.35]])


def derivative(fun,x,eps=2e-6):
    x=np.asarray(x,float)
    return np.stack([(fun((x.ravel()+d).reshape(x.shape))-fun((x.ravel()-d).reshape(x.shape)))/(2*eps)
                       for d in np.eye(x.size)*eps],axis=-1)


def test_decimal_uses_own_context_and_encloses_sqrt_and_sign_intervals():
    b=Bounds(38)
    two=Decimal(2)
    with localcontext() as context:
        context.prec=6
        out=b.sqrt(I(two,two))
        s=b.sq(I(Decimal("-1.12345678912345"),Decimal("-1.12345678891234")))
    with localcontext() as context:
        context.prec=80
        assert out.lo<two.sqrt()<out.hi
        assert s.lo<=Decimal("-1.12345678891234")**2
        assert s.hi>=Decimal("-1.12345678912345")**2
        assert out.hi-out.lo<Decimal("1e-35")


def test_tie_strict_objective_and_all_stationary_coverage():
    t,p=data(tie=True)
    h=FrozenHeight(t,p)
    result=h.minimum()
    assert result["status"]=="gap_certified"
    assert Decimal(result["lower"])<=Decimal("1.66015625")<=Decimal(result["upper"])
    assert Decimal(result["gap"])<=Decimal("1e-9")
    assert profile_minimum(t,p).ties
    roots=h.stationary()
    # Below-branch derivative expressed in h=-Z.
    other=brentq(lambda x: 2*x+3/16-(45/16)*x/np.sqrt(1+x*x),.01,.5)
    for z in [-.75,-other]:
        assert any(Decimal(box["lo"])<=Decimal.from_float(z)<=Decimal(box["hi"])
                   for box in roots["boxes"])
    assert any(x["kind"]=="unique_maximum" for x in roots["boxes"])
    assert roots["status"]=="covered"
    assess=h.minimum(max_boxes=0,timeout_s=None)
    assert assess["status"]=="budget"
    assert Decimal(assess["lower"])<=Decimal("1.66015625")<=Decimal(assess["upper"])
    assert h.minimum(timeout_s=0)["status"]=="timeout"
    assert h.stationary(max_boxes=0)["boxes"][0]["kind"]=="unresolved_budget"


def test_correlated_certificate_agrees_with_float_profile_within_tolerance():
    t,p=data()
    h=FrozenHeight(t,p)
    bounds=h.minimum(gap="1e-10")
    v=profile_minimum(t,p).cost
    assert bounds["status"]=="gap_certified"
    assert float(bounds["lower"])-1e-12<=v<=float(bounds["upper"])+1e-12
    assert float(bounds["gap"])<=1e-10


def test_nadir_cost_supported_but_nonsmooth_search_flagged():
    t=Track([0.,0.],np.zeros((2,2)),[0.,2.],[1.,3.],np.eye(2),(-4.,0.))
    p=np.zeros((2,3))
    h=FrozenHeight(t,p)
    r=h.minimum()
    assert r["status"]=="gap_certified" and float(r["upper"])<1e-9
    assert any(x["kind"]=="possible_nonsmooth" for x in h.stationary()["boxes"])
    with pytest.raises(ValueError,match="nadir"):
        profile_pair(t,p,RadialLoss())


@pytest.mark.parametrize("loss",[RadialLoss(),matched_huber(2),RadialLoss("cauchy_penalty",3),
                                  RadialLoss("student",4),RadialLoss("mixture")])
def test_likelihood_transform_preserves_cost_gradient_and_zero(loss):
    for e in (np.zeros(2),np.array([1.2,.3]),np.array([-5.,8.])):
        v,A=transform(e,np.eye(2),loss)
        t=e@e
        phi,slope=loss.value_slope(t,2)
        assert v@v/2==pytest.approx(phi,abs=1e-12)
        assert np.allclose(A,derivative(lambda e:transform(e,np.eye(2),loss)[0],e),atol=1e-6)
        assert np.allclose(A.T@v,2*slope*e,atol=1e-10)


def test_block_invariance_and_huber_nominal_mass():
    l=matched_huber(2)
    assert chi2.cdf(l.scale**2,2)==pytest.approx(.9973002039367398)
    t,p=data()
    a,_,gauss=profile_pair(t,p,RadialLoss())
    b,G,robust=profile_pair(t,p,l)
    assert robust==gauss
    C=np.array([[0.,-1.],[1.,0.]])
    r=np.array([8.,4.])
    e,J=transform(r,np.eye(2),l)
    f,K=transform(C@r,C,l)
    assert np.allclose(f,C@e) and np.allclose(K,C@J)
    assert b@b<=a@a
    assert np.allclose(G,derivative(lambda q:profile_pair(t,q,l)[0],p),atol=1e-6)


def test_covariance_includes_shared_errors_and_rejects_non_psd():
    t,p=data()
    R=np.diag([.3,.5])
    B=np.zeros((10,10))
    B[:2,:2]=np.eye(2)*.4
    C=propagate(t,p,-12.,R,B)
    assert C[0,1]!=0 and np.linalg.eigvalsh(C).min()>0
    B[3,3]=-1
    with pytest.raises(ValueError,match="PSD"):
        propagate(t,p,-12.,R,B)


def test_pairing_inputs_are_nested_and_holdout_is_independent():
    s=scale_calibration()
    assert not set(s["seeds"])&set(range(53000,53024))
    a=observations(Case(views=2),53000,s["scale"])
    b=observations(Case(views=6),53000,s["scale"])
    assert np.array_equal(a.initial_poses,b.initial_poses[:2])
    for x,y in zip(a.tracks,b.tracks):
        assert np.array_equal(x.xy,y.xy) and np.array_equal(x.ranges,y.ranges[:2])
        assert np.array_equal(x.lever,y.lever[:2])
    for method in ("M0","M0_plain","M1_Gaussian","M1_Robust","M2"):
        assert fit(a,method)["reported_error_m"]>=0


def test_failed_numerical_fit_is_preserved(monkeypatch):
    from research.sonar_range_factor_v3 import experiment
    s=observations(Case(),53000,1.)
    def fail(*a,**kw):
        raise ValueError("test inactive set")
    monkeypatch.setattr(experiment,"profile_pair",fail)
    r=fit(s,"M1_Robust")
    assert not r["ok"] and "inactive set" in r["reason"]
    assert r["reported_error_m"]==np.linalg.norm(s.true_shift)


@pytest.mark.parametrize("incremental",[False,True])
@pytest.mark.parametrize("failure",[False,True])
def test_gtsam_succeeds_or_explicitly_replays_frozen_m0(incremental,failure):
    gtsam=pytest.importorskip("gtsam")
    from standalone.SideScanMappedRangeChange import MappedRangeChange
    t,p=data(tie=failure)
    keys=[gtsam.symbol("x",i) for i in range(2)]

    def builder(mode):
        g=gtsam.NonlinearFactorGraph()
        v=gtsam.Values()
        for i in range(2):
            q=gtsam.Pose2(*p[i])
            g.add(gtsam.PriorFactorPose2(keys[i],q,gtsam.noiseModel.Isotropic.Sigma(3,.3)))
            v.insert(keys[i],q)
            if incremental and i==0:
                yield g,v
                g,v=gtsam.NonlinearFactorGraph(),gtsam.Values()
        if mode=="M1":
            g.add(make_m1(t,*keys,matched_huber(2)))
        else:
            d=MappedRangeChange(t.xy[None],t.lever[:1],t.lever[1:],t.sensor_z[:1],
                                t.sensor_z[1:],t.ranges[:1],t.ranges[1:],np.ones(1))
            g.add(make_m0(d,*keys))
        yield g,v

    r=solve_with_replay(builder,incremental=incremental)
    assert r.selected==("M0" if failure else "M1")
    assert (r.primary_failure is not None)==failure
    assert all(np.isfinite(r.values.atPose2(k).x()) for k in keys)


def test_adapter_has_correct_local_pose2_jacobian_with_full_covariance():
    gtsam=pytest.importorskip("gtsam")
    t,p=data()
    keys=[gtsam.symbol("x",i) for i in range(2)]
    original=[gtsam.Pose2(*x) for x in p]
    graph=make_m1(t,*keys,RadialLoss("student",4))
    def v(delta):
        values=gtsam.Values()
        for i,(k,x) in enumerate(zip(keys,original)):
            values.insert(k,x.retract(delta[i*3:i*3+3]))
        return values
    z=np.zeros(6)
    lin=graph.linearize(v(z))
    assert np.allclose(np.array(lin.getA(),copy=True),
                       derivative(lambda delta:graph.unwhitenedError(v(delta)),z),atol=2e-6)


def test_statistical_linear_projection_has_the_expected_dof_and_coverage():
    from research.sonar_range_factor_v3.diagnostics import consistency
    x=consistency(n=6000)
    for key in ("-0.72","0.72"):
        r=x["results"][key]["correct"]
        assert abs(r["NIS_95"]-.95)<.015 and abs(r["NEES_95"]-.95)<.015
    assert x["results"]["-0.72"]["diagonal"]["NIS_mean"]>1.05
    assert x["results"]["0.72"]["diagonal"]["NIS_mean"]<.95


def test_replay_reuses_locked_nodes_background_and_exact_truth_time(tmp_path):
    import inspect
    import json
    from pathlib import Path
    pytest.importorskip("gtsam")
    core=pytest.importorskip("oceansim.slam.pose_graph")
    from research.sonar_range_factor_v3.replay import replay
    t,p=data()
    stem=tmp_path/"factor.json"
    states=tmp_path/"states.npz"
    strong=np.eye(3).tolist()
    note=dict(landmark_xy=t.xy[None].tolist(),origin_a=t.lever[:1].tolist(),
              origin_b=t.lever[1:].tolist(),sensor_z_a=t.sensor_z[:1].tolist(),
              sensor_z_b=t.sensor_z[1:].tolist(),range_a=t.ranges[:1].tolist(),
              range_b=t.ranges[1:].tolist(),standard_deviation_m=[1.])
    stem.write_text(json.dumps(dict(schema="oceansim.planar_pose_graph_factor_stream.v1",
        pose_initializations=p.tolist(),pose_timestamps_s=[0.,1.],objective_configuration={},
        gauge_pose=p[0].tolist(),edges=[dict(source=0,target=1,kind="mapped_range_change")],
        priors=[dict(target=1,measurement=p[1].tolist(),information=strong)],
        mapped_range_changes=[dict(edge_index=0,source=0,target=1,observations=note)])))
    np.savez(states,timestamps=[0.,1.],positions=p[:,None,:])
    frozen=Path(inspect.getfile(core)).resolve().parent
    result=replay(stem,states,frozen)
    assert result["counts"]["nodes"]==2 and result["range_pairs"]==1
    assert result["batches"]["M0"]["graph_factors"]==result["batches"]["M1_Robust"]["graph_factors"]
    assert all(r["status"]=="returned_estimate" for r in result["batches"].values())
    np.savez(states,timestamps=[0.,2.],positions=p[:,None,:])
    with pytest.raises(ValueError,match="exactly"):
        replay(stem,states,frozen)
