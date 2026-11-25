from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.sonar_range_factor_v2.models import Track
from research.sonar_range_factor_v4.adaptive import Policy,decide
from research.sonar_range_factor_v4.batch import Batch
from research.sonar_range_factor_v5.ablations import interaction,summary_single
from research.sonar_range_factor_v5.calibration import (
    map_semivariance,paired_covariance,residuals,scale_and_clock,
)
from research.sonar_range_factor_v5.diagnostics import decompose,prefix_predictions,select_masks,snapshots,tube_prediction
from research.sonar_range_factor_v5.gate_study import early_gate,independent_holdout
from research.sonar_range_factor_v5.geometry_study import Design,build,spectrum
from research.sonar_range_factor_v5.replay import Study,make_tracks,score
from standalone.SideScanMappedRangeChange import MappedRangeChange


def pair(dz):
    p=np.array([[0.,0.,0.],[0.,10.,0.]])
    L=np.array([[0.,45.],[0.,52.],[0.,60.],[0.,75.]])
    rng=np.sqrt((L[:,1,None]-p[None,:,1])**2+(16+np.array([0.,dz]))**2)
    u=np.zeros((len(L),2))
    old=MappedRangeChange(L,u,u,np.zeros(len(L)),np.full(len(L),dz),rng[:,0],rng[:,1],np.ones(len(L)))
    return p,old


@pytest.mark.parametrize("dz",[0.,1e-6,14.])
def test_local_information_and_collinear_yaw_geometry(dz):
    p,old=pair(dz)
    b=Batch(make_tracks(old,.25))
    row,g=snapshots(SimpleNamespace(old={0:old},batch={0:b}),0,p)
    assert np.allclose(row["M0"]["information"],row["M0"]["jacobian_local"].transpose(0,2,1)@
                       row["M0"]["jacobian_local"])
    assert np.allclose(row["M1"]["information"],row["M1"]["jacobian_local"].transpose(0,2,1)@
                       row["M1"]["jacobian_local"])
    for m in ("M0","M1_Robust"):
        v=g[m]["conditional_target_xy"]
        assert v["rank"]<=1 and v["condition"] is None
        if v["rank"]:
            assert abs(v["right_strongest"][1])==pytest.approx(1.)
    if dz:
        assert g["M0"]["conditional_target_xy"]["rank"]==0
        assert g["M1_Robust"]["conditional_target_xy"]["rank"]==1


def test_same_pair_raw_covariance_assumption_keeps_difference_variance():
    _,a=pair(4.)
    for rho in (-.5,0,.25,.75):
        t=make_tracks(a,rho)[0]
        g=2*t.ranges*np.array([1.,-1.])/t.ranges.sum()
        assert g@t.covariance@g==pytest.approx(1.)
        assert t.covariance[0,1]/np.sqrt(np.prod(t.covariance.diagonal()))==pytest.approx(rho)


def test_truth_free_masks_and_interaction_not_summed_as_effects():
    s=SimpleNamespace(ids=[0,1],times=np.array([0.,16.]),arrival={0:3.,1:15.})
    def r(a,f):
        return dict(observability_causal={m:dict(conditional_target_xy=dict(singular=a)) for m in ["M0","M1_Robust"]},
                    causal_tube_fraction=f)
    rec={0:r(np.array([1.,0.]),1.),1:r(np.array([1.,.6]),1.)}
    rel={0:dict(identical_loop_current_window=True,identical_loop_endpoints=False),
         1:dict(identical_loop_current_window=False,identical_loop_endpoints=False)}
    masks=select_masks(s,rec,rel)
    assert masks["E_overlap_risk"]==masks["F_weak_geometry"]==masks["G_only_high_quality"]==[0]
    assert masks["D_q1"]==[0] and masks["D_q4"]==[1]
    assert interaction(5.,[1.,2.])==2.
    assert interaction(5.,[1.,None]) is None
    assert interaction(None,[1.,2.]) is None


def test_c_single_sign_and_failed_scoring():
    t=np.zeros((2,2))
    ref=np.array([[1.,0.,0.],[1.,0.,0.]])
    good=score(dict(pose=np.zeros((2,3)),status="returned"),t,ref,[0,1])
    fail=score(dict(status="failed",error="unresolved"),t,ref,[0,1])
    assert good["delta_rmse_m"]==-1 and good["endpoint_changes_m"]==[1.,1.]
    assert "rmse_m" not in fail
    stats=summary_single([dict(edge_id=1,result=good),dict(edge_id=2,result=fail)])
    assert stats["failed"]==1 and stats["loss_when_retained_ge_1mm"]==1


@pytest.mark.parametrize("dz",[0.,14.])
def test_shortcircuit_same_action_and_prevents_future_keys(dz):
    p,old=pair(dz)
    b=Batch(make_tracks(old,.25))
    policy=Policy(2.,3.,4.2362)
    confidence=np.array([.2,.55,.95,np.nan])
    kwargs=dict(arrival_s=30.,node_times_s=[10.,25.])
    x,skip=early_gate(b,p,old,confidence,policy,**kwargs)
    y=decide(b,p,old,confidence,policy,**kwargs)
    assert all(np.array_equal(getattr(x,k),getattr(y,k)) for k in [
        "mode","reason","information","feasible","deadband_sigma"])
    assert skip==(dz==0)
    with pytest.raises(ValueError,match="Future"):
        early_gate(b,p,old,confidence,policy,arrival_s=15.,node_times_s=[10.,25.])


def test_covariance_shadow_and_holdout_tradeoff():
    p,old=pair(4.)
    b=Batch(make_tracks(old,.25))
    o=b.evaluate(p)
    s=tube_prediction(b,p,o,[None,np.eye(3)*4.])
    assert np.allclose(s[:,0],np.sqrt(np.array([t.covariance[0,0] for t in b.tracks])))
    assert np.all(s[:,1]>np.sqrt(np.array([t.covariance[1,1] for t in b.tracks])))
    result=independent_holdout(Policy(2.,3.,4.2362),12)
    for d in result["table"]:
        s=d["passes"]
        assert s["correct_predicted_marginal_pass"]>=s["correct_nominal_pass"]
        assert s["wrong_predicted_marginal_pass"]>=s["wrong_nominal_pass"]


def test_delay_scale_needs_range_span_and_holds_on_unseen_targets():
    r=np.arange(30.,150.,3.)
    v=scale_and_clock(r,r+.18+r*.0017)
    assert v["offset_m"]==pytest.approx(.18) and v["fractional_scale"]==pytest.approx(.0017)
    assert v["time_zero_s"]==pytest.approx(.36/1480.)
    assert np.linalg.norm(residuals([12.,166.],[12.+.18+12*.0017,166.+.18+166*.0017],v))<1e-12
    with pytest.raises(ValueError,match="Single-distance"):
        scale_and_clock(np.ones(8)*80,np.ones(8)*81)


def test_negative_correlation_and_map_semivariance():
    rng=np.random.default_rng(733)
    x=rng.multivariate_normal([0,0],[[1.,-.5],[-.5,1.]],4096)
    c=paired_covariance(*x.T,block_ids=np.arange(len(x))//256,resamples=100)
    assert -.56<c["rho"]<-.44 and len(c["rho_ci95_block"])==2
    g=map_semivariance([[0,0],[1,0],[4,0]],[[0,0],[1,0],[3,0]],[0,2,8])
    assert g[0]["pairs"]==1 and g[1]["pairs"]==2
    assert g[0]["semivariance_m2"]<g[1]["semivariance_m2"]


def test_nested_true_view_sources_and_logged_information():
    a=build(Design(views=2),81282)
    b=build(Design(views=6),81282)
    assert np.array_equal(a.initial_poses,b.initial_poses[:2])
    for x,y in zip(a.tracks,b.tracks):
        assert np.array_equal(x.ranges,y.ranges[:2])
    x,bad=spectrum(b)
    assert x["M1_Robust"]["rank"]==2 and x["M1_Robust"]["condition"]>1 and bad==0
    assert decompose(np.zeros((2,2)))["right_strongest"] is None


def test_real_archived_simulator_prefix_does_not_read_truth(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    core=root/"output/论文数据资料库_20261124/02_匹配导航算法/复现实验_冻结导航运行时/runtime/core"
    if not core.is_dir():
        pytest.skip("local immutable archived simulator dependencies are optional")
    pytest.importorskip("oceansim.slam.pose_graph")
    s=Study("qiandao2",core)
    def forbidden():
        raise AssertionError("A gate diagnosis is not permitted to read truth.")
    monkeypatch.setattr(s,"truth",forbidden)
    p=prefix_predictions(s)
    assert len(p)==8
    for i,q in p.items():
        assert q["last_committed_node_time_s"]<=s.arrival[i]
        assert q["n_previous_edges"]<=i
        assert all(x is None or np.isfinite(x).all() for x in q["covs"])
