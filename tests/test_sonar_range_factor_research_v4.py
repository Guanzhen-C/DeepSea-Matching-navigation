from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from research.sonar_range_factor_v2.models import Track, _profiled, geometry, physical_feasibility
from research.sonar_range_factor_v3.certified import FrozenHeight
from research.sonar_range_factor_v3.experiment import Case,observations
from research.sonar_range_factor_v3.likelihood import matched_huber,profile_pair,transform
from research.sonar_range_factor_v4.adaptive import Policy,decide,physical_tube
from research.sonar_range_factor_v4.batch import Batch
from research.sonar_range_factor_v4.graph import make_batch,make_hybrid
from research.sonar_range_factor_v4.online_probe import Epoch,Ledger,fixed_lag_probe
from standalone.SideScanMappedRangeChange import MappedRangeChange


def fixture(dz=15.):
    xy=np.array([30.,46.])
    p=np.array([[0.,0.,.12],[3.,2.,.4]])
    z=np.array([0.,dz])
    u=np.zeros((2,2))
    r=np.sqrt(np.sum((xy-p[:,:2])**2,axis=1)+(z+16.)**2)
    t=Track(xy,u,z,r,[[.25,.08],[.08,.35]],(-150.,0.))
    return t,p


def frozen_old(t,n):
    return MappedRangeChange(np.tile(t.xy,(n,1)),np.tile(t.lever[0],(n,1)),np.tile(t.lever[1],(n,1)),
            np.full(n,t.sensor_z[0]),np.full(n,t.sensor_z[1]),np.full(n,t.ranges[0]),
            np.full(n,t.ranges[1]),np.ones(n))


@pytest.mark.parametrize("dz",[0.,.00001,4.,16.])
def test_equivalence_profile_hessian_and_cached_jacobian(dz):
    s=observations(Case(dz=dz,views=2),4281,1.)
    t,p=s.tracks,s.initial_poses
    f=matched_huber(2)
    b=Batch(t,f)
    for pos in (p,p+[[0.,.7,.002],[2.,0.,.015]]):
        result=b.evaluate(pos)
        assert b.evaluate(pos) is result and not result.jacobian.flags.writeable
        for i,track in enumerate(t):
            m,e,J,H,zp=_profiled(track,geometry(track,pos))
            r,j=transform(e,J[:,:-1]+np.outer(J[:,-1],zp),f)
            assert result.status[i].startswith("ok")
            assert result.height[i]==pytest.approx(m.height,abs=1e-7)
            assert result.cost[i]==pytest.approx(m.cost,abs=1e-8)
            assert np.allclose(result.residual[i],r,atol=1e-8)
            assert np.allclose(result.jacobian[i],j,atol=1e-8)
            assert np.allclose(result.hessian[i],H[:-1,:-1]+np.outer(H[:-1,-1],zp),atol=1e-7)
    assert b.cache_hits==2


def test_tie_nadir_and_budget_not_silently_accepted():
    p=np.array([[0.,0.,0.],[1.,0.,0.]])
    t=Track([1.,0.],np.zeros((2,2)),[0.,2.],[2.8125,1.8125],np.eye(2),(-4.,0.))
    bad=Track([1.,0.],np.zeros((2,2)),[2.,0.],[3.,1.],np.eye(2),(-4.,0.))
    a=Batch([t,bad]).evaluate(p)
    assert list(a.status)==["ties_or_unresolved","nadir"]
    cert=FrozenHeight(t,p).minimum()
    assert cert["status"]=="gap_certified" and float(cert["gap"])<1e-9
    t,p=fixture()
    b=Batch([t])
    assert b.evaluate(p,max_boxes=0,allow_reference=False).status[0]=="uncertain_or_budget"
    assert b.evaluate(p,max_boxes=0).status[0]=="ok_reference"
    out=b.evaluate(p)
    assert out.status[0]=="ok"
    # Changing the warm seed cannot change the chosen minimum.
    b.warm[:]=-149.
    assert np.allclose(b.evaluate(p,use_cache=False).height,out.height,atol=1e-8)


def test_physical_tube_and_causal_latched_modes():
    t,p=fixture()
    bf=Batch([t,t,t,t])
    eps=np.zeros((4,2))+1e-9
    assert physical_tube(bf,p,eps).all() and physical_feasibility(t,p).feasible
    old=frozen_old(t,4)
    pol=Policy(2.,3.,3.)
    confidence=np.array([.95,.6,.2,np.nan])
    g=decide(bf,p,old,confidence,pol,arrival_s=40,node_times_s=[10,40])
    assert list(g.mode)==["M1","M0","reject","M1"]
    assert list(g.information)==[1.,.5,0.,.5]
    with pytest.raises(ValueError,match="Future"):
        decide(bf,p,old,confidence,pol,arrival_s=39,node_times_s=[10,40])
    wrong=np.array([[200.,0.,.12],[-100.,2.,.4]])
    a=decide(bf,wrong,old,confidence,pol,arrival_s=40,node_times_s=[10,40])
    assert not a.feasible.any() and not (a.mode=="M1").any()
    t,p=fixture(dz=0.)
    a=decide(Batch([t]),p,frozen_old(t,1),[.99],pol,arrival_s=40,node_times_s=[10,40])
    assert a.reason[0]=="m0_narrow_deadband"


def test_batched_gtsam_preserves_full_objective_local_linearization():
    g=pytest.importorskip("gtsam")
    t,p=fixture()
    tracks=[t,t]
    keys=[g.symbol("x",i) for i in range(2)]
    loss=matched_huber(2)
    f=make_batch(tracks,keys,loss)
    x=g.Values()
    pp=[g.Pose2(*q) for q in p]
    for k,q in zip(keys,pp):
        x.insert(k,q)
    assert f.error(x)==pytest.approx(sum(np.linalg.norm(profile_pair(t,p,loss)[0])**2/2 for t in tracks))
    J=[]
    for j in range(6):
        d=np.eye(6)[j]*1e-6
        def values(delta):
            q=g.Values()
            for i,k in enumerate(keys):
                q.insert(k,pp[i].retract(delta[i*3:i*3+3]))
            return q
        J.append((f.unwhitenedError(values(d))-f.unwhitenedError(values(-d)))/2e-6)
    l=f.linearize(x)
    assert np.allclose(np.array(l.getA(),copy=True),np.column_stack(J),atol=1e-7)
    old=frozen_old(t,2)
    policy=Policy(2.,3.,3.)
    gate=decide(Batch(tracks),p,old,[.95,.2],policy,arrival_s=40,node_times_s=[10,40])
    hybrid=make_hybrid(tracks,old,keys,loss,gate,policy)
    rebuilt=make_hybrid(tracks,old,keys,loss,gate,policy,rebuild_as_m0=True)
    assert len(hybrid.unwhitenedError(x))==2 and len(rebuilt.unwhitenedError(x))==1


def test_isam2_insert_remove_relinearize_and_transaction_recovery():
    g=pytest.importorskip("gtsam")
    t,p=fixture()
    keys=[g.symbol("x",i) for i in range(2)]
    noise=g.noiseModel.Isotropic.Sigma(3,.3)
    l=Ledger()
    _,r=l.advance(Epoch(tuple((f"prior{i}",lambda i=i:g.PriorFactorPose2(
        keys[i],g.Pose2(*p[i]),noise)) for i in range(2)),
        tuple((k,tuple(q)) for k,q in zip(keys,p))))
    x,r=l.advance(Epoch((("range",lambda:make_batch([t],keys,matched_huber(2))),)))
    assert "range" in l.ids and r.getVariablesRelinearized()>=0
    def fail(_f,_values,_J):
        raise ValueError("test chart switch")
    bad=lambda:g.CustomFactor(g.noiseModel.Unit.Create(1),keys,fail)
    with pytest.raises((ValueError,RuntimeError),match="chart switch"):
        l.advance(Epoch((("bad",bad),),remove=("range",)))
    assert "range" in l.ids and "bad" not in l.ids and len(l.events)==2
    assert np.allclose(l.solver.calculateEstimate().atPose2(keys[1]).translation(),
                       x.atPose2(keys[1]).translation())
    y,_=l.advance(Epoch((),remove=("range",)))
    assert "range" not in l.ids
    l.restore()
    assert np.allclose(l.solver.calculateEstimate().atPose2(keys[1]).translation(),
                       y.atPose2(keys[1]).translation())


def test_fixed_lag_key_outside_window_not_inserted():
    pytest.importorskip("gtsam")
    x=fixed_lag_probe()
    assert x["expired_first"] and x["latest_present"] and not x["expired_edge_allowed"]
    assert x["active_counts"]==[1,2,3,3]


def test_cpp_eigen_convex_kernel(tmp_path):
    cxx=shutil.which("c++")
    pkg=shutil.which("pkg-config")
    if not cxx or not pkg or subprocess.run([pkg,"--exists","eigen3"]).returncode:
        pytest.skip("Eigen/C++ optional toolchain")
    flags=subprocess.check_output([pkg,"--cflags","eigen3"],text=True).split()
    path=Path(__file__).resolve().parents[1]/"research/sonar_range_factor_v4/eigen_probe.cpp"
    executable=tmp_path/"kernel_probe"
    subprocess.run([cxx,"-std=c++17","-O3","-fno-fast-math",*flags,str(path),"-o",str(executable)],check=True)
    z,stat,it,bad,invalid=subprocess.check_output([str(executable)],text=True).split()
    assert float(z)==pytest.approx(-11.,abs=1e-10) and 0<int(it)<20
    assert [int(x) for x in (stat,bad,invalid)]==[0,2,1]
