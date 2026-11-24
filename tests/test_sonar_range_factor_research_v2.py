import numpy as np
import pytest
from scipy.optimize import least_squares, minimize_scalar

from research.sonar_range_factor_v2.models import (
    Track, joint_residual_jacobian, physical_feasibility, physical_interval,
    profile_minimum, profile_value_gradient_hessian, profiled_residual_jacobian,
    projected_linearization,
)


def fixture(*, noise=(0., 0.), covariance=None, bounds=(-70., 0.3), height=-12.):
    xy, u, z = np.array([20., 11.]), np.array([[1., -.4], [2., .5]]), np.array([.3, 3.])
    poses = np.array([[-3., 1., .13], [5., 3., -.7]])
    d = np.array([xy-p[:2]-[[np.cos(p[2]), -np.sin(p[2])],
                           [np.sin(p[2]), np.cos(p[2])]]@l for p, l in zip(poses, u)])
    ranges = np.sqrt(np.sum(d*d, axis=1)+(height-z)**2)+noise
    return Track(xy, u, z, ranges, np.diag([.36, .64]) if covariance is None else covariance,
                 bounds), poses


def diff(fun, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    return np.stack([(fun((x.ravel()+dx).reshape(x.shape))-
                      fun((x.ravel()-dx).reshape(x.shape)))/(2*h)
                     for dx in np.eye(x.size)*h], axis=-1)


def test_original_m0_counterexample_and_exact_gate():
    from standalone.SideScanMappedRangeChange import (
        MappedRangeChange, profiled_range_residual_jacobians,
    )
    xy, poses = np.array([0., 20.]), np.zeros((2, 3))
    t = Track(xy, np.zeros((2, 2)), np.zeros(2), np.full(2, 5.), np.eye(2), (-250., 0.))
    m0 = MappedRangeChange(xy[None], np.zeros((1, 2)), np.zeros((1, 2)),
                           *[np.zeros(1)]*2, *[np.full(1, 5.)]*2, np.ones(1))
    assert profiled_range_residual_jacobians(*poses, m0)[0][0] == 0
    check = physical_feasibility(t, poses)
    assert not check.feasible and np.array_equal(check.eta, [-375., -375.])
    assert profile_minimum(t, poses).cost == pytest.approx(225.)


@pytest.mark.parametrize("dz,angle,baseline,err,height,expected", [
    (0., 0., 2., 0., -10., True),
    (4., 0., 2., 0., -10., True),
    (4., 1.5, 2., 0., -10., True),
    (-4., -2.5, 20., 0., -10., True),
    (1e-7, .4, 4., 0., -10., True),
    (5., .4, 4., 0., -40., True),
    (5., .4, 4., 0., 0., True),
    (5., .4, 4., .12, -10., False),
    (5., .4, 4., 8., -10., False),
])
def test_physical_cases(dz, angle, baseline, err, height, expected):
    from research.sonar_range_factor_v2.models import geometry
    z = np.array([0., dz])
    poses = np.array([[0., 0., 0.], [baseline, 3., angle]])
    lever = np.array([[1., 0.], [2., 1.]])
    bounds = (-40., min(z))
    init = Track([14., 18.], lever, z, [1., 1.], np.eye(2), bounds)
    _, _, d2 = geometry(init, poses)
    rs = np.sqrt(d2+(height-z)**2)+[0, err]
    t = Track(init.xy, lever, z, rs, np.eye(2), bounds)
    assert physical_feasibility(t, poses).feasible == expected
    assert physical_feasibility(t, poses, range_tolerance=abs(err)+1e-8).feasible
    p = profile_minimum(t, poses)
    if expected:
        assert p.height == pytest.approx(height, abs=1e-6)
        assert p.cost < 1e-20
    else:
        assert p.cost > 1e-6


@pytest.mark.parametrize("noise,bounds,branch", [
    ((.1, -.2), (-70., .3), "interior"),
    ((.1, -.2), (-7., .3), "lower"),
    ((.1, -.2), (-70., -16.), "upper"),
    ((0., 0.), (-12., -12.), "singleton"),
])
def test_derivatives_interior_and_strict_boundaries(noise, bounds, branch):
    track, poses = fixture(noise=noise, bounds=bounds)
    opt, g, H = profile_value_gradient_hessian(track, poses)
    r, J, _ = profiled_residual_jacobian(track, poses)
    assert opt.branch == branch
    assert np.allclose(g, diff(lambda p: profile_minimum(track, p).cost, poses), atol=1e-6)
    assert np.allclose(H, diff(lambda p: profile_value_gradient_hessian(track, p)[1],
                               poses), atol=1e-6)
    assert np.allclose(J, diff(lambda p: profiled_residual_jacobian(track, p)[0],
                               poses), atol=1e-6)
    assert np.allclose(H, H.T, atol=1e-10)
    assert opt.cost == pytest.approx(r@r/2, abs=1e-13)


def test_zero_is_regular_but_active_set_switch_is_not_treated_as_smooth():
    track, poses = fixture()
    e, j, opt = profiled_residual_jacobian(track, poses)
    assert np.linalg.norm(e) < 1e-12 and np.isfinite(j).all() and np.linalg.matrix_rank(j, 1e-8) == 1
    lr = Track(track.xy, track.lever, track.sensor_z, track.ranges, track.covariance, (-12., .3))
    assert profile_minimum(lr, poses).cost < 1e-20
    with pytest.raises(ValueError, match="Active-set"):
        profiled_residual_jacobian(lr, poses)


def test_nonunique_global_minimum_even_with_independent_ranges():
    t = Track([1.,0.], np.zeros((2,2)), [0.,2.], [2.8125,1.8125],
              np.eye(2), (-4.,0.))
    p = np.array([[0.,0.,0.], [1.,0.,0.]])
    costs = [np.linalg.norm(joint_residual_jacobian(t,p,z)[0])**2/2 for z in (0.,-.75)]
    assert costs[0] == pytest.approx(costs[1], abs=1e-13)
    opt = profile_minimum(t,p)
    assert opt.ties and opt.cost == pytest.approx(costs[0], abs=1e-13)
    with pytest.raises(ValueError, match="Competing minima"):
        profiled_residual_jacobian(t,p)


def test_equal_depth_ceiling_with_structurally_zero_slope():
    t = Track([12.,0.], np.zeros((2,2)), [0.,0.], [10.,8.],
              np.eye(2), (-25.,0.))
    p = np.array([[0.,0.,0.], [3.,0.,0.]])
    r, j, opt = profiled_residual_jacobian(t,p)
    assert opt.branch == "upper" and np.array_equal(r, [2.,1.])
    assert np.allclose(j, diff(lambda poses: profiled_residual_jacobian(t,poses)[0],p))


def test_correlated_profile_and_projected_linear_problem():
    track, poses = fixture(noise=(.1, -.2), covariance=[[.6, .23], [.23, .9]])
    opt, g, H = profile_value_gradient_hessian(track, poses)
    J = lambda h: np.linalg.norm(joint_residual_jacobian(track, poses, h)[0])**2/2
    grid = np.linspace(*track.bounds, 200)
    assert opt.cost <= min(J(h) for h in grid)
    assert opt.cost == pytest.approx(minimize_scalar(J, bounds=track.bounds,
                                                     method="bounded").fun, abs=1e-10)
    assert np.allclose(g, diff(lambda p: profile_minimum(track, p).cost, poses), atol=1e-6)
    assert np.allclose(H, diff(lambda p: profile_value_gradient_hessian(track, p)[1],
                               poses), atol=1e-6)
    r, j = joint_residual_jacobian(track, poses, opt.height)
    pr, pa = projected_linearization(r, j)
    dx = np.linspace(-.01, .01, 6)
    b = j[:, -1]
    dz = -b@(r+j[:, :-1]@dx)/(b@b)
    assert np.linalg.norm(pr+pa@dx)**2 == pytest.approx(
        np.linalg.norm(r+j[:, :-1]@dx+b*dz)**2)


def test_shared_height_uses_all_views_and_matches_elimination():
    z = np.array([0., 3., 1., 6.])
    poses = np.array([[0., 0., 0.], [3., 8., 1.], [-4., 10., .5], [12., 6., -.2]])
    r = np.sqrt(np.sum((poses[:, :2]-[25., 11.])**2, axis=-1)+(z+14.)**2)
    data = Track([25., 11.], np.zeros((4, 2)), z, r, np.eye(4), physical_interval(z, 250.))
    assert physical_feasibility(data, poses).feasible
    opt = profile_minimum(data, poses)
    fit = least_squares(lambda h: joint_residual_jacobian(data, poses, h[0])[0],
                        [-4.], bounds=data.bounds)
    assert opt.height == pytest.approx(fit.x[0], abs=1e-8)
    e, j = joint_residual_jacobian(data, poses, opt.height)
    _, J_shared = projected_linearization(e,j)
    assert np.linalg.matrix_rank(J_shared, tol=1e-8) == 3
    height_pairs = np.column_stack([j[:,-1]*[1,1,0,0], j[:,-1]*[0,0,1,1]])
    paired_projection = np.eye(4)-height_pairs@np.linalg.pinv(height_pairs)
    J_paired = paired_projection@j[:,:-1]
    gain = J_shared.T@J_shared-J_paired.T@J_paired
    assert np.linalg.matrix_rank(J_paired, tol=1e-8) == 2
    assert np.min(np.linalg.eigvalsh(gain)) >= -1e-12
    assert np.max(np.linalg.eigvalsh(gain)) > .1
    broken = Track(data.xy, data.lever, z, r+[0., 0., 2., 2.], np.eye(4), data.bounds)
    assert not physical_feasibility(broken, poses).feasible
    near = Track(data.xy, data.lever, z, r+.1, np.eye(4), data.bounds)
    assert physical_feasibility(near, poses, range_tolerance=.11).feasible


def test_gtsam_custom_factor_uses_local_pose2_retract():
    gtsam = pytest.importorskip("gtsam")
    from research.sonar_range_factor_v2.gtsam_adapter import make_m1_factor
    track, poses = fixture(noise=(.1, -.2))
    keys = [gtsam.symbol("x", i) for i in range(2)]
    f = make_m1_factor(track, *keys)
    p = [gtsam.Pose2(*x) for x in poses]

    def g_values(dx):
        v = gtsam.Values()
        for k, pose, delta in zip(keys, p, dx.reshape(2, 3)):
            v.insert(k, pose.retract(delta))
        return v

    v = g_values(np.zeros(6))
    jac_factor = f.linearize(v)
    assert np.allclose(np.array(jac_factor.getA(), copy=True),
                       diff(lambda d: f.unwhitenedError(g_values(d)), np.zeros(6)), atol=1e-6)
    assert f.error(v) == pytest.approx(profile_minimum(track, poses).cost, abs=1e-12)
    graph = gtsam.NonlinearFactorGraph()
    graph.add(f)
    for k, pose in zip(keys, p):
        graph.add(gtsam.PriorFactorPose2(k, pose, gtsam.noiseModel.Isotropic.Sigma(3, .5)))
    v0 = g_values(np.array([0., 0., 0., .3, -.3, .02]))
    solved = gtsam.LevenbergMarquardtOptimizer(graph, v0).optimize()
    assert graph.error(solved) < graph.error(v0)


def test_reject_bad_calibration_and_nadir_singular():
    with pytest.raises(ValueError, match="symmetric"):
        fixture(covariance=[[1., .1], [.2, 1.]])
    with pytest.raises(np.linalg.LinAlgError):
        fixture(covariance=[[1., 2.], [2., 1.]])
    with pytest.raises(ValueError, match="below-sensor"):
        fixture(bounds=(-40., 1.))
    with pytest.raises(ValueError, match="nadir"):
        profile_minimum(Track([0., 0.], np.zeros((2, 2)), [0., 0.], [1., 1.],
                              np.eye(2), (-50., 0.)), np.zeros((2, 3)))


def test_global_candidate_search_against_dense_and_multiple_local_intervals():
    from research.sonar_range_factor_v2.models import geometry
    rng = np.random.default_rng(90671)
    for i in range(30):
        poses = np.c_[rng.uniform(-5.,5.,(2,2)), rng.uniform(-3.,3.,2)]
        z = np.sort(rng.uniform(-2.,8.,2))
        cross = (0., -.5, .45)[i%3]
        t = Track([10.,0.], np.zeros((2,2)), z, rng.uniform(2.,35.,2),
                  [[1.,cross],[cross,1.]], (-40.,z.min()))
        opt = profile_minimum(t,poses)
        _, _, d2 = geometry(t,poses)

        def objective(h):
            q = np.sqrt(d2+(h-z)**2)-t.ranges
            return q@t.precision@q/2

        grid = np.linspace(*t.bounds,1001)
        losses = np.array([objective(z) for z in grid])
        candidates = [losses[0],losses[-1]]
        for k in np.where((losses[1:-1]<losses[:-2])&(losses[1:-1]<losses[2:]))[0]+1:
            candidates.append(minimize_scalar(objective,method="bounded",
                              bounds=(grid[k-1],grid[k+1])).fun)
        assert opt.cost == pytest.approx(min(candidates),abs=1e-9)
