"""Two-view SSS range/scan-plane loop geometry with UNKNOWN 3-D landmarks.

Inspired by the measurement model in Zhang et al., DIASSS (IET RSN 2024).
This is an independent planar/pressure-assisted implementation, not copied
code or an exact reproduction. No DEM/MBES/flat-bottom depth prior is used.
Motion only stabilizes the nonlinear fit; it is EXCLUDED from the returned
Schur-marginalized information, so odometry is not counted as sonar data.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def wrap(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))


def range_plane_residual_jacobian(pose, landmarks, a, b, delta_z, range_std=.7, plane_std=.7):
    """4 residuals per match; columns are pose3 followed by each landmark3."""
    p = np.asarray(pose); l = np.asarray(landmarks); n = len(l); r = rotation(p[2])
    pb = np.column_stack([(l[:, :2]-p[:2])@r, l[:, 2]-delta_z])
    da, db = l-a['origins'], pb-b['origins']
    ra, rb = np.maximum(np.linalg.norm(da, axis=1), 1e-9), np.maximum(np.linalg.norm(db, axis=1), 1e-9)
    ua, ub = da/ra[:, None], db/rb[:, None]
    residual = np.column_stack([(ra-a['ranges'])/range_std,
        np.einsum('ij,ij->i', a['normals'], da[:, :2])/plane_std,
        (rb-b['ranges'])/range_std,
        np.einsum('ij,ij->i', b['normals'], db[:, :2])/plane_std])
    jac = np.zeros((n*4, 3+3*n))
    for i in range(n):
        j = slice(3+3*i, 6+3*i)
        jac[4*i, j] = ua[i]/range_std
        jac[4*i+1, j] = np.r_[a['normals'][i], 0.]/plane_std
        world_ub = ub[i, :2]@r.T
        world_nb = b['normals'][i]@r.T
        jac[4*i+2, :2] = -world_ub/range_std
        jac[4*i+3, :2] = -world_nb/plane_std
        jac[4*i+2, 2] = (ub[i, 0]*pb[i, 1]-ub[i, 1]*pb[i, 0])/range_std
        jac[4*i+3, 2] = (b['normals'][i, 0]*pb[i, 1]-b['normals'][i, 1]*pb[i, 0])/plane_std
        jac[4*i+2, j] = np.r_[world_ub, ub[i, 2]]/range_std
        jac[4*i+3, j] = np.r_[world_nb, 0.]/plane_std
    return residual.ravel(), jac


def data_only_schur_information(jacobian, points):
    """Eliminate landmark variables. No optimizer/motion prior rows allowed."""
    j = np.asarray(jacobian, float)
    if j.shape != (4*points, 3+3*points):
        raise ValueError('Exactly data-only range/plane rows required, without seed regularization.')
    result = np.zeros((3, 3))
    for i in range(points):
        rows = slice(4*i, 4*i+4)
        jp, jl = j[rows, :3], j[rows, 3+3*i:6+3*i]
        null_projector = np.eye(4)-jl@np.linalg.pinv(jl, rcond=1e-10)
        result += jp.T@null_projector@jp
    result = (result+result.T)*.5
    values, basis = np.linalg.eigh(result)
    values[values < max(float(values.max()), 1.)*1e-9] = 0.
    return (basis*values)@basis.T


def inflate_information(information, extra_covariance):
    """Add model uncertainty without ever filling a true information null."""
    values, u = np.linalg.eigh((information+information.T)*.5)
    root = (u*np.sqrt(np.maximum(values, 0)))@u.T
    result = root@np.linalg.solve(np.eye(len(root))+root@extra_covariance@root, root)
    return (result+result.T)*.5


def fit_range_plane_loop(a, b, seed_pose, delta_z=0., *, maximum_features=48):
    n = len(a['ranges'])
    if n < 6 or len(b['ranges']) != n:
        return dict(accepted=False, reason='not_enough_independent_feature_correspondences')
    for observation in (a, b):
        if (np.asarray(observation['origins']).shape != (n, 3)
                or np.asarray(observation['normals']).shape != (n, 2)
                or np.asarray(observation['sides']).shape != (n,)
                or any(not np.isfinite(np.asarray(v)).all() for v in observation.values())
                or np.any(np.asarray(observation['ranges']) <= 0)):
            raise ValueError('Observed slant ranges, scan origins and plane normals required.')
    if n > maximum_features:
        # Deterministic spread over observed ranges, never truth/geological labels.
        order = np.argsort(a['ranges'])
        keep = order[np.linspace(0, n-1, maximum_features).round().astype(int)]
        a = {k: np.asarray(v)[keep] for k, v in a.items()}
        b = {k: np.asarray(v)[keep] for k, v in b.items()}
        n = len(keep)
    seed = np.asarray(seed_pose, float).reshape(3)
    std_seed = np.array([50., 50., np.deg2rad(30.)])
    z_ceiling = np.minimum(a['origins'][:, 2], b['origins'][:, 2]+delta_z)-.01
    lower = np.full(3+3*n, -np.inf); upper = np.full(3+3*n, np.inf)
    lower[5::3] = -2*max(float(np.max(a['ranges'])), float(np.max(b['ranges'])))
    upper[5::3] = z_ceiling
    solutions = []
    for fraction in (.25, .6, .9):
        h = np.minimum(a['ranges']*fraction, a['ranges']*.99)
        lateral = np.sqrt(np.maximum(a['ranges']**2-h**2, 0))*a['sides']
        tangent = np.column_stack([-a['normals'][:, 1], a['normals'][:, 0]])
        landmarks = a['origins'].copy()
        landmarks[:, :2] += tangent*lateral[:, None]
        landmarks[:, 2] = np.minimum(a['origins'][:, 2]-h, z_ceiling-.01)
        x0 = np.r_[seed, landmarks.ravel()]
        def functions(x):
            residual, jac = range_plane_residual_jacobian(x[:3], x[3:].reshape(n, 3), a, b, delta_z)
            prior = x[:3]-seed; prior[2] = wrap(prior[2])
            return np.r_[residual, prior/std_seed], np.vstack([jac,
                np.column_stack([np.diag(1/std_seed), np.zeros((3, 3*n))])])
        fit = least_squares(lambda x: functions(x)[0], x0,
            jac=lambda x: csr_matrix(functions(x)[1]), bounds=(lower, upper),
            loss='soft_l1', f_scale=1., max_nfev=100, tr_solver='lsmr')
        residual, raw_jac = range_plane_residual_jacobian(fit.x[:3], fit.x[3:].reshape(n, 3), a, b, delta_z)
        # Selection is data residual with the weak seed solely breaking ties;
        # no true pose/height is available to this solver.
        cost = float(np.mean(np.log1p(residual**2)))+.001*float(np.sum(((fit.x[:3]-seed)/std_seed)**2))
        solutions.append((cost, fit, residual, raw_jac))
    cost, fit, residual, jac = min(solutions, key=lambda x: x[0])
    landmarks = fit.x[3:].reshape(n, 3)
    pb = np.column_stack([(landmarks[:, :2]-fit.x[:2])@rotation(fit.x[2]), landmarks[:, 2]-delta_z])
    for observations, local_points in ((a, landmarks), (b, pb)):
        tangent = np.column_stack([-observations['normals'][:, 1], observations['normals'][:, 0]])
        signed = np.einsum('ij,ij->i', tangent, local_points[:, :2]-observations['origins'][:, :2])*observations['sides']
        if np.any(signed < -3*.7):
            return dict(accepted=False, reason='historical_feature_on_wrong_physical_sonar_side')
    robust_weight = (1+residual**2)**(-.25)
    info = data_only_schur_information(jac*robust_weight[:, None], n)*min(24, n)/n
    info = inflate_information(info, np.diag([2.5**2, 2.5**2, np.deg2rad(1.5)**2]))
    values = np.linalg.eigvalsh(info)
    rank = int(np.count_nonzero(values > max(float(values.max()), 1.)*1e-9))
    standardized_rmse = float(np.sqrt(np.mean(residual**2)))
    if rank == 0 or not np.isfinite(fit.x).all() or standardized_rmse > 2.:
        return dict(accepted=False, reason='range_plane_geometry_not_supported',
                    standardized_rmse=standardized_rmse, observable_rank=rank)
    pose = fit.x[:3].copy(); pose[2] = wrap(pose[2])
    return dict(accepted=True, relative_pose2=pose, information=info,
        observable_rank=rank, information_eigenvalues=values, standardized_rmse=standardized_rmse,
        features=n, landmark_depth_prior_used=False, motion_seed_information_added_to_measurement=False,
        measured_height_or_DEM_used=False, estimator='unknown_landmark_range_plane_Schur',
        caveat='Planar, pressure-assisted local geometry; landmark depth ambiguities remain.')


def lift_loop_to_graph_nodes(relative_pose, information, old_node_to_frame, new_node_to_frame,
                             *, offset_covariance=None):
    from SideScanFeatureMap import pose_matrix, matrix_pose
    a, b = np.asarray(old_node_to_frame), np.asarray(new_node_to_frame)
    inverse_b = np.linalg.inv(b)
    m = pose_matrix(relative_pose)
    measurement = matrix_pose(a@m@inverse_b)
    lever = a[:2, :2]@m[:2, :2]@np.array([-inverse_b[1, 2], inverse_b[0, 2]])
    jac = np.eye(3); jac[:2, :2] = a[:2, :2]; jac[:2, 2] = lever
    inverse = np.linalg.inv(jac)
    transformed = inverse.T@information@inverse
    if offset_covariance is not None: transformed = inflate_information(transformed, offset_covariance)
    return measurement, (transformed+transformed.T)*.5
