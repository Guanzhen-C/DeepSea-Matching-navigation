"""Observable-subspace side-scan scan-plane registration against XY landmarks.

Only measured scan-plane normals/offsets and matched map XY are inputs. Slant
range is NOT a horizontal measurement. Straight/opposite straight scan planes
retain the real translation null; diverse planes may support three DOFs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

PARAMETER_SCALES = np.array([20., 20., np.deg2rad(5.)])


def plane_residual_jacobian(parameters, points, normals, offsets):
    """Translation is in candidate body frame, heading relative to map seed."""
    x, y, theta = parameters
    c, s = np.cos(theta), np.sin(theta)
    rotated = points@np.array([[c, -s], [s, c]])
    residual = np.einsum('ij,ij->i', normals, rotated-[x, y])-offsets
    jac = np.column_stack([-normals, normals[:, 0]*rotated[:, 1]-normals[:, 1]*rotated[:, 0]])
    return residual, jac


def observed_basis(jac, minimum_effective_support=0.):
    # Balance angle/translation units for a numerical rank test, not a prior
    # or invented measurement. A 180-degree normal sign change adds no rank.
    scale = np.diag(PARAMETER_SCALES)
    effective_count = min(24, len(jac))
    whitened = jac@scale/.7*np.sqrt(effective_count/len(jac))
    u, values, vh = np.linalg.svd(whitened, full_matrices=False)
    # An otherwise missing DOF can be fitted by ONE false correspondence.
    # Require distributed measurement support, using Kish effective sample
    # size of each singular direction, not true pose/error thresholds.
    effective_support = 1./np.maximum(np.sum(u**4, axis=0), 1e-30)
    # Numerical full rank is not statistical observability. If motion across
    # the whole proposal scale cannot change a residual above its noise, do
    # not invert that weak singular direction. This is a measurement-SNR test,
    # never a true-position error gate or a new measurement prior.
    supported = (values >= 1.) & (values > max(float(values.max()), 1.)*1e-7) & (effective_support >= minimum_effective_support)
    rank = int(np.count_nonzero(supported))
    return scale@vh[supported].T, rank


def shrink_information(information, covariance):
    values, u = np.linalg.eigh((information+information.T)*.5)
    root = (u*np.sqrt(np.maximum(values, 0)))@u.T
    result = root@np.linalg.solve(np.eye(3)+root@covariance@root, root)
    return (result+result.T)*.5


def fit_observed_scan_planes(points, normals, offsets, *, seed=31, minimum_inliers=24):
    points, normals, offsets = np.asarray(points, float), np.asarray(normals, float), np.asarray(offsets, float)
    n = len(points)
    if (points.shape != (n, 2) or normals.shape != (n, 2) or offsets.shape != (n,)
            or not all(np.isfinite(x).all() for x in (points, normals, offsets))):
        raise ValueError('Finite matched XY landmarks and measured scan planes required.')
    if n < minimum_inliers: return dict(accepted=False, reason='not_enough_scan_plane_correspondences', inliers=0)
    base, jac = plane_residual_jacobian(np.zeros(3), points, normals, offsets)
    minimum_directional_support = max(3., minimum_inliers/4.)
    basis, rank = observed_basis(jac, minimum_directional_support)
    if rank < 2: return dict(accepted=False, reason='insufficient_scan_plane_observability', inliers=0)
    linear = jac@basis; rng = np.random.default_rng(seed); best = None
    for _ in range(256):
        ii = rng.choice(n, rank, replace=False)
        if np.linalg.cond(linear[ii]) > 1e8: continue
        beta = basis@np.linalg.solve(linear[ii], -base[ii])
        beta[2] = np.arctan2(np.sin(beta[2]), np.cos(beta[2]))
        residual, _ = plane_residual_jacobian(beta, points, normals, offsets)
        good = abs(residual) <= .7
        cost = float(np.sum(np.minimum(residual**2, .7**2)))
        if best is None or (int(good.sum()), -cost) > (best[0], -best[1]):
            best = int(good.sum()), cost, good, beta
    if best is None or best[0] < minimum_inliers:
        return dict(accepted=False, reason='insufficient_scan_plane_consensus', inliers=0 if best is None else best[0])
    good, initial = best[2], best[3]
    # Recompute the subspace from INLIERS: false correspondences cannot supply
    # a missing DOF. Unsupported components remain at the original motion seed.
    for _ in range(2):
        _, j = plane_residual_jacobian(initial, points[good], normals[good], offsets[good])
        basis, rank = observed_basis(j, minimum_directional_support)
        if rank < 2: return dict(accepted=False, reason='inlier_geometry_unobservable', inliers=int(good.sum()))
        normalization = np.diag(1./PARAMETER_SCALES)
        eta0 = np.linalg.pinv(normalization@basis)@(normalization@initial)
        fit = least_squares(lambda eta: plane_residual_jacobian(basis@eta, points[good], normals[good], offsets[good])[0],
            eta0,
            jac=lambda eta: plane_residual_jacobian(basis@eta, points[good], normals[good], offsets[good])[1]@basis,
            loss='huber', f_scale=.3)
        initial = basis@fit.x
        residual, _ = plane_residual_jacobian(initial, points, normals, offsets)
        new_good = abs(residual) <= .7
        if new_good.sum() < minimum_inliers:
            return dict(accepted=False, reason='refined_scan_plane_consensus_insufficient', inliers=int(new_good.sum()))
        if np.array_equal(new_good, good): break
        good = new_good
    residual, jac = plane_residual_jacobian(initial, points[good], normals[good], offsets[good])
    # Information from data ONLY; no optimizer seed rows. Effective feature
    # count is capped at 24 to avoid treating dense image points as independent.
    weight = np.minimum(1., .3/np.maximum(abs(residual), 1e-12))
    normalization = np.diag(1./PARAMETER_SCALES)
    projection = basis@np.linalg.pinv(normalization@basis)@normalization
    j = jac@projection
    information = (j.T*weight)@j/.7**2*min(24, len(j))/len(j)
    information = shrink_information(information, np.diag([2.5**2, 2.5**2, np.deg2rad(1.5)**2]))
    return dict(accepted=True, parameters=initial, information=information, inlier_mask=good,
        observable_rank=rank, inliers=int(good.sum()), scan_plane_rmse_m=float(np.sqrt(np.mean(residual**2))),
        optimizer_seed_information_included=False, slant_range_used_as_horizontal=False)


def fit_to_world_pose(seed_pose2, parameters, information):
    """Transform BOTH estimate and precision to graph's world-coordinate error."""
    theta = seed_pose2[2]+parameters[2]; c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s], [s, c]])
    pose = np.r_[np.asarray(seed_pose2)[:2]+r@parameters[:2], theta]
    jac = np.eye(3); jac[:2, :2] = r
    jac[:2, 2] = r@np.array([-parameters[1], parameters[0]])
    inverse = np.linalg.inv(jac)
    world_information = inverse.T@information@inverse
    return pose, (world_information+world_information.T)*.5
