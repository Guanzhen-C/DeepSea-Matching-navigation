"""Isolated M1/M2 research; no imports or registration from production navigation."""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class Track:
    """One hypothesized landmark, fixed map XY, one independent pose slot per view."""

    xy: np.ndarray
    lever: np.ndarray
    sensor_z: np.ndarray
    ranges: np.ndarray
    covariance: np.ndarray
    bounds: tuple[float, float]
    whitener: np.ndarray = field(init=False, repr=False)
    precision: np.ndarray = field(init=False, repr=False)
    diagonal: bool = field(init=False, repr=False)

    def __post_init__(self):
        ranges = np.asarray(self.ranges, dtype=float)
        n = ranges.size
        shapes = dict(xy=(2,), lever=(n, 2), sensor_z=(n,), ranges=(n,), covariance=(n, n))
        for name, shape in shapes.items():
            arr = np.array(getattr(self, name), dtype=float, copy=True)
            if arr.shape != shape or not np.isfinite(arr).all():
                raise ValueError(f"Expected finite {name} with shape {shape}")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        lo, hi = map(float, self.bounds)
        if (n < 1 or np.any(ranges <= 0) or not np.isfinite([lo, hi]).all()
                or lo > hi or hi > np.min(self.sensor_z)):
            raise ValueError("Positive ranges and a finite physical below-sensor interval required")
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-13, rtol=1e-13):
            raise ValueError("Range covariance must be symmetric")
        chol = np.linalg.cholesky((self.covariance + self.covariance.T) * .5)
        whitener = np.linalg.solve(chol, np.eye(n))
        precision = whitener.T @ whitener
        whitener.setflags(write=False)
        precision.setflags(write=False)
        object.__setattr__(self, "bounds", (lo, hi))
        object.__setattr__(self, "whitener", whitener)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "diagonal", bool(np.count_nonzero(
            self.covariance - np.diag(self.covariance.diagonal())) == 0))

    def subset(self, indices):
        idx = np.asarray(indices)
        return Track(self.xy, self.lever[idx], self.sensor_z[idx], self.ranges[idx],
                     self.covariance[np.ix_(idx, idx)], self.bounds)


def physical_interval(sensor_z, maximum_range):
    """Necessary noise-independent envelope from sensor elevations and max range."""
    z = np.asarray(sensor_z, dtype=float)
    if z.ndim != 1 or not z.size or not np.isfinite(z).all() or not (
            np.isfinite(maximum_range) and maximum_range > 0):
        raise ValueError("Invalid height or maximum range")
    return float(np.max(z - maximum_range)), float(np.min(z))


def geometry(track, poses):
    poses = np.asarray(poses, dtype=float)
    if poses.shape != (len(track.ranges), 3) or not np.isfinite(poses).all():
        raise ValueError("A finite world-coordinate Pose2 is required for every view")
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    u, w = track.lever.T
    rotated = np.column_stack((c*u - s*w, s*u + c*w))
    v = track.xy - (poses[:, :2] + rotated)
    return v, rotated, np.einsum("ni,ni->n", v, v)


@dataclass(frozen=True)
class Feasibility:
    eta: np.ndarray
    roots: np.ndarray
    intersection: tuple[float, float]
    feasible: bool


def physical_feasibility(track, poses, *, range_tolerance=1e-9):
    """Exact below-branch diagnostic up to range_tolerance metres, not a noise gate.

    For tolerance eps, intersect the inverse image of each [r-eps, r+eps].
    """
    if not np.isfinite(range_tolerance) or range_tolerance < 0:
        raise ValueError("Invalid range tolerance")
    _, _, d2 = geometry(track, poses)
    r, z = track.ranges, track.sensor_z
    eta = r*r - d2
    roots = np.where(eta >= 0, z-np.sqrt(np.maximum(eta, 0)), np.nan)
    minimum = np.maximum(r-range_tolerance, 0)
    low = minimum*minimum-d2
    high = (r+range_tolerance)**2-d2
    if np.any(high < 0):
        return Feasibility(eta, roots, (np.inf, -np.inf), False)
    lo = max(track.bounds[0], float(np.max(z-np.sqrt(high))))
    hi = min(track.bounds[1], float(np.min(z-np.sqrt(np.maximum(low, 0)))))
    roundoff = 32*np.finfo(float).eps*max(1., abs(lo), abs(hi))
    return Feasibility(eta, roots, (lo, hi), bool(lo <= hi+roundoff))


def _height_terms(d2, sensor_z, height):
    dz = height-sensor_z
    rho = np.sqrt(d2+dz*dz)
    return rho, dz/rho, d2/rho**3


def _ensure_smooth_domain(track, d2):
    # The minimum rho on the downward branch is attained at the ceiling.
    if np.any(d2+(track.bounds[1]-track.sensor_z)**2 <= 1e-20):
        raise ValueError("Zero predicted range at domain ceiling: nonsmooth/nadir case")


def _product_bounds(a, b):
    p = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
    return min(p), max(p)


def _scale_add(bounds, product, scale):
    x, y = (scale*product[0], scale*product[1])
    return bounds[0]+min(x, y), bounds[1]+max(x, y)


def _corr_bounds(track, d2, l, h):
    """Derivative boxes on a downward branch, using monotonic scalar terms."""
    rl, bl, kl = _height_terms(d2, track.sensor_z, l)
    rh, bh, kh = _height_terms(d2, track.sensor_z, h)
    ql, qh = rl-track.ranges, rh-track.ranges
    w, n = track.precision, len(rl)
    curv = (float(np.dot(w.diagonal(), 1-track.ranges*d2/rh**3)),
            float(np.dot(w.diagonal(), 1-track.ranges*d2/rl**3)))
    deriv = (0., 0.)
    for i in range(n):
        qi, bi, ki = (qh[i], ql[i]), (bl[i], bh[i]), (kl[i], kh[i])
        for j in range(n):
            deriv = _scale_add(deriv, _product_bounds(qi, (bl[j], bh[j])), w[i, j])
            if j != i:
                curv = _scale_add(curv, _product_bounds(bi, (bl[j], bh[j])), w[i, j])
                curv = _scale_add(curv, _product_bounds(ki, (qh[j], ql[j])), w[i, j])
    return deriv, curv


@dataclass(frozen=True)
class ProfileMinimum:
    height: float
    cost: float
    branch: str
    ties: bool
    unresolved_intervals: int


def _minimum(track, d2, *, height_tol=1e-7, subdivision_limit=8192):
    """Find global candidates on the physical branch, not a single-start local fit."""
    _ensure_smooth_domain(track, d2)
    z, r, w = track.sensor_z, track.ranges, track.precision
    lo, hi = track.bounds
    weights = w.diagonal()

    def g(h):
        rho, b, _ = _height_terms(d2, z, h)
        return float((rho-r) @ w @ b)

    def diag_hessian(h):
        rho = np.sqrt(d2+(h-z)**2)
        return float(weights @ (1-r*d2/rho**3))

    candidates = [lo, hi]
    unresolved = 0
    if lo < hi and track.diagonal:
        # F'' is nonincreasing as the height approaches the sensors. Only
        # the increasing part of F' can contain an interior minimum.
        if diag_hessian(lo) > 0:
            peak = hi if diag_hessian(hi) >= 0 else brentq(diag_hessian, lo, hi, xtol=1e-12)
            if g(lo) < 0 < g(peak):
                candidates.append(brentq(g, lo, peak, xtol=1e-12))
    elif lo < hi:
        pending = [(lo, hi)]
        explored = 0
        while pending:
            explored += 1
            if explored > subdivision_limit:
                raise RuntimeError("Correlated profile search exhausted subdivision budget")
            l, h = pending.pop()
            deriv, curv = _corr_bounds(track, d2, l, h)
            pad = 1e-12*(1+sum(abs(x) for x in deriv+curv))
            if deriv[0] > pad or deriv[1] < -pad or curv[1] < -pad:
                continue  # no stationary point, or only an interior maximum
            if curv[0] > pad:
                gl, gh = g(l), g(h)
                if gl <= 0 <= gh:
                    candidates.append(brentq(g, l, h, xtol=1e-12))
                continue
            mid = (l+h)*.5
            if h-l <= height_tol:
                candidates.extend((l, mid, h))
                unresolved += 1
            else:
                pending.extend(((l, mid), (mid, h)))

    def cost(h):
        q = np.sqrt(d2+(h-z)**2)-r
        e = track.whitener @ q
        return float(e @ e)*.5

    values = sorted((cost(h), h) for h in candidates)
    f, h = values[0]
    ties = any(abs(other-h) > 8*height_tol and abs(f_other-f) <= 1e-11*(1+f)
               for f_other, other in values[1:])
    branch = ("singleton" if lo == hi else "lower" if h == lo else
              "upper" if h == hi else "interior")
    return ProfileMinimum(float(h), f, branch, bool(ties), unresolved)


def profile_minimum(track, poses):
    """M1 pair minimum (also usable offline for an M2 track of arbitrary length)."""
    return _minimum(track, geometry(track, poses)[2])


def _joint(track, geom, height, *, exact_hessian=False):
    """Whitened signed residual, full world Jacobian, optional exact F Hessian."""
    v, u, d2 = geom
    rho, b, c = _height_terms(d2, track.sensor_z, height)
    q = rho-track.ranges
    n = len(q)
    G = np.zeros((n, 2, 3))
    G[:, 0, 0] = G[:, 1, 1] = 1
    G[:, :, 2] = np.column_stack((-u[:, 1], u[:, 0]))
    ap = -np.einsum("ni,nij->nj", v, G)/rho[:, None]
    jac = np.zeros((n, 3*n+1))
    cols = np.arange(n)[:, None]*3+np.arange(3)
    jac[np.arange(n)[:, None], cols] = ap
    jac[:, -1] = b
    C = track.whitener
    r, J = C@q, C@jac
    if not exact_hessian:
        return r, J, None
    H = J.T@J
    for i, qi in enumerate(track.precision@q):
        ai, Gi, idx = ap[i], G[i], slice(3*i, 3*i+3)
        ii = (Gi.T@Gi-np.outer(ai, ai))/rho[i]
        ii[2, 2] += (v[i] @ u[i])/rho[i]
        iz = -ai*b[i]/rho[i]
        H[idx, idx] += qi * ii
        H[idx, -1] += qi * iz
        H[-1, idx] += qi * iz
        H[-1, -1] += qi*c[i]
    return r, J, H


def joint_residual_jacobian(track, poses, height):
    """M2 explicit raw-range factor; callers own the shared bounded Z variable."""
    if not np.isfinite(height) or not track.bounds[0] <= height <= track.bounds[1]:
        raise ValueError("Height outside the physical interval")
    geom = geometry(track, poses)
    _ensure_smooth_domain(track, geom[2])
    r, J, _ = _joint(track, geom, height)
    return r, J


def profile_value_gradient_hessian(track, poses):
    """M1 smooth profile derivatives; reject a switch, a tie or a singular optimum."""
    geom = geometry(track, poses)
    opt, r, J, H, dz_dp = _profiled(track, geom)
    grad = J[:, :-1].T@r
    return opt, grad, H[:-1, :-1]+np.outer(H[:-1, -1], dz_dp)


def _profiled(track, geom):
    opt = _minimum(track, geom[2])
    if opt.ties or opt.unresolved_intervals:
        raise ValueError("Competing minima or unresolved profile; derivatives not certified")
    r, J, H = _joint(track, geom, opt.height, exact_hessian=True)
    if opt.branch == "interior":
        if H[-1, -1] <= 1e-10:
            raise ValueError("Near-singular interior profile curvature")
        dz_dp = -H[-1, :-1]/H[-1, -1]
    else:
        # All z_i equal to the fixed ceiling implies F_Z(ceiling)=0 for
        # every nearby pose. Positive curvature then gives a stable ceiling
        # solution, not an active-set switch.
        stable_flat_ceiling = (opt.branch == "upper"
            and np.all(track.sensor_z == track.bounds[1]) and H[-1, -1] > 1e-10)
        if opt.branch != "singleton" and not stable_flat_ceiling and abs(J[:, -1]@r) <= 1e-9:
            raise ValueError("Active-set transition; no ordinary profile Hessian")
        dz_dp = np.zeros(J.shape[1]-1)
    return opt, r, J, H, dz_dp


def profiled_residual_jacobian(track, poses):
    """M1 signed vector at Z*(p); preserves 2F without sqrt(profile objective)."""
    opt, r, J, _, dz_dp = _profiled(track, geometry(track, poses))
    return r, J[:, :-1]+np.outer(J[:, -1], dz_dp), opt


def projected_linearization(residual, joint_jacobian, *, tolerance=1e-12):
    """Eliminate an *unconstrained height increment* in a linearized whitened system."""
    b, A = joint_jacobian[:, -1], joint_jacobian[:, :-1]
    norm = np.linalg.norm(b)
    if norm <= tolerance:
        return residual.copy(), A.copy()
    u = b/norm
    return residual-u*(u@residual), A-np.outer(u, u@A)
