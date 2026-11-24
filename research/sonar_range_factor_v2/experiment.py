"""Reproducible paired synthetic study; no production writes or real-data inputs."""
import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.stats import chi2

from standalone.SideScanMappedRangeChange import (
    MappedRangeChange, conservative_range_change_sigma, profiled_range_residual_jacobians,
)
from .models import (
    Track, geometry, joint_residual_jacobian, physical_feasibility, physical_interval,
    profiled_residual_jacobian,
)

RMAX = 250.
PRIOR_STD = 4.
M0_SHA256 = "a5f3640b954547669aa5ce01f3c2ea34110267e7baee2b16f4568006ca3bef42"
LOOP_BASELINE = "1efc20322a7a0e8dd0fcc13b3ac9bc59df30a8df"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    height_difference: float = 2.
    heading_difference_deg: float = 30.
    terrain_relief: float = 3.
    range_std: float = .35
    outlier_fraction: float = 0.
    map_std: float = 0.
    repetitions: int = 4
    lateral_baseline: float = 20.
    features: int = 8


@dataclass(frozen=True)
class Scene:
    config: Config
    initial_poses: np.ndarray
    true_shift: np.ndarray
    tracks: list
    sides: np.ndarray
    height: np.ndarray
    n_swapped: int

    @property
    def affected(self):
        return np.arange(self.config.repetitions) % 2 == 1

    @property
    def true_poses(self):
        return self.at(self.true_shift)

    def at(self, shift):
        p = self.initial_poses.copy()
        p[self.affected, :2] += shift
        return p

    def contractions(self):
        n = self.config.repetitions
        B = np.zeros((n*3, 2))
        for i in np.flatnonzero(self.affected):
            B[3*i:3*i+2] = np.eye(2)
        return B


def synthesize(config, seed):
    """Each landmark has a different ping on each line; headings are known."""
    rng = np.random.default_rng(seed)
    n, m = config.repetitions, config.features
    if n < 2 or n % 2 or m < 6:
        raise ValueError("Paired study needs even visits and at least six landmarks")
    point = rng.uniform([-25., -17.], [25., 17.], (m, 2))
    terrain = np.sin(point[:, 0]/9)*np.cos(point[:, 1]/7)
    height = -17. + config.terrain_relief*terrain
    # Draw all nuisance templates even when the swept multiplier is zero.
    chart = point + config.map_std*rng.normal(size=(m, 2))
    shift = rng.normal(0., PRIOR_STD, size=2)
    k = np.arange(n)
    angle = np.deg2rad(config.heading_difference_deg*(k % 2))
    forward = np.column_stack([np.cos(angle), np.sin(angle)])
    left = np.column_stack([-np.sin(angle), np.cos(angle)])
    z = config.height_difference*(k % 2) + .4*config.height_difference*np.floor(k/2)
    # Distribute centers to both sides on later traversals. Ensure same geometry
    # for M0/M1/M2 and reject no samples based on a competing method.
    cross = np.where(k % 4 < 2, 48., -48.) - (k % 2)*config.lateral_baseline
    centers = cross[:, None]*left
    old_pose = np.column_stack([centers, angle])
    old_pose[k % 2 == 1, :2] -= shift
    along = point@forward.T
    along += rng.uniform(-.25, .25, (m, n))
    levers = np.stack([along, np.zeros_like(along)], axis=-1)
    origins = centers[None, :, :] + along[:, :, None]*forward[None]
    displacement = point[:, None]-origins
    ranges = np.sqrt(np.sum(displacement**2, axis=-1)+(height[:, None]-z[None])**2)
    ranges += config.range_std*rng.normal(size=ranges.shape)
    sides = np.sign(np.einsum("mni,ni->mn", displacement, left))
    n_bad = int(round(m*config.outlier_fraction))
    if n_bad >= 2:
        # Swap actual measurements among targets, not a large arbitrary value.
        idx = rng.choice(m, n_bad, replace=False)
        ranges[idx, -1] = ranges[np.roll(idx, 1), -1]
    else:
        n_bad = 0
    cov = np.eye(n)*max(config.range_std, .05)**2
    bounds = physical_interval(z, RMAX)
    tracks = [Track(chart[i], levers[i], z, ranges[i], cov, bounds) for i in range(m)]
    return Scene(config, old_pose, shift, tracks, sides, height, n_bad)


def m0_batches(scene, *, untempered=False):
    from numpy import array
    result = []
    t = scene.tracks
    for a in range(0, scene.config.repetitions, 2):
        b = a+1
        ranges_a, ranges_b = (array([tr.ranges[i] for tr in t]) for i in (a, b))
        baseline = np.linalg.norm(scene.initial_poses[a, :2]-scene.initial_poses[b, :2])
        sigma = conservative_range_change_sigma(
            ranges_a, ranges_b, baseline, range_std_m=max(scene.config.range_std, .05),
            map_xy_std_m=scene.config.map_std, receiver_offset_std_m=0.)
        data = MappedRangeChange(array([tr.xy for tr in t]),
            array([tr.lever[a] for tr in t]), array([tr.lever[b] for tr in t]),
            array([tr.sensor_z[a] for tr in t]), array([tr.sensor_z[b] for tr in t]),
            ranges_a, ranges_b, sigma, maximum_range_m=RMAX,
            information_budget=1. if untempered else .5)
        result.append((a, b, data))
    return result


def with_map_covariance(scene, track):
    """Freeze G_L at the recorded initial pose and a fixed non-truth height."""
    v, _, d2 = geometry(track, scene.initial_poses)
    h = np.clip(-12., *track.bounds)
    G = v/np.sqrt(d2+(h-track.sensor_z)**2)[:, None]
    covariance = track.covariance + scene.config.map_std**2*(G@G.T)
    return Track(track.xy, track.lever, track.sensor_z, track.ranges, covariance, track.bounds)


def solve(scene, label, loop=None):
    """Same 2D shift prior/initialization; height starts are not truth estimates."""
    B = scene.contractions()
    tracks = ([with_map_covariance(scene, t) for t in scene.tracks]
              if label.endswith("_map_cov") else scene.tracks)
    m = len(tracks)
    p0 = np.zeros(2)
    bounded = (np.full(2, -np.inf), np.full(2, np.inf))
    if label.startswith("M2"):
        lo, hi = (np.array([t.bounds[k] for t in tracks]) for k in [0, 1])
        p0 = np.r_[p0, np.clip(np.full(m, -12.), lo+1e-6, hi-1e-6)]
        bounded = np.r_[bounded[0], lo], np.r_[bounded[1], hi]
    if label.startswith("M0"):
        batches = m0_batches(scene, untempered=(label == "M0_plain"))
    elif label.startswith("M1"):
        batches = [(i, i+1, t.subset([i, i+1])) for t in tracks
                   for i in range(0, scene.config.repetitions, 2)]
    dim = len(p0)
    if loop is not None:
        target, root = loop

    def evaluate(x):
        pose = scene.at(x[:2])
        re = [x[:2]/PRIOR_STD]
        ja = [np.pad(np.eye(2)/PRIOR_STD, ((0, 0), (0, dim-2)))]
        if loop is not None:
            re.append(root@(x[:2]-target))
            ja.append(np.pad(root, ((0, 0), (0, dim-2))))
        if label.startswith("M0"):
            for a, b, data in batches:
                e, A, D = profiled_range_residual_jacobians(pose[a], pose[b], data,
                                                           robust=(label == "M0"))
                re.append(e)
                ja.append(A@B[3*a:3*a+3]+D@B[3*b:3*b+3])
        elif label.startswith("M1"):
            for a, b, t in batches:
                e, jac, _ = profiled_residual_jacobian(t, pose[[a, b]])
                re.append(e)
                ja.append(jac[:, :3]@B[3*a:3*a+3]+jac[:, 3:]@B[3*b:3*b+3])
        elif label.startswith("M2"):
            for i, t in enumerate(tracks):
                e, j = joint_residual_jacobian(t, pose, x[2+i])
                J = np.zeros((len(e), dim))
                J[:, :2] = j[:, :-1]@B
                J[:, i+2] = j[:, -1]
                re.append(e)
                ja.append(J)
        return np.concatenate(re), np.vstack(ja)

    memo = None

    def evaluate_cached(x):
        nonlocal memo
        if memo is None or not np.array_equal(memo[0], x):
            memo = x.copy(), evaluate(x)
        return memo[1]

    start = time.perf_counter()
    try:
        fit = least_squares(lambda x: evaluate_cached(x)[0], p0,
                            jac=lambda x: evaluate_cached(x)[1], bounds=bounded,
                            max_nfev=100, ftol=1e-9, gtol=1e-9, xtol=1e-9)
    except (ValueError, RuntimeError) as exc:
        return dict(error_xy_m=None, success=False, reason=str(exc),
                    ms=(time.perf_counter()-start)*1000)
    ms = (time.perf_counter()-start)*1000
    return dict(error_xy_m=(fit.x[:2]-scene.true_shift).tolist(), success=bool(fit.success),
                ms=ms, nfev=fit.nfev, half_cost=float(fit.cost))


def feasibility(scene):
    """Run the diagnostic for all candidate tracks, never discard by truth."""
    levels = [("initial", scene.initial_poses), ("truth_geometry", scene.true_poses)]
    # Tolerance is declared for this synthetic diagnostic, not a calibrated
    # acceptance threshold. Map errors are not included in this simple tolerance.
    eps = max(scene.config.range_std, .05)*2
    summary = {}
    for key, poses in levels:
        exact = tol = negative = pair_exact = pair_tol = pair_negative = 0
        for t in scene.tracks:
            check = physical_feasibility(t, poses)
            exact += check.feasible
            negative += np.any(check.eta < 0)
            tol += physical_feasibility(t, poses, range_tolerance=eps).feasible
            for a in range(0, scene.config.repetitions, 2):
                ti, pi = t.subset([a, a+1]), poses[[a, a+1]]
                check = physical_feasibility(ti, pi)
                pair_exact += check.feasible
                pair_negative += np.any(check.eta < 0)
                pair_tol += physical_feasibility(ti, pi, range_tolerance=eps).feasible
        summary[key] = dict(exact=int(exact), tolerance=int(tol), negative_eta=int(negative),
                            pair_exact=int(pair_exact), pair_tolerance=int(pair_tol),
                            pair_negative=int(pair_negative))
    return summary


def scan_design():
    opts = {
        "height_difference": [0., 1e-6, .5, 2., 8., 16.],
        "heading_difference_deg": [0., 30., 60., 90., 135., 180.],
        "terrain_relief": [0., 3., 8., 12.],
        "range_std": [0., .15, .35, .7, 1.4],
        "outlier_fraction": [0., .25, .5],
        "map_std": [0., .35, 1., 2.5],
        "repetitions": [2, 4, 6],
        "lateral_baseline": [3., 20., 36.],
    }
    return [(k, v, replace(Config(), **{k: v})) for k, values in opts.items() for v in values]


def summary(runs, label):
    records = [r["models"][label] for r in runs]
    ok = [r for r in records if r["success"] and r["error_xy_m"] is not None]
    err = np.linalg.norm([r["error_xy_m"] for r in ok], axis=1) if ok else np.array([])
    ms = [r["ms"] for r in records]
    return dict(n=len(records), failed=len(records)-len(ok),
                rmse_m=float(np.sqrt(np.mean(err**2))) if len(err) else None,
                p95_m=float(np.quantile(err,.95)) if len(err) else None,
                median_ms=float(np.median(ms)), p95_ms=float(np.quantile(ms,.95)))


def paired_bootstrap(runs, a, b):
    pairs = [(r["models"][a]["error_xy_m"], r["models"][b]["error_xy_m"]) for r in runs
             if r["models"][a]["success"] and r["models"][b]["success"]]
    if not pairs:
        return None
    squares = np.sum(np.array(pairs)**2, axis=2)
    rng = np.random.default_rng(4203)
    group = rng.integers(0, len(squares), (3000, len(squares)))
    differences = np.sqrt(squares[group, 0].mean(axis=1))-np.sqrt(squares[group, 1].mean(axis=1))
    return dict(n_paired=len(squares), delta_rmse_m=float(np.diff(
        np.sqrt(squares.mean(axis=0))[::-1])[0]),
        ci95_m=np.quantile(differences, [.025, .975]).tolist())


def statistics_demo(seed=58217, n=6000):
    """Exact linear-Gaussian fixture with physical nuisance Jacobians."""
    scene = synthesize(Config(repetitions=6), seed=809)
    t = scene.tracks[3]
    poses = scene.true_poses
    # Unwhiten by building a copy with identity covariance at the known geometry.
    identity = Track(t.xy,t.lever,t.sensor_z,t.ranges,np.eye(len(t.ranges)),t.bounds)
    _, J = joint_residual_jacobian(identity, poses, scene.height[3])
    m = len(J)
    S_xy = -np.array([J[i,3*i:3*i+2] for i in range(m)])
    rng = np.random.default_rng(seed)
    A = J[:,:-1]@scene.contractions()
    independent = .18**2*np.eye(m)
    # Illustrative, not estimated: common map error, shared installation and
    # vertical offset, range-proportional sound speed and time-zero range bias.
    G = np.column_stack([S_xy*.45, -S_xy*.25, -J[:,-1]*.18,
                         t.ranges*.004, np.full(m,.15)])
    S = independent+G@G.T
    P = np.eye(2)*PRIOR_STD**2
    x = rng.multivariate_normal(np.zeros(2), P, n)
    y = x@A.T+rng.multivariate_normal(np.zeros(m), S, n)
    outputs = {}
    for key, assumed in dict(full=S, diagonal=np.diag(S.diagonal())).items():
        Sy = A@P@A.T+assumed
        K = np.linalg.solve(Sy, A@P).T
        posterior = np.linalg.inv(np.linalg.inv(P)+A.T@np.linalg.solve(assumed,A))
        error = x-y@K.T
        nis = np.einsum("ni,ij,nj->n", y, np.linalg.inv(Sy), y)
        nees = np.einsum("ni,ij,nj->n", error, np.linalg.inv(posterior), error)
        outputs[key] = dict(mean_NIS=float(nis.mean()), mean_NEES=float(nees.mean()),
                            NIS_95_coverage=float((nis<=chi2.ppf(.95,m)).mean()),
                            NEES_95_coverage=float((nees<=chi2.ppf(.95,2)).mean()))
    return dict(seed=seed, n=n, NIS_dof=m, NEES_dof=2, outputs=outputs, covariance=S.tolist())


def loop_fixture():
    """Run the actual frozen range+plane geometry with synthetic known associations.

    This is not the historical candidate detector nor an independent observation
    when combined with range: both arms consume the first two range views.
    """
    path = "standalone/SideScanLoopGeometry.py"
    source = subprocess.check_output(["git","show",f"{LOOP_BASELINE}:{path}"], cwd=ROOT)
    namespace = {"__name__": "frozen_loop_research_only"}
    exec(compile(source, f"{LOOP_BASELINE}:{path}", "exec"), namespace)
    case = synthesize(Config(repetitions=2, heading_difference_deg=60, range_std=.15), seed=829)
    p = case.initial_poses
    a, b = [dict(origins=np.column_stack([np.array([t.lever[i] for t in case.tracks]),
                    np.zeros(len(case.tracks))]),
                 normals=np.tile([1.,0.],(len(case.tracks),1)),
                 sides=case.sides[:,i],
                 ranges=np.array([t.ranges[i] for t in case.tracks])) for i in range(2)]
    seed = p[1]-p[0]  # first pose yaw is exactly 0 by this controlled fixture
    fit = namespace["fit_range_plane_loop"](a,b,seed,delta_z=case.tracks[0].sensor_z[1])
    if not fit["accepted"]:
        return dict(accepted=False, reason=fit["reason"], source_sha256=hashlib.sha256(source).hexdigest())
    lam, Q = np.linalg.eigh(fit["information"][:2,:2])
    root = (Q*np.sqrt(np.maximum(lam, 0)))@Q.T
    loop = np.asarray(fit["relative_pose2"][:2])-seed[:2], root
    labels = ["none","M0","M1","M2"]
    return dict(accepted=True, rank=fit["observable_rank"],
                source_sha256=hashlib.sha256(source).hexdigest(), seed=829,
                standalone={name:solve(case,name) for name in labels},
                with_loop={name:solve(case,name,loop=loop) for name in labels})


def run(count, verbose):
    observed_hash = hashlib.sha256((ROOT/"standalone/SideScanMappedRangeChange.py").read_bytes()).hexdigest()
    if observed_hash != M0_SHA256:
        raise RuntimeError("M0 differs from the audited frozen source")
    scenarios = []
    labels = ["none", "M0", "M0_plain", "M1", "M2"]
    seeds = list(range(13700, 13700+count))
    for field, value, config in scan_design():
        records = []
        executed = labels + (["M1_map_cov","M2_map_cov"] if field == "map_std" else [])
        for seed in seeds:
            s = synthesize(config, seed)
            # Alternate method order to reduce timing drift from CPU warmup.
            n = seed%len(executed)
            order = executed[n:]+executed[:n]
            records.append(dict(seed=seed, swapped_tracks=s.n_swapped,
                                gate=feasibility(s), models={m:solve(s,m) for m in order}))
        scenarios.append(dict(field=field, level=value, config=config.__dict__,
            metrics={m:summary(records,m) for m in executed},
            paired_bootstrap={m:paired_bootstrap(records,m,"M0") for m in ("M1","M2")},
            trials=records))
        if verbose:
            print(f"{len(scenarios):02d} {field}={value}", flush=True)
    return dict(schema="sonar_range_factor_v2_synthetic", seeds=seeds,
                M0_sha256=observed_hash, versions=dict(python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, system=platform.platform()),
                scenarios=scenarios, covariance_fixture=statistics_demo(), loop_fixture=loop_fixture())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--trials", type=int, default=24, help="paired independent scene seeds per level")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; refuses overwrite")
    args = parser.parse_args()
    if not 1 <= args.trials <= 240 or args.output.exists():
        parser.error("choose 1–240 trials and a new output path")
    evidence = run(args.trials, verbose=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
