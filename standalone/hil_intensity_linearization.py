"""Opt-in batched intensity Jacobians, guarded by the frozen source hash.

Keep the original two matrix products in the original association order. Keep
the yaw dot product and photometric/surface scalar equations per row, because
changing their reduction layout can change floating-point results. No sampling,
information, gate, sensor rate or state dimension is changed.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import textwrap

import numpy as np

EXPECTED_LINEARIZATION_SHA256 = "736a42e9e8e173109fb4335baca99bfae9185eb53946e1a8dd553b7b1c463780"


def _batch_rows(estimator, state, points_body, points_world, measured, groups, samples, yaw_axis):
    count = min(len(points_body), len(points_world), len(measured), len(groups), len(samples))
    selected = np.array([i for i, sample in enumerate(samples[:count]) if sample is not None], dtype=np.int64)
    if not selected.size:
        return (np.empty(0), np.empty(0), np.empty((0, 15)), np.empty((0, 3)), np.empty(0),
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    retained = [samples[i] for i in selected]
    gradients = np.array([[s.gradient_xy[0], s.gradient_xy[1], 0.] for s in retained], dtype=np.float64)
    points = points_body[selected]
    skew = np.zeros((len(selected), 3, 3), dtype=np.float64)
    skew[:, 0, 1], skew[:, 0, 2] = -points[:, 2], points[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = points[:, 2], -points[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -points[:, 1], points[:, 0]
    rotation_jacobians = (gradients[:, None, :] @ (-state.rotation @ skew))[:, 0, :]
    # Matrix-vector reduction is intentionally NOT replaced by one GEMV.
    yaw = np.array([float(row @ yaw_axis) for row in rotation_jacobians], dtype=np.float64)
    jacobians = np.zeros((len(selected), 15), dtype=np.float64)
    if estimator.config.intensity_planar_only:
        jacobians[:, :3] = yaw[:, None] * yaw_axis
        jacobians[:, 3:5] = gradients[:, :2]
    else:
        jacobians[:, :3] = rotation_jacobians
        jacobians[:, 3:6] = gradients
    variances = []
    for index, sample in zip(selected, retained):
        value = estimator.config.intensity_noise_std**2 + sample.variance
        if np.isfinite(sample.surface_z):
            scale_squared = (estimator.config.intensity_surface_consistency_std**2
                             + max(0., float(sample.surface_variance)))
            elevation_residual = float(points_world[index, 2] - sample.surface_z)
            weight = scale_squared / (scale_squared + elevation_residual**2)
            value /= max(weight, 1e-12)
        variances.append(value)
    return (np.asarray([float(measured[i]) for i in selected], dtype=np.float64),
            np.asarray([s.intensity for s in retained], dtype=np.float64), jacobians,
            np.column_stack((gradients[:, :2], yaw)), np.asarray(variances, dtype=np.float64),
            np.asarray([int(s.reference_observation_id) for s in retained], dtype=np.int64),
            np.asarray([int(s.anchor_id) for s in retained], dtype=np.int64),
            np.asarray([int(groups[i]) for i in selected], dtype=np.int64))


def build_batched_linearization(original):
    source = textwrap.dedent(inspect.getsource(original))
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_LINEARIZATION_SHA256:
        raise ValueError("unrecognized intensity linearization; refusing deployment transformation")
    tree = ast.parse(source)
    method = tree.body[0]
    indices = [i for i, node in enumerate(method.body) if isinstance(node, ast.For)
               and "point_body" in ast.unparse(node.target)]
    if len(indices) != 1:
        raise ValueError("ambiguous intensity measurement loop")
    method.body[indices[0]:] = ast.parse('''
return _hil_batch_rows(self, state, points_body, points_world, measured_intensity,
    input_group_ids, intensity_samples, world_yaw_axis_body)
''').body
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _hil_batch_rows=_batch_rows)
    exec(compile(tree, "<hil-batched-intensity-linearization>", "exec"), namespace)
    candidate = namespace[original.__name__]
    candidate.frozen_method_sha256 = digest
    return candidate


def install_batched_linearization(audit):
    from oceansim.slam.esikf import AcousticESIKF
    candidate = build_batched_linearization(AcousticESIKF._intensity_linearization)
    AcousticESIKF._intensity_linearization = candidate
    audit["intensity_linearization_candidate"] = dict(
        frozen_method_sha256=candidate.frozen_method_sha256,
        per_row_yaw_reduction_and_variance_equations_retained=True,
        paired_full_engine_bitwise_validation_required=True)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("action", choices=("replay", "profile"))
    p.add_argument("--canonical-correspondences", action="store_true")
    args, remaining = p.parse_known_args()
    if args.action == "replay":
        from standalone.hil import replay as runner
    else:
        from standalone import profile_hil_engine as runner
    from standalone.hil_intensity_packing import install_layout
    original_initialize = runner.initialize

    def initialize_candidate(*values, **kwargs):
        engine, audit = original_initialize(*values, **kwargs)
        install_layout(audit, "packed-and-feature-batch")
        install_batched_linearization(audit)
        if args.canonical_correspondences:
            from standalone.hil_stable_correspondences import install_canonical_order
            install_canonical_order(audit)
        return engine, audit

    runner.initialize = initialize_candidate
    sys.argv = [sys.argv[0], *remaining]
    runner.main()


if __name__ == "__main__":
    main()
