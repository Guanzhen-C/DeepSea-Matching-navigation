"""Opt-in, exact layout-only deployment candidate; never installed on import.

The frozen lake/sea source files remain unchanged. Only repeated neighbourhood
packing is replaced; candidate order, radius comparisons, plane fitting,
causal/heading/anchor rules, gates and estimator state updates are untouched.
Unknown source versions fail closed. This candidate needs paired full-engine
validation before use; unit equivalence alone is not navigation acceptance.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import textwrap

import numpy as np

EXPECTED_METHOD_SHA256 = "324a8f2c942152e562db815c2a0dcf557cb2d919aad1a6cf1c5b274e1e4d35b4"


def _pack_fixed_anchor(points, finite_indices, entries_by_base, base_inverse,
                       radius_squared, arrays):
    positions, intensities, variances, elevations, z_variances, counts, observation_ids, support = arrays
    if not finite_indices.size:
        return
    width = support.shape[1]
    base_xy = np.zeros((len(entries_by_base), width, 2), dtype=np.float64)
    # Match scalar-to-float conversions of the original destination arrays.
    base_values = np.zeros((len(entries_by_base), width, 5), dtype=np.float64)
    base_ids = np.zeros((len(entries_by_base), width), dtype=np.int64)
    occupied = np.zeros((len(entries_by_base), width), dtype=bool)
    for base, entries in enumerate(entries_by_base):
        if len(entries) > width:
            raise ValueError("unexpected fixed-anchor neighbourhood width")
        for slot, (_anchor_id, cell) in enumerate(entries):
            base_xy[base, slot] = cell.mean_xy
            base_values[base, slot] = (cell.mean_intensity, cell.variance, cell.mean_z,
                                      cell.z_variance, cell.count)
            base_ids[base, slot] = cell.last_observation_id
            occupied[base, slot] = True
    dx = base_xy[base_inverse, :, 0] - points[finite_indices, 0, None]
    dy = base_xy[base_inverse, :, 1] - points[finite_indices, 1, None]
    retained = occupied[base_inverse] & (dx * dx + dy * dy <= radius_squared)
    rows, columns = np.nonzero(retained)
    if not rows.size:
        return
    destination_rows = finite_indices[rows]
    # Stable compaction, exactly preserving the original per-point entry order.
    destination_columns = (np.cumsum(retained, axis=1) - 1)[rows, columns]
    bases = base_inverse[rows]
    positions[destination_rows, destination_columns] = base_xy[bases, columns]
    selected = base_values[bases, columns]
    for index, destination in enumerate((intensities, variances, elevations, z_variances, counts)):
        destination[destination_rows, destination_columns] = selected[:, index]
    observation_ids[destination_rows, destination_columns] = base_ids[bases, columns]
    support[destination_rows, destination_columns] = True


def build_packed_query(original):
    """Return an uninstalled candidate for a recognized frozen query method."""
    source = textwrap.dedent(inspect.getsource(original))
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_METHOD_SHA256:
        raise ValueError("unrecognized intensity query; refusing deployment transformation")
    tree = ast.parse(source)
    loops = [node for node in tree.body[0].body if isinstance(node, ast.For)
             and ast.unparse(node.target) == "(local_index, point_index)"]
    if len(loops) != 1:
        raise ValueError("ambiguous original neighbourhood packing loop")
    loop = loops[0]
    replacement = ast.parse('''
_hil_pack_fixed_anchor(points, finite_indices, entries_by_base, base_inverse,
    radius_squared, (positions, intensities, variances, elevations,
    elevation_variances, counts, observation_ids, support))
''').body
    offset = tree.body[0].body.index(loop)
    tree.body[0].body[offset:offset + 1] = replacement
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _hil_pack_fixed_anchor=_pack_fixed_anchor)
    exec(compile(tree, "<hil-layout-only-intensity-packing>", "exec"), namespace)
    candidate = namespace[original.__name__]
    candidate.frozen_method_sha256 = digest
    return candidate


def main():
    """Explicitly labelled experimental replay/profile, never the ROS receiver."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("action", choices=("replay", "profile"))
    parser.add_argument("--batch-feature-descriptors", action="store_true",
                        help="additional explicit candidate: batch patch mean/std, preserve decisions")
    args, remaining = parser.parse_known_args()
    if args.action == "replay":
        from standalone.hil import replay as runner
    else:
        from standalone import profile_hil_engine as runner
    original_initialize = runner.initialize

    def experimental_initialize(*values, **kwargs):
        engine, audit = original_initialize(*values, **kwargs)
        from oceansim.slam.intensity_map import AcousticIntensityMap
        candidate = build_packed_query(AcousticIntensityMap.query_fixed_anchor_batch)
        AcousticIntensityMap.query_fixed_anchor_batch = candidate
        audit["deployment_candidate"] = dict(name="layout_only_intensity_packing",
            frozen_method_sha256=candidate.frozen_method_sha256,
            altered_plane_fit_or_thresholds=False, paired_full_engine_validation_required=True)
        if args.batch_feature_descriptors:
            from standalone.hil_feature_batching import build_batched_canonical
            from oceansim.slam.feature_sonar_odometry import FeatureSonarOdometry
            canonical = build_batched_canonical(FeatureSonarOdometry._canonical_frame)
            FeatureSonarOdometry._canonical_frame = canonical
            audit["deployment_candidate"].update(name="layout_packing_and_descriptor_reductions",
                descriptor_method_sha256=canonical.frozen_method_sha256)
        return engine, audit

    runner.initialize = experimental_initialize
    sys.argv = [sys.argv[0], *remaining]
    runner.main()


if __name__ == "__main__":
    main()
