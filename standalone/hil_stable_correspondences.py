"""Explicit numerical-portability variant, NOT a bitwise frozen replay.

KLT still chooses exactly the original correspondences. Only the order of that
completed set entering uniform RANSAC is made spatially canonical. Thus the
fixed-seed finite sample realization changes; compare this variant separately
against the original PC/ARM trajectories. No gate, sensor or state is changed.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
import textwrap

EXPECTED_KLT_SHA256 = "2a14b8902b37f883204517859e7e5b4bbdc28c650c29547bdf08a86982f66911"


def canonical_pairs(pairs, previous, current):
    """Use pixel geometry, not detector index or floating confidence ranking."""
    def key(pair):
        current_index, previous_index = pair
        cx, cy = current.keypoint_pixels[current_index]
        px, py = previous.keypoint_pixels[previous_index]
        return float(cy), float(cx), float(py), float(px)
    return sorted(pairs, key=key)


def build_canonical_klt(original):
    digest = hashlib.sha256(textwrap.dedent(inspect.getsource(original)).encode()).hexdigest()
    if digest != EXPECTED_KLT_SHA256:
        raise ValueError("unrecognized KLT association; refusing canonical-order variant")

    def spatially_ordered(self, previous, current, predicted_transform):
        # Association, greedy mutual ownership and all quality decisions remain
        # in the original function. Never sort before it chooses the set.
        pairs = original(self, previous, current, predicted_transform)
        return canonical_pairs(pairs, previous, current)

    spatially_ordered.frozen_method_sha256 = digest
    return spatially_ordered


def install_canonical_order(audit):
    from oceansim.slam.feature_sonar_odometry import FeatureSonarOdometry
    method = build_canonical_klt(FeatureSonarOdometry._klt_pairs)
    FeatureSonarOdometry._klt_pairs = method
    audit["numerical_portability_variant"] = dict(
        name="spatially_canonical_completed_KLT_correspondence_order",
        original_klt_method_sha256=method.frozen_method_sha256,
        original_correspondence_set_and_quality_decisions_retained=True,
        uniform_RANSAC_sampling_distribution_retained=True,
        fixed_seed_sample_realization_changes=True,
        bitwise_equality_to_original_trajectory_expected=False,
        separate_PC_ARM_and_original_accuracy_comparisons_required=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("action", choices=("replay", "profile"))
    parser.add_argument("--layout", choices=("original", "packed", "packed-and-feature-batch"),
                        default="packed-and-feature-batch")
    args, remaining = parser.parse_known_args()
    if args.action == "replay":
        from standalone.hil import replay as runner
    else:
        from standalone import profile_hil_engine as runner
    from standalone.hil_intensity_packing import install_layout
    original_initialize = runner.initialize

    def initialize_variant(*values, **kwargs):
        engine, audit = original_initialize(*values, **kwargs)
        install_layout(audit, args.layout)
        install_canonical_order(audit)
        return engine, audit

    runner.initialize = initialize_variant
    sys.argv = [sys.argv[0], *remaining]
    runner.main()


if __name__ == "__main__":
    main()
