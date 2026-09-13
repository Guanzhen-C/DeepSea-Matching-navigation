#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/build_bbs3d_bridge.sh"

# Controlled full-route candidate for the operational TRN + Global-UTAN
# recovery path. This is an independent OceanSim implementation of published
# algorithmic ideas, not copied MB-System or unavailable Global-UTAN code. The sensor model,
# route, seed, exact initial state, local estimator and GTSAM backend remain
# identical to the fixed-lag control.  The candidate places persistent
# point-mass particle cohorts, an explicit null state, raw MBES/side-scan map
# likelihoods jointly normalized across every active hypothesis, and
# independent-window confirmation in front of both ordinary
# terrain factors and whole-map recovery.  While multiple spatial modes exist,
# the leading MAP cohort is the provisional, output-only map->odom target;
# active whole-map recovery cohorts take priority, single-mode ordinary output
# uses GTSAM, and graph insertion retains the stricter confirmation policy.  A
# deterministic basin-local raw-
# likelihood MAP/Laplace approximation supplies the current acquisition's
# rank-aware information to GTSAM even when the compact matcher has no
# proposal; the recursive PF posterior is never reused as a unary factor.
# A change-point branch gives newly retrieved map candidates probability mass
# from the complete prior, so an old confident alias cannot become immortal.
# Provenance-labelled persistent proposals cannot be overwritten by an
# unlabelled whole-map alias during association.  Ordinary
# terrain priors retain the baseline Huber loss; only a cohort-confirmed,
# re-seeded global relocalization uses the separate Cauchy policy.
exec "${SCRIPT_DIR}/run_backend_ab_gtsam_full_route_isaac.sh" \
    --enable-terrain-cohort-particle-backend \
    --enable-global-cohort-tracking \
    --enable-bbs3d-global-verification \
    --terrain-global-yaw-window-deg 180 \
    --terrain-global-maximum-cohorts 48 \
    --terrain-global-particles-per-cohort 256 \
    --terrain-global-particle-resample-ess-fraction 0.50 \
    --terrain-global-particle-kernel-bandwidth 0.15 \
    --terrain-global-minimum-independent-evidence 3 \
    --terrain-global-cohort-confirmation-probability 0.97 \
    --terrain-global-cohort-confirmation-odds 20 \
    --terrain-particle-online-output multimodal-map \
    --enable-map-to-odom-soft-correction \
    --metrics /tmp/oceansim_gtsam_global_utan_full_route_truth_ui.json \
    "$@"
