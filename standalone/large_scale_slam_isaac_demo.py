#!/usr/bin/env python3
"""Live Isaac Sim UI for the large-area acoustic-inertial SLAM system."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, replace
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live OceanSim FastSonar-LIVO2 demonstration using measured "
            "Monterey Canyon bathymetry and a full-scale LAUV"
        )
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="wall-clock seconds before closing; zero runs until the window closes",
    )
    parser.add_argument("--time-scale", type=float, default=8.0)
    parser.add_argument(
        "--ui-sonar-refresh-hz",
        type=float,
        default=0.0,
        help=(
            "display-only MBES/side-scan/status refresh limit in simulated "
            "Hz; zero presents the final ping before every rendered frame. "
            "All pings still enter the sensor and SLAM pipelines"
        ),
    )
    parser.add_argument(
        "--ping-hz",
        type=float,
        default=2.5,
        help=(
            "simulated ping rate; 2.5 Hz keeps the assumed 1500 m/s "
            "two-way travel time below the 250 m V2 range interval"
        ),
    )
    parser.add_argument(
        "--sonar-pings-in-water",
        type=int,
        default=1,
        help=(
            "number of concurrent, distinguishable coded/interleaved acoustic "
            "pings; one enforces the conventional single-ping travel-time limit"
        ),
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--initial-position-offset-x-m",
        type=float,
        default=0.0,
        help="deterministic east-axis error applied to the navigation launch seed",
    )
    parser.add_argument(
        "--initial-position-offset-y-m",
        type=float,
        default=0.0,
        help="deterministic north-axis error applied to the navigation launch seed",
    )
    parser.add_argument(
        "--initial-yaw-offset-deg",
        type=float,
        default=0.0,
        help="deterministic yaw error applied to the navigation launch seed",
    )
    parser.add_argument(
        "--initialize-from-truth",
        action="store_true",
        help=(
            "evaluation-only mode: initialize position, attitude, and velocity "
            "from the simulated t=0 truth instead of the sensor startup window"
        ),
    )
    parser.add_argument(
        "--survey-speed-mps",
        type=float,
        default=1.5,
        help=(
            "arc-length-parameterized AUV speed; zero retains the legacy "
            "360 s phase trajectory"
        ),
    )
    parser.add_argument(
        "--survey-laps",
        type=float,
        default=0,
        help=(
            "stop after this many simulated survey laps (fractions allowed); "
            "zero disables"
        ),
    )
    parser.add_argument(
        "--survey-start-distance-m",
        type=float,
        default=0.0,
        help=(
            "start a fresh estimator run at this arc distance along the "
            "periodic survey route; this changes only the truth-route phase "
            "and is not a serialized estimator checkpoint"
        ),
    )
    parser.add_argument(
        "--survey-segment-distance-m",
        type=float,
        default=0.0,
        help=(
            "stop after travelling this 3-D arc distance from the selected "
            "route start; zero leaves the existing lap/window termination"
        ),
    )
    parser.add_argument("--survey-center-x-m", type=float, default=0.0)
    parser.add_argument("--survey-center-y-m", type=float, default=0.0)
    parser.add_argument(
        "--survey-x-amplitude-m", type=float, default=320.0
    )
    parser.add_argument(
        "--survey-y-amplitude-m", type=float, default=220.0
    )
    parser.add_argument(
        "--survey-pattern-yaw-deg", type=float, default=0.0
    )
    parser.add_argument(
        "--survey-pattern",
        choices=("figure-eight", "full-map-route"),
        default="figure-eight",
        help=(
            "figure-eight uses the amplitude/centre arguments; full-map-route "
            "loads a closed measured-terrain waypoint mission"
        ),
    )
    parser.add_argument(
        "--survey-route",
        type=Path,
        default=(
            ROOT
            / "data/monterey_canyon_full_5m/full_map_traversal_route.json"
        ),
        help="JSON waypoint mission used by --survey-pattern full-map-route",
    )
    parser.add_argument("--survey-altitude-m", type=float, default=28.0)
    parser.add_argument(
        "--survey-maximum-pitch-deg",
        type=float,
        default=None,
        help=(
            "optional pitch limit for full-map terrain following; the "
            "vehicle climbs early and may fly higher than the minimum altitude"
        ),
    )
    parser.add_argument(
        "--sonar-profile",
        choices=("v2-standard", "v2-high-resolution"),
        default="v2-standard",
        help=(
            "operating mode transcribed from 多波束测深侧扫作业及成像V2; "
            "unknown hardware fields remain explicit simulation assumptions"
        ),
    )
    parser.add_argument(
        "--sonar-swath-width-deg",
        type=float,
        default=170.0,
        help=(
            "ray-model vertical fan used to expose the documented 250 m "
            "range at 28 m altitude; alpha is not specified by the V2 document"
        ),
    )
    parser.add_argument(
        "--slam-max-points",
        type=int,
        default=384,
        help="maximum motion-compensated MBES points in each ESIKF update",
    )
    parser.add_argument(
        "--local-sonar-noise-floor-m",
        type=float,
        default=0.30,
        help=(
            "continuous point-to-surfel model standard deviation used by the "
            "local acoustic ESIKF; this is a covariance weight, not an "
            "accept/reject threshold"
        ),
    )
    parser.add_argument(
        "--local-intensity-noise-std",
        type=float,
        default=0.09,
        help=(
            "continuous normalized-backscatter residual standard deviation "
            "used by the local direct-intensity ESIKF"
        ),
    )
    parser.add_argument(
        "--local-intensity-surface-consistency-std-m",
        type=float,
        default=0.75,
        help=(
            "Cauchy scale for disagreement between a direct side-scan "
            "sample and its co-registered local bathymetric surface; this "
            "continuously changes covariance and is not a rejection gate"
        ),
    )
    parser.add_argument(
        "--local-observability-mode",
        choices=("continuous", "hard", "legacy"),
        default="continuous",
        help=(
            "weak-direction handling for the local MBES ESIKF: continuous "
            "spectral weighting is the production mode; hard projection and "
            "the former whole-frame scalar weighting are reproducible ablations"
        ),
    )
    parser.add_argument(
        "--mbes-group-pings",
        type=int,
        default=1,
        help=(
            "integrated acquisitions per FAST-LIVO2-style geometry update; "
            "V2 already supplies eight simultaneous rows, so one is physical"
        ),
    )
    parser.add_argument(
        "--enable-mbes-sliding-window",
        action="store_true",
        help=(
            "treat --mbes-group-pings as a causal sliding local submap; "
            "after seeding, every physical ping produces one update while "
            "only the oldest aligned ping enters the frozen reference map"
        ),
    )
    parser.add_argument(
        "--sonar-beams",
        type=int,
        default=96,
        help=(
            "published bathymetry samples selected from the unified raw echo "
            "grid; must be even for symmetric port/starboard coverage"
        ),
    )
    parser.add_argument(
        "--sidescan-imaging-rays-per-side",
        type=int,
        default=1024,
        help=(
            "ray-model quadrature directions per side and per documented "
            "along-track beam; this is not the 2 cm output-bin count"
        ),
    )
    parser.add_argument(
        "--sidescan-slam-stride",
        type=int,
        default=8,
        help=(
            "deterministic decimation of correlated dense imaging XYZ samples "
            "before direct SLAM; raw/display products retain every ray"
        ),
    )
    parser.add_argument(
        "--association-map-half-extent-m",
        type=float,
        default=275.0,
        help=(
            "half extent of the non-destructive local surfel snapshot; zero "
            "packs the complete historical map"
        ),
    )
    parser.add_argument(
        "--slam-voxel-size-m",
        type=float,
        default=1.8,
        help="incremental bathymetric surfel-map voxel size",
    )
    parser.add_argument(
        "--intensity-voxel-size-m",
        type=float,
        default=3.0,
        help="direct side-scan intensity-map cell size",
    )
    parser.add_argument(
        "--map-insert-stride",
        type=int,
        default=1,
        help="deterministic grouped-MBES insertion stride; filtering keeps all rays",
    )
    parser.add_argument(
        "--intensity-reference-delay-pings",
        type=int,
        default=75,
        help=(
            "minimum causal separation before a side-scan strip becomes a "
            "FAST-LIVO2 intensity-map reference; one permits the immediately "
            "preceding acquisition but never the current acquisition"
        ),
    )
    parser.add_argument(
        "--intensity-group-pings",
        type=int,
        default=1,
        help=(
            "integrated acquisitions per direct-intensity update; use one for "
            "the V2 ping that already contains eight simultaneous rows"
        ),
    )
    parser.add_argument(
        "--intensity-min-group-span-m",
        type=float,
        default=1.0,
        help=(
            "minimum total strip aperture, including simultaneous intra-ping "
            "row coverage and any inter-ping vehicle motion"
        ),
    )
    parser.add_argument(
        "--intensity-submap-anchor-interval-s",
        type=float,
        default=0.0,
        help=(
            "minimum lifetime of one local side-scan intensity-submap "
            "anchor; zero follows every global pose-graph keyframe"
        ),
    )
    parser.add_argument(
        "--intensity-effective-independent-samples",
        type=int,
        default=96,
        help=(
            "maximum independent-equivalent samples carried by one direct "
            "side-scan strip update; dense samples sharing an acoustic "
            "footprint remain in the residual but do not multiply information"
        ),
    )
    parser.add_argument(
        "--intensity-min-singular-ratio",
        type=float,
        default=2.0e-3,
        help=(
            "minimum whitened direct-intensity singular-value ratio used "
            "to declare a planar x/y/yaw direction observable"
        ),
    )
    parser.add_argument(
        "--intensity-factor-min-interval-s",
        type=float,
        default=0.0,
        help=(
            "minimum cadence of accepted side-scan pose-graph factors; "
            "zero preserves unrestricted short-run behaviour"
        ),
    )
    parser.add_argument(
        "--intensity-factor-optimize-interval",
        type=int,
        default=10,
        help="optimize after this many accepted direct-intensity graph factors",
    )
    parser.add_argument(
        "--trajectory-sample-interval-s",
        type=float,
        default=0.0,
        help=(
            "stored trajectory cadence only; zero retains every IMU state "
            "and does not alter filter propagation"
        ),
    )
    parser.add_argument(
        "--enable-terrain-map-matching",
        action="store_true",
        help=(
            "align motion-compensated MBES patches to the measured prior "
            "bathymetry and add independent absolute factors to the global "
            "pose graph; the local acoustic ESIKF remains unchanged"
        ),
    )
    parser.add_argument(
        "--enable-backscatter-map-matching",
        action="store_true",
        help=(
            "refine accepted bathymetric absolute fixes against the measured "
            "side-scan backscatter mosaic after range/side trend removal"
        ),
    )
    parser.add_argument(
        "--backscatter-match-maximum-points",
        type=int,
        default=2048,
        help="maximum registered side-scan texture voxels per absolute match",
    )
    parser.add_argument(
        "--backscatter-match-yaw-window-deg",
        type=float,
        default=2.5,
        help=(
            "symmetric side-scan texture yaw-refinement bound around the "
            "accepted bathymetric fix"
        ),
    )
    parser.add_argument(
        "--backscatter-ci-bathymetry-weight",
        type=float,
        default=-1.0,
        help=(
            "bathymetry weight in covariance intersection with an accepted "
            "co-registered backscatter fix; the side-scan weight is one minus "
            "this value and no independence assumption is made; -1 selects "
            "the minimum-log-determinant CI weight separately for each patch"
        ),
    )
    parser.add_argument(
        "--cross-modal-large-correction-m",
        type=float,
        default=5.0,
        help=(
            "after global initialization, withhold a bathymetric correction "
            "larger than this when the co-registered backscatter likelihood "
            "is multi-modal; zero disables the compatibility veto"
        ),
    )
    parser.add_argument(
        "--backscatter-boundary-yaw-retry-window-deg",
        type=float,
        default=5.0,
        help=(
            "wider yaw bound used only when the primary side-scan optimum "
            "hits its yaw boundary while translation remains interior; zero "
            "disables the retry"
        ),
    )
    parser.add_argument(
        "--terrain-match-window-pings",
        type=int,
        default=30,
        help="rolling MBES pings in each prior-map terrain-matching patch",
    )
    parser.add_argument(
        "--terrain-patch-motion-source",
        choices=("acoustic-fixed-lag", "acoustic-predictive", "inertial"),
        default="acoustic-predictive",
        help=(
            "trajectory used to deskew terrain/backscatter patches; "
            "acoustic-fixed-lag smoothly distributes each causal acoustic "
            "endpoint correction over IMU/DVL relative motion, while "
            "acoustic-predictive snapshots the fused state before the current "
            "ping update, while inertial uses IMU/DVL/depth dead reckoning"
        ),
    )
    parser.add_argument(
        "--terrain-match-maximum-points",
        type=int,
        default=768,
        help=(
            "deterministic point budget retained from each motion-compensated "
            "MBES patch; statistical information remains correlation-capped"
        ),
    )
    parser.add_argument(
        "--terrain-patch-minimum-voxel-size-m",
        type=float,
        default=0.0,
        help=(
            "minimum horizontal averaging scale for the terrain observation; "
            "zero uses only the adaptive point-budget scale"
        ),
    )
    parser.add_argument(
        "--terrain-require-yaw-observable",
        action="store_true",
        help=(
            "require a rank-three primary bathymetric solution; after all "
            "other safety gates, a yaw-unobservable solution may still be "
            "downgraded to an x/y-only prior"
        ),
    )
    parser.add_argument(
        "--terrain-initialize-global-gauge",
        action="store_true",
        help=(
            "use the first accepted rank-three absolute terrain match to "
            "rigidly initialize the otherwise arbitrary planar pose-graph "
            "gauge; later ordinary terrain factors remain robust"
        ),
    )
    parser.add_argument(
        "--terrain-gauge-confirmation-patches",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "accepted disjoint terrain patches used before initializing the "
            "global gauge; two confirms the common basin and averages the "
            "transported terrain-yaw estimates instead of trusting one strip"
        ),
    )
    parser.add_argument(
        "--terrain-confirm-every-prior",
        action="store_true",
        help=(
            "validate every ordinary terrain prior against a preceding "
            "temporally disjoint fix propagated by local acoustic odometry, "
            "then average the two current-time pose estimates"
        ),
    )
    parser.add_argument(
        "--terrain-gauge-minimum-baseline-m",
        type=float,
        default=60.0,
        help=(
            "minimum seed and matched-position separation for two-patch "
            "global-gauge initialization"
        ),
    )
    parser.add_argument(
        "--terrain-gauge-maximum-position-residual-m",
        type=float,
        default=5.0,
        help=(
            "maximum horizontal disagreement between the first propagated "
            "map fix and the second independent map fix"
        ),
    )
    parser.add_argument(
        "--terrain-gauge-maximum-yaw-residual-deg",
        type=float,
        default=5.0,
        help=(
            "maximum yaw disagreement between a prior terrain fix propagated "
            "by local acoustic odometry and the current independent map fix"
        ),
    )
    parser.add_argument(
        "--terrain-position-only-priors",
        action="store_true",
        help=(
            "marginalize terrain-match yaw before graph fusion; successive "
            "absolute x/y fixes still constrain graph heading through motion"
        ),
    )
    parser.add_argument(
        "--terrain-prior-horizontal-std-floor-m",
        type=float,
        default=0.0,
        help=(
            "horizontal one-sigma model-error floor added in quadrature to "
            "terrain-prior covariance; zero disables the floor"
        ),
    )
    parser.add_argument(
        "--terrain-prior-yaw-std-floor-deg",
        type=float,
        default=0.0,
        help=(
            "yaw one-sigma map/model-error floor added in quadrature to "
            "terrain-prior covariance; zero disables the floor"
        ),
    )
    parser.add_argument(
        "--terrain-prior-surface",
        choices=("low-pass", "measured"),
        default="low-pass",
        help=(
            "low-pass uses the separately smoothed survey surface as a "
            "non-identical prior; measured is an optimistic same-DEM upper bound"
        ),
    )
    parser.add_argument(
        "--terrain-match-translation-window-m",
        type=float,
        default=30.0,
        help="symmetric x/y coarse-search bound around the global graph seed",
    )
    parser.add_argument(
        "--terrain-recovery-translation-window-m",
        type=float,
        default=0.0,
        help=(
            "strict-rank-three recovery x/y bound after consecutive failed "
            "terrain-prior observations; zero disables recovery"
        ),
    )
    parser.add_argument(
        "--terrain-recovery-after-rejections",
        type=int,
        default=3,
        help=(
            "consecutive observations without an inserted terrain prior "
            "before invoking the wide-window matcher"
        ),
    )
    parser.add_argument(
        "--terrain-emergency-recovery-translation-window-m",
        type=float,
        default=0.0,
        help=(
            "truth-free, two-patch-confirmed emergency x/y bound after the "
            "normal and recovery matchers both fail; zero disables this "
            "final layer and values above 150 m are forbidden"
        ),
    )
    parser.add_argument(
        "--enable-global-terrain-relocalization",
        action="store_true",
        help=(
            "after every bounded terrain-recovery layer fails, correlate a "
            "long motion-compensated MBES/side-scan strip against the complete "
            "measured map; the proposal still needs local refinement and a "
            "second temporally disjoint confirmation before graph fusion"
        ),
    )
    parser.add_argument(
        "--terrain-global-window-pings",
        type=int,
        default=1200,
        help="pings retained for rare whole-map relocalization proposals",
    )
    parser.add_argument(
        "--terrain-global-pyramid-stride",
        type=int,
        default=4,
        help="source-map cells per whole-map proposal-lattice cell",
    )
    parser.add_argument(
        "--terrain-global-yaw-window-deg",
        type=float,
        default=45.0,
        help="whole-map proposal yaw half-window around IMU/DVL dead reckoning",
    )
    parser.add_argument(
        "--terrain-global-yaw-step-deg",
        type=float,
        default=5.0,
        help="whole-map proposal yaw sampling interval",
    )
    parser.add_argument(
        "--terrain-global-minimum-correlation",
        type=float,
        default=0.30,
        help="minimum joint bathymetry/backscatter proposal correlation",
    )
    parser.add_argument(
        "--terrain-global-maximum-bathymetry-rmse-m",
        type=float,
        default=6.0,
        help=(
            "maximum absolute depth residual for a whole-map proposal; "
            "pressure depth and the registered map share the vertical datum"
        ),
    )
    parser.add_argument(
        "--terrain-global-bathymetry-absolute-weight",
        type=float,
        default=0.50,
        help=(
            "fraction of the bathymetry proposal score assigned to absolute "
            "pressure-depth consistency rather than zero-mean relief shape"
        ),
    )
    parser.add_argument(
        "--terrain-global-minimum-score-gap",
        type=float,
        default=0.08,
        help="minimum best-versus-second spatial-basin proposal score gap",
    )
    parser.add_argument(
        "--enable-global-cohort-tracking",
        action="store_true",
        help=(
            "retain every separated whole-map candidate across sonar epochs "
            "with an explicit null hypothesis; only a dominant cohort "
            "supported by independent ping windows may enter local refinement"
        ),
    )
    parser.add_argument(
        "--enable-terrain-cohort-particle-backend",
        action="store_true",
        help=(
            "filter ordinary bathymetry/side-scan absolute matches through a "
            "joint point-mass/cohort terrain-navigation filter before GTSAM "
            "insertion"
        ),
    )
    parser.add_argument(
        "--terrain-particle-online-output",
        choices=("graph", "multimodal-map", "always-map"),
        default="multimodal-map",
        help=(
            "published online map-pose source when the ordinary terrain "
            "particle backend is enabled: graph only, particle MAP only "
            "while multiple ordinary modes or active whole-map recovery "
            "modes exist, or ordinary particle MAP whenever a spatial "
            "hypothesis beats the explicit null state"
        ),
    )
    parser.add_argument(
        "--terrain-global-maximum-cohorts",
        type=int,
        default=48,
        help="maximum persistent whole-map pose hypotheses",
    )
    parser.add_argument(
        "--terrain-global-particles-per-cohort",
        type=int,
        default=256,
        help="SE(2) particles maintained inside each global pose cohort",
    )
    parser.add_argument(
        "--terrain-global-particle-resample-ess-fraction",
        type=float,
        default=0.50,
        help="systematic-resampling trigger as a fraction of cohort particles",
    )
    parser.add_argument(
        "--terrain-global-particle-kernel-bandwidth",
        type=float,
        default=0.15,
        help="Liu-West regularization bandwidth applied after resampling",
    )
    parser.add_argument(
        "--terrain-global-minimum-independent-evidence",
        type=int,
        default=3,
        help=(
            "nonoverlapping sonar acquisition windows required before a "
            "global cohort may be confirmed"
        ),
    )
    parser.add_argument(
        "--terrain-global-cohort-confirmation-probability",
        type=float,
        default=0.97,
        help="minimum normalized posterior probability of the leading cohort",
    )
    parser.add_argument(
        "--terrain-global-cohort-confirmation-odds",
        type=float,
        default=20.0,
        help="minimum leading-cohort odds over the runner-up or null state",
    )
    parser.add_argument(
        "--enable-bbs3d-global-verification",
        action="store_true",
        help=(
            "verify every persistent whole-map candidate with the pinned "
            "official 3D-BBS GPU implementation while holding pressure "
            "depth and IMU roll/pitch fixed"
        ),
    )
    parser.add_argument(
        "--bbs3d-global-translation-window-m",
        type=float,
        default=30.0,
        help="residual x/y half-window for 3D-BBS candidate verification",
    )
    parser.add_argument(
        "--bbs3d-global-yaw-window-deg",
        type=float,
        default=6.0,
        help="residual yaw half-window for 3D-BBS candidate verification",
    )
    parser.add_argument(
        "--bbs3d-global-minimum-score-fraction",
        type=float,
        default=0.25,
        help="minimum fraction of MBES source points occupying map voxels",
    )
    parser.add_argument(
        "--bbs3d-global-timeout-ms",
        type=int,
        default=5000,
        help="per-candidate 3D-BBS search timeout; zero disables the timeout",
    )
    parser.add_argument(
        "--terrain-global-confirmation-position-tolerance-m",
        type=float,
        default=25.0,
        help=(
            "maximum disagreement between two disjoint whole-map strip "
            "matches after IMU/DVL propagation"
        ),
    )
    parser.add_argument(
        "--terrain-global-confirmation-yaw-tolerance-deg",
        type=float,
        default=5.0,
        help=(
            "maximum yaw disagreement between two disjoint whole-map strip "
            "matches after IMU/DVL propagation"
        ),
    )
    parser.add_argument(
        "--terrain-match-yaw-window-deg",
        type=float,
        default=8.0,
        help="symmetric yaw coarse-search bound around the global graph seed",
    )
    parser.add_argument(
        "--terrain-prior-optimize-interval",
        type=int,
        default=10,
        help="accepted terrain priors between online pose-graph optimizations",
    )
    parser.add_argument(
        "--terrain-prior-robust-kernel",
        choices=("huber", "cauchy"),
        default="huber",
        help=(
            "robust loss for ordinary absolute terrain factors; Cauchy gives "
            "bounded influence to isolated perceptual aliases"
        ),
    )
    parser.add_argument(
        "--disable-intensity-updates",
        action="store_true",
        help=(
            "build the acoustic appearance map but disable its ESIKF state "
            "update for a geometry-only ablation"
        ),
    )
    parser.add_argument(
        "--sonar-backend",
        choices=("auto", "warp-cuda", "warp-cpu", "physx"),
        default="auto",
        help=(
            "acoustic ray backend; auto selects Warp CUDA when available "
            "and otherwise retains the PhysX CPU-query implementation"
        ),
    )
    parser.add_argument(
        "--verify-sonar-backend",
        action="store_true",
        help="compare Warp ray ranges with the exact PhysX collision mesh at startup",
    )
    parser.add_argument(
        "--slam-association-backend",
        choices=("auto", "warp-cuda", "warp-cpu", "cpu"),
        default="auto",
        help=(
            "point-to-surfel data association backend; auto uses exact Warp "
            "CUDA lookup for batches >=256 when available and otherwise uses "
            "the Python voxel map"
        ),
    )
    parser.add_argument(
        "--bathymetry",
        type=Path,
        default=ROOT / "data/monterey_canyon_full_5m",
        help=(
            "prepared complete-map directory or legacy NPZ; the default is "
            "the complete 29.2 x 17.7 km USGS Monterey Canyon product"
        ),
    )
    parser.add_argument(
        "--terrain-display-stride",
        type=int,
        default=8,
        help=(
            "whole-map display LOD in native 5 m cells; all source arrays "
            "remain loaded and NoData holes are never capped"
        ),
    )
    parser.add_argument(
        "--sonar-terrain-stride",
        type=int,
        default=2,
        help=(
            "whole-map acoustic mesh LOD in native 5 m cells; use 1 for "
            "all native triangles when memory permits"
        ),
    )
    parser.add_argument(
        "--camera-mode",
        choices=("chase", "survey-overview", "overview"),
        default="chase",
    )
    parser.add_argument("--keyframe-distance-m", type=float, default=25.0)
    parser.add_argument(
        "--keyframe-max-interval-s", type=float, default=20.0
    )
    parser.add_argument(
        "--pose-graph-odometry-reference-interval-s",
        type=float,
        default=60.0,
        help=(
            "physical duration over which pose-graph odometry standard "
            "deviations are defined; decouples random-walk information from "
            "keyframe cadence"
        ),
    )
    parser.add_argument(
        "--pose-graph-odometry-xy-std-m",
        type=float,
        default=0.32,
        help=(
            "horizontal standard deviation of a pose-graph odometry edge "
            "over the configured physical reference interval"
        ),
    )
    parser.add_argument(
        "--pose-graph-odometry-yaw-std-deg",
        type=float,
        default=0.8,
        help=(
            "yaw standard deviation of a pose-graph odometry edge over the "
            "configured physical reference interval"
        ),
    )
    parser.add_argument(
        "--pose-graph-backend",
        choices=("numpy-batch", "gtsam-fixed-lag-isam2"),
        default="numpy-batch",
        help=(
            "planar pose-graph solver; selecting GTSAM changes only the "
            "optimizer and preserves front-end states, factors, and matchers"
        ),
    )
    parser.add_argument(
        "--pose-graph-fixed-lag-s",
        type=float,
        default=180.0,
        help="GTSAM IncrementalFixedLagSmoother window duration",
    )
    parser.add_argument(
        "--pose-graph-isam-relinearize-threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--pose-graph-isam-relinearize-skip",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--enable-map-to-odom-soft-correction",
        "--enable-map-to-odom-soft-feedback",
        dest="enable_map_to_odom_soft_correction",
        action="store_true",
        help=(
            "continuously follow pose-graph corrections in the published "
            "map->odom transform; the local ESIKF and matcher seed remain "
            "unchanged"
        ),
    )
    parser.add_argument(
        "--map-to-odom-translation-time-constant-s",
        type=float,
        default=5.0,
        help="first-order time constant for published horizontal correction",
    )
    parser.add_argument(
        "--map-to-odom-yaw-time-constant-s",
        type=float,
        default=5.0,
        help="first-order time constant for published yaw correction",
    )
    parser.add_argument(
        "--map-to-odom-maximum-translation-rate-mps",
        type=float,
        default=5.0,
        help="maximum map-frame correction speed at the current AUV pose",
    )
    parser.add_argument(
        "--map-to-odom-maximum-yaw-rate-deg-s",
        type=float,
        default=5.0,
        help="maximum published map-frame yaw correction rate",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--progress-distance-m",
        type=float,
        default=2000.0,
        help="print a compact long-run checkpoint at this travelled distance",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "demo/large_scale_slam/isaac_live_metrics.json",
    )
    args, kit_args = parser.parse_known_args()
    if (
        args.duration < 0.0
        or not all(
            math.isfinite(value)
            for value in (
                args.initial_position_offset_x_m,
                args.initial_position_offset_y_m,
                args.initial_yaw_offset_deg,
            )
        )
        or args.time_scale <= 0.0
        or not math.isfinite(args.ui_sonar_refresh_hz)
        or args.ui_sonar_refresh_hz < 0.0
        or args.ping_hz <= 0.0
        or args.sonar_pings_in_water < 1
        or args.slam_max_points < 32
        or not math.isfinite(args.local_sonar_noise_floor_m)
        or args.local_sonar_noise_floor_m <= 0.0
        or not math.isfinite(args.local_intensity_noise_std)
        or args.local_intensity_noise_std <= 0.0
        or not math.isfinite(
            args.local_intensity_surface_consistency_std_m
        )
        or args.local_intensity_surface_consistency_std_m <= 0.0
        or args.mbes_group_pings < 1
        or args.association_map_half_extent_m < 0.0
        or 0.0 < args.association_map_half_extent_m < 150.0
        or args.slam_voxel_size_m <= 0.0
        or args.intensity_voxel_size_m <= 0.0
        or args.map_insert_stride < 1
        or args.survey_speed_mps < 0.0
        or args.survey_laps < 0
        or not math.isfinite(args.survey_start_distance_m)
        or args.survey_start_distance_m < 0.0
        or not math.isfinite(args.survey_segment_distance_m)
        or args.survey_segment_distance_m < 0.0
        or args.survey_x_amplitude_m <= 0.0
        or args.survey_y_amplitude_m <= 0.0
        or args.survey_altitude_m <= 0.0
        or (
            args.survey_maximum_pitch_deg is not None
            and not 0.0 <= args.survey_maximum_pitch_deg < 90.0
        )
        or not 0.0 < args.sonar_swath_width_deg < 180.0
        or args.sonar_beams < 32
        or args.sonar_beams % 2 != 0
        or args.sidescan_imaging_rays_per_side < 16
        or args.sidescan_slam_stride < 1
        or args.intensity_reference_delay_pings < 1
        or args.intensity_group_pings < 1
        or args.intensity_effective_independent_samples < 1
        or not 0.0 < args.intensity_min_singular_ratio < 1.0
        or not math.isfinite(args.intensity_min_group_span_m)
        or args.intensity_min_group_span_m < 0.0
        or not math.isfinite(args.intensity_submap_anchor_interval_s)
        or args.intensity_submap_anchor_interval_s < 0.0
        or args.intensity_factor_min_interval_s < 0.0
        or args.intensity_factor_optimize_interval < 1
        or args.trajectory_sample_interval_s < 0.0
        or args.terrain_match_window_pings < 2
        or args.terrain_match_maximum_points < 30
        or args.backscatter_match_maximum_points < 120
        or not math.isfinite(args.backscatter_match_yaw_window_deg)
        or args.backscatter_match_yaw_window_deg <= 0.5
        or args.backscatter_match_yaw_window_deg >= 15.0
        or (
            args.backscatter_ci_bathymetry_weight != -1.0
            and not 0.0 <= args.backscatter_ci_bathymetry_weight <= 1.0
        )
        or not math.isfinite(args.cross_modal_large_correction_m)
        or args.cross_modal_large_correction_m < 0.0
        or not math.isfinite(
            args.backscatter_boundary_yaw_retry_window_deg
        )
        or args.backscatter_boundary_yaw_retry_window_deg < 0.0
        or args.backscatter_boundary_yaw_retry_window_deg >= 15.0
        or (
            args.backscatter_boundary_yaw_retry_window_deg > 0.0
            and args.backscatter_boundary_yaw_retry_window_deg
            <= args.backscatter_match_yaw_window_deg
        )
        or not math.isfinite(args.terrain_patch_minimum_voxel_size_m)
        or args.terrain_patch_minimum_voxel_size_m < 0.0
        or not math.isfinite(args.terrain_gauge_minimum_baseline_m)
        or args.terrain_gauge_minimum_baseline_m <= 0.0
        or not math.isfinite(
            args.terrain_gauge_maximum_position_residual_m
        )
        or args.terrain_gauge_maximum_position_residual_m <= 0.0
        or not math.isfinite(args.terrain_gauge_maximum_yaw_residual_deg)
        or args.terrain_gauge_maximum_yaw_residual_deg <= 0.0
        or args.terrain_gauge_maximum_yaw_residual_deg >= 45.0
        or not math.isfinite(args.terrain_prior_horizontal_std_floor_m)
        or args.terrain_prior_horizontal_std_floor_m < 0.0
        or not math.isfinite(args.terrain_prior_yaw_std_floor_deg)
        or args.terrain_prior_yaw_std_floor_deg < 0.0
        or args.terrain_match_translation_window_m <= 0.0
        or not math.isfinite(args.terrain_recovery_translation_window_m)
        or args.terrain_recovery_translation_window_m < 0.0
        or args.terrain_recovery_after_rejections < 1
        or not math.isfinite(
            args.terrain_emergency_recovery_translation_window_m
        )
        or args.terrain_emergency_recovery_translation_window_m < 0.0
        or args.terrain_emergency_recovery_translation_window_m > 150.0
        or args.terrain_global_window_pings < 2
        or args.terrain_global_pyramid_stride < 1
        or not math.isfinite(args.terrain_global_yaw_window_deg)
        or args.terrain_global_yaw_window_deg <= 0.0
        or args.terrain_global_yaw_window_deg > 180.0
        or not math.isfinite(args.terrain_global_yaw_step_deg)
        or args.terrain_global_yaw_step_deg <= 0.0
        or args.terrain_global_yaw_step_deg
        > 2.0 * args.terrain_global_yaw_window_deg
        or not math.isfinite(args.terrain_global_minimum_correlation)
        or not 0.0 < args.terrain_global_minimum_correlation <= 1.0
        or not math.isfinite(
            args.terrain_global_maximum_bathymetry_rmse_m
        )
        or args.terrain_global_maximum_bathymetry_rmse_m <= 0.0
        or not math.isfinite(
            args.terrain_global_bathymetry_absolute_weight
        )
        or not 0.0
        <= args.terrain_global_bathymetry_absolute_weight
        <= 1.0
        or not math.isfinite(args.terrain_global_minimum_score_gap)
        or not 0.0 < args.terrain_global_minimum_score_gap <= 2.0
        or args.terrain_global_maximum_cohorts < 1
        or args.terrain_global_particles_per_cohort < 1
        or not math.isfinite(
            args.terrain_global_particle_resample_ess_fraction
        )
        or not 0.0
        < args.terrain_global_particle_resample_ess_fraction
        < 1.0
        or not math.isfinite(
            args.terrain_global_particle_kernel_bandwidth
        )
        or not 0.0 < args.terrain_global_particle_kernel_bandwidth < 1.0
        or args.terrain_global_minimum_independent_evidence < 1
        or not math.isfinite(
            args.terrain_global_cohort_confirmation_probability
        )
        or not 0.0
        < args.terrain_global_cohort_confirmation_probability
        < 1.0
        or not math.isfinite(args.terrain_global_cohort_confirmation_odds)
        or args.terrain_global_cohort_confirmation_odds <= 1.0
        or not math.isfinite(args.bbs3d_global_translation_window_m)
        or args.bbs3d_global_translation_window_m <= 0.0
        or not math.isfinite(args.bbs3d_global_yaw_window_deg)
        or args.bbs3d_global_yaw_window_deg <= 0.0
        or args.bbs3d_global_yaw_window_deg >= 45.0
        or not math.isfinite(args.bbs3d_global_minimum_score_fraction)
        or not 0.0 < args.bbs3d_global_minimum_score_fraction < 1.0
        or args.bbs3d_global_timeout_ms < 0
        or not math.isfinite(
            args.terrain_global_confirmation_position_tolerance_m
        )
        or args.terrain_global_confirmation_position_tolerance_m <= 0.0
        or not math.isfinite(
            args.terrain_global_confirmation_yaw_tolerance_deg
        )
        or args.terrain_global_confirmation_yaw_tolerance_deg <= 0.0
        or args.terrain_global_confirmation_yaw_tolerance_deg >= 45.0
        or args.terrain_match_yaw_window_deg <= 0.0
        or args.terrain_match_yaw_window_deg >= 45.0
        or args.terrain_prior_optimize_interval < 1
        or args.keyframe_distance_m <= 0.0
        or args.keyframe_max_interval_s <= 0.0
        or not math.isfinite(
            args.pose_graph_odometry_reference_interval_s
        )
        or args.pose_graph_odometry_reference_interval_s <= 0.0
        or not math.isfinite(args.pose_graph_odometry_xy_std_m)
        or args.pose_graph_odometry_xy_std_m <= 0.0
        or not math.isfinite(args.pose_graph_odometry_yaw_std_deg)
        or args.pose_graph_odometry_yaw_std_deg <= 0.0
        or not math.isfinite(args.pose_graph_fixed_lag_s)
        or args.pose_graph_fixed_lag_s <= 0.0
        or not math.isfinite(
            args.pose_graph_isam_relinearize_threshold
        )
        or args.pose_graph_isam_relinearize_threshold <= 0.0
        or args.pose_graph_isam_relinearize_skip < 1
        or not math.isfinite(
            args.map_to_odom_translation_time_constant_s
        )
        or args.map_to_odom_translation_time_constant_s <= 0.0
        or not math.isfinite(args.map_to_odom_yaw_time_constant_s)
        or args.map_to_odom_yaw_time_constant_s <= 0.0
        or not math.isfinite(
            args.map_to_odom_maximum_translation_rate_mps
        )
        or args.map_to_odom_maximum_translation_rate_mps <= 0.0
        or not math.isfinite(
            args.map_to_odom_maximum_yaw_rate_deg_s
        )
        or args.map_to_odom_maximum_yaw_rate_deg_s <= 0.0
        or args.terrain_display_stride < 1
        or args.sonar_terrain_stride < 1
        or args.progress_distance_m <= 0.0
    ):
        parser.error(
            "duration must be non-negative; rates must be positive; "
            "--slam-max-points must be >= 32; ping groups must be >= 1; "
            "--sonar-beams must be an even integer >= 32; "
            "--sonar-pings-in-water must be >= 1; "
            "--sidescan-imaging-rays-per-side must be >= 16; "
            "survey geometry, sonar swath, SLAM stride, and keyframe cadence "
            "must be positive; survey speed/laps must be non-negative; "
            "association half extent "
            "must be zero or at least 150 m; map/terrain strides and voxel "
            "sizes must be positive; progress distance must be positive"
        )
    if args.survey_pattern == "full-map-route" and not args.survey_route.is_file():
        parser.error(f"survey route does not exist: {args.survey_route}")
    if args.initialize_from_truth and any(
        abs(value) > 1.0e-12
        for value in (
            args.initial_position_offset_x_m,
            args.initial_position_offset_y_m,
            args.initial_yaw_offset_deg,
        )
    ):
        parser.error(
            "--initialize-from-truth requires all deterministic initial "
            "position/yaw offsets to be zero"
        )
    if args.survey_pattern == "full-map-route" and args.survey_speed_mps <= 0.0:
        parser.error("--survey-speed-mps must be positive for full-map-route")
    if (
        (args.survey_start_distance_m > 0.0
         or args.survey_segment_distance_m > 0.0)
        and args.survey_speed_mps <= 0.0
    ):
        parser.error(
            "survey start/segment distances require positive "
            "--survey-speed-mps"
        )
    if args.terrain_recovery_translation_window_m > 0.0:
        if (
            args.terrain_recovery_translation_window_m
            <= args.terrain_match_translation_window_m
        ):
            parser.error(
                "--terrain-recovery-translation-window-m must be wider than "
                "--terrain-match-translation-window-m"
            )
        if not args.terrain_require_yaw_observable:
            parser.error(
                "terrain recovery requires --terrain-require-yaw-observable "
                "so every wide-window recovery remains rank three"
            )
    if args.terrain_initialize_global_gauge:
        if not args.enable_terrain_map_matching:
            parser.error(
                "--terrain-initialize-global-gauge requires "
                "--enable-terrain-map-matching"
            )
        if not args.terrain_require_yaw_observable:
            parser.error(
                "--terrain-initialize-global-gauge requires "
                "--terrain-require-yaw-observable so the fixed planar gauge "
                "is never initialized from a rank-two match"
            )
    if (
        args.enable_backscatter_map_matching
        and not args.enable_terrain_map_matching
    ):
        parser.error(
            "--enable-backscatter-map-matching requires "
            "--enable-terrain-map-matching"
        )
    if (
        args.enable_terrain_cohort_particle_backend
        and not args.enable_terrain_map_matching
    ):
        parser.error(
            "--enable-terrain-cohort-particle-backend requires "
            "--enable-terrain-map-matching"
        )
    if args.terrain_confirm_every_prior:
        if not args.enable_terrain_map_matching:
            parser.error(
                "--terrain-confirm-every-prior requires "
                "--enable-terrain-map-matching"
            )
        if args.terrain_gauge_confirmation_patches != 2:
            parser.error(
                "--terrain-confirm-every-prior requires "
                "--terrain-gauge-confirmation-patches 2"
            )
    if args.terrain_emergency_recovery_translation_window_m > 0.0:
        if args.terrain_recovery_translation_window_m <= 0.0:
            parser.error(
                "emergency terrain recovery requires the preceding "
                "--terrain-recovery-translation-window-m layer"
            )
        if (
            args.terrain_emergency_recovery_translation_window_m
            <= args.terrain_recovery_translation_window_m
        ):
            parser.error(
                "--terrain-emergency-recovery-translation-window-m must be "
                "wider than --terrain-recovery-translation-window-m"
            )
        if not args.terrain_require_yaw_observable:
            parser.error(
                "emergency terrain recovery requires "
                "--terrain-require-yaw-observable"
            )
    if args.enable_global_terrain_relocalization:
        if args.terrain_emergency_recovery_translation_window_m <= 0.0:
            parser.error(
                "global terrain relocalization requires all preceding bounded "
                "terrain-recovery layers"
            )
        if not args.enable_backscatter_map_matching:
            parser.error(
                "global terrain relocalization requires co-registered "
                "backscatter map matching"
            )
        if args.terrain_global_window_pings <= args.terrain_match_window_pings:
            parser.error(
                "--terrain-global-window-pings must exceed the ordinary "
                "terrain-matching window"
            )
    if (
        args.enable_global_cohort_tracking
        and not args.enable_global_terrain_relocalization
    ):
        parser.error(
            "--enable-global-cohort-tracking requires "
            "--enable-global-terrain-relocalization"
        )
    if (
        args.enable_bbs3d_global_verification
        and not args.enable_global_cohort_tracking
    ):
        parser.error(
            "--enable-bbs3d-global-verification requires "
            "--enable-global-cohort-tracking"
        )
    sys.argv = [sys.argv[0], *kit_args]
    return args


ARGS = parse_args()
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# Seed both process-global generators before Isaac Sim or any OceanSim sensor
# objects are constructed.  The sensor models also receive their own explicit
# seeds below, so seeded replay does not depend on incidental generator use by
# the UI or another module.
random.seed(ARGS.seed)
np.random.seed(ARGS.seed)

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RayTracedLighting",
        "width": 1440,
        "height": 900,
        "window_width": 1720,
        "window_height": 1080,
        "anti_aliasing": 0,
        "multi_gpu": False,
        "sync_loads": True,
    }
)

import isaacsim


local_package = str(ROOT / "isaacsim")
if local_package not in isaacsim.__path__:
    isaacsim.__path__.append(local_package)

import carb
import omni.usd
import trimesh
from isaacsim.core.api import World
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

from isaacsim.oceansim.sensors.BathymetricSideScanSonarSensor import (
    BathymetricSideScanSonarSensor,
)
from isaacsim.oceansim.sensors.bathymetric_sidescan_profiles import (
    acoustic_two_way_travel_time_s,
    get_v2_profile,
    maximum_supported_ping_rate_hz,
    required_distinguishable_pings_in_water,
)
from isaacsim.oceansim.slam import (
    AcousticESIKF,
    AcousticTerrainParticleLikelihood,
    BackscatterMapMatcher,
    BackscatterMatcherConfig,
    BBS3DCandidateVerifier,
    BBS3DVerifierConfig,
    BathymetricTerrainMatcher,
    ESIKFConfig,
    FastSonarSLAM,
    FastSonarSLAMConfig,
    GlobalPoseEvidence,
    GlobalTerrainRelocalizer,
    GlobalTerrainRelocalizerConfig,
    InertialState,
    NavigationInitialization,
    ParticleLikelihoodObservation,
    PoseGraphConfig,
    TerrainEmergencyRecoveryController,
    TerrainRecoveryController,
    TerrainMatcherConfig,
    TerrainNavigationFilter,
    TerrainNavigationFilterConfig,
    TerrainParticleLikelihoodConfig,
    covariance_intersection_pose2,
    fixed_lag_fused_deskew_states,
    gauge_anchored_initial_covariance,
    initialize_from_imu_dvl_depth,
    information_with_covariance_floor,
    reporting_covariance_from_information,
    map_to_odom_change_metrics,
    make_terrain_recovery_config,
    motion_compensate_to_body,
    sanitize_for_strict_json,
    se3_relative_pose_error_by_distance,
    terrain_prior_prefit_bypass_allowed,
    voxel_average_backscatter_patch,
    voxel_average_terrain_patch,
)
from isaacsim.oceansim.slam.large_scale_sim import (
    LargeScaleSensorSimulator,
)
from isaacsim.oceansim.slam.math3d import rotation_from_rpy, rpy_from_rotation
from isaacsim.oceansim.slam.real_bathymetry import (
    MeasuredBathymetryGrid,
    TerrainFollowingSurvey,
    WaypointTerrainFollowingSurvey,
)


LAUV_MESH_ROOT = ROOT / "assets/vehicles/lauv/meshes"


def _xform_ops(prim, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)):
    xformable = UsdGeom.Xformable(prim)
    translate = xformable.AddTranslateOp()
    translate.Set(Gf.Vec3d(*(float(value) for value in translation)))
    rotate = xformable.AddRotateXYZOp()
    rotate.Set(Gf.Vec3f(*(float(value) for value in rotation)))
    return translate, rotate


def _make_cube(
    stage,
    path: str,
    *,
    position,
    scale,
    color,
    rotation=(0.0, 0.0, 0.0),
    collision=True,
    opacity=1.0,
):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    cube.CreateDisplayOpacityAttr([float(opacity)])
    translate, rotate = _xform_ops(cube.GetPrim(), position, rotation)
    del translate, rotate
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(*(float(value) for value in scale))
    )
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _make_sphere(stage, path: str, position, radius: float, color):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _xform_ops(sphere.GetPrim(), position)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    return sphere


def _make_terrain(
    stage,
    model: MeasuredBathymetryGrid,
    *,
    stride: int,
    collision_enabled: bool,
):
    """Build a whole-footprint display mesh without filling survey gaps."""

    points, indices = model.triangle_mesh(stride=stride, compact=True)
    if points.shape[0] == 0:
        raise RuntimeError("the measured map contains no renderable triangles")
    mesh = UsdGeom.Mesh.Define(stage, "/World/LargeRange/Seabed")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(*(float(value) for value in point)) for point in points]
    )
    mesh.CreateFaceVertexCountsAttr([3] * (indices.size // 3))
    mesh.CreateFaceVertexIndicesAttr(indices.tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    colours = model.seabed_color(points[:, 0], points[:, 1]).reshape(-1, 3)
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
        [Gf.Vec3f(*colour) for colour in colours]
    )
    # Smooth only rendered lighting.  Warp owns the separate acoustic mesh;
    # PhysX collision is created only when that fallback/verification is used.
    normals = model.normal(points[:, 0], points[:, 1]).reshape(-1, 3)
    normals = np.nan_to_num(normals, nan=0.0)
    invalid_normals = np.linalg.norm(normals, axis=1) < 1.0e-8
    normals[invalid_normals] = np.array([0.0, 0.0, 1.0])
    mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in normals])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if collision_enabled:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        collision.CreateApproximationAttr().Set("none")
    mesh.GetPrim().SetCustomDataByKey("oceansim:sourceGridStride", int(stride))
    mesh.GetPrim().SetCustomDataByKey(
        "oceansim:completeMeasuredFootprint", True
    )
    return mesh


def _make_acoustic_backend(model: MeasuredBathymetryGrid):
    """Create the static-mesh backend for the integrated sonar ping."""

    if ARGS.sonar_backend == "physx":
        from isaacsim.oceansim.sensors.raycast_backend import PhysXRaycastBackend

        carb.log_info("[OceanSim sonar] using PhysX CPU scene queries")
        return PhysXRaycastBackend()

    from isaacsim.oceansim.sensors.warp_raycast_backend import (
        WarpMeshRaycastBackend,
    )

    if ARGS.sonar_backend in ("auto", "warp-cuda"):
        if not WarpMeshRaycastBackend.cuda_available():
            if ARGS.sonar_backend == "warp-cuda":
                raise RuntimeError(
                    "--sonar-backend warp-cuda was requested but Warp cannot "
                    "access a CUDA device"
                )
            carb.log_warn(
                "[OceanSim sonar] Warp CUDA unavailable; falling back to "
                "PhysX CPU scene queries"
            )
            from isaacsim.oceansim.sensors.raycast_backend import (
                PhysXRaycastBackend,
            )

            return PhysXRaycastBackend()
        device = "cuda:0"
    else:
        device = "cpu"

    points, indices = model.triangle_mesh(
        stride=ARGS.sonar_terrain_stride, compact=True
    )
    # Preserve the geometry in the small radiometric gaps, but use zero return
    # power there.  This avoids inventing texture while allowing measured MBES
    # ranges to remain available throughout the valid bathymetry footprint.
    measured_reflectivity = model.reflectivity(
        points[:, 0], points[:, 1], measured_only=True
    ).reshape(-1)
    reflectivity = np.nan_to_num(
        measured_reflectivity, nan=0.0, posinf=0.0, neginf=0.0
    )
    backend = WarpMeshRaycastBackend(
        points,
        indices,
        vertex_reflectivities=reflectivity,
        prim_path="/World/LargeRange/Seabed",
        device=device,
    )
    carb.log_info(
        f"[OceanSim sonar] {backend.execution_provider} on "
        f"{backend.device_name}: {backend.triangle_count:,} seabed triangles, "
        f"whole-map stride {ARGS.sonar_terrain_stride} "
        f"({ARGS.sonar_terrain_stride * model.resolution_m:.0f} m), "
        f"{np.count_nonzero(reflectivity):,}/{reflectivity.size:,} vertices "
        f"with measured backscatter, BVH build {backend.build_ms:.1f} ms"
    )
    return backend


def _acoustic_backend_label(backend) -> str:
    if backend is None:
        return "PhysX CPU"
    if not getattr(backend, "is_cuda", False):
        return backend.execution_provider
    return f"{backend.execution_provider} ({backend.device_name})"


def _make_slam_associator():
    """Create the exact point-to-surfel lookup backend for the ESIKF."""

    if ARGS.slam_association_backend == "cpu":
        carb.log_info("[OceanSim SLAM] using Python CPU voxel association")
        return None
    if (
        ARGS.slam_association_backend == "auto"
        and ARGS.slam_max_points < 256
    ):
        carb.log_info(
            "[OceanSim SLAM] retaining Python CPU association below 256 "
            "points per update (measured launch/transfer crossover)"
        )
        return None

    from isaacsim.oceansim.slam.warp_surfel_association import (
        WarpSurfelAssociator,
    )

    if ARGS.slam_association_backend in ("auto", "warp-cuda"):
        if not WarpSurfelAssociator.cuda_available():
            if ARGS.slam_association_backend == "warp-cuda":
                raise RuntimeError(
                    "--slam-association-backend warp-cuda was requested but "
                    "Warp cannot access a CUDA device"
                )
            carb.log_warn(
                "[OceanSim SLAM] Warp CUDA unavailable; retaining Python CPU "
                "voxel association"
            )
            return None
        device = "cuda:0"
        minimum_surfels = 0
    else:
        device = "cpu"
        minimum_surfels = 0

    associator = WarpSurfelAssociator(
        device=device,
        minimum_gpu_surfels=minimum_surfels,
    )
    warmup_ms = associator.warm_up()
    carb.log_info(
        f"[OceanSim SLAM] {associator.execution_provider} exact 27-neighbour "
        f"association on {associator.device_name}; warm-up {warmup_ms:.1f} ms"
    )
    return associator


def _slam_association_label(associator) -> str:
    if associator is None:
        return "Python CPU"
    return f"{associator.execution_provider} ({associator.device_name})"


def _finite_float_or_none(value) -> float | None:
    """Return a strict-JSON finite float, or ``None`` for absent/nonfinite data."""

    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _chi_square_sample_summary(
    values: list[float],
) -> dict[str, float | int | None]:
    """Return strict-JSON audit statistics for performed prefit tests."""

    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        maximum = percentile_95 = mean = None
    else:
        maximum = float(np.max(finite))
        percentile_95 = float(np.percentile(finite, 95.0))
        mean = float(np.mean(finite))
    return {
        "count": int(samples.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(samples.size - finite.size),
        "max": maximum,
        "p95": percentile_95,
        "mean": mean,
    }


def _position_error_metrics(
    estimates: np.ndarray, truth: np.ndarray
) -> dict[str, float | int]:
    """Summarize synchronized 3-D position errors without trajectory alignment."""

    estimates = np.asarray(estimates, dtype=np.float64).reshape(-1, 3)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 3)
    if estimates.shape != truth.shape or estimates.shape[0] == 0:
        raise ValueError("estimate and truth arrays must have equal nonzero shape")
    delta = estimates - truth
    error_3d = np.linalg.norm(delta, axis=1)
    error_horizontal = np.linalg.norm(delta[:, :2], axis=1)
    error_vertical = np.abs(delta[:, 2])
    return {
        "samples": int(error_3d.size),
        "rmse_3d_m": float(np.sqrt(np.mean(error_3d**2))),
        "mean_3d_m": float(np.mean(error_3d)),
        "median_3d_m": float(np.median(error_3d)),
        "p95_3d_m": float(np.percentile(error_3d, 95.0)),
        "p99_3d_m": float(np.percentile(error_3d, 99.0)),
        "max_3d_m": float(np.max(error_3d)),
        "final_3d_m": float(error_3d[-1]),
        "horizontal_rmse_m": float(
            np.sqrt(np.mean(error_horizontal**2))
        ),
        "vertical_rmse_m": float(np.sqrt(np.mean(error_vertical**2))),
        "final_horizontal_m": float(error_horizontal[-1]),
        "final_vertical_m": float(error_vertical[-1]),
    }


def _relative_position_error_metrics(
    estimates: np.ndarray,
    truth: np.ndarray,
    cumulative_distance_m: np.ndarray,
    interval_m: float,
) -> dict[str, float | int | bool | None]:
    """Legacy world-frame displacement error over fixed route intervals.

    This quantity is intentionally retained for result-file compatibility.  It
    is not a gauge-invariant relative-pose error: a constant global rotation of
    an otherwise exact trajectory changes the reported value.  Use
    ``se3_relative_pose_error_by_distance`` for odometry-accuracy claims.
    """

    estimates = np.asarray(estimates, dtype=np.float64).reshape(-1, 3)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 3)
    distance = np.asarray(cumulative_distance_m, dtype=np.float64).reshape(-1)
    if (
        estimates.shape != truth.shape
        or estimates.shape[0] != distance.size
        or interval_m <= 0.0
    ):
        raise ValueError("RPE inputs must share shape and interval must be positive")
    target = distance + float(interval_m)
    ending = np.searchsorted(distance, target, side="left")
    starting = np.arange(distance.size, dtype=np.int64)
    valid = ending < distance.size
    starting = starting[valid]
    ending = ending[valid]
    if starting.size == 0:
        return {
            "available": False,
            "samples": 0,
            "requested_interval_m": float(interval_m),
            "mean_actual_interval_m": None,
            "rmse_translation_m": None,
            "mean_translation_m": None,
            "p95_translation_m": None,
            "max_translation_m": None,
        }
    truth_delta = truth[ending] - truth[starting]
    estimate_delta = estimates[ending] - estimates[starting]
    errors = np.linalg.norm(estimate_delta - truth_delta, axis=1)
    return {
        "available": True,
        "samples": int(errors.size),
        "requested_interval_m": float(interval_m),
        "mean_actual_interval_m": float(
            np.mean(distance[ending] - distance[starting])
        ),
        "rmse_translation_m": float(np.sqrt(np.mean(errors**2))),
        "mean_translation_m": float(np.mean(errors)),
        "p95_translation_m": float(np.percentile(errors, 95.0)),
        "max_translation_m": float(np.max(errors)),
    }


def _orientation_error_metrics(
    estimates: np.ndarray, truth: np.ndarray
) -> dict[str, float | int | dict[str, float]]:
    """Summarize synchronized attitude error using relative SO(3) rotations.

    Rotation matrices are composed as ``R_truth.T @ R_estimate`` before their
    intrinsic roll/pitch/yaw error is extracted.  This is the matrix-equivalent
    of forming the normalized relative quaternion; it avoids the invalid
    component-wise subtraction of quaternions and naturally wraps yaw.
    """

    estimates = np.asarray(estimates, dtype=np.float64).reshape(-1, 3, 3)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 3, 3)
    if estimates.shape != truth.shape or estimates.shape[0] == 0:
        raise ValueError("estimate and truth rotations must have equal nonzero shape")

    relative = np.einsum("nji,njk->nik", truth, estimates)
    pitch = np.arcsin(np.clip(-relative[:, 2, 0], -1.0, 1.0))
    nonsingular = np.abs(np.cos(pitch)) > 1.0e-7
    roll = np.zeros_like(pitch)
    yaw = np.empty_like(pitch)
    roll[nonsingular] = np.arctan2(
        relative[nonsingular, 2, 1], relative[nonsingular, 2, 2]
    )
    yaw[nonsingular] = np.arctan2(
        relative[nonsingular, 1, 0], relative[nonsingular, 0, 0]
    )
    yaw[~nonsingular] = np.arctan2(
        -relative[~nonsingular, 0, 1], relative[~nonsingular, 1, 1]
    )
    errors_deg = np.rad2deg(np.column_stack((roll, pitch, yaw)))

    trace = np.trace(relative, axis1=1, axis2=2)
    geodesic_deg = np.rad2deg(
        np.arccos(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))
    )

    def component(values: np.ndarray) -> dict[str, float]:
        absolute = np.abs(values)
        return {
            "rmse_deg": float(np.sqrt(np.mean(values**2))),
            "p95_abs_deg": float(np.percentile(absolute, 95.0)),
            "max_abs_deg": float(np.max(absolute)),
            "final_abs_deg": float(absolute[-1]),
            "final_signed_deg": float(values[-1]),
        }

    return {
        "samples": int(errors_deg.shape[0]),
        "roll": component(errors_deg[:, 0]),
        "pitch": component(errors_deg[:, 1]),
        "yaw": component(errors_deg[:, 2]),
        "total_rotation": component(geodesic_deg),
    }


def _pose_graph_corrected_rotations(
    pose_graph,
    timestamps: np.ndarray,
    raw_rotations: np.ndarray,
    *,
    optimized_poses: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the pose graph's interpolated yaw correction to attitude history."""

    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    rotations = np.asarray(raw_rotations, dtype=np.float64).reshape(-1, 3, 3)
    if times.size != rotations.shape[0]:
        raise ValueError("rotation timestamps and samples must have equal length")
    if not pose_graph.keyframes or times.size == 0:
        return rotations.copy()
    key_times, key_yaw_delta, _ = pose_graph._corrections(optimized_poses)
    yaw_delta = np.interp(times, key_times, np.unwrap(key_yaw_delta))
    cosine = np.cos(yaw_delta)
    sine = np.sin(yaw_delta)
    corrections = np.zeros_like(rotations)
    corrections[:, 0, 0] = cosine
    corrections[:, 0, 1] = -sine
    corrections[:, 1, 0] = sine
    corrections[:, 1, 1] = cosine
    corrections[:, 2, 2] = 1.0
    return np.einsum("nij,njk->nik", corrections, rotations)


def _five_kilometre_error_segments(
    estimates: np.ndarray,
    truth: np.ndarray,
    cumulative_distance_m: np.ndarray,
) -> list[dict[str, float | int]]:
    """Return unaligned ATE summaries in consecutive 5 km truth bins."""

    estimates = np.asarray(estimates, dtype=np.float64).reshape(-1, 3)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 3)
    distance = np.asarray(cumulative_distance_m, dtype=np.float64).reshape(-1)
    results: list[dict[str, float | int]] = []
    start_m = 0.0
    total_m = float(distance[-1])
    while start_m < total_m + 1.0e-9:
        stop_m = min(start_m + 5000.0, total_m)
        if stop_m >= total_m:
            selected = (distance >= start_m) & (distance <= stop_m + 1.0e-9)
        else:
            selected = (distance >= start_m) & (distance < stop_m)
        if np.any(selected):
            summary = _position_error_metrics(
                estimates[selected], truth[selected]
            )
            summary["start_distance_m"] = float(start_m)
            summary["stop_distance_m"] = float(stop_m)
            results.append(summary)
        if stop_m >= total_m:
            break
        start_m = stop_m
    return results


def _verify_acoustic_backend(mbes, backend, repeats: int = 20) -> dict:
    """Check CUDA ranges against PhysX and time both closest-hit paths."""

    if backend is None or not getattr(backend, "is_cuda", False):
        return {}
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    from isaacsim.oceansim.sensors.raycast_backend import PhysXRaycastBackend

    origin, orientation = mbes.get_world_pose()
    origin = np.asarray(origin, dtype=np.float64)
    directions = mbes.get_beam_directions() @ quat_to_rot_matrix(orientation).T
    physx = PhysXRaycastBackend(
        default_reflectivity=1.0,
        reflectivity_resolver=lambda _hit: 1.0,
    )

    # Warm both paths before timing so JIT compilation and collision cooking
    # are not reported as steady-state sensor latency.
    backend.cast(origin, directions, 140.0)
    physx.cast(origin, directions, 140.0)

    gpu_started = time.perf_counter()
    for _ in range(repeats):
        warp_hits = backend.cast(origin, directions, 140.0)
    warp_ms = 1.0e3 * (time.perf_counter() - gpu_started) / repeats

    physx_started = time.perf_counter()
    for _ in range(repeats):
        physx_hits = physx.cast(origin, directions, 140.0)
    physx_ms = 1.0e3 * (time.perf_counter() - physx_started) / repeats

    agreement = float(np.mean(warp_hits.hit_mask == physx_hits.hit_mask))
    common = warp_hits.hit_mask & physx_hits.hit_mask
    if not np.any(common):
        raise RuntimeError("Warp/PhysX validation produced no common seabed hits")
    range_error = warp_hits.distances[common] - physx_hits.distances[common]
    result = {
        "rays": int(directions.shape[0]),
        "hit_agreement": agreement,
        "common_hits": int(np.count_nonzero(common)),
        "range_rmse_m": float(np.sqrt(np.mean(np.square(range_error)))),
        "range_max_abs_m": float(np.max(np.abs(range_error))),
        "warp_geometry_ms": float(warp_ms),
        "physx_python_ms": float(physx_ms),
        "speedup": float(physx_ms / max(warp_ms, 1.0e-12)),
    }
    if result["hit_agreement"] < 0.99 or result["range_max_abs_m"] > 0.01:
        raise RuntimeError(
            "Warp/PhysX seabed validation failed: " + json.dumps(result)
        )
    print(
        f"[OceanSim sonar] Warp/PhysX validation: {json.dumps(result)}",
        flush=True,
    )
    return result


def _sample_mesh_colours(geometry, fallback) -> np.ndarray:
    """Bake a Collada texture into USD vertex colours when UVs are present."""

    count = geometry.vertices.shape[0]
    visual = geometry.visual
    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)
    if texture is None or uv is None or len(uv) != count:
        return np.broadcast_to(np.asarray(fallback), (count, 3)).copy()
    image = np.asarray(texture.convert("RGB"), dtype=np.float64) / 255.0
    u = np.mod(np.asarray(uv[:, 0], dtype=np.float64), 1.0)
    v = np.mod(np.asarray(uv[:, 1], dtype=np.float64), 1.0)
    column = np.clip(np.rint(u * (image.shape[1] - 1)), 0, image.shape[1] - 1)
    row = np.clip(
        np.rint((1.0 - v) * (image.shape[0] - 1)), 0, image.shape[0] - 1
    )
    return image[row.astype(np.int64), column.astype(np.int64)]


def _make_lauv_component(
    stage,
    path: str,
    mesh_path: Path,
    *,
    position=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
    fallback_colour=(0.32, 0.34, 0.36),
):
    """Load one LAUV Collada component as native OpenUSD triangle meshes."""

    if not mesh_path.is_file():
        raise FileNotFoundError(f"LAUV mesh is missing: {mesh_path}")
    root = UsdGeom.Xform.Define(stage, path)
    _xform_ops(root.GetPrim(), position, rotation)
    scene = trimesh.load(mesh_path, force="scene")
    for index, node_name in enumerate(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        usd_mesh = UsdGeom.Mesh.Define(stage, f"{path}/Part_{index:02d}")
        usd_mesh.CreatePointsAttr(
            [Gf.Vec3f(*vertex) for vertex in np.asarray(geometry.vertices)]
        )
        faces = np.asarray(geometry.faces, dtype=np.int64)
        usd_mesh.CreateFaceVertexCountsAttr([3] * faces.shape[0])
        usd_mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        colours = _sample_mesh_colours(geometry, fallback_colour)
        usd_mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
            [Gf.Vec3f(*colour) for colour in colours]
        )
        normals = np.asarray(geometry.vertex_normals, dtype=np.float64)
        usd_mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in normals])
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return root


def _make_curve(stage, path: str, color, width: float):
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    curve.CreateWidthsAttr([float(width)])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr([Gf.Vec3f(0.0), Gf.Vec3f(0.0)])
    return curve


def _update_curve(curve, positions: list[np.ndarray], lift: float = 0.0):
    if not positions:
        return
    values = positions if len(positions) > 1 else [positions[0], positions[0]]
    if len(values) > 6_000:
        indices = np.linspace(0, len(values) - 1, 6_000, dtype=np.int64)
        values = [values[int(index)] for index in indices]
    points = [
        Gf.Vec3f(float(p[0]), float(p[1]), float(p[2] + lift)) for p in values
    ]
    curve.GetPointsAttr().Set(points)
    curve.GetCurveVertexCountsAttr().Set([len(points)])


def _make_status_window(model: MeasuredBathymetryGrid):
    if ARGS.headless:
        return None, None
    import omni.ui as ui

    window = ui.Window(
        "OceanSim | Monterey Canyon + LAUV | LIVE SLAM",
        width=540,
        height=326,
        visible=True,
    )
    window.position_x = 16
    window.position_y = 48
    with window.frame:
        with ui.VStack(spacing=5):
            ui.Label(
                "LIVE ACOUSTIC-INERTIAL SLAM",
                height=28,
                style={"color": 0xFF5FE8FF, "font_size": 18},
            )
            status = ui.Label(
                (
                    f"Loading COMPLETE USGS Monterey Canyon "
                    f"{model.width / 1000.0:.2f} x "
                    f"{model.length / 1000.0:.2f} km measured acoustic map..."
                ),
                height=234,
                word_wrap=True,
                style={"color": 0xFFE4F7FA, "font_size": 14},
            )
            ui.Label(
                "USGS 5 m MBES + measured backscatter | no crop/exaggeration",
                height=22,
                style={"color": 0xFFACCBD2, "font_size": 12},
            )
    return window, status


def _position_sensor_windows(sonar):
    if ARGS.headless:
        return
    for viewport, (x, y) in zip(
        (sonar._visual_viewport, sonar._sidescan_viewport),
        ((16, 680), (690, 680)),
    ):
        try:
            viewport._window.position_x = x
            viewport._window.position_y = y
        except (AttributeError, RuntimeError):
            pass


def _restore_standard_workspace() -> None:
    """Restore the standard Isaac authoring panes used by the original view."""

    if ARGS.headless:
        return
    import omni.ui as ui

    for name in ("Stage", "Property", "Content"):
        ui.Workspace.show_window(name, True)


def build_scene(
    world_model: MeasuredBathymetryGrid,
    *,
    terrain_stride: int,
    terrain_collision: bool,
):
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Scope.Define(stage, "/World/LargeRange")
    _make_terrain(
        stage,
        world_model,
        stride=terrain_stride,
        collision_enabled=terrain_collision,
    )

    vehicle = UsdGeom.Xform.Define(stage, "/World/Vehicle")
    vehicle_translate, vehicle_rotate = _xform_ops(vehicle.GetPrim())
    _make_lauv_component(
        stage,
        "/World/Vehicle/LAUVBody",
        LAUV_MESH_ROOT / "body.dae",
        fallback_colour=(0.72, 0.05, 0.025),
    )
    _make_lauv_component(
        stage,
        "/World/Vehicle/Propeller",
        LAUV_MESH_ROOT / "propeller.dae",
        position=(-0.53994, 0.0, 0.0),
        rotation=(0.0, 0.0, 180.0),
        fallback_colour=(0.78, 0.48, 0.09),
    )
    for index, (mesh_name, position, rotation, colour) in enumerate(
        (
            ("red_fin.dae", (-0.4, 0.0, 0.04), (0.0, 0.0, 0.0), (0.75, 0.03, 0.02)),
            ("black_fin.dae", (-0.4, -0.04, 0.0), (90.0, 0.0, 0.0), (0.025, 0.025, 0.03)),
            ("black_fin.dae", (-0.4, 0.0, -0.04), (180.0, 0.0, 0.0), (0.025, 0.025, 0.03)),
            ("black_fin.dae", (-0.4, 0.04, 0.0), (-90.0, 0.0, 0.0), (0.025, 0.025, 0.03)),
        )
    ):
        _make_lauv_component(
            stage,
            f"/World/Vehicle/ControlFin_{index}",
            LAUV_MESH_ROOT / mesh_name,
            position=position,
            rotation=rotation,
            fallback_colour=colour,
        )
    _make_lauv_component(
        stage,
        "/World/Vehicle/IntegratedSonarHousing",
        LAUV_MESH_ROOT / "side_scan_sonar.dae",
        position=(0.10, 0.0, -0.15),
        fallback_colour=(0.12, 0.12, 0.13),
    )
    light = UsdLux.SphereLight.Define(stage, "/World/Vehicle/WorkLight")
    light.CreateIntensityAttr(12_000.0)
    light.CreateRadiusAttr(0.18)
    light.CreateColorAttr(Gf.Vec3f(0.30, 0.78, 0.91))
    _xform_ops(light.GetPrim(), (0.48, 0.0, -0.04))

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(720.0)
    sun.CreateColorAttr(Gf.Vec3f(0.38, 0.63, 0.74))
    _xform_ops(sun.GetPrim(), rotation=(35.0, -25.0, -25.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/Ambient")
    dome.CreateIntensityAttr(260.0)
    dome.CreateColorAttr(Gf.Vec3f(0.025, 0.17, 0.24))

    curves = {
        "truth": _make_curve(stage, "/World/Tracks/Truth", (0.15, 1.0, 0.35), 0.07),
        "local": _make_curve(stage, "/World/Tracks/Local", (1.0, 0.42, 0.05), 0.06),
        "global": _make_curve(stage, "/World/Tracks/Global", (0.05, 0.85, 1.0), 0.09),
    }
    map_points = UsdGeom.Points.Define(stage, "/World/Tracks/OptimizedMBES")
    map_points.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.55, 1.0)])
    map_points.CreateWidthsAttr([0.38])
    map_points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    return stage, vehicle_translate, vehicle_rotate, curves, map_points


def _set_vehicle_pose(translate_op, rotate_op, position, rotation):
    rpy = np.rad2deg(rpy_from_rotation(rotation))
    translate_op.Set(Gf.Vec3d(*(float(value) for value in position)))
    rotate_op.Set(Gf.Vec3f(*(float(value) for value in rpy)))


def _set_live_camera(truth, world_model: MeasuredBathymetryGrid) -> None:
    if ARGS.camera_mode == "overview":
        span = max(world_model.width, world_model.length)
        centre_z = float(world_model.navigation_height(0.0, 0.0))
        eye = np.array(
            [0.34 * span, -0.58 * span, centre_z + 0.62 * span]
        )
        target = np.array([0.0, 0.0, centre_z])
        _set_viewport_camera(eye, target)
        return
    if ARGS.camera_mode == "survey-overview":
        centre_x = float(ARGS.survey_center_x_m)
        centre_y = float(ARGS.survey_center_y_m)
        centre_z = float(world_model.navigation_height(centre_x, centre_y))
        span = 1.25 * max(
            2.0 * ARGS.survey_x_amplitude_m,
            2.0 * ARGS.survey_y_amplitude_m,
        )
        target = np.array([centre_x, centre_y, centre_z])
        eye = target + np.array([0.34 * span, -0.58 * span, 0.62 * span])
        _set_viewport_camera(eye, target)
        return
    forward = truth.rotation[:, 0]
    port = truth.rotation[:, 1]
    up = np.array([0.0, 0.0, 1.0])
    eye = truth.position - 5.2 * forward - 2.1 * port + 2.2 * up
    target = truth.position + 4.0 * forward - 3.8 * up
    _set_viewport_camera(eye, target)


def _set_viewport_camera(eye, target) -> None:
    """Set the original perspective camera using Kit's viewport camera state."""

    viewport = get_active_viewport()
    if viewport is None:
        return
    camera = ViewportCameraState("/OmniverseKit_Persp", viewport)
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    camera.set_position_world(Gf.Vec3d(*(float(value) for value in eye)), False)
    camera.set_target_world(Gf.Vec3d(*(float(value) for value in target)), True)


def _follow_live_camera(truth) -> None:
    """Follow the AUV while preserving the user's orbit and zoom distance."""

    if ARGS.headless or ARGS.camera_mode in ("survey-overview", "overview"):
        return
    forward = truth.rotation[:, 0]
    up = np.array([0.0, 0.0, 1.0])
    target = truth.position + 4.0 * forward - 3.8 * up
    viewport = get_active_viewport()
    if viewport is None:
        return
    camera = ViewportCameraState("/OmniverseKit_Persp", viewport)
    camera.set_target_world(
        Gf.Vec3d(*(float(value) for value in np.asarray(target))),
        False,
    )


def main() -> int:
    world_model = MeasuredBathymetryGrid.from_path(ARGS.bathymetry)
    profile_mode = {
        "v2-standard": "standard",
        "v2-high-resolution": "high",
    }[ARGS.sonar_profile]
    sonar_profile = get_v2_profile(profile_mode)
    sound_speed_mps = 1500.0
    two_way_travel_time_s = acoustic_two_way_travel_time_s(
        sonar_profile.maximum_range_m, sound_speed_mps
    )
    required_pings_in_water = required_distinguishable_pings_in_water(
        ARGS.ping_hz,
        sonar_profile.maximum_range_m,
        sound_speed_mps,
    )
    maximum_scheduled_ping_hz = maximum_supported_ping_rate_hz(
        sonar_profile.maximum_range_m,
        sound_speed_mps,
        ARGS.sonar_pings_in_water,
    )
    if ARGS.ping_hz > maximum_scheduled_ping_hz + 1.0e-12:
        raise ValueError(
            f"{ARGS.ping_hz:g} Hz requires at least "
            f"{required_pings_in_water} distinguishable pings in water for "
            f"{sonar_profile.maximum_range_m:g} m at {sound_speed_mps:g} m/s; "
            f"only {ARGS.sonar_pings_in_water} configured"
        )
    if ARGS.survey_speed_mps > 0.0:
        minimum_gap_free_ping_hz = (
            ARGS.survey_speed_mps
            / sonar_profile.along_track_coverage_per_ping_m
        )
        if ARGS.ping_hz + 1.0e-12 < minimum_gap_free_ping_hz:
            raise ValueError(
                f"{ARGS.ping_hz:g} Hz leaves along-track gaps at "
                f"{ARGS.survey_speed_mps:g} m/s in {profile_mode} mode; "
                f"use at least {minimum_gap_free_ping_hz:g} Hz"
            )
    survey_route_metadata: dict = {}
    if ARGS.survey_pattern == "full-map-route":
        with ARGS.survey_route.open("r", encoding="utf-8") as stream:
            survey_route_metadata = json.load(stream)
        if survey_route_metadata.get("schema") != (
            "oceansim.measured-terrain-route/v1"
        ):
            raise ValueError("unsupported or missing survey-route schema")
        route_map = survey_route_metadata.get("map", {})
        if (
            float(route_map.get("resolution_m", -1.0))
            != float(world_model.resolution_m)
            or list(route_map.get("scene_extent_m", []))
            != [float(world_model.width), float(world_model.length)]
        ):
            raise ValueError(
                "survey route was prepared for a different terrain grid"
            )
        route_definition = survey_route_metadata.get(
            "route_definition", {}
        )
        survey = WaypointTerrainFollowingSurvey(
            world_model,
            np.asarray(
                survey_route_metadata.get("control_points_xy_m", []),
                dtype=np.float64,
            ),
            speed_mps=ARGS.survey_speed_mps,
            altitude_m=ARGS.survey_altitude_m,
            smoothing_iterations=int(
                route_definition.get("smoothing_iterations", 4)
            ),
            maximum_pitch_deg=ARGS.survey_maximum_pitch_deg,
        )
    else:
        survey = TerrainFollowingSurvey(
            world_model,
            speed_mps=(
                ARGS.survey_speed_mps
                if ARGS.survey_speed_mps > 0.0
                else None
            ),
            center_x_m=ARGS.survey_center_x_m,
            center_y_m=ARGS.survey_center_y_m,
            x_amplitude=ARGS.survey_x_amplitude_m,
            y_amplitude=ARGS.survey_y_amplitude_m,
            pattern_yaw_deg=ARGS.survey_pattern_yaw_deg,
            altitude_m=ARGS.survey_altitude_m,
        )
    route_report = survey.validate_route(
        sample_count=8_192,
        minimum_clearance_m=sonar_profile.maximum_range_m,
        require_measured_backscatter=True,
    )
    route_length_m = float(route_report.path_length_m)
    if ARGS.survey_start_distance_m >= route_length_m:
        raise ValueError(
            "--survey-start-distance-m must be smaller than the validated "
            f"route length ({route_length_m:.3f} m)"
        )
    if (
        ARGS.survey_segment_distance_m > 0.0
        and ARGS.survey_start_distance_m
        + ARGS.survey_segment_distance_m
        > route_length_m + 1.0e-9
    ):
        raise ValueError(
            "the requested survey segment crosses the periodic route seam; "
            "choose a shorter segment so the focused run remains contiguous"
        )
    survey_start_time_offset_s = (
        ARGS.survey_start_distance_m / ARGS.survey_speed_mps
        if ARGS.survey_speed_mps > 0.0
        else 0.0
    )
    survey_segment_duration_s = (
        ARGS.survey_segment_distance_m / ARGS.survey_speed_mps
        if ARGS.survey_segment_distance_m > 0.0
        else None
    )

    def survey_state(segment_timestamp_s: float):
        """Evaluate truth on a selected route segment using a fresh clock."""

        return survey.state(
            float(segment_timestamp_s) + survey_start_time_offset_s
        )

    navigation = LargeScaleSensorSimulator(
        world_model,
        survey,
        max_range=sonar_profile.maximum_range_m,
        seed=ARGS.seed,
    )
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    terrain_collision = bool(
        ARGS.sonar_backend == "physx" or ARGS.verify_sonar_backend
    )
    terrain_stride = (
        ARGS.sonar_terrain_stride
        if terrain_collision
        else ARGS.terrain_display_stride
    )
    _, vehicle_translate, vehicle_rotate, curves, map_prim = build_scene(
        world_model,
        terrain_stride=terrain_stride,
        terrain_collision=terrain_collision,
    )
    acoustic_backend = _make_acoustic_backend(world_model)
    acoustic_backend_label = _acoustic_backend_label(acoustic_backend)
    slam_associator = _make_slam_associator()
    slam_association_label = _slam_association_label(slam_associator)

    sonar = BathymetricSideScanSonarSensor(
        prim_path="/World/Vehicle/bathymetric_sidescan",
        name="Integrated bathymetric side-scan",
        translation=np.array([0.10, 0.0, -0.15]),
        num_beams=ARGS.sonar_beams,
        swath_width_deg=ARGS.sonar_swath_width_deg,
        range_bins=sonar_profile.range_bins_per_side,
        imaging_rays_per_side=ARGS.sidescan_imaging_rays_per_side,
        along_track_beams_per_side=(
            sonar_profile.rows_per_side_per_ping
        ),
        # V2 defines 0.15/0.075 m as along-track spatial sampling, while
        # 0.14 deg is the individual beam opening.  Parallel virtual
        # footprint centres preserve that spacing at every slant range; the
        # unavailable hardware steering/phase-centre table remains unknown.
        along_track_origin_spacing_m=(
            sonar_profile.along_track_sample_spacing_m
        ),
        waterfall_lines=260,
        waterfall_decimation=10,
        beam_footprint_bins=1,
        min_range=0.0,
        max_range=sonar_profile.maximum_range_m,
        # The V2 document does not state frequency. Zero is the existing
        # message-level sentinel for "unspecified", not a physical 0 Hz ping.
        center_frequency_hz=0.0,
        range_noise_std=0.028,
        multiplicative_noise_std=0.04,
        additive_noise_std=0.008,
        attenuation=0.003,
        gain=1.0,
        random_seed=ARGS.seed + 1,
        raycast_backend=acoustic_backend,
        viewport=not ARGS.headless,
        draw_in_scene=not ARGS.headless,
        viewport_resolution=(640, 330),
        sidescan_viewport_resolution=(640, 330),
    )
    status_window, status_label = _make_status_window(world_model)

    truth0 = survey_state(0.0)
    startup_specific_force = []
    startup_dvl = []
    startup_depth = []
    for _ in range(32):
        _, specific_force = navigation.imu(truth0)
        startup_specific_force.append(specific_force)
        startup_dvl.append(navigation.dvl(truth0))
        startup_depth.append(navigation.depth(truth0))
    if ARGS.initialize_from_truth:
        initial = NavigationInitialization(
            rotation=truth0.rotation.copy(),
            position=truth0.position.copy(),
            velocity=truth0.velocity.copy(),
        )
    else:
        initial = initialize_from_imu_dvl_depth(
            np.asarray(startup_specific_force),
            np.asarray(startup_dvl),
            np.asarray(startup_depth),
            yaw_gauge_rad=(
                float(rpy_from_rotation(truth0.rotation)[2])
                + np.deg2rad(ARGS.initial_yaw_offset_deg)
            ),
            horizontal_origin_m=(
                truth0.position[:2]
                + np.array(
                    [
                        ARGS.initial_position_offset_x_m,
                        ARGS.initial_position_offset_y_m,
                    ],
                    dtype=np.float64,
                )
            ),
        )
    imu_bias_limit = np.deg2rad(1.0) / 3600.0
    initial_covariance = gauge_anchored_initial_covariance(
        gyro_bias_std_rad_s=imu_bias_limit / 2.0
    )
    slam = FastSonarSLAM(
        FastSonarSLAMConfig(
            filter=ESIKFConfig(
                gyro_bias_max_norm=imu_bias_limit,
                sonar_max_points=ARGS.slam_max_points,
                sonar_noise_floor=ARGS.local_sonar_noise_floor_m,
                intensity_noise_std=ARGS.local_intensity_noise_std,
                intensity_surface_consistency_std=(
                    ARGS.local_intensity_surface_consistency_std_m
                ),
                intensity_effective_independent_samples=(
                    ARGS.intensity_effective_independent_samples
                ),
                intensity_min_singular_ratio=(
                    ARGS.intensity_min_singular_ratio
                ),
                sonar_continuous_observability_weighting=(
                    ARGS.local_observability_mode == "continuous"
                ),
                sonar_observable_subspace_projection=(
                    ARGS.local_observability_mode == "hard"
                ),
            ),
            voxel_size=ARGS.slam_voxel_size_m,
            intensity_voxel_size=ARGS.intensity_voxel_size_m,
            map_insert_stride=ARGS.map_insert_stride,
            map_max_range=sonar_profile.maximum_range_m + 5.0,
            association_map_half_extent_m=(
                ARGS.association_map_half_extent_m
            ),
            mbes_group_pings=ARGS.mbes_group_pings,
            mbes_sliding_window_enabled=ARGS.enable_mbes_sliding_window,
            sidescan_min_range=0.0,
            sidescan_max_range=sonar_profile.maximum_range_m,
            intensity_min_separation_pings=(
                ARGS.intensity_reference_delay_pings
            ),
            intensity_group_pings=ARGS.intensity_group_pings,
            intensity_min_group_span_m=ARGS.intensity_min_group_span_m,
            intensity_submap_anchor_interval_s=(
                ARGS.intensity_submap_anchor_interval_s
            ),
            intensity_updates_enabled=not ARGS.disable_intensity_updates,
            trajectory_sample_interval_s=ARGS.trajectory_sample_interval_s,
            map_to_odom_soft_correction_enabled=(
                ARGS.enable_map_to_odom_soft_correction
            ),
            map_to_odom_translation_time_constant_s=(
                ARGS.map_to_odom_translation_time_constant_s
            ),
            map_to_odom_yaw_time_constant_s=(
                ARGS.map_to_odom_yaw_time_constant_s
            ),
            map_to_odom_maximum_translation_rate_mps=(
                ARGS.map_to_odom_maximum_translation_rate_mps
            ),
            map_to_odom_maximum_yaw_rate_deg_s=(
                ARGS.map_to_odom_maximum_yaw_rate_deg_s
            ),
            pose_graph=PoseGraphConfig(
                optimizer_backend=ARGS.pose_graph_backend.replace("-", "_"),
                fixed_lag_seconds=ARGS.pose_graph_fixed_lag_s,
                isam_relinearize_threshold=(
                    ARGS.pose_graph_isam_relinearize_threshold
                ),
                isam_relinearize_skip=(
                    ARGS.pose_graph_isam_relinearize_skip
                ),
                keyframe_distance_m=ARGS.keyframe_distance_m,
                keyframe_max_interval_s=ARGS.keyframe_max_interval_s,
                odometry_reference_interval_s=(
                    ARGS.pose_graph_odometry_reference_interval_s
                ),
                odometry_xy_std_m=(
                    ARGS.pose_graph_odometry_xy_std_m
                ),
                odometry_yaw_std_deg=(
                    ARGS.pose_graph_odometry_yaw_std_deg
                ),
                loop_min_separation_s=(
                    300.0
                    if ARGS.survey_pattern == "full-map-route"
                    else max(300.0, 0.70 * survey.period)
                ),
                loop_search_radius_m=110.0,
                loop_period_tolerance_s=10.0,
                intensity_factor_min_interval_s=(
                    ARGS.intensity_factor_min_interval_s
                ),
                intensity_factor_optimize_interval=(
                    ARGS.intensity_factor_optimize_interval
                ),
                terrain_prior_optimize_interval=(
                    ARGS.terrain_prior_optimize_interval
                ),
                terrain_prior_robust_kernel=(
                    ARGS.terrain_prior_robust_kernel
                ),
            ),
        ),
        sensor_translation_body=np.array([0.10, 0.0, -0.15]),
        surfel_associator=slam_associator,
    )
    slam.initialize(
        0.0,
        rotation=initial.rotation,
        position=initial.position,
        velocity=initial.velocity,
        covariance=initial_covariance,
    )
    dead_reckoning = AcousticESIKF(
        ESIKFConfig(gyro_bias_max_norm=imu_bias_limit)
    )
    dead_reckoning.initialize(
        0.0,
        rotation=initial.rotation,
        position=initial.position,
        velocity=initial.velocity,
        covariance=initial_covariance,
    )

    terrain_prior_model = world_model
    terrain_prior_model_kind = "measured_5m_same_dem_upper_bound"
    terrain_prior_vertical_model_std_m = 0.50
    if (
        ARGS.enable_terrain_map_matching
        and ARGS.terrain_prior_surface == "low-pass"
    ):
        # Avoid the classic inverse crime in the default validation: acoustic
        # rays hit the native measured 5 m relief, whereas localization sees
        # only the preparation pipeline's 2.2-cell Gaussian navigation
        # surface. It is still derived from the same public survey, not an
        # independent field campaign, and is labelled accordingly in metrics.
        terrain_prior_model = MeasuredBathymetryGrid(
            world_model.navigation_elevation_m,
            resolution_m=world_model.resolution_m,
            navigation_elevation_m=world_model.navigation_elevation_m,
            valid_mask=world_model.valid_mask,
            metadata={
                **world_model.metadata,
                "terrain_matching_role": (
                    "same-survey low-pass prior; Gaussian sigma 2.2 cells"
                ),
            },
            source_path=world_model.source_path,
        )
        terrain_prior_model_kind = "same_survey_low_pass_sigma_2p2_cells"
        # The native-versus-low-pass grid difference is 1.046 m RMS over all
        # measured cells. Include it with sonar/datum uncertainty rather than
        # pretending dense beams are centimetre-independent measurements.
        terrain_prior_vertical_model_std_m = 1.50
    terrain_matcher_config = None
    if ARGS.enable_terrain_map_matching:
        terrain_matcher_config = TerrainMatcherConfig(
            measurement_std_m=terrain_prior_vertical_model_std_m,
            huber_delta_m=(1.5 * terrain_prior_vertical_model_std_m),
            inlier_threshold_m=(2.5 * terrain_prior_vertical_model_std_m),
            maximum_points=ARGS.terrain_match_maximum_points,
            maximum_translation_m=ARGS.terrain_match_translation_window_m,
            maximum_yaw_deg=ARGS.terrain_match_yaw_window_deg,
            require_yaw_observable=ARGS.terrain_require_yaw_observable,
            # The seed is the corrected global pose while this covariance
            # belongs to the independent local ESIKF. Pose-graph robust
            # factors perform the statistically valid compatibility check;
            # applying the local-filter NIS gate here would mix frames.
            enable_innovation_nis_gate=False,
        )
    terrain_matcher = (
        BathymetricTerrainMatcher(terrain_prior_model, terrain_matcher_config)
        if terrain_matcher_config is not None
        else None
    )
    terrain_position_fallback_matcher = (
        BathymetricTerrainMatcher(
            terrain_prior_model,
            replace(
                terrain_matcher_config,
                require_yaw_observable=False,
            ),
        )
        if (
            terrain_matcher_config is not None
            and terrain_matcher_config.require_yaw_observable
        )
        else None
    )
    backscatter_matcher = (
        BackscatterMapMatcher(
            world_model,
            BackscatterMatcherConfig(
                maximum_points=ARGS.backscatter_match_maximum_points,
                maximum_yaw_deg=ARGS.backscatter_match_yaw_window_deg,
            ),
        )
        if ARGS.enable_backscatter_map_matching
        else None
    )
    backscatter_boundary_yaw_retry_matcher = (
        BackscatterMapMatcher(
            world_model,
            replace(
                backscatter_matcher.config,
                maximum_yaw_deg=(
                    ARGS.backscatter_boundary_yaw_retry_window_deg
                ),
            ),
        )
        if (
            backscatter_matcher is not None
            and ARGS.backscatter_boundary_yaw_retry_window_deg > 0.0
        )
        else None
    )
    global_backscatter_refinement_matcher = (
        BackscatterMapMatcher(
            world_model,
            replace(
                backscatter_matcher.config,
                maximum_points=max(
                    4_096, ARGS.backscatter_match_maximum_points
                ),
                maximum_translation_m=(
                    ARGS.terrain_match_translation_window_m
                ),
                maximum_yaw_deg=max(
                    6.0, ARGS.terrain_global_yaw_step_deg
                ),
                coarse_translation_step_m=min(
                    5.0, ARGS.terrain_match_translation_window_m
                ),
                coarse_yaw_step_deg=1.0,
                basin_translation_separation_m=max(
                    10.0, 2.0 * world_model.resolution_m
                ),
                basin_yaw_separation_deg=2.0,
                scale_space_cell_sizes_m=(
                    2.0 * world_model.resolution_m,
                    5.0 * world_model.resolution_m,
                    10.0 * world_model.resolution_m,
                ),
                scale_space_weight=0.35,
                context_consistency_weight=0.20,
                context_length_m=10.0 * world_model.resolution_m,
                minimum_scale_cells=12,
                minimum_context_points=24,
            ),
        )
        if ARGS.enable_global_terrain_relocalization
        else None
    )
    terrain_recovery_config = (
        make_terrain_recovery_config(
            terrain_matcher_config,
            maximum_translation_m=(
                ARGS.terrain_recovery_translation_window_m
            ),
        )
        if (
            terrain_matcher_config is not None
            and ARGS.terrain_recovery_translation_window_m > 0.0
        )
        else None
    )
    terrain_recovery_matcher = (
        BathymetricTerrainMatcher(
            terrain_prior_model,
            terrain_recovery_config,
        )
        if terrain_recovery_config is not None
        else None
    )
    terrain_emergency_recovery_config = (
        make_terrain_recovery_config(
            terrain_matcher_config,
            maximum_translation_m=(
                ARGS.terrain_emergency_recovery_translation_window_m
            ),
        )
        if (
            terrain_matcher_config is not None
            and ARGS.terrain_emergency_recovery_translation_window_m > 0.0
        )
        else None
    )
    terrain_emergency_recovery_matcher = (
        BathymetricTerrainMatcher(
            terrain_prior_model,
            terrain_emergency_recovery_config,
        )
        if terrain_emergency_recovery_config is not None
        else None
    )
    global_terrain_relocalizer = (
        GlobalTerrainRelocalizer(
            world_model,
            GlobalTerrainRelocalizerConfig(
                pyramid_stride=ARGS.terrain_global_pyramid_stride,
                yaw_half_window_deg=ARGS.terrain_global_yaw_window_deg,
                yaw_step_deg=ARGS.terrain_global_yaw_step_deg,
                minimum_joint_correlation=(
                    ARGS.terrain_global_minimum_correlation
                ),
                maximum_bathymetry_rmse_m=(
                    ARGS.terrain_global_maximum_bathymetry_rmse_m
                ),
                bathymetry_absolute_weight=(
                    ARGS.terrain_global_bathymetry_absolute_weight
                ),
                minimum_basin_score_gap=(
                    ARGS.terrain_global_minimum_score_gap
                ),
            ),
        )
        if ARGS.enable_global_terrain_relocalization
        else None
    )
    terrain_particle_likelihood_config = TerrainParticleLikelihoodConfig(
        bathymetry_measurement_std_m=terrain_prior_vertical_model_std_m,
        maximum_bathymetry_points=min(
            768, ARGS.terrain_match_maximum_points
        ),
        maximum_backscatter_points=min(
            768, ARGS.backscatter_match_maximum_points
        ),
    )
    global_cohort_tracker = (
        TerrainNavigationFilter(
            TerrainNavigationFilterConfig(
                maximum_cohorts=ARGS.terrain_global_maximum_cohorts,
                particles_per_cohort=(
                    ARGS.terrain_global_particles_per_cohort
                ),
                random_seed=ARGS.seed,
                resample_effective_sample_fraction=(
                    ARGS.terrain_global_particle_resample_ess_fraction
                ),
                resample_kernel_bandwidth=(
                    ARGS.terrain_global_particle_kernel_bandwidth
                ),
                minimum_total_evidence=(
                    ARGS.terrain_global_minimum_independent_evidence
                ),
                minimum_independent_evidence=(
                    ARGS.terrain_global_minimum_independent_evidence
                ),
                confirmation_probability=(
                    ARGS.terrain_global_cohort_confirmation_probability
                ),
                confirmation_odds_over_second=(
                    ARGS.terrain_global_cohort_confirmation_odds
                ),
            )
        )
        if ARGS.enable_global_cohort_tracking
        else None
    )
    terrain_backend_particle_tracker = (
        TerrainNavigationFilter(
            TerrainNavigationFilterConfig(
                maximum_cohorts=ARGS.terrain_global_maximum_cohorts,
                particles_per_cohort=(
                    ARGS.terrain_global_particles_per_cohort
                ),
                random_seed=ARGS.seed + 1,
                resample_effective_sample_fraction=(
                    ARGS.terrain_global_particle_resample_ess_fraction
                ),
                resample_kernel_bandwidth=(
                    ARGS.terrain_global_particle_kernel_bandwidth
                ),
                minimum_total_evidence=(
                    ARGS.terrain_global_minimum_independent_evidence
                ),
                minimum_independent_evidence=(
                    ARGS.terrain_global_minimum_independent_evidence
                ),
                confirmation_probability=(
                    ARGS.terrain_global_cohort_confirmation_probability
                ),
                confirmation_odds_over_second=(
                    ARGS.terrain_global_cohort_confirmation_odds
                ),
            )
        )
        if ARGS.enable_terrain_cohort_particle_backend
        else None
    )
    bbs3d_candidate_verifier = (
        BBS3DCandidateVerifier(
            world_model,
            BBS3DVerifierConfig(
                minimum_level_resolution_m=world_model.resolution_m,
                maximum_level=3,
                maximum_translation_m=(
                    ARGS.bbs3d_global_translation_window_m
                ),
                maximum_yaw_deg=ARGS.bbs3d_global_yaw_window_deg,
                minimum_score_fraction=(
                    ARGS.bbs3d_global_minimum_score_fraction
                ),
                minimum_source_points=min(
                    128, ARGS.terrain_match_maximum_points
                ),
                maximum_source_points=max(
                    8_192, 4 * ARGS.terrain_match_maximum_points
                ),
                timeout_milliseconds=ARGS.bbs3d_global_timeout_ms,
            ),
            require_native=True,
        )
        if ARGS.enable_bbs3d_global_verification
        else None
    )
    terrain_recovery = TerrainRecoveryController(
        enabled=terrain_recovery_matcher is not None,
        after_rejections=ARGS.terrain_recovery_after_rejections,
    )
    terrain_emergency_recovery = TerrainEmergencyRecoveryController(
        enabled=terrain_emergency_recovery_matcher is not None,
        recovery_translation_window_m=(
            150.0
            if terrain_emergency_recovery_config is None
            else terrain_emergency_recovery_config.maximum_translation_m
        ),
        confirmation_minimum_time_separation_s=(
            ARGS.terrain_match_window_pings / ARGS.ping_hz
        ),
    )
    terrain_patch_scans = deque(maxlen=ARGS.terrain_match_window_pings)
    terrain_patch_inertial_states = deque(
        maxlen=ARGS.terrain_match_window_pings
    )
    global_terrain_patch_scans = deque(
        maxlen=ARGS.terrain_global_window_pings
    )
    global_terrain_patch_inertial_states = deque(
        maxlen=ARGS.terrain_global_window_pings
    )
    last_terrain_retained_full_swath_pings = 0
    last_terrain_patch_raw_points = 0
    last_terrain_patch_reduced_points = 0
    last_terrain_patch_voxel_size_m = 0.0
    terrain_match_attempts = 0
    terrain_match_accepts = 0
    terrain_prior_accepts = 0
    terrain_match_mode_stats = {
        mode: {
            "invocations": 0,
            "accepts": 0,
            "rejections": 0,
            "rejection_reasons": Counter(),
            "rank_counts": Counter(),
            "prior_factor_attempts": 0,
            "prior_factor_accepts": 0,
            "prior_factor_rejections": 0,
            "wall_time_s": 0.0,
        }
        for mode in ("normal", "recovery", "emergency", "global")
    }
    terrain_backend_particle_stats = {
        "updates": 0,
        "raw_likelihood_updates": 0,
        "candidate_seed_updates": 0,
        "unseeded_tracking_updates": 0,
        "confirmed_updates": 0,
        "tracking_updates": 0,
        "no_fix_updates": 0,
        "factor_authorizations": 0,
        "factor_holds": 0,
        "nonindependent_factor_reuses_prevented": 0,
        "posterior_as_measurement_reuses_prevented": 0,
        "raw_likelihood_refinements": 0,
        "raw_likelihood_refinement_translation_sum_m": 0.0,
        "raw_likelihood_refinement_translation_max_m": 0.0,
        "raw_likelihood_refinement_abs_yaw_sum_deg": 0.0,
        "raw_likelihood_refinement_abs_yaw_max_deg": 0.0,
        "cohort_basin_refinements": 0,
        "compact_proposal_refinements": 0,
        "raw_likelihood_factor_attempts": 0,
        "raw_likelihood_factor_accepts": 0,
        "raw_likelihood_factor_rejections": 0,
    }
    terrain_backend_particle_records: list[dict] = []
    terrain_backend_particle_last_patch_end_observation_id = -1
    terrain_navigation_output_stats = {
        "policy": ARGS.terrain_particle_online_output,
        "particle_map_updates": 0,
        "ordinary_particle_map_updates": 0,
        "global_particle_map_updates": 0,
        "pose_graph_fallback_updates": 0,
        "last_source": "pose_graph",
        "last_particle_map_pose2": None,
    }
    global_relocalization_stats = {
        "invocations": 0,
        "accepted_proposals": 0,
        "rejected_proposals": 0,
        "proposal_rejection_reasons": Counter(),
        "bathymetry_refinement_accepts": 0,
        "bathymetry_refinement_rejections": 0,
        "bathymetry_refinement_rejection_reasons": Counter(),
        "bathymetry_refinement_seed_attempts": 0,
        "bathymetry_refinement_appearance_seeds_selected": 0,
        "bathymetry_refinement_bbs3d_seeds_selected": 0,
        "bathymetry_position_fallback_invocations": 0,
        "bathymetry_position_fallback_accepts": 0,
        "bathymetry_position_fallback_rejections": 0,
        "bathymetry_position_fallback_rejection_reasons": Counter(),
        "dual_modal_refinement_accepts": 0,
        "dual_modal_refinement_rejections": 0,
        "dual_modal_refinement_rejection_reasons": Counter(),
        "wall_time_s": 0.0,
        "last_best_score": float("nan"),
        "last_second_basin_score": float("nan"),
        "last_score_gap": float("nan"),
        "last_candidate_count": 0,
        "last_evaluated_yaws": 0,
        "last_finite_lattice_candidates": 0,
        "cohort_updates": 0,
        "cohort_confirmed_updates": 0,
        "cohort_tracking_updates": 0,
        "cohort_no_fix_updates": 0,
        "candidate_verification_invocations": 0,
        "candidate_verification_accepts": 0,
        "candidate_verification_rejections": 0,
        "bbs3d_verification_invocations": 0,
        "bbs3d_verification_accepts": 0,
        "bbs3d_verification_rejections": 0,
        "bbs3d_verification_rejection_reasons": Counter(),
        "bbs3d_verification_wall_time_s": 0.0,
        "last_bbs3d_score_fraction": float("nan"),
        "last_cohort_count": 0,
        "last_cohort_null_probability": float("nan"),
        "last_cohort_best_probability": float("nan"),
        "last_cohort_odds_over_second": float("nan"),
    }
    global_relocalization_records: list[dict] = []
    global_cohort_records: list[dict] = []
    terrain_accepted_invocation_stats = {
        "count": 0,
        "rmse_before_sum": 0.0,
        "rmse_after_sum": 0.0,
        "translation_correction_sum": 0.0,
        "yaw_correction_sum_deg": 0.0,
        "maximum_translation_correction_m": 0.0,
        "maximum_abs_yaw_correction_deg": 0.0,
    }
    terrain_prior_factor_stats = {
        "attempts": 0,
        "accepts": 0,
        "rejections": 0,
        "rejection_reasons": Counter(),
    }
    terrain_prefit_accepted_chi2: list[float] = []
    terrain_prefit_rejected_chi2: list[float] = []
    terrain_recovery_invocation_records: list[dict] = []
    terrain_match_truth_evaluation_records: list[dict] = []
    terrain_match_truth_evaluation_sums = {
        "count": 0,
        "seed_horizontal_squared_m2": 0.0,
        "matched_horizontal_squared_m2": 0.0,
        "seed_yaw_squared_deg2": 0.0,
        "matched_yaw_squared_deg2": 0.0,
    }
    terrain_consistency_records: list[dict] = []
    backscatter_match_records: list[dict] = []
    backscatter_match_stats = {
        "invocations": 0,
        "matcher_calls": 0,
        "accepts": 0,
        "rejections": 0,
        "rejection_reasons": Counter(),
        "wall_time_s": 0.0,
        "fused_fixes": 0,
        "automatic_ci_fixes": 0,
        "bathymetry_ci_weight_sum": 0.0,
        "last_bathymetry_ci_weight": None,
        "compatibility_vetoes": 0,
        "compatibility_veto_reasons": Counter(),
        "boundary_yaw_retries": 0,
        "boundary_yaw_retry_accepts": 0,
        "primary_rejection_reasons": Counter(),
    }
    last_backscatter_patch_raw_points = 0
    last_backscatter_patch_reduced_points = 0
    last_backscatter_patch_voxel_size_m = 0.0
    terrain_position_fallback_stats = {
        "invocations": 0,
        "accepts": 0,
        "rejections": 0,
        "rejection_reasons": Counter(),
        "wall_time_s": 0.0,
    }
    pending_initial_terrain_gauge = None
    terrain_gauge_initialization_stats = {
        "pending_patches": 0,
        "confirmed_pairs": 0,
        "confirmed_initializations": 0,
        "inconsistent_pairs": 0,
        "inconsistent_position_pairs": 0,
        "inconsistent_yaw_pairs": 0,
        "full_pose_confirmations": 0,
        "position_only_confirmations": 0,
        "unobservable_yaw_patches_held": 0,
        "overlapping_pairs": 0,
        "short_baseline_pairs": 0,
        "last_status": "inactive",
        "last_seed_baseline_m": None,
        "last_odometry_baseline_m": None,
        "last_matched_baseline_m": None,
        "last_position_fit_residual_m": None,
        "last_yaw_correction_deg": None,
    }
    last_terrain_match = None
    last_terrain_match_mode = None

    def record_terrain_match_invocation(
        mode,
        result,
        elapsed_s,
        *,
        truth_state,
    ):
        """Record matcher diagnostics; truth remains evaluation-only."""

        stats = terrain_match_mode_stats[mode]
        stats["invocations"] += 1
        stats["wall_time_s"] += float(elapsed_s)
        rank_label = (
            f"r{result.observable_rank}_"
            f"t{result.translation_observable_rank}_"
            f"yaw{int(result.yaw_observable)}"
        )
        stats["rank_counts"][rank_label] += 1

        truth_pose2 = np.array(
            [
                truth_state.position[0],
                truth_state.position[1],
                rpy_from_rotation(truth_state.rotation)[2],
            ],
            dtype=np.float64,
        )
        seed_pose2 = np.array(
            [
                result.base_position[0],
                result.base_position[1],
                rpy_from_rotation(result.base_rotation)[2],
            ],
            dtype=np.float64,
        )
        matched_pose2 = np.asarray(result.matched_pose2, dtype=np.float64)

        def yaw_error_deg(pose2):
            delta = float(pose2[2] - truth_pose2[2])
            return float(
                np.rad2deg(math.atan2(math.sin(delta), math.cos(delta)))
            )

        seed_horizontal_error_m = float(
            np.linalg.norm(seed_pose2[:2] - truth_pose2[:2])
        )
        matched_horizontal_error_m = float(
            np.linalg.norm(matched_pose2[:2] - truth_pose2[:2])
        )
        seed_yaw_error_deg = yaw_error_deg(seed_pose2)
        matched_yaw_error_deg = yaw_error_deg(matched_pose2)
        terrain_match_truth_evaluation_records.append(
            {
                "timestamp_s": float(result.state_timestamp),
                "mode": mode,
                "matcher_accepted": bool(result.accepted),
                "rejection_reason": result.rejection_reason,
                "truth_pose2": truth_pose2.tolist(),
                "seed_pose2": seed_pose2.tolist(),
                "matched_pose2": matched_pose2.tolist(),
                "seed_horizontal_error_m": seed_horizontal_error_m,
                "matched_horizontal_error_m": matched_horizontal_error_m,
                "seed_yaw_error_deg": seed_yaw_error_deg,
                "matched_yaw_error_deg": matched_yaw_error_deg,
                "local_acoustic_odometry_pose2": [
                    float(slam.state.position[0]),
                    float(slam.state.position[1]),
                    float(rpy_from_rotation(slam.state.rotation)[2]),
                ],
                "dead_reckoning_pose2": [
                    float(dead_reckoning.state.position[0]),
                    float(dead_reckoning.state.position[1]),
                    float(rpy_from_rotation(dead_reckoning.state.rotation)[2]),
                ],
                "reported_pose_std": np.sqrt(
                    np.maximum(np.diag(result.covariance), 0.0)
                ).tolist(),
            }
        )
        if result.accepted:
            evaluation = terrain_match_truth_evaluation_sums
            evaluation["count"] += 1
            evaluation["seed_horizontal_squared_m2"] += (
                seed_horizontal_error_m**2
            )
            evaluation["matched_horizontal_squared_m2"] += (
                matched_horizontal_error_m**2
            )
            evaluation["seed_yaw_squared_deg2"] += seed_yaw_error_deg**2
            evaluation["matched_yaw_squared_deg2"] += (
                matched_yaw_error_deg**2
            )

        if not result.accepted:
            stats["rejections"] += 1
            stats["rejection_reasons"][
                result.rejection_reason or "unspecified"
            ] += 1
            return
        stats["accepts"] += 1
        aggregate = terrain_accepted_invocation_stats
        aggregate["count"] += 1
        aggregate["rmse_before_sum"] += result.rmse_before
        aggregate["rmse_after_sum"] += result.rmse_after
        aggregate["translation_correction_sum"] += (
            result.translation_correction_m
        )
        aggregate["yaw_correction_sum_deg"] += result.yaw_correction_deg
        aggregate["maximum_translation_correction_m"] = max(
            aggregate["maximum_translation_correction_m"],
            result.translation_correction_m,
        )
        aggregate["maximum_abs_yaw_correction_deg"] = max(
            aggregate["maximum_abs_yaw_correction_deg"],
            abs(result.yaw_correction_deg),
        )

    def refine_terrain_fix_with_backscatter(
        terrain_result,
        patch_points_body,
        patch_intensities,
        *,
        truth_state,
        use_global_refinement=False,
    ):
        """Return a covariance-weighted bathymetry/backscatter absolute fix."""

        selected_backscatter_matcher = (
            global_backscatter_refinement_matcher
            if use_global_refinement
            else backscatter_matcher
        )
        selected_boundary_retry_matcher = (
            None
            if use_global_refinement
            else backscatter_boundary_yaw_retry_matcher
        )
        if selected_backscatter_matcher is None:
            return (
                terrain_result.matched_pose2,
                terrain_result.information,
                bool(terrain_result.yaw_observable),
                True,
                False,
                False,
                None,
            )
        bathymetric_pose2 = np.asarray(
            terrain_result.matched_pose2, dtype=np.float64
        )
        backscatter_seed = InertialState(
            timestamp=float(terrain_result.state_timestamp),
            rotation=np.asarray(
                terrain_result.base_rotation, dtype=np.float64
            ).copy(),
            position=np.asarray(
                terrain_result.base_position, dtype=np.float64
            ).copy(),
            velocity=np.zeros(3, dtype=np.float64),
            covariance=np.eye(15, dtype=np.float64),
        )
        base_yaw = float(
            rpy_from_rotation(backscatter_seed.rotation)[2]
        )
        yaw_delta = math.atan2(
            math.sin(bathymetric_pose2[2] - base_yaw),
            math.cos(bathymetric_pose2[2] - base_yaw),
        )
        cosine = math.cos(yaw_delta)
        sine = math.sin(yaw_delta)
        yaw_rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        backscatter_seed.rotation = yaw_rotation @ backscatter_seed.rotation
        backscatter_seed.position[:2] = bathymetric_pose2[:2]

        match_started = time.perf_counter()
        primary_backscatter_result = selected_backscatter_matcher.match(
            patch_points_body, patch_intensities, backscatter_seed
        )
        elapsed_s = time.perf_counter() - match_started
        stats = backscatter_match_stats
        stats["invocations"] += 1
        stats["matcher_calls"] += 1
        stats["wall_time_s"] += elapsed_s
        backscatter_result = primary_backscatter_result
        boundary_retry_attempted = False
        if not primary_backscatter_result.accepted:
            stats["primary_rejection_reasons"][
                primary_backscatter_result.rejection_reason or "unspecified"
            ] += 1
        primary_config = selected_backscatter_matcher.config
        primary_translation_boundary = bool(
            np.linalg.norm(primary_backscatter_result.correction[:2])
            >= primary_config.maximum_translation_m
            - 0.5 * primary_config.refinement_translation_step_m
        )
        primary_yaw_boundary = bool(
            abs(np.rad2deg(primary_backscatter_result.correction[2]))
            >= primary_config.maximum_yaw_deg
            - 0.5 * primary_config.refinement_yaw_step_deg
        )
        if (
            selected_boundary_retry_matcher is not None
            and primary_backscatter_result.rejection_reason
            == "search_boundary_reached"
            and primary_yaw_boundary
            and not primary_translation_boundary
        ):
            boundary_retry_attempted = True
            stats["boundary_yaw_retries"] += 1
            match_started = time.perf_counter()
            retry_result = selected_boundary_retry_matcher.match(
                patch_points_body, patch_intensities, backscatter_seed
            )
            retry_elapsed_s = time.perf_counter() - match_started
            stats["matcher_calls"] += 1
            stats["wall_time_s"] += retry_elapsed_s
            backscatter_result = retry_result
            if retry_result.accepted:
                stats["boundary_yaw_retry_accepts"] += 1
        if backscatter_result.accepted:
            stats["accepts"] += 1
        else:
            stats["rejections"] += 1
            stats["rejection_reasons"][
                backscatter_result.rejection_reason or "unspecified"
            ] += 1

        fused_pose2 = bathymetric_pose2.copy()
        fused_information = np.asarray(
            terrain_result.information, dtype=np.float64
        ).copy()
        information_includes_model_floors = False
        if backscatter_result.accepted:
            # Apply low-pass bathymetry model error before covariance
            # intersection. Applying it to the already fused result would
            # incorrectly downweight the separately observed side-scan yaw a
            # second time.
            fused_information = terrain_prior_information(
                terrain_result,
                "full",
                base_information=fused_information,
                apply_model_floors=True,
            )
            information_includes_model_floors = True
            intensity_information = np.asarray(
                backscatter_result.information, dtype=np.float64
            )
            # Bathymetry and side-scan share the same vehicle trajectory,
            # acoustic geometry and source survey, so their cross-covariance
            # is unknown and independence must not be assumed. Equal-weight
            # covariance intersection retains the complementary optimum while
            # avoiding the over-confident information sum.
            configured_weight = float(
                ARGS.backscatter_ci_bathymetry_weight
            )
            (
                fused_pose2,
                fused_information,
                bathymetric_weight,
            ) = covariance_intersection_pose2(
                bathymetric_pose2,
                fused_information,
                backscatter_result.matched_pose2,
                intensity_information,
                first_weight=(
                    None if configured_weight < 0.0 else configured_weight
                ),
            )
            stats["fused_fixes"] += 1
            stats["automatic_ci_fixes"] += int(configured_weight < 0.0)
            stats["bathymetry_ci_weight_sum"] += bathymetric_weight
            stats["last_bathymetry_ci_weight"] = bathymetric_weight

        # Once the global gauge is established, a correction exceeding one
        # configured map-scale cell needs cross-modal support.  A multi-modal
        # side-scan likelihood does not identify which absolute basin is
        # correct, so retain both matcher outputs for diagnosis but do not add
        # the bathymetric hypothesis as a world-frame factor.  This is a veto,
        # not a substitute measurement: no ambiguous texture candidate is
        # injected into the estimator.
        bathymetric_correction_m = float(
            terrain_result.translation_correction_m
        )
        compatibility_threshold_m = float(
            ARGS.cross_modal_large_correction_m
        )
        cross_modal_compatible = not bool(
            terrain_prior_factor_stats["accepts"] > 0
            and compatibility_threshold_m > 0.0
            and bathymetric_correction_m > compatibility_threshold_m
            and not backscatter_result.accepted
            and backscatter_result.rejection_reason
            == "ambiguous_backscatter_match"
        )
        compatibility_veto_reason = None
        if not cross_modal_compatible:
            compatibility_veto_reason = (
                "large_bathymetric_correction_with_ambiguous_backscatter"
            )
            stats["compatibility_vetoes"] += 1
            stats["compatibility_veto_reasons"][
                compatibility_veto_reason
            ] += 1

        truth_pose2 = np.array(
            [
                truth_state.position[0],
                truth_state.position[1],
                rpy_from_rotation(truth_state.rotation)[2],
            ],
            dtype=np.float64,
        )

        def errors(pose2):
            yaw_error = math.atan2(
                math.sin(pose2[2] - truth_pose2[2]),
                math.cos(pose2[2] - truth_pose2[2]),
            )
            return (
                float(np.linalg.norm(pose2[:2] - truth_pose2[:2])),
                float(np.rad2deg(yaw_error)),
            )

        bathymetric_errors = errors(bathymetric_pose2)
        backscatter_errors = errors(backscatter_result.matched_pose2)
        fused_errors = errors(fused_pose2)
        backscatter_match_records.append(
            {
                "refinement_scope": (
                    "global_relocalization"
                    if use_global_refinement
                    else "ordinary_tracking"
                ),
                "timestamp_s": float(terrain_result.state_timestamp),
                "primary_accepted": bool(
                    primary_backscatter_result.accepted
                ),
                "primary_rejection_reason": (
                    primary_backscatter_result.rejection_reason
                ),
                "primary_yaw_boundary": primary_yaw_boundary,
                "primary_translation_boundary": (
                    primary_translation_boundary
                ),
                "boundary_yaw_retry_attempted": boundary_retry_attempted,
                "accepted": bool(backscatter_result.accepted),
                "rejection_reason": backscatter_result.rejection_reason,
                "correlation_before": float(
                    backscatter_result.correlation_before
                ),
                "correlation_after": float(
                    backscatter_result.correlation_after
                ),
                "second_basin_correlation": float(
                    backscatter_result.second_basin_correlation
                ),
                "basin_correlation_gap": float(
                    backscatter_result.basin_correlation_gap
                ),
                "correspondences": int(backscatter_result.correspondences),
                "valid_fraction": float(backscatter_result.valid_fraction),
                "observation_texture_std": float(
                    backscatter_result.observation_texture_std
                ),
                "map_texture_std": float(
                    backscatter_result.map_texture_std
                ),
                "fine_correlation_after": float(
                    backscatter_result.fine_correlation_after
                ),
                "scale_space_correlation_after": float(
                    backscatter_result.scale_space_correlation_after
                ),
                "context_consistency_after": float(
                    backscatter_result.context_consistency_after
                ),
                "valid_contexts_after": int(
                    backscatter_result.valid_contexts_after
                ),
                "correction_from_bathymetry": (
                    backscatter_result.correction.tolist()
                ),
                "bathymetric_translation_correction_m": (
                    bathymetric_correction_m
                ),
                "cross_modal_compatible": bool(cross_modal_compatible),
                "compatibility_veto_reason": compatibility_veto_reason,
                "bathymetric_horizontal_error_m": bathymetric_errors[0],
                "bathymetric_yaw_error_deg": bathymetric_errors[1],
                "backscatter_horizontal_error_m": backscatter_errors[0],
                "backscatter_yaw_error_deg": backscatter_errors[1],
                "fused_horizontal_error_m": fused_errors[0],
                "fused_yaw_error_deg": fused_errors[1],
                "bathymetry_ci_weight": (
                    None
                    if not backscatter_result.accepted
                    else float(bathymetric_weight)
                ),
                "automatic_ci_weight": bool(
                    backscatter_result.accepted
                    and ARGS.backscatter_ci_bathymetry_weight < 0.0
                ),
                "truth_evaluation_only": True,
                "truth_used_by_matcher_or_fusion": False,
                "wall_time_s": elapsed_s,
            }
        )
        return (
            fused_pose2,
            fused_information,
            bool(
                terrain_result.yaw_observable
                or backscatter_result.accepted
            ),
            bool(cross_modal_compatible),
            bool(information_includes_model_floors),
            bool(backscatter_result.accepted),
            backscatter_result,
        )

    def build_global_relocalization_patch():
        """Build the long acoustic-inertial deskewed strip used after loss."""

        scans = list(global_terrain_patch_scans)
        inertial_states = list(global_terrain_patch_inertial_states)
        if (
            global_terrain_relocalizer is None
            or len(scans) != ARGS.terrain_global_window_pings
            or len(inertial_states) != len(scans)
        ):
            return None
        motion_states = [scan[0].copy() for scan in scans]
        if ARGS.terrain_patch_motion_source == "acoustic-fixed-lag":
            motion_states = fixed_lag_fused_deskew_states(
                motion_states, inertial_states
            )
        reference_state = motion_states[-1].copy()
        terrain_chunks = [
            motion_compensate_to_body(
                scan[1],
                acquisition_state,
                reference_state,
                sensor_rotation_body=slam.filter.sensor_rotation_body,
                sensor_translation_body=slam.filter.sensor_translation_body,
            )
            for acquisition_state, scan in zip(motion_states, scans)
            if scan[1].size
        ]
        if not terrain_chunks:
            return None
        terrain_full = np.vstack(terrain_chunks)
        terrain_points, terrain_voxel_size_m = voxel_average_terrain_patch(
            terrain_full,
            max(8_192, 4 * ARGS.terrain_match_maximum_points),
            minimum_voxel_size_m=world_model.resolution_m,
        )

        backscatter_point_chunks = []
        backscatter_value_chunks = []
        for acquisition_state, scan in zip(motion_states, scans):
            if scan[2].size == 0:
                continue
            backscatter_point_chunks.append(
                motion_compensate_to_body(
                    scan[2],
                    acquisition_state,
                    reference_state,
                    sensor_rotation_body=slam.filter.sensor_rotation_body,
                    sensor_translation_body=slam.filter.sensor_translation_body,
                )
            )
            backscatter_value_chunks.append(
                np.asarray(scan[3], dtype=np.float64).reshape(-1)
            )
        if not backscatter_point_chunks:
            return None
        backscatter_full = np.vstack(backscatter_point_chunks)
        backscatter_values_full = np.concatenate(backscatter_value_chunks)
        (
            backscatter_points,
            backscatter_values,
            backscatter_voxel_size_m,
        ) = voxel_average_backscatter_patch(
            backscatter_full,
            backscatter_values_full,
            max(8_192, 4 * ARGS.backscatter_match_maximum_points),
            minimum_voxel_size_m=world_model.resolution_m,
        )
        if backscatter_points.size == 0:
            return None
        return {
            "reference_state": reference_state,
            "terrain_points_body": terrain_points,
            "terrain_raw_points": int(terrain_full.shape[0]),
            "terrain_voxel_size_m": float(terrain_voxel_size_m),
            "backscatter_points_body": backscatter_points,
            "backscatter_values": backscatter_values,
            "backscatter_raw_points": int(backscatter_full.shape[0]),
            "backscatter_voxel_size_m": float(backscatter_voxel_size_m),
            "patch_start_timestamp": float(scans[0][4]),
            "patch_end_timestamp": float(scans[-1][4]),
            "patch_start_observation_id": int(scans[0][5]),
            "patch_end_observation_id": int(scans[-1][5]),
        }

    def verify_global_candidate(candidate, global_patch, *, truth_state):
        """Run 3D-BBS, independent 5 m MBES and side-scan checks.

        The whole-map appearance stage is intentionally permissive when the
        cohort tracker is enabled.  Every separated basin is therefore
        subjected first to full 3-D occupancy verification and then to the
        same metric bathymetry and backscatter registration before it may
        contribute positive persistent evidence.
        """

        global_stats = global_relocalization_stats
        candidate_pose2 = np.asarray(candidate.pose2, dtype=np.float64)
        bbs3d_result = None
        bbs3d_wall_time_s = 0.0
        if bbs3d_candidate_verifier is not None:
            bbs3d_started = time.perf_counter()
            bbs3d_result = bbs3d_candidate_verifier.verify(
                global_patch["terrain_points_body"],
                global_patch["reference_state"],
                candidate_pose2,
            )
            bbs3d_wall_time_s = time.perf_counter() - bbs3d_started
            global_stats["bbs3d_verification_invocations"] += 1
            global_stats["bbs3d_verification_wall_time_s"] += (
                bbs3d_wall_time_s
            )
            global_stats["last_bbs3d_score_fraction"] = (
                bbs3d_result.score_fraction
            )
            if not bbs3d_result.accepted:
                global_stats["bbs3d_verification_rejections"] += 1
                global_stats["bbs3d_verification_rejection_reasons"][
                    bbs3d_result.rejection_reason or "unspecified"
                ] += 1
                return {
                    "terrain_result": None,
                    "refine_elapsed_s": float(bbs3d_wall_time_s),
                    "accepted": False,
                    "matched_pose2": np.asarray(
                        bbs3d_result.matched_pose2, dtype=np.float64
                    ),
                    "information": np.zeros((3, 3), dtype=np.float64),
                    "information_has_floors": False,
                    "yaw_observable": False,
                    "observable_rank": 0,
                    "translation_observable_rank": 0,
                    "backscatter_result": None,
                    "bbs3d_result": bbs3d_result,
                    "terrain_seed_source": None,
                    "terrain_seed_attempts": [],
                    "geometry_score": float(bbs3d_result.score_fraction),
                    "backscatter_observability": 0.0,
                }
            global_stats["bbs3d_verification_accepts"] += 1

        # 3D-BBS uses quantized occupancy.  Several translations can therefore
        # attain the same coarse score, especially over a broad sloping patch.
        # It remains a required geometric veto, but its argmax must not erase
        # an already good appearance proposal.  Refine from both independent
        # seeds and select only by matcher diagnostics (never simulator truth).
        metric_seeds = [("appearance", candidate_pose2.copy())]
        if bbs3d_result is not None:
            bbs3d_pose2 = np.asarray(
                bbs3d_result.matched_pose2, dtype=np.float64
            )
            seed_translation_separation = float(
                np.linalg.norm(bbs3d_pose2[:2] - candidate_pose2[:2])
            )
            seed_yaw_separation_deg = abs(
                math.degrees(
                    math.atan2(
                        math.sin(bbs3d_pose2[2] - candidate_pose2[2]),
                        math.cos(bbs3d_pose2[2] - candidate_pose2[2]),
                    )
                )
            )
            if (
                seed_translation_separation
                > 0.25 * world_model.resolution_m
                or seed_yaw_separation_deg > 0.10
            ):
                metric_seeds.append(("bbs3d", bbs3d_pose2))

        refinement_attempts = []
        refinement_wall_time_s = 0.0
        for seed_source, seed_pose2 in metric_seeds:
            proposal_state = global_patch["reference_state"].copy()
            proposal_rpy = rpy_from_rotation(proposal_state.rotation)
            proposal_state.rotation = rotation_from_rpy(
                float(proposal_rpy[0]),
                float(proposal_rpy[1]),
                float(seed_pose2[2]),
            )
            proposal_state.position[:2] = seed_pose2[:2]

            refine_started = time.perf_counter()
            seed_result = terrain_matcher.match(
                global_patch["terrain_points_body"],
                proposal_state,
                points_are_body_frame=True,
            )
            seed_elapsed_s = time.perf_counter() - refine_started
            refinement_wall_time_s += seed_elapsed_s
            global_stats["bathymetry_refinement_seed_attempts"] += 1
            record_terrain_match_invocation(
                "global",
                seed_result,
                seed_elapsed_s,
                truth_state=truth_state,
            )
            if (
                not seed_result.accepted
                and seed_result.rejection_reason == "yaw_unobservable"
                and terrain_position_fallback_matcher is not None
            ):
                global_stats[
                    "bathymetry_position_fallback_invocations"
                ] += 1
                fallback_started = time.perf_counter()
                seed_result = terrain_position_fallback_matcher.match(
                    global_patch["terrain_points_body"],
                    proposal_state,
                    points_are_body_frame=True,
                )
                fallback_elapsed_s = time.perf_counter() - fallback_started
                refinement_wall_time_s += fallback_elapsed_s
                record_terrain_match_invocation(
                    "global",
                    seed_result,
                    fallback_elapsed_s,
                    truth_state=truth_state,
                )
                if seed_result.accepted:
                    global_stats[
                        "bathymetry_position_fallback_accepts"
                    ] += 1
                else:
                    global_stats[
                        "bathymetry_position_fallback_rejections"
                    ] += 1
                    global_stats[
                        "bathymetry_position_fallback_rejection_reasons"
                    ][seed_result.rejection_reason or "unspecified"] += 1
            refinement_attempts.append((seed_source, seed_pose2, seed_result))

        def refinement_quality(attempt):
            _source, _seed, match_result = attempt
            likelihood_ratio = float(
                match_result.basin_log_likelihood_ratio
            )
            if not np.isfinite(likelihood_ratio):
                likelihood_ratio = 1.0e300 if likelihood_ratio > 0.0 else -1.0e300
            rmse = float(match_result.rmse_after)
            if not np.isfinite(rmse):
                rmse = 1.0e300
            return (
                int(match_result.accepted),
                int(not match_result.reached_search_boundary),
                int(match_result.observable_rank),
                likelihood_ratio,
                -rmse,
                -float(match_result.translation_correction_m),
            )

        selected_attempt = max(
            refinement_attempts, key=refinement_quality
        )
        selected_seed_source, _selected_seed_pose2, refined_result = (
            selected_attempt
        )
        global_stats[
            "bathymetry_refinement_"
            + selected_seed_source
            + "_seeds_selected"
        ] += 1
        refine_elapsed_s = bbs3d_wall_time_s + refinement_wall_time_s
        public_refinement_attempts = [
            {
                "seed_source": seed_source,
                "seed_pose2": seed_pose2.tolist(),
                "accepted": bool(match_result.accepted),
                "rejection_reason": match_result.rejection_reason,
                "matched_pose2": match_result.matched_pose2.tolist(),
                "rmse_after_m": _finite_float_or_none(
                    match_result.rmse_after
                ),
                "observable_rank": int(match_result.observable_rank),
                "reached_search_boundary": bool(
                    match_result.reached_search_boundary
                ),
            }
            for seed_source, seed_pose2, match_result in refinement_attempts
        ]

        result = {
            "terrain_result": refined_result,
            "refine_elapsed_s": float(refine_elapsed_s),
            "accepted": False,
            "matched_pose2": np.asarray(
                refined_result.matched_pose2, dtype=np.float64
            ),
            "information": np.asarray(
                refined_result.information, dtype=np.float64
            ),
            "information_has_floors": False,
            "yaw_observable": bool(refined_result.yaw_observable),
            "observable_rank": int(refined_result.observable_rank),
            "translation_observable_rank": int(
                refined_result.translation_observable_rank
            ),
            "backscatter_result": None,
            "bbs3d_result": bbs3d_result,
            "terrain_seed_source": selected_seed_source,
            "terrain_seed_attempts": public_refinement_attempts,
            "geometry_score": 0.0,
            "backscatter_observability": 0.0,
        }
        if not refined_result.accepted:
            global_stats["bathymetry_refinement_rejections"] += 1
            global_stats["bathymetry_refinement_rejection_reasons"][
                refined_result.rejection_reason or "unspecified"
            ] += 1
            return result

        global_stats["bathymetry_refinement_accepts"] += 1
        (
            refined_pose2,
            refined_information,
            refined_yaw_observable,
            refined_compatible,
            refined_information_has_floors,
            refined_backscatter_accepted,
            refined_backscatter_result,
        ) = refine_terrain_fix_with_backscatter(
            refined_result,
            global_patch["backscatter_points_body"],
            global_patch["backscatter_values"],
            truth_state=truth_state,
            use_global_refinement=True,
        )
        dual_modal_accepted = bool(
            refined_compatible and refined_backscatter_accepted
        )
        if dual_modal_accepted:
            global_stats["dual_modal_refinement_accepts"] += 1
        else:
            global_stats["dual_modal_refinement_rejections"] += 1
            rejection_reason = (
                "backscatter_refinement_rejected"
                if not refined_backscatter_accepted
                else "cross_modal_incompatible"
            )
            global_stats["dual_modal_refinement_rejection_reasons"][
                rejection_reason
            ] += 1

        inlier_ratio = float(
            refined_result.inliers / max(refined_result.correspondences, 1)
        )
        chi_square_excess = max(
            0.0, float(refined_result.reduced_chi_square) - 1.0
        )
        geometry_score = float(
            np.clip(
                0.5 * inlier_ratio + 0.5 * math.exp(-0.5 * chi_square_excess),
                0.0,
                1.0,
            )
        )
        if bbs3d_result is not None:
            # Both values derive from MBES geometry, so they are not treated
            # as independent likelihood channels.  Their geometric mean is a
            # conservative agreement score: either verifier can lower it.
            geometry_score = float(
                math.sqrt(
                    max(0.0, geometry_score)
                    * max(0.0, bbs3d_result.score_fraction)
                )
            )
        backscatter_observability = 0.0
        if refined_backscatter_result is not None:
            texture_scale = max(
                5.0
                * global_backscatter_refinement_matcher.config.minimum_texture_std,
                1.0e-12,
            )
            backscatter_observability = float(
                np.clip(
                    refined_backscatter_result.observation_texture_std
                    / texture_scale,
                    0.0,
                    1.0,
                )
                * np.clip(
                    refined_backscatter_result.valid_fraction, 0.0, 1.0
                )
            )
        result.update(
            {
                "accepted": dual_modal_accepted,
                "matched_pose2": np.asarray(
                    refined_pose2, dtype=np.float64
                ),
                "information": np.asarray(
                    refined_information, dtype=np.float64
                ),
                "information_has_floors": bool(
                    refined_information_has_floors
                ),
                "yaw_observable": bool(refined_yaw_observable),
                "observable_rank": (
                    3
                    if (
                        refined_backscatter_accepted
                        and refined_result.translation_observable_rank == 2
                    )
                    else int(refined_result.observable_rank)
                ),
                "translation_observable_rank": int(
                    refined_result.translation_observable_rank
                ),
                "backscatter_result": refined_backscatter_result,
                "geometry_score": geometry_score,
                "backscatter_observability": backscatter_observability,
            }
        )
        return result

    def terrain_prior_information(
        result,
        component_mode="full",
        base_information=None,
        *,
        apply_model_floors=True,
    ):
        """Return a map-resolution-aware graph information matrix."""

        source_information = (
            np.asarray(result.information, dtype=np.float64)
            if base_information is None
            else np.asarray(base_information, dtype=np.float64).reshape(3, 3)
        )
        source_covariance = np.linalg.pinv(
            source_information, rcond=1.0e-12, hermitian=True
        )
        floor_variance = (
            ARGS.terrain_prior_horizontal_std_floor_m**2
            if apply_model_floors
            else 0.0
        )
        yaw_floor_variance = (
            np.deg2rad(ARGS.terrain_prior_yaw_std_floor_deg) ** 2
            if apply_model_floors
            else 0.0
        )
        if ARGS.terrain_position_only_priors or component_mode == "position":
            # Marginalize the terrain yaw rather than taking a 3x3 information
            # sub-block, then add survey/model error in covariance space.
            covariance_xy = np.asarray(
                source_covariance[:2, :2], dtype=np.float64
            ).copy()
            marginal_information_xy = np.linalg.pinv(
                covariance_xy, rcond=1.0e-12, hermitian=True
            )
            information = np.zeros((3, 3), dtype=np.float64)
            information[:2, :2] = information_with_covariance_floor(
                marginal_information_xy,
                floor_variance * np.eye(2, dtype=np.float64),
            )
            return information
        if component_mode != "full":
            raise ValueError(f"unsupported terrain-prior mode: {component_mode}")
        if floor_variance <= 0.0 and yaw_floor_variance <= 0.0:
            return source_information
        floor_covariance = np.diag(
            [floor_variance, floor_variance, yaw_floor_variance]
        )
        return information_with_covariance_floor(
            source_information, floor_covariance
        )

    def prepare_initial_terrain_gauge(
        result,
        *,
        matched_pose2_override=None,
        yaw_observable_override=None,
        odometry_pose2,
        initializing,
        patch_start_timestamp,
        patch_end_timestamp,
        patch_start_observation_id,
        patch_end_observation_id,
    ):
        """Confirm an initial SE(2) gauge without simulator truth."""

        nonlocal pending_initial_terrain_gauge
        seed_pose2 = np.array(
            [
                result.base_position[0],
                result.base_position[1],
                rpy_from_rotation(result.base_rotation)[2],
            ],
            dtype=np.float64,
        )
        odometry_pose2 = np.asarray(
            odometry_pose2, dtype=np.float64
        ).reshape(3).copy()
        matched_pose2 = np.asarray(
            result.matched_pose2
            if matched_pose2_override is None
            else matched_pose2_override,
            dtype=np.float64,
        ).reshape(3)
        measurement_yaw_observable = bool(
            result.yaw_observable
            if yaw_observable_override is None
            else yaw_observable_override
        )
        stats = terrain_gauge_initialization_stats
        if initializing and not measurement_yaw_observable:
            # A position-only absolute match cannot establish the global yaw
            # gauge. Preserve any earlier rank-three candidate and wait for a
            # full independent observation instead of inventing heading.
            stats["unobservable_yaw_patches_held"] += 1
            stats["last_status"] = "waiting_for_yaw_observable_patch"
            return None, True, None
        if ARGS.terrain_gauge_confirmation_patches == 1:
            stats["confirmed_pairs"] += 1
            if initializing:
                stats["confirmed_initializations"] += 1
            stats["last_status"] = (
                "single_patch_initialized"
                if initializing
                else "single_patch_prior_ready"
            )
            stats["last_yaw_correction_deg"] = float(
                np.rad2deg(
                    math.atan2(
                        math.sin(matched_pose2[2] - seed_pose2[2]),
                        math.cos(matched_pose2[2] - seed_pose2[2]),
                    )
                )
            )
            return matched_pose2.copy(), False, "full"

        current = {
            "seed_pose2": seed_pose2.copy(),
            "odometry_pose2": odometry_pose2.copy(),
            "matched_pose2": matched_pose2.copy(),
            "patch_start_timestamp": float(patch_start_timestamp),
            "patch_end_timestamp": float(patch_end_timestamp),
            "patch_start_observation_id": int(patch_start_observation_id),
            "patch_end_observation_id": int(patch_end_observation_id),
        }
        if pending_initial_terrain_gauge is None:
            pending_initial_terrain_gauge = current
            stats["pending_patches"] += 1
            stats["last_status"] = "pending_first_disjoint_patch"
            return None, True, None

        first = pending_initial_terrain_gauge
        windows_disjoint = bool(
            current["patch_start_timestamp"] > first["patch_end_timestamp"]
            and current["patch_start_observation_id"]
            > first["patch_end_observation_id"]
        )
        if not windows_disjoint:
            stats["overlapping_pairs"] += 1
            stats["last_status"] = "waiting_for_disjoint_patch"
            return None, True, None

        first_odometry = first["odometry_pose2"]
        first_matched = first["matched_pose2"]
        odometry_baseline = odometry_pose2[:2] - first_odometry[:2]
        matched_baseline = matched_pose2[:2] - first_matched[:2]
        odometry_baseline_m = float(np.linalg.norm(odometry_baseline))
        matched_baseline_m = float(np.linalg.norm(matched_baseline))
        # Retain the older metric name for readers of previous result files,
        # but make the source explicit in the new field.  The global matcher
        # seed is graph-corrected and is therefore not a continuous odometry
        # frame; the independent local acoustic trajectory is.
        stats["last_seed_baseline_m"] = odometry_baseline_m
        stats["last_odometry_baseline_m"] = odometry_baseline_m
        stats["last_matched_baseline_m"] = matched_baseline_m
        if min(odometry_baseline_m, matched_baseline_m) < (
            ARGS.terrain_gauge_minimum_baseline_m
        ):
            stats["short_baseline_pairs"] += 1
            stats["last_status"] = "waiting_for_longer_baseline"
            return None, True, None

        # Propagate the first independent map fix by the intervening local
        # acoustic-odometry trajectory. Agreement with the second fix validates
        # the basin; circular yaw averaging suppresses the systematic
        # over-rotation of one low-pass strip.
        delta_world = odometry_pose2[:2] - first_odometry[:2]
        first_seed_cosine = math.cos(float(first_odometry[2]))
        first_seed_sine = math.sin(float(first_odometry[2]))
        delta_local = np.array(
            [
                first_seed_cosine * delta_world[0]
                + first_seed_sine * delta_world[1],
                -first_seed_sine * delta_world[0]
                + first_seed_cosine * delta_world[1],
            ],
            dtype=np.float64,
        )
        first_match_cosine = math.cos(float(first_matched[2]))
        first_match_sine = math.sin(float(first_matched[2]))
        propagated_first = np.empty(3, dtype=np.float64)
        propagated_first[:2] = first_matched[:2] + np.array(
            [
                first_match_cosine * delta_local[0]
                - first_match_sine * delta_local[1],
                first_match_sine * delta_local[0]
                + first_match_cosine * delta_local[1],
            ]
        )
        seed_yaw_increment = math.atan2(
            math.sin(odometry_pose2[2] - first_odometry[2]),
            math.cos(odometry_pose2[2] - first_odometry[2]),
        )
        propagated_first[2] = math.atan2(
            math.sin(first_matched[2] + seed_yaw_increment),
            math.cos(first_matched[2] + seed_yaw_increment),
        )
        position_disagreement = float(
            np.linalg.norm(propagated_first[:2] - matched_pose2[:2])
        )
        yaw_disagreement = math.atan2(
            math.sin(matched_pose2[2] - propagated_first[2]),
            math.cos(matched_pose2[2] - propagated_first[2]),
        )
        yaw_disagreement_deg = float(abs(np.rad2deg(yaw_disagreement)))
        stats["last_position_fit_residual_m"] = position_disagreement
        position_consistent = bool(
            position_disagreement
            <= ARGS.terrain_gauge_maximum_position_residual_m
        )
        yaw_consistent = bool(
            measurement_yaw_observable
            and yaw_disagreement_deg
            <= ARGS.terrain_gauge_maximum_yaw_residual_deg
        )
        stats["inconsistent_position_pairs"] += int(not position_consistent)
        stats["inconsistent_yaw_pairs"] += int(
            measurement_yaw_observable and not yaw_consistent
        )
        consistency_record = {
            "timestamp_s": float(result.state_timestamp),
            "initializing": bool(initializing),
            "reference_timestamp_s": float(first["patch_end_timestamp"]),
            "odometry_baseline_m": odometry_baseline_m,
            "position_innovation_m": position_disagreement,
            "yaw_innovation_deg": yaw_disagreement_deg,
            "position_consistent": position_consistent,
            "yaw_consistent": yaw_consistent,
            "measurement_yaw_observable": measurement_yaw_observable,
            "propagated_reference_pose2": propagated_first.tolist(),
            "current_matched_pose2": matched_pose2.tolist(),
        }
        terrain_consistency_records.append(consistency_record)
        if initializing and not (position_consistent and yaw_consistent):
            # Before initialization neither hypothesis is trusted, so roll
            # forward and seek a completely consistent pair.
            pending_initial_terrain_gauge = current
            stats["inconsistent_pairs"] += 1
            stats["pending_patches"] += 1
            stats["last_status"] = "inconsistent_pair_restarted"
            consistency_record["decision"] = "initialization_restarted"
            return None, True, None

        gauge_pose2 = matched_pose2.copy()
        component_mode = "full"
        if measurement_yaw_observable and yaw_consistent:
            # The first fix and intervening local odometry are used only to
            # confirm that the current absolute match occupies a consistent
            # basin.  Averaging their propagated heading into this unary
            # measurement would count the same odometry twice: once here and
            # again in the graph edge.  Position already retains the current
            # independent match; yaw must follow the same rule.
            gauge_pose2[2] = matched_pose2[2]
            stats["full_pose_confirmations"] += 1
        else:
            # Height-only terrain matching can have a good horizontal basin
            # but a noisy yaw optimum. Preserve its independent x/y evidence,
            # propagate yaw from the last confirmed fix, and remove yaw and
            # x/y-yaw cross-information from this unary graph factor.
            gauge_pose2[2] = propagated_first[2]
            component_mode = "position"
            stats["position_only_confirmations"] += 1
        if not position_consistent:
            # Once the global gauge is initialized, local odometry is not an
            # authority over an independently accepted absolute map match: its
            # purpose is precisely to drift between fixes. Record the temporal
            # disagreement for diagnosis, but let the matcher's overlap,
            # observability, residual, and 100:1 basin gates plus the graph's
            # robust kernel govern the x/y factor.
            stats["last_status"] = "position_innovation_recorded_not_gated"
        consistency_record["decision"] = component_mode
        yaw_correction = math.atan2(
            math.sin(gauge_pose2[2] - seed_pose2[2]),
            math.cos(gauge_pose2[2] - seed_pose2[2]),
        )
        stats["last_yaw_correction_deg"] = float(
            np.rad2deg(yaw_correction)
        )
        # A sliding confirmation window preserves the ordinary factor cadence:
        # the current independently acquired patch becomes the reference for
        # the next patch.  Its pose is used only once as an absolute position
        # measurement; reuse is limited to validation and yaw averaging.
        if ARGS.terrain_confirm_every_prior:
            # Carry the filtered current-time posterior and the local
            # acoustic-odometry epoch into the next independent confirmation.
            current["matched_pose2"] = gauge_pose2.copy()
            pending_initial_terrain_gauge = current
        else:
            pending_initial_terrain_gauge = None
        stats["confirmed_pairs"] += 1
        if initializing:
            stats["confirmed_initializations"] += 1
        stats["last_status"] = (
            "two_disjoint_patches_initialized"
            if initializing
            else "two_disjoint_patches_prior_ready"
        )
        return gauge_pose2, False, component_mode

    def update_terrain_backend_particle_filter(
        *,
        timestamp,
        observation_id,
        mode,
        patch_body,
        patch_reference_state,
        backscatter_patch_body,
        backscatter_patch_values,
        patch_start_timestamp,
        patch_end_timestamp,
        patch_start_observation_id,
        patch_end_observation_id,
        evidences=(),
        evaluation_truth_state=None,
    ):
        """Apply one raw sonar window exactly once to the ordinary cohort PF."""

        nonlocal terrain_backend_particle_last_patch_end_observation_id
        if terrain_backend_particle_tracker is None:
            return None
        if (
            patch_end_observation_id
            <= terrain_backend_particle_last_patch_end_observation_id
        ):
            raise RuntimeError(
                "ordinary terrain particle windows must advance monotonically"
            )
        local_pose2 = np.array(
            [
                slam.state.position[0],
                slam.state.position[1],
                rpy_from_rotation(slam.state.rotation)[2],
            ],
            dtype=np.float64,
        )
        raw_likelihood = AcousticTerrainParticleLikelihood(
            terrain_prior_model,
            patch_reference_state,
            patch_body,
            backscatter_terrain=world_model,
            backscatter_points_body=backscatter_patch_body,
            backscatter_values=backscatter_patch_values,
            config=terrain_particle_likelihood_config,
        )
        particle_observation = ParticleLikelihoodObservation(
            evaluator=raw_likelihood,
            patch_start_timestamp_s=patch_start_timestamp,
            patch_end_timestamp_s=patch_end_timestamp,
            patch_start_observation_id=patch_start_observation_id,
            patch_end_observation_id=patch_end_observation_id,
        )
        original_evidence_tuple = tuple(evidences)
        refinement_records = []
        refined_evidences = []
        truth_yaw = (
            None
            if evaluation_truth_state is None
            else float(
                rpy_from_rotation(evaluation_truth_state.rotation)[2]
            )
        )

        def append_refined_evidence(
            evidence, seed_pose2, refined_pose2, *, role, cohort_id=None
        ):
            seed_pose2 = np.asarray(seed_pose2, dtype=np.float64).reshape(3)
            refined_pose2 = np.asarray(
                refined_pose2, dtype=np.float64
            ).reshape(3)
            translation_m = float(
                np.linalg.norm(refined_pose2[:2] - seed_pose2[:2])
            )
            yaw_deg = float(
                np.rad2deg(
                    math.atan2(
                        math.sin(refined_pose2[2] - seed_pose2[2]),
                        math.cos(refined_pose2[2] - seed_pose2[2]),
                    )
                )
            )
            source_pose2 = np.asarray(
                evidence.map_pose2, dtype=np.float64
            ).reshape(3)
            source_disagreement = refined_pose2 - source_pose2
            source_disagreement[2] = math.atan2(
                math.sin(source_disagreement[2]),
                math.cos(source_disagreement[2]),
            )
            # The compact Gaussian was linearized at source_pose2.  Moving its
            # mean to another raw-likelihood mode without changing covariance
            # would be over-confident.  Treat the mode disagreement as a
            # correlated systematic uncertainty; this continuously weakens
            # the affected directions instead of introducing an error gate.
            refined_information = information_with_covariance_floor(
                evidence.information,
                np.outer(source_disagreement, source_disagreement),
            )
            refined_covariance = reporting_covariance_from_information(
                refined_information,
                yaw_scale_m=terrain_particle_likelihood_config.yaw_rank_scale_m,
                unobservable_variance_scaled_m2=(
                    terrain_particle_likelihood_config
                    .unobservable_variance_scaled_m2
                ),
            )
            refined_position_std_m = float(
                np.sqrt(
                    np.max(
                        np.linalg.eigvalsh(refined_covariance[:2, :2])
                    )
                )
            )
            refined_yaw_std_deg = float(
                np.rad2deg(np.sqrt(refined_covariance[2, 2]))
            )
            refined_evidences.append(
                replace(
                    evidence,
                    map_pose2=refined_pose2,
                    source_index=len(refined_evidences),
                    position_std_m=refined_position_std_m,
                    yaw_std_deg=refined_yaw_std_deg,
                    covariance_matrix=refined_covariance,
                    target_cohort_id=cohort_id,
                    information_matrix=refined_information,
                )
            )
            refinement_records.append(
                {
                    "role": role,
                    "cohort_id": cohort_id,
                    "original_pose2": seed_pose2.tolist(),
                    "refined_pose2": refined_pose2.tolist(),
                    "translation_m": translation_m,
                    "yaw_deg": yaw_deg,
                    "source_gaussian_disagreement": (
                        source_disagreement.tolist()
                    ),
                    "inflated_position_std_m": refined_position_std_m,
                    "inflated_yaw_std_deg": refined_yaw_std_deg,
                    "truth_evaluation": (
                        None
                        if evaluation_truth_state is None
                        else {
                            "original_horizontal_error_m": float(
                                np.linalg.norm(
                                    seed_pose2[:2]
                                    - evaluation_truth_state.position[:2]
                                )
                            ),
                            "refined_horizontal_error_m": float(
                                np.linalg.norm(
                                    refined_pose2[:2]
                                    - evaluation_truth_state.position[:2]
                                )
                            ),
                            "original_yaw_error_deg": float(
                                np.rad2deg(
                                    math.atan2(
                                        math.sin(
                                            seed_pose2[2] - truth_yaw
                                        ),
                                        math.cos(
                                            seed_pose2[2] - truth_yaw
                                        ),
                                    )
                                )
                            ),
                            "refined_yaw_error_deg": float(
                                np.rad2deg(
                                    math.atan2(
                                        math.sin(
                                            refined_pose2[2] - truth_yaw
                                        ),
                                        math.cos(
                                            refined_pose2[2] - truth_yaw
                                        ),
                                    )
                                )
                            ),
                            "used_by_estimator": False,
                        }
                    ),
                }
            )
            terrain_backend_particle_stats[
                "raw_likelihood_refinements"
            ] += 1
            terrain_backend_particle_stats[
                "raw_likelihood_refinement_translation_sum_m"
            ] += translation_m
            terrain_backend_particle_stats[
                "raw_likelihood_refinement_translation_max_m"
            ] = max(
                terrain_backend_particle_stats[
                    "raw_likelihood_refinement_translation_max_m"
                ],
                translation_m,
            )
            terrain_backend_particle_stats[
                "raw_likelihood_refinement_abs_yaw_sum_deg"
            ] += abs(yaw_deg)
            terrain_backend_particle_stats[
                "raw_likelihood_refinement_abs_yaw_max_deg"
            ] = max(
                terrain_backend_particle_stats[
                    "raw_likelihood_refinement_abs_yaw_max_deg"
                ],
                abs(yaw_deg),
            )

        # A persistent cohort selects a data-association basin.  The current
        # raw sonar window alone supplies both the local maximum and its
        # rank-aware curvature, so an independent graph measurement remains
        # available even when the compact whole-map matcher has no proposal.
        # No recursive particle covariance is reused as measurement noise.
        for cohort_id, predicted_pose2 in (
            terrain_backend_particle_tracker.prediction_seeds(local_pose2)
        ):
            approximation = raw_likelihood.approximate_local_measurement(
                predicted_pose2
            )
            basin_pose2 = np.asarray(
                approximation.pose2, dtype=np.float64
            ).reshape(3)
            measurement_covariance = np.asarray(
                approximation.covariance, dtype=np.float64
            ).reshape(3, 3)
            diagnostics = approximation.diagnostics
            bathymetry_score = (
                float(
                    np.tanh(
                        (
                            terrain_particle_likelihood_config
                            .bathymetry_reference_normalized_loss
                            - diagnostics.best_bathymetry_normalized_loss
                        )
                        / terrain_particle_likelihood_config
                        .bathymetry_fit_temperature
                    )
                )
                if np.isfinite(
                    diagnostics.best_bathymetry_normalized_loss
                )
                else 0.0
            )
            backscatter_score = (
                float(
                    np.clip(
                        diagnostics.best_backscatter_correlation,
                        -1.0,
                        1.0,
                    )
                )
                if np.isfinite(
                    diagnostics.best_backscatter_correlation
                )
                else 0.0
            )
            position_std_m = float(
                np.sqrt(
                    np.max(
                        np.linalg.eigvalsh(
                            measurement_covariance[:2, :2]
                        )
                    )
                )
            )
            yaw_std_deg = float(
                np.rad2deg(
                    np.sqrt(measurement_covariance[2, 2])
                )
            )
            refined_evidences.append(
                GlobalPoseEvidence(
                    map_pose2=basin_pose2,
                    bathymetry_score=bathymetry_score,
                    backscatter_score=backscatter_score,
                    bathymetry_observability=(
                        diagnostics.bathymetry_observability
                    ),
                    backscatter_observability=(
                        diagnostics.backscatter_observability
                    ),
                    geometry_score=None,
                    cross_modal_consistent=True,
                    position_std_m=position_std_m,
                    yaw_std_deg=yaw_std_deg,
                    patch_start_timestamp_s=patch_start_timestamp,
                    patch_end_timestamp_s=patch_end_timestamp,
                    patch_start_observation_id=(
                        patch_start_observation_id
                    ),
                    patch_end_observation_id=patch_end_observation_id,
                    source_index=len(refined_evidences),
                    covariance_matrix=measurement_covariance,
                    target_cohort_id=int(cohort_id),
                    information_matrix=approximation.information,
                )
            )
            predicted_pose2 = np.asarray(
                predicted_pose2, dtype=np.float64
            ).reshape(3)
            translation_m = float(
                np.linalg.norm(basin_pose2[:2] - predicted_pose2[:2])
            )
            yaw_deg = float(
                np.rad2deg(
                    math.atan2(
                        math.sin(basin_pose2[2] - predicted_pose2[2]),
                        math.cos(basin_pose2[2] - predicted_pose2[2]),
                    )
                )
            )
            refinement_records.append(
                {
                    "role": "persistent_cohort_basin",
                    "cohort_id": int(cohort_id),
                    "measurement_source": (
                        "current_raw_likelihood_laplace"
                    ),
                    "original_pose2": predicted_pose2.tolist(),
                    "refined_pose2": basin_pose2.tolist(),
                    "translation_m": translation_m,
                    "yaw_deg": yaw_deg,
                    "source_gaussian_disagreement": None,
                    "inflated_position_std_m": position_std_m,
                    "inflated_yaw_std_deg": yaw_std_deg,
                    "observable_rank": int(
                        approximation.observable_rank
                    ),
                    "scaled_information_eigenvalues": (
                        approximation.scaled_information_eigenvalues.tolist()
                    ),
                    "truth_evaluation": (
                        None
                        if evaluation_truth_state is None
                        else {
                            "original_horizontal_error_m": float(
                                np.linalg.norm(
                                    predicted_pose2[:2]
                                    - evaluation_truth_state.position[:2]
                                )
                            ),
                            "refined_horizontal_error_m": float(
                                np.linalg.norm(
                                    basin_pose2[:2]
                                    - evaluation_truth_state.position[:2]
                                )
                            ),
                            "original_yaw_error_deg": float(
                                np.rad2deg(
                                    math.atan2(
                                        math.sin(
                                            predicted_pose2[2] - truth_yaw
                                        ),
                                        math.cos(
                                            predicted_pose2[2] - truth_yaw
                                        ),
                                    )
                                )
                            ),
                            "refined_yaw_error_deg": float(
                                np.rad2deg(
                                    math.atan2(
                                        math.sin(
                                            basin_pose2[2] - truth_yaw
                                        ),
                                        math.cos(
                                            basin_pose2[2] - truth_yaw
                                        ),
                                    )
                                )
                            ),
                            "used_by_estimator": False,
                        }
                    ),
                }
            )
            terrain_backend_particle_stats[
                "raw_likelihood_refinements"
            ] += 1
            terrain_backend_particle_stats[
                "raw_likelihood_refinement_translation_sum_m"
            ] += translation_m
            terrain_backend_particle_stats[
                "raw_likelihood_refinement_translation_max_m"
            ] = max(
                terrain_backend_particle_stats[
                    "raw_likelihood_refinement_translation_max_m"
                ],
                translation_m,
            )
            terrain_backend_particle_stats[
                "raw_likelihood_refinement_abs_yaw_sum_deg"
            ] += abs(yaw_deg)
            terrain_backend_particle_stats[
                "raw_likelihood_refinement_abs_yaw_max_deg"
            ] = max(
                terrain_backend_particle_stats[
                    "raw_likelihood_refinement_abs_yaw_max_deg"
                ],
                abs(yaw_deg),
            )
            terrain_backend_particle_stats[
                "cohort_basin_refinements"
            ] += 1

        for evidence in original_evidence_tuple:
            original_pose2 = np.asarray(
                evidence.map_pose2, dtype=np.float64
            ).reshape(3)
            refined_pose2 = raw_likelihood.refine_pose(original_pose2)
            append_refined_evidence(
                evidence,
                original_pose2,
                refined_pose2,
                role="compact_proposal_basin",
            )
            terrain_backend_particle_stats[
                "compact_proposal_refinements"
            ] += 1
        evidence_tuple = tuple(refined_evidences)
        decision = terrain_backend_particle_tracker.update(
            timestamp_s=timestamp,
            raw_pose2=local_pose2,
            evidences=evidence_tuple,
            particle_observation=particle_observation,
        )
        terrain_backend_particle_last_patch_end_observation_id = int(
            patch_end_observation_id
        )
        terrain_backend_particle_stats["updates"] += 1
        terrain_backend_particle_stats["raw_likelihood_updates"] += 1
        terrain_backend_particle_stats[
            "candidate_seed_updates"
            if evidence_tuple
            else "unseeded_tracking_updates"
        ] += 1
        terrain_backend_particle_stats[
            f"{decision.status}_updates"
        ] += 1
        leading = decision.cohorts[0] if decision.cohorts else None
        diagnostics = raw_likelihood.last_diagnostics
        terrain_backend_particle_records.append(
            {
                "timestamp_s": float(timestamp),
                "observation_id": int(observation_id),
                "mode": mode,
                "status": decision.status,
                "candidate_count": len(evidence_tuple),
                "raw_likelihood_refinements": refinement_records,
                "null_probability": float(decision.null_probability),
                "best_probability": float(decision.best_probability),
                "odds_over_second": float(decision.odds_over_second),
                "cohort_count": len(decision.cohorts),
                "cohorts": [
                    {
                        "cohort_id": int(cohort.cohort_id),
                        "probability": float(cohort.probability),
                        "map_pose2": cohort.map_pose2.tolist(),
                        "independent_evidence": int(
                            cohort.independent_evidence
                        ),
                        "effective_sample_size": float(
                            cohort.effective_sample_size
                        ),
                        "last_observation_log_likelihood_ratio": float(
                            cohort.last_observation_log_likelihood_ratio
                        ),
                        "last_measurement_observable_rank": int(
                            cohort.last_measurement_observable_rank
                        ),
                        "last_measurement_is_graph_ready": bool(
                            cohort.last_measurement_is_graph_ready
                        ),
                    }
                    for cohort in decision.cohorts
                ],
                "best_particle_count": (
                    0 if leading is None else int(leading.particle_count)
                ),
                "best_effective_sample_size": (
                    0.0
                    if leading is None
                    else float(leading.effective_sample_size)
                ),
                "best_independent_evidence": (
                    0
                    if leading is None
                    else int(leading.independent_evidence)
                ),
                "best_map_pose2": (
                    None if leading is None else leading.map_pose2.tolist()
                ),
                "best_map_to_odom_pose2": (
                    None
                    if leading is None
                    else leading.map_to_odom_pose2.tolist()
                ),
                "best_last_measurement_pose2": (
                    None
                    if leading is None
                    else leading.last_measurement_pose2.tolist()
                ),
                "best_last_measurement_position_std_m": (
                    None
                    if leading is None
                    else float(
                        np.sqrt(
                            np.max(
                                np.linalg.eigvalsh(
                                    leading.last_measurement_covariance[:2, :2]
                                )
                            )
                        )
                    )
                ),
                "best_last_measurement_yaw_std_deg": (
                    None
                    if leading is None
                    else float(
                        np.rad2deg(
                            np.sqrt(
                                max(
                                    0.0,
                                    float(
                                        leading.last_measurement_covariance[2, 2]
                                    ),
                                )
                            )
                        )
                    )
                ),
                "best_last_measurement_observable_rank": (
                    None
                    if leading is None
                    else int(leading.last_measurement_observable_rank)
                ),
                "best_last_observation_log_likelihood_ratio": (
                    None
                    if leading is None
                    else float(
                        leading.last_observation_log_likelihood_ratio
                    )
                ),
                "best_last_observation_novel_fraction": (
                    None
                    if leading is None
                    else float(leading.last_observation_novel_fraction)
                ),
                "confirmed_with_new_independent_evidence": bool(
                    decision.confirmed_with_new_independent_evidence
                ),
                "best_measurement_graph_ready": bool(
                    leading is not None
                    and leading.last_measurement_is_graph_ready
                ),
                "bathymetry_observability": (
                    None
                    if diagnostics is None
                    else float(diagnostics.bathymetry_observability)
                ),
                "backscatter_observability": (
                    None
                    if diagnostics is None
                    else float(diagnostics.backscatter_observability)
                ),
                "best_bathymetry_normalized_loss": (
                    None
                    if diagnostics is None
                    else _finite_float_or_none(
                        diagnostics.best_bathymetry_normalized_loss
                    )
                ),
                "best_backscatter_correlation": (
                    None
                    if diagnostics is None
                    else _finite_float_or_none(
                        diagnostics.best_backscatter_correlation
                    )
                ),
                "particle_log_likelihood_ratio_std": (
                    None
                    if diagnostics is None
                    else float(diagnostics.log_likelihood_ratio_std)
                ),
                "uses_truth": False,
            }
        )
        return decision

    def published_online_global_pose():
        """Publish TRN MAP when valid, otherwise retain the graph output.

        The particle posterior remains separate from GTSAM. Whole-map
        relocalization hypotheses take priority while that recovery tracker is
        active; otherwise the ordinary filter's leading mode may drive the
        output-only map->odom transform during multimodal tracking. Only an
        independently confirmed, observable current sonar measurement can
        become a graph factor.
        """

        particle_map_pose2 = None
        particle_map_source = None
        use_global_particle_map = bool(
            global_cohort_tracker is not None
            and ARGS.terrain_particle_online_output != "graph"
            and global_cohort_tracker.active_cohort_count > 0
        )
        if use_global_particle_map:
            local_pose2 = np.array(
                [
                    slam.state.position[0],
                    slam.state.position[1],
                    rpy_from_rotation(slam.state.rotation)[2],
                ],
                dtype=np.float64,
            )
            particle_map_pose2 = (
                global_cohort_tracker.maximum_a_posteriori_pose(local_pose2)
            )
            if particle_map_pose2 is not None:
                particle_map_source = "global_terrain_particle_map"
        use_ordinary_particle_map = bool(
            particle_map_pose2 is None
            and not use_global_particle_map
            and terrain_backend_particle_tracker is not None
            and ARGS.terrain_particle_online_output != "graph"
            and (
                ARGS.terrain_particle_online_output == "always-map"
                or terrain_backend_particle_tracker.active_cohort_count > 1
            )
        )
        if use_ordinary_particle_map:
            local_pose2 = np.array(
                [
                    slam.state.position[0],
                    slam.state.position[1],
                    rpy_from_rotation(slam.state.rotation)[2],
                ],
                dtype=np.float64,
            )
            particle_map_pose2 = (
                terrain_backend_particle_tracker
                .maximum_a_posteriori_pose(local_pose2)
            )
            if particle_map_pose2 is not None:
                particle_map_source = "ordinary_terrain_particle_map"
        if particle_map_pose2 is None:
            terrain_navigation_output_stats[
                "pose_graph_fallback_updates"
            ] += 1
            terrain_navigation_output_stats["last_source"] = "pose_graph"
            return slam.global_pose()
        terrain_navigation_output_stats["particle_map_updates"] += 1
        if particle_map_source == "global_terrain_particle_map":
            terrain_navigation_output_stats[
                "global_particle_map_updates"
            ] += 1
        else:
            terrain_navigation_output_stats[
                "ordinary_particle_map_updates"
            ] += 1
        terrain_navigation_output_stats["last_source"] = particle_map_source
        terrain_navigation_output_stats["last_particle_map_pose2"] = (
            particle_map_pose2.tolist()
        )
        return slam.global_pose(map_pose2_target=particle_map_pose2)

    def try_add_terrain_prior_result(
        result,
        *,
        mode,
        timestamp,
        observation_id,
        patch_body,
        patch_reference_state=None,
        backscatter_patch_body=None,
        backscatter_patch_values=None,
        patch_start_timestamp,
        patch_end_timestamp,
        patch_start_observation_id,
        patch_end_observation_id,
        matched_pose2_override=None,
        information_override=None,
        information_override_includes_model_floors=False,
        yaw_observable_override=None,
        backscatter_result=None,
        evaluation_truth_state=None,
    ):
        """Attempt one factor and retain matcher-versus-graph diagnostics."""

        local_pose2 = np.array(
            [
                slam.state.position[0],
                slam.state.position[1],
                rpy_from_rotation(slam.state.rotation)[2],
            ],
            dtype=np.float64,
        )
        initialize_global_gauge = bool(
            ARGS.terrain_initialize_global_gauge
            and not ARGS.initialize_from_truth
            and terrain_prior_factor_stats["accepts"] == 0
            and mode not in ("emergency", "global")
        )
        gauge_seed_pose2 = None
        factor_matched_pose2 = np.asarray(
            result.matched_pose2
            if matched_pose2_override is None
            else matched_pose2_override,
            dtype=np.float64,
        ).reshape(3)
        particle_filter_confirmed = False
        if (
            terrain_backend_particle_tracker is not None
            and mode in ("normal", "recovery")
        ):
            measurement_information = terrain_prior_information(
                result,
                "full",
                base_information=information_override,
                apply_model_floors=(
                    not information_override_includes_model_floors
                ),
            )
            measurement_covariance = reporting_covariance_from_information(
                measurement_information,
                yaw_scale_m=terrain_particle_likelihood_config.yaw_rank_scale_m,
                unobservable_variance_scaled_m2=(
                    terrain_particle_likelihood_config
                    .unobservable_variance_scaled_m2
                ),
            )
            position_std_m = math.sqrt(
                max(
                    1.0e-12,
                    float(
                        np.linalg.eigvalsh(
                            measurement_covariance[:2, :2]
                        )[-1]
                    ),
                )
            )
            yaw_std_deg = math.degrees(
                math.sqrt(max(1.0e-12, float(measurement_covariance[2, 2])))
            )
            inlier_ratio = float(
                result.inliers / max(result.correspondences, 1)
            )
            reduced_chi_square = float(result.reduced_chi_square)
            chi_square_quality = (
                math.exp(-0.5 * max(0.0, reduced_chi_square - 1.0))
                if math.isfinite(reduced_chi_square)
                else 0.0
            )
            bathymetry_score = float(
                np.clip(
                    0.5 * inlier_ratio + 0.5 * chi_square_quality,
                    -1.0,
                    1.0,
                )
            )
            bathymetry_observability = float(
                np.clip(result.translation_observable_rank / 2.0, 0.0, 1.0)
            )
            backscatter_score = 0.0
            backscatter_observability = 0.0
            if backscatter_result is not None:
                correlation = float(backscatter_result.correlation_after)
                if math.isfinite(correlation):
                    backscatter_score = float(np.clip(correlation, -1.0, 1.0))
                if backscatter_matcher is not None:
                    texture_scale = max(
                        5.0 * backscatter_matcher.config.minimum_texture_std,
                        1.0e-12,
                    )
                    backscatter_observability = float(
                        np.clip(
                            backscatter_result.observation_texture_std
                            / texture_scale,
                            0.0,
                            1.0,
                        )
                        * np.clip(backscatter_result.valid_fraction, 0.0, 1.0)
                    )
            particle_evidence = GlobalPoseEvidence(
                map_pose2=factor_matched_pose2,
                bathymetry_score=bathymetry_score,
                backscatter_score=backscatter_score,
                bathymetry_observability=bathymetry_observability,
                backscatter_observability=backscatter_observability,
                geometry_score=float(np.clip(inlier_ratio, 0.0, 1.0)),
                cross_modal_consistent=True,
                position_std_m=position_std_m,
                yaw_std_deg=yaw_std_deg,
                patch_start_timestamp_s=patch_start_timestamp,
                patch_end_timestamp_s=patch_end_timestamp,
                patch_start_observation_id=patch_start_observation_id,
                patch_end_observation_id=patch_end_observation_id,
                source_index=0,
                covariance_matrix=measurement_covariance,
                information_matrix=measurement_information,
            )
            if patch_reference_state is None:
                raise RuntimeError(
                    "particle-backed terrain priors require a patch reference pose"
                )
            if backscatter_patch_body is None:
                backscatter_patch_body = np.empty((0, 3), dtype=np.float64)
            if backscatter_patch_values is None:
                backscatter_patch_values = np.empty(0, dtype=np.float64)
            particle_decision = update_terrain_backend_particle_filter(
                timestamp=timestamp,
                observation_id=observation_id,
                mode=mode,
                patch_body=patch_body,
                patch_reference_state=patch_reference_state,
                backscatter_patch_body=backscatter_patch_body,
                backscatter_patch_values=backscatter_patch_values,
                patch_start_timestamp=patch_start_timestamp,
                patch_end_timestamp=patch_end_timestamp,
                patch_start_observation_id=patch_start_observation_id,
                patch_end_observation_id=patch_end_observation_id,
                evidences=(particle_evidence,),
                evaluation_truth_state=evaluation_truth_state,
            )
            confirmed_particle_cohort = (
                particle_decision.confirmed_cohort
            )
            if confirmed_particle_cohort is None:
                terrain_backend_particle_stats["factor_holds"] += 1
                return False, True
            if (
                not particle_decision.confirmed_with_new_independent_evidence
                or not confirmed_particle_cohort.last_measurement_is_graph_ready
            ):
                # A confirmed recursive posterior may be returned on an
                # overlapping rolling window.  Reusing it as a new unary
                # factor would count the same historical sonar information
                # again and make GTSAM falsely over-confident.
                terrain_backend_particle_stats["factor_holds"] += 1
                terrain_backend_particle_stats[
                    "nonindependent_factor_reuses_prevented"
                ] += 1
                return False, True

            particle_filter_confirmed = True
            terrain_backend_particle_stats["factor_authorizations"] += 1
            factor_matched_pose2 = np.asarray(
                confirmed_particle_cohort.last_measurement_pose2,
                dtype=np.float64,
            ).copy()
            # This is the local Laplace information from the *current
            # independent acquisition*, not the recursively accumulated PF
            # posterior.  Its map/model floors were already applied when the
            # evidence was constructed, so do not apply them a second time.
            information_override = np.asarray(
                confirmed_particle_cohort.last_measurement_information,
                dtype=np.float64,
            ).copy()
            information_override_includes_model_floors = True
            terrain_backend_particle_stats[
                "posterior_as_measurement_reuses_prevented"
            ] += 1

        if particle_filter_confirmed:
            factor_component_mode = "full"
            # Only the first confirmed PF output establishes/re-establishes
            # the graph gauge. Subsequent PF factors pass normal graph prefit.
            if initialize_global_gauge:
                gauge_seed_pose2 = factor_matched_pose2.copy()
        elif initialize_global_gauge:
            (
                gauge_seed_pose2,
                held_for_confirmation,
                factor_component_mode,
            ) = (
                prepare_initial_terrain_gauge(
                    result,
                    matched_pose2_override=factor_matched_pose2,
                    yaw_observable_override=yaw_observable_override,
                    odometry_pose2=local_pose2,
                    initializing=True,
                    patch_start_timestamp=patch_start_timestamp,
                    patch_end_timestamp=patch_end_timestamp,
                    patch_start_observation_id=patch_start_observation_id,
                    patch_end_observation_id=patch_end_observation_id,
                )
            )
            if held_for_confirmation:
                return False, True
            factor_matched_pose2 = gauge_seed_pose2
        elif ARGS.terrain_confirm_every_prior and mode not in (
            "emergency",
            "global",
        ):
            (
                factor_matched_pose2,
                held_for_confirmation,
                factor_component_mode,
            ) = (
                prepare_initial_terrain_gauge(
                    result,
                    matched_pose2_override=factor_matched_pose2,
                    yaw_observable_override=yaw_observable_override,
                    odometry_pose2=local_pose2,
                    initializing=False,
                    patch_start_timestamp=patch_start_timestamp,
                    patch_end_timestamp=patch_end_timestamp,
                    patch_start_observation_id=patch_start_observation_id,
                    patch_end_observation_id=patch_end_observation_id,
                )
            )
            if held_for_confirmation:
                return False, True
        else:
            factor_component_mode = "full"

        terrain_prior_factor_stats["attempts"] += 1
        terrain_match_mode_stats[mode]["prior_factor_attempts"] += 1
        added = slam.pose_graph.add_terrain_prior(
            timestamp=timestamp,
            observation_id=observation_id,
            prior_pose2=local_pose2,
            raw_rotation_world_body=slam.state.rotation,
            raw_position_world=slam.state.position,
            matched_pose2=factor_matched_pose2,
            information_world=terrain_prior_information(
                result,
                factor_component_mode,
                base_information=information_override,
                apply_model_floors=(
                    not information_override_includes_model_floors
                ),
            ),
            points_body=patch_body,
            # Recovery candidates still pass graph prefit; only a disjoint-
            # strip-confirmed emergency/global re-anchor must affect the next
            # seed immediately.
            optimize_interval=(
                1
                if mode in ("recovery", "emergency", "global")
                else ARGS.terrain_prior_optimize_interval
            ),
            # Initial two-patch x/y baselines estimate a common SE(2) gauge;
            # emergency re-seeding has its own stricter confirmation policy.
            relocalization_seed=terrain_prior_prefit_bypass_allowed(
                mode=mode,
                initializes_global_gauge=gauge_seed_pose2 is not None,
            ),
            relocalization_seed_pose2=gauge_seed_pose2,
            # Normal PF-authorized measurements already use the pose graph's
            # bounded Huber kernel.  Select the separate Cauchy policy only
            # when an externally confirmed basin also re-seeds the graph;
            # the pose-graph API intentionally forbids that relocalization
            # kernel without a corresponding seed operation.
            relocalization_prior_robust=bool(
                (
                    mode == "global" and global_cohort_tracker is not None
                )
                or (
                    particle_filter_confirmed
                    and gauge_seed_pose2 is not None
                )
            ),
        )
        diagnostics = slam.pose_graph.diagnostics()
        if diagnostics.last_terrain_prefit_performed:
            prefit_samples = (
                terrain_prefit_accepted_chi2
                if added
                else terrain_prefit_rejected_chi2
            )
            prefit_samples.append(float(diagnostics.last_terrain_prefit_chi2))
        if added:
            terrain_prior_factor_stats["accepts"] += 1
            terrain_match_mode_stats[mode]["prior_factor_accepts"] += 1
        else:
            terrain_prior_factor_stats["rejections"] += 1
            terrain_match_mode_stats[mode]["prior_factor_rejections"] += 1
            terrain_prior_factor_stats["rejection_reasons"][
                diagnostics.last_terrain_prior_rejection_reason or "unspecified"
            ] += 1
        return bool(added), False

    def try_add_current_raw_particle_prior(
        decision,
        *,
        timestamp,
        observation_id,
        patch_body,
    ):
        """Insert one current-window likelihood factor after PF association.

        The PF selects the persistent basin, but neither its recursive mean nor
        its recursive covariance enters this unary factor.  Both measurement
        pose and rank-aware information come only from the current independent
        raw sonar window's local Laplace approximation.
        """

        cohort = None if decision is None else decision.confirmed_cohort
        if cohort is None:
            terrain_backend_particle_stats["factor_holds"] += 1
            return False
        if (
            not decision.confirmed_with_new_independent_evidence
            or not cohort.last_measurement_is_graph_ready
        ):
            terrain_backend_particle_stats["factor_holds"] += 1
            terrain_backend_particle_stats[
                "nonindependent_factor_reuses_prevented"
            ] += 1
            return False

        terrain_backend_particle_stats["factor_authorizations"] += 1
        terrain_backend_particle_stats[
            "posterior_as_measurement_reuses_prevented"
        ] += 1
        terrain_backend_particle_stats[
            "raw_likelihood_factor_attempts"
        ] += 1
        terrain_prior_factor_stats["attempts"] += 1
        terrain_match_mode_stats["normal"]["prior_factor_attempts"] += 1
        local_pose2 = np.array(
            [
                slam.state.position[0],
                slam.state.position[1],
                rpy_from_rotation(slam.state.rotation)[2],
            ],
            dtype=np.float64,
        )
        added = slam.pose_graph.add_terrain_prior(
            timestamp=timestamp,
            observation_id=observation_id,
            prior_pose2=local_pose2,
            raw_rotation_world_body=slam.state.rotation,
            raw_position_world=slam.state.position,
            matched_pose2=cohort.last_measurement_pose2,
            information_world=cohort.last_measurement_information,
            points_body=patch_body,
            optimize_interval=ARGS.terrain_prior_optimize_interval,
            relocalization_seed=False,
            relocalization_seed_pose2=None,
            relocalization_prior_robust=False,
        )
        diagnostics = slam.pose_graph.diagnostics()
        if diagnostics.last_terrain_prefit_performed:
            prefit_samples = (
                terrain_prefit_accepted_chi2
                if added
                else terrain_prefit_rejected_chi2
            )
            prefit_samples.append(
                float(diagnostics.last_terrain_prefit_chi2)
            )
        if added:
            terrain_backend_particle_stats[
                "raw_likelihood_factor_accepts"
            ] += 1
            terrain_prior_factor_stats["accepts"] += 1
            terrain_match_mode_stats["normal"][
                "prior_factor_accepts"
            ] += 1
        else:
            terrain_backend_particle_stats[
                "raw_likelihood_factor_rejections"
            ] += 1
            terrain_prior_factor_stats["rejections"] += 1
            terrain_match_mode_stats["normal"][
                "prior_factor_rejections"
            ] += 1
            terrain_prior_factor_stats["rejection_reasons"][
                diagnostics.last_terrain_prior_rejection_reason
                or "unspecified"
            ] += 1
        return bool(added)

    truth_track: list[np.ndarray] = [truth0.position.copy()]
    truth_rotation_track: list[np.ndarray] = [truth0.rotation.copy()]
    evaluation_timestamps: list[float] = [0.0]
    dead_reckoning_track: list[np.ndarray] = [
        dead_reckoning.state.position.copy()
    ]
    dead_reckoning_rotation_track: list[np.ndarray] = [
        dead_reckoning.state.rotation.copy()
    ]
    local_track: list[np.ndarray] = [initial.position.copy()]
    local_rotation_track: list[np.ndarray] = [initial.rotation.copy()]
    online_global_track: list[np.ndarray] = [initial.position.copy()]
    online_global_rotation_track: list[np.ndarray] = [initial.rotation.copy()]

    # UI tracks are deliberately separate from the full-rate evaluation
    # histories above.  Correcting and converting only this bounded cache
    # avoids revisiting tens of thousands of acoustic samples every refresh.
    display_track_capacity = 6_000
    expected_display_distance_m = float(route_report.path_length_m) * max(
        float(ARGS.survey_laps), 1.0
    )
    display_sample_spacing_m = max(
        expected_display_distance_m / max(display_track_capacity - 1, 1),
        ARGS.survey_speed_mps / ARGS.ping_hz,
        1.0e-6,
    )
    display_timestamps: list[float] = [0.0]
    display_truth_track: list[np.ndarray] = [truth0.position.copy()]
    display_local_track: list[np.ndarray] = [initial.position.copy()]
    display_last_distance_m = 0.0
    display_sample_version = 1
    last_rendered_display_version = -1
    cached_global_display_track: list[np.ndarray] = [initial.position.copy()]
    cached_curve_graph_signature = (-1, -1)
    cached_map_graph_signature = (-1, -1)
    sim_time = 0.0
    imu_dt = 0.02
    dvl_period = 0.2
    depth_period = 0.1
    next_imu = imu_dt
    next_dvl = dvl_period
    next_depth = depth_period
    next_ping = 0.0
    next_sonar_visualization = 0.0
    ping_period = 1.0 / ARGS.ping_hz
    imu_sample_count = 0
    dvl_sample_count = 0
    depth_sample_count = 0
    ping_count = 0
    sonar_visualization_updates = 0
    frame_count = 0
    performance_wall_s: Counter[str] = Counter()
    next_curve_update = 0.0
    next_map_update = 0.0
    next_camera_update = 0.0
    travelled_distance_online_m = 0.0
    next_progress_distance_m = ARGS.progress_distance_m
    started = time.monotonic()
    python_profiler = None
    python_profile_path = os.environ.get("OCEANSIM_CPROFILE")
    if python_profile_path:
        import cProfile

        python_profiler = cProfile.Profile()
        python_profiler.enable()
    run_status = "running"
    termination_reason = "not_terminated"
    return_code = 0
    backend_verification: dict = {}
    latest_gyro = np.zeros(3, dtype=np.float64)
    latest_acceleration = np.mean(
        np.asarray(startup_specific_force, dtype=np.float64), axis=0
    )

    try:
        _set_vehicle_pose(vehicle_translate, vehicle_rotate, truth0.position, truth0.rotation)
        world.reset()
        sonar.initialize()
        if ARGS.verify_sonar_backend:
            # Ensure the USD transform and collision mesh are committed before
            # comparing against PhysX scene queries.
            world.step(render=False)
            backend_verification = _verify_acoustic_backend(
                sonar, acoustic_backend
            )
        if ARGS.headless:
            # A headless benchmark has no sensor windows, curve display, or
            # viewport to refresh.  Committing the initial transforms once is
            # sufficient; every acoustic event below also performs an
            # explicit non-rendering stage step before the ray query.
            world.step(render=False)
        else:
            _position_sensor_windows(sonar)
            for _ in range(20):
                world.step(render=True)
            _restore_standard_workspace()
            _position_sensor_windows(sonar)
            # The default viewport camera is finalized while Kit restores its
            # workspace.  Set the close chase view afterwards so that the first
            # relative-follow update inherits this distance (and later preserves
            # any zoom chosen by the user).
            _set_live_camera(truth0, world_model)
            world.step(render=True)
        world.play()
        print(
            f"[OceanSim] complete measured Monterey Canyon "
            f"{world_model.width / 1000.0:.2f} x "
            f"{world_model.length / 1000.0:.2f} km + full-scale LAUV "
            f"{'headless runtime is active' if ARGS.headless else 'UI is live'}; "
            f"measured backscatter coverage "
            f"{100.0 * world_model.backscatter_coverage_fraction:.2f}%; "
            f"target time scale {ARGS.time_scale:.1f}x; "
            f"sonar backend {acoustic_backend_label}; "
            f"SLAM association {slam_association_label}.",
            flush=True,
        )
        if not ARGS.headless:
            ui_sonar_rate = (
                "the final ping before each rendered frame"
                if ARGS.ui_sonar_refresh_hz == 0.0
                else f"at most {ARGS.ui_sonar_refresh_hz:g} simulated Hz"
            )
            print(
                "[OceanSim UI] rendering is decoupled from acquisition: "
                f"the operator windows present {ui_sonar_rate}; every "
                f"{ARGS.ping_hz:g} Hz ping still enters the unchanged sensor "
                "and SLAM event clocks.",
                flush=True,
            )
        print(
            f"[OceanSim] V2 {profile_mode} profile: "
            f"{sonar_profile.rows_per_side_per_ping} along-track rows/side, "
            f"{sonar_profile.range_sample_spacing_m:.3f} m samples, "
            f"{sonar_profile.range_resolution_m:.3f} m documented resolution, "
            f"{sonar_profile.maximum_range_m:.0f} m interpreted per side; "
            "frequency and vertical opening are not specified in the source document.",
            flush=True,
        )
        adjacent_ping_advance_m = ARGS.survey_speed_mps / ARGS.ping_hz
        adjacent_ping_overlap_m = max(
            0.0,
            sonar_profile.along_track_coverage_per_ping_m
            - adjacent_ping_advance_m,
        )
        print(
            "[OceanSim] integrated bathymetry + side-scan acquisition: "
            f"{ARGS.ping_hz:g} Hz, {ARGS.sonar_pings_in_water} distinguishable "
            f"pings in water (required {required_pings_in_water}), "
            f"two-way time {two_way_travel_time_s:.3f} s; "
            f"advance/overlap {adjacent_ping_advance_m:.3f}/"
            f"{adjacent_ping_overlap_m:.3f} m per ping. Waveform separation "
            "is an explicit scheduling assumption, not synthesized signal physics.",
            flush=True,
        )
        raw_returns_per_ping = (
            2
            * sonar_profile.rows_per_side_per_ping
            * ARGS.sidescan_imaging_rays_per_side
        )
        geometry_update_description = (
            f"{ARGS.mbes_group_pings}-ping causal sliding geometry "
            "window updates at the native ping rate; "
            if ARGS.enable_mbes_sliding_window
            else (
                f"{ARGS.mbes_group_pings} acquisition(s) per geometry "
                "update; "
            )
        )
        print(
            "[OceanSim SLAM] co-registered acoustic front end: "
            f"{raw_returns_per_ping} raw XYZ+intensity samples produce both "
            f"bathymetry and the 16-row side-scan image; "
            f"{geometry_update_description}"
            f"{ARGS.intensity_group_pings} acquisition(s) per update at "
            f"{ARGS.ping_hz / ARGS.intensity_group_pings:g} Hz; "
            f"causal reference separation {ARGS.intensity_reference_delay_pings} "
            f"ping, registration cells {ARGS.intensity_voxel_size_m:g} m; "
            f"local weak directions use {ARGS.local_observability_mode} weighting.",
            flush=True,
        )
        print(
            f"[OceanSim] route envelope "
            f"{route_report.x_range_m[1] - route_report.x_range_m[0]:.0f} x "
            f"{route_report.y_range_m[1] - route_report.y_range_m[0]:.0f} m, "
            f"{route_report.path_length_m / 1000.0:.3f} km/lap, measured "
            f"bathymetry/backscatter clearance "
            f"{route_report.bathymetry_min_clearance_m:.0f}/"
            f"{route_report.backscatter_min_clearance_m:.0f} m.",
            flush=True,
        )
        if terrain_matcher is not None:
            print(
                "[OceanSim] absolute bathymetric terrain matching enabled: "
                f"{ARGS.terrain_match_window_pings} motion-compensated pings, "
                f"+/-{ARGS.terrain_match_translation_window_m:g} m x/y, "
                f"+/-{ARGS.terrain_match_yaw_window_deg:g} deg yaw, "
                f"prior {terrain_prior_model_kind}; "
                + (
                    "accepted matches first enter the cohort particle filter."
                    if terrain_backend_particle_tracker is not None
                    else "accepted matches enter only the global pose graph."
                ),
                flush=True,
            )
            if terrain_backend_particle_tracker is not None:
                print(
                    "[OceanSim] joint point-mass/cohort TRN enabled: "
                    f"{terrain_backend_particle_tracker.config.particles_per_cohort} "
                    "SE(2) particles/cohort, explicit null hypothesis, "
                    f"{terrain_backend_particle_tracker.config.minimum_independent_evidence} "
                    "nonoverlapping observations before GTSAM authorization; "
                    f"online output policy {ARGS.terrain_particle_online_output}; "
                    "particle output never authorizes a graph factor by itself.",
                    flush=True,
                )
            if backscatter_matcher is not None:
                ci_description = (
                    "adaptive minimum-logdet per patch"
                    if ARGS.backscatter_ci_bathymetry_weight < 0.0
                    else (
                        f"fixed {ARGS.backscatter_ci_bathymetry_weight:g}/"
                        f"{1.0 - ARGS.backscatter_ci_bathymetry_weight:g}"
                    )
                )
                print(
                    "[OceanSim] measured side-scan map refinement enabled: "
                    "log-radiometric trend removal + normalized correlation; "
                    "accepted texture and bathymetric fixes use covariance "
                    "intersection with bathymetry/backscatter weighting "
                    f"{ci_description} "
                    "(same-survey upper bound).",
                    flush=True,
                )
            if terrain_recovery_matcher is not None:
                print(
                    "[OceanSim] strict-rank-three terrain recovery enabled: "
                    f"after {ARGS.terrain_recovery_after_rejections} "
                    "consecutive observations without an inserted prior, "
                    f"retry the same {ARGS.terrain_match_window_pings}-ping "
                    "patch at +/-"
                    f"{ARGS.terrain_recovery_translation_window_m:g} m; "
                    "10 m coarse step, 16 candidates, 20 m coarse-seed "
                    "separation, 5 m refined-basin separation, and 5 m "
                    "boundary margin.",
                    flush=True,
                )
            if terrain_emergency_recovery_matcher is not None:
                print(
                    "[OceanSim] confirmed emergency terrain recovery enabled: "
                    f"after {terrain_emergency_recovery.failed_prior_threshold} "
                    "failed prior observations, at least "
                    f"{terrain_emergency_recovery.minimum_raw_arc_m:g} m raw "
                    "estimated arc and "
                    f"{terrain_emergency_recovery.minimum_time_since_prior_s:g} s, "
                    "retry at +/-"
                    f"{terrain_emergency_recovery.recovery_translation_window_m:g} m; "
                    "the first strict rank-three match is held pending and a "
                    f"second strip within {terrain_emergency_recovery.confirmation_max_future_observations} "
                    "observations, at least "
                    f"{terrain_emergency_recovery.confirmation_minimum_time_separation_s:g} s later, "
                    "must agree within "
                    f"{terrain_emergency_recovery.confirmation_position_tolerance_m:g} m/"
                    f"{terrain_emergency_recovery.confirmation_yaw_tolerance_deg:g} deg.",
                    flush=True,
                )
            if global_terrain_relocalizer is not None:
                cohort_description = (
                    ""
                    if global_cohort_tracker is None
                    else (
                        "; persistent cohort particle filter enabled with "
                        f"up to {global_cohort_tracker.config.maximum_cohorts} "
                        "modes and "
                        f"{global_cohort_tracker.config.particles_per_cohort} "
                        "SE(2) particles per mode, explicit null state, and "
                        f"{global_cohort_tracker.config.minimum_independent_evidence} "
                        "nonoverlapping windows required"
                    )
                )
                bbs3d_description = (
                    ""
                    if bbs3d_candidate_verifier is None
                    else (
                        "; every cohort proposal is independently checked by "
                        "the pinned official 3D-BBS GPU solver over +/-"
                        f"{bbs3d_candidate_verifier.config.maximum_translation_m:g} m/"
                        f"{bbs3d_candidate_verifier.config.maximum_yaw_deg:g} deg "
                        "with pressure depth and IMU roll/pitch fixed"
                    )
                )
                print(
                    "[OceanSim] whole-map joint terrain relocalization enabled: "
                    f"{ARGS.terrain_global_window_pings} acoustic-inertially deskewed pings "
                    f"({ARGS.terrain_global_window_pings / ARGS.ping_hz:.0f} s, "
                    f"nominal {ARGS.terrain_global_window_pings * ARGS.survey_speed_mps / ARGS.ping_hz:.0f} m), "
                    f"{global_terrain_relocalizer.resolution_m:g} m proposal grid, "
                    f"+/-{ARGS.terrain_global_yaw_window_deg:g} deg around "
                    "local acoustic-inertial yaw; joint bathymetry/backscatter proposals require "
                    "5 m dual-modal refinement (global side-scan window "
                    f"+/-{global_backscatter_refinement_matcher.config.maximum_translation_m:g} m/"
                    f"{global_backscatter_refinement_matcher.config.maximum_yaw_deg:g} deg) "
                    "and a disjoint-strip confirmation"
                    f"{cohort_description}{bbs3d_description}.",
                    flush=True,
                )

        while simulation_app.is_running():
            if ARGS.duration and time.monotonic() - started >= ARGS.duration:
                run_status = "completed"
                termination_reason = "requested_wall_duration_reached"
                break
            if ARGS.max_frames and frame_count >= ARGS.max_frames:
                run_status = "completed"
                termination_reason = "requested_max_frames_reached"
                break
            if (
                survey_segment_duration_s is not None
                and sim_time + 1.0e-10 >= survey_segment_duration_s
            ):
                run_status = "completed"
                termination_reason = "requested_survey_segment_reached"
                break
            if ARGS.survey_laps and sim_time >= ARGS.survey_laps * survey.period:
                run_status = "completed"
                termination_reason = "requested_survey_laps_reached"
                break
            # Drive every sensor from an absolute event clock.  A fractional
            # render-frame tail must never become an extra IMU step or satisfy
            # a rounded modulo test: at accelerated display rates that pattern
            # duplicated DVL/depth observations and biased the estimator.
            target_time = (frame_count + 1) * ARGS.time_scale / 60.0
            if survey_segment_duration_s is not None:
                target_time = min(target_time, survey_segment_duration_s)
            event_loop_started = time.perf_counter()
            while True:
                event_time = min(next_imu, next_dvl, next_depth, next_ping)
                if event_time > target_time + 1.0e-10:
                    break
                sim_time = event_time
                truth = survey_state(sim_time)

                if next_imu <= sim_time + 1.0e-10:
                    gyro, acceleration = navigation.imu(truth)
                    slam.process_imu(sim_time, gyro, acceleration)
                    dead_reckoning.propagate(sim_time, gyro, acceleration)
                    latest_gyro = gyro
                    latest_acceleration = acceleration
                    imu_sample_count += 1
                    next_imu = (imu_sample_count + 1) * imu_dt
                if next_dvl <= sim_time + 1.0e-10:
                    dvl = navigation.dvl(truth)
                    slam.process_dvl(dvl)
                    dead_reckoning.update_dvl(dvl)
                    dvl_sample_count += 1
                    next_dvl = (dvl_sample_count + 1) * dvl_period
                if next_depth <= sim_time + 1.0e-10:
                    depth = navigation.depth(truth)
                    slam.process_depth(depth)
                    dead_reckoning.update_depth(depth)
                    depth_sample_count += 1
                    next_depth = (depth_sample_count + 1) * depth_period
                if sim_time + 1.0e-9 >= next_ping:
                    if slam.state.timestamp < sim_time - 1.0e-12:
                        # Sonar clocks need not be integer multiples of the
                        # 50 Hz IMU clock. Propagate to the exact acoustic
                        # timestamp with a zero-order hold of the latest IMU
                        # sample, without inventing an additional sensor
                        # sample or changing the reported IMU rate/count.
                        slam.process_imu(
                            sim_time, latest_gyro, latest_acceleration
                        )
                        dead_reckoning.propagate(
                            sim_time, latest_gyro, latest_acceleration
                        )
                    # The USD vehicle transform is needed by the acoustic
                    # launch, but writing it at every 50 Hz IMU-only event is
                    # pure UI/stage overhead: navigation sensors consume the
                    # analytic truth state above and no render occurs inside
                    # this event loop.  The end-of-frame pose is still written
                    # below so the visible AUV remains current.
                    stage_pose_started = time.perf_counter()
                    _set_vehicle_pose(
                        vehicle_translate,
                        vehicle_rotate,
                        truth.position,
                        truth.rotation,
                    )
                    performance_wall_s["acoustic_stage_pose"] += (
                        time.perf_counter() - stage_pose_started
                    )
                    following_ping_time = (ping_count + 1) * ping_period
                    final_ping_before_render = (
                        following_ping_time > target_time + 1.0e-10
                    )
                    sonar_visualization_due = (
                        not ARGS.headless
                        and final_ping_before_render
                        and (
                            ARGS.ui_sonar_refresh_hz == 0.0
                            or sim_time + 1.0e-10
                            >= next_sonar_visualization
                        )
                    )
                    # PhysX scene queries require a committed parent
                    # transform.  The static Warp backend reads the authored
                    # USD pose directly and must not advance Kit once per
                    # ping; doing so adds an artificial physics clock and
                    # dominates accelerated headless missions.
                    if acoustic_backend is None or ARGS.verify_sonar_backend:
                        world.step(render=False)
                    sonar_scan_started = time.perf_counter()
                    sonar_ping = sonar.scan(
                        timestamp=sim_time,
                        update_visualization=sonar_visualization_due,
                    )
                    performance_wall_s["sonar_scan"] += (
                        time.perf_counter() - sonar_scan_started
                    )
                    if sonar_visualization_due:
                        sonar_visualization_updates += 1
                        if ARGS.ui_sonar_refresh_hz > 0.0:
                            next_sonar_visualization = (
                                math.floor(
                                    sim_time * ARGS.ui_sonar_refresh_hz
                                    + 1.0e-10
                                )
                                + 1
                            ) / ARGS.ui_sonar_refresh_hz
                    terrain_points_sensor = np.asarray(
                        sonar_ping.bathymetry.points_sensor,
                        dtype=np.float64,
                    ).reshape(-1, 3)
                    terrain_points_sensor = terrain_points_sensor[
                        np.isfinite(terrain_points_sensor).all(axis=1)
                    ]
                    (
                        raw_acoustic_points,
                        raw_acoustic_backscatter,
                        _raw_source_indices,
                    ) = sonar_ping.valid_acoustic_samples(frame="sensor")
                    slam_acoustic_points = raw_acoustic_points[
                        :: ARGS.sidescan_slam_stride
                    ]
                    slam_acoustic_backscatter = raw_acoustic_backscatter[
                        :: ARGS.sidescan_slam_stride
                    ]
                    # Snapshot the causal acoustic-inertial prediction before
                    # consuming this ping.  It may contain information from
                    # earlier overlapping pings, but never the observation it
                    # is about to deskew, avoiding an immediate self-update.
                    acoustic_predictive_state = slam.state.copy()
                    slam_update_started = time.perf_counter()
                    slam.process_integrated_ping(
                        sonar_ping,
                        sample_stride=ARGS.sidescan_slam_stride,
                    )
                    performance_wall_s["local_slam_update"] += (
                        time.perf_counter() - slam_update_started
                    )
                    observation_id = ping_count + 1
                    if terrain_matcher is not None and terrain_points_sensor.size:
                        # Terrain matching remains an absolute prior-map
                        # factor.  For causal fused deskew, each scan uses the
                        # state immediately before that scan's own update; the
                        # explicit inertial mode remains available as an
                        # ablation. A short window needs relative motion only,
                        # so either trajectory's arbitrary global gauge cancels.
                        patch_motion_state = (
                            acoustic_predictive_state
                            if ARGS.terrain_patch_motion_source
                            in ("acoustic-predictive", "acoustic-fixed-lag")
                            else dead_reckoning.state
                        )
                        stored_terrain_scan = (
                            patch_motion_state.copy(),
                            terrain_points_sensor.copy(),
                            slam_acoustic_points.copy(),
                            slam_acoustic_backscatter.copy(),
                            float(sonar_ping.timestamp),
                            int(observation_id),
                        )
                        stored_inertial_state = dead_reckoning.state.copy()
                        terrain_patch_scans.append(stored_terrain_scan)
                        terrain_patch_inertial_states.append(stored_inertial_state)
                        if global_terrain_relocalizer is not None:
                            # Both deques reference the same immutable scan
                            # snapshots, so the longer recovery history does not
                            # duplicate acoustic arrays in memory.
                            global_terrain_patch_scans.append(stored_terrain_scan)
                            global_terrain_patch_inertial_states.append(
                                stored_inertial_state
                            )
                    if (
                        terrain_matcher is not None
                        and len(terrain_patch_scans)
                        == ARGS.terrain_match_window_pings
                    ):
                        exact_geometry_keyframes = [
                            frame
                            for frame in slam.pose_graph.keyframes
                            if frame.loop_eligible
                            and frame.observation_id == observation_id
                            and abs(frame.timestamp - sim_time) <= 1.0e-9
                        ]
                        if len(exact_geometry_keyframes) == 1:
                            patch_start_timestamp = float(
                                terrain_patch_scans[0][4]
                            )
                            patch_end_timestamp = float(
                                terrain_patch_scans[-1][4]
                            )
                            patch_start_observation_id = int(
                                terrain_patch_scans[0][5]
                            )
                            patch_end_observation_id = int(
                                terrain_patch_scans[-1][5]
                            )
                            patch_motion_states = [
                                scan[0] for scan in terrain_patch_scans
                            ]
                            if (
                                ARGS.terrain_patch_motion_source
                                == "acoustic-fixed-lag"
                            ):
                                patch_motion_states = (
                                    fixed_lag_fused_deskew_states(
                                        patch_motion_states,
                                        list(terrain_patch_inertial_states),
                                    )
                                )
                            local_reference_state = (
                                patch_motion_states[-1].copy()
                            )
                            full_patch_body = np.vstack(
                                [
                                    motion_compensate_to_body(
                                        points,
                                        acquisition_state,
                                        local_reference_state,
                                        sensor_rotation_body=(
                                            slam.filter.sensor_rotation_body
                                        ),
                                        sensor_translation_body=(
                                            slam.filter.sensor_translation_body
                                        ),
                                    )
                                    for acquisition_state, (
                                        _stored_state,
                                        points,
                                        _imaging_points,
                                        _imaging_values,
                                        _acquisition_timestamp,
                                        _acquisition_observation_id,
                                    ) in zip(
                                        patch_motion_states,
                                        terrain_patch_scans,
                                    )
                                ]
                            )
                            (
                                patch_body,
                                last_terrain_patch_voxel_size_m,
                            ) = voxel_average_terrain_patch(
                                full_patch_body,
                                ARGS.terrain_match_maximum_points,
                                minimum_voxel_size_m=(
                                    ARGS.terrain_patch_minimum_voxel_size_m
                                ),
                            )
                            last_terrain_retained_full_swath_pings = len(
                                terrain_patch_scans
                            )
                            last_terrain_patch_raw_points = int(
                                full_patch_body.shape[0]
                            )
                            last_terrain_patch_reduced_points = int(
                                patch_body.shape[0]
                            )
                            backscatter_patch_body = np.empty(
                                (0, 3), dtype=np.float64
                            )
                            backscatter_patch_values = np.empty(
                                0, dtype=np.float64
                            )
                            if backscatter_matcher is not None:
                                backscatter_point_chunks = []
                                backscatter_value_chunks = []
                                for acquisition_state, (
                                    _stored_state,
                                    _terrain_points,
                                    imaging_points,
                                    imaging_values,
                                    _acquisition_timestamp,
                                    _acquisition_observation_id,
                                ) in zip(
                                    patch_motion_states,
                                    terrain_patch_scans,
                                ):
                                    if imaging_points.size == 0:
                                        continue
                                    backscatter_point_chunks.append(
                                        motion_compensate_to_body(
                                            imaging_points,
                                            acquisition_state,
                                            local_reference_state,
                                            sensor_rotation_body=(
                                                slam.filter
                                                .sensor_rotation_body
                                            ),
                                            sensor_translation_body=(
                                                slam.filter
                                                .sensor_translation_body
                                            ),
                                        )
                                    )
                                    backscatter_value_chunks.append(
                                        np.asarray(
                                            imaging_values,
                                            dtype=np.float64,
                                        ).reshape(-1)
                                    )
                                if backscatter_point_chunks:
                                    full_backscatter_patch_body = np.vstack(
                                        backscatter_point_chunks
                                    )
                                    full_backscatter_patch_values = (
                                        np.concatenate(
                                            backscatter_value_chunks
                                        )
                                    )
                                    last_backscatter_patch_raw_points = int(
                                        full_backscatter_patch_body.shape[0]
                                    )
                                    (
                                        backscatter_patch_body,
                                        backscatter_patch_values,
                                        last_backscatter_patch_voxel_size_m,
                                    ) = voxel_average_backscatter_patch(
                                        full_backscatter_patch_body,
                                        full_backscatter_patch_values,
                                        ARGS.backscatter_match_maximum_points,
                                        # The prior backscatter raster has a
                                        # 5 m ground sample distance. Keeping
                                        # denser cells would only duplicate
                                        # correlated evidence.
                                        minimum_voxel_size_m=5.0,
                                    )
                                    last_backscatter_patch_reduced_points = (
                                        int(backscatter_patch_body.shape[0])
                                    )
                            # Match against the actual graph target.  The
                            # map->odom soft follower is an output/UI contract
                            # only; using it here would make an already
                            # verified correction needlessly lag the matcher.
                            global_rotation_seed, global_position_seed = (
                                slam.raw_global_pose()
                            )
                            global_seed_state = local_reference_state.copy()
                            global_seed_state.rotation = global_rotation_seed
                            global_seed_state.position = global_position_seed
                            terrain_match_attempts += 1
                            observation_matcher_accepted = False
                            terrain_prior_added = False
                            terrain_prior_held = False
                            recovery_invoked = False
                            emergency_recovery_invoked = False
                            preceding_factor_rejected = False

                            match_started = time.perf_counter()
                            terrain_result = terrain_matcher.match(
                                patch_body,
                                global_seed_state,
                                points_are_body_frame=True,
                            )
                            match_elapsed_s = (
                                time.perf_counter() - match_started
                            )
                            record_terrain_match_invocation(
                                "normal",
                                terrain_result,
                                match_elapsed_s,
                                truth_state=truth,
                            )
                            last_terrain_match = terrain_result
                            last_terrain_match_mode = "normal"
                            if (
                                not terrain_result.accepted
                                and terrain_result.rejection_reason
                                == "yaw_unobservable"
                                and terrain_position_fallback_matcher
                                is not None
                            ):
                                # Re-run the identical deterministic solve with
                                # rank-two output permitted. This is necessary
                                # because the strict matcher stops at the yaw
                                # gate before evaluating its remaining safety
                                # checks (boundary, residual, ambiguity). The
                                # fallback therefore does not bypass any gate.
                                match_started = time.perf_counter()
                                terrain_result = (
                                    terrain_position_fallback_matcher.match(
                                        patch_body,
                                        global_seed_state,
                                        points_are_body_frame=True,
                                    )
                                )
                                match_elapsed_s = (
                                    time.perf_counter() - match_started
                                )
                                fallback_stats = (
                                    terrain_position_fallback_stats
                                )
                                fallback_stats["invocations"] += 1
                                fallback_stats["wall_time_s"] += (
                                    match_elapsed_s
                                )
                                if terrain_result.accepted:
                                    fallback_stats["accepts"] += 1
                                else:
                                    fallback_stats["rejections"] += 1
                                    fallback_stats["rejection_reasons"][
                                        terrain_result.rejection_reason
                                        or "unspecified"
                                    ] += 1
                                record_terrain_match_invocation(
                                    "normal",
                                    terrain_result,
                                    match_elapsed_s,
                                    truth_state=truth,
                                )
                                last_terrain_match = terrain_result
                                last_terrain_match_mode = (
                                    "normal_position_fallback"
                                )
                            preceding_recovery_rejection_reason = (
                                terrain_result.rejection_reason
                            )
                            if terrain_result.accepted:
                                observation_matcher_accepted = True
                                matched_pose2_override = None
                                information_override = None
                                yaw_observable_override = None
                                cross_modal_compatible = True
                                backscatter_refinement_accepted = False
                                backscatter_refinement_result = None
                                information_override_includes_model_floors = (
                                    False
                                )
                                if backscatter_matcher is not None:
                                    (
                                        matched_pose2_override,
                                        information_override,
                                        yaw_observable_override,
                                        cross_modal_compatible,
                                        information_override_includes_model_floors,
                                        backscatter_refinement_accepted,
                                        backscatter_refinement_result,
                                    ) = refine_terrain_fix_with_backscatter(
                                        terrain_result,
                                        backscatter_patch_body,
                                        backscatter_patch_values,
                                        truth_state=truth,
                                    )
                                if cross_modal_compatible:
                                    (
                                        terrain_prior_added,
                                        terrain_prior_held,
                                    ) = try_add_terrain_prior_result(
                                        terrain_result,
                                        mode="normal",
                                        timestamp=sim_time,
                                        observation_id=observation_id,
                                        patch_body=patch_body,
                                        patch_reference_state=(
                                            local_reference_state
                                        ),
                                        backscatter_patch_body=(
                                            backscatter_patch_body
                                        ),
                                        backscatter_patch_values=(
                                            backscatter_patch_values
                                        ),
                                        patch_start_timestamp=(
                                            patch_start_timestamp
                                        ),
                                        patch_end_timestamp=patch_end_timestamp,
                                        patch_start_observation_id=(
                                            patch_start_observation_id
                                        ),
                                        patch_end_observation_id=(
                                            patch_end_observation_id
                                        ),
                                        matched_pose2_override=(
                                            matched_pose2_override
                                        ),
                                        information_override=(
                                            information_override
                                        ),
                                        information_override_includes_model_floors=(
                                            information_override_includes_model_floors
                                        ),
                                        yaw_observable_override=(
                                            yaw_observable_override
                                        ),
                                        backscatter_result=(
                                            backscatter_refinement_result
                                            if backscatter_refinement_accepted
                                            else None
                                        ),
                                        evaluation_truth_state=truth,
                                    )
                                else:
                                    terrain_prior_factor_stats["attempts"] += 1
                                    terrain_prior_factor_stats[
                                        "rejections"
                                    ] += 1
                                    terrain_prior_factor_stats[
                                        "rejection_reasons"
                                    ]["cross_modal_compatibility_veto"] += 1
                                    terrain_match_mode_stats["normal"][
                                        "prior_factor_attempts"
                                    ] += 1
                                    terrain_match_mode_stats["normal"][
                                        "prior_factor_rejections"
                                    ] += 1
                                preceding_factor_rejected = bool(
                                    not terrain_prior_added
                                    and not terrain_prior_held
                                )
                                if preceding_factor_rejected:
                                    preceding_recovery_rejection_reason = (
                                        "preceding_factor_rejected"
                                    )

                            if (
                                not terrain_result.accepted
                                and terrain_recovery.should_attempt_recovery(
                                    normal_prior_added=terrain_prior_added
                                )
                            ):
                                # This is intentionally the identical 300-ping
                                # motion-compensated patch and identical graph
                                # seed.  Recovery expands only the bounded
                                # terrain search; it does not use truth or a
                                # longer temporal window.
                                recovery_invoked = True
                                failed_prior_streak_before = (
                                    terrain_recovery
                                    .consecutive_failed_observations
                                )
                                match_started = time.perf_counter()
                                terrain_result = terrain_recovery_matcher.match(
                                    patch_body,
                                    global_seed_state,
                                    points_are_body_frame=True,
                                )
                                match_elapsed_s = (
                                    time.perf_counter() - match_started
                                )
                                record_terrain_match_invocation(
                                    "recovery",
                                    terrain_result,
                                    match_elapsed_s,
                                    truth_state=truth,
                                )
                                last_terrain_match = terrain_result
                                last_terrain_match_mode = "recovery"
                                preceding_recovery_rejection_reason = (
                                    terrain_result.rejection_reason
                                )
                                recovery_prior_attempted = False
                                recovery_graph_diagnostics = None
                                if terrain_result.accepted:
                                    observation_matcher_accepted = True
                                    recovery_prior_attempted = True
                                    (
                                        terrain_prior_added,
                                        terrain_prior_held,
                                    ) = try_add_terrain_prior_result(
                                        terrain_result,
                                        mode="recovery",
                                        timestamp=sim_time,
                                        observation_id=observation_id,
                                        patch_body=patch_body,
                                        patch_reference_state=(
                                            local_reference_state
                                        ),
                                        backscatter_patch_body=(
                                            backscatter_patch_body
                                        ),
                                        backscatter_patch_values=(
                                            backscatter_patch_values
                                        ),
                                        patch_start_timestamp=(
                                            patch_start_timestamp
                                        ),
                                        patch_end_timestamp=(
                                            patch_end_timestamp
                                        ),
                                        patch_start_observation_id=(
                                            patch_start_observation_id
                                        ),
                                        patch_end_observation_id=(
                                            patch_end_observation_id
                                        ),
                                        evaluation_truth_state=truth,
                                    )
                                    recovery_graph_diagnostics = (
                                        slam.pose_graph.diagnostics()
                                    )
                                    preceding_factor_rejected = (
                                        not terrain_prior_added
                                        and not terrain_prior_held
                                    )
                                    if preceding_factor_rejected:
                                        preceding_recovery_rejection_reason = (
                                            "preceding_factor_rejected"
                                        )

                                # Preserve every wide-window invocation, not
                                # only successes.  These records make a long
                                # run independently auditable without using
                                # truth inside the matching or graph gates.
                                terrain_recovery_invocation_records.append(
                                    {
                                        "mode": "recovery",
                                        "timestamp_s": float(sim_time),
                                        "observation_id": int(observation_id),
                                        "failed_prior_streak_before": int(
                                            failed_prior_streak_before
                                        ),
                                        "patch_ping_count": int(
                                            len(terrain_patch_scans)
                                        ),
                                        "patch_start_timestamp_s": (
                                            patch_start_timestamp
                                        ),
                                        "patch_end_timestamp_s": (
                                            patch_end_timestamp
                                        ),
                                        "patch_start_observation_id": (
                                            patch_start_observation_id
                                        ),
                                        "patch_end_observation_id": (
                                            patch_end_observation_id
                                        ),
                                        "patch_point_count": int(
                                            patch_body.shape[0]
                                        ),
                                        "match_wall_time_s": float(
                                            match_elapsed_s
                                        ),
                                        "seed_pose2": [
                                            float(global_position_seed[0]),
                                            float(global_position_seed[1]),
                                            float(rpy_from_rotation(
                                                global_rotation_seed
                                            )[2]),
                                        ],
                                        "matcher_accepted": bool(
                                            terrain_result.accepted
                                        ),
                                        "matcher_rejection_reason": (
                                            terrain_result.rejection_reason
                                        ),
                                        "correction_xyyaw": [
                                            _finite_float_or_none(value)
                                            for value in terrain_result.correction
                                        ],
                                        "translation_correction_m": (
                                            _finite_float_or_none(
                                                terrain_result
                                                .translation_correction_m
                                            )
                                        ),
                                        "yaw_correction_deg": (
                                            _finite_float_or_none(
                                                terrain_result
                                                .yaw_correction_deg
                                            )
                                        ),
                                        "matched_pose2": [
                                            _finite_float_or_none(value)
                                            for value in terrain_result
                                            .matched_pose2
                                        ],
                                        "correspondences": int(
                                            terrain_result.correspondences
                                        ),
                                        "inliers": int(
                                            terrain_result.inliers
                                        ),
                                        "observable_rank": int(
                                            terrain_result.observable_rank
                                        ),
                                        "translation_observable_rank": int(
                                            terrain_result
                                            .translation_observable_rank
                                        ),
                                        "yaw_observable": bool(
                                            terrain_result.yaw_observable
                                        ),
                                        "rmse_before_m": _finite_float_or_none(
                                            terrain_result.rmse_before
                                        ),
                                        "rmse_after_m": _finite_float_or_none(
                                            terrain_result.rmse_after
                                        ),
                                        "reduced_chi_square": (
                                            _finite_float_or_none(
                                                terrain_result
                                                .reduced_chi_square
                                            )
                                        ),
                                        "reached_search_boundary": bool(
                                            terrain_result
                                            .reached_search_boundary
                                        ),
                                        "distinct_basin_count": int(
                                            terrain_result
                                            .distinct_basin_count
                                        ),
                                        "best_basin_cost": (
                                            _finite_float_or_none(
                                                terrain_result
                                                .best_basin_cost
                                            )
                                        ),
                                        "second_best_basin_cost": (
                                            _finite_float_or_none(
                                                terrain_result
                                                .second_best_basin_cost
                                            )
                                        ),
                                        "basin_log_likelihood_ratio": (
                                            _finite_float_or_none(
                                                terrain_result
                                                .basin_log_likelihood_ratio
                                            )
                                        ),
                                        "minimum_basin_log_likelihood_ratio": (
                                            _finite_float_or_none(
                                                terrain_result
                                                .minimum_basin_log_likelihood_ratio
                                            )
                                        ),
                                        "graph_prior_attempted": bool(
                                            recovery_prior_attempted
                                        ),
                                        "graph_optimize_interval": 1,
                                        "graph_prior_accepted": bool(
                                            recovery_prior_attempted
                                            and terrain_prior_added
                                        ),
                                        "graph_rejection_reason": (
                                            None
                                            if recovery_graph_diagnostics is None
                                            else recovery_graph_diagnostics
                                            .last_terrain_prior_rejection_reason
                                        ),
                                        "graph_prefit_performed": bool(
                                            recovery_graph_diagnostics
                                            is not None
                                            and recovery_graph_diagnostics
                                            .last_terrain_prefit_performed
                                        ),
                                        "graph_prefit_bypassed_for_relocalization": bool(
                                            recovery_graph_diagnostics
                                            is not None
                                            and recovery_graph_diagnostics
                                            .last_terrain_prefit_bypassed_for_relocalization
                                        ),
                                        "graph_prefit_chi2": (
                                            None
                                            if recovery_graph_diagnostics is None
                                            or not recovery_graph_diagnostics
                                            .last_terrain_prefit_performed
                                            else _finite_float_or_none(
                                                recovery_graph_diagnostics
                                                .last_terrain_prefit_chi2
                                            )
                                        ),
                                        "graph_prefit_dof": (
                                            None
                                            if recovery_graph_diagnostics is None
                                            or not recovery_graph_diagnostics
                                            .last_terrain_prefit_performed
                                            else int(
                                                recovery_graph_diagnostics
                                                .last_terrain_prefit_dof
                                            )
                                        ),
                                    }
                                )

                            raw_pose2 = np.array(
                                [
                                    dead_reckoning.state.position[0],
                                    dead_reckoning.state.position[1],
                                    rpy_from_rotation(
                                        dead_reckoning.state.rotation
                                    )[2],
                                ],
                                dtype=np.float64,
                            )
                            emergency_request = (
                                terrain_emergency_recovery
                                .observe_preceding_recovery_outcome(
                                    timestamp=sim_time,
                                    raw_pose2=raw_pose2,
                                    prior_added=terrain_prior_added,
                                    preceding_recovery_rejection_reason=(
                                        preceding_recovery_rejection_reason
                                    ),
                                    preceding_recovery_invoked=(
                                        recovery_invoked
                                    ),
                                    preceding_factor_rejected=(
                                        preceding_factor_rejected
                                    ),
                                )
                            )
                            if (
                                terrain_prior_added
                                and global_cohort_tracker is not None
                            ):
                                # Any successfully inserted absolute prior has
                                # re-established the graph gauge.  Old outage
                                # cohorts are then stale and must not survive
                                # into a later, unrelated loss episode.
                                global_cohort_tracker.reset()
                            if emergency_request.should_attempt:
                                emergency_outage_snapshot = {
                                    "observation_index": int(
                                        terrain_emergency_recovery
                                        .observation_index
                                    ),
                                    "consecutive_without_prior": int(
                                        terrain_emergency_recovery
                                        .consecutive_without_prior
                                    ),
                                    "estimated_raw_arc_since_prior_m": float(
                                        terrain_emergency_recovery
                                        .raw_arc_since_prior_m
                                    ),
                                    "estimated_time_since_prior_s": float(
                                        terrain_emergency_recovery
                                        .time_since_prior_s
                                    ),
                                    "pending_before_match": bool(
                                        terrain_emergency_recovery
                                        .has_pending_hypothesis
                                    ),
                                }
                                # This final bounded layer still receives the
                                # identical estimated-state patch and graph
                                # seed.  Its first rank-three solution is held
                                # as a hypothesis; only a later independent
                                # strip can authorize the current result for
                                # graph prefit and insertion.
                                emergency_recovery_invoked = True
                                match_started = time.perf_counter()
                                terrain_result = (
                                    terrain_emergency_recovery_matcher.match(
                                        patch_body,
                                        global_seed_state,
                                        points_are_body_frame=True,
                                    )
                                )
                                match_elapsed_s = (
                                    time.perf_counter() - match_started
                                )
                                record_terrain_match_invocation(
                                    "emergency",
                                    terrain_result,
                                    match_elapsed_s,
                                    truth_state=truth,
                                )
                                confirmation_result = terrain_result
                                confirmation_mode = "emergency"
                                confirmation_matched_pose2 = np.asarray(
                                    terrain_result.matched_pose2,
                                    dtype=np.float64,
                                )
                                confirmation_information_override = None
                                confirmation_information_has_floors = False
                                confirmation_yaw_observable = bool(
                                    terrain_result.yaw_observable
                                )
                                confirmation_observable_rank = int(
                                    terrain_result.observable_rank
                                )
                                confirmation_translation_observable_rank = int(
                                    terrain_result.translation_observable_rank
                                )
                                confirmation_accepted = bool(
                                    terrain_result.accepted
                                )
                                confirmation_patch_body = patch_body
                                confirmation_patch_ping_count = len(
                                    terrain_patch_scans
                                )
                                confirmation_patch_start_timestamp = (
                                    patch_start_timestamp
                                )
                                confirmation_patch_end_timestamp = (
                                    patch_end_timestamp
                                )
                                confirmation_patch_start_observation_id = (
                                    patch_start_observation_id
                                )
                                confirmation_patch_end_observation_id = (
                                    patch_end_observation_id
                                )
                                global_search_result = None
                                global_patch = None
                                pending_confirmation_source = (
                                    terrain_emergency_recovery
                                    .pending_confirmation_source
                                )

                                # A bounded failure may mean the navigation
                                # estimate has simply left every local search
                                # window.  The rare whole-map layer uses a much
                                # longer strip deskewed only by IMU/DVL/depth,
                                # proposes on a 20 m pyramid, and then returns to
                                # the ordinary 5 m matchers for metric refinement.
                                if (
                                    global_terrain_relocalizer is not None
                                    and (
                                        (
                                            not confirmation_accepted
                                            and pending_confirmation_source
                                            in (None, "global")
                                        )
                                        or pending_confirmation_source
                                        == "global"
                                    )
                                ):
                                    # Once a global hypothesis is pending, a
                                    # bathymetry-only bounded result cannot
                                    # self-confirm it.  This observation must
                                    # supply another complete global chain.
                                    if pending_confirmation_source == "global":
                                        confirmation_mode = "global"
                                        confirmation_accepted = False
                                    global_patch = (
                                        build_global_relocalization_patch()
                                    )
                                    if global_patch is not None:
                                        global_started = time.perf_counter()
                                        global_search_result = (
                                            global_terrain_relocalizer.search(
                                                global_patch[
                                                    "terrain_points_body"
                                                ],
                                                global_patch[
                                                    "reference_state"
                                                ],
                                                backscatter_points_body=(
                                                    global_patch[
                                                        "backscatter_points_body"
                                                    ]
                                                ),
                                                backscatter_values=(
                                                    global_patch[
                                                        "backscatter_values"
                                                    ]
                                                ),
                                                yaw_center_rad=float(
                                                    rpy_from_rotation(
                                                        global_patch[
                                                            "reference_state"
                                                        ].rotation
                                                    )[2]
                                                ),
                                            )
                                        )
                                        global_elapsed_s = (
                                            time.perf_counter() - global_started
                                        )
                                        global_stats = global_relocalization_stats
                                        global_stats["invocations"] += 1
                                        global_stats["wall_time_s"] += (
                                            global_elapsed_s
                                        )
                                        global_stats[
                                            "last_best_score"
                                        ] = global_search_result.best_score
                                        global_stats[
                                            "last_second_basin_score"
                                        ] = (
                                            global_search_result
                                            .second_basin_score
                                        )
                                        global_stats[
                                            "last_score_gap"
                                        ] = global_search_result.basin_score_gap
                                        global_stats[
                                            "last_candidate_count"
                                        ] = len(global_search_result.candidates)
                                        global_stats[
                                            "last_evaluated_yaws"
                                        ] = global_search_result.evaluated_yaws
                                        global_stats[
                                            "last_finite_lattice_candidates"
                                        ] = (
                                            global_search_result
                                            .finite_lattice_candidates
                                        )
                                        if global_search_result.accepted:
                                            global_stats[
                                                "accepted_proposals"
                                            ] += 1
                                        else:
                                            global_stats[
                                                "rejected_proposals"
                                            ] += 1
                                            global_stats[
                                                "proposal_rejection_reasons"
                                            ][
                                                global_search_result
                                                .rejection_reason
                                                or "unspecified"
                                            ] += 1

                                        truth_pose2 = np.array(
                                            [
                                                truth.position[0],
                                                truth.position[1],
                                                rpy_from_rotation(
                                                    truth.rotation
                                                )[2],
                                            ],
                                            dtype=np.float64,
                                        )
                                        global_relocalization_records.append(
                                            {
                                                "timestamp_s": float(sim_time),
                                                "observation_id": int(
                                                    observation_id
                                                ),
                                                "accepted": bool(
                                                    global_search_result.accepted
                                                ),
                                                "rejection_reason": (
                                                    global_search_result
                                                    .rejection_reason
                                                ),
                                                "best_score": (
                                                    _finite_float_or_none(
                                                        global_search_result
                                                        .best_score
                                                    )
                                                ),
                                                "second_basin_score": (
                                                    _finite_float_or_none(
                                                        global_search_result
                                                        .second_basin_score
                                                    )
                                                ),
                                                "basin_score_gap": (
                                                    _finite_float_or_none(
                                                        global_search_result
                                                        .basin_score_gap
                                                    )
                                                ),
                                                "evaluated_yaws": int(
                                                    global_search_result
                                                    .evaluated_yaws
                                                ),
                                                "finite_lattice_candidates": int(
                                                    global_search_result
                                                    .finite_lattice_candidates
                                                ),
                                                "patch_ping_count": int(
                                                    len(
                                                        global_terrain_patch_scans
                                                    )
                                                ),
                                                "patch_start_timestamp_s": (
                                                    global_patch[
                                                        "patch_start_timestamp"
                                                    ]
                                                ),
                                                "patch_end_timestamp_s": (
                                                    global_patch[
                                                        "patch_end_timestamp"
                                                    ]
                                                ),
                                                "terrain_raw_points": (
                                                    global_patch[
                                                        "terrain_raw_points"
                                                    ]
                                                ),
                                                "terrain_reduced_points": int(
                                                    global_patch[
                                                        "terrain_points_body"
                                                    ].shape[0]
                                                ),
                                                "backscatter_raw_points": (
                                                    global_patch[
                                                        "backscatter_raw_points"
                                                    ]
                                                ),
                                                "backscatter_reduced_points": int(
                                                    global_patch[
                                                        "backscatter_points_body"
                                                    ].shape[0]
                                                ),
                                                "yaw_center_rad": float(
                                                    rpy_from_rotation(
                                                        global_patch[
                                                            "reference_state"
                                                        ].rotation
                                                    )[2]
                                                ),
                                                "candidates": [
                                                    {
                                                        "pose2": (
                                                            candidate.pose2.tolist()
                                                        ),
                                                        "joint_correlation": float(
                                                            candidate
                                                            .joint_correlation
                                                        ),
                                                        "bathymetry_correlation": float(
                                                            candidate
                                                            .bathymetry_correlation
                                                        ),
                                                        "bathymetry_evidence_score": float(
                                                            candidate
                                                            .bathymetry_evidence_score
                                                        ),
                                                        "bathymetry_rmse_m": float(
                                                            candidate
                                                            .bathymetry_rmse_m
                                                        ),
                                                        "backscatter_correlation": float(
                                                            candidate
                                                            .backscatter_correlation
                                                        ),
                                                        "horizontal_error_m": float(
                                                            np.linalg.norm(
                                                                candidate.pose2[
                                                                    :2
                                                                ]
                                                                - truth_pose2[
                                                                    :2
                                                                ]
                                                            )
                                                        ),
                                                        "yaw_error_deg": float(
                                                            np.rad2deg(
                                                                math.atan2(
                                                                    math.sin(
                                                                        candidate.pose2[
                                                                            2
                                                                        ]
                                                                        - truth_pose2[
                                                                            2
                                                                        ]
                                                                    ),
                                                                    math.cos(
                                                                        candidate.pose2[
                                                                            2
                                                                        ]
                                                                        - truth_pose2[
                                                                            2
                                                                        ]
                                                                    ),
                                                                )
                                                            )
                                                        ),
                                                    }
                                                    for candidate in (
                                                        global_search_result
                                                        .candidates
                                                    )
                                                ],
                                                "truth_evaluation_only": True,
                                                "truth_used_by_search": False,
                                                "wall_time_s": float(
                                                    global_elapsed_s
                                                ),
                                            }
                                        )

                                        proposal_authorized = bool(
                                            global_search_result.accepted
                                        )
                                        proposal_index = 0
                                        global_tracking_decision = None
                                        global_candidate_verifications = {}
                                        if global_cohort_tracker is not None:
                                            cohort_evidence_list = []
                                            for (
                                                candidate_index,
                                                candidate,
                                            ) in enumerate(
                                                global_search_result.candidates
                                            ):
                                                verification = (
                                                    verify_global_candidate(
                                                        candidate,
                                                        global_patch,
                                                        truth_state=truth,
                                                    )
                                                )
                                                global_candidate_verifications[
                                                    candidate_index
                                                ] = verification
                                                global_stats[
                                                    "candidate_verification_invocations"
                                                ] += 1
                                                if not verification["accepted"]:
                                                    global_stats[
                                                        "candidate_verification_rejections"
                                                    ] += 1
                                                    continue
                                                global_stats[
                                                    "candidate_verification_accepts"
                                                ] += 1
                                                verification_covariance = (
                                                    reporting_covariance_from_information(
                                                        verification[
                                                            "information"
                                                        ],
                                                        yaw_scale_m=(
                                                            terrain_particle_likelihood_config
                                                            .yaw_rank_scale_m
                                                        ),
                                                        unobservable_variance_scaled_m2=(
                                                            terrain_particle_likelihood_config
                                                            .unobservable_variance_scaled_m2
                                                        ),
                                                    )
                                                )
                                                position_variance = max(
                                                    0.0,
                                                    float(
                                                        np.linalg.eigvalsh(
                                                            verification_covariance[
                                                                :2, :2
                                                            ]
                                                        )[-1]
                                                    ),
                                                )
                                                position_std_m = max(
                                                    0.5
                                                    * world_model.resolution_m,
                                                    math.sqrt(
                                                        position_variance
                                                    ),
                                                )
                                                yaw_std_deg = max(
                                                    0.5,
                                                    math.degrees(
                                                        math.sqrt(
                                                            max(
                                                                0.0,
                                                                float(
                                                                    verification_covariance[
                                                                        2, 2
                                                                    ]
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                )
                                                backscatter_result = (
                                                    verification[
                                                        "backscatter_result"
                                                    ]
                                                )
                                                cohort_evidence_list.append(
                                                    GlobalPoseEvidence(
                                                        map_pose2=verification[
                                                            "matched_pose2"
                                                        ],
                                                        bathymetry_score=float(
                                                            candidate
                                                            .bathymetry_evidence_score
                                                        ),
                                                        backscatter_score=float(
                                                            backscatter_result
                                                            .correlation_after
                                                        ),
                                                        bathymetry_observability=(
                                                            global_search_result
                                                            .bathymetry_observability
                                                            * min(
                                                                1.0,
                                                                verification[
                                                                    "translation_observable_rank"
                                                                ]
                                                                / 2.0,
                                                            )
                                                        ),
                                                        backscatter_observability=(
                                                            verification[
                                                                "backscatter_observability"
                                                            ]
                                                        ),
                                                        geometry_score=(
                                                            verification[
                                                                "geometry_score"
                                                            ]
                                                        ),
                                                        cross_modal_consistent=True,
                                                        position_std_m=(
                                                            position_std_m
                                                        ),
                                                        yaw_std_deg=yaw_std_deg,
                                                        patch_start_timestamp_s=(
                                                            global_patch[
                                                                "patch_start_timestamp"
                                                            ]
                                                        ),
                                                        patch_end_timestamp_s=(
                                                            global_patch[
                                                                "patch_end_timestamp"
                                                            ]
                                                        ),
                                                        patch_start_observation_id=(
                                                            global_patch[
                                                                "patch_start_observation_id"
                                                            ]
                                                        ),
                                                        patch_end_observation_id=(
                                                            global_patch[
                                                                "patch_end_observation_id"
                                                            ]
                                                        ),
                                                        source_index=(
                                                            candidate_index
                                                        ),
                                                        covariance_matrix=(
                                                            verification_covariance
                                                        ),
                                                    )
                                                )
                                            cohort_evidences = tuple(
                                                cohort_evidence_list
                                            )
                                            global_particle_likelihood = (
                                                AcousticTerrainParticleLikelihood(
                                                    terrain_prior_model,
                                                    global_patch[
                                                        "reference_state"
                                                    ],
                                                    global_patch[
                                                        "terrain_points_body"
                                                    ],
                                                    backscatter_terrain=(
                                                        world_model
                                                    ),
                                                    backscatter_points_body=(
                                                        global_patch[
                                                            "backscatter_points_body"
                                                        ]
                                                    ),
                                                    backscatter_values=(
                                                        global_patch[
                                                            "backscatter_values"
                                                        ]
                                                    ),
                                                    config=replace(
                                                        terrain_particle_likelihood_config,
                                                        bathymetry_measurement_std_m=max(
                                                            terrain_prior_vertical_model_std_m,
                                                            float(
                                                                global_search_result
                                                                .bathymetry_observation_std_m
                                                            ),
                                                        ),
                                                    ),
                                                )
                                            )
                                            global_particle_observation = (
                                                ParticleLikelihoodObservation(
                                                    evaluator=(
                                                        global_particle_likelihood
                                                    ),
                                                    patch_start_timestamp_s=(
                                                        global_patch[
                                                            "patch_start_timestamp"
                                                        ]
                                                    ),
                                                    patch_end_timestamp_s=(
                                                        global_patch[
                                                            "patch_end_timestamp"
                                                        ]
                                                    ),
                                                    patch_start_observation_id=(
                                                        global_patch[
                                                            "patch_start_observation_id"
                                                        ]
                                                    ),
                                                    patch_end_observation_id=(
                                                        global_patch[
                                                            "patch_end_observation_id"
                                                        ]
                                                    ),
                                                )
                                            )
                                            local_acoustic_pose2 = np.array(
                                                [
                                                    slam.state.position[0],
                                                    slam.state.position[1],
                                                    rpy_from_rotation(
                                                        slam.state.rotation
                                                    )[2],
                                                ],
                                                dtype=np.float64,
                                            )
                                            global_tracking_decision = (
                                                global_cohort_tracker.update(
                                                    timestamp_s=sim_time,
                                                    raw_pose2=(
                                                        local_acoustic_pose2
                                                    ),
                                                    evidences=cohort_evidences,
                                                    particle_observation=(
                                                        global_particle_observation
                                                    ),
                                                )
                                            )
                                            global_stats["cohort_updates"] += 1
                                            global_stats[
                                                f"cohort_{global_tracking_decision.status}_updates"
                                            ] += 1
                                            global_stats[
                                                "last_cohort_count"
                                            ] = len(
                                                global_tracking_decision.cohorts
                                            )
                                            global_stats[
                                                "last_cohort_null_probability"
                                            ] = (
                                                global_tracking_decision
                                                .null_probability
                                            )
                                            global_stats[
                                                "last_cohort_best_probability"
                                            ] = (
                                                global_tracking_decision
                                                .best_probability
                                            )
                                            global_stats[
                                                "last_cohort_odds_over_second"
                                            ] = (
                                                global_tracking_decision
                                                .odds_over_second
                                            )
                                            confirmed_cohort = (
                                                global_tracking_decision
                                                .confirmed_cohort
                                            )
                                            proposal_authorized = bool(
                                                confirmed_cohort is not None
                                                and global_tracking_decision
                                                .confirmed_with_new_independent_evidence
                                                and confirmed_cohort
                                                .last_measurement_is_graph_ready
                                                and confirmed_cohort
                                                .last_source_index
                                                in global_candidate_verifications
                                                and global_candidate_verifications[
                                                    confirmed_cohort
                                                    .last_source_index
                                                ]["accepted"]
                                            )
                                            if proposal_authorized:
                                                proposal_index = int(
                                                    confirmed_cohort
                                                    .last_source_index
                                                )
                                            global_cohort_records.append(
                                                {
                                                    "timestamp_s": float(
                                                        sim_time
                                                    ),
                                                    "observation_id": int(
                                                        observation_id
                                                    ),
                                                    "status": (
                                                        global_tracking_decision
                                                        .status
                                                    ),
                                                    "proposal_authorized": bool(
                                                        proposal_authorized
                                                    ),
                                                    "selected_candidate_index": (
                                                        proposal_index
                                                        if proposal_authorized
                                                        else None
                                                    ),
                                                    "null_probability": float(
                                                        global_tracking_decision
                                                        .null_probability
                                                    ),
                                                    "best_probability": float(
                                                        global_tracking_decision
                                                        .best_probability
                                                    ),
                                                    "odds_over_second": float(
                                                        global_tracking_decision
                                                        .odds_over_second
                                                    ),
                                                    "bathymetry_observation_std_m": float(
                                                        global_search_result
                                                        .bathymetry_observation_std_m
                                                    ),
                                                    "backscatter_observation_log_std": float(
                                                        global_search_result
                                                        .backscatter_observation_log_std
                                                    ),
                                                    "bathymetry_observability": float(
                                                        global_search_result
                                                        .bathymetry_observability
                                                    ),
                                                    "backscatter_observability": float(
                                                        global_search_result
                                                        .backscatter_observability
                                                    ),
                                                    "candidate_verifications": [
                                                        {
                                                            "source_index": int(
                                                                source_index
                                                            ),
                                                            "accepted": bool(
                                                                verification[
                                                                    "accepted"
                                                                ]
                                                            ),
                                                            "bbs3d": (
                                                                None
                                                                if verification[
                                                                    "bbs3d_result"
                                                                ]
                                                                is None
                                                                else {
                                                                    "accepted": bool(
                                                                        verification[
                                                                            "bbs3d_result"
                                                                        ].accepted
                                                                    ),
                                                                    "score_fraction": float(
                                                                        verification[
                                                                            "bbs3d_result"
                                                                        ].score_fraction
                                                                    ),
                                                                    "correction_pose2": verification[
                                                                        "bbs3d_result"
                                                                    ].correction_pose2.tolist(),
                                                                    "source_points": int(
                                                                        verification[
                                                                            "bbs3d_result"
                                                                        ].source_points
                                                                    ),
                                                                    "target_points": int(
                                                                        verification[
                                                                            "bbs3d_result"
                                                                        ].target_points
                                                                    ),
                                                                    "elapsed_ms": float(
                                                                        verification[
                                                                            "bbs3d_result"
                                                                        ].elapsed_milliseconds
                                                                    ),
                                                                    "rejection_reason": verification[
                                                                        "bbs3d_result"
                                                                    ].rejection_reason,
                                                                }
                                                            ),
                                                            "bathymetry_accepted": bool(
                                                                verification[
                                                                    "terrain_result"
                                                                ]
                                                                is not None
                                                                and verification[
                                                                    "terrain_result"
                                                                ].accepted
                                                            ),
                                                            "backscatter_accepted": bool(
                                                                verification[
                                                                    "backscatter_result"
                                                                ]
                                                                is not None
                                                                and verification[
                                                                    "backscatter_result"
                                                                ].accepted
                                                            ),
                                                            "terrain_seed_source": (
                                                                verification[
                                                                    "terrain_seed_source"
                                                                ]
                                                            ),
                                                            "terrain_seed_attempts": (
                                                                verification[
                                                                    "terrain_seed_attempts"
                                                                ]
                                                            ),
                                                            "geometry_score": float(
                                                                verification[
                                                                    "geometry_score"
                                                                ]
                                                            ),
                                                        }
                                                        for source_index, verification in sorted(
                                                            global_candidate_verifications.items()
                                                        )
                                                    ],
                                                    "cohorts": [
                                                        {
                                                            "cohort_id": int(
                                                                cohort.cohort_id
                                                            ),
                                                            "map_pose2": (
                                                                cohort.map_pose2.tolist()
                                                            ),
                                                            "map_to_odom_pose2": (
                                                                cohort
                                                                .map_to_odom_pose2
                                                                .tolist()
                                                            ),
                                                            "probability": float(
                                                                cohort.probability
                                                            ),
                                                            "total_evidence": int(
                                                                cohort
                                                                .total_evidence
                                                            ),
                                                            "independent_evidence": int(
                                                                cohort
                                                                .independent_evidence
                                                            ),
                                                            "consecutive_misses": int(
                                                                cohort
                                                                .consecutive_misses
                                                            ),
                                                            "last_source_index": int(
                                                                cohort
                                                                .last_source_index
                                                            ),
                                                            "particle_count": int(
                                                                cohort
                                                                .particle_count
                                                            ),
                                                            "effective_sample_size": float(
                                                                cohort
                                                                .effective_sample_size
                                                            ),
                                                            "resampling_events": int(
                                                                cohort
                                                                .resampling_events
                                                            ),
                                                        }
                                                        for cohort in (
                                                            global_tracking_decision
                                                            .cohorts
                                                        )
                                                    ],
                                                    "uses_truth": False,
                                                }
                                            )

                                        if (
                                            proposal_authorized
                                            and global_cohort_tracker is not None
                                        ):
                                            selected_verification = (
                                                global_candidate_verifications[
                                                    proposal_index
                                                ]
                                            )
                                            refined_result = selected_verification[
                                                "terrain_result"
                                            ]
                                            match_elapsed_s = (
                                                selected_verification[
                                                    "refine_elapsed_s"
                                                ]
                                            )
                                            confirmation_result = refined_result
                                            confirmation_mode = "global"
                                            confirmation_accepted = bool(
                                                selected_verification["accepted"]
                                            )
                                            confirmation_matched_pose2 = np.asarray(
                                                selected_verification[
                                                    "matched_pose2"
                                                ],
                                                dtype=np.float64,
                                            )
                                            confirmation_information_override = np.asarray(
                                                selected_verification[
                                                    "information"
                                                ],
                                                dtype=np.float64,
                                            )
                                            confirmation_information_has_floors = bool(
                                                selected_verification[
                                                    "information_has_floors"
                                                ]
                                            )
                                            confirmation_yaw_observable = bool(
                                                selected_verification[
                                                    "yaw_observable"
                                                ]
                                            )
                                            confirmation_observable_rank = int(
                                                selected_verification[
                                                    "observable_rank"
                                                ]
                                            )
                                            confirmation_translation_observable_rank = int(
                                                selected_verification[
                                                    "translation_observable_rank"
                                                ]
                                            )
                                            confirmation_patch_body = global_patch[
                                                "terrain_points_body"
                                            ]
                                            confirmation_patch_ping_count = len(
                                                global_terrain_patch_scans
                                            )
                                            confirmation_patch_start_timestamp = (
                                                global_patch[
                                                    "patch_start_timestamp"
                                                ]
                                            )
                                            confirmation_patch_end_timestamp = (
                                                global_patch[
                                                    "patch_end_timestamp"
                                                ]
                                            )
                                            confirmation_patch_start_observation_id = (
                                                global_patch[
                                                    "patch_start_observation_id"
                                                ]
                                            )
                                            confirmation_patch_end_observation_id = (
                                                global_patch[
                                                    "patch_end_observation_id"
                                                ]
                                            )

                                        if (
                                            proposal_authorized
                                            and global_cohort_tracker is None
                                        ):
                                            proposal = (
                                                global_search_result.candidates[
                                                    proposal_index
                                                ]
                                            )
                                            proposal_state = global_patch[
                                                "reference_state"
                                            ].copy()
                                            proposal_rpy = rpy_from_rotation(
                                                proposal_state.rotation
                                            )
                                            proposal_state.rotation = (
                                                rotation_from_rpy(
                                                    float(proposal_rpy[0]),
                                                    float(proposal_rpy[1]),
                                                    float(proposal.pose2[2]),
                                                )
                                            )
                                            proposal_state.position[:2] = (
                                                proposal.pose2[:2]
                                            )
                                            refine_started = time.perf_counter()
                                            refined_result = terrain_matcher.match(
                                                global_patch[
                                                    "terrain_points_body"
                                                ],
                                                proposal_state,
                                                points_are_body_frame=True,
                                            )
                                            refine_elapsed_s = (
                                                time.perf_counter()
                                                - refine_started
                                            )
                                            record_terrain_match_invocation(
                                                "global",
                                                refined_result,
                                                refine_elapsed_s,
                                                truth_state=truth,
                                            )
                                            if (
                                                not refined_result.accepted
                                                and refined_result.rejection_reason
                                                == "yaw_unobservable"
                                                and terrain_position_fallback_matcher
                                                is not None
                                            ):
                                                global_stats[
                                                    "bathymetry_position_fallback_invocations"
                                                ] += 1
                                                refine_started = (
                                                    time.perf_counter()
                                                )
                                                refined_result = (
                                                    terrain_position_fallback_matcher
                                                    .match(
                                                        global_patch[
                                                            "terrain_points_body"
                                                        ],
                                                        proposal_state,
                                                        points_are_body_frame=True,
                                                    )
                                                )
                                                refine_elapsed_s = (
                                                    time.perf_counter()
                                                    - refine_started
                                                )
                                                record_terrain_match_invocation(
                                                    "global",
                                                    refined_result,
                                                    refine_elapsed_s,
                                                    truth_state=truth,
                                                )
                                                if refined_result.accepted:
                                                    global_stats[
                                                        "bathymetry_position_fallback_accepts"
                                                    ] += 1
                                                else:
                                                    global_stats[
                                                        "bathymetry_position_fallback_rejections"
                                                    ] += 1
                                                    global_stats[
                                                        "bathymetry_position_fallback_rejection_reasons"
                                                    ][
                                                        refined_result.rejection_reason
                                                        or "unspecified"
                                                    ] += 1
                                            confirmation_result = refined_result
                                            confirmation_mode = "global"
                                            confirmation_accepted = False
                                            confirmation_observable_rank = int(
                                                refined_result.observable_rank
                                            )
                                            confirmation_translation_observable_rank = int(
                                                refined_result
                                                .translation_observable_rank
                                            )
                                            if refined_result.accepted:
                                                global_stats[
                                                    "bathymetry_refinement_accepts"
                                                ] += 1
                                                (
                                                    refined_pose2,
                                                    refined_information,
                                                    refined_yaw_observable,
                                                    refined_compatible,
                                                    refined_information_has_floors,
                                                    refined_backscatter_accepted,
                                                    refined_backscatter_result,
                                                ) = refine_terrain_fix_with_backscatter(
                                                    refined_result,
                                                    global_patch[
                                                        "backscatter_points_body"
                                                    ],
                                                    global_patch[
                                                        "backscatter_values"
                                                    ],
                                                    truth_state=truth,
                                                    use_global_refinement=True,
                                                )
                                                # Whole-map proposals require
                                                # both modalities at both the
                                                # coarse and local stages.
                                                confirmation_accepted = bool(
                                                    refined_compatible
                                                    and refined_backscatter_accepted
                                                )
                                                if confirmation_accepted:
                                                    global_stats[
                                                        "dual_modal_refinement_accepts"
                                                    ] += 1
                                                    confirmation_mode = "global"
                                                    confirmation_matched_pose2 = (
                                                        refined_pose2
                                                    )
                                                    confirmation_information_override = (
                                                        refined_information
                                                    )
                                                    confirmation_information_has_floors = (
                                                        refined_information_has_floors
                                                    )
                                                    confirmation_yaw_observable = bool(
                                                        refined_yaw_observable
                                                    )
                                                    confirmation_observable_rank = (
                                                        3
                                                        if (
                                                            refined_backscatter_accepted
                                                            and refined_result
                                                            .translation_observable_rank
                                                            == 2
                                                        )
                                                        else int(
                                                            refined_result
                                                            .observable_rank
                                                        )
                                                    )
                                                    confirmation_patch_body = (
                                                        global_patch[
                                                            "terrain_points_body"
                                                        ]
                                                    )
                                                    confirmation_patch_ping_count = len(
                                                        global_terrain_patch_scans
                                                    )
                                                    confirmation_patch_start_timestamp = (
                                                        global_patch[
                                                            "patch_start_timestamp"
                                                        ]
                                                    )
                                                    confirmation_patch_end_timestamp = (
                                                        global_patch[
                                                            "patch_end_timestamp"
                                                        ]
                                                    )
                                                    confirmation_patch_start_observation_id = (
                                                        global_patch[
                                                            "patch_start_observation_id"
                                                        ]
                                                    )
                                                    confirmation_patch_end_observation_id = (
                                                        global_patch[
                                                            "patch_end_observation_id"
                                                        ]
                                                    )
                                                else:
                                                    global_stats[
                                                        "dual_modal_refinement_rejections"
                                                    ] += 1
                                                    rejection_reason = (
                                                        "backscatter_refinement_rejected"
                                                        if not refined_backscatter_accepted
                                                        else "cross_modal_incompatible"
                                                    )
                                                    global_stats[
                                                        "dual_modal_refinement_rejection_reasons"
                                                    ][rejection_reason] += 1
                                            else:
                                                global_stats[
                                                    "bathymetry_refinement_rejections"
                                                ] += 1
                                                global_stats[
                                                    "bathymetry_refinement_rejection_reasons"
                                                ][
                                                    refined_result.rejection_reason
                                                    or "unspecified"
                                                ] += 1
                                last_terrain_match = terrain_result
                                if global_search_result is not None:
                                    last_terrain_match = confirmation_result
                                last_terrain_match_mode = confirmation_mode
                                observation_matcher_accepted |= bool(
                                    confirmation_accepted
                                )
                                seed_pose2 = np.array(
                                    raw_pose2,
                                    dtype=np.float64,
                                )
                                emergency_confirmation = (
                                    terrain_emergency_recovery
                                    .consider_recovery_match(
                                        accepted=confirmation_accepted,
                                        observable_rank=(
                                            confirmation_observable_rank
                                        ),
                                        translation_observable_rank=(
                                            confirmation_translation_observable_rank
                                        ),
                                        yaw_observable=(
                                            confirmation_yaw_observable
                                        ),
                                        seed_pose2=seed_pose2,
                                        matched_pose2=(
                                            confirmation_matched_pose2
                                        ),
                                        patch_start_timestamp=(
                                            confirmation_patch_start_timestamp
                                        ),
                                        patch_end_timestamp=(
                                            confirmation_patch_end_timestamp
                                        ),
                                        patch_start_observation_id=(
                                            confirmation_patch_start_observation_id
                                        ),
                                        patch_end_observation_id=(
                                            confirmation_patch_end_observation_id
                                        ),
                                        confirmation_position_tolerance_m=(
                                            ARGS
                                            .terrain_global_confirmation_position_tolerance_m
                                            if confirmation_mode == "global"
                                            else None
                                        ),
                                        confirmation_yaw_tolerance_deg=(
                                            ARGS
                                            .terrain_global_confirmation_yaw_tolerance_deg
                                            if confirmation_mode == "global"
                                            else None
                                        ),
                                        confirmation_source=confirmation_mode,
                                        externally_confirmed_independent_evidence=(
                                            None
                                            if (
                                                confirmation_mode != "global"
                                                or global_tracking_decision
                                                is None
                                                or global_tracking_decision
                                                .confirmed_cohort
                                                is None
                                            )
                                            else global_tracking_decision
                                            .confirmed_cohort
                                            .independent_evidence
                                        ),
                                    )
                                )
                                emergency_prior_attempted = False
                                emergency_graph_diagnostics = None
                                if emergency_confirmation.fuse_current_match:
                                    emergency_prior_attempted = True
                                    (
                                        terrain_prior_added,
                                        _terrain_prior_held,
                                    ) = try_add_terrain_prior_result(
                                        confirmation_result,
                                        mode=confirmation_mode,
                                        timestamp=sim_time,
                                        observation_id=observation_id,
                                        patch_body=confirmation_patch_body,
                                        patch_start_timestamp=(
                                            confirmation_patch_start_timestamp
                                        ),
                                        patch_end_timestamp=(
                                            confirmation_patch_end_timestamp
                                        ),
                                        patch_start_observation_id=(
                                            confirmation_patch_start_observation_id
                                        ),
                                        patch_end_observation_id=(
                                            confirmation_patch_end_observation_id
                                        ),
                                        matched_pose2_override=(
                                            confirmation_matched_pose2
                                        ),
                                        information_override=(
                                            confirmation_information_override
                                        ),
                                        information_override_includes_model_floors=(
                                            confirmation_information_has_floors
                                        ),
                                        yaw_observable_override=(
                                            confirmation_yaw_observable
                                        ),
                                    )
                                    emergency_graph_diagnostics = (
                                        slam.pose_graph.diagnostics()
                                    )
                                    if terrain_prior_added:
                                        terrain_emergency_recovery.record_prior_insertion()
                                        if global_cohort_tracker is not None:
                                            global_cohort_tracker.reset()
                                        if (
                                            terrain_backend_particle_tracker
                                            is not None
                                        ):
                                            # The externally confirmed global
                                            # basin supersedes all locally
                                            # seeded tracking cohorts.
                                            terrain_backend_particle_tracker.reset()
                                    else:
                                        terrain_emergency_recovery.record_factor_rejection()

                                terrain_recovery_invocation_records.append(
                                    {
                                        "mode": confirmation_mode,
                                        "timestamp_s": float(sim_time),
                                        "observation_id": int(observation_id),
                                        "patch_ping_count": int(
                                            confirmation_patch_ping_count
                                        ),
                                        "patch_start_timestamp_s": (
                                            confirmation_patch_start_timestamp
                                        ),
                                        "patch_end_timestamp_s": (
                                            confirmation_patch_end_timestamp
                                        ),
                                        "patch_start_observation_id": (
                                            confirmation_patch_start_observation_id
                                        ),
                                        "patch_end_observation_id": (
                                            confirmation_patch_end_observation_id
                                        ),
                                        "patch_point_count": int(
                                            confirmation_patch_body.shape[0]
                                        ),
                                        "match_wall_time_s": float(
                                            match_elapsed_s
                                        ),
                                        "seed_pose2": [
                                            float(value) for value in seed_pose2
                                        ],
                                        "policy_request_reason": (
                                            emergency_request.reason
                                        ),
                                        "policy_bypassed_throttle": bool(
                                            emergency_request
                                            .bypassed_throttle
                                        ),
                                        "preceding_recovery_invoked": bool(
                                            recovery_invoked
                                        ),
                                        "preceding_recovery_rejection_reason": (
                                            preceding_recovery_rejection_reason
                                        ),
                                        "preceding_factor_rejected": bool(
                                            preceding_factor_rejected
                                        ),
                                        "outage_observation_index": int(
                                            emergency_outage_snapshot[
                                                "observation_index"
                                            ]
                                        ),
                                        "outage_observations": int(
                                            emergency_outage_snapshot[
                                                "consecutive_without_prior"
                                            ]
                                        ),
                                        "estimated_raw_arc_since_prior_m": (
                                            emergency_outage_snapshot[
                                                "estimated_raw_arc_since_prior_m"
                                            ]
                                        ),
                                        "estimated_time_since_prior_s": (
                                            emergency_outage_snapshot[
                                                "estimated_time_since_prior_s"
                                            ]
                                        ),
                                        "pending_before_match": bool(
                                            emergency_outage_snapshot[
                                                "pending_before_match"
                                            ]
                                        ),
                                        "matcher_accepted": bool(
                                            confirmation_accepted
                                        ),
                                        "matcher_rejection_reason": (
                                            None
                                            if confirmation_accepted
                                            else confirmation_result.rejection_reason
                                            or "global_cross_modal_refinement_rejected"
                                        ),
                                        "correction_xyyaw": [
                                            _finite_float_or_none(value)
                                            for value in confirmation_result.correction
                                        ],
                                        "translation_correction_m": (
                                            _finite_float_or_none(
                                                confirmation_result
                                                .translation_correction_m
                                            )
                                        ),
                                        "yaw_correction_deg": (
                                            _finite_float_or_none(
                                                confirmation_result
                                                .yaw_correction_deg
                                            )
                                        ),
                                        "matched_pose2": [
                                            _finite_float_or_none(value)
                                            for value in confirmation_matched_pose2
                                        ],
                                        "correspondences": int(
                                            confirmation_result.correspondences
                                        ),
                                        "inliers": int(
                                            confirmation_result.inliers
                                        ),
                                        "observable_rank": int(
                                            confirmation_observable_rank
                                        ),
                                        "translation_observable_rank": int(
                                            confirmation_translation_observable_rank
                                        ),
                                        "yaw_observable": bool(
                                            confirmation_yaw_observable
                                        ),
                                        "rmse_before_m": _finite_float_or_none(
                                            confirmation_result.rmse_before
                                        ),
                                        "rmse_after_m": _finite_float_or_none(
                                            confirmation_result.rmse_after
                                        ),
                                        "reduced_chi_square": (
                                            _finite_float_or_none(
                                                confirmation_result
                                                .reduced_chi_square
                                            )
                                        ),
                                        "reached_search_boundary": bool(
                                            confirmation_result
                                            .reached_search_boundary
                                        ),
                                        "distinct_basin_count": int(
                                            confirmation_result
                                            .distinct_basin_count
                                        ),
                                        "best_basin_cost": (
                                            _finite_float_or_none(
                                                confirmation_result
                                                .best_basin_cost
                                            )
                                        ),
                                        "second_best_basin_cost": (
                                            _finite_float_or_none(
                                                confirmation_result
                                                .second_best_basin_cost
                                            )
                                        ),
                                        "basin_log_likelihood_ratio": (
                                            _finite_float_or_none(
                                                confirmation_result
                                                .basin_log_likelihood_ratio
                                            )
                                        ),
                                        "whole_map_search_invoked": bool(
                                            global_search_result is not None
                                        ),
                                        "whole_map_search_accepted": bool(
                                            global_search_result is not None
                                            and global_search_result.accepted
                                        ),
                                        "whole_map_search_score_gap": (
                                            None
                                            if global_search_result is None
                                            else _finite_float_or_none(
                                                global_search_result
                                                .basin_score_gap
                                            )
                                        ),
                                        "confirmation_status": (
                                            emergency_confirmation.status
                                        ),
                                        "confirmation_pending": bool(
                                            emergency_confirmation.pending
                                        ),
                                        "confirmation_reference_observation_index": (
                                            emergency_confirmation
                                            .reference_observation_index
                                        ),
                                        "confirmation_acquisition_windows_disjoint": bool(
                                            emergency_confirmation
                                            .acquisition_windows_disjoint
                                        ),
                                        "confirmation_position_disagreement_m": (
                                            _finite_float_or_none(
                                                emergency_confirmation
                                                .position_disagreement_m
                                            )
                                        ),
                                        "confirmation_yaw_disagreement_deg": (
                                            _finite_float_or_none(
                                                emergency_confirmation
                                                .yaw_disagreement_deg
                                            )
                                        ),
                                        "graph_prior_attempted": bool(
                                            emergency_prior_attempted
                                        ),
                                        "graph_optimize_interval": 1,
                                        "graph_prior_accepted": bool(
                                            emergency_prior_attempted
                                            and terrain_prior_added
                                        ),
                                        "graph_relocalization_seeded": bool(
                                            emergency_prior_attempted
                                            and terrain_prior_added
                                        ),
                                        "graph_relocalization_anchor_index": (
                                            None
                                            if emergency_graph_diagnostics is None
                                            else emergency_graph_diagnostics
                                            .last_terrain_relocalization_anchor_index
                                        ),
                                        "graph_relocalization_translation_m": (
                                            None
                                            if emergency_graph_diagnostics is None
                                            else _finite_float_or_none(
                                                emergency_graph_diagnostics
                                                .last_terrain_relocalization_translation_m
                                            )
                                        ),
                                        "graph_relocalization_yaw_deg": (
                                            None
                                            if emergency_graph_diagnostics is None
                                            else _finite_float_or_none(
                                                emergency_graph_diagnostics
                                                .last_terrain_relocalization_yaw_deg
                                            )
                                        ),
                                        "graph_rejection_reason": (
                                            None
                                            if emergency_graph_diagnostics is None
                                            else emergency_graph_diagnostics
                                            .last_terrain_prior_rejection_reason
                                        ),
                                        "graph_prefit_performed": bool(
                                            emergency_graph_diagnostics
                                            is not None
                                            and emergency_graph_diagnostics
                                            .last_terrain_prefit_performed
                                        ),
                                        "graph_prefit_bypassed_for_relocalization": bool(
                                            emergency_graph_diagnostics
                                            is not None
                                            and emergency_graph_diagnostics
                                            .last_terrain_prefit_bypassed_for_relocalization
                                        ),
                                        "graph_prefit_chi2": (
                                            None
                                            if emergency_graph_diagnostics is None
                                            or not emergency_graph_diagnostics
                                            .last_terrain_prefit_performed
                                            else _finite_float_or_none(
                                                emergency_graph_diagnostics
                                                .last_terrain_prefit_chi2
                                            )
                                        ),
                                        "graph_prefit_dof": (
                                            None
                                            if emergency_graph_diagnostics is None
                                            or not emergency_graph_diagnostics
                                            .last_terrain_prefit_performed
                                            else int(
                                                emergency_graph_diagnostics
                                                .last_terrain_prefit_dof
                                            )
                                        ),
                                    }
                                )

                            if (
                                terrain_backend_particle_tracker is not None
                                and terrain_backend_particle_last_patch_end_observation_id
                                < patch_end_observation_id
                            ):
                                # Even when the compact local optimizer has no
                                # certified proposal, the current raw MBES/
                                # side-scan likelihood updates every cohort.
                                # Its basin-local Laplace approximation is an
                                # independent, rank-aware graph measurement;
                                # the recursive PF posterior is never reused.
                                raw_only_decision = (
                                    update_terrain_backend_particle_filter(
                                        timestamp=sim_time,
                                        observation_id=observation_id,
                                        mode="raw_likelihood_only",
                                        patch_body=patch_body,
                                        patch_reference_state=(
                                            local_reference_state
                                        ),
                                        backscatter_patch_body=(
                                            backscatter_patch_body
                                        ),
                                        backscatter_patch_values=(
                                            backscatter_patch_values
                                        ),
                                        patch_start_timestamp=(
                                            patch_start_timestamp
                                        ),
                                        patch_end_timestamp=(
                                            patch_end_timestamp
                                        ),
                                        patch_start_observation_id=(
                                            patch_start_observation_id
                                        ),
                                        patch_end_observation_id=(
                                            patch_end_observation_id
                                        ),
                                        evidences=(),
                                        evaluation_truth_state=truth,
                                    )
                                )
                                if raw_only_decision is not None:
                                    terrain_prior_added = (
                                        try_add_current_raw_particle_prior(
                                            raw_only_decision,
                                            timestamp=sim_time,
                                            observation_id=observation_id,
                                            patch_body=patch_body,
                                        )
                                    )

                            terrain_match_accepts += int(
                                observation_matcher_accepted
                            )
                            terrain_prior_accepts += int(terrain_prior_added)
                            terrain_recovery.complete_observation(
                                prior_added=terrain_prior_added,
                                recovery_invoked=(
                                    recovery_invoked
                                    or emergency_recovery_invoked
                                ),
                            )
                    ping_count += 1
                    next_ping = ping_count * ping_period
                    global_rotation, global_position = (
                        published_online_global_pose()
                    )
                    ping_samples = (
                        truth.position.copy(),
                        dead_reckoning.state.position.copy(),
                        slam.state.position.copy(),
                        global_position.copy(),
                    )
                    ping_rotation_samples = (
                        truth.rotation.copy(),
                        dead_reckoning.state.rotation.copy(),
                        slam.state.rotation.copy(),
                        global_rotation.copy(),
                    )
                    if np.isclose(
                        evaluation_timestamps[-1], sim_time, rtol=0.0, atol=1.0e-12
                    ):
                        truth_track[-1] = ping_samples[0]
                        truth_rotation_track[-1] = ping_rotation_samples[0]
                        dead_reckoning_track[-1] = ping_samples[1]
                        dead_reckoning_rotation_track[-1] = ping_rotation_samples[1]
                        local_track[-1] = ping_samples[2]
                        local_rotation_track[-1] = ping_rotation_samples[2]
                        online_global_track[-1] = ping_samples[3].copy()
                        online_global_rotation_track[-1] = (
                            ping_rotation_samples[3].copy()
                        )
                        if not ARGS.headless:
                            display_truth_track[-1] = ping_samples[0].copy()
                            display_local_track[-1] = ping_samples[2].copy()
                            display_sample_version += 1
                    else:
                        travelled_distance_online_m += float(
                            np.linalg.norm(ping_samples[0] - truth_track[-1])
                        )
                        evaluation_timestamps.append(sim_time)
                        truth_track.append(ping_samples[0])
                        truth_rotation_track.append(ping_rotation_samples[0])
                        dead_reckoning_track.append(ping_samples[1])
                        dead_reckoning_rotation_track.append(
                            ping_rotation_samples[1]
                        )
                        local_track.append(ping_samples[2])
                        local_rotation_track.append(ping_rotation_samples[2])
                        online_global_track.append(ping_samples[3].copy())
                        online_global_rotation_track.append(
                            ping_rotation_samples[3].copy()
                        )
                        if (
                            not ARGS.headless
                            and travelled_distance_online_m
                            >= display_last_distance_m
                            + display_sample_spacing_m
                        ):
                            display_timestamps.append(sim_time)
                            display_truth_track.append(ping_samples[0].copy())
                            display_local_track.append(ping_samples[2].copy())
                            display_last_distance_m = travelled_distance_online_m
                            display_sample_version += 1
                            if len(display_timestamps) > display_track_capacity:
                                display_timestamps[:] = display_timestamps[::2]
                                display_truth_track[:] = display_truth_track[::2]
                                display_local_track[:] = display_local_track[::2]
                                display_sample_spacing_m *= 2.0
                                display_sample_version += 1

                    if (
                        travelled_distance_online_m + 1.0e-9
                        >= next_progress_distance_m
                    ):
                        diagnostics = slam.diagnostics()
                        local_error = float(
                            np.linalg.norm(
                                slam.state.position - truth.position
                            )
                        )
                        global_error = float(
                            np.linalg.norm(global_position - truth.position)
                        )
                        elapsed_wall_s = time.monotonic() - started
                        checkpoint = {
                            "status": "running",
                            "survey_pattern": ARGS.survey_pattern,
                            "sim_time_s": sim_time,
                            "wall_time_s": elapsed_wall_s,
                            "travelled_distance_m": travelled_distance_online_m,
                            "route_completion_fraction": (
                                travelled_distance_online_m
                                / max(route_report.path_length_m, 1.0e-12)
                            ),
                            "pings": ping_count,
                            "local_error_m": local_error,
                            "online_global_error_m": global_error,
                            "keyframes": diagnostics.pose_graph.keyframes,
                            "loop_edges": diagnostics.pose_graph.loop_edges,
                            "pose_graph_optimizer_backend": (
                                diagnostics.pose_graph.optimizer_backend
                            ),
                            "pose_graph_isam_updates": (
                                diagnostics.pose_graph.isam_updates
                            ),
                            "pose_graph_fixed_lag_updates": (
                                diagnostics.pose_graph.fixed_lag_updates
                            ),
                            "pose_graph_fixed_lag_active_variables": (
                                diagnostics.pose_graph.fixed_lag_active_variables
                            ),
                            "pose_graph_fixed_lag_marginalized_variables": (
                                diagnostics.pose_graph
                                .fixed_lag_marginalized_variables
                            ),
                            "pose_graph_global_anchor_bridge_updates": (
                                diagnostics.pose_graph
                                .global_anchor_bridge_updates
                            ),
                            "terrain_matches_accepted": terrain_match_accepts,
                            "terrain_priors_accepted": terrain_prior_accepts,
                            "backscatter_matches_accepted": int(
                                backscatter_match_stats["accepts"]
                            ),
                            "backscatter_match_invocations": int(
                                backscatter_match_stats["invocations"]
                            ),
                            "terrain_recovery_invocations": (
                                terrain_match_mode_stats["recovery"][
                                    "invocations"
                                ]
                            ),
                            "terrain_emergency_recovery_invocations": (
                                terrain_match_mode_stats["emergency"][
                                    "invocations"
                                ]
                            ),
                            "terrain_global_relocalization_invocations": int(
                                global_relocalization_stats["invocations"]
                            ),
                            "terrain_global_relocalization_accepts": int(
                                global_relocalization_stats[
                                    "accepted_proposals"
                                ]
                            ),
                            "terrain_global_active_cohorts": (
                                0
                                if global_cohort_tracker is None
                                else int(
                                    global_cohort_tracker.diagnostics()[
                                        "active_cohorts"
                                    ]
                                )
                            ),
                            "terrain_global_cohort_confirmations": (
                                0
                                if global_cohort_tracker is None
                                else int(global_cohort_tracker.confirmations)
                            ),
                            "terrain_consecutive_failed_observations": (
                                terrain_recovery.consecutive_failed_observations
                            ),
                            "geometry_map_voxels": diagnostics.map_voxels,
                            "intensity_map_cells": diagnostics.intensity_map_cells,
                            "achieved_time_scale": (
                                sim_time / max(elapsed_wall_s, 1.0e-12)
                            ),
                        }
                        progress_path = ARGS.metrics.with_name(
                            f"{ARGS.metrics.stem}_progress.json"
                        )
                        progress_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary_progress_path = progress_path.with_suffix(
                            f"{progress_path.suffix}.tmp"
                        )
                        with temporary_progress_path.open(
                            "w", encoding="utf-8"
                        ) as stream:
                            json.dump(
                                checkpoint,
                                stream,
                                indent=2,
                                allow_nan=False,
                            )
                            stream.write("\n")
                        temporary_progress_path.replace(progress_path)
                        print(
                            "[OceanSim progress] "
                            f"{travelled_distance_online_m / 1000.0:.1f} km / "
                            f"{route_report.path_length_m / 1000.0:.1f} km, "
                            f"local/global {local_error:.3f}/{global_error:.3f} m, "
                            f"KF {diagnostics.pose_graph.keyframes}, "
                            f"loops {diagnostics.pose_graph.loop_edges}, "
                            f"terrain {terrain_prior_accepts}/"
                            f"{terrain_match_attempts} "
                            "(recovery "
                            f"{terrain_match_mode_stats['recovery']['invocations']}, "
                            "emergency "
                            f"{terrain_match_mode_stats['emergency']['invocations']}), "
                            f"{checkpoint['achieved_time_scale']:.1f}x achieved",
                            flush=True,
                        )
                        while (
                            next_progress_distance_m
                            <= travelled_distance_online_m + 1.0e-9
                        ):
                            next_progress_distance_m += ARGS.progress_distance_m

                    if status_label is not None and sonar_visualization_due:
                        diagnostics = slam.diagnostics()
                        dead_reckoning_error = np.linalg.norm(
                            dead_reckoning.state.position - truth.position
                        )
                        local_error = np.linalg.norm(
                            slam.state.position - truth.position
                        )
                        global_error = np.linalg.norm(global_position - truth.position)
                        altitude = truth.position[2] - float(
                            world_model.seabed_height(
                                truth.position[0], truth.position[1]
                            )
                        )
                        status_label.text = (
                            "ACOUSTIC FAST-LIVO2 | IMU + DVL + DEPTH + "
                            "INTEGRATED BATHYMETRIC SIDE-SCAN\n"
                            f"SONAR {acoustic_backend_label} | "
                            f"cast {getattr(acoustic_backend, 'last_cast_ms', 0.0):.2f} ms\n"
                            f"SLAM ASSOC {slam_association_label} | "
                            f"{slam.filter.last_association_ms:.2f} ms\n"
                            "COMPLETE USGS MONTEREY 5 m + MEASURED BACKSCATTER\n"
                            f"sim {sim_time:6.1f}s (target {ARGS.time_scale:.1f}x, "
                            f"achieved {sim_time / max(time.monotonic() - started, 1.0e-12):.2f}x)  "
                            f"ping {ping_count:05d}  "
                            f"travel {sim_time * ARGS.survey_speed_mps / 1000.0:5.2f}/"
                            f"{route_report.path_length_m / 1000.0:5.2f} km\n"
                            f"DR {dead_reckoning_error:6.2f} m   "
                            f"local {local_error:6.2f} m   "
                            f"global {global_error:6.2f} m   "
                            f"depth {-truth.position[2]:5.1f} m   altitude {altitude:4.1f} m\n"
                            f"V2 {profile_mode.upper()} | "
                            f"INTEGRATED {ARGS.ping_hz:g} Hz | "
                            f"UI {ARGS.ui_sonar_refresh_hz or (60.0 / ARGS.time_scale):g} Hz max | "
                            f"{sonar_profile.rows_per_side_per_ping} rows/side | "
                            f"{sonar_profile.range_sample_spacing_m * 100:.0f} cm samples | "
                            f"{sonar_profile.maximum_range_m:.0f} m/side\n"
                            f"ONE RAW PING: DEPTH + "
                            f"{2 * sonar_profile.rows_per_side_per_ping}-ROW IMAGE "
                            f"{np.count_nonzero(sonar_ping.bathymetry.hit_mask):03d}/"
                            f"{ARGS.sonar_beams:03d}   "
                            f"raw {raw_acoustic_points.shape[0]:05d}/"
                            f"{2 * sonar_profile.rows_per_side_per_ping * ARGS.sidescan_imaging_rays_per_side:05d}   "
                            f"SLAM {slam_acoustic_points.shape[0]:04d}\n"
                            f"ESIKF <= {ARGS.slam_max_points:03d} pts   "
                            f"map {diagnostics.map_points:7d} pts   "
                            f"KF {diagnostics.pose_graph.keyframes:03d}   "
                            f"loops {diagnostics.pose_graph.loop_edges:03d}\n"
                            f"geometry {diagnostics.accepted_sonar_updates:04d}/"
                            f"{diagnostics.rejected_sonar_updates:04d}   "
                            f"intensity {diagnostics.accepted_intensity_updates:04d}/"
                            f"{diagnostics.rejected_intensity_updates:04d}   "
                            f"terrain {terrain_prior_accepts:04d}/"
                            f"{terrain_match_attempts:04d}   "
                            f"texture {backscatter_match_stats['accepts']:04d}/"
                            f"{backscatter_match_stats['invocations']:04d}   "
                            f"recovery {terrain_match_mode_stats['recovery']['invocations']:04d}   "
                            f"streak {terrain_recovery.consecutive_failed_observations:03d}"
                        )

            performance_wall_s["sensor_and_slam_event_loop"] += (
                time.perf_counter() - event_loop_started
            )
            sim_time = target_time
            truth = survey_state(sim_time)
            display_stage_started = time.perf_counter()
            _set_vehicle_pose(
                vehicle_translate, vehicle_rotate, truth.position, truth.rotation
            )

            if not ARGS.headless and sim_time >= next_curve_update:
                graph_diagnostics = slam.diagnostics().pose_graph
                curve_graph_signature = (
                    graph_diagnostics.keyframes,
                    graph_diagnostics.optimization_runs,
                )
                if display_sample_version != last_rendered_display_version:
                    _update_curve(curves["truth"], display_truth_track, lift=0.8)
                    _update_curve(curves["local"], display_local_track, lift=1.0)
                if (
                    display_sample_version != last_rendered_display_version
                    or curve_graph_signature != cached_curve_graph_signature
                ):
                    corrected_display = slam.pose_graph.correct_trajectory(
                        np.asarray(display_timestamps, dtype=np.float64),
                        np.asarray(display_local_track, dtype=np.float64),
                    )
                    cached_global_display_track = [
                        point.copy() for point in corrected_display
                    ]
                    _update_curve(
                        curves["global"], cached_global_display_track, lift=1.2
                    )
                    cached_curve_graph_signature = curve_graph_signature
                last_rendered_display_version = display_sample_version
                next_curve_update = sim_time + 3.0
            if not ARGS.headless and sim_time >= next_map_update:
                graph_diagnostics = slam.diagnostics().pose_graph
                map_graph_signature = (
                    graph_diagnostics.keyframes,
                    graph_diagnostics.optimization_runs,
                )
                if map_graph_signature != cached_map_graph_signature:
                    points = slam.optimized_map_points()
                    if points.shape[0] > 6_000:
                        indices = np.linspace(
                            0, points.shape[0] - 1, 6_000, dtype=np.int64
                        )
                        points = points[indices]
                    if points.size:
                        map_prim.GetPointsAttr().Set(
                            [Gf.Vec3f(*point) for point in points]
                        )
                    cached_map_graph_signature = map_graph_signature
                next_map_update = sim_time + 5.0
            if not ARGS.headless and sim_time >= next_camera_update:
                _follow_live_camera(truth)
                next_camera_update = sim_time + 0.25
            performance_wall_s["display_stage_updates"] += (
                time.perf_counter() - display_stage_started
            )

            kit_step_started = time.perf_counter()
            world.step(render=not ARGS.headless)
            performance_wall_s["kit_render_step"] += (
                time.perf_counter() - kit_step_started
            )
            frame_count += 1
        if run_status == "running":
            run_status = "stopped"
            termination_reason = "simulation_app_closed"
    except KeyboardInterrupt:
        run_status = "interrupted"
        termination_reason = "keyboard_interrupt"
        return_code = 130
        carb.log_info("[OceanSim] interrupted by user")
        print("[OceanSim] interrupted by KeyboardInterrupt", flush=True)
        traceback.print_exc()
    except BaseException as exc:
        run_status = "failed"
        termination_reason = f"{type(exc).__name__}: {exc}"
        print(
            f"[OceanSim] terminating after {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        if python_profiler is not None:
            python_profiler.disable()
            python_profiler.dump_stats(python_profile_path)
        # Online updates may be deliberately throttled; the report always
        # consumes every accepted geometry, intensity, and loop factor.
        slam.pose_graph.optimize()
        diagnostics = slam.diagnostics()
        _, global_position = published_online_global_pose()
        truth = survey_state(sim_time)
        sample_times = np.asarray(evaluation_timestamps, dtype=np.float64)
        truth_samples = np.asarray(truth_track, dtype=np.float64)
        truth_rotation_samples = np.asarray(
            truth_rotation_track, dtype=np.float64
        )
        dead_samples = np.asarray(dead_reckoning_track, dtype=np.float64)
        dead_rotation_samples = np.asarray(
            dead_reckoning_rotation_track, dtype=np.float64
        )
        local_samples = np.asarray(local_track, dtype=np.float64)
        local_rotation_samples = np.asarray(
            local_rotation_track, dtype=np.float64
        )
        online_global_samples = np.asarray(
            online_global_track, dtype=np.float64
        )
        online_global_rotation_samples = np.asarray(
            online_global_rotation_track, dtype=np.float64
        )
        global_times, optimized_global_trajectory = slam.global_trajectory()
        optimized_global_samples = np.column_stack(
            [
                np.interp(
                    sample_times,
                    global_times,
                    optimized_global_trajectory[:, axis],
                )
                for axis in range(3)
            ]
        )
        optimized_global_rotation_samples = _pose_graph_corrected_rotations(
            slam.pose_graph,
            sample_times,
            local_rotation_samples,
        )
        cumulative_distance_m = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    np.linalg.norm(np.diff(truth_samples, axis=0), axis=1)
                ),
            )
        )
        estimate_tracks = {
            "dead_reckoning_imu_dvl_depth": dead_samples,
            "local_acoustic_esikf": local_samples,
            "online_global_slam": online_global_samples,
            "final_optimized_global_slam": optimized_global_samples,
        }
        estimate_rotation_tracks = {
            "dead_reckoning_imu_dvl_depth": dead_rotation_samples,
            "local_acoustic_esikf": local_rotation_samples,
            "online_global_slam": online_global_rotation_samples,
            "final_optimized_global_slam": (
                optimized_global_rotation_samples
            ),
        }
        full_history_poses = np.asarray(
            getattr(
                slam.pose_graph.graph,
                "global_isam_poses",
                np.empty((0, 3), dtype=np.float64),
            ),
            dtype=np.float64,
        ).reshape(-1, 3)
        if (
            full_history_poses.shape[0] > 0
            and full_history_poses.shape[0] == len(slam.pose_graph.keyframes)
        ):
            full_history_name = "gtsam_full_history_isam2_diagnostic"
            estimate_tracks[full_history_name] = (
                slam.pose_graph.correct_trajectory(
                    sample_times,
                    local_samples,
                    optimized_poses=full_history_poses,
                )
            )
            estimate_rotation_tracks[full_history_name] = (
                _pose_graph_corrected_rotations(
                    slam.pose_graph,
                    sample_times,
                    local_rotation_samples,
                    optimized_poses=full_history_poses,
                )
            )
        position_errors = {
            "definition": (
                "unaligned Euclidean position error at synchronized "
                f"{ARGS.ping_hz:g} Hz acoustic ping times, including the "
                "t=0 ping/initialization sample exactly once"
            ),
            "dead_reckoning_imu_dvl_depth": _position_error_metrics(
                dead_samples, truth_samples
            ),
            "local_acoustic_esikf": _position_error_metrics(
                local_samples, truth_samples
            ),
            "online_global_slam": _position_error_metrics(
                online_global_samples, truth_samples
            ),
            "final_optimized_global_slam": _position_error_metrics(
                optimized_global_samples, truth_samples
            ),
        }
        if "gtsam_full_history_isam2_diagnostic" in estimate_tracks:
            position_errors["gtsam_full_history_isam2_diagnostic"] = (
                _position_error_metrics(
                    estimate_tracks["gtsam_full_history_isam2_diagnostic"],
                    truth_samples,
                )
            )
        position_errors["translation_rpe_by_truth_distance"] = {
            name: {
                f"{int(interval_m)}m": _relative_position_error_metrics(
                    estimates,
                    truth_samples,
                    cumulative_distance_m,
                    interval_m,
                )
                for interval_m in (100.0, 1000.0, 10000.0)
            }
            for name, estimates in estimate_tracks.items()
        }
        position_errors["translation_rpe_definition"] = (
            "legacy difference between estimate and truth displacement vectors "
            "expressed in the shared world frame; retained for compatibility "
            "and not invariant to a constant position/yaw gauge"
        )
        position_errors["se3_rpe_by_truth_distance"] = {
            name: {
                f"{int(interval_m)}m": se3_relative_pose_error_by_distance(
                    estimates,
                    estimate_rotation_tracks[name],
                    truth_samples,
                    truth_rotation_samples,
                    cumulative_distance_m,
                    interval_m,
                )
                for interval_m in (100.0, 1000.0, 10000.0)
            }
            for name, estimates in estimate_tracks.items()
        }
        position_errors["five_kilometre_segments"] = {
            name: _five_kilometre_error_segments(
                estimates, truth_samples, cumulative_distance_m
            )
            for name, estimates in estimate_tracks.items()
        }
        orientation_errors = {
            "definition": (
                "truth-to-estimate relative SO(3) rotation at synchronized "
                f"{ARGS.ping_hz:g} Hz acoustic ping times, decomposed into "
                "intrinsic roll/pitch/yaw error; yaw is wrapped and no "
                "component-wise quaternion subtraction is used"
            ),
            **{
                name: _orientation_error_metrics(
                    estimates, truth_rotation_samples
                )
                for name, estimates in estimate_rotation_tracks.items()
            },
        }
        map_to_odom_audit = map_to_odom_change_metrics(
            sample_times,
            local_samples,
            local_rotation_samples,
            online_global_samples,
            online_global_rotation_samples,
        )
        dead_rmse = position_errors[
            "dead_reckoning_imu_dvl_depth"
        ]["rmse_3d_m"]
        local_rmse = position_errors["local_acoustic_esikf"]["rmse_3d_m"]
        global_rmse = position_errors[
            "final_optimized_global_slam"
        ]["rmse_3d_m"]
        position_errors["rmse_reduction_vs_dead_reckoning_percent"] = {
            "local_acoustic_esikf": float(
                100.0 * (dead_rmse - local_rmse) / max(dead_rmse, 1.0e-12)
            ),
            "final_optimized_global_slam": float(
                100.0 * (dead_rmse - global_rmse) / max(dead_rmse, 1.0e-12)
            ),
        }
        error_columns = [
            sample_times,
            np.linalg.norm(dead_samples - truth_samples, axis=1),
            np.linalg.norm(local_samples - truth_samples, axis=1),
            np.linalg.norm(online_global_samples - truth_samples, axis=1),
            np.linalg.norm(optimized_global_samples - truth_samples, axis=1),
        ]
        error_column_names = [
            "timestamp_s",
            "dead_reckoning_3d_m",
            "local_acoustic_esikf_3d_m",
            "online_global_slam_3d_m",
            "final_optimized_global_slam_3d_m",
        ]
        if "gtsam_full_history_isam2_diagnostic" in estimate_tracks:
            error_columns.append(
                np.linalg.norm(
                    estimate_tracks["gtsam_full_history_isam2_diagnostic"]
                    - truth_samples,
                    axis=1,
                )
            )
            error_column_names.append(
                "gtsam_full_history_isam2_diagnostic_3d_m"
            )
        error_traces = np.column_stack(error_columns)
        error_csv = ARGS.metrics.with_name(
            f"{ARGS.metrics.stem}_position_errors.csv"
        )
        ARGS.metrics.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            error_csv,
            error_traces,
            delimiter=",",
            header=",".join(error_column_names),
            comments="",
        )
        factor_log = ARGS.metrics.with_name(
            f"{ARGS.metrics.stem}_pose_graph_factors.json"
        )
        factor_payload = slam.pose_graph.factor_stream_payload()
        temporary_factor_log = factor_log.with_suffix(
            f"{factor_log.suffix}.tmp"
        )
        with temporary_factor_log.open("w", encoding="utf-8") as stream:
            json.dump(factor_payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary_factor_log.replace(factor_log)
        travelled_distance_m = float(cumulative_distance_m[-1])
        survey_laps_completed = float(sim_time / survey.period)
        completion_tolerance_laps = 1.0e-9
        completed_requested_route = bool(
            ARGS.survey_laps > 0.0
            and survey_laps_completed + completion_tolerance_laps
            >= float(ARGS.survey_laps)
        )
        full_map_completed = bool(
            ARGS.survey_pattern == "full-map-route"
            and survey_laps_completed + completion_tolerance_laps >= 1.0
        )
        position_errors["final_drift_percent_of_distance"] = {
            name: float(
                100.0
                * position_errors[name]["final_3d_m"]
                / max(travelled_distance_m, 1.0e-12)
            )
            for name in estimate_tracks
        }
        normal_terrain_stats = terrain_match_mode_stats["normal"]
        recovery_terrain_stats = terrain_match_mode_stats["recovery"]
        emergency_terrain_stats = terrain_match_mode_stats["emergency"]
        global_terrain_stats = terrain_match_mode_stats["global"]
        terrain_match_invocations = int(
            normal_terrain_stats["invocations"]
            + recovery_terrain_stats["invocations"]
            + emergency_terrain_stats["invocations"]
            + global_terrain_stats["invocations"]
        )
        terrain_match_invocation_accepts = int(
            normal_terrain_stats["accepts"]
            + recovery_terrain_stats["accepts"]
            + emergency_terrain_stats["accepts"]
            + global_terrain_stats["accepts"]
        )
        terrain_match_elapsed_s = float(
            normal_terrain_stats["wall_time_s"]
            + recovery_terrain_stats["wall_time_s"]
            + emergency_terrain_stats["wall_time_s"]
            + global_terrain_stats["wall_time_s"]
        )
        terrain_match_rejection_reasons = (
            normal_terrain_stats["rejection_reasons"]
            + recovery_terrain_stats["rejection_reasons"]
            + emergency_terrain_stats["rejection_reasons"]
            + global_terrain_stats["rejection_reasons"]
        )
        terrain_match_rank_counts = (
            normal_terrain_stats["rank_counts"]
            + recovery_terrain_stats["rank_counts"]
            + emergency_terrain_stats["rank_counts"]
            + global_terrain_stats["rank_counts"]
        )
        accepted_invocation_count = int(
            terrain_accepted_invocation_stats["count"]
        )
        terrain_truth_evaluation_count = int(
            terrain_match_truth_evaluation_sums["count"]
        )

        def terrain_truth_evaluation_rmse(sum_key):
            if terrain_truth_evaluation_count == 0:
                return None
            return float(
                math.sqrt(
                    terrain_match_truth_evaluation_sums[sum_key]
                    / terrain_truth_evaluation_count
                )
            )

        accepted_backscatter_records = [
            record
            for record in backscatter_match_records
            if record["accepted"]
        ]

        def backscatter_truth_evaluation_rmse(records, key):
            if not records:
                return None
            return float(
                math.sqrt(
                    sum(float(record[key]) ** 2 for record in records)
                    / len(records)
                )
            )

        metrics = {
            "run_status": run_status,
            "termination_reason": termination_reason,
            "completed_requested_route": completed_requested_route,
            "full_map_completed": full_map_completed,
            "algorithm": (
                "OceanSim acoustic FAST-LIVO2 adaptation with joint "
                "point-mass/cohort particle terrain navigation and "
                "rank-aware absolute bathymetry/backscatter factors"
                if terrain_matcher is not None
                else "OceanSim acoustic FAST-LIVO2 adaptation"
            ),
            "algorithm_tracks": {
                "dead_reckoning_imu_dvl_depth": (
                    "IMU propagation with DVL velocity and pressure depth"
                ),
                "local_acoustic_esikf": (
                    "self-built surfel/intensity-map relative acoustic ESIKF; "
                    "no prior-map correction"
                ),
                "online_global_slam": (
                    (
                        "GTSAM IncrementalFixedLagSmoother active-window "
                        "estimate; marginalized poses retain their last "
                        "online value, while confirmed cross-window factors "
                        "supply a rigid global map anchor"
                        + (
                            "; while multiple spatial modes exist, the "
                            "terrain-particle MAP drives the provisional "
                            "online target, while single-mode output uses "
                            "the graph and independent confirmation still "
                            "protects factor insertion; active whole-map "
                            "relocalization cohorts take output priority"
                            if terrain_backend_particle_tracker is not None
                            else ""
                        )
                        + (
                            "; the published map->odom transform follows "
                            "that target continuously with bounded rates"
                            if ARGS.enable_map_to_odom_soft_correction
                            else ""
                        )
                    )
                    if ARGS.pose_graph_backend == "gtsam-fixed-lag-isam2"
                    else (
                        "online NumPy batch pose graph including accepted "
                        "loop, intensity, and absolute terrain factors"
                        + (
                            "; the published map->odom transform follows "
                            "that target continuously with bounded rates"
                            if ARGS.enable_map_to_odom_soft_correction
                            else ""
                        )
                    )
                ),
                "final_optimized_global_slam": (
                    (
                        "the fixed-lag online trajectory after consuming all "
                        "factors that remain representable in its live window"
                    )
                    if ARGS.pose_graph_backend == "gtsam-fixed-lag-isam2"
                    else (
                        "the same global factor graph after consuming every "
                        "accepted factor in a final optimization"
                    )
                ),
                **(
                    {
                        "gtsam_full_history_isam2_diagnostic": (
                            "separate full-history iSAM2 diagnostic; retained "
                            "for old-node constraints and not used as the "
                            "online map-to-odom output"
                        )
                    }
                    if ARGS.pose_graph_backend == "gtsam-fixed-lag-isam2"
                    else {}
                ),
            },
            "navigation_initialization": {
                "roll_pitch_velocity_and_depth": (
                    "simulator t=0 truth (evaluation only)"
                    if ARGS.initialize_from_truth
                    else "estimated from stationary IMU, DVL, and pressure samples"
                ),
                "horizontal_origin_and_yaw_gauge": (
                    "simulator t=0 truth (evaluation only)"
                    if ARGS.initialize_from_truth
                    else (
                        "known launch pose plus the explicitly reported "
                        "deterministic capture-offset test"
                    )
                ),
                "uses_simulator_truth": bool(ARGS.initialize_from_truth),
                "horizontal_seed_offset_m": [
                    ARGS.initial_position_offset_x_m,
                    ARGS.initial_position_offset_y_m,
                ],
                "yaw_seed_offset_deg": ARGS.initial_yaw_offset_deg,
                "interpretation": (
                    "controlled front-end odometry evaluation with an exact "
                    "initial state; not an operational initialization claim"
                    if ARGS.initialize_from_truth
                    else (
                        "bounded coarse surface-launch initialization; the "
                        "matcher is not an unbounded kidnapped-vehicle global "
                        "search"
                    )
                ),
            },
            "sensors": [
                "IMU",
                "DVL",
                "pressure_depth",
                "integrated_bathymetric_side_scan",
            ],
            "sonar_acquisition_model": (
                "single_raw_return_grid_derived_bathymetry_and_backscatter"
            ),
            "simulation_clock_model": (
                "exact application-clock sensor events; kinematic AUV pose; "
                "zero-order-held IMU propagation to asynchronous sonar "
                "timestamps; Isaac physics timeline is not the navigation time base"
            ),
            "pose_graph_odometry_noise_model": {
                "model": "continuous-time random walk",
                "reference_interval_s": float(
                    ARGS.pose_graph_odometry_reference_interval_s
                ),
                "horizontal_std_m_per_reference_interval": float(
                    slam.pose_graph.config.odometry_xy_std_m
                ),
                "yaw_std_deg_per_reference_interval": float(
                    slam.pose_graph.config.odometry_yaw_std_deg
                ),
                "keyframe_max_interval_s": float(
                    ARGS.keyframe_max_interval_s
                ),
                "edge_scaling": "standard deviation times sqrt(elapsed/reference)",
            },
            "pose_graph_backend_configuration": {
                "selected": ARGS.pose_graph_backend,
                "implementation": diagnostics.pose_graph.optimizer_backend,
                "fixed_lag_seconds": float(ARGS.pose_graph_fixed_lag_s),
                "isam_relinearize_threshold": float(
                    ARGS.pose_graph_isam_relinearize_threshold
                ),
                "isam_relinearize_skip": int(
                    ARGS.pose_graph_isam_relinearize_skip
                ),
                "online_output_source": (
                    "incremental_fixed_lag_active_window_with_rigid_"
                    "global_anchor_for_cross_window_factors"
                    if ARGS.pose_graph_backend == "gtsam-fixed-lag-isam2"
                    else "numpy_batch_full_graph"
                ),
            },
            "reproducibility": {
                "random_seed": int(ARGS.seed),
                "python_random_seed": int(ARGS.seed),
                "numpy_legacy_random_seed": int(ARGS.seed),
                "navigation_sensor_seed": int(ARGS.seed),
                "sonar_sensor_seed": int(ARGS.seed + 1),
                "deterministic_event_clock": True,
                "deterministic_sampling_and_decimation": True,
                "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                "strict_bitwise_determinism": False,
                "determinism_scope": (
                    "sensor noise, event scheduling, sampling, and decimation "
                    "are seeded/deterministic; bitwise identity is not claimed "
                    "across Isaac, Warp, driver, device, or thread configurations"
                ),
            },
            "scene_m": [world_model.width, world_model.length],
            "terrain_source": world_model.metadata.get("source_product", "unknown"),
            "terrain_resolution_m": world_model.resolution_m,
            "terrain_matching_reference": (
                {
                    "enabled": terrain_matcher is not None,
                    "map_role": "georeferenced prior bathymetry",
                    "surface_model": terrain_prior_model_kind,
                    "source_product": world_model.metadata.get(
                        "source_product", "unknown"
                    ),
                    "resolution_m": world_model.resolution_m,
                    "vertical_model_std_m": (
                        terrain_prior_vertical_model_std_m
                    ),
                    "low_pass_sigma_cells": (
                        2.2
                        if ARGS.terrain_prior_surface == "low-pass"
                        else None
                    ),
                    "uses_simulator_truth_pose": False,
                    "simulation_and_matcher_share_source_survey": True,
                    "simulation_and_matcher_use_identical_surface": (
                        ARGS.terrain_prior_surface == "measured"
                    ),
                    "independent_survey_prior": False,
                }
            ),
            "backscatter_map_matching": {
                "enabled": backscatter_matcher is not None,
                "map_role": (
                    "georeferenced side-scan texture refinement of accepted "
                    "bathymetric fixes"
                ),
                "source_product": world_model.metadata.get(
                    "backscatter_source_product",
                    world_model.metadata.get("source_product", "unknown"),
                ),
                "resolution_m": world_model.resolution_m,
                "radiometry": (
                    "unitless normalized survey mosaic; port/starboard and "
                    "smooth cross-track trends removed before zero-mean "
                    "normalized correlation; not calibrated target strength"
                ),
                "fusion": (
                    "weighted covariance intersection in local SE(2) "
                    "coordinates; bathymetry/backscatter independence is "
                    "not assumed"
                ),
                "covariance_intersection_weight_mode": (
                    "automatic_minimum_logdet"
                    if ARGS.backscatter_ci_bathymetry_weight < 0.0
                    else "fixed"
                ),
                "covariance_intersection_bathymetry_weight": (
                    None
                    if ARGS.backscatter_ci_bathymetry_weight < 0.0
                    else float(ARGS.backscatter_ci_bathymetry_weight)
                ),
                "maximum_points": ARGS.backscatter_match_maximum_points,
                "maximum_yaw_deg": ARGS.backscatter_match_yaw_window_deg,
                "boundary_yaw_retry_window_deg": float(
                    ARGS.backscatter_boundary_yaw_retry_window_deg
                ),
                "large_correction_compatibility_threshold_m": float(
                    ARGS.cross_modal_large_correction_m
                ),
                "horizontal_std_floor_m": (
                    None
                    if backscatter_matcher is None
                    else backscatter_matcher.config.horizontal_std_m
                ),
                "yaw_std_floor_deg": (
                    None
                    if backscatter_matcher is None
                    else backscatter_matcher.config.yaw_std_deg
                ),
                "simulation_and_matcher_share_source_backscatter": bool(
                    backscatter_matcher is not None
                ),
                "independent_survey_prior": False,
                "inverse_crime_risk": (
                    "optimistic same-survey texture upper bound; noise, "
                    "attenuation and incidence effects are simulated, but "
                    "the reference mosaic is not from an independent survey"
                    if backscatter_matcher is not None
                    else None
                ),
                "uses_simulator_truth_pose": False,
                "invocations": int(backscatter_match_stats["invocations"]),
                "matcher_calls": int(
                    backscatter_match_stats["matcher_calls"]
                ),
                "accepts": int(backscatter_match_stats["accepts"]),
                "rejections": int(backscatter_match_stats["rejections"]),
                "rejection_reasons": dict(
                    backscatter_match_stats["rejection_reasons"]
                ),
                "primary_rejection_reasons": dict(
                    backscatter_match_stats["primary_rejection_reasons"]
                ),
                "boundary_yaw_retries": int(
                    backscatter_match_stats["boundary_yaw_retries"]
                ),
                "boundary_yaw_retry_accepts": int(
                    backscatter_match_stats[
                        "boundary_yaw_retry_accepts"
                    ]
                ),
                "fused_fixes": int(backscatter_match_stats["fused_fixes"]),
                "compatibility_vetoes": int(
                    backscatter_match_stats["compatibility_vetoes"]
                ),
                "compatibility_veto_reasons": dict(
                    backscatter_match_stats[
                        "compatibility_veto_reasons"
                    ]
                ),
                "wall_time_s": float(backscatter_match_stats["wall_time_s"]),
                "mean_time_ms": float(
                    1.0e3
                    * backscatter_match_stats["wall_time_s"]
                    / max(backscatter_match_stats["invocations"], 1)
                ),
                "last_patch_raw_points": int(
                    last_backscatter_patch_raw_points
                ),
                "last_patch_reduced_points": int(
                    last_backscatter_patch_reduced_points
                ),
                "last_patch_voxel_size_m": float(
                    last_backscatter_patch_voxel_size_m
                ),
                "truth_evaluation": {
                    "evaluation_only": True,
                    "used_by_matcher_or_estimator": False,
                    "accepted_match_count": len(
                        accepted_backscatter_records
                    ),
                    "bathymetric_seed_horizontal_rmse_m": (
                        backscatter_truth_evaluation_rmse(
                            accepted_backscatter_records,
                            "bathymetric_horizontal_error_m",
                        )
                    ),
                    "backscatter_horizontal_rmse_m": (
                        backscatter_truth_evaluation_rmse(
                            accepted_backscatter_records,
                            "backscatter_horizontal_error_m",
                        )
                    ),
                    "fused_horizontal_rmse_m": (
                        backscatter_truth_evaluation_rmse(
                            accepted_backscatter_records,
                            "fused_horizontal_error_m",
                        )
                    ),
                    "bathymetric_seed_yaw_rmse_deg": (
                        backscatter_truth_evaluation_rmse(
                            accepted_backscatter_records,
                            "bathymetric_yaw_error_deg",
                        )
                    ),
                    "backscatter_yaw_rmse_deg": (
                        backscatter_truth_evaluation_rmse(
                            accepted_backscatter_records,
                            "backscatter_yaw_error_deg",
                        )
                    ),
                    "fused_yaw_rmse_deg": (
                        backscatter_truth_evaluation_rmse(
                            accepted_backscatter_records,
                            "fused_yaw_error_deg",
                        )
                    ),
                    "records": backscatter_match_records,
                },
            },
            "terrain_source_shape_pixels": list(
                world_model.elevation_m.shape
            ),
            "terrain_geographically_cropped": bool(
                world_model.metadata.get("is_geographically_cropped", True)
            ),
            "terrain_valid_measured_cells": int(
                np.count_nonzero(world_model.valid_mask)
            ),
            "terrain_display_stride": int(terrain_stride),
            "sonar_terrain_stride": int(ARGS.sonar_terrain_stride),
            "measured_backscatter_coverage_fraction": float(
                world_model.backscatter_coverage_fraction
            ),
            "bounds_utm_m": world_model.metadata.get("bounds_utm_m", {}),
            "vehicle": "LSTS/OceanScan LAUV",
            "sonar_profile": sonar_profile.to_metadata(),
            "sonar_profile_interpretation": {
                "maximum_range": "treated as per-side slant range",
                "along_track_rows": (
                    "modelled as eight parallel virtual footprint centres per "
                    "side, spaced by the documented along-track sample interval"
                ),
                "along_track_row_center_offsets_m": (
                    (
                        np.arange(
                            sonar_profile.rows_per_side_per_ping,
                            dtype=np.float64,
                        )
                        - 0.5 * (sonar_profile.rows_per_side_per_ping - 1)
                    )
                    * sonar_profile.along_track_sample_spacing_m
                ).tolist(),
                "along_track_coverage_boundaries_m": [
                    -0.5 * sonar_profile.along_track_coverage_per_ping_m,
                    0.5 * sonar_profile.along_track_coverage_per_ping_m,
                ],
                "terrain_resolution_limit_m": (
                    ARGS.sonar_terrain_stride * world_model.resolution_m
                ),
                "warning": (
                    "2 cm output sampling and 2.5 cm documented sensor "
                    "resolution do not create independent centimetre-scale "
                    "seabed information from the 5 m source raster"
                ),
            },
            "sonar_simulation_assumptions": {
                "sound_speed_mps": sound_speed_mps,
                "ping_rate_hz": ARGS.ping_hz,
                "two_way_travel_time_at_maximum_range_s": (
                    two_way_travel_time_s
                ),
                "distinguishable_pings_in_water": (
                    ARGS.sonar_pings_in_water
                ),
                "required_distinguishable_pings_in_water": (
                    required_pings_in_water
                ),
                "maximum_scheduled_ping_rate_hz": (
                    maximum_scheduled_ping_hz
                ),
                "multiping_waveform_separation_modelled": False,
                "vertical_fan_width_deg": ARGS.sonar_swath_width_deg,
                "horizontal_beam_opening_deg": (
                    sonar_profile.horizontal_beam_opening_deg
                ),
                "along_track_footprint_model": (
                    "parallel virtual row centres; finite 0.14 deg aperture is "
                    "recorded but not coherently integrated"
                ),
                "center_frequency_hz": None,
                "range_noise_std_m": 0.028,
                "attenuation_coefficient_per_m": 0.003,
                "installation_translation_body_m": [0.10, 0.0, -0.15],
                "installation_orientation_body_rpy_deg": [0.0, 0.0, 0.0],
            },
            "survey_speed_mps": ARGS.survey_speed_mps,
            "survey_pattern": ARGS.survey_pattern,
            "survey_period_s": survey.period,
            "requested_survey_laps": ARGS.survey_laps,
            "survey_laps_completed": survey_laps_completed,
            "survey_travelled_distance_m": travelled_distance_m,
            "survey_route_completion_fraction": (
                travelled_distance_m
                / max(float(route_report.path_length_m), 1.0e-12)
            ),
            "survey_route_file": (
                str(ARGS.survey_route)
                if ARGS.survey_pattern == "full-map-route"
                else None
            ),
            "survey_route_definition": survey_route_metadata.get(
                "route_definition"
            ),
            "survey_center_m": (
                [ARGS.survey_center_x_m, ARGS.survey_center_y_m]
                if ARGS.survey_pattern == "figure-eight"
                else None
            ),
            "survey_amplitude_m": (
                [ARGS.survey_x_amplitude_m, ARGS.survey_y_amplitude_m]
                if ARGS.survey_pattern == "figure-eight"
                else None
            ),
            "survey_pattern_yaw_deg": (
                ARGS.survey_pattern_yaw_deg
                if ARGS.survey_pattern == "figure-eight"
                else None
            ),
            "survey_altitude_m": ARGS.survey_altitude_m,
            "survey_maximum_pitch_deg": ARGS.survey_maximum_pitch_deg,
            "survey_route_validation": asdict(route_report),
            "ping_rate_hz": ARGS.ping_hz,
            "sonar_pings_in_water": ARGS.sonar_pings_in_water,
            "sonar_beams": ARGS.sonar_beams,
            "published_bathymetry_samples_per_ping": ARGS.sonar_beams,
            "bathymetry_is_raw_return_subset": True,
            "slam_uses_unified_xyz_intensity_returns": True,
            "documented_sidescan_rows_per_side_per_ping": (
                sonar_profile.rows_per_side_per_ping
            ),
            "sidescan_imaging_rays_per_side": (
                sonar_profile.rows_per_side_per_ping
                * ARGS.sidescan_imaging_rays_per_side
            ),
            "sidescan_imaging_rays_per_side_per_row": (
                ARGS.sidescan_imaging_rays_per_side
            ),
            "sidescan_imaging_rays_per_ping": (
                2
                * sonar_profile.rows_per_side_per_ping
                * ARGS.sidescan_imaging_rays_per_side
            ),
            "unified_raw_returns_per_ping": (
                2
                * sonar_profile.rows_per_side_per_ping
                * ARGS.sidescan_imaging_rays_per_side
            ),
            "sidescan_slam_sample_stride": ARGS.sidescan_slam_stride,
            "shared_rays_per_ping": (
                int(sonar_ping.shared_ray_count)
                if ping_count > 0
                else int(
                    sonar._shared_ray_directions.shape[0]
                )
            ),
            "sonar_swath_width_deg": ARGS.sonar_swath_width_deg,
            "sidescan_range_bins_per_side": (
                sonar_profile.range_bins_per_side
            ),
            "sidescan_range_sample_spacing_m": (
                sonar_profile.range_sample_spacing_m
            ),
            "sidescan_documented_range_resolution_m": (
                sonar_profile.range_resolution_m
            ),
            "sidescan_maximum_range_m": sonar_profile.maximum_range_m,
            "intensity_reference_delay_pings": (
                ARGS.intensity_reference_delay_pings
            ),
            "intensity_group_pings": ARGS.intensity_group_pings,
            "intensity_update_rate_hz": (
                ARGS.ping_hz / ARGS.intensity_group_pings
            ),
            "intensity_min_group_span_m": (
                ARGS.intensity_min_group_span_m
            ),
            "intensity_submap_anchor_interval_s": (
                ARGS.intensity_submap_anchor_interval_s
            ),
            "intensity_effective_independent_samples": (
                ARGS.intensity_effective_independent_samples
            ),
            "local_sonar_noise_floor_m": (
                ARGS.local_sonar_noise_floor_m
            ),
            "local_intensity_noise_std": (
                ARGS.local_intensity_noise_std
            ),
            "local_intensity_surface_consistency_std_m": (
                ARGS.local_intensity_surface_consistency_std_m
            ),
            "local_observability_mode": ARGS.local_observability_mode,
            "local_intensity_map_coregisters_surface_elevation": True,
            "intensity_min_singular_ratio": (
                ARGS.intensity_min_singular_ratio
            ),
            "intensity_factor_min_interval_s": (
                ARGS.intensity_factor_min_interval_s
            ),
            "intensity_factor_optimize_interval": (
                ARGS.intensity_factor_optimize_interval
            ),
            "intensity_updates_enabled": not ARGS.disable_intensity_updates,
            "terrain_map_matching_enabled": terrain_matcher is not None,
            "terrain_prior_surface": ARGS.terrain_prior_surface,
            "terrain_match_window_pings": ARGS.terrain_match_window_pings,
            "terrain_patch_motion_compensation": (
                {
                    "acoustic-fixed-lag": (
                        "causal_fixed_lag_imu_dvl_motion_with_acoustic_endpoints"
                    ),
                    "acoustic-predictive": (
                        "causal_acoustic_inertial_prediction_before_current_ping"
                    ),
                    "inertial": "independent_imu_dvl_depth_dead_reckoning",
                }[ARGS.terrain_patch_motion_source]
            ),
            "terrain_match_maximum_points": (
                ARGS.terrain_match_maximum_points
            ),
            "terrain_patch_minimum_voxel_size_m": (
                ARGS.terrain_patch_minimum_voxel_size_m
            ),
            "terrain_position_only_priors": (
                ARGS.terrain_position_only_priors
            ),
            "terrain_prior_horizontal_std_floor_m": (
                ARGS.terrain_prior_horizontal_std_floor_m
            ),
            "terrain_prior_yaw_std_floor_deg": (
                ARGS.terrain_prior_yaw_std_floor_deg
            ),
            "terrain_gauge_confirmation_patches": (
                ARGS.terrain_gauge_confirmation_patches
            ),
            "terrain_confirm_every_prior": (
                ARGS.terrain_confirm_every_prior
            ),
            "terrain_gauge_minimum_baseline_m": (
                ARGS.terrain_gauge_minimum_baseline_m
            ),
            "terrain_gauge_maximum_position_residual_m": (
                ARGS.terrain_gauge_maximum_position_residual_m
            ),
            "terrain_gauge_maximum_yaw_residual_deg": (
                ARGS.terrain_gauge_maximum_yaw_residual_deg
            ),
            "terrain_gauge_initialization": {
                **terrain_gauge_initialization_stats,
                "pending_at_end": pending_initial_terrain_gauge is not None,
                "uses_simulator_truth": False,
            },
            "terrain_fix_consistency_records": terrain_consistency_records,
            "last_terrain_retained_full_swath_pings": int(
                last_terrain_retained_full_swath_pings
            ),
            "terrain_patch_reduction": (
                "all_ping_horizontal_voxel_centroid_averaging"
            ),
            "last_terrain_patch_raw_points": int(
                last_terrain_patch_raw_points
            ),
            "last_terrain_patch_reduced_points": int(
                last_terrain_patch_reduced_points
            ),
            "last_terrain_patch_voxel_size_m": float(
                last_terrain_patch_voxel_size_m
            ),
            "terrain_match_translation_window_m": (
                ARGS.terrain_match_translation_window_m
            ),
            "terrain_recovery_translation_window_m": (
                ARGS.terrain_recovery_translation_window_m
            ),
            "terrain_recovery_after_rejections": (
                ARGS.terrain_recovery_after_rejections
            ),
            "terrain_recovery_policy": {
                "enabled": terrain_recovery.enabled,
                "strict_rank3": bool(
                    terrain_recovery_config is not None
                    and terrain_recovery_config.require_yaw_observable
                ),
                "same_patch_pings": ARGS.terrain_match_window_pings,
                "coarse_translation_step_m": (
                    None
                    if terrain_recovery_config is None
                    else terrain_recovery_config.coarse_translation_step_m
                ),
                "coarse_top_k": (
                    None
                    if terrain_recovery_config is None
                    else terrain_recovery_config.coarse_top_k
                ),
                "coarse_seed_translation_separation_m": (
                    None
                    if terrain_recovery_config is None
                    else terrain_recovery_config
                    .coarse_seed_translation_separation_m
                ),
                "coarse_seed_yaw_separation_deg": (
                    None
                    if terrain_recovery_config is None
                    else terrain_recovery_config
                    .coarse_seed_yaw_separation_deg
                ),
                "minimum_basin_translation_separation_m": (
                    None
                    if terrain_recovery_config is None
                    else terrain_recovery_config
                    .minimum_basin_translation_separation_m
                ),
                "minimum_basin_yaw_separation_deg": (
                    None
                    if terrain_recovery_config is None
                    else terrain_recovery_config
                    .minimum_basin_yaw_separation_deg
                ),
                "boundary_translation_margin_m": (
                    None
                    if terrain_recovery_config is None
                    else terrain_recovery_config.boundary_translation_margin_m
                ),
                "minimum_basin_log_likelihood_ratio": (
                    None
                    if terrain_recovery_config is None
                    else terrain_recovery_config
                    .minimum_basin_log_likelihood_ratio
                ),
                "uses_truth": False,
                "accepted_factor_optimize_interval": 1,
            },
            "terrain_emergency_recovery_translation_window_m": (
                ARGS.terrain_emergency_recovery_translation_window_m
            ),
            "terrain_emergency_recovery_policy": {
                "enabled": terrain_emergency_recovery.enabled,
                "requires_preceding_recovery_layer": True,
                "strict_rank3": bool(
                    terrain_emergency_recovery_config is not None
                    and terrain_emergency_recovery_config
                    .require_yaw_observable
                ),
                "same_patch_pings": ARGS.terrain_match_window_pings,
                "failed_prior_threshold": (
                    terrain_emergency_recovery.failed_prior_threshold
                ),
                "minimum_estimated_raw_arc_m": (
                    terrain_emergency_recovery.minimum_raw_arc_m
                ),
                "minimum_time_since_prior_s": (
                    terrain_emergency_recovery
                    .minimum_time_since_prior_s
                ),
                "ordinary_attempt_interval_observations": (
                    terrain_emergency_recovery
                    .ordinary_attempt_interval_observations
                ),
                "allowed_preceding_rejection_reasons": [
                    "search_boundary_reached",
                    "insufficient_inliers",
                    "yaw_unobservable",
                    "unobservable_terrain",
                ],
                "ambiguity_expansion_allowed": False,
                "coarse_translation_step_m": (
                    None
                    if terrain_emergency_recovery_config is None
                    else terrain_emergency_recovery_config
                    .coarse_translation_step_m
                ),
                "coarse_top_k": (
                    None
                    if terrain_emergency_recovery_config is None
                    else terrain_emergency_recovery_config.coarse_top_k
                ),
                "coarse_seed_translation_separation_m": (
                    None
                    if terrain_emergency_recovery_config is None
                    else terrain_emergency_recovery_config
                    .coarse_seed_translation_separation_m
                ),
                "minimum_basin_translation_separation_m": (
                    None
                    if terrain_emergency_recovery_config is None
                    else terrain_emergency_recovery_config
                    .minimum_basin_translation_separation_m
                ),
                "minimum_basin_log_likelihood_ratio": (
                    None
                    if terrain_emergency_recovery_config is None
                    else terrain_emergency_recovery_config
                    .minimum_basin_log_likelihood_ratio
                ),
                "first_match_fused": False,
                "confirmed_factor_optimize_interval": 1,
                "confirmed_factor_relocalization_seed": True,
                "confirmed_factor_robust_kernel": False,
                "confirmation_requires_actual_nonoverlapping_patch_ids": True,
                "confirmation_requires_actual_nonoverlapping_patch_times": True,
                "confirmation_minimum_estimated_raw_arc_m": (
                    terrain_emergency_recovery
                    .confirmation_minimum_raw_arc_m
                ),
                "confirmation_minimum_time_separation_s": (
                    terrain_emergency_recovery
                    .confirmation_minimum_time_separation_s
                ),
                "confirmation_max_future_observations": (
                    terrain_emergency_recovery
                    .confirmation_max_future_observations
                ),
                "confirmation_position_tolerance_m": (
                    terrain_emergency_recovery
                    .confirmation_position_tolerance_m
                ),
                "confirmation_yaw_tolerance_deg": (
                    terrain_emergency_recovery
                    .confirmation_yaw_tolerance_deg
                ),
                "cooldown_observations": (
                    terrain_emergency_recovery.cooldown_observations
                ),
                "maximum_automatic_translation_window_m": 150.0,
                "uses_truth": False,
            },
            "terrain_cohort_particle_backend": {
                "enabled": terrain_backend_particle_tracker is not None,
                "filter": "joint_point_mass_cohort_particle_trn",
                "role": (
                    "persistent_multimodal_data_association_and_current_"
                    "window_rank_aware_factor_authorization_before_gtsam"
                ),
                "state": "map-frame vehicle SE(2)",
                "motion_source": "local_acoustic_inertial_odometry_se2_increment",
                "persistent_proposal_provenance_enforced": True,
                "graph_measurement_source": (
                    "current_independent_raw_sonar_likelihood_local_laplace"
                ),
                "online_output_source": (
                    "policy_selected_propagated_MAP_cohort_or_pose_graph"
                ),
                "online_output_is_factor_authorization": False,
                "online_output_statistics": dict(
                    terrain_navigation_output_stats
                ),
                "recursive_particle_posterior_used_as_graph_measurement": False,
                "unobservable_directions_preserve_zero_information": True,
                "all_cohorts_share_one_raw_likelihood_normalization": True,
                "candidate_birth_mass_drawn_from_complete_prior": True,
                "particles_per_cohort": (
                    None
                    if terrain_backend_particle_tracker is None
                    else terrain_backend_particle_tracker.config
                    .particles_per_cohort
                ),
                "explicit_null_hypothesis": True,
                "statistics": dict(terrain_backend_particle_stats),
                "diagnostics": (
                    None
                    if terrain_backend_particle_tracker is None
                    else terrain_backend_particle_tracker.diagnostics()
                ),
                "records": terrain_backend_particle_records,
                "uses_truth": False,
            },
            "terrain_global_relocalization": {
                "enabled": global_terrain_relocalizer is not None,
                "search_scope": "complete_measured_map",
                "proposal_only": True,
                "requires_preceding_bounded_failure": True,
                "requires_joint_bathymetry_backscatter": True,
                "requires_local_bathymetry_refinement": True,
                "requires_local_backscatter_refinement": True,
                "requires_second_disjoint_strip_confirmation": (
                    global_cohort_tracker is None
                ),
                "persistent_cohort_confirmation_replaces_legacy_hold": (
                    global_cohort_tracker is not None
                ),
                "confirmation_motion_source": (
                    "local_acoustic_inertial_odometry_se2_increment"
                ),
                "patch_motion_source": {
                    "acoustic-fixed-lag": (
                        "causal_fixed_lag_imu_dvl_motion_with_acoustic_endpoints"
                    ),
                    "acoustic-predictive": (
                        "causal_acoustic_inertial_prediction_before_current_ping"
                    ),
                    "inertial": "independent_imu_dvl_depth_dead_reckoning",
                }[ARGS.terrain_patch_motion_source],
                "window_pings": ARGS.terrain_global_window_pings,
                "nominal_window_duration_s": float(
                    ARGS.terrain_global_window_pings / ARGS.ping_hz
                ),
                "nominal_window_along_track_m": float(
                    ARGS.terrain_global_window_pings
                    * ARGS.survey_speed_mps
                    / ARGS.ping_hz
                ),
                "pyramid_stride_source_cells": (
                    ARGS.terrain_global_pyramid_stride
                ),
                "pyramid_resolution_m": (
                    None
                    if global_terrain_relocalizer is None
                    else global_terrain_relocalizer.resolution_m
                ),
                "yaw_half_window_deg": ARGS.terrain_global_yaw_window_deg,
                "yaw_step_deg": ARGS.terrain_global_yaw_step_deg,
                "minimum_joint_correlation": (
                    ARGS.terrain_global_minimum_correlation
                ),
                "maximum_bathymetry_rmse_m": (
                    ARGS.terrain_global_maximum_bathymetry_rmse_m
                ),
                "bathymetry_absolute_weight": (
                    ARGS.terrain_global_bathymetry_absolute_weight
                ),
                "minimum_basin_score_gap": (
                    ARGS.terrain_global_minimum_score_gap
                ),
                "persistent_cohort_tracking": {
                    "enabled": global_cohort_tracker is not None,
                    "state": "map-frame vehicle pose",
                    "motion_source": "imu_dvl_depth_dead_reckoning_se2_increment",
                    "filter": "joint_point_mass_cohort_particle_trn",
                    "explicit_null_hypothesis": True,
                    "all_spatially_separated_candidates_retained": True,
                    "all_cohorts_share_one_raw_likelihood_normalization": True,
                    "candidate_birth_mass_drawn_from_complete_prior": True,
                    "same_observation_refined_basin_nms": True,
                    "overlapping_window_evidence_discounted": True,
                    "same_observation_nms_position_m": (
                        None
                        if global_cohort_tracker is None
                        else global_cohort_tracker.config
                        .same_observation_nms_position_m
                    ),
                    "same_observation_nms_yaw_deg": (
                        None
                        if global_cohort_tracker is None
                        else global_cohort_tracker.config
                        .same_observation_nms_yaw_deg
                    ),
                    "maximum_cohorts": (
                        None
                        if global_cohort_tracker is None
                        else global_cohort_tracker.config.maximum_cohorts
                    ),
                    "particles_per_cohort": (
                        None
                        if global_cohort_tracker is None
                        else global_cohort_tracker.config.particles_per_cohort
                    ),
                    "resample_effective_sample_fraction": (
                        None
                        if global_cohort_tracker is None
                        else global_cohort_tracker.config
                        .resample_effective_sample_fraction
                    ),
                    "resample_kernel_bandwidth": (
                        None
                        if global_cohort_tracker is None
                        else global_cohort_tracker.config
                        .resample_kernel_bandwidth
                    ),
                    "minimum_independent_evidence": (
                        None
                        if global_cohort_tracker is None
                        else global_cohort_tracker.config
                        .minimum_independent_evidence
                    ),
                    "confirmation_probability": (
                        None
                        if global_cohort_tracker is None
                        else global_cohort_tracker.config
                        .confirmation_probability
                    ),
                    "confirmation_odds_over_second": (
                        None
                        if global_cohort_tracker is None
                        else global_cohort_tracker.config
                        .confirmation_odds_over_second
                    ),
                    "diagnostics": (
                        None
                        if global_cohort_tracker is None
                        else global_cohort_tracker.diagnostics()
                    ),
                    "uses_truth": False,
                },
                "bbs3d_geometric_verification": {
                    "enabled": bbs3d_candidate_verifier is not None,
                    "backend": "KOKIAOKI/3d_bbs@41529a34a2fb9618b5ff560fb3c2363f1615666d",
                    "role": "candidate_verifier_not_standalone_absolute_factor",
                    "search_state": "residual_x_y_yaw",
                    "pressure_depth_fixed": True,
                    "imu_roll_pitch_fixed": True,
                    "minimum_level_resolution_m": (
                        None
                        if bbs3d_candidate_verifier is None
                        else bbs3d_candidate_verifier.config
                        .minimum_level_resolution_m
                    ),
                    "maximum_translation_m": (
                        None
                        if bbs3d_candidate_verifier is None
                        else bbs3d_candidate_verifier.config
                        .maximum_translation_m
                    ),
                    "maximum_yaw_deg": (
                        None
                        if bbs3d_candidate_verifier is None
                        else bbs3d_candidate_verifier.config.maximum_yaw_deg
                    ),
                    "minimum_score_fraction": (
                        None
                        if bbs3d_candidate_verifier is None
                        else bbs3d_candidate_verifier.config
                        .minimum_score_fraction
                    ),
                    "timeout_milliseconds": (
                        None
                        if bbs3d_candidate_verifier is None
                        else bbs3d_candidate_verifier.config
                        .timeout_milliseconds
                    ),
                    "uses_truth": False,
                },
                "local_backscatter_refinement": {
                    "maximum_points": (
                        None
                        if global_backscatter_refinement_matcher is None
                        else global_backscatter_refinement_matcher
                        .config.maximum_points
                    ),
                    "maximum_translation_m": (
                        None
                        if global_backscatter_refinement_matcher is None
                        else global_backscatter_refinement_matcher
                        .config.maximum_translation_m
                    ),
                    "maximum_yaw_deg": (
                        None
                        if global_backscatter_refinement_matcher is None
                        else global_backscatter_refinement_matcher
                        .config.maximum_yaw_deg
                    ),
                    "basin_translation_separation_m": (
                        None
                        if global_backscatter_refinement_matcher is None
                        else global_backscatter_refinement_matcher
                        .config.basin_translation_separation_m
                    ),
                    "basin_yaw_separation_deg": (
                        None
                        if global_backscatter_refinement_matcher is None
                        else global_backscatter_refinement_matcher
                        .config.basin_yaw_separation_deg
                    ),
                    "minimum_correlation": (
                        None
                        if global_backscatter_refinement_matcher is None
                        else global_backscatter_refinement_matcher
                        .config.minimum_correlation
                    ),
                    "minimum_basin_correlation_gap": (
                        None
                        if global_backscatter_refinement_matcher is None
                        else global_backscatter_refinement_matcher
                        .config.minimum_basin_correlation_gap
                    ),
                    "sc_match_inspired_scale_space_context": {
                        "exact_paper_implementation": False,
                        "reason": (
                            "runtime input is geocoded reflectivity points "
                            "against a prior mosaic, not two raw waterfall images"
                        ),
                        "scale_cell_sizes_m": (
                            None
                            if global_backscatter_refinement_matcher is None
                            else list(
                                global_backscatter_refinement_matcher.config
                                .scale_space_cell_sizes_m
                            )
                        ),
                        "scale_space_weight": (
                            None
                            if global_backscatter_refinement_matcher is None
                            else global_backscatter_refinement_matcher.config
                            .scale_space_weight
                        ),
                        "context_consistency_weight": (
                            None
                            if global_backscatter_refinement_matcher is None
                            else global_backscatter_refinement_matcher.config
                            .context_consistency_weight
                        ),
                        "context_length_m": (
                            None
                            if global_backscatter_refinement_matcher is None
                            else global_backscatter_refinement_matcher.config
                            .context_length_m
                        ),
                    },
                },
                "confirmation_position_tolerance_m": (
                    ARGS.terrain_global_confirmation_position_tolerance_m
                ),
                "confirmation_yaw_tolerance_deg": (
                    ARGS.terrain_global_confirmation_yaw_tolerance_deg
                ),
                "uses_truth": False,
            },
            "terrain_match_yaw_window_deg": (
                ARGS.terrain_match_yaw_window_deg
            ),
            "terrain_require_yaw_observable": (
                ARGS.terrain_require_yaw_observable
            ),
            "terrain_position_only_fallback": {
                "enabled": terrain_position_fallback_matcher is not None,
                "policy": (
                    "only after a strict primary result is rejected solely "
                    "as yaw_unobservable; the identical solve is rerun with "
                    "rank-two output permitted so boundary, residual, "
                    "ambiguity and innovation checks still execute"
                ),
                "invocations": int(
                    terrain_position_fallback_stats["invocations"]
                ),
                "accepts": int(
                    terrain_position_fallback_stats["accepts"]
                ),
                "rejections": int(
                    terrain_position_fallback_stats["rejections"]
                ),
                "rejection_reasons": dict(
                    terrain_position_fallback_stats["rejection_reasons"]
                ),
                "wall_time_s": float(
                    terrain_position_fallback_stats["wall_time_s"]
                ),
                "uses_simulator_truth": False,
            },
            "terrain_initialize_global_gauge": bool(
                ARGS.terrain_initialize_global_gauge
            ),
            "terrain_prior_optimize_interval": (
                ARGS.terrain_prior_optimize_interval
            ),
            "terrain_prior_robust_kernel": (
                ARGS.terrain_prior_robust_kernel
            ),
            "terrain_relocalization_prior_robust_kernel": (
                slam.config.pose_graph.relocalization_prior_robust_kernel
            ),
            "terrain_relocalization_prior_robust_delta": (
                slam.config.pose_graph.robust_relocalization_prior_delta
            ),
            # Observation counts remain independent of the one-to-three matcher
            # invocations used by the adaptive policy.
            "terrain_match_attempts": terrain_match_attempts,
            "terrain_match_observations": terrain_match_attempts,
            "accepted_terrain_matches": terrain_match_accepts,
            "rejected_terrain_matches": (
                terrain_match_attempts - terrain_match_accepts
            ),
            "accepted_terrain_match_observations": terrain_match_accepts,
            "rejected_terrain_match_observations": (
                terrain_match_attempts - terrain_match_accepts
            ),
            "terrain_match_invocations": terrain_match_invocations,
            "accepted_terrain_match_invocations": (
                terrain_match_invocation_accepts
            ),
            "rejected_terrain_match_invocations": (
                terrain_match_invocations
                - terrain_match_invocation_accepts
            ),
            "accepted_terrain_prior_factors": terrain_prior_accepts,
            "accepted_terrain_prior_observations": terrain_prior_accepts,
            "rejected_terrain_prior_observations": (
                terrain_match_attempts - terrain_prior_accepts
            ),
            "terrain_prior_factor_attempts": int(
                terrain_prior_factor_stats["attempts"]
            ),
            "rejected_terrain_prior_factor_attempts": int(
                terrain_prior_factor_stats["rejections"]
            ),
            "terrain_prior_factor_rejection_reasons": dict(
                terrain_prior_factor_stats["rejection_reasons"]
            ),
            "normal_terrain_match_invocations": int(
                normal_terrain_stats["invocations"]
            ),
            "normal_terrain_match_accepts": int(
                normal_terrain_stats["accepts"]
            ),
            "normal_terrain_match_rejections": int(
                normal_terrain_stats["rejections"]
            ),
            "normal_terrain_match_rejection_reasons": dict(
                normal_terrain_stats["rejection_reasons"]
            ),
            "normal_terrain_match_observable_rank_counts": dict(
                normal_terrain_stats["rank_counts"]
            ),
            "normal_terrain_prior_factor_attempts": int(
                normal_terrain_stats["prior_factor_attempts"]
            ),
            "normal_terrain_prior_factor_accepts": int(
                normal_terrain_stats["prior_factor_accepts"]
            ),
            "normal_terrain_prior_factor_rejections": int(
                normal_terrain_stats["prior_factor_rejections"]
            ),
            "normal_terrain_matching_wall_time_s": float(
                normal_terrain_stats["wall_time_s"]
            ),
            "recovery_terrain_match_invocations": int(
                recovery_terrain_stats["invocations"]
            ),
            "recovery_terrain_match_accepts": int(
                recovery_terrain_stats["accepts"]
            ),
            "recovery_terrain_match_rejections": int(
                recovery_terrain_stats["rejections"]
            ),
            "recovery_terrain_match_rejection_reasons": dict(
                recovery_terrain_stats["rejection_reasons"]
            ),
            "recovery_terrain_match_observable_rank_counts": dict(
                recovery_terrain_stats["rank_counts"]
            ),
            "recovery_terrain_prior_factor_attempts": int(
                recovery_terrain_stats["prior_factor_attempts"]
            ),
            "recovery_terrain_prior_factor_accepts": int(
                recovery_terrain_stats["prior_factor_accepts"]
            ),
            "recovery_terrain_prior_factor_rejections": int(
                recovery_terrain_stats["prior_factor_rejections"]
            ),
            "recovery_terrain_matching_wall_time_s": float(
                recovery_terrain_stats["wall_time_s"]
            ),
            "emergency_terrain_match_invocations": int(
                emergency_terrain_stats["invocations"]
            ),
            "emergency_terrain_match_accepts": int(
                emergency_terrain_stats["accepts"]
            ),
            "emergency_terrain_match_rejections": int(
                emergency_terrain_stats["rejections"]
            ),
            "emergency_terrain_match_rejection_reasons": dict(
                emergency_terrain_stats["rejection_reasons"]
            ),
            "emergency_terrain_match_observable_rank_counts": dict(
                emergency_terrain_stats["rank_counts"]
            ),
            "emergency_terrain_prior_factor_attempts": int(
                emergency_terrain_stats["prior_factor_attempts"]
            ),
            "emergency_terrain_prior_factor_accepts": int(
                emergency_terrain_stats["prior_factor_accepts"]
            ),
            "emergency_terrain_prior_factor_rejections": int(
                emergency_terrain_stats["prior_factor_rejections"]
            ),
            "emergency_terrain_matching_wall_time_s": float(
                emergency_terrain_stats["wall_time_s"]
            ),
            "global_terrain_match_invocations": int(
                global_terrain_stats["invocations"]
            ),
            "global_terrain_match_accepts": int(
                global_terrain_stats["accepts"]
            ),
            "global_terrain_match_rejections": int(
                global_terrain_stats["rejections"]
            ),
            "global_terrain_match_rejection_reasons": dict(
                global_terrain_stats["rejection_reasons"]
            ),
            "global_terrain_match_observable_rank_counts": dict(
                global_terrain_stats["rank_counts"]
            ),
            "global_terrain_prior_factor_attempts": int(
                global_terrain_stats["prior_factor_attempts"]
            ),
            "global_terrain_prior_factor_accepts": int(
                global_terrain_stats["prior_factor_accepts"]
            ),
            "global_terrain_prior_factor_rejections": int(
                global_terrain_stats["prior_factor_rejections"]
            ),
            "global_terrain_matching_wall_time_s": float(
                global_terrain_stats["wall_time_s"]
            ),
            "global_relocalization_proposal_statistics": {
                key: (
                    dict(value)
                    if isinstance(value, Counter)
                    else (
                        _finite_float_or_none(value)
                        if isinstance(value, (float, np.floating))
                        else value
                    )
                )
                for key, value in global_relocalization_stats.items()
            },
            "global_relocalization_records": global_relocalization_records,
            "global_cohort_records": global_cohort_records,
            "emergency_recovery_attempts": int(
                terrain_emergency_recovery.recovery_attempts
            ),
            "emergency_pending_hypotheses_created": int(
                terrain_emergency_recovery.pending_hypotheses
            ),
            "emergency_confirmed_recoveries": int(
                terrain_emergency_recovery.confirmed_recoveries
            ),
            "emergency_inconsistent_hypotheses": int(
                terrain_emergency_recovery.inconsistent_hypotheses
            ),
            "emergency_expired_hypotheses": int(
                terrain_emergency_recovery.expired_hypotheses
            ),
            "emergency_preceding_factor_rejections": int(
                terrain_emergency_recovery.preceding_factor_rejections
            ),
            "emergency_pending_at_end": bool(
                terrain_emergency_recovery.has_pending_hypothesis
            ),
            "terrain_consecutive_failed_prior_observations": (
                terrain_recovery.consecutive_failed_observations
            ),
            "terrain_maximum_consecutive_failed_prior_observations": (
                terrain_recovery.maximum_consecutive_failed_observations
            ),
            "terrain_recovery_events": terrain_recovery.recovery_events,
            "terrain_recovery_invocation_records": (
                terrain_recovery_invocation_records
            ),
            "terrain_match_truth_evaluation": {
                "evaluation_only": True,
                "used_by_matcher_or_estimator": False,
                "definition": (
                    "unaligned matcher seed and output error against the "
                    "synchronized simulated pose at each matcher invocation; "
                    "truth is recorded only after matching for diagnosis"
                ),
                "accepted_invocation_count": terrain_truth_evaluation_count,
                "seed_horizontal_rmse_m": terrain_truth_evaluation_rmse(
                    "seed_horizontal_squared_m2"
                ),
                "matched_horizontal_rmse_m": terrain_truth_evaluation_rmse(
                    "matched_horizontal_squared_m2"
                ),
                "seed_yaw_rmse_deg": terrain_truth_evaluation_rmse(
                    "seed_yaw_squared_deg2"
                ),
                "matched_yaw_rmse_deg": terrain_truth_evaluation_rmse(
                    "matched_yaw_squared_deg2"
                ),
                "records": terrain_match_truth_evaluation_records,
            },
            "terrain_recovery_active_at_end": terrain_recovery.recovery_active,
            "terrain_prior_prefit_chi2": {
                "accepted": _chi_square_sample_summary(
                    terrain_prefit_accepted_chi2
                ),
                "rejected": _chi_square_sample_summary(
                    terrain_prefit_rejected_chi2
                ),
                "rank2_threshold": float(
                    slam.pose_graph.config
                    .terrain_prior_prefit_chi2_rank2
                ),
                "rank3_threshold": float(
                    slam.pose_graph.config
                    .terrain_prior_prefit_chi2_rank3
                ),
            },
            "terrain_match_rejection_reasons": dict(
                terrain_match_rejection_reasons
            ),
            "terrain_match_invocation_rejection_reasons": dict(
                terrain_match_rejection_reasons
            ),
            "terrain_match_observable_rank_counts": dict(
                terrain_match_rank_counts
            ),
            "mean_accepted_terrain_rmse_before_m": (
                terrain_accepted_invocation_stats["rmse_before_sum"]
                / max(accepted_invocation_count, 1)
            ),
            "mean_accepted_terrain_rmse_after_m": (
                terrain_accepted_invocation_stats["rmse_after_sum"]
                / max(accepted_invocation_count, 1)
            ),
            "mean_accepted_terrain_translation_correction_m": (
                terrain_accepted_invocation_stats[
                    "translation_correction_sum"
                ]
                / max(accepted_invocation_count, 1)
            ),
            "mean_accepted_terrain_yaw_correction_deg": (
                terrain_accepted_invocation_stats["yaw_correction_sum_deg"]
                / max(accepted_invocation_count, 1)
            ),
            "maximum_accepted_terrain_translation_correction_m": (
                terrain_accepted_invocation_stats[
                    "maximum_translation_correction_m"
                ]
            ),
            "maximum_accepted_terrain_abs_yaw_correction_deg": (
                terrain_accepted_invocation_stats[
                    "maximum_abs_yaw_correction_deg"
                ]
            ),
            "terrain_matching_wall_time_s": terrain_match_elapsed_s,
            "terrain_matching_mean_time_ms": (
                1.0e3
                * terrain_match_elapsed_s
                / max(terrain_match_invocations, 1)
            ),
            "last_terrain_match_mode": last_terrain_match_mode,
            "last_terrain_match": (
                None
                if last_terrain_match is None
                else {
                    "accepted": last_terrain_match.accepted,
                    "rejection_reason": last_terrain_match.rejection_reason,
                    "correspondences": last_terrain_match.correspondences,
                    "inliers": last_terrain_match.inliers,
                    "observable_rank": last_terrain_match.observable_rank,
                    "translation_observable_rank": (
                        last_terrain_match.translation_observable_rank
                    ),
                    "yaw_observable": last_terrain_match.yaw_observable,
                    "rmse_before_m": last_terrain_match.rmse_before,
                    "rmse_after_m": last_terrain_match.rmse_after,
                    "reduced_chi_square": (
                        last_terrain_match.reduced_chi_square
                    ),
                    "innovation_nis_diagnostic_only": (
                        last_terrain_match.innovation_nis
                    ),
                    "translation_correction_m": (
                        last_terrain_match.translation_correction_m
                    ),
                    "yaw_correction_deg": (
                        last_terrain_match.yaw_correction_deg
                    ),
                    "distinct_basin_count": int(getattr(
                        last_terrain_match, "distinct_basin_count", 0
                    )),
                    "best_basin_cost": _finite_float_or_none(getattr(
                        last_terrain_match, "best_basin_cost", None
                    )),
                    "second_best_basin_cost": _finite_float_or_none(getattr(
                        last_terrain_match, "second_best_basin_cost", None
                    )),
                    "basin_cost_gap": _finite_float_or_none(getattr(
                        last_terrain_match, "basin_cost_gap", None
                    )),
                    "basin_log_likelihood_ratio": _finite_float_or_none(getattr(
                        last_terrain_match,
                        "basin_log_likelihood_ratio",
                        None,
                    )),
                    "minimum_basin_log_likelihood_ratio": _finite_float_or_none(getattr(
                        last_terrain_match,
                        "minimum_basin_log_likelihood_ratio",
                        None,
                    )),
                }
            ),
            "sonar_backend": acoustic_backend_label,
            "sonar_backend_gpu": bool(
                getattr(acoustic_backend, "is_cuda", False)
            ),
            "sonar_mesh_triangles": int(
                getattr(acoustic_backend, "triangle_count", 0)
            ),
            "sonar_bvh_build_ms": float(
                getattr(acoustic_backend, "build_ms", 0.0)
            ),
            "sonar_mean_cast_ms": float(
                getattr(acoustic_backend, "mean_cast_ms", 0.0)
            ),
            "sonar_cast_calls": int(
                getattr(acoustic_backend, "cast_count", 0)
            ),
            "sonar_backend_verification": backend_verification,
            "slam_association_backend": slam_association_label,
            "slam_max_points": ARGS.slam_max_points,
            "slam_voxel_size_m": ARGS.slam_voxel_size_m,
            "intensity_voxel_size_m": ARGS.intensity_voxel_size_m,
            "map_insert_stride": ARGS.map_insert_stride,
            "trajectory_sample_interval_s": (
                ARGS.trajectory_sample_interval_s
            ),
            "ui_display_trajectory_samples_cached": (
                len(display_timestamps) if not ARGS.headless else 0
            ),
            "ui_display_trajectory_sample_spacing_m": (
                display_sample_spacing_m if not ARGS.headless else None
            ),
            "mbes_group_pings": ARGS.mbes_group_pings,
            "mbes_sliding_window_enabled": (
                ARGS.enable_mbes_sliding_window
            ),
            "mbes_sliding_window_information_scale": (
                1.0 / ARGS.mbes_group_pings
                if ARGS.enable_mbes_sliding_window
                else 1.0
            ),
            "association_map_half_extent_m": (
                ARGS.association_map_half_extent_m
            ),
            "slam_association_gpu": bool(
                getattr(slam_associator, "is_cuda", False)
            ),
            "slam_association_warmup_ms": float(
                getattr(slam_associator, "warmup_ms", 0.0)
            ),
            "slam_association_mean_ms": float(
                slam.filter.mean_association_ms
            ),
            "slam_association_calls": int(
                slam.filter.association_calls
            ),
            "slam_gpu_call_mean_ms": float(
                getattr(slam_associator, "mean_association_ms", 0.0)
            ),
            "slam_association_gpu_calls": int(
                getattr(slam_associator, "gpu_calls", 0)
            ),
            "slam_association_cpu_fallbacks": int(
                getattr(slam_associator, "cpu_fallbacks", 0)
            ),
            "sim_time_s": sim_time,
            "wall_time_s": time.monotonic() - started,
            "requested_time_scale": ARGS.time_scale,
            "achieved_time_scale_including_startup": (
                sim_time / max(time.monotonic() - started, 1.0e-12)
            ),
            "ui_sonar_refresh_limit_hz": ARGS.ui_sonar_refresh_hz,
            "sonar_visualization_updates": sonar_visualization_updates,
            "effective_sonar_visualization_rate_sim_hz": (
                sonar_visualization_updates / max(sim_time, 1.0e-12)
            ),
            "performance_wall_time_breakdown_s": {
                key: float(value)
                for key, value in sorted(performance_wall_s.items())
            },
            "frames": frame_count,
            "imu_rate_hz": 1.0 / imu_dt,
            "imu_samples": imu_sample_count,
            "dvl_rate_hz": 1.0 / dvl_period,
            "dvl_samples": dvl_sample_count,
            "depth_rate_hz": 1.0 / depth_period,
            "depth_samples": depth_sample_count,
            "pings": ping_count,
            "mbes_map_points": diagnostics.map_points,
            "mbes_total_committed_points": (
                diagnostics.total_geometry_points_committed
            ),
            "mbes_association_map_voxels": (
                diagnostics.association_map_voxels
            ),
            "mbes_map_voxels": diagnostics.map_voxels,
            "mbes_mature_surfels": diagnostics.mature_surfels,
            "sidescan_lines": diagnostics.sidescan_lines,
            "integrated_sonar_pings": diagnostics.integrated_sonar_pings,
            "coregistered_intensity_samples": (
                diagnostics.coregistered_intensity_samples
            ),
            "legacy_range_attached_intensity_samples": (
                diagnostics.legacy_range_attached_intensity_samples
            ),
            "accepted_geometry_updates": diagnostics.accepted_sonar_updates,
            "rejected_geometry_updates": diagnostics.rejected_sonar_updates,
            "accepted_intensity_updates": diagnostics.accepted_intensity_updates,
            "accepted_local_intensity_updates": (
                diagnostics.accepted_local_intensity_updates
            ),
            "accepted_local_intensity_updates_with_graph_factor": (
                diagnostics.accepted_local_intensity_updates_with_graph_factor
            ),
            "accepted_local_intensity_updates_without_graph_factor": (
                diagnostics.accepted_local_intensity_updates_without_graph_factor
            ),
            "accepted_graph_seeded_intensity_updates": (
                diagnostics.accepted_graph_seeded_intensity_updates
            ),
            "rejected_intensity_updates": diagnostics.rejected_intensity_updates,
            "committed_geometry_groups": diagnostics.committed_geometry_groups,
            "quarantined_geometry_groups": diagnostics.quarantined_geometry_groups,
            "committed_intensity_batches": diagnostics.committed_intensity_batches,
            "quarantined_intensity_batches": diagnostics.quarantined_intensity_batches,
            "sonar_rejection_reasons": diagnostics.sonar_rejection_reasons,
            "sonar_observable_rank_counts": (
                diagnostics.sonar_observable_rank_counts
            ),
            "intensity_rejection_reasons": (
                diagnostics.intensity_rejection_reasons
            ),
            "intensity_planar_rank_counts": (
                diagnostics.intensity_planar_rank_counts
            ),
            "mean_accepted_intensity_correlation": (
                diagnostics.mean_accepted_intensity_correlation
            ),
            "minimum_accepted_intensity_correlation": (
                diagnostics.minimum_accepted_intensity_correlation
            ),
            "mean_accepted_intensity_overlap": (
                diagnostics.mean_accepted_intensity_overlap
            ),
            "cumulative_intensity_translation_correction_m": (
                diagnostics.cumulative_intensity_translation_correction_m
            ),
            "cumulative_intensity_rotation_correction_deg": (
                diagnostics.cumulative_intensity_rotation_correction_deg
            ),
            "pending_intensity_batches": diagnostics.pending_intensity_batches,
            "pending_intensity_points": diagnostics.pending_intensity_points,
            "intensity_map_points": diagnostics.intensity_map_points,
            "intensity_map_cells": diagnostics.intensity_map_cells,
            "formed_intensity_strips": diagnostics.formed_intensity_strips,
            "skipped_intensity_strips": diagnostics.skipped_intensity_strips,
            "last_intensity_strip_span_m": (
                diagnostics.last_intensity_strip_span_m
            ),
            "keyframes": diagnostics.pose_graph.keyframes,
            "loop_edges": diagnostics.pose_graph.loop_edges,
            "relocalization_edges": (
                diagnostics.pose_graph.relocalization_edges
            ),
            "terrain_prior_edges": (
                diagnostics.pose_graph.terrain_prior_count
            ),
            "pose_graph_accepted_terrain_priors": (
                diagnostics.pose_graph.accepted_terrain_priors
            ),
            "pose_graph_rejected_terrain_priors": (
                diagnostics.pose_graph.rejected_terrain_priors
            ),
            "terrain_prior_factor_rejection_reasons": (
                diagnostics.pose_graph.terrain_prior_rejection_reasons
            ),
            "last_terrain_prior_factor_rejection_reason": (
                diagnostics.pose_graph.last_terrain_prior_rejection_reason
            ),
            "last_terrain_prefit_performed": getattr(
                diagnostics.pose_graph, "last_terrain_prefit_performed", False
            ),
            "last_terrain_prefit_bypassed_for_relocalization": getattr(
                diagnostics.pose_graph,
                "last_terrain_prefit_bypassed_for_relocalization",
                False,
            ),
            "last_terrain_prefit_chi2": _finite_float_or_none(getattr(
                diagnostics.pose_graph, "last_terrain_prefit_chi2", None
            )),
            "last_terrain_prefit_dof": int(getattr(
                diagnostics.pose_graph, "last_terrain_prefit_dof", 0
            )),
            "terrain_relocalization_seeds": int(getattr(
                diagnostics.pose_graph, "terrain_relocalization_seeds", 0
            )),
            "last_terrain_relocalization_translation_m": float(getattr(
                diagnostics.pose_graph,
                "last_terrain_relocalization_translation_m",
                0.0,
            )),
            "last_terrain_relocalization_yaw_deg": float(getattr(
                diagnostics.pose_graph,
                "last_terrain_relocalization_yaw_deg",
                0.0,
            )),
            "last_terrain_relocalization_anchor_index": getattr(
                diagnostics.pose_graph,
                "last_terrain_relocalization_anchor_index",
                None,
            ),
            "intensity_factor_edges": (
                diagnostics.pose_graph.intensity_edges
            ),
            "rejected_intensity_factors": (
                diagnostics.pose_graph.rejected_intensity_factors
            ),
            "throttled_intensity_factors": (
                diagnostics.pose_graph.throttled_intensity_factors
            ),
            "intensity_factor_rejection_reasons": (
                diagnostics.pose_graph.intensity_factor_rejection_reasons
            ),
            "last_intensity_factor_rejection_reason": (
                diagnostics.pose_graph.last_intensity_factor_rejection_reason
            ),
            "last_intensity_prefit_performed": (
                diagnostics.pose_graph.last_intensity_prefit_performed
            ),
            "last_intensity_prefit_chi2": (
                diagnostics.pose_graph.last_intensity_prefit_chi2
            ),
            "last_intensity_prefit_dof": (
                diagnostics.pose_graph.last_intensity_prefit_dof
            ),
            "pose_graph_optimization_runs": (
                diagnostics.pose_graph.optimization_runs
            ),
            "pose_graph_optimizer_backend": (
                diagnostics.pose_graph.optimizer_backend
            ),
            "pose_graph_isam_updates": diagnostics.pose_graph.isam_updates,
            "pose_graph_fixed_lag_updates": (
                diagnostics.pose_graph.fixed_lag_updates
            ),
            "pose_graph_fixed_lag_active_variables": (
                diagnostics.pose_graph.fixed_lag_active_variables
            ),
            "pose_graph_fixed_lag_marginalized_variables": (
                diagnostics.pose_graph.fixed_lag_marginalized_variables
            ),
            "pose_graph_fixed_lag_global_only_factors": (
                diagnostics.pose_graph.fixed_lag_global_only_factors
            ),
            "pose_graph_optimizer_rebuilds": (
                diagnostics.pose_graph.optimizer_rebuilds
            ),
            "pose_graph_global_anchor_bridge_updates": (
                diagnostics.pose_graph.global_anchor_bridge_updates
            ),
            "pose_graph_global_anchor_bridge_max_translation_m": (
                diagnostics.pose_graph
                .global_anchor_bridge_max_translation_m
            ),
            "pose_graph_global_anchor_bridge_max_abs_yaw_deg": (
                diagnostics.pose_graph
                .global_anchor_bridge_max_abs_yaw_deg
            ),
            "pose_graph_global_isam_initial_cost": (
                diagnostics.pose_graph.global_isam_initial_cost
            ),
            "pose_graph_global_isam_final_cost": (
                diagnostics.pose_graph.global_isam_final_cost
            ),
            "online_map_to_odom_change_audit": map_to_odom_audit,
            "map_to_odom_soft_correction": {
                "enabled": bool(ARGS.enable_map_to_odom_soft_correction),
                "role": "online_output_only_not_frontend_state_feedback",
                "matcher_seed_uses_unsmoothed_graph_target": True,
                "translation_time_constant_s": (
                    ARGS.map_to_odom_translation_time_constant_s
                ),
                "yaw_time_constant_s": (
                    ARGS.map_to_odom_yaw_time_constant_s
                ),
                "maximum_translation_rate_mps": (
                    ARGS.map_to_odom_maximum_translation_rate_mps
                ),
                "maximum_yaw_rate_deg_s": (
                    ARGS.map_to_odom_maximum_yaw_rate_deg_s
                ),
                "final_diagnostics": (
                    slam.map_to_odom_corrector.diagnostics()
                ),
            },
            "local_error_m": float(np.linalg.norm(slam.state.position - truth.position)),
            "global_error_m": float(np.linalg.norm(global_position - truth.position)),
            "position_error_statistics": position_errors,
            "orientation_error_statistics": orientation_errors,
            "position_error_csv": str(error_csv),
            "pose_graph_factor_log": str(factor_log),
        }
        metrics, nonfinite_replacements = sanitize_for_strict_json(metrics)
        metrics["json_serialization"] = {
            "strict_json": True,
            "nonfinite_diagnostic_values_replaced_with_null": int(
                nonfinite_replacements
            ),
        }
        with ARGS.metrics.open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(
            "[OceanSim] live metrics: "
            f"{json.dumps(metrics, allow_nan=False)}",
            flush=True,
        )
        try:
            world.stop()
        except Exception:
            pass
        sonar.close()
        if acoustic_backend is not None:
            acoustic_backend.close()
        if slam_associator is not None:
            slam_associator.close()
        if status_window is not None:
            try:
                status_window.destroy()
            except Exception:
                pass
        simulation_app.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
