# Mature TRN + Fast-LIVO2-style front-end regression (2026-09-13)

## Scope

This change integrates a mature terrain-relative-navigation architecture with
OceanSim's acoustic Fast-LIVO2-style front end. It is an independent
BSD-3-Clause implementation informed by public work; it does not copy
MB-System/libtrnav GPL-3.0 source and it is not the official FAST-LIVO2
camera/LiDAR implementation.

The controlled run uses only the modeled integrated multibeam/side-scan sonar,
IMU, bottom-track DVL, and pressure depth sensor streams. The initial state is
simulator truth for this regression. Truth is used to synthesize observations
and score results, but map-frame x/y/yaw truth is not supplied to a matcher,
particle filter, or factor graph.

## Implemented chain

1. The existing 15-state acoustic-inertial ESIKF remains the local odometry
   front end: rotation, position, velocity, gyro bias, and accelerometer bias.
   No gravity state or DVL scale state was added.
2. Joint MBES relief/depth and co-registered side-scan appearance propose
   absolute map basins. Native 3D-BBS verifies global candidates.
3. A map-frame point-mass/cohort particle filter propagates each SE(2) basin
   using local odometry. Every cohort is evaluated in one joint raw-sonar
   likelihood call, with an explicit invalid/null state and change-point birth
   mass for recovery from a formerly confident alias.
4. Particle MAP is an output-only target while two or more spatial modes are
   active. It passes through the existing bounded `map -> odom` soft follower
   and never overwrites local ESIKF state.
5. Only a posterior-confirmed cohort with a current independent observation
   of rank at least two may add a robust GTSAM factor. The recursive particle
   posterior is never reused as a fresh graph measurement; the factor comes
   from a current-window raw-sonar Laplace approximation.
6. Rank-deficient information remains rank deficient in GTSAM. A separate
   finite positive-definite covariance is used only for reporting and particle
   sampling, eliminating the previous non-positive-definite covariance crash.

The architecture follows operational PMF/PF terrain navigation described by
[MBARI Terrain Relative Navigation](https://www.mbari.org/project/terrain-relative-navigation/)
and the public
[MB-System libtrnav tree](https://github.com/dwcaress/MB-System/tree/master/src/mbtrnav),
while global candidate/cohort handling follows the published
[Global UTAN](https://doi.org/10.1109/TIM.2026.3716492) structure. Candidate
geometry uses the pinned official
[3D-BBS implementation](https://github.com/KOKIAOKI/3d_bbs) at
`41529a34a2fb9618b5ff560fb3c2363f1615666d`.

## 3.3 km critical regression

Command:

```bash
standalone/run_backend_ab_gtsam_trn_full_route_isaac.sh \
  --headless --time-scale 3600 \
  --survey-segment-distance-m 3300 \
  --terrain-display-stride 32 \
  --metrics /tmp/oceansim_trn_final_regression_3300m_20260913.json
```

The archived seed-31 sensor stream, route, local ESIKF, and graph configuration
are unchanged between the graph-only and hybrid output runs.

| Result | Distance/status | Online global RMSE / P95 / max / final (m) | Final fixed-lag RMSE / max / final (m) |
| --- | --- | ---: | ---: |
| Previous implementation | failed at 3.060 km; non-PD covariance | 2.598 / 5.731 / 7.519 / 7.519 (incomplete run) | 1.915 / 7.519 / 7.519 (incomplete run) |
| Joint PF, graph-only online output | completed 3.300 km | 2.898 / 6.729 / 10.383 / 1.536 | 1.182 / 2.361 / 1.462 |
| Joint PF, multimodal MAP soft output | completed 3.300 km | **2.412 / 4.238 / 8.221 / 1.535** | **1.182 / 2.361 / 1.462** |

Against graph-only online output, the multimodal policy reduced online RMSE by
16.76%, P95 by 37.01%, and maximum error by 20.82%. It does not change the
final graph solution; it improves the causal output while GTSAM deliberately
holds ambiguous factors.

For the completed run:

| Track | 3-D RMSE (m) | Final error (m) |
| --- | ---: | ---: |
| IMU/DVL/depth dead reckoning | 82.527 | 225.258 |
| Local acoustic ESIKF | 21.786 | 58.921 |
| Online global SLAM | 2.412 | 1.535 |
| Final fixed-lag global SLAM | 1.182 | 1.462 |
| Full-history iSAM2 diagnostic (not online output) | 0.990 | 1.467 |

The run used 22,001 synchronized evaluation samples, traveled 3,299.999 m,
and completed normally. This is not a 72.3 km full-route result.

## Critical alias transition

At 2.835 km (`t=1890 s`) a second basin appeared. The old implementation
inserted factors from the formerly dominant but wrong basin. The new filter
held graph factors while ambiguous:

| Time (s) | Leading cohort | Posterior | Odds over runner-up | Decision |
| ---: | ---: | ---: | ---: | --- |
| 1890 | 1 | 0.881 | 7.50 | hold |
| 1920 | 1 | 0.858 | 6.13 | hold |
| 1950 | 1 | 0.732 | 2.77 | hold |
| 1980 | 2 | 0.629 | 1.72 | mode switch, hold |
| 2010 | 2 | 0.957 | 24.29 | hold: probability below production 0.97 |
| 2040 | 2 | 0.9967 | 577.00 | confirm current rank-3 observation |
| 2160 | 2 | 0.9989 | 890.47 | old cohort pruned |

The final tracker recorded two cohorts created, one posterior mode switch, one
old cohort pruned, 73 joint raw-likelihood updates, 20 resampling events, 63
factor authorizations, and 10 factor holds.

## Independent integration validation

```bash
PYTHONPATH=isaacsim \
  /home/cgz/.conda/envs/isaacsim-6.0/bin/python \
  standalone/validate_global_relocalization_chain.py \
  --output /tmp/oceansim_trn_global_relocalization_chain_20260913.json
```

This test injects a 1,000 m / -800 m / 20 degree local-frame transform and
uses non-overlapping sonar windows from the full Monterey product. With the
validator's 0.95 posterior and 20:1 odds requirement, it remained unconfirmed
for six windows and confirmed on window seven at posterior 0.9564 and odds
21.95. The current raw measurement had full rank, the robust GTSAM factor was
retained, and graph horizontal/yaw error was 0.309 m / 0.0018 degree. All 16
integration checks passed. The production launcher remains stricter at 0.97
and 20:1.

Repository verification completed with `335 passed`; Python compilation,
launcher syntax checks, and `git diff --check` also passed. The sole pytest
warning is the pre-existing SciPy requirement notice for the installed NumPy
version.

## Scientific boundary

The Monterey bathymetry and backscatter are measured products, but simulation
and matching use different processing of the same source survey. Side-scan is
normalized reflectivity rather than calibrated target strength. These runs are
therefore deterministic algorithm-integration evidence and a same-survey upper
bound, not independent-chart field validation, hydrographic certification, or
a sea-trial accuracy claim. The next decisive validation is the complete
72.3 km run followed by an independent-survey prior test.
