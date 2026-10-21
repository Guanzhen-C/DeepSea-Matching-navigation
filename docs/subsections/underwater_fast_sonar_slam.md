# FastSonar-LIVO2 underwater SLAM

OceanSim now includes a runnable acoustic-inertial SLAM stack for an AUV with
only an IMU, bottom-track DVL, pressure depth, multibeam echosounder (MBES),
and co-registered side-scan backscatter from one integrated sonar. Its
high-rate error-state filter and sequential geometry/appearance structure are
inspired by FAST-LIVO2, but the implementation is original OceanSim acoustic
SLAM code: it is neither the FAST-LIVO2 source code nor its camera photometric
model. It is also not FAST-LIO2 and does not require a camera, LiDAR,
compass, AHRS heading, GNSS, or simulated ground-truth pose measurements.

The implementation is original BSD-3-Clause OceanSim code.  Public GPL
projects were studied as architectural references; their source code was not
copied into this repository.

## Architecture

The local front end in `isaacsim/oceansim/slam/` contains:

1. A 15-state error-state iterated Kalman filter with rotation, position,
   velocity, gyroscope bias, and accelerometer bias.
2. High-rate IMU propagation plus robust bottom-track DVL velocity and
   positive-down pressure-depth updates.
3. A sensor-only startup window: gravity aligns roll/pitch, DVL initializes
   velocity, and pressure initializes depth. Absolute yaw and horizontal
   translation are explicitly fixed as local-frame gauge freedoms, not
   treated as measurements.
4. Each 5 Hz integrated V2 ping retains its eight simultaneous along-track
   rows and the exact MBES XYZ return that generated every backscatter pixel.
   These rows are one physical acquisition, not eight temporal pings. Four
   distinct consecutive pings form one causal feature-odometry subframe.
5. IMU/DVL propagation supplies the complete SE(3) prediction for pyramid KLT
   tracking between canonical side-scan images. Forward/backward-consistent
   GFTT tracks are lifted through the co-registered MBES XYZ grid. CUDA evaluates
   256 prediction-centred geometric hypotheses in one batch; the winning
   consensus is refined in float64. Because the short aperture does not
   independently observe smooth-terrain cross-track translation or yaw, only
   the observable along-track scalar is fused. A state projector prevents this
   measurement from directly changing cross-track position, attitude, depth,
   velocity, or IMU biases. The retired direct point-to-self-map MBES state
   update is not present.
6. MBES still supplies the causal 30 s surfel map, descriptors, graph geometry,
   feature lifting, and Global terrain likelihood. Expired ping batches are
   subtracted exactly from voxel-local sufficient statistics, avoiding
   cancellation at UTM coordinates. Co-registered side-scan intensity may
   align against an anchored appearance surface and export a conservatively
   weighted GTSAM factor, but this same-echo alignment is non-committing and
   never contracts Local a second time. Gain/offset nuisance terms are
   projected out; surface disagreement continuously inflates variance; causal
   separation, coverage, correlation, inlier, rank, view and two-sided-support
   checks protect factor export. This is acoustic map matching, not a camera
   photometric residual.
7. Fixed IMU-to-sonar and IMU-to-DVL calibration interfaces. DVL prediction
   includes the rotational lever-arm velocity at the DVL phase centre. Each
   co-registered echo can also carry a pose-sampling time offset and is
   deskewed with the current angular rate, translation velocity, and sonar
   lever arm. The current ray-cast simulator is a synchronous snapshot and
   therefore supplies zero offsets. Two-way acoustic travel time is not
   silently reinterpreted as a pose timestamp; a real driver must provide the
   instrument's actual transmit/receive timing convention.
8. Conservative innovation/correction limits and recoverable DVL weighting so
   an acoustic correction cannot permanently lock out velocity observations.

The global back end adds:

1. Fixed-interval acoustic keyframes.
2. Combined MBES bathymetry/backscatter and robust rolling side-scan
   waterfall descriptors for repeat-place candidates. A strict descriptor
   search over all historical keyframes can propose a place even outside the
   drifting odometric search radius.
3. Ordered-beam registration followed by trimmed ICP geometric verification.
4. Consecutive-match confirmation, uncertainty-weighted loop edges, global
   relocalization branch seeding, robust x/y/yaw pose-graph optimization, and
   a separately corrected global map and trajectory. An accepted direct
   intensity match that also passes graph-export validation adds a separate
   anchor-to-current pose factor
   from its measurement-only normal equation; it is not folded into or counted
   again as an odometry edge. Factor uncertainty is floored at 1 m and 2 deg
   for the 5 m source raster/3 m online voxels, partly covering correlation
   with the self-built reference submap, and a rank-aware 99% pre-fit
   normalized-residual gate rejects a graph-inconsistent factor before it can
   create or modify any node. This is a compatibility test, not a claim that
   the compact graph computes full endpoint-marginal NIS. Roll, pitch, and
   depth remain owned by the ESIKF.
   After a geometrically verified loop has corrected the global branch, that
   graph pose may seed a non-committing intensity re-alignment in an old anchor
   frame. A passing match refines the global graph only; it never jumps the
   live local inertial state and still has to pass every radiometric gate.

This split is intentional: loop corrections do not jump the live inertial
state or invalidate its local map.

## Deterministic large-area validation

Run the estimator regression suite in the Isaac Sim environment:

```bash
PYTHONPATH=isaacsim \
  /home/cgz/.conda/envs/isaacsim-6.0/bin/python -m pytest -q
```

The deterministic whole-map retrieval, cohort confirmation, native 3-D BBS,
GTSAM factor insertion and output correction check is:

```bash
standalone/build_bbs3d_bridge.sh
PYTHONPATH=isaacsim \
  /home/cgz/.conda/envs/isaacsim-6.0/bin/python \
  standalone/validate_global_relocalization_chain.py
```

Run the measured-map route headlessly with
`standalone/run_v2_full_map_slam_isaac.sh --headless`.

## Live Isaac Sim UI

Isaac Sim 6.0.1 is installed in
`/home/cgz/.conda/envs/isaacsim-6.0`.  Start the live scene with:

```bash
standalone/run_v2_full_map_slam_isaac.sh
```

The default live UI loads the **complete, geographically un-cropped** USGS
Monterey Canyon 5 m raster mosaic. Its 5,842 x 3,534 grid spans 29.205 x
17.665 km between sample centres. It contains 11,296,546 measured bathymetry
cells; source NoData remains `NaN`, so preprocessing does not bridge or invent
seafloor in unmapped areas. The display mesh uses the complete footprint at
stride 8 by default, while the static Warp acoustic mesh uses stride 2. These
are level-of-detail choices, not geographic crops. Strict mesh construction
also preserves native NoData holes instead of spanning them with coarse
triangles.

The paired texture is a mosaic of the public USGS backscatter products A, B,
C, and D. It provides 11,145,229 measured cells, or 98.6605% of the valid
bathymetry; the remaining 151,317 bathymetry cells have `NaN` backscatter.
Because the four products use heterogeneous native encodings, instruments,
and frequencies, each layer is independently clipped and linearly normalized
between its 1st and 99th percentiles before equal-weight averaging in overlap
cells. The result retains measured spatial texture and per-cell A/B/C/D source
lineage, but it is **not absolute-calibrated target strength**. Complete bounds,
source URLs, archive and array checksums, normalization details, and masks are
recorded in `data/monterey_canyon_full_5m/metadata.json`.

The vehicle is the full-scale 1.08 m Light Autonomous Underwater Vehicle
(LAUV) developed by LSTS at the University of Porto and OceanScan-MST.  Its
upstream Apache-2.0 Collada body, control fins, propeller, texture, and license
are stored under `assets/vehicles/lauv/`. The live trajectory maintains a
28 m terrain clearance over a low-pass navigation copy of the DEM. By default,
one integrated V2 ping casts one common grid of eight simultaneous rows per
side and 256 quadrature returns per row: 4,096 raw returns in one backend call.
Each raw return contains measured range/XYZ and backscatter. The 96-sample
published bathymetry cloud is selected from this grid rather than cast as an
independent fan, while the identically decimated raw `XYZ + intensity` array
feeds both local geometry and direct acoustic matching. The port/starboard range-bin
waterfall is retained for display and place descriptors only; it is not
back-projected into invented 3-D points. `--sonar-backend physx` retains the
general collision-scene fallback.

The generic matcher can optionally update the ESIKF, but the formal full-map
preset runs it non-committing because intensity and geometry come from the
same echo. Only when its freshly re-associated measurement-only optimum also
passes the graph-export checks does it produce an explicitly anchored x/y/yaw
pose-graph factor with its own conservative information matrix. The exported
factor is derived after projecting out gain/offset, rather than from the
filter's posterior covariance, and remains separate from the same interval's
pre-image inertial/DVL/depth edge. A non-committing graph-seeded match is counted as
accepted only after that graph factor also passes the transactional
compatibility gate. This architecture is implemented and its component gates
are unit-tested, but
it is not evidence by itself that measured backscatter improves localization.
Use `--disable-intensity-updates` and identical seeds for a controlled A/B run;
rolling descriptors and geometrically verified loop closure remain active in
that mode.

### Archived pre-correction direct-intensity ablation (2026-09-04)

The following results predate the one-raw-grid/per-ping correction and must not
be used as validation of the current acquisition model. A fresh matched-seed
ablation is required before making a localization-accuracy claim.

The current dense, anchored implementation was run for the complete 360 s
two-lap numerical survey with three fixed random seeds. `ON` enables the
direct intensity ESIKF update and anchored graph factors; `OFF` disables those
updates but deliberately retains rolling side-scan descriptors, so this table
isolates direct fusion rather than the total value of side-scan for proposing
loop candidates.

| Seed | Local RMSE ON/OFF (m) | Global RMSE ON/OFF (m) | Accepted updates | Intensity factors |
| --- | ---: | ---: | ---: | ---: |
| 21 | 0.260258 / 0.276660 | 0.223512 / 0.236169 | 107 | 25 |
| 31 | 0.386127 / 0.386298 | 0.287776 / 0.272710 | 106 | 33 |
| 41 | 0.353901 / 0.341703 | 0.292648 / 0.287315 | 106 | 29 |
| Mean | 0.333429 / 0.334887 | 0.267979 / 0.265398 | 106.3 | 29.0 |

Across these three seeds, direct intensity improves mean local RMSE by only
0.44% and worsens mean optimized global RMSE by 0.97%. The signs also vary by
seed. With only three deterministic trials this is not a confidence interval,
but it is sufficient to reject a claim of a large or robust gain. The result
is expected when geometry and loop closure are already strong, adjacent dense
rays share the same low-resolution texture, and most accepted image updates
apply only centimetre-scale corrections to a self-built map.

The measured mosaic itself is not featureless: over 11,145,229 valid cells its
normalized standard deviation is 0.1925 and the median 5 m neighbour
difference is 0.0512. The limitation is independence and scale. Valid-cell
correlation remains approximately 0.76--0.77 at a 10 m lag, while the default
stride-2 acoustic mesh samples the native 5 m raster at 10 m vertices. Thus a
512-ray-per-side fan heavily oversamples interpolated, correlated texture; it
does not create hundreds of independent sub-metre localization features.

The complete-map 1.1-lap run provides a second, harder paired ablation. Both
runs use the same seed, 8,561 pings, complete 29.205 x 17.665 km scene, and 54
MBES-verified loop edges; `OFF` again retains side-scan descriptors.

| Complete-map metric | Direct intensity ON | Direct intensity OFF | ON change |
| --- | ---: | ---: | ---: |
| Dead-reckoning RMSE (m) | 28.223693 | 28.223693 | identical |
| Local acoustic ESIKF RMSE (m) | 7.552456 | 7.637383 | -1.11% |
| Local final error (m) | 2.098617 | 3.823041 | -45.11% |
| Final optimized global RMSE (m) | 3.994212 | 3.969828 | +0.61% |
| Final optimized global final error (m) | 5.449938 | 4.530961 | +20.28% |

The ON run attempted 1,060 strips, accepted 77 (74 local and 3 graph-seeded),
and exported only 6 explicit graph factors; 962 attempts lacked enough old-map
correspondences. Of the 74 local acceptances, only 3 passed the stricter fresh
association needed for graph export, while 71 entered the odometry chain as a
local posterior. This supports a narrow conclusion: direct intensity helps the
revisited local tail in this run, but it does not yet improve the optimized
global trajectory. High mean accepted correlation (0.994) is internal
photometric consistency, not proof of geographic accuracy.

The live navigation model keeps simulated gyro bias within the
LSTS-published 1 degree/hour INS maximum. That data-sheet envelope is used as
a configurable bias bound; no AHRS/compass heading is generated or fused.

The UI shows a chase-camera 3-D scene, live MBES beams/returns, a live
MBES swath, a side-scan waterfall, truth/local/global tracks, optimized MBES
points, and a status panel.  The 360-second route repeats for loop closure.
Close Isaac Sim to stop it.  Use `--camera-mode overview` for a fixed view of
the complete survey instead of the default vehicle-following view.

The complete V2 full-map launcher uses a 20x render time scale and limits only
the operator sonar-window upload to 2 simulated Hz. Every 5 Hz integrated
sonar ping still updates the raw waterfall and enters the unchanged sensor and
SLAM event clocks. At 1.5 m/s, adjacent acquisitions advance 0.30 m and their
1.2 m footprints overlap by 0.90 m. Two distinguishable coded/interleaved
pings in water are an explicit scheduling assumption for the 250 m range and
0.333 s two-way travel time; waveform separation itself is not synthesized.
Four distinct physical pings form one feature-odometry subframe. Side-scan
alignment uses a four-ping causal reference separation (0.8 s / about 1.2 m),
32 independent-equivalent samples, and a 120 s stable intensity-map anchor,
but never writes Local a second time. The 150-ping ordinary and 600-ping
whole-map terrain windows preserve their 30 s / 120 s durations. No estimator
measurement is decimated to accelerate the UI:

```bash
standalone/run_v2_full_map_slam_isaac.sh
```

The status panel and shutdown metrics report the acoustic ray backend and
feature frontend independently. The formal launcher requests CUDA for the
batched feature consensus and fails explicitly if it is unavailable. Image
tracking, float64 final refinement, the 15-state filter update, map statistics,
terrain likelihood, and GTSAM remain their validated implementations; the
retired point-to-surfel Local state write and its association-backend selector
are not initialized.

Useful finite checks are:

```bash
standalone/run_v2_full_map_slam_isaac.sh \
  --headless --sonar-backend warp-cuda --verify-sonar-backend \
  --max-frames 40 --time-scale 4

standalone/run_v2_full_map_slam_isaac.sh \
  --duration 30 --time-scale 8
```

To recreate the complete bathymetry/backscatter mosaic from the official USGS
archives (the default paths are under `downloads/`):

```bash
PYTHONPATH=. python standalone/prepare_real_bathymetry.py --download
```

The source catalog and direct data download are published by the USGS at
<https://doi.org/10.5066/F7XD0ZQ4>.  The LAUV model is sourced from
<https://github.com/uuvsimulator/lauv_gazebo> at the revision recorded in
`assets/vehicles/lauv/SOURCE.md`.

Live metrics are written to
`demo/large_scale_slam/isaac_live_metrics.json` during shutdown.

![Live Isaac Sim large-area SLAM UI](../../demo/large_scale_slam/isaac_live_ui.png)

## Whole-map relocalization after a navigation outage

The normal terrain matcher is deliberately local. When repeated bounded
matches fail, the optional recovery path searches the complete measured map
and all yaw angles, but it does not turn the largest correlation peak directly
into a pose-graph factor. The recovery chain is:

1. Joint coarse bathymetry/backscatter retrieval returns several spatially
   separated basins. This is an OceanSim implementation inspired by the
   cohort-filtering architecture of Global UTAN, not that paper's reference
   code.
2. Every basin must pass the pinned official
   [3D-BBS](https://github.com/KOKIAOKI/3d_bbs) branch-and-bound occupancy
   verifier. IMU roll/pitch and pressure depth remain fixed, so only residual
   horizontal translation and yaw are searched. The coarse 3D-BBS argmax and
   the original appearance proposal are both retained as metric-refinement
   seeds; diagnostics, never truth, choose the better converged basin.
3. A 5 m MBES terrain fit and multi-scale side-scan cell correlation check the
   candidate. The side-scan context treatment follows the reliability idea of
   SC-Match, but is not an exact SC-Match implementation because OceanSim
   matches geocoded samples to a prior raster rather than matching two raw
   waterfalls.
4. Refined duplicates from the same acquisition are merged. Genuinely
   separated aliases become separate particle cohorts alongside an explicit
   `no valid fix` state. Each cohort contains 256 weighted SE(2) particles;
   the FAST-LIVO2-style acoustic/IMU/DVL/depth front end propagates every
   particle. All particles from all cohorts are evaluated in one raw-sonar
   point-mass update, so the best residual is common to the complete posterior
   instead of being renormalized separately inside each alias. Low effective
   sample size triggers systematic resampling followed by covariance-preserving
   regularization. A Bayesian change-point branch reserves proposal mass from
   the complete prior, allowing a newly retrieved candidate to replace an old
   but formerly confident false mode. Overlapping windows receive discounted
   weight, and three non-overlapping ping/time windows must support the same
   cohort before confirmation.
5. The production estimator publishes a particle pose only after the leading
   whole-map or ordinary terrain cohort passes the independent-evidence,
   posterior-probability, odds, covariance, and current-observation checks.
   Every unconfirmed mode remains a shadow tracker state, even when it is the
   particle MAP and currently outranks the explicit null hypothesis. GTSAM
   supplies the online output during that ambiguity. This prevents repetitive
   or weak terrain from moving `map -> odom` before the estimator has enough
   independent evidence, while preserving every cohort for continued tracking
   and later confirmation. This is the only published-output policy.

The same particle policy also sits in front of ordinary bounded terrain
factors. Persistent proposals retain
their originating cohort ID, so a broad whole-map alias cannot steal or
overwrite an established mode through covariance-normalized association. A
single accepted bathymetry/side-scan match only initializes/tracks a cohort;
three non-overlapping observations must support the same propagated mode
before GTSAM authorization.

The graph never receives the recursive particle posterior as if it were a new
sensor measurement. The formal hybrid output can publish the current MAP mode
during ambiguity/recovery, but this provisional output cannot authorize a
graph factor.
For every independent 300-ping window, the confirmed
cohort selects only a local basin. A deterministic search and finite-
difference Laplace approximation then derive the pose and information matrix
from that current raw MBES/side-scan likelihood alone. This also supplies
ordinary factors when the compact matcher has no usable Gaussian proposal.
The curvature is projected onto its positive-semidefinite observable subspace,
and chart/model covariance floors are applied only inside that subspace; flat,
textureless directions therefore contribute zero information rather than a
false precise constraint. A separate finite reporting covariance assigns broad
variance to null-space directions; its inverse is never substituted for the
exact rank-aware graph information.

This posterior follows operational terrain-relative-navigation practice used
by MB-System/MBARI. Its public `libtrnav` tree includes both
`TNavPointMassFilter` and `TNavParticleFilter`: odometry supplies the motion
update, raw multibeam soundings are compared directly with a prior bathymetric
map, and the posterior is monitored for convergence. In OceanSim, the
Fast-LIVO2-style acoustic/IMU/DVL/depth front end supplies that odometry and
co-registered side-scan adds a conservative appearance likelihood. The
implementation is independent OceanSim code. MB-System is GPL-3.0, so no
MB-System source was copied into this BSD-3-Clause project.

Build the native verifier and run the deterministic, UI-free integration
check with:

```bash
standalone/build_bbs3d_bridge.sh
PYTHONPATH=isaacsim \
  /home/cgz/.conda/envs/isaacsim-6.0/bin/python \
  standalone/validate_global_relocalization_chain.py
```

That check samples a sequence of disjoint strips from the complete Monterey
product and stops only after the configured posterior/odds criteria confirm a
cohort. It injects a kilometre-class rigid local-frame error and validates
whole-map retrieval, native 3D-BBS, both metric refinements, persistent
confirmation, the robust graph factor, and continuous output correction. The
validator uses 0.95 posterior and 20:1 odds; the production launcher retains
the stricter 0.97 posterior and 20:1 odds. Truth synthesizes
the sonar window and ideal roll/pitch/depth inputs; map-frame horizontal
position and yaw remain hidden from the matchers, filter, and graph, and are
inspected only afterward to calculate error. Bathymetry is matched against a
separately low-pass navigation surface, but side-scan observations and the
prior still come from the same public backscatter survey. It is therefore a
deterministic integration and wiring upper bound, not an independent
chart-to-field or sea-trial accuracy result.

Run the complete route without a UI through the sole production launcher:

```bash
standalone/run_v2_full_map_slam_isaac.sh --headless
```

It preserves the archived seed-31 sensor, route, local front end, and fixed-lag
GTSAM settings while enabling complete-yaw search, persistent cohorts, the
native verifier, robust relocalization factors, and output-only soft
correction. The 72 km run is intentionally separate from the short check
because it takes hours and is the evidence required for a full-route accuracy
claim.

The controlled 2 km same-seed A/B result and its scientific limitations
are recorded in
[`demo/large_scale_slam/global_utan_cohort_pf_ab_20260911.md`](../../demo/large_scale_slam/global_utan_cohort_pf_ab_20260911.md).
The joint-likelihood, rank-safe 3.3 km critical regression is recorded in
[`demo/large_scale_slam/mature_trn_fastlivo2_regression_20260913.md`](../../demo/large_scale_slam/mature_trn_fastlivo2_regression_20260913.md).
The matched-seed 5 Hz causal acoustic front-end regression is recorded in
[`demo/large_scale_slam/fast_sonar_frontend_5hz_regression_20260913.md`](../../demo/large_scale_slam/fast_sonar_frontend_5hz_regression_20260913.md).

## Production MBES–SSS feature Local frontend

The co-registered MBES–SSS feature observation is the only Local acoustic
frontend. It does not treat one
eight-row ping as a terrestrial camera frame. GFTT corners are followed between
consecutive physical pings by pyramid KLT initialized from the complete
IMU/DVL SE(3) motion. Forward/backward-consistent tracks are lifted to the exact
MBES XYZ returns that generated their pixels, and deterministic consensus
estimates an error about the propagated pose while preserving roll, pitch,
pressure depth, and the sonar lever arm.

The V2 aperture robustly observes row displacement but does not independently
observe cross-track translation or yaw on a smooth seabed. Consequently the
accepted side-scan factor enters the ESIKF as one scalar projection along the
anchor sensor x axis. A state-space projector prevents incidental covariance
from directly changing cross-track position, attitude, or depth. MBES remains
active in the feature lift, local map, graph, and Global matcher, but there is
no direct local-map ESIKF state write: the kilometre ablation showed that this
self-map correction eventually outweighed its short-term benefit. DVL, IMU,
and the pressure sensor retain their normal responsibilities. A continuous
Cauchy consistency response reduces periodic-
texture aliases without a hard state-innovation gate. Four distinct physical
pings form one factor; no echo is counted as two simultaneous pings.

`--feature-compute-backend cuda` evaluates all 256 prediction-centred RANSAC
hypotheses as one float64 PyTorch CUDA batch. The final all-inlier refinement,
information-eigenvalue observability check, covariance, and ESIKF write remain
unchanged on CPU. GFTT and bidirectional KLT intentionally remain in OpenCV:
on the 32 x 1024 canonical image they take about 0.4 and 4 ms per ping, so a
host/device round trip would cost more than it saves. Explicit `cuda` fails
loudly when unavailable, while `auto` records and permits a CPU fallback.
Diagnostics and output metrics include the requested/active backend, device,
GPU/CPU call counts, fallback count, and per-stage timing.

The USGS backscatter supplied with the complete map is registered at 5 m,
whereas the documented standard-mode row interval is 0.15 m. The optional
`--subgrid-reflectivity-amplitude` therefore adds a deterministic world-fixed,
band-limited unresolved-scattering field for sensor-resolution experiments.
Zero preserves the measured mosaic exactly. A nonzero value retains the USGS
data as its low-frequency envelope but is explicitly synthetic below the map
cell size; metrics record that provenance. This option changes intensity only,
never terrain, ray geometry, range, or the MBES/side-scan shared-return
lineage.

Run the estimator with `standalone/run_v2_full_map_slam_isaac.sh`.
The cross-region tests, negative control, exact numbers, and limitations are
recorded in
[`feature_hybrid_frontend_validation_20260915.md`](../../demo/large_scale_slam/feature_hybrid_frontend_validation_20260915.md).
The archived 1 km replay records the selection experiment; its removed A/B
runner is not a production entry. These simulation improvements are not
presented as a full-route or sea-trial accuracy claim.

## Python sensor interface

The estimator has no Isaac Sim dependency, so the same object can consume a
ROS bag, hardware driver, or simulated sensors:

```python
from oceansim.slam import (
    FastSonarSLAM,
    gauge_anchored_initial_covariance,
    initialize_from_imu_dvl_depth,
)

slam = FastSonarSLAM()
initial = initialize_from_imu_dvl_depth(
    startup_specific_force,
    startup_dvl_velocity,
    startup_depth,
)
slam.initialize(
    t0,
    rotation=initial.rotation,
    position=initial.position,
    velocity=initial.velocity,
    covariance=gauge_anchored_initial_covariance(),
)

slam.process_imu(t, gyro_rad_s, specific_force_mps2)
slam.process_dvl(velocity_body_mps)
slam.process_depth(depth_positive_down_m)
# `ping` is the driver's BathymetricSideScanPing built from one acquisition.
slam.process_integrated_ping(ping, sample_stride=8)

R_local, p_local = slam.state.rotation, slam.state.position
R_global, p_global = slam.global_pose()
times, global_track = slam.global_trajectory()
points_global = slam.optimized_map_points()
```

Inputs must be synchronized and use metres, seconds, radians, a local
`+X forward, +Y port, +Z up` vehicle/sensor convention, and a Z-up navigation
frame. Fixed IMU-to-sonar and IMU-to-DVL rotation/translation calibrations can
be passed to the `FastSonarSLAM` constructor. The estimator state remains the
15 components `rotation, position, velocity, gyro bias, accelerometer bias`;
gravity is a fixed navigation-frame constant, and DVL scale is a fixed
externally calibrated quantity rather than an online state. Calibrate those
extrinsics, echo pose epochs, DVL scale, pressure offset, and sound speed before
real-data use. Since the
allowed sensors provide no absolute heading or horizontal position, compare
real trajectories only after fixing or aligning that gauge.

## Public references and scope

The design was checked against these public projects, specifications, and papers:

- [LSTS LAUV specification](https://lsts.fe.up.pt/index.php/vehicles/lauv):
  standard navigation/INS capabilities and the published 1 degree/hour
  maximum gyroscope bias.

- [FAST-LIVO2 repository](https://github.com/hku-mars/FAST-LIVO2) and
  [paper](https://arxiv.org/abs/2408.14035): sequential iterated Kalman
  updates for heterogeneous geometric and direct photometric observations.
- [Warp kernel guide](https://nvidia.github.io/warp/latest/user_guide/basics.html)
  and [mesh-query API](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.MeshQueryRay.html):
  typed SIMT kernels and closest-hit mesh outputs used by the two GPU paths.
- [RA-SIDO](https://doi.org/10.3390/jmse14161520): tightly coupled
  sonar/inertial/depth odometry with anisotropic sonar uncertainty and
  degeneracy handling; this is the closest published underwater analogue.
- [DISO](https://github.com/jiahao-Ma/DISO): direct acoustic intensity,
  DVL/IMU/depth fusion, and acoustic odometry design choices.
- [MINS](https://www.sciencedirect.com/science/article/pii/S2468013325000579): multibeam/inertial
  navigation and bathymetric matching.
- [Dense side-scan acoustic SLAM](https://github.com/halajun/acoustic_slam):
  side-scan place constraints and mapping.
- [3D-BBS repository](https://github.com/KOKIAOKI/3d_bbs) and
  [paper](https://arxiv.org/abs/2310.10023): the exact pinned native
  branch-and-bound implementation used only for candidate verification.
- [Global UTAN](https://doi.org/10.1109/TIM.2026.3716492): appearance
  retrieval, geometric verification, and persistent cohort reasoning used as
  an architectural reference for whole-map recovery.
- [MBARI Terrain Relative Navigation](https://www.mbari.org/project/terrain-relative-navigation/)
  and [MB-System libtrnav](https://github.com/dwcaress/MB-System/tree/master/src/mbtrnav): operational
  multibeam-to-prior-map navigation and the PMF/PF architecture used as the
  mature terrain-navigation reference. OceanSim does not copy the GPL-3.0
  implementation.
- [SC-Match](https://arxiv.org/abs/2607.25937): multi-scale side-scan context
  consistency used as a design reference, not copied reference code.

This is a research and simulation implementation, not a claim of
hydrographic-grade navigation.  The current global graph assumes a
roll/pitch-stabilised survey AUV and optimizes x/y/yaw.  A platform performing
large 6-DoF maneuvers needs a full SE(3) graph. The timestamp/deskew interface
does not itself simulate propagation physics. Sound-speed refraction,
multipath, magnetic disturbances, time-varying DVL scale, pressure bias,
clock offset, and online extrinsic calibration are not yet modeled.  Validate
all noise, latency, calibration, and observability settings on the target
vehicle before field deployment.

## Jetson 处理器在环部署：阶段性证据（2026-10-21）

以下是仿真传感器驱动实际 Jetson AGX Orin 的导航处理器在环验证，
不是实船湖试/海试，也不是带执行机构的控制器闭环。完整两航程尚未完成。
本节不以短程精度、CUDA 单测或同机回放代替全程验证。

### 冻结算法与数据边界

- 湖底基线：`lake-sim-baseline-20261021` / `204d3e2`，Chelan 地形。
- 海底基线：`sea-sim-baseline-20261021` / `8b5df53`，Monterey 地形。
- PC 独立保留仿真真值；板端仅收到受控初值及原始 IMU、DVL、深度和
  同源测深/侧扫数据。原始传感器记录无跳帧，声呐 5 Hz。
- 板端使用完整原前端、粒子地形跟踪、原生 3D-BBS、GTSAM 固定滞后及
  `map→odom` 链路。不是 NumPy 后端，也不是只回放地形粒子滤波。
- 两份冻结源和地图都校验哈希；部署实验位于独立 `hil/jetson-lake-sea-20261021` 分支。

### 已完成的 600 m 锁步输入验证

两场景各有 28,003 个输入事件和 2,001 个在线输出；实际 CUDA 前端无
CPU 回退，实际 GTSAM 后端运行。锁步保证每个输入消费完再发送下一个，
**不能用其墙钟速度宣称 1 倍速实时运行**。

| 场景 | PC Global RMSE / P95（m） | Jetson Global RMSE / P95（m） | 状态 |
| --- | --- | --- | --- |
| 湖底 600 m | 1.36554 / 1.73788 | 1.36098 / 1.73319 | 短程跨架构资格通过 |
| 海底 600 m，从原路线 30 km 新初始化 | 0.83121 / 1.35714 | 0.86543 / 1.50537 | P95 增加 10.922%，待审查 |

事前提出的 RMSE/P95 相对退化 10% 审查线没有放宽。流水线因此停在
`after_build_qualification_v2/pipeline_status.json` 的 `needs_review`，
没有自动重复运行或推进全航程。海底最大板端与 PC 输出位置差为 0.15810 m，
Local 最大差为 0.00345 m；相对百分比超线与绝对数值均应如实报告。

证据目录：`demo/hil/jetson_20261021/after_build_qualification_v2/`。
真值评分和图形仅在 PC 上进行，二维/三维图工具为
`standalone/plot_hil_results.py`；`--require-full-route` 拒绝以短程冒充全程。

### 数值与性能审查

`sea_portability_audit/` 中保留以下独立对照，不覆盖原测量：

1. 相同保存因子流在两端原 GTSAM 中重算，最终 XY 差约 `3.5e-12 m`，
   不能把原来的海底差异归咎于后端求解器。
2. PC 独立 Python 3.12 环境使用与板端相同的 NumPy 1.26.4，完整
   2,001 帧 DR/Local/Global 与原 PC NumPy 2.3.1 结果逐帧差为零。
   没有降级原电脑环境，也未借此更换评分基线。
3. 选取相同协方差和种子的原高斯采样差约 `7.1e-15 m`；该对照不能
   支持本批差异由采样基底符号翻转引起的猜测。
4. 板端 30 s 原始前缀性能剖析耗时 44.34 s，其中侧扫强度处理约
   21.4 s，固定锚邻域查询约 14.3 s，CUDA 共识约 2.76 s。剖析有额外
   开销，且故意截断，不是完整 HIL 接受结果。

具体前端/匹配数值分支的首次差异仍在排查；不能把上述排除性检查写成
“跨架构精度问题已解决”。另行启动的海底 600 m 1 倍速原始输入实验
标记为**性能诊断**，不撤销 `needs_review`。该实验已完整消费 28,003 个
事件、返回 2,001 帧，输出与锁步板端结果一致；但 400 s 传感器数据实际
耗时 583.22 s，最大结果到达滞后 183.15 s，故**没有达到实时运行要求**。
不能把模式名 `realtime` 或零丢帧写成实时性能通过。

独立候选 `standalone/hil_intensity_packing.py` 只批量装填重复邻域，
保持原邻域顺序、半径、锚、因果截止、平面方程和所有门限不变；未知
源方法哈希拒绝转换。它不会在导入时自动安装，也未接入正式 ROS 接收器。
湖底与海底同输入各完整 2,001 帧候选回放的位置和姿态差全部为零；仍需
板端完整一致性、性能和实时链路验证，不能据此宣称板端提速已验收。

第一候选的板端完整 600 m 离线对照现已结束：2,001 帧 DR/Local/Global
位置和姿态与原锁步板端输出的最大差全部为零，CUDA 回退为零，仍为
原生 GTSAM。400 s 输入耗时 466.96 s（0.8566 倍），**仍不足实时**。
该离线速度不能直接与原 ROS 测试相除，冒充同条件的算法提速百分比。

第二候选另将描述子均值/标准差按行归约，但保留原 OpenCV 重采样、
逐行范数、特征归属顺序和判定。海底 PC 完整回放 2,001 帧的位置/姿态
差为零；ARM 合成逐字段一致性测试通过。同一 30 s ARM 剖析前缀为
32.12 s（原 44.34 s），声呐处理中位 0.188 s。但剖析包含启动期特殊
开销，中位声呐时间也不含全部传感器/通信预算，故仍需完整实时测试。

ROS 接口新增显式 `--deployment-layout`，默认 `original`，另外可选
`packed` 或 `packed-and-feature-batch`。选择值写入启动和提取审计，
两份适配源码也进入 PC/板端一致性哈希检查。该参数仅用于部署候选
验证，不会自动启用“最快”模式或更改两份冻结算法/评分参考。

进一步的第 91 s 单对特征隔离实验找到一个可复现的跨平台来源：完全
相同输入产生相同 147 组 KLT 对应集合，但按光流误差排序的次序不同。
这使固定种子的均匀 RANSAC 具体抽样不同，PC/ARM 分别留下 61/60 个
内点，相对变换最大元素差 `0.008213 m`。仅在隔离实验中让 ARM 复用
同一对应顺序后，变换差降为零，协方差差约 `4.4e-23`。
这证明该帧的差异由对应排列影响有限随机抽样引起，不是 GTSAM 或
同输入 CUDA 几何求解器误差；尚不能据单帧证明全程尾部差异均已解释。
生产对应排序尚未修改，原 10% 审查失败仍保留。证据为
`sea_portability_audit/pair_order_causal_audit_91s.json`。

进一步加速的第二轮 ROS 600 m 已结束：仍为 28,003 个事件/2,001 个
输出、0 跳过，6 组位置/姿态相对原 ARM 输出逐帧零差异。耗时从同条件
原 ROS 的 583.22 s 降到 449.49 s，但最大结果滞后仍为 49.44 s，
**仍未达到实时容量要求**。不能以剖析中位数替代这项实测结论。

后续等价候选 `hil_intensity_linearization.py` 批量计算强度观测的两个
矩阵乘积，保留原逐行 yaw 点积和表面一致性方差算式，避免改变归约
次序。PC 原始 600 m 完整回放的 6 组位置/姿态差为零；两端合成输入
逐字段测试通过。强度邻域打包还只在单次调用内复用共享单元统计，不
跨地图更新保留缓存。这些候选仍需完整板端和网络性能检查。

`hil_stable_correspondences.py` 则是另行标记的数值移植变体：保留 KLT
选出的对应集合和质量判定，只在进入均匀 RANSAC 前按匹配像素位置
排序。它不改变门限或抽样分布，但改变固定种子的具体样本，故**不是
原轨迹的逐帧零差异加速**。PC 600 m 对照的 Global RMSE/P95 为
0.83785/1.45313 m，原 PC 为 0.83121/1.35714 m，分别增加约 0.8%/7.1%。
原参考和原失败记录保留，不将此新变体当作旧版参考来隐藏退化；完整
ARM 同变体、5 km/转弯与全程效果仍需验证。

组合候选的海底 ARM 600 m 离线回放随后完整结束：400 s 输入耗时
371.30 s，平均 1.077 倍；CUDA 无回退，仍为原生 GTSAM。Global RMSE/P95
为 0.80270/1.27746 m，相对**原 PC**分别降低 3.43%/5.87%，此次短程
原 10% 退化检查通过（并不撤销旧变体原来的失败记录）。
但该变体与同变体 PC 的最大 Local/Global 位置差仍为 0.00616/0.34979 m，
**跨架构并未逐帧完全一致**；单帧排序根因不是全部差异的证明。
完整网络实时、湖底、关键转弯及两全航程仍须分别验证。

`score_hil_variants.py` 在同一份主机真值上并列保留原 PC、原 ARM 和
各新变体，禁止以新结果替换 `original_PC`，不做轨迹对齐，且明确不
单凭位置误差统计声称生命周期、实时或全程验收完成。

组合海底 600 m 的网络输入试验现已结束：全部 28,003 个事件和
2,001 个在线输出完整，网络版与该 ARM 离线版的六通道位置/姿态逐帧
零差异，Global RMSE/P95 仍为 0.80270/1.27746 m。400 s 输入实际耗时
401.31 s，平均吞吐约 1 倍；输出延迟最大 4.12 s、P95 3.28 s，属于
仍有突发延迟的近实时处理，**不是硬实时或控制器闭环通过**。
此项使用已记录输入，不能冒充真实运行的 Isaac 源链路；后续须实际
接通 `--live-bundle` 的传感器生成、IPC、ROS 和板端处理全过程。

首个湖底 600 m 实际 Isaac 源链路随后完整完成：原生源、IPC/ROS、
板端均正常结束，28,003 个事件/2,001 个输出一致，明确记录
`live_Isaac_sensor_stream` / `live_ROS2_to_Jetson`。排除仅输出路径不同的
boot 元数据后，28,002 个事件与原生录入对照逐包相同；不是用录入文件
假装实时生成。ARM Global RMSE/P95 为 1.35751/1.71522 m，短程完整性
和相对原 PC 的精度检查通过。400 s 输入耗时 402.19 s，最大/P95 输出
延迟 4.45/3.12 s，仍是近实时处理，不是硬实时通过。

该次主机观测最低可用内存约 15.08 GiB，没有 OOM kill；源进程的内核
累计峰值触及 7 GiB 硬上限，并发生文件缓存回收，不能宣称资源完全
没有压力。部分 GPU 时间线在观测启动前，没有补造早期样本。
`observe_hil_host_resources.py` 只读取主机及任务 cgroup，不停止或调整
任何任务；后续启动器自动从启动期记录主机资源，板端仍记录独立的
tegrastats。首个 live 短程并不等于两个完整航程已通过。

### 后续验证次序与限制

#### 独立的短时相对运动诊断

`standalone/analyze_hil_relative_motion.py` 对**已经完整结束并通过输入完整性
核验的** HIL 记录计算 10、30、60 s 固定时间窗 SE(3) 相对位姿误差。
位置和姿态都使用各窗起始时刻的机体系，而不是直接相减两个绝对误差。
常量世界坐标系平移/旋转会抵消，随时间变化的漂移、尺度和航向误差不会。
不插值、不拟合、不将真值送回估计器，也不删除高误差窗口。

```bash
/home/cgz/.conda/envs/isaacsim-6.0/bin/python -m standalone.analyze_hil_relative_motion \
  --run demo/hil/jetson_20261021/lake_live_600m_canonical_v3 \
  --output /tmp/oceansim-lake-600m-rpe-new-report.json
```

输出文件必须尚不存在；该工具不接受用未结束记录代替完整试验。
湖底实际链路 600 m 记录中，Local 的 10/30/60 s 平移 RPE RMSE 为
0.157/0.377/0.636 m，DR 为 0.174/0.457/0.861 m。这说明**该短程输入**
中 Local 的短时平移增量优于 DR，并不能据此保证长程不漂移。
Global 也单独报告，但其短时变化包含地图校正，不能作为前端增量使用。
滑动窗口彼此重叠，不是独立重复试验，亦不单凭 RPE 宣布全程验收。

#### 尚待完成的长程证据

`standalone/run_hil_full_missions.py` 是主机侧有限串行执行器，不改变导航
计算。它先观察**已经运行的**海底 5 km pilot，完整结束后回收真实板端
日志，逐帧核对网络返回的六组位姿与板端日志完全相同，再做原 PC
参考的 RMSE/P95 审查、10/30/60 s RPE 和二维/三维图。沿用先前提出的
10% 相对退化审查规则，不改变估计器门限，也不以新变体替换原 PC。
真实 live 标记、所有输入计数、CUDA 无回退、原 GTSAM、无持续真值输入和
资源记录均要核验。单凭 systemd 已回收临时 unit 后显示的默认 success，
不能声称独立观察到了 exit 0；完成数据和失败记录也必须分别检查。

显式开启 `--run-full-missions` 后，它复核已完成的湖底 600 m，另外
先做湖底自己的 5 km live 验证，以检查较大底图的资源行为，再依次
运行湖底约 47.116 km 和海底约 73.070 km 的完整航程。每次启动仍经过
原启动器的代码/地图哈希、工作树、唯一任务、内存、磁盘和显存检查。
资格验证后若部署源码发生变化，会拒绝沿用旧验证启动新航程。

```bash
/home/cgz/.conda/envs/isaacsim-6.0/bin/python -m standalone.run_hil_full_missions \
  --pilot demo/hil/jetson_20261021/sea_live_5km_canonical_v3 \
  --output demo/hil/jetson_20261021/full_missions_pipeline_new_run \
  --run-full-missions
```

这是执行入口示例，不要在已有执行器运行时再次执行；输出目录必须是新的。
适合通过有内存限制的独立用户服务及单实例锁运行，不能依赖终端窗口。
任何检查失败只终止后续队列并写明原因，**不停止、重启、重试当前试验，
不放宽审查线，不删除失败数据**。SSH 暂时失败会重查相同任务；观察
期限耗尽也不会自动认定原任务已结束。图和阶段报告不代表最终整体验收，
更不代表物理传感器、实船试验或硬实时资格通过。

保留现有失败证据，完成数值审查及板端实时容量检查后，再验证实时
Isaac 原始传感器链路、5 km/关键转弯，最后串行执行湖底 47.116 km 和
海底约 73.070 km。全程数据应包含无损输入、实际在线返回位姿、时间戳、
延迟和资源监测；不得事后对齐轨迹或用 PC 位姿替换板端输出。
当前真实地形和已有传感器模型的科学边界仍沿用前文，部署并不使其
自动成为真实声学传播或实船硬件试验。
