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

### 后续验证次序与限制

保留现有失败证据，完成数值审查及板端实时容量检查后，再验证实时
Isaac 原始传感器链路、5 km/关键转弯，最后串行执行湖底 47.116 km 和
海底约 73.070 km。全程数据应包含无损输入、实际在线返回位姿、时间戳、
延迟和资源监测；不得事后对齐轨迹或用 PC 位姿替换板端输出。
当前真实地形和已有传感器模型的科学边界仍沿用前文，部署并不使其
自动成为真实声学传播或实船硬件试验。
