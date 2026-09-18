# Changelog

## 改进版：确认后发布的全图重定位（2026-09-18）

- 全图重定位候选继续由 cohort particle filter 跟踪，但在通过独立证据、
  后验概率、优势比和协方差检查前仅作为影子状态，不能改变在线
  `map -> odom` 输出；未确认期间回退到普通多模态粒子 MAP 或 GTSAM。
- 在相同 73.070 km 路线、相同 243,569 个 5 Hz 样本和相同随机种子下，
  与 `stable-long-range-slam-20260917` 严格 A/B 对比：Online Global RMSE
  从 18.55 m 降至 6.36 m，P95 从 14.00 m 降至 11.71 m，最大误差从
  459.95 m 降至 53.85 m；原 59--60.5 km 风险区间峰值从 459.95 m
  降至 14.78 m，最终误差保持 4.35 m。DR、Local、固定滞后图优化和
  全历史 iSAM2 诊断逐样本不变，说明差异来自在线发布仲裁，而非随机
  波动或前端变化。
- 本版保留科学边界：31.77 km 仍出现 53.85 m 的短时 Online Global
  峰值，后续应独立分析；本次结果是确定性仿真验证，不是海试精度声明。

- Consolidated the large-area SLAM demo into one production estimator. Local
  is now always the co-registered MBES/side-scan feature frontend with no
  direct point-to-self-map or same-echo intensity state write. Global is now
  always the cohort particle terrain tracker, native 3-D BBS verification,
  GTSAM incremental fixed-lag graph, and bounded `map -> odom` correction. The
  validated multimodal output rule is fixed: only independently confirmed
  whole-map recovery cohorts take priority, unconfirmed whole-map candidates
  remain shadow states, multiple ordinary terrain modes temporarily publish
  the particle MAP, and a single stable mode publishes GTSAM. Removed the
  NumPy batch graph, Direct/Hybrid frontend switches, graph-only/always-map
  output policies, A/B launchers, factor replay/comparison programs, and the
  extra wrapper. The sole production entry is
  `standalone/run_v2_full_map_slam_isaac.sh`. Historical entries below describe
  experiments that led to this selection; their retired switches and launchers
  are no longer present.
- Added an optional CUDA compute backend for the feature Local frontend. The
  256 prediction-centred RANSAC hypotheses are evaluated in one PyTorch CUDA
  batch in float64; the selected consensus, observability test, covariance,
  and filter update retain the CPU implementation. Canonical shared-return
  raster construction was also vectorized without changing pixel-to-XYZ
  ownership. A 60 m matched-seed replay produced bit-identical sampled
  trajectories with zero fallbacks, reduced mean RANSAC time from 54.25 to
  6.42 ms, and reduced the complete replay from 24.78 to 15.29 s. Explicit
  `cuda` fails if unavailable; `auto` records and permits a CPU fallback.
- Added an experimental water-adapted MBES–SSS feature-hybrid Local frontend.
  Motion-seeded pyramid KLT follows GFTT side-scan features across the eight
  simultaneous V2 rows and lifts tracks back to their co-registered XYZ
  returns. The estimator preserves the propagated full SE(3) motion and fuses
  only the physically supported along-track scalar; a state projector prevents
  the side-scan observation from directly changing cross-track position,
  attitude, or depth. In feature mode, direct point-to-self-built-map MBES
  state writes are now opt-in: the returns remain in feature lifting, mapping,
  graph, and Global, but a 1 km ablation showed that their Local correction
  eventually reinforced strip-map deformation. Added a deterministic replay
  A/B runner, a dedicated Isaac launcher, and an opt-in world-fixed unresolved-
  scattering model for the 0.15 m sensor/5 m public-raster resolution gap. The
  model is explicitly reported as synthetic and zero preserves the measured
  map. Seven 60 m cross-region/seed runs reduced mean RMSE from 0.403 m DR and
  0.439 m direct MBES to 0.356 m, winning all seven RMSE comparisons. The 1 km
  run reduced RMSE from 4.390/5.374 m to 3.788 m; retaining both Local state
  writes produced 4.773 m. These are simulation regressions, not full-route or
  sea-trial claims.
- Replaced the formal front end's overlapping eight-physical-ping source
  window with one current integrated ping per 5 Hz geometry update. The
  current ping is matched exactly once against an older 30 s fixed-lag surfel
  submap and inserted only after convergence. Per-voxel removable sufficient
  statistics now use voxel-local coordinates, making repeated retirement
  stable on UTM maps. Adjacent 1.2 m footprints retain every residual but
  receive continuous information weighting from their approximately 25% new
  coverage. The production Local correction is position-only; single-ping
  MBES does not write yaw/velocity/IMU biases, while multi-ping Global
  bathymetry/backscatter matching continues to estimate absolute x/y/yaw.
  Same-ping side-scan alignment remains available to the graph without a
  second Local-state write. No new residual or truth-dependent gate was added.
- Changed both V2 Isaac Sim presets to one physically shared 5 Hz MBES +
  side-scan acquisition clock. At 1.5 m/s, adjacent 1.2 m footprints advance
  0.30 m and overlap by 0.90 m. The 250 m full-map preset explicitly allows
  two distinguishable pings in water. Its ping-count parameters were rescaled
  to preserve physical support: an eight-ping geometry window spans about
  3.3 m, four pings preserve the 0.8 s direct-intensity reference delay, and
  150/600 pings preserve the 30/120 s terrain-matching windows.
- Added a causal sliding acoustic submap to the FAST-LIVO2-inspired front end.
  After an independent startup map, every physical ping produces an ESIKF
  update and only the oldest aligned ping is committed. Uniform point capping
  covers the complete window, and the likelihood is scaled by the reciprocal
  window length so overlapping windows do not manufacture information.
- Replaced whole-frame acoustic degeneracy scaling with continuous,
  direction-wise spectral observability weighting. Added co-registered
  intensity/surface fitting and continuous relief-consistency covariance,
  fixed DVL rotation/lever-arm support, and per-echo pose-time deskew inputs.
  The filter deliberately remains a 15-state rotation/position/velocity/bias
  ESIKF with fixed gravity and no online DVL-scale state.
- Vectorized point-to-plane residual/Jacobian/noise construction and exact
  batch insertion for surfel and co-registered intensity maps. Reused one
  factorization for simultaneous intensity/surface plane fitting and limited
  Warp map uploads to the populated staging prefix. These changes reduce
  Python and host/device overhead without lowering any sensor or estimator
  update rate.
- Tuned the one-ping V2 acoustic FAST-LIVO2 front end without adding acceptance
  thresholds. The 10 Hz geometry and direct-intensity updates remain per
  physical ping, while the intensity reference excludes the latest eight pings
  (0.8 s / about 1.2 m, one complete standard-mode strip), its correlated
  return weight is capped at 32
  independent-equivalent samples, and one local intensity-map anchor persists
  for 120 s. Every ping remains a 10 Hz update and every converged strip enters
  the causal local map. In the deterministic truth-initialized 71.4 m
  comparison this reduced local RMSE from 0.754 m to 0.345 m and final error
  from 1.055 m to 0.383 m. Geometry groups of two/four pings, 3 m geometry
  voxels, 5 m intensity
  voxels, and a 0.60 m geometry noise floor were tested and rejected because
  they did not improve the paired run.
- Corrected the V2 integrated-sonar acquisition model to follow the manual's
  one-ping semantics. One ping now creates one per-echo range/XYZ/backscatter
  grid containing eight simultaneous along-track rows per side; the published
  bathymetry cloud is a deterministic subset of that grid and the 16-row
  side-scan image is binned from the same measurements. Runtime invariants and
  tests reject independent range, hit, point, or intensity arrays masquerading
  as a shared ping. The V2 SLAM path consumes one identically decimated
  `XYZ + intensity` array for geometry and direct intensity, and updates once
  per physical ping instead of recombining eight already multi-row pings.
- Raised the whole-map integrated bathymetric side-scan schedule to 10 Hz. The
  250 m model now records an explicit four-distinguishable-pings-in-water
  scheduling assumption instead of silently violating the 0.333 s two-way
  travel time. At 1.5 m/s adjacent 1.2 m footprints overlap by 1.05 m. All
  eight simultaneous rows in each acquisition form the direct-intensity strip,
  enabling a 10 Hz FAST-LIVO2-style acoustic update without a second temporal
  grouping stage.
- Added document-derived standard and high-resolution profiles for
  `多波束测深侧扫作业及成像V2.docx`. Both retain eight along-track image rows per
  side and ping, 2 cm range sampling, 2.5 cm documented range resolution, a
  0.14 deg horizontal opening, and the stated 250 m maximum. Standard mode
  records 0.15 m row spacing, 1.2 m coverage, and constant 0.15 m resolution
  through 60 m; high mode records 0.075 m spacing, 0.6 m coverage, and constant
  0.075 m resolution through 30 m. Beyond those breakpoints both use
  `R * sin(0.14 deg)`. The resulting raw contract is 16 beams by 12,500 samples
  per ping, while the historical one-row API remains available.
- Preserved all quantities absent from V2 as explicit unknowns, including the
  instrument identity, acoustic frequency, vertical opening alpha, ping rate,
  pulse/bandwidth/source level, calibrated intensity unit, accuracy, exact
  eight-beam steering table, and installation calibration. The live V2 route's
  1500 m/s sound speed, 2.5 Hz ping rate, 170 deg ray fan, 1.5 m/s speed, 28 m
  altitude, and 0.10/0.00/-0.15 m identity-orientation mounting are labelled
  simulation assumptions rather than device specifications.
- Extended the integrated bathymetric side-scan sensor with backward-compatible
  per-ray origins. The V2 model uses eight parallel, cell-centred virtual
  footprint strips at the documented 0.15/0.075 m spatial interval; it no
  longer misuses the 0.14 deg beam opening as row-centre steering. One
  timestamped Warp launch contains only the common quadrature return grid;
  96 published bathymetry samples are selected from it rather than launched as
  a second fan. Every direct intensity sample therefore has the same measured
  range and explicit XYZ coordinate. The virtual origins describe image
  sampling because the hardware phase-centre table is unavailable; they are
  not a claimed transducer layout. The rolling waterfall is used only for display and
  descriptors. Standalone MBES and SSS models remain available for genuinely
  separate instruments.
- Added the now-retired `standalone/run_v2_large_area_slam_isaac.sh` UI
  demonstration launcher.
  Its complete-map route covers an approximately 2.42 x 1.81 km envelope and
  7.94 km per lap and is validated against bathymetry, measured-backscatter,
  and 250 m NoData-clearance requirements before simulation starts. Ping-rate
  validation also enforces two-way travel time and gap-free along-track
  coverage for the selected profile.
- Replaced accelerated-frame modulo sampling with an absolute multi-rate event
  clock. IMU, DVL, pressure depth, and sonar now remain exactly at 50, 5, 10,
  and 2.5 Hz without duplicate measurements on fractional render-frame tails.
  The V2 launcher requests 4x display time and reports the requested and
  achieved scale separately; the kinematic navigation clock is explicitly not
  presented as the Isaac physics timeline.
- Reworked direct side-scan fusion from per-line pseudo-footprints into
  true multi-row per-ping strips, retaining optional motion-compensated temporal
  grouping only for genuinely single-row devices. Added
  causal reference separation,
  overlap, normalized-correlation, inlier, planar-rank, and correction gates;
  direct intensity now updates only x/y/world-yaw. Intensity map cells retain
  their pose-graph anchor, per-ping gain/offset nuisance directions are
  robustly projected out using the same IRLS metric as the pose Hessian,
  and graph-export-eligible accepted matches create separate
  anchor-to-current pose factors
  rather than being counted again as odometry constraints. This is original
  acoustic matching code, not FAST-LIVO2 source or a camera photometric model.
  A verified global loop can now seed a non-committing old-submap intensity
  refinement, allowing metre-scale local drift to be handled without jumping
  the live inertial state. Dense correlated rays are capped at 96
  independent-equivalent samples; graph factors use 1 m/2 deg uncertainty
  floors and a rank-aware 99% transactional pre-fit compatibility gate.
- Added non-destructive local association snapshots backed by a coarse
  spatial block index. The complete historical surfel map is retained for
  revisits while only the current sensor-reach window is packed for Warp.
- Added descriptor-first global relocalization outside the odometric search
  radius, followed by MBES registration, consecutive-match confirmation, and
  consistent pose-graph branch re-seeding.
- Added an exact Warp point-to-surfel data-association backend for the acoustic
  ESIKF. It preserves the CPU map's ordered 27-voxel candidates and only moves
  match indices across the device boundary. The live scene now uses up to 384
  motion-compensated MBES points per update, selects CPU/GPU association by
  workload, reports timings, and has CPU/Warp estimator-equivalence tests.
- Added a Warp CUDA backend for the acoustic sensors. It builds one BVH for
  static measured bathymetry and fuses closest-hit queries, normals, spatial
  reflectivity, and return-power evaluation. Added explicit PhysX fallback,
  backend/timing UI and metrics, and numerical cross-validation.
- Replaced the live SLAM demo's primitive AUV and synthetic 300 x 220 m
  seabed with a full-scale LSTS/OceanScan LAUV and the complete, un-cropped
  public USGS Monterey Canyon 5 m mosaic: 5,842 x 3,534 samples over
  29.205 x 17.665 km. It retains 11,296,546 measured bathymetry cells and
  11,145,229 measured backscatter cells (98.6605% coverage), with missing data
  left as `NaN`. Products A/B/C/D are independently percentile-normalized and
  are measured spatial texture, not absolute-calibrated target strength.
  Added full-footprint stride-8 display and stride-2 Warp acoustic meshes,
  terrain-following motion, chase/overview cameras, and data provenance. The
  source scene remains a 5 m raster (10 m acoustic vertices at default stride
  2), so 2 cm output bins cannot be interpreted as independent centimetre-scale
  terrain information or as validation of 2.5 cm sensor resolving power.
- Removed the temporary AHRS/compass assumption. The estimator now consumes
  only IMU, DVL, pressure depth, MBES, and side-scan data, with explicit
  yaw/x/y gauge handling and sensor-only startup initialization.
- Added a FAST-LIVO2-inspired, independently implemented direct acoustic
  intensity update with view-aware, temporally separated map support, plus
  full-state grouped-MBES updates and physically consistent continuous-time
  IMU noise densities.

## [0.2.0] - 2026-08-()

### Added

- Isaac Sim 5.0 compatibility
- Better ROS2 tutorial and demo
- Standalone scripts for our sea urchin detection ([paper](TODO))
- Geometric multibeam echosounder with per-beam ranges, points, intensity, and ROS 2 PointCloud2 output
- Dual-channel side-scan sonar with acoustic shadows, configurable noise, waterfall output, and ROS 2 mono8 images
- Live MBES swath and side-scan waterfall windows, with 3D USD beam/return visualization
- Presentation-ready offline acoustic SLAM dashboard PNG/GIF generator
- FAST-LIVO2-inspired, independently implemented underwater ESIKF with IMU,
  DVL, depth, grouped MBES point-to-surfel updates, acoustic uncertainty, and
  degeneracy handling
- Side-scan place descriptors, geometrically verified loop closures, robust
  pose-graph optimization, and corrected global trajectory/map output
- Deterministic 300 x 220 m, 1.70 km two-lap validation and a matching live
  Isaac Sim scene with truth/local/global tracks

## [0.1.0] - 2025-01-08

### Added

- Initial version of OceanSim Extension
