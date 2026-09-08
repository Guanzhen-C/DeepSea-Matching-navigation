# Changelog

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
- Added `standalone/run_v2_large_area_slam_isaac.sh` for the UI demonstration.
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
