# Pose-graph backend A/B protocol

This experiment changes only the planar pose-graph solver.  The sensor model,
random seed, local Python acoustic ESIKF, terrain/backscatter matchers, route,
initial state, rendering settings, and factor construction must remain
identical between runs.

## Fixed experiment inputs

- Source baseline: `7b5c1809d3587cacba841f8fe3ce5bcedab5dc0e`
- Random seed: `31`
- Initialization: simulator truth at `t=0` (evaluation only)
- Route: `data/monterey_canyon_full_5m/full_map_traversal_route.json`
- Requested route: `1.01` laps, approximately `72.347 km`
- Vehicle speed: `1.5 m/s`
- Integrated sonar: V2 standard profile at `10 Hz`
- Local estimator: OceanSim Python 15-state acoustic ESIKF
- Terrain patch: 300 pings, acoustic-predictive motion compensation
- Prior-map modes: low-pass bathymetry plus same-survey backscatter
- UI: overview camera, 20x requested time scale, 2 simulated Hz display cap

## Run A: baseline

- Backend: NumPy batch planar pose graph
- Live output: `/tmp/oceansim_7b5c180_full_route_truth_ui.json`
- Durable final metrics: `numpy_batch_full_route_seed31.json`
- Durable position errors: `numpy_batch_full_route_seed31_position_errors.csv`

## Run B: candidate

- Backend: GTSAM iSAM2 with `IncrementalFixedLagSmoother`
- Live output: `/tmp/oceansim_gtsam_fixed_lag_full_route_truth_ui.json`
- Durable final metrics: `gtsam_fixed_lag_full_route_seed31.json`
- Durable position errors: `gtsam_fixed_lag_full_route_seed31_position_errors.csv`
- Durable factor stream: `gtsam_fixed_lag_full_route_seed31_pose_graph_factors.json`

The online candidate trajectory is produced by GTSAM's
`IncrementalFixedLagSmoother`.  Poses leaving the 180 s window retain their
last online estimate.  A separate complete-history iSAM2 instance is recorded
as `gtsam_full_history_isam2_diagnostic`; it supports old-node constraints and
diagnosis but is never substituted for the online fixed-lag trajectory.  When
a newly accepted factor crosses the 180 s boundary, its solved latest-pose
effect is transferred as one rigid `map->odom` anchor over the live window so
confirmed long-baseline relocalization is not discarded.

The candidate must be selected explicitly at launch.  The NumPy backend stays
available as the reference implementation.  No native FAST-LIVO2 front end or
other stashed experimental code is part of this A/B.

## Required comparison

Record and compare, using the synchronized ping timestamps without trajectory
alignment:

- dead-reckoning, local, online-global, and final-global position RMSE;
- final, median, p95, p99, and maximum position error;
- 100 m, 1 km, and 10 km translation RPE;
- roll, pitch, yaw, and geodesic orientation error;
- error statistics for every 5 km route segment;
- accepted/rejected geometry, intensity, terrain, backscatter, loop, and
  relocalization factors;
- terrain-factor outages and normal/recovery/emergency invocations;
- optimizer update count, active fixed-lag variables, marginalized variables,
  wall time, and achieved simulation time scale;
- non-finite states, discontinuities, and maximum online `map->odom`
  correction change.

If the GTSAM run disagrees with the batch solution, first replay an identical
recorded factor stream through both solvers.  Fix factor-frame conventions,
incremental synchronization, marginalization, or correction interpolation as
indicated by that replay before repeating the full route.

## Acceptance gates

- Both runs must report `completed` and the requested 1.01 laps.
- Synchronized timestamp arrays and every fixed sensor/front-end/matcher input
  must agree; dead-reckoning statistics are an additional determinism control.
- The candidate must report nonzero iSAM2 and fixed-lag updates, live and
  marginalized variables, finite trajectory metrics, and zero non-finite
  `map->odom` samples.
- Candidate online RMSE/P95/max/final error is compared with baseline online
  output.  The complete-history iSAM2 diagnostic is compared with the NumPy
  final batch output, including 100 m, 1 km, and 10 km RPE.
- A regression beyond the documented relative-plus-absolute allowances in
  `standalone/compare_pose_graph_backend_runs.py` fails the experiment and
  triggers factor-stream replay and correction before another full run.

## Prepared implementation

- Isolated branch: `backend-ab-gtsam-fixed-lag`
- Current prepared revision: `858c6f9`
- Full unit suite: 221 passed (one pre-existing SciPy/NumPy compatibility
  warning)
- The main worktree remains at `7b5c180` until Run A finishes.
