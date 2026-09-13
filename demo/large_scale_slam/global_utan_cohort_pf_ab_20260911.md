# Global-UTAN-inspired cohort particle-filter A/B check

Date: 2026-09-11

## Scope

This is a controlled 2 km simulation comparison beginning 40.5 km into the
72.347 km Monterey survey route. Both runs use seed 31, exact initial truth,
the same 10 Hz co-registered bathymetric side-scan stream, IMU/DVL/depth
measurements, local acoustic ESIKF, GTSAM fixed-lag smoother and bounded
`map -> odom` output correction. The candidate alone enables the ordinary
cohort particle backend and its direct raw-sonar likelihood factors.

The candidate is inspired by the public Global UTAN cohort architecture; it
is not the authors' implementation. OceanSim retains its own bathymetry/
backscatter likelihood and native 3D-BBS verification chain.

## 2 km result

| 3-D position metric | Direct-factor control | Cohort/raw-likelihood candidate | Reduction |
|---|---:|---:|---:|
| Online global RMSE | 6.791 m | 0.755 m | 88.89% |
| Online global P95 | 12.608 m | 1.574 m | 87.51% |
| Online global maximum | 13.338 m | 2.104 m | 84.22% |
| Online global final | 13.332 m | 1.714 m | 87.15% |
| Final optimized RMSE | 6.182 m | 0.493 m | 92.02% |

The dead-reckoning and local acoustic ESIKF columns in the two CSV files are
byte-for-byte equal. In both runs, dead-reckoning RMSE/final error are
46.547/141.042 m and local RMSE/final error are 2.204/6.767 m. The candidate
therefore reduces online RMSE by 65.76% relative to the unchanged local
front end; the gain is not caused by different simulated sensor noise or
front-end tuning.

The control repeatedly entered a wrong terrain basin: its online error was
7.740 m at 1.0 km, 7.755 m at 1.5 km and 11.241 m at 1.8 km. The candidate
reported 0.762, 0.700 and 0.670 m at the same checkpoints. It accepted 42
terrain factors, including 29 windows for which the compact matcher supplied
no usable Gaussian proposal. The control accepted 16 factors and invoked
recovery five times; the candidate required no recovery or emergency reset.

Candidate `map -> odom` output remained continuous: maximum sampled
translation/yaw changes were 0.734 m and 0.0053 degrees, with no changes over
5 m or 1 degree. Its online yaw RMSE was 0.379 degrees versus 0.552 degrees
for the local front end.

## Why the candidate remains statistically separated

1. Each persistent cohort carries 256 weighted SE(2) particles and is
   propagated only by local acoustic-inertial odometry. The explicit null
   hypothesis can retain probability when a sonar window is uninformative.
2. Raw MBES Student-t height likelihood and detrended side-scan correlation
   are evaluated directly at every cohort. The co-registered modalities use
   a conservative logarithmic opinion pool rather than being counted as
   independent sensors.
3. A proposal refined from a propagated cohort retains that cohort's ID.
   A distant whole-map compact proposal may form another cohort, but it cannot
   overwrite the persistent cohort merely through covariance-based nearest
   association.
4. The cohort selects only a data-association basin. Inside that basin, a
   deterministic local search and finite-difference Laplace approximation use
   the current independent sonar window alone to form the GTSAM measurement.
   The recursive PF mean/covariance are never reused as a unary factor.
5. Curvature is projected into its positive-semidefinite observable subspace,
   model covariance floors are applied only in that subspace, and exact
   measurement information is handed to the graph separately from the finite
   reporting covariance. A flat, textureless unit-test map produces exactly
   zero information rather than a false pose constraint.

## Verification

- Full Python suite: `328 passed`, with one pre-existing SciPy/NumPy version
  warning.
- Focused particle/likelihood tests: `39 passed`.
- 160 m integration smoke: local/global/final-optimized RMSE
  `0.501/0.493/0.493 m`.
- 600 m integration: local/global/final-optimized RMSE
  `0.684/0.459/0.267 m`; four of eleven accepted factors were raw-only.
- 2 km candidate and control both terminated at 1999.950 m with 13,334
  synchronized 10 Hz evaluation samples.

## Artifacts

- 2 km candidate:
  `/tmp/oceansim_pf_raw_laplace_candidate_2000m_v2_20260911.json`
- 2 km candidate error CSV:
  `/tmp/oceansim_pf_raw_laplace_candidate_2000m_v2_20260911_position_errors.csv`
- 2 km candidate factor stream:
  `/tmp/oceansim_pf_raw_laplace_candidate_2000m_v2_20260911_pose_graph_factors.json`
- 2 km direct-factor control:
  `/tmp/oceansim_pf_raw_laplace_control_2000m_20260911.json`
- 600 m candidate:
  `/tmp/oceansim_pf_raw_laplace_candidate_600m_20260911.json`

## Scientific boundary

This is a single-seed, exact-initial-state, 2 km controlled simulation, not a
full-route, multi-seed or sea-trial accuracy claim. The bathymetric prior is a
low-pass version of the same Monterey survey family used to construct the
simulation terrain, and the side-scan observation/reference share the same
public backscatter mosaic. That same-survey coupling is an inverse-crime risk
and makes the result an optimistic upper bound.

The current sonar is ray-cast/phenomenological: it models geometry,
incidence, attenuation, noise and normalized reflectivity, but not emitted
waveforms, sound-speed refraction, multipath, speckle statistics or calibrated
target strength. Hardware data from an independent survey, timestamp/
extrinsic/sound-speed calibration, multi-seed evaluation and the complete
72.347 km route remain required before an operational claim.
