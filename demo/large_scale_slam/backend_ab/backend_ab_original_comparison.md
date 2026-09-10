# Pose-graph backend A/B comparison

- Structural verdict: `pass`
- Baseline: `/home/cgz/OceanSim/demo/large_scale_slam/backend_ab/numpy_batch_full_route_seed31.json`
- Candidate: `/home/cgz/OceanSim/demo/large_scale_slam/backend_ab/gtsam_fixed_lag_full_route_seed31.json`
- Timestamp streams identical: `True`
- Fixed-input mismatches: 0
- Accuracy verdict: `fail`

## Position error

| Track | Metric | Baseline | Candidate | Delta |
|---|---|---:|---:|---:|
| dead_reckoning_imu_dvl_depth | rmse_3d_m | 5123.5 | 5123.5 | +0 |
| dead_reckoning_imu_dvl_depth | median_3d_m | 4546.77 | 4546.77 | +0 |
| dead_reckoning_imu_dvl_depth | p95_3d_m | 8167.5 | 8167.5 | +0 |
| dead_reckoning_imu_dvl_depth | p99_3d_m | 8301.99 | 8301.99 | +0 |
| dead_reckoning_imu_dvl_depth | max_3d_m | 8358.19 | 8358.19 | +0 |
| dead_reckoning_imu_dvl_depth | final_3d_m | 8151.68 | 8151.68 | +0 |
| local_acoustic_esikf | rmse_3d_m | 8430.78 | 8430.78 | +0 |
| local_acoustic_esikf | median_3d_m | 6668.07 | 6668.07 | +0 |
| local_acoustic_esikf | p95_3d_m | 15636.5 | 15636.5 | +0 |
| local_acoustic_esikf | p99_3d_m | 17237.4 | 17237.4 | +0 |
| local_acoustic_esikf | max_3d_m | 17564.9 | 17564.9 | +0 |
| local_acoustic_esikf | final_3d_m | 12398.7 | 12398.7 | +0 |
| online_global_slam | rmse_3d_m | 394.812 | 471.421 | +76.609 |
| online_global_slam | median_3d_m | 17.1251 | 21.4594 | +4.33424 |
| online_global_slam | p95_3d_m | 938.887 | 1187.17 | +248.285 |
| online_global_slam | p99_3d_m | 1045.12 | 1260.67 | +215.551 |
| online_global_slam | max_3d_m | 1056.93 | 1270.97 | +214.043 |
| online_global_slam | final_3d_m | 505.591 | 732.533 | +226.942 |
| final_optimized_global_slam | rmse_3d_m | 377.764 | 473.447 | +95.6828 |
| final_optimized_global_slam | median_3d_m | 10.6642 | 18.8506 | +8.18639 |
| final_optimized_global_slam | p95_3d_m | 873.233 | 1198.9 | +325.668 |
| final_optimized_global_slam | p99_3d_m | 981.183 | 1268.22 | +287.033 |
| final_optimized_global_slam | max_3d_m | 988.712 | 1277.4 | +288.693 |
| final_optimized_global_slam | final_3d_m | 506.447 | 741.486 | +235.04 |

## Legacy world-frame displacement-vector error

This historical metric is not invariant to a constant global yaw gauge.

| Track | Interval | Baseline (m) | Candidate (m) | Delta (m) |
|---|---|---:|---:|---:|
| dead_reckoning_imu_dvl_depth | 100m | 30.0255 | 30.0255 | +0 |
| dead_reckoning_imu_dvl_depth | 1000m | 274.542 | 274.542 | +0 |
| dead_reckoning_imu_dvl_depth | 10000m | 2116.45 | 2116.45 | +0 |
| local_acoustic_esikf | 100m | 85.2335 | 85.2335 | +0 |
| local_acoustic_esikf | 1000m | 830.368 | 830.368 | +0 |
| local_acoustic_esikf | 10000m | 6210.68 | 6210.68 | +0 |
| online_global_slam | 100m | 12.3106 | 12.6678 | +0.357178 |
| online_global_slam | 1000m | 99.1846 | 103.244 | +4.05898 |
| online_global_slam | 10000m | 500.261 | 578.784 | +78.5232 |
| final_optimized_global_slam | 100m | 10.4683 | 12.21 | +1.74167 |
| final_optimized_global_slam | 1000m | 98.6949 | 106.029 | +7.33449 |
| final_optimized_global_slam | 10000m | 504.919 | 586.822 | +81.9034 |

## Gauge-invariant local odometry (candidate factor stream)

The older stream lacks raw local posteriors; these numbers exclude unrecoverable local intensity steps and are not the complete local trajectory.

| Baseline | Translation RMSE (m) | Yaw RMSE (deg) |
|---|---:|---:|
| Pre-image odometry-factor ablation: one keyframe | 0.493441 | 0.321274 |
| 100m | 1.6172 | 0.687914 |
| 1000m | 30.9029 | 2.89727 |
| 10000m | 1114.16 | 15.6902 |

## Optimizer and runtime

```json
{
  "baseline_backend": "numpy_batch_preinstrumentation",
  "candidate_backend": "gtsam_isam2_incremental_fixed_lag",
  "baseline_optimization_runs": 645,
  "candidate_optimization_runs": 643,
  "candidate_isam_updates": 643,
  "candidate_fixed_lag_updates": 4205,
  "candidate_fixed_lag_active_variables": 8,
  "candidate_fixed_lag_marginalized_variables": 1773,
  "candidate_global_only_factors": 0,
  "candidate_global_anchor_bridge_updates": 0,
  "candidate_global_anchor_bridge_max_translation_m": 0.0,
  "candidate_global_anchor_bridge_max_abs_yaw_deg": 0.0,
  "wall_time_s": {
    "baseline": 57914.95208274701,
    "candidate": 58087.62737508491,
    "delta": 172.67529233789537,
    "change_percent": 0.29815321627337704
  },
  "achieved_time_scale": {
    "baseline": 0.8411241814722948,
    "candidate": 0.8386237976503025,
    "delta": -0.002500383821992358,
    "change_percent": -0.2972669050621886
  }
}
```
