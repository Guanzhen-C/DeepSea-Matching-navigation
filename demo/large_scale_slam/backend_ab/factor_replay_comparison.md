# Pose-graph factor replay comparison

- Verdict: `pass`
- Scope: solver comparison on the same final factor snapshot; this is not an exact replay of online factor arrival or in-place factor replacement timing

## Full-window solver equivalence

| Quantity | Value |
|---|---:|
| Translation RMSE | 0.00175721849 m |
| Maximum translation difference | 0.00890787053 m |
| Yaw RMSE | 0.000530581968 deg |
| Maximum yaw difference | 0.00852091923 deg |
| NumPy final cost | 1907.11118 |
| GTSAM full-window final cost | 1907.11626 |

## Checks

- `factor_counts_identical`: `true`
- `all_replays_finite`: `true`
- `full_window_translation_max_le_0p02m`: `true`
- `full_window_yaw_max_le_0p02deg`: `true`
- `full_window_cost_equivalent`: `true`
- `full_window_retains_all_variables`: `true`
- `fixed_lag_actually_marginalizes`: `true`
