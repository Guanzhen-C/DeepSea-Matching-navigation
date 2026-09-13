# 5 Hz causal acoustic front-end regression (2026-09-13)

## Scope

This is a deterministic, UI-free regression of the local OceanSim
FAST-LIVO2-inspired acoustic front end. It is not a Monterey full-route result
and is not a sea-trial accuracy claim. The simulator uses one shared
bathymetry/backscatter acquisition clock and the same seed, route, sensor rate,
and 128-point registration budget in the controlled front-end A/B.

## Controlled 60 s A/B

Both runs use seed 31, 5 Hz acoustics, 50 Hz IMU propagation, the 180 s survey
period, 300 physical pings, and a 283.079849 m truth path. Continuous
observability weighting is enabled in both. Direct intensity produced no
accepted local update in this short leading-edge segment, so the measured
difference is attributable to the geometry front-end path rather than a
photometric correction.

| Metric | One-ping source | Eight-ping causal sliding source | Change |
| --- | ---: | ---: | ---: |
| Runtime (s) | 7.450499 | 6.688606 | -10.23% |
| Local RMSE (m) | 0.648418 | 0.301609 | -53.49% |
| Local P95 (m) | 0.913099 | 0.421779 | -53.81% |
| Local final error (m) | 1.030076 | 0.488803 | -52.55% |
| Dead-reckoning RMSE (m) | 0.334272 | 0.334272 | identical |
| Dead-reckoning final error (m) | 0.573615 | 0.573615 | identical |

The sliding implementation updates once per new physical ping after startup.
It caps points uniformly over the complete source window, commits only the
oldest aligned ping after convergence, and scales each overlapping likelihood
by 1/8. Thus the result does not come from counting each reused return eight
times as independent evidence.

Commands:

```bash
PYTHONPATH=isaacsim /home/cgz/.conda/envs/isaacsim-6.0/bin/python \
  standalone/large_scale_underwater_slam.py \
  --duration 60 --seed 31 --acoustic-hz 5 --slam-max-points 128 \
  --mbes-group-pings 1 --local-observability-mode continuous \
  --no-plot --output /tmp/oceansim_frontend_5hz_group1

PYTHONPATH=isaacsim /home/cgz/.conda/envs/isaacsim-6.0/bin/python \
  standalone/large_scale_underwater_slam.py \
  --duration 60 --seed 31 --acoustic-hz 5 --slam-max-points 128 \
  --mbes-group-pings 8 --enable-mbes-sliding-window \
  --local-observability-mode continuous \
  --no-plot --output /tmp/oceansim_frontend_5hz_sliding8
```

## 360 s local drift check

The same 5 Hz sliding configuration completed the two-lap 1,699.984150 m
route with 1,800 physical pings:

| Metric | Dead reckoning | Local acoustic ESIKF | Compact global graph |
| --- | ---: | ---: | ---: |
| RMSE (m) | 0.838436 | 0.540586 | 0.782338 |
| P95 (m) | 1.593364 | 0.844167 | 1.195952 |
| Final error (m) | 0.619693 | 0.473429 | 1.086782 |

The local result is the relevant check here: its RMSE is 35.52% lower than
dead reckoning and remains bounded over two laps. The compact synthetic-scene
pose graph is worse than the local result and is not used as evidence for the
full-map mature terrain/GTSAM/cohort-particle back end.

## Throughput boundary

On the same host, a 10 Hz / 15-ping run with the same approximate 3.3 m source
aperture took 11.768555 s, versus 6.688606 s for 5 Hz / eight pings. This is a
43.16% end-to-end runtime reduction while physical acoustic events fall from
600 to 300. Cross-frequency estimator accuracy is not compared because the
different acoustic call count changes the simulator RNG consumption order.

The repository test suite completed with 343 passing tests. A live Isaac Sim
short/full-map regression is intentionally deferred until the already-running
UI process finishes; this report does not claim its result in advance.

## Approximately 5 km continuous run

A subsequent uninterrupted run used the same seed-31, 5 Hz, eight-ping
sliding configuration for 1,058.833 simulated seconds. Discrete 50 Hz truth
arc integration measured 4,985.041316 m (0.30% below the nominal 5 km target),
with 5,294 physically shared bathymetry/backscatter pings. It completed in
397.499167 wall seconds while the existing Isaac Sim UI process continued in
parallel.

| Metric | Dead reckoning | Local acoustic ESIKF | Compact global graph |
| --- | ---: | ---: | ---: |
| RMSE (m) | 8.166607 | 0.584979 | 1.016225 |
| Mean (m) | 6.186178 | 0.553683 | 0.928103 |
| P95 (m) | 14.977955 | 0.908096 | 1.405980 |
| Maximum (m) | 20.014818 | 1.031455 | 2.452235 |
| Final (m) | 11.757151 | 0.541309 | 1.012910 |

The local front end reduces RMSE by 92.84% relative to dead reckoning and
keeps every sampled position error below 1.032 m. It accepted 5,120 geometry
updates and rejected 153. The compact graph accumulated 430 loop edges and
210 direct-intensity factors, but its RMSE is 0.431 m worse than the local
front end. Consequently the script's aggregate quality flag is false only for
its strict `global_rmse <= 1.0 m` check (missed by 0.016225 m); all local,
P95, finite-state, map, and loop checks pass. This result supports the new
local front end but does not validate the distinct Monterey mature
terrain/GTSAM/cohort-particle back end.

Artifacts from this run are stored outside the repository at
`/tmp/oceansim_frontend_5km_5hz/` for inspection during this session.
