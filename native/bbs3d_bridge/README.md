# OceanSim 3D-BBS bridge

This directory is a small C ABI adapter around the public GPU implementation
of [KOKIAOKI/3d_bbs](https://github.com/KOKIAOKI/3d_bbs). The build is pinned
to commit `41529a34a2fb9618b5ff560fb3c2363f1615666d`; it does not track an
unreviewed upstream branch at runtime.

Build from the repository root:

```bash
standalone/build_bbs3d_bridge.sh
```

The script checks out the exact upstream revision under the ignored
`downloads/3d_bbs_upstream` directory, builds its GPU library, and creates
the ignored `native/bbs3d_bridge/build/liboceansim_bbs3d.so`. No system-wide
installation is performed. `BBS3DCandidateVerifier` loads that library by
default; `OCEANSIM_BBS3D_LIBRARY` can point to another explicitly managed
build.

## Scientific role

The upstream solver is used unchanged for branch-and-bound occupancy scoring.
OceanSim constrains its search for the underwater sensor state:

- IMU roll and pitch pre-rotate the source MBES strip.
- The candidate yaw is pre-applied; 3D-BBS searches only a bounded residual
  yaw and horizontal translation.
- Pressure depth is embedded in source `z`; vertical translation is fixed to
  zero so a wrong horizontal basin cannot be rescued by moving the seabed up
  or down.
- The returned score only verifies a whole-map candidate. It is not inserted
  as a factor until metric MBES refinement, independent side-scan refinement,
  and persistent non-overlapping-window confirmation also succeed.

This is therefore an application-specific adapter, not a claim that the
3D-BBS paper evaluated multibeam sonar or this complete OceanSim pipeline.

## Licensing and attribution

3D-BBS is copyright Koki Aoki and distributed under the MIT License in its
upstream repository. The checkout retains the upstream `LICENSE` file. The
OceanSim bridge code remains covered by OceanSim's repository license.

Please cite the upstream work when using this path:

> K. Aoki, K. Koide, S. Oishi, M. Yokozuka, A. Banno, and J. Meguro,
> “3D-BBS: Global Localization for 3D Point Cloud Scan Matching Using
> Branch-and-Bound Algorithm,” ICRA 2024.
