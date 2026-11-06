"""Explicit synthetic AUV truth/sensors on a measured-source local scene.

No GPS. Programmed truth is only available in the simulator/evaluator; sensor
objects are stripped to sensor frame before entering the frozen estimator.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np


class VirtualAUVTrajectory:
    def __init__(self, length_m=250., speed_mps=.8):
        if not 0 < length_m <= 270 or speed_mps <= 0:
            raise ValueError('Explicit bounded small-region virtual trajectory required.')
        self.length_m = float(length_m)
        self.speed_mps = float(speed_mps)
        self.duration_s = self.length_m/self.speed_mps

    def position(self, t):
        t = float(t)
        return np.array([-self.length_m/2+self.speed_mps*t,
                         30.+2.*np.sin(2*np.pi*t/180.),
                         -8.+.3*np.sin(2*np.pi*t/95.)])

    def rotation(self, t):
        from oceansim.slam.math3d import rotation_from_rpy
        yv = 2*(2*np.pi/180.)*np.cos(2*np.pi*t/180.)
        return rotation_from_rpy(np.deg2rad(1.)*np.sin(2*np.pi*t/55.),
                                 np.deg2rad(1.2)*np.sin(2*np.pi*t/70.),
                                 np.arctan2(yv, self.speed_mps))

    def state(self, t):
        from oceansim.slam.large_scale_sim import GroundTruthState
        from oceansim.slam.math3d import so3_log
        t = float(t)
        v = np.array([self.speed_mps, 2*(2*np.pi/180.)*np.cos(2*np.pi*t/180.),
                      .3*(2*np.pi/95.)*np.cos(2*np.pi*t/95.)])
        a = np.array([0., -2*(2*np.pi/180.)**2*np.sin(2*np.pi*t/180.),
                      -.3*(2*np.pi/95.)**2*np.sin(2*np.pi*t/95.)])
        r = self.rotation(t)
        h = .001
        omega = so3_log(self.rotation(t-h).T@self.rotation(t+h))/(2*h)
        return GroundTruthState(t, r, self.position(t), v, a, omega,
                                r.T@(a+np.array([0., 0., 9.81])))


def measured_field_sensor_class():
    """Keep original geometry/noise/lineage; do not filter/colour measurements."""
    from oceansim.slam.large_scale_sim import LargeScaleSensorSimulator
    from oceansim.utils.acoustic_sonar import compose_sidescan_line

    class MeasuredFieldSensor(LargeScaleSensorSimulator):
        def _acoustic_reflectivity(self, x, y):
            return self.world.reflectivity(x, y, measured_only=True)

        def bathymetric_sidescan_ping(self, truth):
            base = super().bathymetric_sidescan_ping(truth)
            # These maps store already-observed envelope proxy, not an
            # independently calibrated scattering coefficient. Do not apply
            # another incidence/attenuation law to the same measured values.
            def observed(points, hit):
                values = np.zeros(len(points), float)
                values[hit] = self.world.reflectivity(points[hit, 0], points[hit, 1], measured_only=True)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Measured scalar appearance outside common support.')
                return values

            imaging = replace(base.imaging, intensities=observed(base.imaging.points_world, base.imaging.hit_mask))
            bathymetry = replace(base.bathymetry,
                intensities=observed(base.bathymetry.points_world, base.bathymetry.hit_mask))
            port, starboard, _, _ = self._registered_sidescan_lines(
                imaging.points_sensor, imaging.intensities, imaging.hit_mask,
                imaging.beam_directions, imaging.ray_origins_sensor)
            side = replace(base.side_scan, port=port, starboard=starboard,
                display_line=np.rint(np.clip(compose_sidescan_line(port, starboard), 0, 1)*255).astype(np.uint8))
            result = replace(base, imaging=imaging, bathymetry=bathymetry, side_scan=side)
            self.last_integrated_ping = result
            return result

    return MeasuredFieldSensor


def deterministic_schedule(duration_s, ping_period_s):
    """All 50Hz IMU, 5Hz DVL, 10Hz pressure events; no sensor decimation."""
    schedule = []
    for kind, period, priority in [('imu', .02, 0), ('dvl', .2, 1), ('depth', .1, 2), ('sonar', ping_period_s, 3)]:
        first = 0 if kind == 'sonar' else 1
        for k in range(first, int(math.floor(duration_s/period))+1):
            schedule.append((round(k*period, 9), priority, kind))
    return [(t, kind) for t, _, kind in sorted(schedule)]
