"""Mapped SSS landmark revisits without a bathymetry or flat-bottom prior.

Two receptions of one mapped XY landmark carry range-change information.
For source heights za, zb and UNKNOWN landmark height Z,

  ||Lxy-ob||² - ||Lxy-oa||² - (rb²-ra²) = 2*(zb-za)*(Z-(za+zb)/2).

Z is NOT set to a DEM, a guessed flat bottom, or an estimated true height.
The sonar's physical range and below-sensor hemisphere only bound it. We
profile over that entire interval: any admissible Z gives ZERO residual.
At equal sensor depths Z cancels exactly. Nonzero depth changes enlarge the
dead zone and therefore REMOVE information, rather than inventing depth.

This is a new, conservative compatibility factor, not a reproduction of a
published SLAM system. The fixed map XY and data association still have
geocoding/association uncertainty. No truth, terrain, or altitude interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class MappedRangeChange:
    landmark_xy: np.ndarray
    origin_a: np.ndarray
    origin_b: np.ndarray
    sensor_z_a: np.ndarray
    sensor_z_b: np.ndarray
    range_a: np.ndarray
    range_b: np.ndarray
    standard_deviation_m: np.ndarray
    maximum_range_m: float = 250.
    pressure_std_m: float = .03
    independent_equivalent_features: int = 24
    information_budget: float = .5
    robust_delta: float = 3.

    def __post_init__(self):
        arrays = ('landmark_xy', 'origin_a', 'origin_b', 'sensor_z_a', 'sensor_z_b',
                  'range_a', 'range_b', 'standard_deviation_m')
        n = len(np.asarray(self.range_a))
        if n == 0:
            raise ValueError('At least one actually associated landmark is required')
        for name in arrays:
            value = np.array(getattr(self, name), dtype=float, copy=True)
            expected = (n, 2) if name in arrays[:3] else (n,)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError('Finite observed arrays with consistent shapes required: '+name)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (not np.isfinite(self.maximum_range_m) or self.maximum_range_m <= 0
                or not np.isfinite(self.pressure_std_m) or self.pressure_std_m < 0
                or not np.isfinite(self.information_budget) or not 0 < self.information_budget <= 1
                or self.independent_equivalent_features <= 0
                or not np.isfinite(self.robust_delta) or self.robust_delta <= 0
                or np.any(self.range_a <= 0) or np.any(self.range_b <= 0)
                or np.any(self.range_a > self.maximum_range_m)
                or np.any(self.range_b > self.maximum_range_m)
                or np.any(self.standard_deviation_m <= 0)):
            raise ValueError('Physical ranges, positive noise and a conservative information budget required')

    @property
    def features(self):
        return len(self.range_a)

    def payload(self):
        result = {name: value.tolist() if isinstance(value, np.ndarray) else value
                  for name, value in self.__dict__.items()}
        result.update(schema='oceansim.sss.mapped_range_change.v1',
                      landmark_height_profiled_over_physical_interval=True,
                      DEM_or_altitude_used=False, motion_seed_as_measurement=False)
        return result


def world_origins(pose, origins):
    pose = np.asarray(pose, float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError('Finite planar graph pose required')
    r = rotation(pose[2])
    points = origins@r.T+pose[:2]
    yaw_derivative = origins@np.array([[0., 1.], [-1., 0.]])@r.T
    return points, yaw_derivative


def profiled_range_residual_jacobians(pose_a, pose_b, data, *, robust=True):
    """Residual and two WORLD-coordinate Jacobians for independent checking.

    Input origins are measured receiver offsets in each graph-node frame;
    sensor_z is pressure-derived vertical position at that actual reception.
    The denominator and sigma depend on observations, not optimized poses.
    The range-normalized squared difference has metres as its unit.
    """
    oa, dya = world_origins(pose_a, data.origin_a)
    ob, dyb = world_origins(pose_b, data.origin_b)
    va, vb = data.landmark_xy-oa, data.landmark_xy-ob
    difference = (np.einsum('ij,ij->i', vb, vb)-np.einsum('ij,ij->i', va, va)
                  -data.range_b**2+data.range_a**2)
    dz = data.sensor_z_b-data.sensor_z_a
    middle = (data.sensor_z_a+data.sensor_z_b)/2
    # The union of admissible noisy sensor-height intervals is conservative.
    # It is NOT a normal-distribution prior centred at an arbitrary seabed Z.
    upper_z = np.minimum(data.sensor_z_a, data.sensor_z_b)+3*data.pressure_std_m
    lower_z = np.minimum(data.sensor_z_a, data.sensor_z_b)-data.maximum_range_m-3*data.pressure_std_m
    e0 = 2*dz*(lower_z-middle)
    e1 = 2*dz*(upper_z-middle)
    low, high = np.minimum(e0, e1), np.maximum(e0, e1)
    difference_to_interval = difference-np.clip(difference, low, high)
    # Exactly equal depths have an equality, not a flat interval with zero
    # derivative at its optimum. Nonzero intervals retain their true null.
    active = (difference < low) | (difference > high) | (low == high)
    denominator = (data.range_a+data.range_b)*data.standard_deviation_m
    error = difference_to_interval/denominator
    ja = np.column_stack([2*va, 2*np.einsum('ij,ij->i', va, dya)])
    jb = np.column_stack([-2*vb, -2*np.einsum('ij,ij->i', vb, dyb)])
    ja *= (active/denominator)[:, None]
    jb *= (active/denominator)[:, None]
    if robust:
        # Per-feature pseudo-Huber loss represented as an exact transformed
        # least-squares residual. This is not a true-pose or residual gate.
        v = np.sqrt(1+(error/data.robust_delta)**2)
        derivative = np.sqrt((v+1)/2)/v
        error = error*np.sqrt(2/(v+1))
        ja *= derivative[:, None]
        jb *= derivative[:, None]
    weight = np.sqrt(data.information_budget*min(data.independent_equivalent_features,
                                               data.features)/data.features)
    return error*weight, ja*weight, jb*weight


def conservative_range_change_sigma(range_a, range_b, baseline_m, *, range_std_m=.7,
                                    map_xy_std_m=2.5, receiver_offset_std_m=1.):
    """Freeze noise from measured ranges and a causal *estimated* baseline.

    The same uncertain map XY occurs in both ranges: its difference Jacobian
    is proportional to baseline, not twice a fictitious independent map fix.
    Receiver-offset uncertainty covers within-submap odometry approximation.
    """
    ra, rb = np.asarray(range_a, float), np.asarray(range_b, float)
    if np.any(ra <= 0) or np.any(rb <= 0) or baseline_m < 0:
        raise ValueError('Positive observed ranges and nonnegative estimated baseline required')
    total = ra+rb
    variance = 4*(ra**2+rb**2)*range_std_m**2/total**2
    variance += (2*float(baseline_m)*map_xy_std_m/total)**2+2*receiver_offset_std_m**2
    return np.sqrt(variance)
