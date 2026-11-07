"""Closed, curvature-continuous lawnmower trajectory in a declared local frame.

This is a programmed *virtual* trajectory, not a recovered vessel route. The
arc-length parameterization keeps speed constant through both survey lines
and turns; quintic Bezier turns have zero curvature at their endpoints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator


def bezier(control, u):
    control = np.asarray(control, dtype=float)
    u = np.asarray(u, dtype=float)
    degree = len(control)-1
    weights = np.stack([comb(degree, k)*u**k*(1-u)**(degree-k)
                        for k in range(degree+1)], axis=-1)
    return weights@control


class PathSegment:
    def __init__(self, control, kind):
        self.control = np.asarray(control, float)
        if self.control.ndim != 2 or self.control.shape[1] != 2 or len(self.control) not in (2, 6):
            raise ValueError('A finite line or quintic planar Bezier is required.')
        if not np.isfinite(self.control).all():
            raise ValueError('Finite control points required.')
        self.kind = str(kind)
        self.first = (len(self.control)-1)*np.diff(self.control, axis=0)
        self.second = (len(self.first)-1)*np.diff(self.first, axis=0)
        u = np.linspace(0, 1, 8193)
        speed = np.linalg.norm(bezier(self.first, u), axis=1)
        if np.any(speed <= 0):
            raise ValueError('Stationary or reversing Bezier tangent is not allowed.')
        s = np.r_[0., cumulative_trapezoid(speed, u)]
        self.length_m = float(s[-1])
        self.inverse_arclength = PchipInterpolator(s, u, extrapolate=False)

    def evaluate(self, arclength):
        s = np.asarray(arclength, float)
        if np.any((s < -1e-8) | (s > self.length_m+1e-8)):
            raise ValueError('Segment distance is outside the programmed path.')
        u = self.inverse_arclength(np.clip(s, 0, self.length_m))
        xy = bezier(self.control, u)
        derivative = bezier(self.first, u)
        norm = np.linalg.norm(derivative, axis=-1)
        tangent = derivative/norm[..., None]
        if len(self.control) == 2:
            curvature = np.zeros(np.shape(u))
        else:
            second = bezier(self.second, u)
            curvature = (derivative[..., 0]*second[..., 1]
                         - derivative[..., 1]*second[..., 0])/norm**3
        return xy, tangent, curvature


@dataclass(frozen=True)
class CombSurveyParameters:
    lanes: int = 12
    line_length_m: float = 250.
    line_spacing_m: float = 28.
    return_corner_width_m: float = 24.
    speed_mps: float = .8
    vehicle_z_m: float = -8.
    virtual_flat_bottom_z_m: float = -50.

    def __post_init__(self):
        if self.lanes < 2 or self.lanes % 2:
            raise ValueError('Closed return geometry requires an even lane count >=2.')
        values = np.array([self.line_length_m, self.line_spacing_m,
            self.return_corner_width_m, self.speed_mps], float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Finite positive route dimensions/speed required.')
        height = (self.lanes-1)*self.line_spacing_m
        if 2*self.return_corner_width_m >= height:
            raise ValueError('Return corners must leave a positive vertical return leg.')
        if not np.isfinite([self.vehicle_z_m, self.virtual_flat_bottom_z_m]).all() or self.vehicle_z_m <= self.virtual_flat_bottom_z_m:
            raise ValueError('Declared vehicle altitude above the virtual flat plane required.')


def quarter_turn(start, end, initial_heading, final_heading, width):
    p0, p5 = np.asarray(start, float), np.asarray(end, float)
    t0 = np.array([np.cos(initial_heading), np.sin(initial_heading)])
    t1 = np.array([np.cos(final_heading), np.sin(final_heading)])
    a, b = .30*width, .65*width
    return np.array([p0, p0+a*t0, p0+b*t0, p5-b*t1, p5-a*t1, p5])


class CombSurveyTrajectory:
    def __init__(self, parameters=None):
        self.parameters = parameters or CombSurveyParameters()
        p = self.parameters
        self.speed_mps = p.speed_mps
        left, right = -p.line_length_m/2, p.line_length_m/2
        y0 = -(p.lanes-1)*p.line_spacing_m/2
        self.lane_y_m = y0+np.arange(p.lanes)*p.line_spacing_m
        self.segments = []
        for k, y in enumerate(self.lane_y_m):
            direction = 1 if k % 2 == 0 else -1
            begin, end = (left, right) if direction > 0 else (right, left)
            self.segments.append(PathSegment([[begin, y], [end, y]], 'survey_line'))
            if k < p.lanes-1:
                d = p.line_spacing_m
                a, b = .4*d, .75*d
                control = np.array([[0, 0], [a, 0], [b, 0], [b, d], [a, d], [0, d]], float)
                control[:, 0] = end+direction*control[:, 0]
                control[:, 1] += y
                self.segments.append(PathSegment(control, 'survey_uturn'))
        w, top, bottom = p.return_corner_width_m, self.lane_y_m[-1], self.lane_y_m[0]
        self.segments.append(PathSegment(quarter_turn([left, top], [left-w, top-w],
            np.pi, -np.pi/2, w), 'return_turn'))
        self.segments.append(PathSegment([[left-w, top-w], [left-w, bottom+w]], 'return_line'))
        self.segments.append(PathSegment(quarter_turn([left-w, bottom+w], [left, bottom],
            -np.pi/2, 0., w), 'return_turn'))
        self.ends_m = np.cumsum([s.length_m for s in self.segments])
        self.length_m = float(self.ends_m[-1])
        self.duration_s = self.length_m/self.speed_mps

    def evaluate_distance(self, distance):
        distance = np.asarray(distance, float)
        if not np.isfinite(distance).all() or np.any((distance < -1e-8) | (distance > self.length_m+1e-8)):
            raise ValueError('Distance outside the single declared survey loop.')
        distance = np.clip(distance, 0, self.length_m)
        original_shape = distance.shape
        flat = distance.reshape(-1)
        segment = np.minimum(np.searchsorted(self.ends_m, flat, side='right'), len(self.segments)-1)
        starts = np.r_[0, self.ends_m[:-1]]
        xy, tangent, curvature = np.empty((len(flat), 2)), np.empty((len(flat), 2)), np.empty(len(flat))
        for index in np.unique(segment):
            selected = segment == index
            xy[selected], tangent[selected], curvature[selected] = self.segments[index].evaluate(flat[selected]-starts[index])
        return (xy.reshape(original_shape+(2,)), tangent.reshape(original_shape+(2,)),
                curvature.reshape(original_shape), segment.reshape(original_shape))

    def position(self, t):
        xy = self.evaluate_distance(float(t)*self.speed_mps)[0]
        return np.r_[xy, self.parameters.vehicle_z_m]

    def state(self, t):
        # Imported only when actual sensor simulation is requested. Planning
        # and unit tests need neither Isaac Kit nor the navigation engine.
        from oceansim.slam.large_scale_sim import GroundTruthState
        xy, tangent, curvature, _ = self.evaluate_distance(float(t)*self.speed_mps)
        yaw = np.arctan2(tangent[1], tangent[0])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        velocity = np.r_[self.speed_mps*tangent, 0.]
        acceleration = np.r_[self.speed_mps**2*curvature*np.array([-tangent[1], tangent[0]]), 0.]
        return GroundTruthState(float(t), rotation, np.r_[xy, self.parameters.vehicle_z_m],
            velocity, acceleration, np.array([0., 0., self.speed_mps*curvature]),
            rotation.T@(acceleration+[0., 0., 9.81]))

    def manifest(self):
        return dict(schema='oceansim.qiandao.curvature-continuous-comb/v1',
            parameters=asdict(self.parameters), length_m=self.length_m,
            duration_s=self.duration_s, laps=1, closed=True,
            coordinate_frame='nominal local frame, NOT geographic coordinates',
            source='programmed virtual AUV, not measured vessel navigation', GPS_used=False,
            segments=[dict(kind=s.kind, length_m=s.length_m, control_points_xy_m=s.control.tolist())
                      for s in self.segments])
