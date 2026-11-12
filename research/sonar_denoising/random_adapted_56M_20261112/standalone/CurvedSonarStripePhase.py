"""Continuous curved candidate-stripe phase fitted in native coordinates.

Fit the gradient of an analytic envelope carrier, avoiding global phase
unwrapping. A scalar cubic phase enforces cross-range integrability. Neither
a physical circular wave source nor a pure-noise decomposition is claimed.
"""
from __future__ import annotations

import numpy as np


PROTOCOL = "native_analytic_carrier_gradient_integrable_cubic_phase_v1"
POWERS = ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
          (3, 0), (2, 1), (1, 2), (0, 3))


def phase_values(geometry, rows, columns, *, gradient=None):
    """Evaluate native phase cycles; tangent-continue beyond fitted range."""
    y, x = np.broadcast_arrays(np.asarray(rows, np.float64), np.asarray(columns, np.float64))
    if geometry.get('kind') == 'elliptic_ripple_morphology':
        cy, cx = geometry['centre_native_coordinates']
        aspect = geometry['native_range_to_record_axis_scale']
        dy, dx = y - cy, x - cx
        radius = np.sqrt(dy * dy + (aspect * dx)**2)
        if np.any(radius <= 0):
            raise ValueError('Ripple centre cannot be a synthesized source sample.')
        frequency = geometry['radial_frequency_cycles_per_scaled_sample']
        if gradient is None:
            return frequency * (radius - geometry['reference_radius'])
        if gradient == 'record':
            return frequency * dy / radius
        if gradient == 'range':
            return frequency * aspect**2 * dx / radius
        raise ValueError('Gradient must be record or range.')
    lo, hi = geometry["fit_range_interval"]
    clamped = np.clip(x, lo, hi)
    sy, sx = geometry["coordinate_scale"]
    oy, ox = geometry["coordinate_origin"]
    yn, xn = (y - oy) / sy, (clamped - ox) / sx
    result = np.zeros(y.shape, np.float64)
    tangent = np.zeros_like(result)
    for coefficient, (py, px) in zip(geometry["coefficients_cycles"], POWERS):
        if gradient is None:
            result += coefficient * yn**py * xn**px
            if px:
                tangent += coefficient * px * yn**py * xn**(px - 1) / sx
        elif gradient == "range":
            if px:
                result += coefficient * px * yn**py * xn**(px - 1) / sx
        elif gradient == "record":
            if py:
                result += coefficient * py * yn**(py - 1) * xn**px / sy
            if py and px:
                tangent += coefficient * py * px * yn**(py - 1) * xn**(px - 1) / (sy * sx)
        else:
            raise ValueError("Gradient must be record or range.")
    return result + tangent * (x - clamped) if gradient != "range" else result


def ripple_geometry(fitted, records, range_bins):
    """Explicit nested-arc augmentation, with measured local period/direction.

    Finite focal distance is a MORPHOLOGY DESIGN, not a measured source
    location. Preserve that distinction instead of pretending this is the
    unique physically identified noise geometry.
    """
    fy = float(fitted['fitted_record_frequency_q05_q50_q95'][1])
    fx = abs(float(fitted['fitted_range_frequency_q05_q50_q95'][1]))
    if min(fy, fx, records, range_bins) <= 0:
        raise ValueError('Positive measured frequencies and native dimensions required.')
    y0 = (records - 1) / 2
    x0 = float(np.mean(fitted['fit_range_interval']))
    cy, cx = -.75 * records, 0.
    dy, dx = y0 - cy, x0 - cx
    aspect = np.sqrt((fx / fy) * dy / dx)
    radius = np.sqrt(dy**2 + (aspect * dx)**2)
    frequency = fy * radius / dy
    geometry = dict(protocol='elliptic_nested_ripple_native_phase_morphology_v1',
        kind='elliptic_ripple_morphology', centre_native_coordinates=[cy, cx],
        native_range_to_record_axis_scale=float(aspect),
        radial_frequency_cycles_per_scaled_sample=float(frequency), reference_radius=float(radius),
        reference_native_point=[y0, x0], matched_local_record_frequency=fy, matched_local_range_frequency=fx,
        curvature_and_finite_focal_distance_are_design_NOT_device_calibration=True,
        physical_circle_or_source_location_identified=False)
    xx = np.array([0., (range_bins - 1) / 2, range_bins - 1.])
    yy = cy + np.sqrt(radius**2 - aspect**2 * (xx - cx)**2)
    geometry['reference_arc_midpoint_bow_native_records'] = float(yy[1] - (yy[0] + yy[2]) / 2)
    return geometry


def fit_analytic_gradient(analytic, *, first_record=0, first_column=0):
    a = np.asarray(analytic)
    if (a.ndim != 2 or a.shape[0] < 64 or a.shape[1] < 128 or not np.iscomplexobj(a)
            or not np.isfinite(a).all()):
        raise ValueError("Finite native analytic carrier with at least64x128 samples required.")
    h, w = a.shape
    origin = [first_record + (h - 1) / 2, first_column + (w - 1) / 2]
    scale = [(h - 1) / 2, (w - 1) / 2]
    geometry = dict(protocol=PROTOCOL, coefficients_cycles=[0.] * len(POWERS),
                    coordinate_origin=origin, coordinate_scale=scale,
                    fit_range_interval=[first_column, first_column + w - 1],
                    fit_record_interval=[first_record, first_record + h - 1],
                    physical_circle_or_source_location_identified=False)
    matrices, targets, weights = [], [], []
    for axis, bandwidth in ((0, .025), (1, .003)):
        head, tail = [slice(None)] * 2, [slice(None)] * 2
        head[axis], tail[axis] = slice(1, None), slice(None, -1)
        first, previous = a[tuple(head)], a[tuple(tail)]
        frequency = np.angle(first * previous.conj()) / (2 * np.pi)
        confidence = np.sqrt(np.abs(first) * np.abs(previous))
        yy, xx = np.meshgrid(np.arange(frequency.shape[0], dtype=float) + first_record,
                             np.arange(frequency.shape[1], dtype=float) + first_column, indexing="ij")
        if axis == 0:
            yy += .5
        else:
            xx += .5
        # Subsample the estimator only. No source/output samples are resized.
        select = (slice(16, -16, 8), slice(32, -32, 16))
        yy, xx, frequency, confidence = (v[select].ravel() for v in (yy, xx, frequency, confidence))
        yn, xn = (yy - origin[0]) / scale[0], (xx - origin[1]) / scale[1]
        columns = []
        for py, px in POWERS:
            if axis == 0:
                term = py * yn**(py - 1) * xn**px / scale[0] if py else np.zeros_like(yn)
            else:
                term = px * yn**py * xn**(px - 1) / scale[1] if px else np.zeros_like(xn)
            columns.append(term / bandwidth)
        matrices.append(np.stack(columns, axis=1))
        targets.append(frequency / bandwidth)
        reference = max(float(np.quantile(confidence, .9)), 1e-30)
        weights.append(np.minimum(confidence / reference, 1.))
    matrix, target, base_weight = np.concatenate(matrices), np.concatenate(targets), np.concatenate(weights)
    if base_weight.sum() <= 0:
        raise ValueError("Nonzero analytic carrier support required.")
    weight = base_weight.copy()
    for _ in range(8):
        root = np.sqrt(weight)
        coefficients = np.linalg.lstsq(matrix * root[:, None], target * root, rcond=1e-10)[0]
        residual = matrix @ coefficients - target
        robust_scale = max(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 1e-6)
        weight = base_weight * np.minimum(1., 1.5 * robust_scale / np.maximum(np.abs(residual), 1e-30))
    geometry["coefficients_cycles"] = coefficients.tolist()
    yy, xx = np.meshgrid(np.linspace(first_record, first_record + h - 1, 25),
                         np.linspace(first_column, first_column + w - 1, 33), indexing="ij")
    gy, gx = (phase_values(geometry, yy, xx, gradient=k) for k in ("record", "range"))
    geometry["fitted_record_frequency_q05_q50_q95"] = np.quantile(gy, [.05, .5, .95]).tolist()
    geometry["fitted_range_frequency_q05_q50_q95"] = np.quantile(gx, [.05, .5, .95]).tolist()
    geometry["local_slope_range_not_fixed_straight_line"] = [float(np.min(-gx / gy)), float(np.max(-gx / gy))]
    geometry["normalized_gradient_fit_residual_RMS"] = float(np.sqrt(np.average(residual**2, weights=base_weight)))
    geometry["no_global_unwrap_required"] = True
    return geometry


def fit_curved_phase(candidate, *, first_record=0, first_column=0):
    d = np.asarray(candidate)
    if d.ndim != 2 or not np.iscomplexobj(d) or not np.isfinite(d).all():
        raise ValueError("Actual finite complex candidate field required.")
    q = np.log(np.maximum(np.abs(d).astype(np.float64), 1e-12))
    q -= np.median(q, axis=0, keepdims=True)
    h, w = q.shape
    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    transform = np.fft.fft2(q * window)
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    eligible = (fy >= 1 / 24) & (fy <= 1 / 4) & (np.abs(fx) <= .04)
    power = np.abs(transform)**2
    yi, xi = np.unravel_index(np.where(eligible, power, -1).argmax(), q.shape)
    centre_y, centre_x = float(fy[yi, 0]), float(fx[0, xi])
    selector = np.exp(-.5 * (((fy - centre_y) / .025)**2 + ((fx - centre_x) / .003)**2)) * (fy > 0)
    analytic = np.fft.ifft2(transform * selector)
    geometry = fit_analytic_gradient(analytic, first_record=first_record, first_column=first_column)
    geometry.update(analytic_carrier_centre=[centre_y, centre_x],
                    analytic_bandwidth_cycles_per_native_sample=[.025, .003],
                    carrier_phase_is_candidate_modulation_NOT_geology_truth=True)
    return geometry
