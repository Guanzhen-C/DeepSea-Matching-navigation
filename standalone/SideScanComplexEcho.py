"""Equivalent beamformed complex sonar echoes from a fixed surface field.

This is a declared point-scatterer/stop-and-hop simulation, not recovered
measured phase, element-level beamforming, a full-wave solver, or calibrated
electronic interference. The world geometry stays private to the simulator.
The measured denoised field is an AMPLITUDE APPEARANCE prior, not a calibrated
scattering coefficient: do not apply its original TVG/attenuation a second time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class ComplexEchoParameters:
    # Values read from the supplied SysPara; semantics of undocumented device
    # flags are NOT inferred here. Receiver processing stage is explicit.
    sound_speed_mps: float = 1480.
    sample_rate_hz: float = 37650.
    carrier_hz: float = 450000.
    demodulation_hz: float = 451800.
    chirp_rate_hz_s: float = -6000000.
    pulse_duration_s: float = .005
    max_range_m: float = 250.
    along_rows: int = 8
    row_spacing_m: float = .15
    scatter_spacing_m: float = .15
    along_footprint_sigma_m: float = .15
    half_swath_deg: float = 80.
    thermal_noise_rms: float = 0.
    interference_rms: float = 0.
    seed: int = 31
    receiver_forward_m: float = .10
    receiver_left_m: float = 0.
    receiver_up_m: float = -.15

    def __post_init__(self):
        if self.along_rows != 8 or not 0 < self.half_swath_deg < 90:
            raise ValueError('Eight simultaneous rows and an open downward fan required.')
        positive = [self.sound_speed_mps, self.sample_rate_hz, self.carrier_hz, self.pulse_duration_s,
            self.max_range_m, self.row_spacing_m, self.scatter_spacing_m, self.along_footprint_sigma_m]
        if not np.isfinite(list(asdict(self).values())).all() or min(positive) <= 0:
            raise ValueError('Finite positive physical dimensions/rates required.')
        bandwidth = abs(self.chirp_rate_hz_s)*self.pulse_duration_s
        offset = abs(self.carrier_hz-self.demodulation_hz)
        if offset+bandwidth/2 >= self.sample_rate_hz/2:
            raise ValueError('The complex baseband LFM must fit inside the Nyquist interval.')
        if min(self.thermal_noise_rms, self.interference_rms) < 0:
            raise ValueError('Nonnegative explicitly simulated noise amplitudes required.')
        if isinstance(self.seed, bool) or self.seed < 0 or int(self.seed) != self.seed:
            raise ValueError('A nonnegative integer simulator seed is required.')

    @property
    def range_bins(self):
        return int(np.ceil(2*self.max_range_m/self.sound_speed_mps*self.sample_rate_hz))+1

    @property
    def row_centres_m(self):
        return (np.arange(self.along_rows)-3.5)*self.row_spacing_m

    @property
    def receiver_translation_body(self):
        return np.array([self.receiver_forward_m, self.receiver_left_m, self.receiver_up_m])

    @property
    def navigation_row_centres_m(self):
        """Scan-plane offsets in the IMU/body frame, not array-local frame."""
        return self.row_centres_m+self.receiver_forward_m


def lattice_in_projected_strip(position, rotation, half_width, max_range, spacing):
    """Fixed WORLD lattice in a rotated fan footprint, without a huge AABB.

    Scanline bounds are the intersection of four projected half-planes.
    Vertical terms are conservatively bounded by max_range for a tilted
    sonar. No sampling density or world phase changes during a turn.
    """
    axis = np.asarray(rotation)[:2, :2].T
    limits = np.array([half_width, max_range])+max_range*abs(np.asarray(rotation)[2, :2])
    if abs(np.linalg.det(axis)) < 1e-6:
        raise ValueError('Near-vertical footprint requires a 3-D spatial index, not silent clipping.')
    corners = np.array([[x, y] for x in [-limits[0], limits[0]] for y in [-limits[1], limits[1]]])
    xy = corners@np.linalg.inv(axis).T+np.asarray(position)[:2]
    ix = np.arange(int(np.floor(xy[:, 0].min()/spacing)), int(np.ceil(xy[:, 0].max()/spacing))+1)
    x = ix*spacing
    ymin, ymax = np.full(len(x), -np.inf), np.full(len(x), np.inf)
    alive = np.ones(len(x), bool)
    for (a, b), limit in zip(axis, limits):
        v = a*(x-position[0])
        if abs(b) < 1e-12:
            alive &= abs(v) <= limit+1e-9
        else:
            low, high = position[1]+(-limit-v)/b, position[1]+(limit-v)/b
            ymin = np.maximum(ymin, np.minimum(low, high))
            ymax = np.minimum(ymax, np.maximum(low, high))
    lo = np.ceil((ymin-1e-9)/spacing).astype(np.int64)
    hi = np.floor((ymax+1e-9)/spacing).astype(np.int64)
    count = np.where(alive, np.maximum(hi-lo+1, 0), 0)
    total = int(count.sum())
    if total > 3000000:
        raise ValueError('Actual footprint needs spatial tiling; never silently decimate echoes.')
    prefix = np.r_[0, np.cumsum(count)[:-1]]
    return np.repeat(ix, count), np.repeat(lo, count)+np.arange(total)-np.repeat(prefix, count)


def fixed_scatterer_phase(ix, iy, seed):
    """Stateless world-indexed phase: no redraw when another ping sees a cell."""
    x, y = np.asarray(ix, np.int64).astype(np.uint64), np.asarray(iy, np.int64).astype(np.uint64)
    with np.errstate(over='ignore'):
        h = x*np.uint64(0x9e3779b97f4a7c15)^y*np.uint64(0xbf58476d1ce4e5b9)^np.uint64(seed)
        h = (h^(h >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        h = (h^(h >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        h ^= h >> np.uint64(31)
    return 2*np.pi*((h >> np.uint64(11)).astype(float)/2**53)-np.pi


def fractional_impulse(ranges_m, coefficients, parameters):
    """Fractional delay and propagation phase at the BASEBAND REFERENCE LO.

    pulse() already includes fc-LO as a delayed frequency component. Using
    fc again for this channel phase would double-count the carrier offset.
    """
    p = parameters
    r, a = np.asarray(ranges_m, float), np.asarray(coefficients, complex)
    if r.ndim != 1 or r.shape != a.shape or not np.isfinite(r).all() or not np.isfinite(a).all():
        raise ValueError('Finite aligned scatterer ranges and complex amplitudes required.')
    sample = 2*r/p.sound_speed_mps*p.sample_rate_hz
    lo = np.floor(sample).astype(int); fraction = sample-lo
    keep = (lo >= 0)&(lo+1 < p.range_bins)
    path_phase = np.exp(-2j*np.pi*np.remainder(2*p.demodulation_hz*r/p.sound_speed_mps, 1.))
    values = a*path_phase
    out = np.zeros(p.range_bins, np.complex128)
    np.add.at(out, lo[keep], values[keep]*(1-fraction[keep]))
    np.add.at(out, lo[keep]+1, values[keep]*fraction[keep])
    return out


class PointScattererIQSimulator:
    """World interface: scalar height and amplitude sampling, simulator only."""
    def __init__(self, world, parameters=None):
        self.world = world
        self.parameters = parameters or ComplexEchoParameters()
        p = self.parameters
        half = int(np.ceil(p.pulse_duration_s*p.sample_rate_hz/2))
        self.pulse_times_s = np.arange(-half, half+1)/p.sample_rate_hz
        active = abs(self.pulse_times_s) <= p.pulse_duration_s/2
        self.pulse = active*np.exp(2j*np.pi*((p.carrier_hz-p.demodulation_hz)*self.pulse_times_s
                                 + .5*p.chirp_rate_hz_s*self.pulse_times_s**2))
        self.pulse /= np.linalg.norm(self.pulse)
        self.pulse = self.pulse.astype(np.complex64)
        self.rng = np.random.default_rng(p.seed+190001)
        self.observation_id = 0

    def sample_scatterers(self, position, rotation):
        p = self.parameters
        width = max(abs(p.row_centres_m))+4*p.along_footprint_sigma_m
        ix, iy = lattice_in_projected_strip(position, rotation, width, p.max_range_m, p.scatter_spacing_m)
        x, y = ix*p.scatter_spacing_m, iy*p.scatter_spacing_m
        z = self.world.seabed_height(x, y)
        amplitude = self.world.reflectivity(x, y, measured_only=True)
        keep = np.isfinite(z)&np.isfinite(amplitude)&(amplitude > 0)
        xyz = np.column_stack([x, y, z])[keep]
        body = (xyz-position)@rotation
        keep_body = (abs(body[:, 0]) <= width)&(body[:, 2] < 0)
        xyz, body, amplitude = xyz[keep_body], body[keep_body], amplitude[keep][keep_body]
        phases = fixed_scatterer_phase(ix[keep][keep_body], iy[keep][keep_body], p.seed)
        return body, amplitude*np.exp(1j*phases)

    def ping(self, timestamp, position, rotation):
        p = self.parameters
        position, rotation = np.asarray(position, float), np.asarray(rotation, float)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError('Explicit simulator world pose required.')
        if (not np.isfinite(timestamp) or not np.isfinite(position).all() or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-7) or np.linalg.det(rotation) < 0):
            raise ValueError('Finite timestamp/position and a proper rotation required.')
        origin = position+rotation@p.receiver_translation_body
        points, coefficients = self.sample_scatterers(origin, rotation)
        clean = np.zeros((2, 8, p.range_bins), np.complex64)
        for beam, centre in enumerate(p.row_centres_m):
            delta = points-np.array([centre, 0., 0.])
            ranges = np.linalg.norm(delta, axis=1)
            footprint = np.exp(-.5*(delta[:, 0]/p.along_footprint_sigma_m)**2)
            eligible = (ranges <= p.max_range_m)&(abs(delta[:, 0]) <= 4*p.along_footprint_sigma_m)
            eligible &= abs(np.arctan2(delta[:, 1], -delta[:, 2])) <= np.deg2rad(p.half_swath_deg)
            # Density normalization is fixed by declared grid/beam width,
            # never fitted to the truth amplitude of this particular ping.
            weights = footprint*p.scatter_spacing_m/p.along_footprint_sigma_m
            for side in (0, 1):
                keep = eligible&((delta[:, 1] >= 0) if side == 0 else (delta[:, 1] < 0))
                impulse = fractional_impulse(ranges[keep], coefficients[keep]*weights[keep], p)
                clean[side, beam] = fftconvolve(impulse, self.pulse, mode='same')
        # Stop-and-hop coherent oscillator assumption; changing ranges
        # already account for inter-ping motion phase, not random phases.
        clean *= np.exp(2j*np.pi*np.remainder((p.carrier_hz-p.demodulation_hz)*timestamp, 1.))
        noise = self.rng.normal(size=clean.shape)+1j*self.rng.normal(size=clean.shape)
        noise *= p.thermal_noise_rms/np.sqrt(2)
        # Explicit synthetic instrument-like coherent bands. This is NOT a
        # claim to have identified the institute's electronic interference.
        n = np.arange(p.range_bins)
        crest = np.maximum(np.cos(2*np.pi*(timestamp/2.9+.0009*n+.00000002*n*n)), 0.)**6
        carrier = np.exp(2j*np.pi*4200*n/p.sample_rate_hz)
        interference = p.interference_rms*crest*carrier
        observed = np.asarray(clean+noise+interference[None, None], np.complex64)
        self.observation_id += 1
        return dict(timestamp_s=float(timestamp), observation_id=self.observation_id,
            clean_simulated_IQ=clean, observed_IQ=observed,
            pulse=self.pulse.copy(), parameters=asdict(p),
            stage='equivalent_beamformed_complex_baseband_before_pulse_compression',
            original_measured_phase_recovered=False, full_wave_or_element_level_simulation=False,
            single_scattering_no_multipath_or_terrain_occlusion=True,
            calibrated_scattering_and_interference=False, scene_geometry_private_to_simulator=True)

    def pulse_compress(self, iq):
        iq = np.asarray(iq)
        if not np.iscomplexobj(iq) or iq.shape[-1] != self.parameters.range_bins:
            raise ValueError('Native complex beamformed range traces required.')
        kernel = self.pulse.conj()[::-1]
        return fftconvolve(iq, kernel.reshape((1,)*(iq.ndim-1)+(-1,)), mode='same', axes=-1).astype(np.complex64)
