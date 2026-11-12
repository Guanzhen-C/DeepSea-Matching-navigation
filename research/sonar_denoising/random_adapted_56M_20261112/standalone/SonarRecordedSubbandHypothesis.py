"""Finite, centered complex subband and LFM controls, NOT vendor reproduction.

Recorded target frequency is evidence for a processing domain, not proof that
all other energy is noise. Post-beamforming filtering may not commute with the
unknown upstream near-field processor. Actual range origin is not calibrated.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from SonarCompressionHypothesis import lfm_template


def complex_bandpass(fs, band, *, taps=257, beta=8.6):
    """Odd, centered FIR with unity frequency response at the band center.

    The real lowpass is modulated to a possibly negative complex frequency;
    it is not a real filter passing both positive and negative sidebands.
    This normalization is a design property, not image/peak gain adjustment.
    """
    if not math.isfinite(fs) or fs <= 0 or taps < 3 or taps % 2 != 1:
        raise ValueError('positive sampling frequency and odd finite FIR length required')
    if len(band) != 2 or not all(math.isfinite(x) for x in band):
        raise ValueError('two finite complex passband boundaries required')
    low, high = band
    if not -fs/2 < low < high < fs/2:
        raise ValueError('complex subband must be strictly inside Nyquist')
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('finite nonnegative Kaiser beta required')
    center = (low+high)/2
    real = signal.firwin(taps, (high-low)/2, fs=fs, window=('kaiser', beta))
    n = np.arange(taps)-(taps-1)//2
    return real*np.exp(2j*np.pi*center*n/fs)


def centered_filter(values, kernel):
    z = np.asarray(values); h = np.asarray(kernel)
    if (z.ndim < 1 or not np.iscomplexobj(z) or h.ndim != 1 or len(h) % 2 != 1
            or not np.isfinite(z).all() or not np.isfinite(h).all()):
        raise ValueError('finite original complex traces and an odd complex FIR required')
    # Linear, zero-extended convolution. The odd centered kernel introduces
    # no integer indexing delay; crop margins must still be excluded in tests.
    return signal.fftconvolve(z, h.reshape((1,)*(z.ndim-1)+(-1,)),
                              mode='same', axes=-1)


def bandlimited_chirp(fs, duration, slope, center, band, *, taps=257):
    original = lfm_template(fs=fs, duration=duration, slope=slope,
                            center_frequency=center)
    h = complex_bandpass(fs, band, taps=taps)
    # Keep the entire pulse and both finite FIR tails; cropping the template
    # back to its old length would silently change timing/spectral support.
    limited = signal.fftconvolve(original, h, mode='full')
    return original, limited, h


def unit_energy_correlate(values, template):
    z = np.asarray(values); s = np.asarray(template)
    if (z.ndim < 1 or not np.iscomplexobj(z) or s.ndim != 1 or len(s) < 2
            or not np.isfinite(z).all() or not np.isfinite(s).all()
            or np.linalg.norm(s) == 0):
        raise ValueError('finite native complex field and nonempty pulse template required')
    h = np.conj(s[::-1])/np.linalg.norm(s)
    return signal.fftconvolve(z, h.reshape((1,)*(z.ndim-1)+(-1,)),
                              mode='same', axes=-1)


def contiguous_half_power_width(amplitudes, peak):
    a = np.asarray(amplitudes)
    if a.ndim != 1 or not 0 <= peak < len(a) or not np.isfinite(a).all() or a[peak] <= 0:
        raise ValueError('positive known peak in a finite amplitude profile required')
    cutoff = a[peak]/math.sqrt(2)
    left = right = peak
    while left > 0 and a[left-1] >= cutoff: left -= 1
    while right+1 < len(a) and a[right+1] >= cutoff: right += 1
    return right-left+1


def synthetic_point_control(fs, duration, slope, center, band, *, sound_speed=1480.):
    original, limited, h = bandlimited_chirp(fs, duration, slope, center, band)
    length = 4096; start = 1000
    z = np.zeros((1, 1, length), np.complex128)
    z[0, 0, start:start+len(original)] = original
    results = {}
    for name, template in (('fullband_unit_energy_correlation', original),
                            ('recorded_subband_unit_energy_correlation', limited)):
        a = np.abs(unit_energy_correlate(z, template)[0, 0])
        peak = int(np.argmax(a))
        width = contiguous_half_power_width(a, peak)
        results[name] = dict(peak_index_0based=peak, pulse_leading_edge_index_0based=start,
            numerical_peak_offset_from_known_pulse_leading_edge_samples=peak-start,
            peak_amplitude=float(a[peak]), half_power_mainlobe_width_samples=width,
            hypothetical_half_power_width_m=width*sound_speed/(2*fs),
            template_length_samples=len(template),
            unit_energy_filter_not_unit_echo_amplitude_gain=True)
    bpf = np.abs(centered_filter(z, h)[0, 0]); raw = np.abs(z[0, 0])
    results['bandpass_without_correlation'] = dict(
        envelope_peak_index_0based=int(np.argmax(bpf)),
        envelope_peak_shift_from_pulse_leading_edge_samples=int(np.argmax(bpf))-start,
        raw_pulse_envelope_width_samples=contiguous_half_power_width(raw, start),
        bandpass_pulse_envelope_width_samples=contiguous_half_power_width(bpf, int(np.argmax(bpf))),
        envelope_peak_is_not_a_calibrated_target_range=True)
    return dict(noiseless_known_transmitted_chirp=True, native_length=length,
        sampling_frequency_Hz=fs, transmitted_pulse_samples=len(original),
        assumed_chirp_slope_Hz_per_s=slope, assumed_baseband_center_Hz=center,
        recorded_comparison_subband_Hz=list(band), results=results,
        synthetic_control_does_not_calibrate_real_range_origin=True,
        negative_subband_is_not_assumed_true_noise_free_signal=True)
