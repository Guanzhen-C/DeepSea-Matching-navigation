from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanComplexEcho import (ComplexEchoParameters, PointScattererIQSimulator,
                                fixed_scatterer_phase, fractional_impulse)


class FlatAmplitudeWorld:
    def seabed_height(self, x, y):
        return np.full_like(x, -10.)
    def reflectivity(self, x, y, measured_only):
        assert measured_only
        return .5+.1*np.sin(x*2)+.1*np.cos(y)


def test_fixed_scatterer_phase_depends_on_world_cell_not_ping_or_array_order():
    a = fixed_scatterer_phase(np.array([1, -2, 3]), np.array([4, 5, -6]), 31)
    b = fixed_scatterer_phase(np.array([3, 1]), np.array([-6, 4]), 31)
    np.testing.assert_array_equal(a[[2, 0]], b)
    assert not np.all(a == fixed_scatterer_phase([1, -2, 3], [4, 5, -6], 32))


def test_fractional_delay_and_two_way_carrier_phase_are_not_independent_random_draws():
    p = ComplexEchoParameters()
    r = np.array([10.1234])
    a = fractional_impulse(r, np.array([1.+0j]), p)
    sample = 2*r[0]/p.sound_speed_mps*p.sample_rate_hz
    assert np.flatnonzero(abs(a) > 0).tolist() == [int(np.floor(sample)), int(np.floor(sample))+1]
    expected = np.exp(-4j*np.pi*p.demodulation_hz*r[0]/p.sound_speed_mps)
    assert a.sum() == pytest.approx(expected, abs=1e-10)
    # At a fixed delayed envelope origin, propagation uses the declared LO.
    b = fractional_impulse(r+p.sound_speed_mps/p.demodulation_hz/4, np.array([1.+0j]), p)
    assert b.sum() == pytest.approx(-a.sum(), abs=1e-10)


def test_actual_syspara_baseband_fits_and_aliasing_is_rejected():
    p = ComplexEchoParameters()
    assert p.range_bins == 12721
    assert p.row_centres_m[0] == pytest.approx(-.525)
    with pytest.raises(ValueError, match='Nyquist'):
        ComplexEchoParameters(sample_rate_hz=10000.)


def test_same_pose_gives_same_clean_IQ_and_noise_does_not_change_it():
    p = ComplexEchoParameters(max_range_m=20., scatter_spacing_m=.3,
                              thermal_noise_rms=.01, interference_rms=.02)
    sim = PointScattererIQSimulator(FlatAmplitudeWorld(), p)
    a = sim.ping(0., [0., 0., -2.], np.eye(3))
    b = sim.ping(0., [0., 0., -2.], np.eye(3))
    assert a['observed_IQ'].shape == (2, 8, p.range_bins)
    assert np.iscomplexobj(a['observed_IQ'])
    assert np.isfinite(a['observed_IQ']).all() and abs(a['clean_simulated_IQ']).max() > 0
    np.testing.assert_array_equal(a['clean_simulated_IQ'], b['clean_simulated_IQ'])
    assert not np.array_equal(a['observed_IQ'], b['observed_IQ'])
    c = sim.ping(0., [.025, 0., -2.], np.eye(3))
    assert not np.array_equal(a['clean_simulated_IQ'], c['clean_simulated_IQ'])


def test_pulse_compression_peak_has_correct_range_and_no_group_delay_bias():
    p = ComplexEchoParameters(max_range_m=20.)
    sim = PointScattererIQSimulator(FlatAmplitudeWorld(), p)
    # Integer envelope delay avoids the declared linear fractional-delay approximation.
    sample = 500
    impulse = fractional_impulse(np.array([sample*p.sound_speed_mps/(2*p.sample_rate_hz)]), np.array([1.]), p)
    from scipy.signal import fftconvolve
    received = fftconvolve(impulse, sim.pulse, mode='same')
    compressed = sim.pulse_compress(received)
    assert abs(compressed).argmax() == sample
    assert abs(compressed[sample]) == pytest.approx(1., abs=1e-6)


def test_received_IQ_matches_direct_RF_delay_then_LO_demodulation():
    """Independent analytic receiver oracle catches double-counted offsets."""
    from scipy.signal import fftconvolve
    p = ComplexEchoParameters(max_range_m=20.)
    sim = PointScattererIQSimulator(FlatAmplitudeWorld(), p)
    sample = 500; delay = sample/p.sample_rate_hz
    ranges = np.array([delay*p.sound_speed_mps/2])
    received = fftconvolve(fractional_impulse(ranges, np.ones(1), p), sim.pulse, mode='same')
    fast_time = np.arange(p.range_bins)/p.sample_rate_hz
    active = abs(fast_time-delay) < p.pulse_duration_s/2
    # Transmit carrier at fc, delay the physical waveform, then mix by LO
    # at the RECEIVER'S time, not a second delayed transmit time.
    expected = np.exp(2j*np.pi*((p.carrier_hz-p.demodulation_hz)*fast_time
        -p.carrier_hz*delay+.5*p.chirp_rate_hz_s*(fast_time-delay)**2))
    expected *= active/np.sqrt(np.count_nonzero(abs(sim.pulse) > 0))
    np.testing.assert_allclose(received, expected, atol=2e-7)
