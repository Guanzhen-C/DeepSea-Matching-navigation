from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanIQHistory import packet_from_compressed_IQ
from SideScanComplexEcho import ComplexEchoParameters


def test_native_complex_packet_preserves_sides_rows_and_exports_no_clean_or_geometry():
    p = ComplexEchoParameters(max_range_m=20.)
    z = np.zeros((2, 8, p.range_bins), np.complex64)
    z[0, 3, 400:450] = 2j
    z[1, 5, 700:750] = .5+.5j
    original = z.copy()
    packet = packet_from_compressed_IQ(2., z, p, bins=64)
    assert packet.timestamp == 2.
    assert set(packet.sides) == {0, 1}
    assert set(packet.rows[packet.sides == 0]) == {3}
    assert set(packet.rows[packet.sides == 1]) == {5}
    assert np.all((packet.ranges_m > 0)&(packet.ranges_m < p.max_range_m))
    np.testing.assert_array_equal(z, original)
    for forbidden in ['clean_IQ', 'world_pose', 'height', 'points_sensor', 'beam_directions']:
        assert not hasattr(packet, forbidden)


def test_FFT_roundoff_does_not_create_observed_terrain_support():
    p = ComplexEchoParameters(max_range_m=20.)
    z = np.full((2, 8, p.range_bins), 1e-15+1e-15j, np.complex64)
    z[0, 0, 500:550] = 1.+1j
    packet = packet_from_compressed_IQ(0., z, p, bins=64)
    assert set(packet.rows) == {0} and set(packet.sides) == {0}


def test_envelope_adapter_rejects_real_image_disguised_as_complex():
    p = ComplexEchoParameters(max_range_m=20.)
    with pytest.raises(ValueError, match='complex'):
        packet_from_compressed_IQ(0., np.ones((2, 8, p.range_bins)), p)
