from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanRecordedInputs import SideScanInputRecorder, RecordedSideScanInputs
from SideScanIQHistory import packet_from_compressed_IQ
from SideScanComplexEcho import ComplexEchoParameters


def create_recording(tmp_path):
    p = ComplexEchoParameters(max_range_m=20.)
    recorder = SideScanInputRecorder(tmp_path/'inputs', dict(imu=2, dvl=1, depth=1, sonar=2), bins=32)
    z = np.zeros((2, 8, p.range_bins), np.complex64); z[0, 3, 300:400] = 1.+2j
    packets = [packet_from_compressed_IQ(t, z, p, bins=32) for t in [0., .03]]
    recorder.add_side_scan(packets[0]); recorder.add_navigation('imu', .02, np.arange(6))
    recorder.add_navigation('dvl', .02, [1., 2., 3.]); recorder.add_navigation('depth', .02, [8.])
    recorder.add_side_scan(packets[1]); recorder.add_navigation('imu', .04, np.arange(6)+1)
    recorder.finish()
    return packets


def test_all_received_navigation_samples_and_physical_sonar_packets_replay_exactly(tmp_path):
    packets = create_recording(tmp_path)
    replay = RecordedSideScanInputs(tmp_path/'inputs')
    events = list(replay.events())
    assert [(t, k) for t, k, _ in events] == [(0., 'sonar'), (.02, 'imu'), (.02, 'dvl'),
        (.02, 'depth'), (.03, 'sonar'), (.04, 'imu')]
    np.testing.assert_array_equal(events[1][2], np.arange(6))
    for i, packet in enumerate(packets):
        restored = replay.packet(i)
        for key in ['ranges_m', 'amplitudes', 'rows', 'sides', 'sample_sequence', 'row_centres_m']:
            np.testing.assert_array_equal(getattr(restored, key), getattr(packet, key))
    assert not replay.manifest['world_XYZ_or_heights_saved']
    assert not replay.manifest['extra_denoising']


def test_incomplete_or_modified_stream_cannot_be_used_as_a_valid_replay(tmp_path):
    r = SideScanInputRecorder(tmp_path/'incomplete', dict(imu=1, dvl=0, depth=0, sonar=1))
    with pytest.raises(ValueError, match='Incomplete'): r.finish()
    create_recording(tmp_path)
    target = tmp_path/'inputs/navigation_sensors.npy'
    with target.open('r+b') as f:
        f.seek(-1, 2); f.write(b'X')
    with pytest.raises(ValueError, match='changed'): RecordedSideScanInputs(tmp_path/'inputs')
