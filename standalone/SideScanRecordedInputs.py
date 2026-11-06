"""Lossless navigation-input recording/replay, with no simulator geometry.

Keep the scalar sonar observations and every IMU/DVL/depth sample so later
algorithm-only comparisons need not rerender a scene or invent a new stream.
Unsupported sonar bins remain zero. No clean IQ, world XYZ or DEM is stored.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from PureSideScanFeatures import SideScanFeaturePacket


KINDS = {'imu': 1, 'dvl': 2, 'depth': 3}
NAV_DTYPE = np.dtype([('timestamp_s', '<f8'), ('kind', 'u1'), ('values', '<f8', (6,))])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


class SideScanInputRecorder:
    def __init__(self, directory, expected_events, *, bins=512, initialization=None):
        self.root = Path(directory); self.root.mkdir(exist_ok=False)
        self.expected = dict(expected_events); self.bins = int(bins)
        self.counts = dict.fromkeys(self.expected, 0)
        self.initialization = initialization
        self.navigation = np.lib.format.open_memmap(self.root/'navigation_sensors.npy', mode='w+',
            dtype=NAV_DTYPE, shape=(sum(v for k, v in self.expected.items() if k != 'sonar'),))
        self.amplitude = np.lib.format.open_memmap(self.root/'received_side_scan_amplitudes.npy', mode='w+',
            dtype='<f8', shape=(self.expected['sonar'], 2, 8, self.bins))
        self.timestamps = np.lib.format.open_memmap(self.root/'side_scan_timestamps_s.npy', mode='w+',
            dtype='<f8', shape=(self.expected['sonar'],))
        self.index = 0; self.last_navigation_key = (-np.inf, 0)
        self.calibration = None

    def add_navigation(self, kind, timestamp, values):
        values = np.asarray(values, float).reshape(-1)
        size = {'imu': 6, 'dvl': 3, 'depth': 1}[kind]
        key = (float(timestamp), KINDS[kind])
        if values.size != size or not np.isfinite(values).all() or key <= self.last_navigation_key:
            raise ValueError('Finite ordered native navigation observations required.')
        if self.index >= len(self.navigation): raise ValueError('Too many navigation events.')
        self.navigation[self.index] = (timestamp, KINDS[kind], np.pad(values, (0, 6-size)))
        self.index += 1; self.counts[kind] += 1; self.last_navigation_key = key

    def add_side_scan(self, packet):
        i = self.counts['sonar']
        if i >= len(self.timestamps) or (i and packet.timestamp <= self.timestamps[i-1]):
            raise ValueError('Ordered unique physical sonar timestamps required.')
        if np.any(packet.sample_sequence >= self.bins): raise ValueError('Recorded native navigation raster changed.')
        calibration = dict(max_range_m=packet.max_range_m, row_centres_m=packet.row_centres_m.tolist(), bins=self.bins)
        if self.calibration is not None and self.calibration != calibration:
            raise ValueError('Changing calibration cannot be hidden in one recorded observation stream.')
        self.calibration = calibration
        expected_ranges = (packet.sample_sequence+.5)*packet.max_range_m/self.bins
        if not np.array_equal(expected_ranges, packet.ranges_m):
            # Some producers evaluate index/bins*range in another order;
            # a genuine sample-grid change must still be rejected.
            if not np.allclose(expected_ranges, packet.ranges_m, rtol=0, atol=1e-12):
                raise ValueError('Only explicit uniform range sample labels can be recorded here.')
        flat = (packet.sides*8+packet.rows)*self.bins+packet.sample_sequence
        if len(np.unique(flat)) != len(flat): raise ValueError('Duplicate observed sonar bins are forbidden.')
        self.amplitude[i] = 0.
        self.amplitude[i, packet.sides, packet.rows, packet.sample_sequence] = packet.amplitudes
        self.timestamps[i] = packet.timestamp
        self.counts['sonar'] += 1

    def finish(self):
        if self.counts != self.expected: raise ValueError('Incomplete input stream cannot be certified for replay.')
        for array in (self.navigation, self.amplitude, self.timestamps): array.flush()
        files = {name: sha(self.root/name) for name in
            ('navigation_sensors.npy', 'received_side_scan_amplitudes.npy', 'side_scan_timestamps_s.npy')}
        report = dict(status='complete_lossless_navigation_inputs', files_SHA256=files,
            events=self.counts, calibration=self.calibration, initialization=self.initialization,
            full_navigation_precision='float64', world_XYZ_or_heights_saved=False,
            clean_IQ_or_future_truth_saved=False, extra_denoising=False)
        (self.root/'input_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        return report


class RecordedSideScanInputs:
    def __init__(self, directory):
        self.root = Path(directory)
        self.manifest = json.loads((self.root/'input_manifest.json').read_text())
        if self.manifest['status'] != 'complete_lossless_navigation_inputs':
            raise ValueError('Only terminal complete input records can be replayed.')
        for name, expected in self.manifest['files_SHA256'].items():
            if sha(self.root/name) != expected: raise ValueError('Recorded inputs changed: '+name)
        self.navigation = np.load(self.root/'navigation_sensors.npy', mmap_mode='r')
        self.amplitude = np.load(self.root/'received_side_scan_amplitudes.npy', mmap_mode='r')
        self.timestamps = np.load(self.root/'side_scan_timestamps_s.npy', mmap_mode='r')

    def packet(self, i):
        amp = self.amplitude[i]; side, row, index = np.nonzero(amp > 0)
        c = self.manifest['calibration']
        return SideScanFeaturePacket(timestamp=float(self.timestamps[i]),
            ranges_m=(index+.5)/c['bins']*c['max_range_m'], amplitudes=amp[side, row, index],
            rows=row, sides=side, sample_sequence=index, row_centres_m=np.array(c['row_centres_m']),
            max_range_m=c['max_range_m'])

    def events(self):
        ni, si = 0, 0
        names = {v: k for k, v in KINDS.items()}
        while ni < len(self.navigation) or si < len(self.timestamps):
            nt = float(self.navigation[ni]['timestamp_s']) if ni < len(self.navigation) else np.inf
            st = float(self.timestamps[si]) if si < len(self.timestamps) else np.inf
            if nt <= st:
                r = self.navigation[ni]; kind = names[int(r['kind'])]
                size = {'imu': 6, 'dvl': 3, 'depth': 1}[kind]
                yield nt, kind, np.array(r['values'][:size])
                ni += 1
            else:
                yield st, 'sonar', self.packet(si)
                si += 1
