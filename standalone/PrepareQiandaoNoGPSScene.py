"""Build a bounded small real-sonar-derived scene WITHOUT opening GPS.

Reference/query acoustic samples use disjoint even/odd physical records. The
along-track layout is an explicit constant-speed assumption, not a recovered
boat trajectory. This prepares a simulation world, not independent field GT.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import h5py
import numpy as np

from BathymetrySideScanDrawing import PingFile
from QiandaoLocalScene import (bottom_points, sample_native_profiles, grid_average,
                              bounded_interpolation, physical_partition, reject_gps_file)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--raw-cache', required=True, type=Path)
    p.add_argument('--baseline-gain', required=True, type=Path)
    p.add_argument('--edge-gain', required=True, type=Path)
    a = p.parse_args()
    sys.addaudithook(reject_gps_file)
    definition = json.loads((a.experiment/'experiment_definition.json').read_text())
    out = a.experiment/'scene'
    if out.exists():
        raise FileExistsError('Preserve prior or failed scene attempts.')
    if definition['GPS_read_or_used'] is not False:
        raise ValueError('GPS is forbidden by the current user instruction.')
    out.mkdir()
    start = time.monotonic()
    raw = np.load(a.raw_cache/'原始MAT逐束包络_float64.npy', mmap_mode='r')
    baseline = np.load(a.baseline_gain, mmap_mode='r')
    edge = np.load(a.edge_gain, mmap_mode='r')
    if raw.shape != (2, 8, 2020, 12189) or baseline.shape != raw.shape or edge.shape != raw.shape[:-1]+(512,):
        raise ValueError('Exact native source and stopped-algorithm gain fields required.')
    ledger = json.loads((a.raw_cache/'run_summary.json').read_text())
    if not ledger['original_all_native_samples_reproduced_exactly']:
        raise ValueError('Previously verified original magnitude cache required.')
    for path, stamp in ledger['source_guards'].items():
        s = Path(path).stat()
        if stamp != dict(size=s.st_size, mtime_ns=s.st_mtime_ns):
            raise ValueError('Raw source changed: '+path)
    ids = np.arange(definition['physical_record_start'], definition['physical_record_start']+definition['physical_record_count'])
    reference_ids, query_ids = physical_partition(ids)
    if max(ids) >= 1033:
        raise ValueError('First small-region protocol explicitly covers Data01 only.')
    with h5py.File(a.source/'Data01/SysPara.mat', 'r') as h:
        keys = ('c', 'BeamSigFs', 'InstallAngle', 'BeamSpacingAlongTrackDisRes', 'BeamAlongTrackNum')
        parameters = {k: float(h['SysPara'][k][0, 0]) for k in keys}
    if parameters['BeamAlongTrackNum'] != 8 or parameters['BeamSpacingAlongTrackDisRes'] != .15:
        raise ValueError('Supplied eight-synchronous-row layout required.')
    speed = definition['scene_layout_speed_mps_assumed']
    resolution = definition['map_resolution_m']
    coordinates = {'reference': [], 'query': []}
    observations = {'reference': {'baseline': []}, 'query': {n: [] for n in ('raw', 'baseline', 'edge')}}
    timestamps = []
    source_paths = []
    with ExitStack() as stack:
        readers = []
        for side in ('Port', 'Stbd'):
            f = a.source/'Data01'/(side+'NormalModeDepthTotal.mat')
            source_paths.append(f)
            readers.append(stack.enter_context(PingFile(f)))
        t0 = readers[0].identity(int(ids[0]))
        origin = t0[1]+t0[2]*1e-6
        last = readers[0].identity(int(ids[-1]))
        duration = last[1]+last[2]*1e-6-origin
        extent_x = speed*duration+6
        shape = (int(np.ceil(490/resolution))+1, int(np.ceil(extent_x/resolution))+1)
        center = speed*duration/2
        for index in ids:
            identity = readers[0].identity(int(index))
            if readers[1].identity(int(index)) != identity:
                raise ValueError('Sides disagree on physical acquisition identity.')
            stamp = identity[1]+identity[2]*1e-6-origin
            timestamps.append(stamp)
            partition = 'reference' if index % 2 == 0 else 'query'
            for side, reader in enumerate(readers):
                angle, delay = reader.depth(int(index))
                xyz, ranges, beams, _ = bottom_points(angle, delay, side=side,
                    sound_speed=parameters['c'], installation_deg=parameters['InstallAngle'],
                    beam_spacing_m=parameters['BeamSpacingAlongTrackDisRes'],
                    vehicle_along_m=speed*stamp-center,
                    platform_depth_m=definition['source_platform_depth_m_assumed'])
                magnitude = np.asarray(raw[side, :, index])
                gain = np.asarray(baseline[side, :, index])
                values = {'baseline': magnitude*np.exp(gain.astype(float))}
                if partition == 'query':
                    d = np.array(gain, dtype=float)
                    d[:, -512:] = edge[side, :, index]
                    values.update(raw=magnitude, edge=magnitude*np.exp(d))
                good = np.ones(len(xyz), bool)
                sampled = {}
                for name, field in values.items():
                    sampled[name], valid = sample_native_profiles(field, beams, ranges,
                        sound_speed=parameters['c'], sampling_frequency=parameters['BeamSigFs'])
                    good &= valid & np.isfinite(sampled[name]) & (sampled[name] > 0)
                coordinates[partition].append(xyz[good])
                for name in observations[partition]:
                    observations[partition][name].append(sampled[name][good])
            if len(timestamps) % 80 == 0:
                print(json.dumps(dict(records=len(timestamps), total=len(ids), elapsed_s=time.monotonic()-start)), flush=True)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError('Strict physical timestamps required.')
    fields = {}
    for partition in ('reference', 'query'):
        xyz = np.concatenate(coordinates[partition])
        scalars = {n: np.concatenate(v) for n, v in observations[partition].items()}
        if partition == 'reference':
            scalars['elevation'] = xyz[:, 2]
        mapped, count = grid_average(xyz, scalars, shape=shape, resolution=resolution)
        fields[partition] = {}
        for name, values in mapped.items():
            field, mask = bounded_interpolation(values, count, resolution=resolution,
                max_distance_m=definition['maximum_local_interpolation_distance_m'],
                smooth_sigma_m=.6 if name == 'elevation' else 0.)
            fields[partition][name] = field, mask
        np.save(out/(partition+'_observed_cell_counts.npy'), count.astype(np.uint16))
    elevation, bath_valid = fields['reference'].pop('elevation')
    common = bath_valid.copy()
    for v in fields.values():
        for field, mask in v.values():
            common &= mask
    for partition, values in fields.items():
        for name, (field, _) in values.items():
            folder = out/(partition+'_'+name)
            folder.mkdir()
            np.save(folder/'backscatter.npy', field)
            np.save(folder/'backscatter_valid_mask.npy', common)
            np.save(folder/'elevation_m.npy', elevation)
            np.save(folder/'bathymetry_valid_mask.npy', bath_valid)
            metadata = dict(resolution_m=resolution, source_product='Qiandao measured angle/delay local scene proxy WITHOUT GPS',
                coordinate_frame='explicit nominal straight-track local simulation frame, NOT geodetic',
                assumed_layout_speed_mps=speed, GPS_read_or_used=False,
                backscatter_units='uncalibrated linear envelope amplitude proxy, NOT calibrated reflectivity',
                metadata_quality='measured-source reconstruction with explicit nominal layout and interpolation',
                simulation_common_query_support=True, partition=partition, variant=name)
            (folder/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    np.save(out/'physical_record_timestamps_s.npy', np.asarray(timestamps))
    np.save(out/'reference_record_ids.npy', reference_ids)
    np.save(out/'query_record_ids.npy', query_ids)
    report = dict(status='completed_GPS_free_disjoint_record_reconstructed_scene_NOT_geodetic_truth',
        definition=definition, parameters=parameters, grid_shape=list(shape), duration_source_s=duration,
        source_nominal_along_track_extent_m=speed*duration, reference_record_count=len(reference_ids),
        query_record_count=len(query_ids), shared_record_IDs=int(np.intersect1d(reference_ids, query_ids).size),
        scene_valid_cells=int(bath_valid.sum()), common_appearance_cells=int(common.sum()),
        soundings_source_files=[str(f) for f in source_paths],
        baseline_gain_SHA256=sha(a.baseline_gain), edge_gain_SHA256=sha(a.edge_gain),
        source_code_SHA256={n: sha(Path(__file__).parent/n) for n in ('QiandaoLocalScene.py', 'PrepareQiandaoNoGPSScene.py')},
        raw_MAT_modified=False, GPS_file_opened=False, generated_navigation_sensors=False,
        actual_vessel_trajectory_or_attitude_recovered=False, independent_cross_voyage_validation=False,
        raw_stripes_are_not_proven_clean_or_noise_labels=True, elapsed_s=time.monotonic()-start)
    (out/'scene_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
