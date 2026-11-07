"""ALL 2020 recorded acquisitions: joint measured depth/denoised intensity.

The full observed field is for display. Physically disjoint even/odd records
form the estimator reference and simulator query respectively. No GPS, no
RGB input, no artificial texture, and no flat-plane replacement of bathymetry.
Nominal straight source motion remains an explicit uncalibrated assumption.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np

from BathymetrySideScanDrawing import PingFile
from QiandaoLocalScene import bounded_interpolation, reject_gps_file
from QiandaoCombTrajectory import CombSurveyTrajectory, CombSurveyParameters
from QiandaoNativeBottomProjection import native_bottom_samples


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def appearance_values(values, lookup):
    """Explicit uncalibrated HSV-V appearance, not recovered acoustic power.

    The fixed lookup is the max RGB component of each of the 256 colours
    transformed by the actual Shotwell engine. The native display quantizer
    is applied before lookup, exactly as in the reviewed palette image.
    This happens before spatial bin averaging, never from a portrait crop.
    """
    values, lookup = np.asarray(values, float), np.asarray(lookup, float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or lookup.shape != (256,) or not np.isfinite(lookup).all()
            or lookup[0] != 0 or lookup.min() < 0 or lookup.max() > 1
            or (np.diff(lookup) < 0).any()):
        raise ValueError('Finite nonnegative native values and a monotonic black-preserving 256-entry LUT required.')
    index = np.minimum((np.clip(values, 0, 1)*256).astype(np.uint16), 255)
    return lookup[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw-cache', type=Path, required=True)
    parser.add_argument('--stopped-experiment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--edge-patch', type=Path,
                        help='Frozen gain_and_commonmode last512 patch; original baseline unchanged if absent.')
    parser.add_argument('--shotwell-appearance-lut', type=Path,
                        help='Fixed 256-entry enhanced HSV-V display appearance, NOT calibrated echo amplitude.')
    a = parser.parse_args()
    sys.addaudithook(reject_gps_file)
    if a.out.exists():
        raise FileExistsError('Keep earlier full-scene preparations and failures.')
    if shutil.disk_usage(a.out.parent).free < 1024**3:
        raise RuntimeError('Need1GiB free disk; do not delete unrelated projects.')
    a.out.mkdir()
    scene = a.out/'scene'
    scene.mkdir()
    started = time.monotonic()
    ledger = json.loads((a.raw_cache/'run_summary.json').read_text())
    audit = json.loads((a.stopped_experiment/'stopped_input_provenance_audit.json').read_text())
    if not ledger['original_all_native_samples_reproduced_exactly'] or not audit['all_2020_physical_records_and_16_beams']:
        raise ValueError('Original native cache and stopped denoising audit required.')
    for path, guard in ledger['source_guards'].items():
        stat = Path(path).stat()
        if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != guard:
            raise ValueError('Original measured source changed.')
    gain_path = a.stopped_experiment/'inputs/fusion_wide_front_supported_log_delta.npy'
    if sha(gain_path) != audit['baseline_gain_file_SHA256']:
        raise ValueError('Frozen full native gain changed.')
    raw_path = a.raw_cache/'原始MAT逐束包络_float64.npy'
    raw, gain = np.load(raw_path, mmap_mode='r'), np.load(gain_path, mmap_mode='r')
    edge = None
    lookup = None
    if a.edge_patch is not None:
        if sha(a.edge_patch) != audit['edge_gain_file_SHA256']:
            raise ValueError('Not the reviewed full native gain_and_commonmode patch.')
        edge = np.load(a.edge_patch, mmap_mode='r')
        if edge.dtype != np.float32 or edge.shape != (2, 8, 2020, 512):
            raise ValueError('Complete original16-beam/2020-record last512 edge patch required.')
    if a.shotwell_appearance_lut is not None:
        if edge is None:
            raise ValueError('The newly approved appearance uses the reviewed edge repair first.')
        lookup = np.load(a.shotwell_appearance_lut)
        appearance_values(np.array([0., 1.]), lookup)  # validate once before geometry
    variant = ('edge_shotwell_HSV_V_appearance' if lookup is not None else
               'gain_and_commonmode_linear_amplitude' if edge is not None else 'frozen_baseline_linear_amplitude')
    identities = np.load(a.raw_cache/'PingNum_PingS_PingUs.npy')
    if raw.shape != (2, 8, 2020, 12189) or gain.shape != raw.shape or identities.shape != (2020, 3):
        raise ValueError('All native2020 records,16beams and12189range samples required.')
    times = identities[:, 1]+identities[:, 2]*1e-6
    times = times-times[0]
    if np.any(np.diff(times) <= 0) or len(np.unique(identities[:, 0])) != 2020:
        raise ValueError('Unique chronological physical acquisition identities required.')
    speed, resolution, source_depth = 4*.5144444444444445, .3, 2.3
    extent = float(speed*times[-1])
    shape = (int(np.ceil(490/resolution))+1, int(np.ceil((extent+6)/resolution))+1)
    zsum = [np.zeros(shape, np.float64), np.zeros(shape, np.float64)]
    asum = [np.zeros(shape, np.float64), np.zeros(shape, np.float64)]
    counts = [np.zeros(shape, np.uint32), np.zeros(shape, np.uint32)]
    diagnostics, parameter_records, source_files = [], [], []
    with (scene/'全2020物理ping索引.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['record_index', 'part', 'part_record_index', 'PingNum', 'PingS', 'PingUs',
            'relative_time_s', 'nominal_X_m', 'partition', 'port_native_bottom_bins', 'starboard_native_bottom_bins',
            'native_rows_each_side', 'native_range_samples', 'raw_cache', 'frozen_gain'])
        for part in ledger['parts']:
            folder = a.source/part['name']
            with h5py.File(folder/'SysPara.mat', 'r') as f:
                p = {k: float(f['SysPara'][k][0, 0]) for k in
                     ('c', 'BeamSigFs', 'InstallAngle', 'BeamSpacingAlongTrackDisRes', 'BeamAlongTrackNum')}
            if p['BeamAlongTrackNum'] != 8 or p['BeamSpacingAlongTrackDisRes'] != .15:
                raise ValueError('Eight synchronous physical rows required.')
            parameter_records.append(dict(part=part['name'], parameters=p))
            source_files.append(dict(path=str(folder/'SysPara.mat'), SHA256=sha(folder/'SysPara.mat')))
            with ExitStack() as stack:
                readers = []
                for side_name in ('Port', 'Stbd'):
                    path = folder/(side_name+'NormalModeDepthTotal.mat')
                    reader = stack.enter_context(PingFile(path))
                    if len(reader) != part['count']:
                        raise ValueError('Depth record count differs from native amplitude part.')
                    readers.append(reader)
                    source_files.append(dict(path=str(path), SHA256=sha(path)))
                for local in range(part['count']):
                    index = part['offset']+local
                    expected = tuple(identities[index])
                    if any(reader.identity(local) != expected for reader in readers):
                        raise ValueError('Same physical depth and side-scan acquisition must align exactly.')
                    partition = index % 2
                    retained = []
                    for side, reader in enumerate(readers):
                        angle, delay = reader.depth(local)
                        current_gain = np.asarray(gain[side, :, index])
                        if edge is not None:
                            current_gain = current_gain.copy()
                            current_gain[:, -512:] = edge[side, :, index]
                        xyz, values, beam, native_bin = native_bottom_samples(angle, delay,
                            np.asarray(raw[side, :, index]), current_gain, side=side,
                            sound_speed=p['c'], sampling_frequency=p['BeamSigFs'], installation_deg=p['InstallAngle'],
                            beam_spacing_m=p['BeamSpacingAlongTrackDisRes'],
                            vehicle_along_m=speed*times[index]-extent/2, platform_depth_m=source_depth)
                        if lookup is not None:
                            values = appearance_values(values, lookup)
                        ix = np.rint(xyz[:, 0]/resolution+(shape[1]-1)/2).astype(int)
                        iy = np.rint(xyz[:, 1]/resolution+(shape[0]-1)/2).astype(int)
                        inside = (ix >= 0)&(ix < shape[1])&(iy >= 0)&(iy < shape[0])
                        ix, iy, xyz, values = ix[inside], iy[inside], xyz[inside], values[inside]
                        np.add.at(zsum[partition], (iy, ix), xyz[:, 2])
                        np.add.at(asum[partition], (iy, ix), values)
                        np.add.at(counts[partition], (iy, ix), 1)
                        retained.append(int(len(values)))
                    diagnostics.append(retained)
                    writer.writerow([index, part['name'], local, *expected, float(times[index]),
                        float(speed*times[index]-extent/2), 'reference_even' if not partition else 'query_odd',
                        *retained, 8, 12189, str(raw_path.resolve()), str(gain_path.resolve())])
                    if index % 16 == 0:
                        raw._mmap.madvise(mmap.MADV_DONTNEED)
                        gain._mmap.madvise(mmap.MADV_DONTNEED)
                    if (index+1) % 160 == 0:
                        print(json.dumps(dict(stage='all_records_joint_geometry', records=index+1,
                                             total=2020, elapsed_s=time.monotonic()-started)), flush=True)
    fields = {}
    # Each query surface comes ONLY from its own odd records; unlike the
    # earlier pilot, it does not inherit the reference bathymetry as truth.
    for name, index in [('reference_baseline', 0), ('query_baseline', 1), ('full_baseline_display_only', None)]:
        count = counts[index] if index is not None else counts[0]+counts[1]
        ztotal = zsum[index] if index is not None else zsum[0]+zsum[1]
        atotal = asum[index] if index is not None else asum[0]+asum[1]
        zmean = np.divide(ztotal, count, out=np.full(shape, np.nan), where=count > 0)
        amean = np.divide(atotal, count, out=np.full(shape, np.nan), where=count > 0)
        z, zm = bounded_interpolation(zmean, count, resolution=resolution, max_distance_m=1.5, smooth_sigma_m=.6)
        amplitude, am = bounded_interpolation(amean, count, resolution=resolution, max_distance_m=1.5)
        fields[name] = (z, amplitude, zm & am, count)
        del zmean, amean
    common = fields['reference_baseline'][2]&fields['query_baseline'][2]
    for name, (z, amplitude, own_mask, count) in fields.items():
        folder = scene/name
        folder.mkdir()
        mask = own_mask if 'display_only' in name else common
        amplitude = amplitude.copy()
        amplitude[~mask] = np.nan
        np.save(folder/'elevation_m.npy', z)
        np.save(folder/'backscatter.npy', amplitude)
        np.save(folder/'bathymetry_valid_mask.npy', mask)
        np.save(folder/'backscatter_valid_mask.npy', mask)
        np.save(folder/'observed_cell_counts.npy', count)
        metadata = dict(resolution_m=resolution,
            source_product='ALL2020 same-echo measured angle-delay bathymetry plus frozen denoised native amplitude',
            coordinate_frame='nominal source straight-track local simulation frame, NOT geodetic',
            backscatter_units='uncalibrated linear envelope amplitude proxy, NOT calibrated reflectivity',
            appearance_variant=variant,
            numeric_map_derived_from_native_beams_not_portrait_pixels=True,
            shotwell_display_value_is_not_calibrated_acoustic_amplitude=lookup is not None,
            radiometric_mapping_stage='per_native_bottom_sample_before_spatial_bin_average',
            assumed_layout_speed_mps=speed, source_platform_depth_m_assumed=source_depth,
            source_platform_attitude_assumed='level straight source platform, not recovered',
            GPS_read_or_used=False, actual_vessel_trajectory_or_attitude_recovered=False,
            physical_record_partition=name, all2020_used_in_complete_scene=True,
            depth_and_backscatter_share_original_raw_return_geometry=True,
            native_intensity_bins_preserved_before_grid_averaging=True,
            native_geometry_interpolated_only_between_adjacent_valid_DOA_nodes=True,
            measured_absolute_bathymetry_accuracy_unknown=True,
            elevation_smoothing_sigma_m=.6, maximum_local_interpolation_distance_m=1.5,
            reference_and_query_depth_are_not_identical_by_construction=True,
            full_map_display_must_not_be_estimator_reference=('display_only' in name))
        if lookup is not None:
            metadata['backscatter_units'] = 'uncalibrated enhanced sRGB HSV-V appearance proxy, NOT physical echo amplitude'
            metadata['source_product'] = 'ALL2020 native samples with frozen denoising, reviewed edge repair and fixed Shotwell appearance'
            metadata['appearance_lut_SHA256'] = sha(a.shotwell_appearance_lut)
        if edge is not None:
            metadata['edge_gain_patch_SHA256'] = sha(a.edge_patch)
        write(folder/'metadata.json', metadata)
    np.save(scene/'physical_record_timestamps_s.npy', times)
    np.save(scene/'reference_record_ids.npy', np.arange(0, 2020, 2))
    np.save(scene/'query_record_ids.npy', np.arange(1, 2020, 2))
    np.save(scene/'all_record_PingNum_PingS_PingUs.npy', identities)
    np.save(scene/'per_record_port_starboard_retained_soundings.npy', np.asarray(diagnostics, np.int32))
    route = CombSurveyTrajectory(CombSurveyParameters(line_length_m=1300.))
    route_manifest = route.manifest()
    write(a.out/'AUV全幅梳形轨迹.json', route_manifest)
    definition = dict(status='full_record_joint_scene_prepared_navigation_not_yet_launched',
        source_physical_records=2020, part_counts={'Data01': 1033, 'Data02': 987},
        expected_route_length_m=route.length_m, route_parameters=route_manifest['parameters'],
        GPS_read_or_used=False, measured_depth_and_denoised_side_scan_enabled=True,
        full_native_image_shape=[8084, 24378], full_native_image_is_not_geodetic_map=True,
        virtual_observations_on_new_route_are_simulated_not_original_comb_measurements=True)
    definition['appearance_variant'] = variant
    write(a.out/'experiment_definition.json', definition)
    report = dict(status='completed_ALL2020_both_parts_joint_measured_scene',
        source_records=2020, parts=parameter_records, grid_shape=list(shape), resolution_m=resolution,
        source_nominal_along_track_extent_m=extent, duration_source_s=float(times[-1]),
        reference_record_count=1010, query_record_count=1010, shared_record_IDs=0,
        full_display_record_count=2020, all_depth_side_identities_exact=True,
        all_physical_records_retained_in_index=True, original_native_array_shape=list(raw.shape),
        per_side_retained_soundings=np.asarray(diagnostics).sum(axis=0).tolist(),
        retained_counts_are_native_bins_with_interpolated_geometry_not_independent_soundings=True,
        every_retained_intensity_is_an_actual_native_bin_after_frozen_gain=True,
        records_with_no_supported_soundings=int(np.count_nonzero(np.sum(diagnostics, axis=1)==0)),
        common_appearance_cells=int(common.sum()), scene_valid_cells=int(common.sum()),
        reference_query_elevation_shared=False, nominal_layout_not_recovered_survey_pose=True,
        depth_quality_policy='original physical gates; .6m raster smoothing; no claim of calibrated absolute precision',
        measured_bottom_geometry_not_flat=True, GPS_file_opened=False,
        baseline_gain_SHA256=sha(gain_path), source_files=source_files,
        source_index_SHA256=sha(a.raw_cache/'PingNum_PingS_PingUs.npy'),
        per_ping_index_SHA256=sha(scene/'全2020物理ping索引.csv'),
        source_code_SHA256={n: sha(Path(__file__).parent/n) for n in
            ('PrepareQiandaoFullFrameJointScene.py', 'QiandaoLocalScene.py', 'QiandaoCombTrajectory.py',
             'QiandaoNativeBottomProjection.py')},
        old_denoising_weights_or_algorithm_changed=False, raw_MAT_modified=False,
        appearance_variant=variant,
        edge_gain_patch_SHA256=None if edge is None else sha(a.edge_patch),
        appearance_lut_SHA256=None if lookup is None else sha(a.shotwell_appearance_lut),
        appearance_mapping_after_gain_before_grid_average=lookup is not None,
        elapsed_s=time.monotonic()-started)
    write(scene/'scene_report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
