"""Pose-aware appearance projection with explicit received-echo provenance.

The lateral slant proxy is an IMAGE COORDINATE, not a measured ground range.
No height/DOA/DEM interface exists. Every populated pixel retains a label for
one dominant received contribution (not an averaged fake scan plane). The
image averages all overlapping received contributions without extra denoising.
Matching must use the labeled physical scan plane/range, never projected XY.
"""
from __future__ import annotations

import json
import numpy as np

from PureSideScanFeatures import resample_supported_echoes, normalize_feature_image
from SideScanFeatureMap import pose_matrix, extract_features


def projected_pixel_indices(query, pixels):
    p = np.asarray(pixels, float).reshape(-1, 2)
    x = np.rint(p[:, 0]).astype(int)
    y = np.rint((p[:, 1]+.5)/query['scale']-.5).astype(int)
    h, w = query['support'].shape
    valid = np.isfinite(p).all(axis=1)&(x >= 0)&(y >= 0)&(x < w)&(y < h)
    clipped_x, clipped_y = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
    valid &= query['source_phase_centre'][clipped_y, clipped_x] >= 0
    return clipped_y, clipped_x, valid


def projected_feature_observations(query, pixels):
    """Return real source observations; no image-derived horizontal position."""
    y, x, valid = projected_pixel_indices(query, pixels)
    if not valid.all(): raise ValueError('A feature must refer to actually received supported echoes.')
    phase = query['source_phase_centre'][y, x]
    column = query['source_range_column'][y, x].astype(int)
    bins, max_range = query['bins'], query['max_range_m']
    raw_side = np.where(column < bins, 1., -1.)
    ranges = np.where(raw_side > 0, bins-.5-column, column-bins+.5)*max_range/bins
    return dict(origins=query['phase_origins'][phase], normals=query['phase_normals'][phase],
        ranges=ranges, sides=raw_side*query['phase_side_sign'][phase],
        packet_ids=query['phase_packet_ids'][phase], beam_rows=query['phase_beam_rows'][phase],
        timestamps_s=query['phase_timestamps_s'][phase])


def build_projected_submap(window, anchor_pose2, *, bins=512, step=.5, scale=1, extract=True):
    scans = list(window.scans); z_values = np.asarray(window.vertical_positions, float)
    if (not scans or len(z_values) != len(scans) or bins < 2 or int(bins) != bins
            or not np.isfinite(step) or step <= 0 or scale < 1
            or int(scale) != scale or not np.isfinite(z_values).all()):
        raise ValueError('A nonempty causal received-ping window with pressure positions is required.')
    anchor_pose2 = np.asarray(anchor_pose2, float).reshape(3)
    if not np.isfinite(anchor_pose2).all(): raise ValueError('Finite ESTIMATED image-anchor pose required.')
    inverse = np.linalg.inv(pose_matrix(anchor_pose2))
    max_range = float(scans[-1][0].max_range_m); step_y = max_range/bins
    signed_slant = np.r_[(bins-.5-np.arange(bins))*step_y, -(np.arange(bins)+.5)*step_y]
    xs, ys, amplitudes, phases, columns = [], [], [], [], []
    origin_table, normal_table, sign_table, id_table, beam_table, time_table = [], [], [], [], [], []
    for k, ((packet, pose, identifier), z) in enumerate(zip(scans, z_values)):
        if packet.max_range_m != max_range: raise ValueError('Changing sonar calibration cannot be hidden.')
        relative = inverse@pose_matrix(pose)
        forward, lateral = relative[:2, 0], relative[:2, 1]
        origins = np.column_stack([relative[:2, 2]+packet.row_centres_m[:, None]*forward[None],
                                   np.full(8, z-z_values[-1])])
        # Plane sign is a convention; physical sides must change with it.
        orientation = 1. if (forward[0] > 1e-12 or (abs(forward[0]) <= 1e-12 and forward[1] >= 0)) else -1.
        origin_table.append(origins); normal_table.append(np.tile(orientation*forward, (8, 1)))
        sign_table.append(np.full(8, orientation, np.int8)); id_table.append(np.full(8, identifier, np.int64))
        beam_table.append(np.arange(8, dtype=np.int8)); time_table.append(np.full(8, packet.timestamp))
        uniform = resample_supported_echoes(packet, bins)
        row, col = np.nonzero(uniform > 0)
        if not len(row): continue
        # The zero-height slant proxy only arranges APPEARANCE. It is never
        # returned as a range/plane measurement or an observed landmark XY.
        projected = origins[row, :2]+signed_slant[col, None]*lateral[None]
        xs.append(projected[:, 0]); ys.append(projected[:, 1]); amplitudes.append(uniform[row, col])
        phases.append((k*8+row).astype(np.int32)); columns.append(col.astype(np.int32))
    if not amplitudes: raise ValueError('No received side-scan support for appearance projection.')
    px, py, value = np.concatenate(xs), np.concatenate(ys), np.concatenate(amplitudes)
    source_phase, source_column = np.concatenate(phases), np.concatenate(columns)
    x0, x1 = np.floor(px.min()/step)*step-step, np.ceil(px.max()/step)*step+step
    y0, y1 = np.floor(py.min()/step_y)*step_y-step_y, np.ceil(py.max()/step_y)*step_y+step_y
    height, width = int(round((x1-x0)/step))+1, int(round((y1-y0)/step_y))+1
    ri, ci = np.rint((px-x0)/step).astype(int), np.rint((y1-py)/step_y).astype(int)
    flat = ri*width+ci; size = height*width
    count = np.bincount(flat, minlength=size).reshape(height, width).astype(np.uint32)
    sums = np.bincount(flat, weights=value, minlength=size).reshape(height, width)
    support = count > 0
    image = np.divide(sums, count, out=np.zeros_like(sums), where=support)
    # Keep a genuine source observation at a blend, rather than averaging
    # incompatible phase centres/normals. Largest received amplitude is a
    # deterministic appearance-dominance proxy, not a correctness oracle.
    order = np.lexsort((np.arange(len(flat)), value, flat))
    sorted_flat = flat[order]
    dominant = order[np.r_[sorted_flat[1:] != sorted_flat[:-1], True]]
    labels = np.full(size, -1, np.int32); range_columns = np.zeros(size, np.int32)
    labels[flat[dominant]] = source_phase[dominant]
    range_columns[flat[dominant]] = source_column[dominant]
    result = dict(assembled_amplitude=image, raw_amplitude=image, observation_count=count,
        support=support, source_phase_centre=labels.reshape(height, width),
        source_range_column=range_columns.reshape(height, width), phase_origins=np.concatenate(origin_table),
        phase_normals=np.concatenate(normal_table), phase_side_sign=np.concatenate(sign_table),
        phase_packet_ids=np.concatenate(id_table), phase_beam_rows=np.concatenate(beam_table),
        phase_timestamps_s=np.concatenate(time_table), anchor_pose2=anchor_pose2.copy(),
        phase_forward_offsets_m=np.concatenate([s[0].row_centres_m for s in scans]),
        x0=float(x0), x1=float(x1),
        grid_y_min_m=float(y0), grid_y_max_m=float(y1), grid_step_y_m=step_y,
        step=step, scale=scale, bins=bins, max_range_m=max_range, anchor_vertical_z_m=float(z_values[-1]),
        timestamp=scans[-1][0].timestamp, patch_start_s=scans[0][0].timestamp,
        patch_start_id=scans[0][2], patch_end_id=scans[-1][2],
        physical_ping_timestamps_s=np.array([q[0].timestamp for q in scans]),
        physical_ping_observation_ids=np.array([q[2] for q in scans]),
        contributing_local_pose2=np.array([q[1] for q in scans]),
        received_contributions=int(len(value)), projected_contributions=int(count.sum()),
        distance_sampling_policy='shared_adjacent_acquisition_range_raster',
        native_input_scalar_observations=int(sum(len(s[0].amplitudes) for s in scans)),
        range_raster_contributions=int(len(value)),
        assembly_json=json.dumps(dict(assembly='ordinary_overlap_average_in_estimated_slant_appearance_projection',
            secondary_denoising_used=False, unknown_cells_filled=False, measured_heights_used=False,
            projected_XY_is_a_horizontal_measurement=False, all_received_heading_directions_included=True,
            geometry_source='dominant_actual_received_contribution_per_pixel_not_averaged_fake_plane')))
    if extract:
        u8, mask = normalize_feature_image(image, support, scale)
        pixels, descriptors = extract_features(u8, mask)
        _, _, valid = projected_pixel_indices(result, pixels)
        pixels, descriptors = pixels[valid], descriptors[valid]
        observation = projected_feature_observations(result, pixels)
        result.update(image=u8, pixels=pixels, descriptors=descriptors,
            normals=observation['normals'], plane_offset_m=np.einsum('ij,ij->i',
                observation['normals'], observation['origins'][:, :2]))
    return result
