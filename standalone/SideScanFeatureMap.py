"""2-D intensity-feature prior and scan-line geometric verification.

The navigation map exposes ONLY XY-indexed scalar intensity. There is no
height/normal/DOA API. Query slant range is a descriptor-image coordinate,
never a horizontal position measurement. Metric verification uses the known
along-track scan planes and geocoded XY feature landmarks.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from PureSideScanFeatures import normalize_feature_image, descriptor_pairs, resample_supported_echoes


def extract_features(image, mask, maximum=2400):
    """Scale-space RootSIFT for accumulated strips, not 8-row ping corners."""
    import cv2
    keypoints, descriptors = cv2.SIFT_create(nfeatures=maximum,
        contrastThreshold=.005).detectAndCompute(image, mask)
    if descriptors is None:
        return np.empty((0, 2)), np.empty((0, 128), np.float32)
    descriptors = np.sqrt(descriptors/np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12))
    return np.asarray([k.pt for k in keypoints]), descriptors


def pose_matrix(p):
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array([[c, -s, p[0]], [s, c, p[1]], [0., 0., 1.]])


def matrix_pose(t):
    return np.array([t[0, 2], t[1, 2], np.arctan2(t[1, 0], t[0, 0])])


class IntensityFeatureMap:
    def __init__(self, image, mask, resolution):
        self.image, self.mask = np.asarray(image), np.asarray(mask, bool)
        self.resolution = float(resolution)

    @classmethod
    def from_path(cls, path):
        p = Path(path)
        metadata = json.loads((p/'metadata.json').read_text())
        # Deliberately do NOT use MeasuredBathymetryGrid.from_path, which
        # would also read elevation/navigation surfaces.
        return cls(np.load(p/'backscatter.npy', mmap_mode='r'),
                   np.load(p/'backscatter_valid_mask.npy', mmap_mode='r'), metadata['resolution_m'])

    def sample(self, x, y):
        x, y = np.broadcast_arrays(x, y)
        row, col = y/self.resolution+(self.image.shape[0]-1)/2, x/self.resolution+(self.image.shape[1]-1)/2
        ix, iy = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = (ix >= 0)&(iy >= 0)&(ix+1 < self.image.shape[1])&(iy+1 < self.image.shape[0])
        result = np.full(x.shape, np.nan)
        if inside.any():
            c, r = ix[inside], iy[inside]
            good = self.mask[r, c]&self.mask[r+1, c]&self.mask[r, c+1]&self.mask[r+1, c+1]
            weight_x, weight_y = col[inside]-c, row[inside]-r
            values = ((1-weight_x)*(1-weight_y)*self.image[r, c]
                + weight_x*(1-weight_y)*self.image[r, c+1]
                + (1-weight_x)*weight_y*self.image[r+1, c]+weight_x*weight_y*self.image[r+1, c+1])
            values[~good] = np.nan; result[inside] = values
        return result

    def seabed_height(self, *args):
        raise AssertionError('Measured heights are forbidden in side-scan feature navigation.')

    def normal(self, *args):
        raise AssertionError('MBES surface normals are forbidden in side-scan feature navigation.')


class SideScanFeatureWindow:
    def __init__(self, pings=150):
        self.scans = deque(maxlen=pings)
        self.pings = int(pings)

    def add(self, packet, local_pose2, observation_id):
        if self.scans and (packet.timestamp <= self.scans[-1][0].timestamp
                          or observation_id <= self.scans[-1][2]):
            raise ValueError('Historical side-scan packets must be strictly causal and ordered.')
        self.scans.append((packet, np.asarray(local_pose2).copy(), int(observation_id)))

    def build(self, current_pose2, bins=512, step=.5, scale=1):
        current_inverse = np.linalg.inv(pose_matrix(current_pose2))
        entries = []
        for packet, local_pose, identifier in self.scans:
            relative = current_inverse@pose_matrix(local_pose)
            normal = relative[:2, 0]
            # A single rectified strip cannot represent scan planes nearly
            # perpendicular to its axis. This is a geometric support failure,
            # not a hidden pose correction or use of a DEM.
            if abs(normal[0]) < .5:
                continue
            intercept = (normal@relative[:2, 2]+packet.row_centres_m)/normal[0]
            entries.append((packet, intercept, normal, identifier))
        if not entries:
            raise ValueError('No common strip scan-plane support.')
        x0 = min(float(x[1].min()) for x in entries)-step
        x1 = max(float(x[1].max()) for x in entries)+step
        height = int(np.ceil((x1-x0)/step))+1
        image = np.zeros((height, 2*bins), float)
        weight = np.zeros_like(image, np.uint32)
        normal_sum = np.zeros((height, 2)); normal_count = np.zeros(height)
        for packet, intercept, normal, identifier in entries:
            row = np.rint((intercept-x0)/step).astype(int)
            uniform = resample_supported_echoes(packet, bins)
            np.add.at(image, row, uniform)
            np.add.at(weight, row, uniform > 0)
            for r in np.unique(row):
                normal_sum[r] += normal; normal_count[r] += 1
        # User explicitly cancelled the extra denoiser. This is ordinary
        # overlap averaging for assembling received scans, not Huber/neural
        # restoration and not an additional smoothing/filtering operation.
        np.divide(image, weight, out=image, where=weight > 0)
        known = normal_count > 0
        normals = np.column_stack([np.interp(np.arange(height), np.flatnonzero(known),
            normal_sum[known, k]/normal_count[known]) for k in range(2)])
        u8, mask = normalize_feature_image(image, image > 0, scale)
        pixel, descriptor = extract_features(u8, mask)
        row_float = (pixel[:, 1]+.5)/scale-.5
        along = x0+row_float*step
        feature_normal = np.column_stack([np.interp(row_float, np.arange(height), normals[:, k]) for k in range(2)])
        feature_normal /= np.linalg.norm(feature_normal, axis=1, keepdims=True)
        return dict(image=u8, raw_amplitude=image,
            assembled_amplitude=image, observation_count=weight, support=weight > 0,
            row_normals=normals,
            assembly_json=json.dumps(dict(secondary_denoising_used=False,
                assembly='ordinary_overlap_average_of_received_scans', unknown_cells_filled=False), sort_keys=True),
            current_ping_amplitude=resample_supported_echoes(self.scans[-1][0], bins),
            physical_ping_timestamps_s=np.array([entry[0].timestamp for entry in entries]),
            physical_ping_observation_ids=np.array([entry[3] for entry in entries]),
            contributing_local_pose2=np.array([x[1] for x in self.scans]),
            pixels=pixel, descriptors=descriptor, bins=bins, along_m=along,
            normals=feature_normal, plane_offset_m=along*feature_normal[:, 0], x0=x0, x1=x1,
            step=step, scale=scale, max_range_m=self.scans[-1][0].max_range_m,
            timestamp=self.scans[-1][0].timestamp,
            patch_start_s=self.scans[0][0].timestamp, patch_start_id=self.scans[0][2], patch_end_id=self.scans[-1][2])


def match_feature_prior(feature_map, query, seed_pose2, margin_m=30., image_matcher=None):
    step, scale, bins = query['step'], query['scale'], query['bins']
    max_range = query['max_range_m']
    x0, x1 = query['x0']-margin_m, query['x1']+margin_m
    xs = np.arange(x0, x1+step/2, step)
    ys = np.r_[(bins-.5-np.arange(bins))*max_range/bins,
               -(np.arange(bins)+.5)*max_range/bins]
    bx, by = np.meshgrid(xs, ys, indexing='ij')
    seed = pose_matrix(seed_pose2)
    wx = seed[0, 0]*bx+seed[0, 1]*by+seed[0, 2]
    wy = seed[1, 0]*bx+seed[1, 1]*by+seed[1, 2]
    amplitude = feature_map.sample(wx, wy)
    valid = np.isfinite(amplitude)&(amplitude > 0)
    try:
        image, mask = normalize_feature_image(np.where(valid, amplitude, 0), valid, scale)
    except ValueError as exc:
        return dict(accepted=False, reason=str(exc), matches=0)
    if image_matcher is None:
        mp, md = extract_features(image, mask, maximum=2400)
        map_frame = dict(image=image, pixels=mp, descriptors=md, bins=bins)
        pairs = descriptor_pairs(query, map_frame)
        qp = query['pixels']
        normals = query['normals'][pairs[:, 0]]
        offsets = query['plane_offset_m'][pairs[:, 0]]
    else:
        qm = np.repeat(query['support'], scale, axis=0).astype(np.uint8)*255
        qp, mp, confidence = image_matcher(query['image'], image, qm, mask)
        same_side = (qp[:, 0] < bins) == (mp[:, 0] < bins)
        qp, mp, confidence = qp[same_side], mp[same_side], confidence[same_side]
        pairs = np.column_stack([np.arange(len(qp))]*2)
        row_float = (qp[:, 1]+.5)/scale-.5
        normals = np.column_stack([np.interp(row_float, np.arange(len(query['row_normals'])),
            query['row_normals'][:, k]) for k in range(2)])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        offsets = (query['x0']+row_float*step)*normals[:, 0]
    debug = dict(query_image=query['image'], map_image=image,
        query_pixels=qp[pairs[:, 0]], map_pixels=mp[pairs[:, 1]],
        query_feature_count=len(qp), map_feature_count=len(mp))
    if len(pairs) < 24:
        return dict(accepted=False, reason='insufficient_mutual_feature_matches', matches=len(pairs), **debug)
    qi, mi = pairs.T
    mx = x0+((mp[mi, 1]+.5)/scale-.5)*step
    my = np.where(mp[mi, 0] < bins, bins-.5-mp[mi, 0], -(mp[mi, 0]-bins+.5))*max_range/bins
    landmarks_body = np.column_stack([mx, my])
    # For near-parallel scan planes the data support along-track/yaw, not an
    # independent cross-track fix. Keep the unsupported component at its
    # motion-prior seed and preserve a rank-deficient information matrix.
    linear = np.column_stack([-normals[:, 0], normals[:, 0]*my-normals[:, 1]*mx])
    base_residual = np.einsum('ij,ij->i', normals, landmarks_body)-offsets
    rng = np.random.default_rng(query['patch_end_id'])
    best = None
    for _ in range(256):
        ids = rng.choice(len(pairs), 2, replace=False)
        if abs(np.linalg.det(linear[ids])) < 1e-6:
            continue
        parameter = np.linalg.solve(linear[ids], -base_residual[ids])
        residual = base_residual+linear@parameter
        good = abs(residual) <= .7
        if best is None or good.sum() > best[0]:
            best = int(good.sum()), good, parameter
    if best is None or best[0] < 24:
        return dict(accepted=False, reason='insufficient_scan_plane_consensus', matches=len(pairs),
            inliers=0 if best is None else best[0], **debug)
    keep = best[1]
    points, normal, offset = landmarks_body[keep], normals[keep], offsets[keep]
    def residual(beta):
        c, s = np.cos(beta[1]), np.sin(beta[1])
        transformed = points@np.array([[c, -s], [s, c]])
        transformed[:, 0] -= beta[0]
        return np.einsum('ij,ij->i', normal, transformed)-offset
    fit = least_squares(residual, best[2], loss='huber', f_scale=.3)
    dx, dyaw = fit.x
    if not np.isfinite(fit.x).all() or abs(dx) > margin_m or abs(dyaw) > np.deg2rad(8):
        return dict(accepted=False, reason='feature_solution_outside_search_support', matches=len(pairs), **debug)
    pose = np.asarray(seed_pose2).copy()
    # residual() rotates landmarks into the candidate body frame BEFORE
    # subtracting dx. Thus dx belongs to the candidate, not seed, forward
    # axis. Preserve this convention in the pose sent to PF/GTSAM.
    pose[2] += dyaw
    pose[:2] += pose_matrix(pose)[:2, 0]*dx
    # Use the original conservative side-scan floors; zero information in
    # the unsupported cross-track direction is exact, not a tiny pseudo-fix.
    observed_basis = pose_matrix(pose)
    info = np.zeros((3, 3)); axis = np.r_[observed_basis[:2, 0], 0.]
    info += np.outer(axis, axis)/2.5**2
    info[2, 2] = 1/np.deg2rad(1.5)**2
    covariance = np.linalg.pinv(info)+np.outer(np.r_[observed_basis[:2, 1], 0.], np.r_[observed_basis[:2, 1], 0.])*20.**2
    world_landmarks = points@seed[:2, :2].T+seed[:2, 2]
    return dict(accepted=True, matches=len(pairs), inliers=best[0], matched_pose2=pose,
        query_feature_count=len(qp), map_feature_count=len(mp),
        information=info, proposal_covariance=covariance, observable_rank=2,
        score=best[0]/len(pairs), scan_plane_rmse_m=float(np.sqrt(np.mean(residual(fit.x)**2))),
        landmark_xy=world_landmarks, scan_normals=normal, scan_offsets=offset,
        local_reference_pose2=None, seed_pose2=np.asarray(seed_pose2).copy(),
        query_pixels=qp[qi[keep]], map_pixels=mp[mi[keep]],
        query_image=query['image'], map_image=image, measured_height_used=False,
        raw_intensity_correlation_used=False, optical_flow_used=False)
