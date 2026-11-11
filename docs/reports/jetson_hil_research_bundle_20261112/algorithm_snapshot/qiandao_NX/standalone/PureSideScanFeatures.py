"""Side-scan image FEATURES only; no flow, DVL altitude, MBES or DEM input.

Only the calibrated along-track aperture converts feature pixels to metres.
Signed slant range is retained as an image coordinate, NEVER declared to be
horizontal ground distance. Its fitted slope is a nuisance image deformation,
not a heading or cross-track position observation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SideScanFeaturePacket:
    timestamp: float
    ranges_m: np.ndarray
    amplitudes: np.ndarray
    rows: np.ndarray
    sides: np.ndarray
    row_centres_m: np.ndarray
    sample_sequence: np.ndarray
    max_range_m: float = 250.

    def __post_init__(self):
        r, a, row, side = map(np.asarray, (self.ranges_m, self.amplitudes, self.rows, self.sides))
        centres = np.asarray(self.row_centres_m, float)
        sequence = np.asarray(self.sample_sequence)
        if r.ndim != 1 or any(x.shape != r.shape for x in (a, row, side)):
            raise ValueError('One raw range/amplitude/side/row per side-scan return required.')
        if centres.shape != (8,) or not np.all(np.diff(centres) > 0):
            raise ValueError('Eight calibrated simultaneous phase centres required.')
        if sequence.shape != r.shape or np.any(sequence < 0) or not np.equal(sequence, np.rint(sequence)).all():
            raise ValueError('Original nonnegative echo acquisition indices required to preserve missing-data gaps.')
        if (not np.isfinite(r).all() or not np.isfinite(a).all() or np.any(r <= 0)
                or np.any(a <= 0) or not np.isin(row, range(8)).all() or not np.isin(side, [0, 1]).all()):
            raise ValueError('Only valid measured side-scan scalars and labels are accepted.')
        for name, values, dtype in [('ranges_m', r, float), ('amplitudes', a, float),
                ('rows', row, np.int64), ('sides', side, np.int64), ('row_centres_m', centres, float),
                ('sample_sequence', sequence, np.int64)]:
            copied = np.array(values, dtype=dtype, copy=True); copied.flags.writeable = False
            object.__setattr__(self, name, copied)


def packet_from_simulated_side_scan(ping):
    """Simulator boundary exports range/amplitude only; geometry is discarded."""
    image = ping.imaging
    ranges = np.asarray(image.ranges, float)
    values = np.asarray(image.intensities, float)
    rows = np.asarray(image.along_track_row_indices)
    sides = np.repeat(np.arange(2), len(ranges)//2)
    sequence = np.zeros(len(ranges), np.int64)
    for side in (0, 1):
        for row in range(8):
            indices = np.flatnonzero((sides == side)&(rows == row))
            sequence[indices] = np.arange(len(indices))
    hit = np.asarray(image.hit_mask)&np.isfinite(ranges)&np.isfinite(values)&(ranges > 0)&(values > 0)
    return SideScanFeaturePacket(float(ping.timestamp), ranges[hit], values[hit], rows[hit], sides[hit],
                                 (np.arange(8)-3.5)*.15, sequence[hit])


def resample_supported_echoes(packet, bins=512):
    """Uniform slant raster, interpolating adjacent received samples only.

    Acquisition indices identify missing echoes, not beam directions or XYZ.
    No extension past the first/last return or across a missing sample.
    """
    result = np.zeros((8, 2*bins), float)
    target = (np.arange(bins)+.5)*packet.max_range_m/bins
    for side in (0, 1):
        for row in range(8):
            ids = np.flatnonzero((packet.sides == side)&(packet.rows == row))
            if len(ids) < 2:
                continue
            ids = ids[np.argsort(packet.ranges_m[ids])]
            ranges, values, sequence = packet.ranges_m[ids], packet.amplitudes[ids], packet.sample_sequence[ids]
            left = np.searchsorted(ranges, target, side='right')-1
            right = left+1
            inside = (left >= 0)&(right < len(ranges))
            li, ri = np.clip(left, 0, len(ranges)-1), np.clip(right, 0, len(ranges)-1)
            inside &= (abs(sequence[ri]-sequence[li]) == 1)&(ranges[ri] > ranges[li])
            line = np.interp(target, ranges, values)
            line[~inside] = 0.
            result[row, side*bins:(side+1)*bins] = line[::-1] if side == 0 else line
    return result


def normalize_feature_image(amplitude, support, scale=4):
    import cv2
    image = np.asarray(amplitude, float)
    support = np.asarray(support, bool)
    if support.sum() < 24:
        raise ValueError('Insufficient side-scan image support.')
    compressed = np.log1p(12*image)/np.log(13.)
    sortable = np.sort(np.where(support, compressed, np.inf), axis=0)
    count = support.sum(axis=0); col = np.arange(image.shape[1])
    gain = .5*(sortable[np.maximum((count-1)//2, 0), col]+sortable[np.maximum(count//2, 0), col])
    gain[count == 0] = 0; known = gain > 0
    if known.sum() < 2:
        raise ValueError('No feature texture.')
    gain = np.interp(col, col[known], gain[known])
    gain = cv2.GaussianBlur(gain[None], (0, 0), sigmaX=9.).ravel()
    normalized = np.zeros_like(image)
    normalized[support] = (compressed/np.maximum(gain[None], 1e-3))[support]
    lo, hi = np.percentile(normalized[support], [2, 98])
    if hi-lo < 1e-9:
        raise ValueError('No radiometric feature contrast.')
    u8 = np.rint(np.clip((normalized-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)
    u8 = cv2.resize(u8, (u8.shape[1], u8.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(support.astype(np.uint8)*255, (support.shape[1], support.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 2)).apply(u8), mask


def extract_features(image, mask, maximum=900):
    import cv2
    # A32px-high image is too anisotropic for generic SIFT detection. Detect
    # corners, refine their subpixel positions, then compute SIFT descriptors.
    corners = cv2.goodFeaturesToTrack(image, maxCorners=maximum, qualityLevel=.003,
        minDistance=4., mask=mask, blockSize=5)
    if corners is None:
        return np.empty((0, 2)), np.empty((0, 128), np.float32)
    h, w = image.shape
    pixels = corners[:, 0]
    keep = (pixels[:, 0] >= 3)&(pixels[:, 0] < w-3)&(pixels[:, 1] >= 3)&(pixels[:, 1] < h-3)
    pixels = pixels[keep].reshape(-1, 1, 2).copy()
    if not len(pixels):
        return np.empty((0, 2)), np.empty((0, 128), np.float32)
    cv2.cornerSubPix(image, pixels, (2, 2), (-1, -1),
        (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, .01))
    keypoints = [cv2.KeyPoint(float(x), float(y), 8.) for x, y in pixels[:, 0]]
    keypoints, descriptor = cv2.SIFT_create().compute(image, keypoints)
    if descriptor is None:
        return np.empty((0, 2)), np.empty((0, 128), np.float32)
    descriptor = descriptor.astype(np.float32)
    descriptor /= np.maximum(np.linalg.norm(descriptor, axis=1, keepdims=True), 1e-12)
    return np.asarray([k.pt for k in keypoints]), descriptor


def image_from_packet(packet, bins=512, scale=4):
    rb = np.minimum((packet.ranges_m/packet.max_range_m*bins).astype(int), bins-1)
    col = np.where(packet.sides == 0, bins-1-rb, bins+rb)
    image = np.zeros((8, 2*bins), float)
    np.maximum.at(image, (packet.rows, col), packet.amplitudes)
    u8, mask = normalize_feature_image(image, image > 0, scale)
    pixels, descriptors = extract_features(u8, mask)
    # Pixel-centre inverse of OpenCV resize. No nearest physical row or XYZ.
    x = packet.row_centres_m[0]+((pixels[:, 1]+.5)/scale-.5)*.15
    r = np.where(pixels[:, 0] < bins, bins-.5-pixels[:, 0],
                 -(pixels[:, 0]-bins+.5))*packet.max_range_m/bins
    return dict(image=u8, pixels=pixels, descriptors=descriptors, along_m=x,
                signed_slant_m=r, bins=bins, timestamp=packet.timestamp)


def descriptor_pairs(current, previous, ratio=.9):
    import cv2
    c, p = current['descriptors'], previous['descriptors']
    if min(len(c), len(p)) < 2:
        return np.empty((0, 2), int)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def select(a, b):
        result = {}
        for matches in matcher.knnMatch(a, b, k=2):
            if len(matches) == 2 and matches[0].distance < ratio*matches[1].distance:
                result[matches[0].queryIdx] = matches[0].trainIdx
        return result
    forward, reverse = select(c, p), select(p, c)
    return np.array([(i, j) for i, j in forward.items() if reverse.get(j) == i
                     and ((current['pixels'][i, 0] < current['bins']) ==
                          (previous['pixels'][j, 0] < previous['bins']))], int).reshape(-1, 2)


def robust_feature_row_motion(current, previous, predicted_along_m):
    pairs = descriptor_pairs(current, previous)
    if len(pairs) < 24:
        return None
    ci, pi = pairs.T
    delta = previous['along_m'][pi]-current['along_m'][ci]
    signed_slant = current['signed_slant_m'][ci]
    # Nuisance image shear absorbs yaw-related range-dependent deformation.
    # It is NOT converted into measured heading or ground/cross-track motion.
    design = np.column_stack([np.ones(len(delta)), signed_slant/100.])
    rng = np.random.default_rng(int(round(current['timestamp']*1e6)) & 0xffffffff)
    best = None
    for _ in range(256):
        indices = rng.choice(len(delta), 2, replace=False)
        if abs(signed_slant[indices[0]]-signed_slant[indices[1]]) < 1.:
            continue
        coefficient = np.linalg.solve(design[indices], delta[indices])
        residual = abs(delta-design@coefficient)
        # Same24 independent-equivalent observations; robust consensus uses
        # physical row resolution, not a purported centimetre XYZ precision.
        inliers = residual <= .15
        quality = int(inliers.sum())
        if best is None or quality > best[0]:
            best = (quality, inliers)
    if best is None or best[0] < 24:
        return None
    inliers = best[1]
    coefficient = np.linalg.lstsq(design[inliers], delta[inliers], rcond=None)[0]
    residual = delta[inliers]-design[inliers]@coefficient
    correction = float(coefficient[0]-predicted_along_m)
    correction /= 1+(correction/.08)**2
    return dict(correction_m=correction, measured_along_m=float(coefficient[0]),
        nuisance_slant_shear=float(coefficient[1]/100.), matches=len(pairs), inliers=best[0],
        variance_m2=.15**2+float(np.mean(residual**2))/min(24, best[0]),
        pairs=pairs, inlier_mask=inliers, descriptor_matching=True, optical_flow_used=False,
        MBES_XYZ_used=False, DVL_altitude_used=False)
