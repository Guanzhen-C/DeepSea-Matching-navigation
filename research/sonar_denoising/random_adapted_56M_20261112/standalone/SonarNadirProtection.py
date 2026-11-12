"""Protect the original display-black component connected to the nadir.

This is an explicit augmentation support policy, not a physical water/terrain
classifier. Isolated dark seabed pixels are not globally threshold-masked.
Every affected input sample is excluded BEFORE synthetic noise is added.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage

from NativeSonarPaletteStream import decoded_index_rows


def mosaic_seed_mask(native_protected):
    a = np.asarray(native_protected)
    if a.ndim != 3 or a.shape[0] != 2 or a.dtype != np.bool_:
        raise ValueError('Both per-ping protected range masks required.')
    _, records, bins = a.shape
    result = np.empty((4 * (records + 1), 2 * bins), bool)
    for p in range(records + 1):
        protected = a[:, 0] if p == 0 else a[:, -1] if p == records else a[:, p - 1] & a[:, p]
        row = np.concatenate((protected[0, ::-1], protected[1]))
        result[4 * p:4 * p + 4] = row
    return result


def protection_from_black(black, source_records, range_bins, *, seed_mask=None):
    a = np.asarray(black)
    expected = (4 * (source_records + 1), 2 * range_bins)
    if a.dtype != np.bool_ or a.shape != expected:
        raise ValueError("Complete native waterfall black-mask geometry required.")
    seed = np.zeros_like(a)
    seed[:, range_bins - 1:range_bins + 1] = a[:, range_bins - 1:range_bins + 1]
    if seed_mask is not None:
        external = np.asarray(seed_mask)
        if external.dtype != np.bool_ or external.shape != a.shape:
            raise ValueError('Aligned observed-water seed mask required.')
        seed |= external & a
    component = ndimage.binary_propagation(seed, mask=a,
                                          structure=ndimage.generate_binary_structure(2, 1))
    if not component.any():
        raise ValueError("No original black nadir component; require explicit review.")
    # A physical ping's eight simultaneous beams occupy groups p and p+1 in
    # the original four-row overlap display. Protect both contributing groups.
    masks = []
    for side in (component[:, :range_bins][:, ::-1], component[:, range_bins:]):
        groups = side.reshape(source_records + 1, 4, range_bins).any(axis=1)
        masks.append(groups[:-1] | groups[1:])
    return np.stack(masks), component


def from_reference_png(path, palette, source_records, range_bins, *, seed_mask=None, maximum_dark_index=8):
    if not 0 <= maximum_dark_index <= 8:
        raise ValueError('Only original near-black palette entries0..8 are supported, not arbitrary dark terrain thresholds.')
    black = np.empty((4 * (source_records + 1), 2 * range_bins), bool)
    rows = 0
    for rows, row in enumerate(decoded_index_rows(path, palette), 1):
        if rows > len(black) or len(row) != black.shape[1]:
            raise ValueError("Original native reference dimensions changed.")
        black[rows - 1] = row <= maximum_dark_index
    if rows != len(black):
        raise ValueError("Incomplete reference black component.")
    masks, component = protection_from_black(black, source_records, range_bins, seed_mask=seed_mask)
    evidence = dict(policy="original_near_black_component_connected_to_nadir_and_observed_water_seeds",
                    maximum_original_palette_index=maximum_dark_index,
                    recorded_range_water_seed_used=seed_mask is not None,
                    original_black_component_pixels=int(component.sum()),
                    protected_native_record_range_cells=[int(v.sum()) for v in masks],
                    eight_simultaneous_beam_and_overlap_provenance_used=True,
                    all_dark_pixels_globally_masked=False,
                    output_pixels_forced_black=False,
                    physical_water_ground_truth_claimed=False)
    return masks, component, evidence
