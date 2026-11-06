"""Experimental boundary gain continuation, plus differential common-mode removal.

This is a statistical amplitude correction, NOT recovered RF calibration or a
new neural model. The frozen baseline is never overwritten. No filtering is
applied to raw I/Q; original phase and sample coordinates can be retained.
Adjacent range cells are assumed to have locally comparable slow gain and
common along-record scene variation. Real features violating that assumption
can be changed, so real-data output remains a candidate pending review.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


PROTOCOL = 'far384_gain_background_continuation_differential_commonmode_v1'
CONTEXT = 1024
PATCH = 512
REPLACE = 384
RAMP = 256
SIGMA_RECORDS = 8.


def smooth_parts(field, parts):
    """No mixing across original acquisition file boundaries."""
    x = np.asarray(field, np.float64)
    if (x.ndim not in (1, 2) or not np.isfinite(x).all() or not parts
            or any(not isinstance(n, (int, np.integer)) or n < 2 for n in parts)
            or sum(parts) != len(x)):
        raise ValueError('Complete finite physical-record field and acquisition partitions required.')
    out = np.empty_like(x)
    first = 0
    for n in parts:
        out[first:first+n] = gaussian_filter1d(x[first:first+n], SIGMA_RECORDS,
                                              axis=0, mode='reflect', truncate=4.)
        first += n
    return out


def repair(old_gain, amplitude, parts, *, full_width, valid_end):
    """Return two last-512 float32 gain fields from the last-1024 context.

    Fully active interior reference: [end-768, end-512). Modified cells:
    (end-384, end). Smoothstep reaches one at end-128. Neither reference nor
    statistics includes zero padding. A single explicit common endpoint is
    required; heterogeneous or interior endpoints need a separate protocol.
    """
    old, a = np.asarray(old_gain), np.asarray(amplitude)
    if (old.dtype != np.float32 or a.dtype != np.float64 or old.ndim != 2
            or old.shape != a.shape or old.shape[1] != CONTEXT
            or not np.isfinite(old).all() or not np.isfinite(a).all() or (a < 0).any()
            or not isinstance(full_width, int) or full_width < 2*CONTEXT
            or not isinstance(valid_end, int) or not full_width-64 <= valid_end <= full_width):
        raise ValueError('Finite native F32 gain/F64 amplitude, full width and explicit near-array endpoint required.')
    end = valid_end - (full_width-CONTEXT)
    if not np.all(a[:, end:] <= 1e-12) or not np.all(a[:, end-REPLACE:end] > 1e-12):
        raise ValueError('Do not fill missing cells or infer a new endpoint from rendered brightness.')
    ref = slice(end-768, end-512)
    if not np.all(a[:, ref] > 1e-12):
        raise ValueError('Complete observed interior reference required.')
    original = old[:, -PATCH:]
    source = a[:, -PATCH:]
    columns = np.arange(full_width-PATCH, full_width)
    u = np.clip((columns-(valid_end-REPLACE))/RAMP, 0., 1.)
    w = u*u*(3.-2.*u)
    w[columns >= valid_end] = 0.
    # Continue the model's slowly varying log-amplitude gain, not rendered
    # image intensities. This deliberately changes the old boundary DC ramp;
    # the previous zero-DC adapter could not remove that artifact by design.
    reference_gain = smooth_parts(np.median(old[:, ref], axis=1), parts)
    baseline_slow = smooth_parts(original, parts)
    gain_only = original.astype(np.float64)+(reference_gain[:, None]-baseline_slow)*w
    gain_only[source <= 1e-12] = original[source <= 1e-12]
    # Estimate EXCESS temporal common-mode at the edge relative to the same
    # physical beam's trusted interior. Range-localised targets do not dominate
    # a median over 128/256 cells; shared scene variation largely cancels.
    # This is a structural prior, not a proof that the difference is all noise.
    qref = np.log(a[:, ref])+old[:, ref].astype(np.float64)
    qedge = np.log(a[:, end-128:end])+gain_only[:, end-128-(CONTEXT-PATCH):end-(CONTEXT-PATCH)]
    contrast = np.median(qedge, axis=1)-np.median(qref, axis=1)
    excess = contrast-smooth_parts(contrast, parts)
    combined = gain_only-excess[:, None]*w
    combined[source <= 1e-12] = original[source <= 1e-12]
    outputs = {'gain_continuity': gain_only.astype(np.float32),
               'gain_and_commonmode': combined.astype(np.float32)}
    for value in outputs.values():
        if (not np.isfinite(value).all() or not np.array_equal(value[:, w == 0], original[:, w == 0])
                or not np.array_equal(value[source <= 1e-12], original[source <= 1e-12])):
            raise ValueError('Protected values or explicit processing support changed.')
    return outputs, dict(protocol=PROTOCOL, context_columns=CONTEXT, output_columns=PATCH,
        modified_valid_columns=REPLACE, smoothstep_width=RAMP,
        reference_native_columns=[valid_end-768, valid_end-512],
        edge_commonmode_native_columns=[valid_end-128, valid_end],
        temporal_background_sigma_physical_records=SIGMA_RECORDS, original_parts=list(parts),
        reference_slow_gain_range=[float(reference_gain.min()), float(reference_gain.max())],
        commonmode_excess_RMS=float(np.sqrt(np.mean(excess**2))),
        RF_filtering=False, interpolation=False, cropping=False, gain_floor=False,
        new_training_or_neural_inference=False, statistical_radiometric_correction=True,
        original_boundary_DC_intentionally_changed=True, protected_zero_tail_retained=True,
        NOT_noise_identification_or_clean_truth=True)
