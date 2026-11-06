"""Control the REALIZED synthetic log-modulation span after envelope shaping.

The source reference is a phase-folded candidate statistic and can include
terrain. Matching a synthetic field's robust span is an appearance-domain
randomization choice, not exact electronic-noise/radiometric calibration.
"""
import torch
from AcousticStripeAugmentation import acoustic_stripes


def empirical_stripes(target,support,generator,profile,identity_probability=.2):
    if target.ndim!=4 or target.shape[:2]!=(1,1) or support.shape!=target.shape:
        raise ValueError('one native log-amplitude plane and positive support required')
    damaged,meta=acoustic_stripes(target.repeat(1,3,1,1),1,generator,profile,identity_probability)
    if meta['identity']:
        return target.clone(),dict(meta,strength_protocol='realized_robust_log_span_v1')
    field=damaged[:,0:1]-target
    values=field[support]
    if values.numel()<64:raise ValueError('insufficient observed support for robust synthetic span')
    # torch.quantile cannot sort arbitrary full-native multi-million-pixel
    # fields. Sample only this synthetic scaling statistic, never resize or
    # omit measured values from inference or output.
    stride=max(1,(values.numel()+65535)//65536)
    probe=values[::stride]
    quantiles=torch.quantile(probe,probe.new_tensor([.05,.95]))
    before=(quantiles[1]-quantiles[0])/2
    if not torch.isfinite(before) or float(before)<=1e-10:
        raise ValueError('degenerate generated stripe field cannot be calibrated')
    # Keep all spatial band coordinates, widths, grains and shape randomness
    # identical. Only realized modulation magnitude is being controlled.
    severity=.75+.75*(meta['severity']-.75)/2.25
    desired=meta['estimated_log_depth']*severity
    factor=desired/before
    delta=field*factor
    result=torch.where(support,target+delta,target)
    actual=delta[support]
    measured=torch.quantile(actual[::stride],actual.new_tensor([.05,.95]))
    return result,dict(meta,strength_protocol='realized_robust_log_span_v1',
        pre_calibration_robust_half_span=float(before),requested_robust_half_span=desired,
        achieved_robust_half_span=float((measured[1]-measured[0])/2),
        empirical_strength_scale=float(factor),empirical_severity=severity,
        robust_span_diagnostic_values=probe.numel(),observed_native_field_values=values.numel(),
        synthetic_span_is_sample_estimate=probe.numel()!=values.numel(),
        geometry_randomization_unchanged=True,original_zero_support_preserved=True,
        statistic_caveat='source phase-folded candidate span vs synthetic field span; appearance approximation, not noise truth')
