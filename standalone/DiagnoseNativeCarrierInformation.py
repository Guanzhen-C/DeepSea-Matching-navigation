"""Does local carrier invariance discard the recorded passband information?

An ideal synthetic frequency-shift control, NOT an actual noise identification
or acoustic calibration. Both complete native fields have exactly the same
envelope. Their frequencies occupy opposite known processing bands. If feature
invariance makes them identical, that family cannot itself encode this band
prior. This alone does NOT establish the cause of real-image failures.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from LocalRelativeSonarFeatures import local_features
from ProbeNativeLocalSonarTransfer import load_selected
from TrainNativeLocalSonarWavelet import sha,array_sha


def ideal_fields():
    h,w=64,2048;y=np.arange(h)[:,None];x=np.arange(w)[None,:]
    amplitude=(.7+.2*np.cos(2*np.pi*3*y/h))*(.8+.15*np.cos(2*np.pi*7*x/w))
    original=amplitude*np.exp(-2j*np.pi*490*x/w)
    shifted=original*np.exp(2j*np.pi*980*x/w)
    return original,shifted,37650.,[-15000.,-3000.]


def inband_fraction(z,fs,band):
    frequency=np.fft.fftfreq(z.shape[1],1/fs)
    power=np.abs(np.fft.fft(z,axis=1))**2
    selected=(frequency>=band[0])&(frequency<=band[1])
    return float(power[:,selected].sum()/power.sum())


def controls():
    a,b,fs,band=ideal_fields()
    f,_,mask=local_features(a);g,_,mask2=local_features(b)
    if not np.array_equal(mask,mask2):raise ValueError('ideal shifted support differs')
    unit_a=a/abs(a);unit_b=b/abs(b)
    ga=unit_a[:,1:]*unit_a[:,:-1].conj();gb=unit_b[:,1:]*unit_b[:,:-1].conj()
    result=dict(synthetic_native_shape=list(a.shape),all_synthetic_native_values_included=True,
        sampling_frequency_Hz=fs,recorded_reference_subband_Hz=band,
        tone_Hz_before=-490/2048*fs,tone_Hz_after=490/2048*fs,
        ideal_inband_energy_fraction_before=inband_fraction(a,fs,band),
        ideal_inband_energy_fraction_after=inband_fraction(b,fs,band),
        max_absolute_envelope_roundoff=float(np.max(abs(abs(a)-abs(b)))),
        finite_relative_features_max_absolute_difference=float(np.max(abs(f-g))),
        raw_adjacent_unit_phase_gradient_RMS_difference=float(np.sqrt(np.mean(abs(ga-gb)**2))),
        original_features_SHA256=array_sha(f),shifted_features_SHA256=array_sha(g),
        complete_synthetic_field_SHA256=[array_sha(a),array_sha(b)],
        test_is_synthetic_frequency_information_control_not_real_noise_calibration=True,
        current_real_failure_cause_proved=False)
    return (f,g,mask),result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve existing frequency information control')
    torch.set_num_threads(2);model,selected=load_selected(args.training,torch.device('cpu'))
    (a,b,mask),result=controls();predictions=[]
    with torch.inference_mode():
        for field in (a,b):
            delta=model.feature_residual(torch.from_numpy(field[None]),torch.from_numpy(mask[None,None]),use_phase=True)
            predictions.append(delta.numpy())
    result.update(status='completed_ideal_full_field_carrier_information_control_not_device_noise_identification',
        selected_checkpoint=selected,model_prediction_max_absolute_log_delta_difference=float(np.max(abs(predictions[0]-predictions[1]))),
        training_report_SHA256=sha(args.training/'run_summary.json'),source_code_SHA256=sha(Path(__file__)),
        implication='Invariant features cannot distinguish these opposite-band ideal controls; retaining calibrated frequency context is a testable next hypothesis, not a proven real-data fix')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
