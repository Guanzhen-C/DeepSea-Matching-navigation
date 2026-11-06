"""Recover the unfinished ALL-16-beam/2-law test from preserved selected weights.

Runs only on the user-authorized 172 host. No training resumes or original
reports are rewritten. All six models see the same realized remote synthetic
fields; device/software/RNG identity is recorded rather than claiming cross-
device bit equality without evidence. Native reference bytes are exact copies.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
import torch

from MatchedSonarSignal import CONDITIONS
from MatchedSonarWavelet import MatchedSonarWavelet
from BandAnchoredNativeSonarWavelet import BandAnchoredNativeSonarWavelet
from BandAnchoredNativeSonarFeatures import PROTOCOL
from TrainMatchedSonarFactorial import prepare_fields,evaluate
from TrainNativeLocalSonarWavelet import sha,aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke',action='store_true');args=p.parse_args();root=args.bundle.absolute()
    bundle=json.loads((root/'bundle_manifest.json').read_text())
    if bundle['user_authorized_host']!='172.18.80.26':raise ValueError('explicit authorized destination required')
    for name,digest in bundle['files'].items():
        if sha(root/name)!=digest:raise ValueError('immutable transferred code/report/checkpoint changed: '+name)
    out=root/('remote_validation_smoke' if args.smoke else 'remote_full_native_validation')
    out.mkdir(exist_ok=False);start=time.monotonic()
    report=dict(status='loading_preserved_weights_for_remote_validation_only',PID=os.getpid(),smoke=args.smoke,
        source_bundle_manifest_SHA256=sha(root/'bundle_manifest.json'),source_training_report_SHA256=bundle['source_training_report_SHA256'],
        source_local_process_terminated=True,original_local_process_exit_code=137,training_restarted=False,
        original_reports_rewritten=False,user_authorized_private_host='172.18.80.26',
        source_code_SHA256=sha(Path(__file__)),native_receipts=[],results_partial={},physical_beams_completed=0,
        models_in_same_test_observe_identical_realized_inputs=True,
        cross_device_noise_realizations_not_assumed_bitwise_identical=True,
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
            CUDA=str(torch.version.cuda),device=str(args.device)),quality_accepted=False)
    def save():
        report['elapsed_s']=time.monotonic()-start
        tmp=out/'run_summary.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');tmp.replace(out/'run_summary.json')
    save();torch.set_num_threads(4);device=torch.device(args.device)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.65,device)
    models={};profile=json.loads((root/'aggregate_appearance_profile.json').read_text())
    for name,item in bundle['selected_weights'].items():
        state=torch.load(root/item['path'],map_location=device,weights_only=True)
        if state['feature_protocol']!=PROTOCOL:raise ValueError('trained phase feature protocol differs')
        if name=='frozen_prior_anchored_raw':
            if state['model_family']!=BandAnchoredNativeSonarWavelet.family:raise ValueError('frozen prior family differs')
            model=BandAnchoredNativeSonarWavelet(state['width']).to(device);front=False
        else:
            condition=CONDITIONS[name]
            if (state['model_family']!=MatchedSonarWavelet.family or state['condition']!=condition
                    or state['condition_name']!=name):raise ValueError('trained preprocessing/aggregation differs')
            model=MatchedSonarWavelet(state['width'],aggregation=condition['aggregation']).to(device);front=condition['frontend']
        model.load_state_dict(state['model'],strict=True);models[name]=(model.eval(),front)
    accumulated={name:[] for name in models};report['status']='validating_full_native_frozen_models_on_172';save()
    try:
        sources=bundle['native_sources'][:1] if args.smoke else bundle['native_sources']
        for source in sources:
            i=source['source_index'];receipt_path=root/'native_arrays'/f'source_{i:02d}_receipt.json'
            receipt=json.loads(receipt_path.read_text());path=root/'native_arrays'/f'source_{i:02d}_complex128.npy'
            values=np.load(path,mmap_mode='r')
            if list(values.shape)!=source['shape'] or values.dtype!=np.complex128:
                raise ValueError('complete transported native sampling grid differs')
            digest=hashlib.sha256(memoryview(values).cast('B')).hexdigest()
            if digest!=receipt['source_payload_SHA256']:raise ValueError('all original complex128 bytes differ')
            report['native_receipts'].append(dict(source_index=i,payload_SHA256=digest,receipt_SHA256=sha(receipt_path)))
            for beam in (range(1) if args.smoke else range(8)):
                original=np.asarray(values[beam],np.complex64)
                if args.smoke:original=original[:64,:768].copy()
                fields=prepare_fields(original,profile,device,i,beam,9189000+i*10+beam,.35)
                measured=evaluate(models,fields,device)
                for name,metrics in measured.items():
                    for case in metrics['cases']:
                        case['original_source_side_payload_SHA256']=digest
                        case['original_source_receipt_SHA256']=sha(receipt_path)
                        accumulated[name].append(case)
                report['physical_beams_completed']+=1
                report['results_partial']={name:aggregate(cases) for name,cases in accumulated.items()};save()
                print(json.dumps(dict(source_index=i,beam=beam,noise_laws=2,models=len(models),elapsed_s=report['elapsed_s'])),flush=True)
            del values
        expected=1 if args.smoke else 16
        if report['physical_beams_completed']!=expected:raise ValueError('native beam coverage incomplete')
        report.update(status='completed_remote_selected_weights_full_native_validation_not_real_quality_acceptance',
            results=report['results_partial'],results_partial={},physical_beams_expected=expected,
            full_native_values_per_model=sum(c['native_values'] for c in next(iter(accumulated.values()))))
    except BaseException as exc:
        report.update(status='failed_preserved_remote_native_validation_no_automatic_restart',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if device.type=='cuda':report['peak_CUDA_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        save()


if __name__=='__main__':main()
