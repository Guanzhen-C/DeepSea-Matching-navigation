"""Complete independent fixed-band experiment replay with bounded side reads.

Checks the entire matched training ledger and every native value of all sixteen
reserved fields, not only a validation crop. Complete-side stream optimization
is independently checked against the old single-beam byte/provenance protocol.
This is reproducibility, not unknown clean real-sonar quality acceptance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from BandAnchoredNativeSonarFeatures import PROTOCOL
from BandAnchoredNativeSonarWavelet import BandAnchoredNativeSonarWavelet
from TrainBandAnchoredNativeSonar import read,verified_comparison,evaluate_model
from TrainNativeLocalSonarWavelet import sha,make_noisy,aggregate
from AuditNativeLocalSonarExperiment import check_reserved_coverage
from StreamNativeProcessedSide import stream_side,beam_view


def check_matched_ledger(report,previous,root,comparison):
    original=(comparison/'shared_training_inputs.jsonl').read_text().splitlines()
    new=(root/'matched_training_inputs.jsonl').read_text().splitlines()
    if len(new)!=len(original) or len(new)!=report['actual_training_examples']:
        raise ValueError('complete prior-budget training ledger required')
    if sha(root/'matched_training_inputs.jsonl')!=report['matched_training_input_file_SHA256']:
        raise ValueError('matched input ledger file changed')
    digest=hashlib.sha256()
    for before,after in zip(original,new):
        a=json.loads(before);b=json.loads(after)
        if (b.pop('comparison_relative_features_SHA256')!=a['features_SHA256']
                or len(b.pop('amplitude_feature_SHA256'))!=64 or len(b['features_SHA256'])!=64):
            raise ValueError('only explicitly identified phase feature replacement allowed')
        b['features_SHA256']=a['features_SHA256']
        if a!=b:raise ValueError('native source/identity/noise/labels/coordinates differ')
        digest.update((json.dumps(b,ensure_ascii=False,sort_keys=True)+'\n').encode())
    if (digest.hexdigest()!=previous['training_input_ledger_file_SHA256']
            or digest.hexdigest()!=report['comparison_input_ledger_reconstructed_SHA256']):
        raise ValueError('reconstructed complete old input ledger does not match')
    return len(new)


def load_selected(root,device):
    report=read(root/'run_summary.json')
    if (report['status']!='completed_matched_fixed_band_phase_experiment_real_denoising_unaccepted'
            or report['smoke'] or report['feature_protocol']!=PROTOCOL
            or report['model_family']!=BandAnchoredNativeSonarWavelet.family
            or not report['full_comparison_training_input_ledger_exact']):
        raise ValueError('completed full-budget matched native experiment required')
    for name,digest in report['source_code_SHA256'].items():
        if sha(root/'source_snapshot'/name)!=digest or sha(Path(__file__).with_name(name))!=digest:
            raise ValueError('frozen execution implementation differs: '+name)
    selected=report['selected_by_complete_native_validation']
    expected=min(report['validation_history'],key=lambda r:(r['metrics']['selection_score'],r['step']))['step']
    if selected['step']!=expected:raise ValueError('declared whole-native validation selection differs')
    for item in report['checkpoints']:
        if sha(root/item['file'])!=item['SHA256']:raise ValueError('native checkpoint changed')
    state=torch.load(root/selected['file'],map_location=device,weights_only=True)
    if (state['model_family']!=report['model_family'] or state['feature_protocol']!=PROTOCOL
            or state['source_code_SHA256']!=report['source_code_SHA256'] or state['use_phase'] is not True):
        raise ValueError('selected native model/protocol differs')
    model=BandAnchoredNativeSonarWavelet(state['width']).to(device);model.load_state_dict(state['model'],strict=True)
    return model.eval(),report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','manifest','h5-python'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    root=args.training.absolute();report=read(root/'run_summary.json');comparison=Path(report['comparison_root'])
    previous,manifest=verified_comparison(comparison,args.manifest)
    entries=check_matched_ledger(report,previous,root,comparison)
    check_reserved_coverage(report['postselection_complete_native_audit']['cases'],previous,manifest)
    out=root/'independent_complete_native_replay_audit.json'
    if out.exists():raise FileExistsError('preserve existing complete native replay')
    start=time.monotonic();audit=dict(status='replaying_fixed_band_all_16_native_fields',PID=os.getpid(),
        matched_training_entries_checked=entries,case_pairs_replayed=0,native_complex_values_replayed=0,
        source_run_summary_SHA256=sha(root/'run_summary.json'),auditor_SHA256=sha(Path(__file__)),
        stream_reader_SHA256=sha(Path(__file__).with_name('StreamNativeProcessedSide.py')),
        source_side_reads=[],private_data_uploaded=False,quality_accepted=False)
    def save():
        audit['elapsed_s']=time.monotonic()-start
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');tmp.replace(out)
    save();torch.set_num_threads(2);device=torch.device(args.device)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.4,device)
    try:
        model,report=load_selected(root,device);profile=read(root/'aggregate_appearance_profile.json')
        if sha(root/'aggregate_appearance_profile.json')!=report['profile_SHA256']:raise ValueError('noise profile changed')
        cases={(c['source_index'],c['beam_0based']):c for c in report['postselection_complete_native_audit']['cases']}
        replayed=[]
        for i in previous['reserved_source_indices']:
            source=manifest['sources'][i];box=(0,0,source['native_records'],source['native_range_bins'])
            values,side_meta=stream_side(args.h5_python,args.manifest,i,box)
            audit['source_side_reads'].append(side_meta);save()
            for beam in range(8):
                original,meta=beam_view(values,side_meta,beam);original=np.asarray(original[0],np.complex64)
                expected=cases[(i,beam)]
                noisy,parameters=make_noisy(original,profile,device,expected['seed'],expected['noise_scale'])
                field=dict(original=original,noisy=noisy,source_index=i,beam=beam,
                           seed=expected['seed'],scale=expected['noise_scale'])
                case=evaluate_model(model,[field],device)['cases'][0];case['extraction']=meta
                case['noise_parameters_SHA256']=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()
                if case!=expected:raise ValueError(f'complete native prediction/identity/provenance/metric replay differs: {i}/{beam}')
                replayed.append(case);audit['case_pairs_replayed']+=1;audit['native_complex_values_replayed']+=original.size
                save();print(json.dumps(dict(source_index=i,beam=beam,checked_native_values=original.size)),flush=True)
            del values
        if aggregate(replayed)!=report['postselection_complete_native_audit']:
            raise ValueError('complete native aggregate replay differs')
        audit.update(status='passed_all_16_native_fixed_band_outputs_identity_and_metrics_replay_not_real_quality_acceptance',
            all_native_prediction_and_identity_SHA256_exact=True,all_metrics_exact=True,
            single_beam_metadata_derived_from_lossless_complete_side_payload=True,
            source_report_unchanged=sha(root/'run_summary.json')==audit['source_run_summary_SHA256'])
    except BaseException as exc:
        audit.update(status='failed_preserved_fixed_band_native_replay_no_automatic_restart',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()


if __name__=='__main__':main()
