"""Independent complete reserved-native replay, not real noise acceptance.

Re-reads every reserved side/beam from original MAT, regenerates each complete
known added-noise field and replays frozen, validation-selected checkpoints.
Checks every output/identity-delta SHA and full native metrics, not just JSON
flags, thumbnail similarity, a subset of range bins or only file headers.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import os
import time

import numpy as np
import torch

from LocalRelativeSonarFeatures import PROTOCOL
from NativeLocalSonarWavelet import NativeLocalSonarWavelet
from TrainNativeLocalSonarWavelet import sha,extract,make_noisy,evaluate,aggregate


def check_training_ledger(report,manifest,path):
    entries=[json.loads(line) for line in path.read_text().splitlines()]
    if sha(path)!=report['training_input_ledger_file_SHA256'] or len(entries)!=report['actual_training_examples']:
        raise ValueError('complete shared-input training ledger differs')
    if len(entries)!=report['steps_completed']*2:
        raise ValueError('two shared native examples per training step required')
    exposures=Counter()
    for ordinal,item in enumerate(entries):
        if (item['step'],item['slot'])!=(ordinal//2+1,ordinal%2):
            raise ValueError('training step/slot sequence differs')
        index=item['source_index'];beam=item['beam']
        if index not in report['train_source_indices'] or not 0<=beam<8:
            raise ValueError('reserved native physical source leaked into training')
        source=manifest['sources'][index]
        if source['split']!='train_known_extra_noise':raise ValueError('reserved physical part appeared in training')
        top,left,bottom,right=item['context'];cy,cx,ch,cw=item['core']
        if (not 0<=top<bottom<=source['native_records'] or not 0<=left<right<=source['native_range_bins']
                or top%4 or left%4 or min(cy,cx)<0 or ch!=64 or cw!=256
                or cy+ch>bottom-top or cx+cw>right-left):
            raise ValueError('training context/core was not on the original native grid')
        for key in ('features_SHA256','noisy_log_SHA256','target_log_SHA256','support_SHA256'):
            if len(item[key])!=64:raise ValueError('complete exact input feature identities required')
        exposures[f'{index}:{beam}']+=1
    if dict(exposures)!=report['source_exposure_counts'] or len(exposures)!=48:
        raise ValueError('all 48 training side/beam sources must be represented without split leakage')
    return dict(complete_training_entries_checked=len(entries),train_side_beam_sources_checked=len(exposures),
                training_source_split_leakage_found=False,
                original_training_feature_values_recomputed=False)


def check_reserved_coverage(cases,report,manifest):
    expected={(i,b) for i in report['reserved_source_indices'] for b in range(8)}
    actual={(c['source_index'],c['beam_0based']) for c in cases}
    if len(cases)!=16 or actual!=expected:raise ValueError('all sixteen reserved native beam fields required')
    for case in cases:
        source=manifest['sources'][case['source_index']]
        if (case['native_values']!=source['native_records']*source['native_range_bins']
                or source['split']!='validation_known_extra_noise'
                or not case['source_observation_is_not_clean_truth']
                or not case['missing_support_delta_exact_zero']):
            raise ValueError('complete native reserved field coverage or observation semantics changed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','manifest','h5-python'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    root=args.training.absolute();report=json.loads((root/'run_summary.json').read_text())
    manifest=json.loads(args.manifest.read_text())
    if (report['status']!='completed_same_device_known_added_noise_pair_real_denoising_not_accepted'
            or report['smoke'] or report['steps_completed']!=report['steps_requested']
            or report['model_family']!=NativeLocalSonarWavelet.family or report['feature_protocol']!=PROTOCOL
            or not report['all_train_side_beam_sources_seen']):
        raise ValueError('completed full native known-noise experiment required')
    if sha(args.manifest)!=report['manifest_SHA256']:raise ValueError('original qualified manifest changed')
    for name,digest in report['source_code_SHA256'].items():
        if sha(root/'source_snapshot'/name)!=digest or sha(Path(__file__).with_name(name))!=digest:
            raise ValueError('exact frozen execution implementation required: '+name)
    for source in manifest['sources']:
        path=Path(source['source_MAT']);stat=path.stat()
        if {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}!=source['source_guard']:
            raise ValueError('original source guard changed')
        if sha(source['parameter_file'])!=source['parameter_SHA256']:
            raise ValueError('original source parameters changed')
    ledger=check_training_ledger(report,manifest,root/'shared_training_inputs.jsonl')
    for name in ('amplitude_only','phase_context'):
        history=report['validation_history']
        expected=min(history,key=lambda r:(r['conditions'][name]['selection_score'],r['step']))['step']
        chosen=report['selected_by_full_native_known_noise_validation'][name]
        if expected!=chosen['step']:raise ValueError('checkpoint was not selected by the declared full-field score')
        for item in report['checkpoints'][name]:
            if sha(root/item['file'])!=item['SHA256']:raise ValueError('saved checkpoint changed')
        check_reserved_coverage(report['postselection_complete_native_audit'][name]['cases'],report,manifest)
    audit_path=root/'independent_complete_native_replay_audit.json'
    if audit_path.exists():raise FileExistsError('preserve existing independent native replay')
    start=time.monotonic();audit=dict(status='replaying_all_16_original_reserved_beam_fields',PID=os.getpid(),
        **ledger,case_pairs_replayed=0,native_complex_values_replayed=0,
        original_source_guards_and_parameters_verified=True,
        source_run_summary_SHA256=sha(root/'run_summary.json'),auditor_SHA256=sha(Path(__file__)),
        private_data_uploaded=False,real_clean_ground_truth_available=False,quality_accepted=False,
        selected_amplitude_step=report['selected_by_full_native_known_noise_validation']['amplitude_only']['step'],
        selected_phase_step=report['selected_by_full_native_known_noise_validation']['phase_context']['step'])
    def save():
        audit['elapsed_s']=time.monotonic()-start
        tmp=audit_path.with_suffix('.tmp');tmp.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');tmp.replace(audit_path)
    save();torch.set_num_threads(2);device=torch.device(args.device)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.4,device)
    models={}
    profile=json.loads((root/'aggregate_appearance_profile.json').read_text())
    if sha(root/'aggregate_appearance_profile.json')!=report['profile_SHA256']:raise ValueError('noise profile changed')
    try:
        for name,use_phase in (('amplitude_only',False),('phase_context',True)):
            chosen=report['selected_by_full_native_known_noise_validation'][name]
            state=torch.load(root/chosen['file'],map_location=device,weights_only=True)
            if (state['model_family']!=report['model_family'] or state['feature_protocol']!=PROTOCOL
                    or state['source_code_SHA256']!=report['source_code_SHA256'] or state['use_phase']!=use_phase):
                raise ValueError('selected model/input protocol differs')
            model=NativeLocalSonarWavelet(state['width']).to(device);model.load_state_dict(state['model'],strict=True)
            models[name]=(model,use_phase)
        replayed={name:[] for name in models}
        index={(c['source_index'],c['beam_0based']):ordinal for ordinal,c in enumerate(
            report['postselection_complete_native_audit']['phase_context']['cases'])}
        for (source_index,beam),ordinal in index.items():
            saved=report['postselection_complete_native_audit']['phase_context']['cases'][ordinal]
            source=manifest['sources'][source_index];box=(0,0,source['native_records'],source['native_range_bins'])
            original,meta=extract(args.h5_python,args.manifest,source_index,box,beam=beam)
            original=np.asarray(original[0],np.complex64)
            noisy,parameters=make_noisy(original,profile,device,saved['seed'],saved['noise_scale'])
            fields=[dict(original=original,noisy=noisy,source_index=source_index,beam=beam,
                         seed=saved['seed'],scale=saved['noise_scale'])]
            actual=evaluate(models,fields,device)
            for name in models:
                case=actual[name]['cases'][0];case['extraction']=meta
                case['noise_parameters_SHA256']=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()
                expected=report['postselection_complete_native_audit'][name]['cases'][ordinal]
                if case!=expected:raise ValueError(f'exact complete native prediction/metric replay differs: {name}/{source_index}/{beam}')
                replayed[name].append(case)
            audit['case_pairs_replayed']+=1;audit['native_complex_values_replayed']+=original.size
            save();print(json.dumps(dict(source_index=source_index,beam=beam,checked_native_values=original.size)),flush=True)
        for name,cases in replayed.items():
            if aggregate(cases)!=report['postselection_complete_native_audit'][name]:
                raise ValueError('complete native aggregate metrics differ')
        audit.update(status='passed_complete_16_beam_native_predictions_and_known_noise_replay_not_real_quality_acceptance',
            all_32_model_observation_case_outputs_and_identity_SHA256_exact=True,
            complete_metrics_exact=True,source_report_unchanged=sha(root/'run_summary.json')==audit['source_run_summary_SHA256'])
    except BaseException as exc:
        audit.update(status='failed_preserved_independent_native_replay_no_automatic_restart',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()


if __name__=='__main__':main()
