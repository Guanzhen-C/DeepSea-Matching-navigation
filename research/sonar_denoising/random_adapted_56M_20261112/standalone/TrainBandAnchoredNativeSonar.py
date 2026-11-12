"""Replay the prior 800-example native experiment changing ONLY phase features.

Actually trains the new phase model; DOES NOT silently reinterpret trained
relative-feature weights. Reuses a checked step-zero state and every prior
source/record/context/noise/identity/label. The amplitude-only control ignores
the changed channels exactly, so its completed prior training/selection is
reused explicitly rather than reporting it as another newly executed training.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import os
import random
import shutil
import time

import numpy as np
import torch

from AuditNativeLocalSonarExperiment import check_training_ledger,check_reserved_coverage
from LocalRelativeSonarFeatures import local_features
from BandAnchoredNativeSonarFeatures import band_features,PROTOCOL,REFERENCE_FS,REFERENCE_BAND
from BandAnchoredNativeSonarWavelet import BandAnchoredNativeSonarWavelet,infer_band_native
from TrainNativeLocalSonarWavelet import sha,array_sha,extract,make_noisy,fidelity_loss,metrics,aggregate


def read(path):return json.loads(Path(path).read_text())


def verified_comparison(root,manifest_path):
    previous=read(root/'run_summary.json');manifest=read(manifest_path)
    audit=read(root/'independent_complete_native_replay_audit.json')
    if (previous['status']!='completed_same_device_known_added_noise_pair_real_denoising_not_accepted'
            or previous['smoke'] or audit['status']!='passed_complete_16_beam_native_predictions_and_known_noise_replay_not_real_quality_acceptance'
            or audit['source_run_summary_SHA256']!=sha(root/'run_summary.json')
            or previous['manifest_SHA256']!=sha(manifest_path)):
        raise ValueError('completed independently replayed native comparison required')
    for name,digest in previous['source_code_SHA256'].items():
        if sha(root/'source_snapshot'/name)!=digest or sha(Path(__file__).with_name(name))!=digest:
            raise ValueError('comparison feature/model/noise/label implementation changed: '+name)
    check_training_ledger(previous,manifest,root/'shared_training_inputs.jsonl')
    for source in manifest['sources']:
        if (source['sampling_frequency_Hz']!=REFERENCE_FS
                or tuple(source['recorded_MatchFilterTargetFreqz'])!=REFERENCE_BAND):
            raise ValueError('all native references must share the explicit recorded frequency convention')
    return previous,manifest


def record_epoch(item):
    quotient,remainder=divmod(item['seed']-9163000-item['source_index']*8-item['beam'],100)
    if quotient<0 or remainder:raise ValueError('prior native corruption seed schedule differs')
    return quotient


def compare_example(saved,item):
    identity=saved['identity'];index=0 if identity else 1
    for name,actual in (
        ('features_SHA256',item['old_feature_SHA256'][index]),
        ('noisy_log_SHA256',array_sha(item['target'] if identity else item['noisy'])),
        ('target_log_SHA256',array_sha(item['target'])),('support_SHA256',array_sha(item['mask'])),
        ('parameters_SHA256',item['parameters_SHA256']),
        ('source_complex128_context_SHA256',item['source_complex128_context_SHA256'])):
        if saved[name]!=actual:raise ValueError('exact previous native input/label replay differs: '+name)
    if list(item['context'])!=saved['context'] or saved['scale']!=item['scale']:
        raise ValueError('native context or known added-noise severity differs')
    return index


def build_bank(epoch,previous,manifest_path,python,profile,device):
    generation=previous['bank_generations'][epoch]
    if generation['epoch']!=epoch:raise ValueError('native bank generation order changed')
    bank={};extractions=[]
    for meta in generation['source_contexts']:
        i=meta['source_index'];box=meta['native_context'];left=box[1]
        z,actual=extract(python,manifest_path,i,box)
        if actual!=meta:raise ValueError('complete original native extraction provenance differs')
        extractions.append(actual)
        for beam in range(8):
            original=np.asarray(z[beam],np.complex64)
            old0,q0,mask=local_features(original,first_column=left)
            new0,qcheck,mcheck=band_features(original,first_column=left)
            if (not np.array_equal(q0,qcheck) or not np.array_equal(mask,mcheck)
                    or not np.array_equal(old0[0],new0[0])):raise ValueError('non-phase channels/labels changed')
            seed=9163000+epoch*100+i*8+beam;scale=.25+random.Random(seed).random()
            noisy,parameters=make_noisy(original,profile,device,seed,scale)
            old1,q1,m1=local_features(noisy,first_column=left)
            new1,qcheck,mcheck=band_features(noisy,first_column=left)
            if (not np.array_equal(q1,qcheck) or not np.array_equal(mask,m1)
                    or not np.array_equal(mask,mcheck) or not np.array_equal(old1[0],new1[0])):
                raise ValueError('non-phase corrupted input channels changed')
            bank[(i,beam,seed)]=dict(features=[new0,new1],old_feature_SHA256=[array_sha(old0),array_sha(old1)],
                amplitude_feature_SHA256=[array_sha(new0[0]),array_sha(new1[0])],
                target=q0,noisy=q1,mask=mask,scale=scale,context=box,
                parameters_SHA256=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest(),
                source_complex128_context_SHA256=actual['array_SHA256'])
    return bank,extractions


def evaluate_model(model,fields,device,zero=False):
    cases=[]
    for f in fields:
        z=f['original'];noisy=f['noisy']
        if zero:delta=identity=np.zeros(z.shape,np.float32)
        else:
            delta=infer_band_native(model,noisy,device)
            identity=infer_band_native(model,z,device)
        case=metrics(z,noisy,delta,identity)
        case.update(source_index=f['source_index'],beam_0based=f['beam'],seed=f['seed'],noise_scale=f['scale'])
        cases.append(case)
    return aggregate(cases)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('comparison','manifest','extract-python','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    previous,manifest=verified_comparison(args.comparison,args.manifest)
    profile_path=args.comparison/'aggregate_appearance_profile.json';profile=read(profile_path)
    if sha(profile_path)!=previous['profile_SHA256']:raise ValueError('prior aggregate noise profile changed')
    entries=[json.loads(s) for s in (args.comparison/'shared_training_inputs.jsonl').read_text().splitlines()]
    steps=4 if args.smoke else previous['steps_requested']
    entries=entries[:steps*2]
    if len(entries)!=steps*2:raise ValueError('complete matched-budget input ledger required')
    root=args.output.absolute();reserve=int(3.75*1024**3);budget=48*1024**2
    if shutil.disk_usage(root.parent).free<reserve+budget:raise ValueError('48MiB artifact budget plus 3.75GiB reserve required')
    root.mkdir(parents=True,exist_ok=False);snapshot=root/'source_snapshot';snapshot.mkdir()
    names=set(previous['source_code_SHA256'])|{Path(__file__).name,'BandAnchoredNativeSonarFeatures.py',
        'BandAnchoredNativeSonarWavelet.py','AuditNativeLocalSonarExperiment.py'}
    for name in sorted(names):shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(profile_path,root/'aggregate_appearance_profile.json')
    report=dict(status='preparing_fixed_band_native_phase_feature_ablation',PID=os.getpid(),smoke=args.smoke,
        model_family=BandAnchoredNativeSonarWavelet.family,feature_protocol=PROTOCOL,width=previous['width'],
        sampling_frequency_Hz=REFERENCE_FS,recorded_reference_subband_Hz=list(REFERENCE_BAND),
        reference_subband_is_recorded_processing_prior_not_clean_phase_or_complete_sensor_calibration=True,
        steps_requested=steps,steps_completed=0,training_examples_requested=len(entries),
        comparison_root=str(args.comparison.absolute()),comparison_report_SHA256=sha(args.comparison/'run_summary.json'),
        comparison_training_inputs_SHA256=sha(args.comparison/'shared_training_inputs.jsonl'),
        manifest_SHA256=sha(args.manifest),profile_SHA256=sha(profile_path),
        source_code_SHA256={f.name:sha(f) for f in snapshot.iterdir()},
        changed_factor='two phase feature channels only; exact prior amplitude channel, source/context/noise/identity/labels/initial tensors/budget/loss/optimizer',
        original_processed_observations_are_not_clean_truth=True,known_added_noise_only=True,
        private_raw_data_uploaded=False,raw_MAT_modified=False,source_pixels_resized=False,
        new_training_conditions=['fixed_band_phase_context'],
        amplitude_control_reused_not_retrained=True,
        reused_amplitude_control=previous['selected_by_full_native_known_noise_validation']['amplitude_only'],
        comparison_amp_control_phase_channels_ignored_exactly=True,
        validation_is_complete_native_beam_fields=not args.smoke,
        validation_scope='cropped_execution_smoke' if args.smoke else 'complete_native_beam_fields',
        reserved_source_indices=previous['reserved_source_indices'],train_source_indices=previous['train_source_indices'],
        reserved_part_is_known_and_previously_inspected_not_independent_scene_holdout=True,
        validation_history=[],checkpoints=[],bank_replays=[],source_exposure_counts={},
        real_quality_accepted=False,artifact_budget_bytes=budget,disk_reserve_bytes=reserve)
    started=time.monotonic()
    def save():
        if shutil.disk_usage(root).free<reserve or sum(f.stat().st_size for f in root.rglob('*') if f.is_file())>budget:
            raise OSError('bounded artifact disk budget exceeded; preserve evidence')
        report['elapsed_s']=time.monotonic()-started
        tmp=root/'run_summary.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');tmp.replace(root/'run_summary.json')
    save();torch.set_num_threads(2);torch.manual_seed(916251);device=torch.device(args.device)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.4,device)
    initial=next(c for c in previous['checkpoints']['phase_context'] if c['step']==0)
    initial_path=args.comparison/initial['file']
    if sha(initial_path)!=initial['SHA256']:raise ValueError('comparison initialization changed')
    state=torch.load(initial_path,map_location='cpu',weights_only=True)
    if state['step']!=0 or any(torch.count_nonzero(state['model'][key]) for key in
            ('network.heads.1.weight','network.heads.1.bias')) or torch.count_nonzero(state['model']['network.input.weight'][:,1:]):
        raise ValueError('only exact zero-residual/zero-phase-input initialization may cross protocols')
    model=BandAnchoredNativeSonarWavelet(previous['width']).to(device);model.load_state_dict(state['model'],strict=True)
    report['initial_model_tensors_exactly_match_comparison_step0']=all(
        torch.equal(value.cpu(),state['model'][name]) for name,value in model.state_dict().items())
    report['initial_checkpoint_SHA256']=initial['SHA256']
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
    def checkpoint(step):
        path=root/f'checkpoint_{step:06d}.pt'
        torch.save(dict(model=model.state_dict(),width=previous['width'],model_family=model.family,
            feature_protocol=PROTOCOL,step=step,use_phase=True,source_code_SHA256=report['source_code_SHA256'],
            sampling_frequency_Hz=REFERENCE_FS,recorded_reference_subband_Hz=list(REFERENCE_BAND),
            initial_checkpoint_SHA256=initial['SHA256'],training_manifest_SHA256=report['manifest_SHA256']),path)
        report['checkpoints'].append(dict(step=step,file=path.name,SHA256=sha(path)))
    try:
        fields=[]
        for ordinal,source_meta in enumerate(previous['validation_inputs'][:1] if args.smoke else previous['validation_inputs']):
            i=source_meta['source_index'];source=manifest['sources'][i]
            box=(0,0,64,512) if args.smoke else (0,0,source['native_records'],source['native_range_bins'])
            original,meta=extract(args.extract_python,args.manifest,i,box,beam=source_meta['beam'])
            original=np.asarray(original[0],np.complex64)
            noisy,parameters=make_noisy(original,profile,device,source_meta['seed'],source_meta['scale'])
            if not args.smoke:
                old_case=previous['validation_history'][0]['conditions']['phase_context']['cases'][ordinal]
                if array_sha(noisy)!=old_case['exact_noisy_complex64_SHA256'] or array_sha(original)!=old_case['exact_complex64_reference_SHA256']:
                    raise ValueError('full native validation input changed from comparison')
            fields.append(dict(original=original,noisy=noisy,source_index=i,beam=source_meta['beam'],
                               seed=source_meta['seed'],scale=source_meta['scale']))
        report['validation_history'].append(dict(step=0,metrics=evaluate_model(model,fields,device,True)))
        checkpoint(0);save();bank={};epoch=None;old_digest=hashlib.sha256();new_digest=hashlib.sha256();exposures=Counter()
        with (root/'matched_training_inputs.jsonl').open('x') as ledger:
            for step in range(1,steps+1):
                batch=entries[(step-1)*2:step*2]
                needed=record_epoch(batch[0])
                if epoch!=needed:
                    report['status']='replaying_previous_native_contexts_and_complex_noise';save()
                    bank,extractions=build_bank(needed,previous,args.manifest,args.extract_python,profile,device)
                    epoch=needed;report['bank_replays'].append(dict(epoch=epoch,source_contexts=extractions,beam_contexts=len(bank)))
                    save();print(json.dumps(dict(epoch=epoch,exact_old_context_bank_replayed=True)),flush=True)
                optimizer.zero_grad(set_to_none=True);loss_value=0.
                for saved in batch:
                    if record_epoch(saved)!=epoch:raise ValueError('bank generation split within matched step')
                    item=bank[(saved['source_index'],saved['beam'],saved['seed'])]
                    which=compare_example(saved,item);f=item['features'][which]
                    if saved['step']!=step:raise ValueError('matched training step order changed')
                    encoded=json.dumps(saved,ensure_ascii=False,sort_keys=True)+'\n';old_digest.update(encoded.encode())
                    changed=dict(saved);changed.update(comparison_relative_features_SHA256=saved['features_SHA256'],
                        features_SHA256=array_sha(f),amplitude_feature_SHA256=item['amplitude_feature_SHA256'][which])
                    encoded=json.dumps(changed,ensure_ascii=False,sort_keys=True)+'\n';ledger.write(encoded);new_digest.update(encoded.encode())
                    q=item['target'] if saved['identity'] else item['noisy']
                    features=torch.from_numpy(f[None]).to(device);mask=torch.from_numpy(item['mask'][None,None]).to(device)
                    target=torch.from_numpy(item['target'][None,None]).to(device);observed=torch.from_numpy(q[None,None]).to(device)
                    cy,cx,ch,cw=saved['core'];region=(slice(None),slice(None),slice(cy,cy+ch),slice(cx,cx+cw))
                    model.train();delta=model.feature_residual(features,mask,use_phase=True)[region]
                    loss=fidelity_loss(delta,(target-observed)[region],target[region],mask[region])
                    if not torch.isfinite(loss):raise ValueError('nonfinite matched native loss')
                    (loss/2).backward();loss_value+=float(loss.detach())/2
                    exposures[f'{saved["source_index"]}:{saved["beam"]}']+=1
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                report.update(status='training_matched_fixed_band_phase_features',steps_completed=step,last_loss=loss_value,
                    source_exposure_counts=dict(exposures),comparison_input_ledger_reconstructed_SHA256=old_digest.hexdigest(),
                    matched_training_input_provenance_SHA256=new_digest.hexdigest())
                if step%25==0 or step==steps:
                    ledger.flush();save();print(json.dumps(dict(step=step,loss=loss_value,elapsed_s=report['elapsed_s'])),flush=True)
                if step in (200,400) or step==steps:
                    report['status']='validating_matched_complete_native_fields';save()
                    measured=evaluate_model(model,fields,device)
                    report['validation_history'].append(dict(step=step,metrics=measured));checkpoint(step);save()
                    print(json.dumps(dict(step=step,validation={k:v for k,v in measured.items() if k!='cases'})),flush=True)
        expected=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in entries).encode()
        if old_digest.hexdigest()!=hashlib.sha256(expected).hexdigest():raise ValueError('matched input sequence reconstruction failed')
        report['all_prior_features_and_noisy_targets_support_reproduced_exactly']=True
        report['amplitude_channel_bitwise_unchanged_in_all_replayed_banks']=True
        report['matched_training_input_file_SHA256']=sha(root/'matched_training_inputs.jsonl')
        report['full_comparison_training_input_ledger_exact']=not args.smoke and old_digest.hexdigest()==previous['training_input_ledger_file_SHA256']
        if not args.smoke and (not report['full_comparison_training_input_ledger_exact'] or len(exposures)!=48):
            raise ValueError('full original training input/budget/source coverage differs')
        selected=min(report['validation_history'],key=lambda r:(r['metrics']['selection_score'],r['step']))
        record=next(c for c in report['checkpoints'] if c['step']==selected['step'])
        report['selected_by_complete_native_validation']=record
        selected_state=torch.load(root/record['file'],map_location=device,weights_only=True);model.load_state_dict(selected_state['model'],strict=True)
        bank.clear();fields.clear();save()
        if not args.smoke:
            report['status']='postselection_all_16_matched_native_beam_fields';report['postselection_partial']=[];save()
            for old_case in previous['postselection_complete_native_audit']['phase_context']['cases']:
                i=old_case['source_index'];beam=old_case['beam_0based'];source=manifest['sources'][i]
                box=(0,0,source['native_records'],source['native_range_bins'])
                original,meta=extract(args.extract_python,args.manifest,i,box,beam=beam);original=np.asarray(original[0],np.complex64)
                noisy,parameters=make_noisy(original,profile,device,old_case['seed'],old_case['noise_scale'])
                if array_sha(original)!=old_case['exact_complex64_reference_SHA256'] or array_sha(noisy)!=old_case['exact_noisy_complex64_SHA256']:
                    raise ValueError('complete reserved known-noise input differs from frozen comparison')
                field=dict(original=original,noisy=noisy,source_index=i,beam=beam,seed=old_case['seed'],scale=old_case['noise_scale'])
                case=evaluate_model(model,[field],device)['cases'][0];case['extraction']=meta
                case['noise_parameters_SHA256']=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()
                report['postselection_partial'].append(case);save()
                print(json.dumps(dict(source_index=i,beam=beam,matched_complete_native_values=original.size)),flush=True)
            check_reserved_coverage(report['postselection_partial'],previous,manifest)
            report['postselection_complete_native_audit']=aggregate(report['postselection_partial']);report['postselection_partial']=[]
        report.update(status='completed_matched_fixed_band_phase_experiment_real_denoising_unaccepted',
            actual_training_examples=steps*2,comparison_report_unchanged=sha(args.comparison/'run_summary.json')==report['comparison_report_SHA256'])
    except BaseException as exc:
        report.update(status='failed_preserved_matched_fixed_band_experiment_no_automatic_restart',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if device.type=='cuda':report['peak_CUDA_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        save()


if __name__=='__main__':main()
