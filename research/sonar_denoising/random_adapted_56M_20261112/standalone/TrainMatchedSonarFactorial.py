"""Five actually-trained native controls plus a frozen prior benchmark.

Mandatory versus learnable local/coherent residual fusion; within the fusion
family a 2x2 acquisition-noise-band/front-end design. Every model is evaluated
on BOTH known test-corruption laws, not only its own training law. Weak noise
and no-added-noise observations remain distinct. No current-voyage clean truth
is invented, no private data is uploaded, and no large disk cache is created.
"""
from pathlib import Path
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
import shutil
import time

import numpy as np
import torch

from TrainBandAnchoredNativeSonar import verified_comparison,record_epoch
from AuditBandAnchoredNativeSonar import load_selected as load_frozen_prior
from TrainNativeLocalSonarWavelet import sha,array_sha,extract,make_noisy,fidelity_loss,metrics,aggregate
from StreamNativeProcessedSide import stream_side,beam_view
from NativeLocalSonarWavelet import context_box
from MatchedSonarSignal import CONDITIONS,HALO,WEAK_TO_STRONG,noise_variants,source_logs
from MatchedSonarWavelet import MatchedSonarWavelet,context_input,infer_group
from BandAnchoredNativeSonarFeatures import PROTOCOL
from PublicEnvelopeCoherentModel import DirectionalNoiseConsensus


def core_plan(saved,source):
    top,left,_,_=saved['context'];cy,cx,h,w=saved['core']
    return context_box((source['native_records'],source['native_range_bins']),
        (top+cy,left+cx,h,w),halo=HALO)


def model_digest(model):
    result=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        result.update(name.encode());result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def ideal_operator_control():
    reference=torch.ones(1,1,21,257);reference[:,:,:,128]=10
    observed=reference+1;oracle=(reference/observed).log()
    with torch.inference_mode():
        averaged=DirectionalNoiseConsensus()(oracle,torch.ones_like(oracle,dtype=torch.bool))
        ratio=float((observed*averaged.exp())[0,0,10,128]/reference[0,0,10,128])
    return dict(known_ideal_amplitude_field=True,geological_clean_truth_claimed=False,
        correct_log_gain_given_to_aggregator=True,bright_amplitude_before=10.,
        reconstructed_bright_amplitude_ratio_after_mandatory_aggregation=ratio,
        limitation_of_operator_not_proof_whole_network_cannot_precompensate=True)


def state_entry(values,left,bins,front):
    f,q,m,_,_=context_input(values,first_model_column=left,full_range_bins=bins,frontend=front)
    return dict(features=f,logs=q,mask=m)


def make_bank(epoch,previous,manifest,manifest_path,python,profile,device,entries):
    relevant=[e for e in entries if record_epoch(e)==epoch]
    per_source={e['source_index']:e for e in relevant};bank={};provenance=[]
    for i,sample in sorted(per_source.items()):
        source=manifest['sources'][i];box,core=core_plan(sample,source);top,left,bottom,right=box
        z,meta=extract(python,manifest_path,i,box)
        # Widening the real FIR/CNN halo must not change any old observed byte.
        old=next(m for m in previous['bank_generations'][epoch]['source_contexts'] if m['source_index']==i)
        ot,ol,ob,orr=old['native_context'];first=max(0,left-1);old_first=max(0,ol-1)
        inner=z[:,ot-top:ob-top,old_first-first:orr-first]
        if array_sha(inner)!=old['array_SHA256']:raise ValueError('expanded real halo changed old source observations')
        provenance.append(dict(extraction=meta,old_inner_context_SHA256=old['array_SHA256'],core_in_context=list(core)))
        for beam in range(8):
            original=np.asarray(z[beam],np.complex64)
            target=source_logs(original[:,1:] if left else original)[0]
            seed=9170000+epoch*100+i*8+beam;scale=WEAK_TO_STRONG[(i*8+beam+epoch)%len(WEAK_TO_STRONG)]
            damaged,parameters=make_noisy(original,profile,device,seed,1.)
            delta=damaged-original;fields,scales=noise_variants(original,delta,scale)
            states={('identity',False):state_entry(original,left,source['native_range_bins'],False),
                    ('identity',True):state_entry(original,left,source['native_range_bins'],True)}
            for law,values in fields.items():
                for front in (False,True):states[(law,front)]=state_entry(values,left,source['native_range_bins'],front)
            masks=[v['mask'] for v in states.values()]
            if any(not np.array_equal(m,masks[0]) for m in masks[1:]):raise ValueError('known augmentation/front end changed source support')
            bank[(i,beam)]=dict(target=target,states=states,core=core,context=box,seed=seed,scale=scale,
                source_context_SHA256=meta['array_SHA256'],unit_known_delta_SHA256=array_sha(delta),actual_added_noise=scales,
                source_recipe_parameters_SHA256=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest())
    return bank,provenance


def prepare_fields(original,profile,device,source_index,beam,seed,scale):
    noisy,parameters=make_noisy(original,profile,device,seed,1.)
    fields,diagnostic=noise_variants(original,noisy-original,scale)
    common=dict(source_index=source_index,beam=beam,seed=seed,scale=scale,
        original_complex64_SHA256=array_sha(original),unit_known_delta_SHA256=array_sha(noisy-original),
        actual_added_noise=diagnostic,source_recipe_parameters_SHA256=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest())
    return [dict(original=original,noisy=values,law=law,**common) for law,values in fields.items()]


def evaluate(models,fields,device,*,zero_neural_for=()):
    cases={name:[] for name in models};identity_cache={};identity_info={}
    for field in fields:
        original=field['original'];identity=(field['source_index'],field['beam'])
        if identity not in identity_cache:
            identity_cache[identity],identity_info[identity]=infer_group(models,original,device,zero_neural_for=zero_neural_for)
        result,info=infer_group(models,field['noisy'],device,zero_neural_for=zero_neural_for)
        for name in models:
            measured=metrics(original,field['noisy'],result[name],identity_cache[identity][name])
            measured.update(source_index=field['source_index'],beam_0based=field['beam'],seed=field['seed'],
                noise_scale=field['scale'],test_noise_law=field['law'],
                noisy_neural_contribution=info['neural'][name],identity_neural_contribution=identity_info[identity]['neural'][name])
            cases[name].append(measured)
    return {name:aggregate(items) for name,items in cases.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sampling-plan','frozen-baseline','manifest','extract-python','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    previous,manifest=verified_comparison(args.sampling_plan,args.manifest)
    old_entries=[json.loads(s) for s in (args.sampling_plan/'shared_training_inputs.jsonl').read_text().splitlines()]
    steps=4 if args.smoke else 400;entries=old_entries[:2*steps]
    profile_path=args.sampling_plan/'aggregate_appearance_profile.json';profile=json.loads(profile_path.read_text())
    root=args.output.absolute();reserve=int(3.75*1024**3);budget=64*1024**2
    if shutil.disk_usage(root.parent).free<reserve+budget:raise ValueError('64MiB run budget plus 3.75GiB reserve required')
    root.mkdir(parents=True,exist_ok=False);snapshot=root/'source_snapshot';snapshot.mkdir()
    sources=set(previous['source_code_SHA256'])|{Path(__file__).name,'MatchedSonarSignal.py','MatchedSonarWavelet.py',
        'TrainBandAnchoredNativeSonar.py','AuditBandAnchoredNativeSonar.py','BandAnchoredNativeSonarFeatures.py',
        'BandAnchoredNativeSonarWavelet.py','AuditNativeLocalSonarExperiment.py','StreamNativeProcessedSide.py'}
    for name in sorted(sources):shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(profile_path,root/'aggregate_appearance_profile.json')
    started=time.monotonic();report=dict(status='preparing_matched_processing_factorial',PID=os.getpid(),smoke=args.smoke,
        steps_requested=steps,steps_completed=0,model_family=MatchedSonarWavelet.family,feature_protocol=PROTOCOL,
        width=previous['width'],conditions=CONDITIONS,training_core_halo=list(HALO),weak_noise_design=list(WEAK_TO_STRONG),
        noise_level_design_is_not_measured_device_SNR_distribution=True,
        acquisition_noise_filter_is_nominal_band_hypothesis_not_sensor_calibration=True,
        acquisition_filter_changes_only_simulated_delta_not_target=True,
        total_added_RMS_matched_between_noise_laws_not_claimed_equal_inband_SNR=True,
        actual_conditions_trained=list(CONDITIONS),frozen_benchmark_retrained=False,
        original_reference_observations_not_clean_truth=True,
        current_noisy_voyage_waveforms_not_used_as_training_targets_or_inputs=True,
        appearance_profile_derived_from_previously_inspected_noisy_voyage=True,
        source_pixels_resized=False,original_MAT_modified=False,private_data_uploaded=False,
        initial_neural_function_zero_but_frontend_can_be_nonidentity=True,
        exact_previous_noise_fields_reused=False,previous_core_coordinates_and_identity_schedule_reused=True,
        sampling_plan_report_SHA256=sha(args.sampling_plan/'run_summary.json'),sampling_plan_ledger_SHA256=sha(args.sampling_plan/'shared_training_inputs.jsonl'),
        sampling_plan_root=str(args.sampling_plan.absolute()),frozen_baseline_root=str(args.frozen_baseline.absolute()),
        frozen_baseline_report_SHA256=sha(args.frozen_baseline/'run_summary.json'),manifest_SHA256=sha(args.manifest),
        noise_profile_SHA256=sha(profile_path),source_code_SHA256={f.name:sha(f) for f in snapshot.iterdir()},
        validation_uses_both_noise_laws_for_every_model=True,validation_is_complete_native_fields=not args.smoke,
        validation_scope='cropped_execution_smoke' if args.smoke else 'complete_native_beam_fields',
        reserved_part_previously_inspected_not_independent_scene_generalization=True,
        checkpoints={name:[] for name in CONDITIONS},validation_history=[],bank_generations=[],source_exposure_counts={},
        disk_reserve_bytes=reserve,artifact_budget_bytes=budget,real_quality_accepted=False)
    def save():
        report['elapsed_s']=time.monotonic()-started
        if shutil.disk_usage(root).free<reserve or sum(f.stat().st_size for f in root.rglob('*') if f.is_file())>budget:
            raise OSError('bounded matched-processing artifact budget exceeded')
        tmp=root/'run_summary.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');tmp.replace(root/'run_summary.json')
    save();device=torch.device(args.device);torch.set_num_threads(2);torch.manual_seed(916251)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.4,device)
    report['ideal_mandatory_aggregation_control']=ideal_operator_control();save()
    frozen,frozen_meta=load_frozen_prior(args.frozen_baseline,device)
    models={'frozen_prior_anchored_raw':(frozen,False)}
    initial_record=next(c for c in previous['checkpoints']['phase_context'] if c['step']==0)
    initial_path=args.sampling_plan/initial_record['file']
    if sha(initial_path)!=initial_record['SHA256']:raise ValueError('verified zero-neural initialization changed')
    state=torch.load(initial_path,map_location='cpu',weights_only=True)
    template=MatchedSonarWavelet(previous['width'])
    absent,unexpected=template.load_state_dict(state['model'],strict=False)
    expected={name for name in template.state_dict() if name.startswith('local_gate.')}
    if set(absent)!=expected or unexpected or any(torch.count_nonzero(state['model'][k]) for k in
        ('network.heads.1.weight','network.heads.1.bias')):raise ValueError('only explicit new local selector parameters may be initialized')
    shared=template.state_dict();report['initial_checkpoint_SHA256']=initial_record['SHA256']
    report['all_shared_initial_tensors_match_prior_step0']=all(torch.equal(shared[k],v) for k,v in state['model'].items())
    for name,condition in CONDITIONS.items():
        model=MatchedSonarWavelet(previous['width'],aggregation=condition['aggregation']).to(device)
        model.load_state_dict(shared,strict=True);models[name]=(model,condition['frontend']);(root/name).mkdir()
    initial_digests={name:model_digest(models[name][0]) for name in CONDITIONS}
    if len(set(initial_digests.values()))!=1:raise ValueError('all newly trained controls require identical complete initial tensors')
    report['initial_full_tensor_SHA256']=initial_digests
    optimizers={name:torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
        for name,(model,_) in models.items() if name in CONDITIONS}
    def checkpoint(step):
        for name in CONDITIONS:
            model=models[name][0];path=root/name/f'checkpoint_{step:06d}.pt'
            torch.save(dict(model=model.state_dict(),width=previous['width'],model_family=model.family,feature_protocol=PROTOCOL,
                condition=CONDITIONS[name],condition_name=name,step=step,source_code_SHA256=report['source_code_SHA256'],
                training_manifest_SHA256=report['manifest_SHA256']),path)
            report['checkpoints'][name].append(dict(step=step,file=str(path.relative_to(root)),SHA256=sha(path)))
    try:
        fields=[]
        for i in previous['reserved_source_indices'][:1] if args.smoke else previous['reserved_source_indices']:
            source=manifest['sources'][i];box=(0,0,64,768) if args.smoke else (0,0,source['native_records'],source['native_range_bins'])
            original,meta=extract(args.extract_python,args.manifest,i,box,beam=0);original=np.asarray(original[0],np.complex64)
            for j,scale in enumerate((.08,) if args.smoke else (.08,.75)):
                fields.extend(prepare_fields(original,profile,device,i,0,9179000+i*10+j,scale))
        report['validation_field_ledger']=[{k:v for k,v in field.items() if k not in ('original','noisy')}|dict(
            actual_input_complex64_SHA256=array_sha(field['noisy'])) for field in fields]
        report['validation_history'].append(dict(step=0,metrics=evaluate(models,fields,device,zero_neural_for=set(CONDITIONS))))
        checkpoint(0);save();bank={};epoch=None;digest=hashlib.sha256();exposure=Counter();weak=identity_count=0
        with (root/'shared_factorial_training_inputs.jsonl').open('x') as ledger:
            for step in range(1,steps+1):
                batch=entries[(step-1)*2:step*2];needed=record_epoch(batch[0])
                if epoch!=needed:
                    bank.clear();gc.collect();report['status']='building_native_compound_halo_factorial_bank';save()
                    bank,meta=make_bank(needed,previous,manifest,args.manifest,args.extract_python,profile,device,entries)
                    epoch=needed;report['bank_generations'].append(dict(epoch=epoch,source_contexts=meta,beam_contexts=len(bank)))
                    save();print(json.dumps(dict(epoch=epoch,beam_contexts=len(bank))),flush=True)
                for opt in optimizers.values():opt.zero_grad(set_to_none=True)
                losses={name:0. for name in CONDITIONS}
                for old in batch:
                    item=bank[(old['source_index'],old['beam'])];is_identity=old['identity'];identity_count+=int(is_identity)
                    weak+=int(not is_identity and item['scale']<=.20)
                    target=torch.from_numpy(item['target'][None,None]).to(device);cy,cx,ch,cw=item['core']
                    region=(slice(None),slice(None),slice(cy,cy+ch),slice(cx,cx+cw))
                    record=dict(step=step,slot=old['slot'],source_index=old['source_index'],beam=old['beam'],identity=is_identity,
                        native_context=list(item['context']),native_core_in_context=list(item['core']),
                        source_complex128_SHA256=item['source_context_SHA256'],unit_known_delta_SHA256=item['unit_known_delta_SHA256'],
                        noise_seed=item['seed'],noise_RMS_scale=0. if is_identity else item['scale'],
                        actual_corrupted_field_RMS=item['actual_added_noise'],source_recipe_parameters_SHA256=item['source_recipe_parameters_SHA256'],
                        target_log_SHA256=array_sha(item['target']),conditions={})
                    for name,condition in CONDITIONS.items():
                        entry=item['states'][('identity' if is_identity else condition['noise'],condition['frontend'])]
                        f=torch.from_numpy(entry['features'][None]).to(device);q=torch.from_numpy(entry['logs'][None,None]).to(device)
                        mask=torch.from_numpy(entry['mask'][None,None]).to(device);model=models[name][0];model.train()
                        delta=model.feature_residual(f,mask,use_phase=True)[region]
                        loss=fidelity_loss(delta,(target-q)[region],target[region],mask[region])
                        if not torch.isfinite(loss):raise ValueError('nonfinite composed native known-noise loss')
                        (loss/2).backward();losses[name]+=float(loss.detach())/2
                        record['conditions'][name]=dict(feature_SHA256=array_sha(entry['features']),input_log_SHA256=array_sha(entry['logs']),
                            support_SHA256=array_sha(entry['mask']))
                    encoded=json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n';ledger.write(encoded);digest.update(encoded.encode())
                    exposure[f'{old["source_index"]}:{old["beam"]}']+=1
                for name,opt in optimizers.items():torch.nn.utils.clip_grad_norm_(models[name][0].parameters(),1.);opt.step()
                report.update(status='training_five_matched_processing_controls',steps_completed=step,last_losses=losses,
                    source_exposure_counts=dict(exposure),weak_corrupted_examples=weak,identity_examples=identity_count,
                    shared_training_input_provenance_SHA256=digest.hexdigest())
                if step%25==0 or step==steps:
                    ledger.flush();save();print(json.dumps(dict(step=step,losses=losses,elapsed_s=report['elapsed_s'])),flush=True)
                if step in (200,400) or step==steps:
                    report['status']='validating_every_control_on_both_full_native_corruption_laws';save()
                    result=evaluate(models,fields,device);report['validation_history'].append(dict(step=step,metrics=result))
                    checkpoint(step);save();print(json.dumps(dict(step=step,validation={n:{k:v for k,v in a.items() if k!='cases'} for n,a in result.items()})),flush=True)
        report['training_ledger_file_SHA256']=sha(root/'shared_factorial_training_inputs.jsonl')
        report['actual_training_examples_per_trained_condition']=steps*2
        report['all_48_train_beam_sources_seen']=len(exposure)==48
        if not args.smoke and not report['all_48_train_beam_sources_seen']:raise ValueError('training source cycling incomplete')
        report['selected_by_complete_common_validation']={}
        for name in CONDITIONS:
            best=min(report['validation_history'],key=lambda r:(r['metrics'][name]['selection_score'],r['step']))
            chosen=next(c for c in report['checkpoints'][name] if c['step']==best['step'])
            report['selected_by_complete_common_validation'][name]=chosen
            selected=torch.load(root/chosen['file'],map_location=device,weights_only=True);models[name][0].load_state_dict(selected['model'],strict=True)
        bank.clear();fields.clear();gc.collect();save()
        if not args.smoke:
            post={name:[] for name in models};report['status']='postselection_full_16_beams_both_noise_laws_all_controls';save()
            for i in previous['reserved_source_indices']:
                source=manifest['sources'][i];box=(0,0,source['native_records'],source['native_range_bins'])
                values,meta=stream_side(args.extract_python,args.manifest,i,box)
                for beam in range(8):
                    original,beam_meta=beam_view(values,meta,beam);original=np.asarray(original[0],np.complex64)
                    test=prepare_fields(original,profile,device,i,beam,9189000+i*10+beam,.35)
                    result=evaluate(models,test,device)
                    for name,a in result.items():
                        for case in a['cases']:case['extraction']=beam_meta;post[name].append(case)
                    report['postselection_partial']={name:aggregate(cases) for name,cases in post.items()};save()
                    print(json.dumps(dict(source_index=i,beam=beam,test_laws=2,source_native_values=original.size)),flush=True)
                del values
            report['postselection_complete_native_audit']={name:aggregate(cases) for name,cases in post.items()}
            report['postselection_partial']={}
        report.update(status='completed_matched_processing_factorial_real_denoising_unaccepted',
            full_native_neural_image_created=False,sampling_plan_unchanged=sha(args.sampling_plan/'run_summary.json')==report['sampling_plan_report_SHA256'])
    except BaseException as exc:
        report.update(status='failed_preserved_matched_factorial_no_automatic_restart',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if device.type=='cuda':report['peak_CUDA_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        save()


if __name__=='__main__':main()
