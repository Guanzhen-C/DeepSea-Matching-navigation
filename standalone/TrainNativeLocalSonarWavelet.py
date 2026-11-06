"""Local same-device Haar experiment with known added stripes, NOT clean GT.

Two identical initial models receive shared native examples; one has observed
phase context, the other does not. Train contexts keep real native neighbors.
Selection uses whole native beam fields from the reserved physical part; a
post-selection audit covers ALL sixteen reserved side/beam fields. No private
measurements leave this machine and no large persistent training cache exists.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import shutil
import struct
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from IrregularComplexSonarStripes import irregular_complex_stripes, RECIPE
from LocalRelativeSonarFeatures import local_features, PROTOCOL
from NativeLocalSonarWavelet import NativeLocalSonarWavelet, context_box, infer_native


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4*1024**2), b''): digest.update(data)
    return digest.hexdigest()


def array_sha(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()


def extract(python, manifest, source_index, context, beam=None):
    # Do not resolve the venv symlink to base Python: that loses h5py's env.
    command = [str(Path(python).absolute()), str(Path(__file__).with_name('ExtractNativeProcessedSonarContext.py')),
               '--manifest',str(manifest),'--source-index',str(source_index),'--context',*map(str,context)]
    if beam is not None: command += ['--beam',str(beam)]
    process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    payload = process.stdout
    if len(payload) < 4: raise ValueError('missing local native binary header')
    length = struct.unpack('>I',payload[:4])[0]
    if not 0 < length < 128*1024: raise ValueError('unexpected native pipe header')
    metadata = json.loads(payload[4:4+length]); view = memoryview(payload)[4+length:]
    if (metadata['dtype'] != '<c16' or len(view) != metadata['array_bytes']
            or hashlib.sha256(view).hexdigest() != metadata['array_SHA256']):
        raise ValueError('original complex128 pipe integrity differs')
    values = np.frombuffer(view, np.complex128).reshape(metadata['native_shape']).copy()
    return values, metadata


def make_noisy(values, profile, device, seed, scale):
    original = np.asarray(values, np.complex64)
    tensor = torch.from_numpy(original.copy())[None,None].to(device)
    with torch.inference_mode():
        noisy, parameters = irregular_complex_stripes(tensor,
            torch.Generator(device=device).manual_seed(seed), profile, 0., noise_scale=scale,
            row_layout='native_record')
        output = noisy[0,0].cpu().numpy().copy()
    return output, parameters


def fidelity_loss(prediction, known, reference, mask):
    """Known added-noise supervision, including linear-amplitude observations.

    No delta clipping/gain bounds. Reference-energy normalization removes only
    a loss's arbitrary physical amplitude units, never output image brightness.
    """
    valid = mask.to(prediction.dtype); count = valid.sum().clamp_min(1)
    error = prediction-known
    robust = (F.smooth_l1_loss(prediction,known,reduction='none',beta=.1)*valid).sum()/count
    maximum = torch.where(mask,reference,reference.new_full((),-torch.inf)).amax()
    maximum = torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
    weights = torch.exp(2*torch.where(mask,reference-maximum,torch.zeros_like(reference)))*valid
    amplitude = (torch.expm1(error).square()*weights).sum()/weights.sum().clamp_min(1e-20)
    gradient = prediction.new_zeros(())
    for dim in (-1,-2):
        a = [slice(None)]*error.ndim; b = a.copy();a[dim]=slice(1,None);b[dim]=slice(None,-1)
        a,b=tuple(a),tuple(b); both=mask[a]&mask[b]
        gradient += ((error[a]-error[b]).abs()*both).sum()/both.sum().clamp_min(1)
    return robust+.2*amplitude+.1*gradient


def metrics(original, noisy, delta, identity_delta):
    reference=np.asarray(original,np.complex64); a=np.abs(reference).astype(np.float64)
    support=a>0; q=np.zeros(a.shape,np.float64);np.log(a,out=q,where=support)
    observed=np.abs(noisy).astype(np.float64); logs=np.zeros(a.shape,np.float64)
    np.log(observed,out=logs,where=support)
    error=(logs+delta-q)[support]; baseline=(logs-q)[support]
    identity=np.asarray(identity_delta)[support].astype(np.float64)
    if not len(error) or not np.isfinite(error).all(): raise ValueError('nonfinite validation restoration')
    maximum=a[support].max(); weights=(a[support]/maximum)**2
    ratios=np.exp(error); top=a[support]>=np.quantile(a[support],.999)
    return dict(native_values=int(a.size),observed_values=int(support.sum()),
        input_log_SSE=float(baseline@baseline),output_log_SSE=float(error@error),identity_log_SSE=float(identity@identity),
        input_log_RMSE=float(np.sqrt(np.mean(baseline**2))),output_log_RMSE=float(np.sqrt(np.mean(error**2))),
        identity_extra_log_RMS=float(np.sqrt(np.mean(identity**2))),
        restored_to_reference_amplitude_energy_error=float(np.average((ratios-1)**2,weights=weights)),
        original_top_0_1_percent_observation_ratio_median=float(np.median(ratios[top])),
        identity_top_0_1_percent_observation_ratio_median=float(np.median(np.exp(identity[top]))),
        exact_complex64_reference_SHA256=array_sha(reference),exact_noisy_complex64_SHA256=array_sha(noisy),
        predicted_log_delta_SHA256=array_sha(delta),identity_log_delta_SHA256=array_sha(identity_delta),
        missing_support_delta_exact_zero=bool(np.all(delta[~support]==0) and np.all(identity_delta[~support]==0)),
        source_observation_is_not_clean_truth=True)


def aggregate(cases):
    count=sum(c['observed_values'] for c in cases)
    output={name:math.sqrt(sum(c[key] for c in cases)/count) for name,key in (
        ('input_log_RMSE','input_log_SSE'),('output_log_RMSE','output_log_SSE'),('identity_extra_log_RMS','identity_log_SSE'))}
    output['selection_score']=output['output_log_RMSE']+output['identity_extra_log_RMS']
    output.update(native_values=sum(c['native_values'] for c in cases),observed_values=count,cases=cases)
    return output


def evaluate(models, fields, device, *, zero=False):
    cases={name:[] for name in models}
    for field in fields:
        for name,(model,use_phase) in models.items():
            z=field['original']; noisy=field['noisy']
            if zero:
                delta=np.zeros(z.shape,np.float32); identity=delta
            else:
                delta=infer_native(model,noisy,device,use_phase=use_phase)
                identity=infer_native(model,np.asarray(z,np.complex64),device,use_phase=use_phase)
            case=metrics(z,noisy,delta,identity)
            case.update(source_index=field['source_index'],beam_0based=field['beam'],
                        seed=field['seed'],noise_scale=field['scale'])
            cases[name].append(case)
    return {name:aggregate(items) for name,items in cases.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','profile','output','extract-python'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=400)
    p.add_argument('--refresh-steps',type=int,default=96);p.add_argument('--validation-interval',type=int,default=200)
    p.add_argument('--width',type=int,default=8);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if min(args.steps,args.refresh_steps,args.validation_interval,args.width)<1:raise ValueError('positive training schedule required')
    manifest=json.loads(args.manifest.read_text());profile=json.loads(args.profile.read_text());sources=manifest['sources']
    if manifest['status']!='prepared_native_processed_observation_teacher_manifest_not_training':raise ValueError('qualified native manifest required')
    if manifest['split_coverage']['same_physical_part_crosses_split']:raise ValueError('physical-part split leakage')
    training=[i for i,s in enumerate(sources) if s['split']=='train_known_extra_noise']
    validation=[i for i,s in enumerate(sources) if s['split']=='validation_known_extra_noise']
    if len(training)!=6 or len(validation)!=2:raise ValueError('six training/two reserved original side files required')
    root=args.output.absolute();root.mkdir(parents=True,exist_ok=False)
    reserve=int(3.75*1024**3);budget=48*1024**2
    if shutil.disk_usage(root).free<reserve+budget:raise ValueError('bounded 48MiB run plus 3.75GiB disk reserve required')
    snapshot=root/'source_snapshot';snapshot.mkdir()
    names=('TrainNativeLocalSonarWavelet.py','LocalRelativeSonarFeatures.py','NativeLocalSonarWavelet.py',
        'ExtractNativeProcessedSonarContext.py','IrregularComplexSonarStripes.py','PhaseAwareSonarWavelet.py',
        'PublicEnvelopeCoherentModel.py','PublicSonarEnvelopeModel.py','PublicSonarWavelet.py',
        'AuditSonarTrainingDomains.py','BathymetrySideScanDrawing.py','SonarComplexSourceStatistics.py')
    for name in names:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(args.profile,root/'aggregate_appearance_profile.json')
    report=dict(status='preparing_same_device_native_known_noise_pair',steps_completed=0,steps_requested=args.steps,
        smoke=args.smoke,model_family=NativeLocalSonarWavelet.family,feature_protocol=PROTOCOL,width=args.width,
        source_code_SHA256={f.name:sha(f) for f in snapshot.iterdir()},manifest_SHA256=sha(args.manifest),
        profile_SHA256=sha(args.profile),corruption_recipe=RECIPE,train_source_indices=training,
        reserved_source_indices=validation,known_added_noise_only=True,originals_are_processed_observations_not_clean=True,
        current_noisy_voyage_has_no_paired_clean_labels=True,private_raw_data_uploaded=False,raw_MAT_modified=False,
        source_pixels_resized=False,training_complex128_converted_to_complex64_explicitly=True,
        finite_feature_means_do_not_modify_output_base=True,context_native_halo=list(NativeLocalSonarWavelet.native_halo),
        core_native_shape=[64,256],training_context_noises_are_not_claimed_crops_of_whole_source_noises=True,
        validation_full_beam_fields_not_crops=not args.smoke,
        validation_scope='cropped_execution_smoke' if args.smoke else 'complete_native_beam_fields',
        selection_uses_beam_0_of_each_reserved_side_only=not args.smoke,
        postselection_audit_covers_all_16_reserved_beam_sources=not args.smoke,
        reserved_part_was_previously_inspected_not_independent_scene_holdout=True,
        validation_history=[],bank_generations=[],source_exposure_counts={},checkpoints={},quality_accepted=False,
        artifact_budget_bytes=budget,disk_reserve_bytes=reserve,device=str(args.device))
    started=time.monotonic()
    def save():
        size=sum(f.stat().st_size for f in root.rglob('*') if f.is_file())
        if size>budget or shutil.disk_usage(root).free<reserve:raise OSError('native experiment disk budget exceeded; preserve evidence')
        report['elapsed_s']=time.monotonic()-started
        temporary=root/'run_summary.tmp';temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(root/'run_summary.json')
    save();device=torch.device(args.device)
    torch.set_num_threads(2);torch.manual_seed(916251)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.4,device)
    initial=NativeLocalSonarWavelet(args.width).state_dict()
    models={}
    for name,use_phase in (('amplitude_only',False),('phase_context',True)):
        model=NativeLocalSonarWavelet(args.width).to(device);model.load_state_dict(initial)
        models[name]=(model,use_phase);(root/name).mkdir();report['checkpoints'][name]=[]
    optimizers={name:torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5) for name,(model,_) in models.items()}
    def checkpoint(step):
        for name,(model,use_phase) in models.items():
            path=root/name/f'checkpoint_{step:06d}.pt'
            torch.save(dict(model=model.state_dict(),model_family=model.family,feature_protocol=PROTOCOL,
                width=args.width,use_phase=use_phase,step=step,source_code_SHA256=report['source_code_SHA256'],
                training_manifest_SHA256=report['manifest_SHA256'],is_clean_truth=False),path)
            report['checkpoints'][name].append(dict(step=step,file=str(path.relative_to(root)),SHA256=sha(path)))
    try:
        # Full native selection fields, one beam on each side, two fixed severities.
        fields=[]
        for i in validation[:1] if args.smoke else validation:
            source=sources[i];h,w=source['native_records'],source['native_range_bins']
            box=(0,0,min(h,64),min(w,512)) if args.smoke else (0,0,h,w)
            z,meta=extract(args.extract_python,args.manifest,i,box,beam=0);z=np.asarray(z[0],np.complex64)
            for j,scale in enumerate((.5,1.0) if not args.smoke else (.75,)):
                seed=9162900+i*10+j;noisy,parameters=make_noisy(z,profile,device,seed,scale)
                fields.append(dict(original=z,noisy=noisy,source_index=i,beam=0,seed=seed,scale=scale,
                    extraction=meta,noise_parameters=parameters))
        report['validation_inputs']=[{k:v for k,v in f.items() if k not in ('original','noisy')} for f in fields]
        # Zero output heads prove identity on every native field, and are checked on actual features.
        test,_,mask=local_features(fields[0]['noisy'][:32,:128])
        with torch.inference_mode():
            for model,phase in models.values():
                actual=model.feature_residual(torch.from_numpy(test[None]).to(device),torch.from_numpy(mask[None,None]).to(device),use_phase=phase)
                if torch.count_nonzero(actual):raise ValueError('initial native model is not exact identity')
        report['initial_identity_function_checked_on_actual_features']=True
        report['validation_history'].append(dict(step=0,conditions=evaluate(models,fields,device,zero=True)))
        checkpoint(0);save()
        rng=random.Random(916252);digest=hashlib.sha256();bank=[];order=[]
        with (root/'shared_training_inputs.jsonl').open('x') as ledger:
            for step in range(1,args.steps+1):
                if not bank or (step-1)%args.refresh_steps==0:
                    bank=[];epoch=(step-1)//args.refresh_steps;generation=[]
                    report['status']='extracting_native_training_contexts_in_RAM';save()
                    for i in training[:1] if args.smoke else training:
                        source=sources[i];h,w=source['native_records'],source['native_range_bins']
                        y=rng.randrange(h-64+1);x=rng.randrange(w-256+1)
                        box,core=context_box((h,w),(y,x,64,256));top,left,bottom,right=box
                        z,meta=extract(args.extract_python,args.manifest,i,box)
                        generation.append(meta)
                        for beam in range(8):
                            original=np.asarray(z[beam],np.complex64)
                            f0,q0,mask=local_features(original,first_column=left)
                            seed=9163000+epoch*100+i*8+beam;scale=.25+random.Random(seed).random()
                            noisy,parameters=make_noisy(original,profile,device,seed,scale)
                            f1,q1,mask1=local_features(noisy,first_column=left)
                            if not np.array_equal(mask,mask1):raise ValueError('training synthetic noise altered missing support')
                            bank.append(dict(original_features=f0,noisy_features=f1,target=q0,noisy=q1,mask=mask,
                                core=core,source_index=i,beam=beam,seed=seed,scale=scale,context=box,
                                source_complex128_context_SHA256=meta['array_SHA256'],parameters_SHA256=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()))
                    report['bank_generations'].append(dict(epoch=epoch,source_contexts=generation,
                        beam_contexts=len(bank),sampling_is_native_contexts_not_whole_training_voyages=True))
                    order=list(range(len(bank)));rng.shuffle(order);save()
                    print(json.dumps(dict(event='native_context_bank_ready',epoch=epoch,beam_contexts=len(bank))),flush=True)
                for opt in optimizers.values():opt.zero_grad(set_to_none=True)
                losses={name:0. for name in models}
                for slot in range(2):
                    if not order:order=list(range(len(bank)));rng.shuffle(order)
                    item=bank[order.pop()];identity=rng.random()<.2
                    features=item['original_features'] if identity else item['noisy_features']
                    noisy=item['target'] if identity else item['noisy'];target=item['target'];mask=item['mask']
                    metadata={key:item[key] for key in ('source_index','beam','seed','scale','core','context','source_complex128_context_SHA256','parameters_SHA256')}
                    metadata.update(step=step,slot=slot,identity=identity,features_SHA256=array_sha(features),
                        noisy_log_SHA256=array_sha(noisy),target_log_SHA256=array_sha(target),support_SHA256=array_sha(mask))
                    encoded=json.dumps(metadata,ensure_ascii=False,sort_keys=True)+'\n';ledger.write(encoded);digest.update(encoded.encode())
                    identity_key=f'{item["source_index"]}:{item["beam"]}'
                    report['source_exposure_counts'][identity_key]=report['source_exposure_counts'].get(identity_key,0)+1
                    f=torch.from_numpy(features[None]).to(device);m=torch.from_numpy(mask[None,None]).to(device)
                    q=torch.from_numpy(noisy[None,None]).to(device);reference=torch.from_numpy(target[None,None]).to(device)
                    cy,cx,ch,cw=item['core'];region=(slice(None),slice(None),slice(cy,cy+ch),slice(cx,cx+cw))
                    for name,(model,phase) in models.items():
                        model.train();delta=model.feature_residual(f,m,use_phase=phase)[region]
                        loss=fidelity_loss(delta,(reference-q)[region],reference[region],m[region])
                        if not torch.isfinite(loss):raise ValueError('nonfinite known-noise native loss')
                        (loss/2).backward();losses[name]+=float(loss.detach())/2
                for name,(model,_) in models.items():
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizers[name].step()
                report.update(status='training_same_device_known_extra_noise_pair',steps_completed=step,last_losses=losses,
                              shared_training_inputs_SHA256=digest.hexdigest())
                if step%25==0 or step==args.steps:
                    ledger.flush();save();print(json.dumps(dict(step=step,losses=losses,elapsed_s=report['elapsed_s'])),flush=True)
                if step%args.validation_interval==0 or step==args.steps:
                    report['status']='validating_complete_reserved_native_beam_fields';save()
                    result=evaluate(models,fields,device)
                    report['validation_history'].append(dict(step=step,conditions=result));checkpoint(step);save()
                    print(json.dumps(dict(step=step,validation_scope=report['validation_scope'],
                        native_validation={n:{k:v for k,v in m.items() if k!='cases'} for n,m in result.items()})),flush=True)
        report['training_input_ledger_file_SHA256']=sha(root/'shared_training_inputs.jsonl')
        report['all_train_side_beam_sources_seen']=len(report['source_exposure_counts'])==(8 if args.smoke else 48)
        if not report['all_train_side_beam_sources_seen']:raise ValueError('native training source cycling incomplete')
        report['selected_by_full_native_known_noise_validation']={}
        for name,(model,_) in models.items():
            best=min(report['validation_history'],key=lambda r:(r['conditions'][name]['selection_score'],r['step']))
            chosen=next(c for c in report['checkpoints'][name] if c['step']==best['step'])
            report['selected_by_full_native_known_noise_validation'][name]=chosen
            saved=torch.load(root/chosen['file'],map_location=device,weights_only=True);model.load_state_dict(saved['model'],strict=True)
        # Drop diagnostic full fields before the complete reserved-coverage pass.
        fields.clear();bank.clear();report['status']='postselection_full_reserved_16_beam_audit';save()
        post={name:[] for name in models}
        if not args.smoke:
            for i in validation:
                source=sources[i];box=(0,0,source['native_records'],source['native_range_bins'])
                for beam in range(8):
                    original,meta=extract(args.extract_python,args.manifest,i,box,beam=beam)
                    original=np.asarray(original[0],np.complex64);seed=9169000+i*10+beam
                    noisy,parameters=make_noisy(original,profile,device,seed,.75)
                    field=dict(original=original,noisy=noisy,source_index=i,beam=beam,seed=seed,scale=.75)
                    result=evaluate(models,[field],device)
                    for name in models:
                        case=result[name]['cases'][0];case['extraction']=meta
                        case['noise_parameters_SHA256']=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()
                        post[name].append(case)
                    report['postselection_complete_native_audit_partial']={name:aggregate(items) for name,items in post.items()}
                    save();print(json.dumps(dict(event='reserved_full_beam_audited',source_index=i,beam=beam,
                        native_values=original.size)),flush=True)
            report['postselection_complete_native_audit']={name:aggregate(items) for name,items in post.items()}
            report['postselection_complete_native_audit_partial']={}
        report.update(status='completed_same_device_known_added_noise_pair_real_denoising_not_accepted',
            actual_training_examples=args.steps*2,private_data_uploaded=False,
            input_manifest_unchanged=sha(args.manifest)==report['manifest_SHA256'])
    except BaseException as exc:
        report.update(status='failed_preserved_native_known_noise_pair_no_automatic_restart',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if device.type=='cuda':report['peak_CUDA_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        save()


if __name__=='__main__':main()
