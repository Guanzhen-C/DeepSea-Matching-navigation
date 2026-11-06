"""Envelope-specific training, calendar-date held-out validation, public only.

References are actual observations plus known injected interference, not true
clean signals. No measured private arrays or RGB-derived fake amplitudes are
used. The existing mixed-domain baseline is never overwritten.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from AcousticStripeAugmentation import acoustic_stripes
from PublicSonarEnvelopeModel import PublicSonarEnvelopeModel
from PublicSonarWavelet import PublicSonarWavelet
from EmpiricalSonarStripeStrength import empirical_stripes
from PublicEnvelopeFullValidation import full_case_statistics


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def save(root,report):
    path=root/'run_summary.tmp';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    path.replace(root/'run_summary.json')


def resolve(path,mappings):
    p=Path(path)
    for left,right in mappings:
        try:return Path(right)/p.relative_to(left)
        except ValueError:pass
    return p


@lru_cache(maxsize=16)
def read_native(path,digest):
    if sha(path)!=digest:raise ValueError('native public cache changed')
    a=np.load(path,mmap_mode='r')
    if a.ndim!=2 or min(a.shape)<128:raise ValueError('at least 128 native records/range samples required')
    return a


@lru_cache(maxsize=16)
def native_origin(path,digest):
    values=read_native(path,digest)
    total=0.;count=0
    for start in range(0,len(values),32):
        block=np.asarray(values[start:start+32],np.float64)
        positive=block[block>0]
        total+=float(np.log(positive).sum(dtype=np.float64));count+=positive.size
    if not count:raise ValueError('no native positive values for source origin')
    return total/count,count


def crop(row,mappings,rng,height=128,width=384,source_context=False):
    a=read_native(str(resolve(row['path'],mappings)),row['cache_sha256'])
    y=rng.randrange(len(a)-height+1);x=rng.randrange(a.shape[1]-width+1)
    data=np.array(a[y:y+height,x:x+width],dtype=np.float64)
    if not np.isfinite(data).all() or (data<0).any():raise ValueError('invalid native public envelope')
    positive=data>0
    q=np.zeros(data.shape,np.float32);q[positive]=np.log(data[positive]).astype(np.float32)
    meta=dict(sample_id=row['sample_id'],row=y,column=x)
    if source_context:
        origin,count=native_origin(str(resolve(row['path'],mappings)),row['cache_sha256'])
        meta.update(source_log_origin=origin,source_positive_values=count)
    return torch.from_numpy(q)[None,None],torch.from_numpy(positive)[None,None],meta


def corruption(target,support,generator,profile,identity_probability,strength_protocol='native_field'):
    if strength_protocol=='realized_log_span':
        return empirical_stripes(target,support,generator,profile,identity_probability)
    if strength_protocol!='native_field':raise ValueError('unreviewed synthetic strength protocol')
    damaged,meta=acoustic_stripes(target.repeat(1,3,1,1),1,generator,profile,identity_probability=identity_probability)
    return torch.where(support,damaged[:,0:1],target),meta


def known_residual_loss(residual,known,support):
    if residual.shape!=known.shape or support.shape!=known.shape:
        raise ValueError('aligned native residual, known perturbation, and support required')
    n=support.sum((-2,-1),keepdim=True).clamp_min(1)
    scale=(known.square()*support).sum((-2,-1),keepdim=True)/n+.01**2
    return (((residual-known).square()*support).sum((-2,-1),keepdim=True)/n/scale).mean()


def source_noisy_origin(target,noisy,support,meta):
    # The source observation is already noisy, not a clean label. Artificial
    # corruption of just this training crop changes the whole input mean by
    # exactly delta_sum / whole_source_observed_count.
    center=noisy.new_tensor(meta['source_log_origin']).reshape(1,1,1,1)
    return center+((noisy-target)*support).sum((-2,-1),keepdim=True)/meta['source_positive_values']


def full_native_validation(rows,mappings,profile,strength_protocol,device):
    cases=[]
    for index,row in enumerate(rows):
        a=read_native(str(resolve(row['path'],mappings)),row['cache_sha256'])
        positive=np.asarray(a)>0;q=np.zeros(a.shape,np.float32)
        for start in range(0,len(a),32):
            values=np.asarray(a[start:start+32]);mask=positive[start:start+32]
            q[start:start+32][mask]=np.log(values[mask]).astype(np.float32)
        target=torch.from_numpy(q)[None,None];support=torch.from_numpy(positive)[None,None]
        generator=torch.Generator(device=device).manual_seed(87105+index)
        noisy,_=corruption(target.to(device),support.to(device),generator,profile,0.,strength_protocol)
        cases.append((target,support,noisy.cpu(),dict(sample_id=row['sample_id'],row=0,column=0,
            full_native_shape=list(a.shape),all_original_native_values_included=True)))
    return cases


@torch.inference_mode()
def evaluate(model,validation,device,legacy=False,origin_protocol='crop',scheme='windows'):
    input_sse=output_sse=identity_sse=0.;count=0
    native_count=0
    for target,support,noisy,meta in validation:
        if scheme=='full_native':
            stats=full_case_statistics(model,target,support,noisy,device,legacy=legacy)
            input_sse+=stats['input_SSE'];output_sse+=stats['output_SSE'];identity_sse+=stats['identity_SSE']
            count+=stats['observed_values'];native_count+=stats['native_values_processed'];continue
        target=target.to(device);support=support.to(device);noisy=noisy.to(device)
        if legacy:
            restored=model(noisy.repeat(1,3,1,1),1)[:,0:1]
            identity=model(target.repeat(1,3,1,1),1)[:,0:1]
        else:
            options={};identity_options={}
            if origin_protocol=='source':
                options=dict(origin=source_noisy_origin(target,noisy,support,meta))
                identity_options=dict(origin=target.new_tensor(meta['source_log_origin']).reshape(1,1,1,1))
            restored=model(noisy,support,**options);identity=model(target,support,**identity_options)
        input_sse+=float(((noisy-target).square()*support).sum())
        output_sse+=float(((restored-target).square()*support).sum())
        identity_sse+=float(((identity-target).square()*support).sum())
        count+=int(support.sum())
        native_count+=target.numel()
    return dict(known_added_input_log_RMSE=(input_sse/count)**.5,
        known_added_output_log_RMSE=(output_sse/count)**.5,
        extra_change_on_original_log_RMSE=(identity_sse/count)**.5,
        validation_selection_score=(output_sse+.5*identity_sse)/count,
        observed_native_values=count,real_clean_reference=False,
        complete_native_values_processed=native_count if scheme=='full_native' else None,
        scope=('every native value of the full held-out validation date, fixed whole-source origins and exact-context tiling'
               if scheme=='full_native' else 'fixed native windows from a whole held-out acquisition date, no resizing'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--initial-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--path-map',action='append',default=[])
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--strength-protocol',choices=('native_field','realized_log_span'),default='native_field')
    p.add_argument('--origin-protocol',choices=('crop','source'),default='crop')
    p.add_argument('--validation-scheme',choices=('windows','full_native'),default='windows')
    args=p.parse_args()
    if args.steps<1 or not 1<=args.batch_size<=8:raise ValueError('bounded training budget/batch required')
    data=json.loads(args.dataset.read_text())
    if data['status']!='completed_public_native_envelope_dataset_not_denoising_acceptance':raise ValueError('complete qualified public dataset required')
    from PublicSonarData import validate_splits
    split_audit=validate_splits(data['records'])
    train=[r for r in data['records'] if r['split']=='train']
    valid=[r for r in data['records'] if r['split']=='validation']
    if not train or not valid or {r['group_id'] for r in train}&{r['group_id'] for r in valid}:raise ValueError('whole-date split required')
    mappings=[x.split('=',1) for x in args.path_map]
    profile=json.loads(args.profile.read_text())
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for name in ('TrainPublicSonarEnvelope.py','PublicSonarEnvelopeModel.py','PublicSonarWavelet.py','AcousticStripeAugmentation.py','EmpiricalSonarStripeStrength.py','PublicEnvelopeFullValidation.py','PublicSonarNativeGrid.py','PublicSonarData.py','InspectPublicSonarSignals.py','InspectPublicSonarDatasets.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(args.profile,root/'acoustic_stripe_style_profile.json')
    torch.set_num_threads(2);torch.manual_seed(20261105)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(.7,device)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.deterministic=True
    state=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=True)
    model=PublicSonarEnvelopeModel(width=state['width']).to(device)
    model.network.load_state_dict(state['model'],strict=True)
    # Other output heads remain frozen inside this independent copy only.
    for index in (0,2):model.network.heads[index].requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-5)
    rng=random.Random(20261105)
    noise=torch.Generator(device=device).manual_seed(20261105)
    validation=[];vrng=random.Random(77105);vnoise=torch.Generator().manual_seed(77105)
    if args.validation_scheme=='full_native':
        validation=full_native_validation(valid,mappings,profile,args.strength_protocol,device)
    else:
        for row in valid:
            for _ in range(8):
                target,support,origin=crop(row,mappings,vrng,source_context=args.origin_protocol=='source')
                noisy,_=corruption(target,support,vnoise,profile,0.,args.strength_protocol)
                validation.append((target,support,noisy,origin))
    digest=hashlib.sha256()
    for _,_,noisy,_ in validation:digest.update(noisy.numpy().tobytes())
    input_hash=digest.hexdigest()
    report=dict(status='training_independent_public_envelope_model',model_family=model.family,
        steps_requested=args.steps,steps_completed=0,batch_size=args.batch_size,
        dataset_sha256=sha(args.dataset),noise_profile_sha256=sha(args.profile),split_audit=split_audit,
        initial_checkpoint_sha256=sha(args.initial_checkpoint),
        source_sha256={s.name:sha(s) for s in snapshot.iterdir()},
        public_only=True,private_sonar_used=False,original_references_are_not_clean_truth=True,
        image_branch_of_original_baseline_modified=False,
        unit_equivariance='one scalar noisy-input log origin; no per-ping/per-range normalization',
        corruption_strength_protocol=args.strength_protocol,
        training_origin_protocol=args.origin_protocol,validation_scheme=args.validation_scheme,
        full_native_tiling_requires_one_fixed_origin_per_beam=True,
        validation_native_windows=len(validation) if args.validation_scheme=='windows' else 0,
        validation_full_native_groups=len(validation) if args.validation_scheme=='full_native' else 0,
        validation_input_sha256=input_hash,
        validation_window_origins=[case[3] for case in validation],
        test_date_used_for_training_or_checkpoint_selection=False,
        train_source_ids=[r['sample_id'] for r in train],actual_train_source_ids=[],
        checkpoints=[],validation_history=[],quality_accepted=False,
        full_native_measured_evaluation_complete=False)
    started=time.monotonic();save(root,report)
    try:
        legacy=PublicSonarWavelet(width=state['width']).to(device).eval()
        legacy.load_state_dict(state['model'],strict=True)
        report['legacy_envelope_on_identical_heldout_validation']=evaluate(legacy,validation,device,legacy=True,scheme=args.validation_scheme)
        del legacy
        model.eval();report['validation_history'].append(dict(step=0,**evaluate(model,validation,device,origin_protocol=args.origin_protocol,scheme=args.validation_scheme)))
        initial=root/'checkpoint_000000.pt'
        torch.save(dict(model=model.state_dict(),width=state['width'],step=0,model_family=model.family,
            input_space='native log amplitude with positive support',source_sha256=report['source_sha256']),initial)
        report['checkpoints'].append(dict(step=0,file=initial.name,sha256=sha(initial)))
        save(root,report)
        model.train();used=set()
        for step in range(1,args.steps+1):
            clean=[];damaged=[];supports=[];origins=[]
            for _ in range(args.batch_size):
                row=train[rng.randrange(len(train))]
                target,support,meta=crop(row,mappings,rng,source_context=args.origin_protocol=='source')
                target=target.to(device);support=support.to(device)
                noisy,_=corruption(target,support,noise,profile,.2,args.strength_protocol)
                clean.append(target);damaged.append(noisy);supports.append(support);used.add(row['sample_id'])
                if args.origin_protocol=='source':origins.append(source_noisy_origin(target,noisy,support,meta))
            target=torch.cat(clean);noisy=torch.cat(damaged);support=torch.cat(supports)
            options=dict(origin=torch.cat(origins)) if args.origin_protocol=='source' else {}
            residual=model.residual(noisy,support,**options)
            known=target-noisy
            # A weak synthetic field must not disappear in the large natural
            # log-amplitude variance. Identity examples have a finite floor.
            loss=known_residual_loss(residual,known,support)
            if not torch.isfinite(loss):raise ValueError('nonfinite envelope training objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            report.update(steps_completed=step,last_loss=float(loss),actual_train_source_ids=sorted(used))
            if step%100==0:
                report['elapsed_s']=time.monotonic()-started;save(root,report)
                print(json.dumps(dict(step=step,steps=args.steps,loss=float(loss),elapsed_s=report['elapsed_s'])),flush=True)
            if step%1000==0 or step==args.steps:
                model.eval();metrics=evaluate(model,validation,device,origin_protocol=args.origin_protocol,scheme=args.validation_scheme)
                report['validation_history'].append(dict(step=step,**metrics))
                path=root/f'checkpoint_{step:06d}.pt'
                torch.save(dict(model=model.state_dict(),width=state['width'],step=step,model_family=model.family,
                    input_space='native log amplitude with positive support',source_sha256=report['source_sha256']),path)
                report['checkpoints'].append(dict(step=step,file=path.name,sha256=sha(path)))
                save(root,report);model.train()
                print(json.dumps(dict(step=step,validation=metrics)),flush=True)
        report.update(status='completed_envelope_training_real_interference_removal_unverified',
            best_validation_step=min(report['validation_history'],key=lambda m:m['validation_selection_score'])['step'])
    except Exception as exc:
        report.update(status='failed_envelope_training_preserved_no_automatic_restart',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        save(root,report)


if __name__=='__main__':main()
