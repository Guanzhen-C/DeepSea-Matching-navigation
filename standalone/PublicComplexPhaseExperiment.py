"""Native complete-field paired inputs and evaluation for a phase ablation.

Original public observations are NOT clean truth. Known synthetic complex
interference is the supervised target. The three declared validation chips
are also in training: this measures new-noise recovery on familiar textures,
not independent-scene/acquisition generalization. No labels are used.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from PublicEnvelopeNativeTraining import context_region
from PublicSonarComplexCorruption import complex_additive_stripes,complex_phase_features


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def array_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass
class ComplexObservation:
    sample_id:str
    path:Path
    source_sha256:str
    features:np.ndarray
    logs:np.ndarray
    support:np.ndarray
    origin:float

    @classmethod
    def read(cls,row,path,device):
        path=Path(path)
        if row.get('domain')!='complex_field' or row.get('split')!='train_aux' or row.get('is_clean_truth') is not False:
            raise ValueError('explicit observed public auxiliary complex field required')
        if file_sha(path)!=row['cache_sha256']:raise ValueError('public original complex file changed')
        values=np.load(path)
        if values.dtype!=np.complex64 or values.ndim!=2 or not np.isfinite(values).all():
            raise ValueError('finite original native complex64 source required')
        with torch.inference_mode():
            features,q,support,origin=complex_phase_features(torch.from_numpy(values)[None,None].to(device))
            arrays=[v[0].cpu().numpy().copy() for v in (features,q,support)]
        for array in arrays:array.flags.writeable=False
        return cls(row['sample_id'],path,row['cache_sha256'],*arrays,float(origin.item()))


@dataclass
class CompleteField:
    features:np.ndarray
    logs:np.ndarray
    origin:float
    metadata:dict


@torch.inference_mode()
def corrupt_complete(ref,profile,device,seed,scale):
    # Use the original complex measurements, never a magnitude-to-phase lift.
    if file_sha(ref.path)!=ref.source_sha256:raise ValueError('original public source changed')
    values=np.load(ref.path)
    z=torch.from_numpy(values)[None,None].to(device)
    observed,parameters=complex_additive_stripes(z,torch.Generator(device=device).manual_seed(seed),
        profile,0.,noise_scale=scale)
    features,q,support,origin=complex_phase_features(observed)
    if not np.array_equal(support[0].cpu().numpy(),ref.support):raise ValueError('native support changed')
    f=features[0].cpu().numpy().copy();logs=q[0].cpu().numpy().copy()
    if not np.isfinite(f).all() or not np.isfinite(logs).all():raise ValueError('nonfinite native features')
    if np.any(logs[~ref.support]!=ref.logs[~ref.support]):raise ValueError('missing support changed')
    f.flags.writeable=False;logs.flags.writeable=False
    metadata=dict(sample_id=ref.sample_id,seed=seed,noise_scale=scale,native_shape=list(values.shape),
        source_complex_sha256=ref.source_sha256,noisy_features_sha256=array_sha(f),
        noisy_log_sha256=array_sha(logs),whole_noisy_source_origin=float(origin.item()),
        parameters=parameters)
    return CompleteField(f,logs,float(origin.item()),metadata)


class CompleteComplexBank:
    def __init__(self,references,profile,device):
        if not references:raise ValueError('nonempty public source set required')
        self.references=references;self.profile=profile;self.device=torch.device(device)
        self.fields=[];self.epoch=None;self.queue=[];self.exposures=Counter()

    def refresh(self,epoch):
        if epoch<0:raise ValueError('nonnegative bank generation required')
        self.fields.clear();self.epoch=None
        for index,ref in enumerate(self.references):
            seed=7400000+epoch*10000+index
            scale=.25+random.Random(seed).random()
            self.fields.append(corrupt_complete(ref,self.profile,self.device,seed,scale))
        self.epoch=epoch
        return dict(epoch=epoch,fields=[f.metadata for f in self.fields],
            noisy_field_RAM_bytes=sum(f.features.nbytes+f.logs.nbytes for f in self.fields))

    def sample(self,rng,*,height=128,width=384,halo=160):
        if self.epoch is None:raise ValueError('complete native fields not prepared')
        if not self.queue:
            self.queue=list(range(len(self.references)));rng.shuffle(self.queue)
        index=self.queue.pop();ref=self.references[index];self.exposures[ref.sample_id]+=1
        h,w=ref.logs.shape[-2:];y=rng.randrange(h-height+1);x=rng.randrange(w-width+1)
        top,left,bottom,right,cy,cx,ch,cw=context_region((h,w),y,x,height,width,halo)
        identity=rng.random()<.2;field=self.fields[index]
        features,logs=(ref.features,ref.logs) if identity else (field.features,field.logs)
        tensors=[torch.from_numpy(v[...,top:bottom,left:right].copy())[None] for v in
                 (features,logs,ref.logs,ref.support)]
        meta=dict(sample_id=ref.sample_id,epoch=self.epoch,identity=identity,
            full_noisy_features_sha256=array_sha(ref.features) if identity else field.metadata['noisy_features_sha256'],
            native_core=[y,x,height,width],native_context=[top,left,bottom,right],
            core_in_context=[cy,cx,ch,cw],full_source_phase_neighbours_preserved=True,
            whole_noisy_source_origin=ref.origin if identity else field.origin)
        return (*tensors,meta)


@torch.inference_mode()
def restore_complete(model,features,support,device,*,use_phase,core=512):
    if features.shape[0]!=3 or support.shape!=(1,*features.shape[-2:]) or core%4:
        raise ValueError('native full-source features, support and aligned core required')
    h,w=support.shape[-2:];delta=np.zeros((1,h,w),np.float32);visited=0
    for y in range(0,h,core):
        for x in range(0,w,core):
            ch,cw=min(core,h-y),min(core,w-x)
            top,left,bottom,right,cy,cx,_,_=context_region((h,w),y,x,ch,cw,model.required_halo)
            f=torch.from_numpy(features[...,top:bottom,left:right].copy())[None].to(device)
            mask=torch.from_numpy(support[...,top:bottom,left:right].copy())[None].to(device)
            result=model.feature_residual(f,mask,use_phase=use_phase)
            delta[...,y:y+ch,x:x+cw]=result[0,...,cy:cy+ch,cx:cx+cw].cpu().numpy()
            visited+=ch*cw
    if visited!=h*w or not np.isfinite(delta).all() or np.count_nonzero(delta[~support]):
        raise ValueError('incomplete or invalid native scalar restoration')
    return delta


def evaluate_pair(models,validation,*,outputs=None):
    reports={};predictions={}
    for condition,(model,device,use_phase) in models.items():
        model.eval();records=[];total_in=total_out=total_id=0.;count=0;identity_cache={}
        predictions[condition]=[]
        for ref,field in validation:
            delta=restore_complete(model,field.features,ref.support,device,use_phase=use_phase)
            if ref.sample_id not in identity_cache:
                identity_cache[ref.sample_id]=restore_complete(model,ref.features,ref.support,device,use_phase=use_phase)
            identity=identity_cache[ref.sample_id];mask=ref.support
            # Float64 metric accumulation; no percentile masking/thresholding.
            before=(field.logs.astype(np.float64)-ref.logs)[mask]
            error=before+delta[mask].astype(np.float64);id_error=identity[mask].astype(np.float64)
            input_sse=float(np.dot(before,before));output_sse=float(np.dot(error,error));identity_sse=float(np.dot(id_error,id_error))
            q=ref.logs[mask].astype(np.float64);weights=np.exp(2*(q-q.max()))
            amp_input=float(np.dot(np.expm1(before)**2,weights)/weights.sum())
            amp_output=float(np.dot(np.expm1(error)**2,weights)/weights.sum())
            if not all(math.isfinite(v) for v in (input_sse,output_sse,identity_sse,amp_input,amp_output)):
                raise ValueError('nonfinite complete-source metrics')
            n=int(mask.sum());count+=n;total_in+=input_sse;total_out+=output_sse;total_id+=identity_sse
            record=dict(sample_id=ref.sample_id,seed=field.metadata['seed'],noise_scale=field.metadata['noise_scale'],
                input_features_sha256=field.metadata['noisy_features_sha256'],native_values=int(mask.size),observed_values=n,
                input_log_RMSE=math.sqrt(input_sse/n),output_log_RMSE=math.sqrt(output_sse/n),
                extra_change_on_original_log_RMSE=math.sqrt(identity_sse/n),
                input_amplitude_energy_relative_RMSE=math.sqrt(amp_input),output_amplitude_energy_relative_RMSE=math.sqrt(amp_output),
                restored_log_sha256=array_sha(field.logs+delta))
            records.append(record)
            if outputs is not None:
                folder=Path(outputs)/condition/f'{ref.sample_id}_scale_{field.metadata["noise_scale"]:.2f}'
                folder.mkdir(parents=True,exist_ok=False)
                for name,values in (('reference_log',ref.logs),('noisy_log',field.logs),
                                     ('restored_log',field.logs+delta),('original_restored_log',ref.logs+identity)):
                    np.save(folder/(name+'.npy'),values[0],allow_pickle=False)
                record['output_directory']=str(folder)
        reports[condition]=dict(cases=records,known_added_input_log_RMSE=math.sqrt(total_in/count),
            known_added_output_log_RMSE=math.sqrt(total_out/count),extra_change_on_original_log_RMSE=math.sqrt(total_id/count),
            selection_score=(total_out+.5*total_id)/count,observed_values=count,
            complete_native_values_processed=sum(r['native_values'] for r in records),
            all_validation_textures_also_in_training=True,independent_scene_generalization=False,real_clean_truth=False)
    return reports
