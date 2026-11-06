"""Complete-source synthetic fields and real-neighbor, core-only supervision.

Only public measurements enter this training bank. Native observations are
never spatially resized; zeros, acquisition order and source units are kept.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch

from TrainPublicSonarEnvelope import corruption, read_native, resolve


def context_region(shape,y,x,height=128,width=384,halo=96):
    h,w=shape
    if min(height,width)<1 or halo<0 or halo%4 or y<0 or x<0 or y+height>h or x+width>w:
        raise ValueError('valid original native core and factor-four halo required')
    top=max(0,((y-halo)//4)*4);left=max(0,((x-halo)//4)*4)
    bottom=min(h,((y+height+halo+3)//4)*4)
    right=min(w,((x+width+halo+3)//4)*4)
    return top,left,bottom,right,y-top,x-left,height,width


@dataclass
class NativeReference:
    sample_id:str
    values:np.ndarray
    support:np.ndarray
    origin:float
    digest:str

    @classmethod
    def from_record(cls,row,mappings):
        if row.get('domain')!='envelope':raise ValueError('explicit real-envelope source required; complex images need their typed adapter')
        a=read_native(str(resolve(row['path'],mappings)),row['cache_sha256'])
        if np.iscomplexobj(a):raise ValueError('complex data must not be silently cast to a real envelope')
        support=np.asarray(a)>0;q=np.zeros(a.shape,np.float32)
        for start in range(0,len(a),32):
            v=np.asarray(a[start:start+32]);mask=support[start:start+32]
            if not np.isfinite(v).all() or (v<0).any():raise ValueError('invalid native public amplitudes')
            q[start:start+32][mask]=np.log(v[mask]).astype(np.float32)
        if not support.any():raise ValueError('empty observed source')
        origin=float(q[support].mean(dtype=np.float64))
        digest=hashlib.sha256(q.tobytes()).hexdigest()
        q.flags.writeable=False;support.flags.writeable=False
        return cls(row['sample_id'],q,support,origin,digest)


class NativeFieldBank:
    def __init__(self,references,profile,device,variants=4):
        if not references or not 1<=variants<=8:raise ValueError('bounded public field bank required')
        self.references,self.profile,self.device,self.variants=references,profile,torch.device(device),variants
        self.fields={};self.epoch=None

    @torch.inference_mode()
    def refresh(self,epoch):
        if epoch<0:raise ValueError('nonnegative bank generation required')
        fresh={};ledger=[]
        for index,ref in enumerate(self.references):
            target=torch.from_numpy(ref.values.copy())[None,None].to(self.device)
            support=torch.from_numpy(ref.support.copy())[None,None].to(self.device)
            entries=[]
            for variant in range(self.variants):
                seed=5100500+epoch*100000+index*100+variant
                generator=torch.Generator(device=self.device).manual_seed(seed)
                noisy,parameters=corruption(target,support,generator,self.profile,0.,'realized_log_span')
                array=noisy[0,0].cpu().numpy().copy()
                if not np.isfinite(array).all() or not np.array_equal(array[~ref.support],ref.values[~ref.support]):
                    raise ValueError('full native synthetic field changed missing support')
                origin=float(array[ref.support].mean(dtype=np.float64))
                digest=hashlib.sha256(array.tobytes()).hexdigest();array.flags.writeable=False
                entries.append(dict(values=array,origin=origin,digest=digest))
                ledger.append(dict(sample_id=ref.sample_id,variant=variant,seed=seed,
                    source_shape=list(array.shape),source_log_values_sha256=ref.digest,
                    noisy_input_sha256=digest,noisy_whole_source_log_origin=origin,
                    corruption_parameters=parameters))
            fresh[index]=entries
            del target,support,noisy
        self.fields=fresh;self.epoch=epoch
        return dict(epoch=epoch,fields=ledger,
            RAM_noisy_field_bytes=sum(entry['values'].nbytes for group in fresh.values() for entry in group),
            synthetic_fields_generated_at_complete_native_source_dimensions=True)

    def sample(self,rng,augmentation_rng,height=128,width=384,halo=96):
        if self.epoch is None:raise ValueError('native field bank has not been generated')
        index=rng.randrange(len(self.references));ref=self.references[index]
        h,w=ref.values.shape
        y=rng.randrange(h-height+1);x=rng.randrange(w-width+1)
        top,left,bottom,right,cy,cx,ch,cw=context_region((h,w),y,x,height,width,halo)
        identity=augmentation_rng.random()<.2
        if identity:
            variant=None;values=ref.values;origin=ref.origin;digest=ref.digest
        else:
            variant=augmentation_rng.randrange(self.variants);entry=self.fields[index][variant]
            values,origin,digest=entry['values'],entry['origin'],entry['digest']
        target=torch.from_numpy(ref.values[top:bottom,left:right].copy())[None,None]
        noisy=torch.from_numpy(values[top:bottom,left:right].copy())[None,None]
        support=torch.from_numpy(ref.support[top:bottom,left:right].copy())[None,None]
        metadata=dict(sample_id=ref.sample_id,bank_epoch=self.epoch,variant=variant,identity=identity,
            original_source_log_sha256=ref.digest,full_noisy_input_sha256=digest,
            native_core=[y,x,height,width],native_context=[top,left,bottom,right],
            core_in_context=[cy,cx,ch,cw],whole_noisy_source_origin=origin,
            context_top_left_factor4_aligned=True,artificial_missing_neighbor_padding=False)
        return target,noisy,support,metadata


def amplitude_fidelity_loss(predicted_delta,known_delta,reference_log,support):
    """Unit-invariant amplitude-energy fidelity, with no intensity cutoff.

    exp(predicted_delta-known_delta) is restored/reference amplitude. Native
    reference energy weights let strong observed returns influence fidelity.
    This is known synthetic corruption recovery, never clean sonar truth.
    """
    if predicted_delta.shape!=known_delta.shape or reference_log.shape!=known_delta.shape or support.shape!=known_delta.shape:
        raise ValueError('aligned native core and amplitude fidelity labels required')
    mask=support.to(reference_log.dtype)
    maximum=torch.where(support,reference_log,reference_log.new_full((),-torch.inf)).amax((-2,-1),keepdim=True)
    maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
    shifted=torch.where(support,reference_log-maximum,torch.zeros_like(reference_log))
    weights=torch.exp(2*shifted)*mask
    total=weights.sum((-2,-1),keepdim=True).clamp_min(1e-20)
    error=torch.expm1(predicted_delta-known_delta)
    input_error=torch.expm1(-known_delta)
    output_energy=(error.square()*weights).sum((-2,-1),keepdim=True)/total
    input_energy=(input_error.square()*weights).sum((-2,-1),keepdim=True)/total
    return (output_energy/(input_energy+.01**2)).mean()
