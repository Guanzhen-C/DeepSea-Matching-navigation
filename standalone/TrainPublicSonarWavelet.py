"""Train our public-data wavelet model on explicitly known injected stripes.

Measured originals are reference observations, NOT absolutely clean targets.
Real interference removal is a separate, still-required evaluation task.
No private MATs, thermal pretrained weights, resizing or test-source fitting.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import shutil
import time

import numpy as np
import torch

from PublicSonarData import PublicSonarSample, encode, load_display, validate_splits
from PublicSonarWavelet import PublicSonarWavelet, haar
from PublicSonarDisplayLuminance import display_luminance
from AcousticStripeAugmentation import acoustic_stripes


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024**2),b""):
            h.update(block)
    return h.hexdigest()


def resolve_path(path, mappings):
    p=Path(path)
    for source,destination in mappings:
        try:return Path(destination)/p.relative_to(source)
        except ValueError:pass
    return p


def protection_boxes(path,height,width):
    mask=np.ones((height,width),np.float32)
    if path is None:return mask,0
    count=0
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        fields=line.split()
        if len(fields)!=5:raise ValueError("reviewed normalized YOLO boxes require five fields")
        values=np.array([float(x) for x in fields])
        if not np.isfinite(values).all() or values[0]!=int(values[0]) or np.any(values[1:]<0) or np.any(values[1:]>1):
            raise ValueError("invalid original object annotation")
        _,cx,cy,w,h=values
        x0,x1=max(0,math.floor((cx-w/2)*width)),min(width,math.ceil((cx+w/2)*width))
        y0,y1=max(0,math.floor((cy-h/2)*height)),min(height,math.ceil((cy+h/2)*height))
        mask[y0:y1,x0:x1]=2.
        count+=1
    return mask,count


@lru_cache(maxsize=12)
def native_record(serialized,mapping_json):
    row=json.loads(serialized)
    mappings=json.loads(mapping_json)
    path=resolve_path(row["path"],mappings)
    expected=row.get("cache_sha256",row["source_sha256"])
    if sha(path)!=expected:raise ValueError("public training input differs from frozen manifest")
    if row["domain"]=="display_rgb":
        sample=load_display(path,sample_id=row["sample_id"],source=row["source"],
                            group=row["group_id"],license=row["license"])
    else:
        sample=PublicSonarSample(row["sample_id"],row["domain"],np.load(path,allow_pickle=False),row)
    if sample.domain=="display_rgb":
        q,support=display_luminance(sample)
    else:
        q,support=encode(sample,amplitude_scale=row.get("fixed_instrument_amplitude_scale",1.))
    label=resolve_path(row["annotation_file"],mappings) if row.get("annotation_file") else None
    weights,boxes=protection_boxes(label,*support.shape)
    weights*=support
    return q,weights,boxes


def crop(row,mapping_json,rng,device,center=False):
    q,weights,boxes=native_record(json.dumps(row,sort_keys=True),mapping_json)
    h,w=min(128,q.shape[1]),min(384,q.shape[2])
    y=(q.shape[1]-h)//2 if center else rng.randrange(q.shape[1]-h+1)
    x=(q.shape[2]-w)//2 if center else rng.randrange(q.shape[2]-w+1)
    target=torch.from_numpy(q[:,y:y+h,x:x+w].copy())[None].to(device)
    weight=torch.from_numpy(weights[y:y+h,x:x+w].copy())[None,None].to(device)
    return target,weight,boxes


def corrupt(target,domain,generator,identity_probability=.2):
    """Measurement-domain hypotheses, not calibrated physical interference."""
    def rand(lo,hi):return lo+(hi-lo)*torch.rand((),device=target.device,generator=generator)
    if float(rand(0,1))<identity_probability:return target.clone(),True
    h,w=target.shape[-2:]
    y=torch.arange(h,device=target.device,dtype=target.dtype)[:,None]-h/2
    x=torch.arange(w,device=target.device,dtype=target.dtype)[None,:]-w/2
    phase=2*torch.pi*(rand(.015,.22)*y+rand(-.015,.015)*x+rand(-.00001,.00001)*x*x)+rand(-torch.pi,torch.pi)
    wave=torch.cos(phase)
    stripe=torch.relu(wave)**rand(1.,4.)
    stripe=stripe+rand(0,.3)*torch.cos(2*phase+rand(-torch.pi,torch.pi))
    stripe-=stripe.mean()
    envelope=.3+.7*torch.sigmoid(rand(-.03,.03)*x+rand(-1,1))
    delta=rand(.04,.35) if domain==0 else rand(.15,.9)
    damaged=target+delta*stripe[None,None]*envelope[None,None]
    if domain==0:damaged=damaged.clamp(0,1) # display-domain saturation only
    return damaged,False


def objective(predicted,target,weight):
    error=predicted-target
    # Keep the original measured texture and target boxes: no unconditional
    # penalty pushes all directional wavelet coefficients toward zero.
    fidelity=(error.square()*weight).sum()/(weight.sum()*target.shape[1]).clamp_min(1)
    h,w=error.shape[-2:]
    even=error[...,:h-h%2,:w-w%2]
    if min(even.shape[-2:])>=2:
        details=haar(even)
        fidelity=fidelity+.02*details.abs().mean()
    return fidelity


@torch.inference_mode()
def evaluate(model,rows,mapping_json,device,corruptor=corrupt):
    rng=random.Random(12041)
    gen=torch.Generator(device=device).manual_seed(12041)
    input_error=output_error=identity_error=0.
    samples=0
    for row in rows:
        target,weight,_=crop(row,mapping_json,rng,device,center=True)
        damaged,_=corruptor(target,0,gen,identity_probability=0.)
        output=model(damaged,0)
        identity=model(target,0)
        count=float(weight.sum())*3
        input_error+=float(((damaged-target).square()*weight).sum())
        output_error+=float(((output-target).square()*weight).sum())
        identity_error+=float(((identity-target).square()*weight).sum())
        samples+=count
    return dict(known_injected_stripe_input_RMSE=math.sqrt(input_error/samples),
        known_injected_stripe_output_RMSE=math.sqrt(output_error/samples),
        extra_change_on_unmodified_measured_reference_RMSE=math.sqrt(identity_error/samples),
        scope="fixed native display-luminance validation windows from a held-out acquisition year; full-image test pending",
        validation_images=len(rows),real_clean_truth=False,real_interference_removal_proven=False)


def save_report(root,report):
    temporary=root/"run_summary.tmp"
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    temporary.replace(root/"run_summary.json")


def appearance_corruptor(profile):
    def apply(target,domain,generator,identity_probability=.2):
        output,metadata=acoustic_stripes(target,domain,generator,profile,identity_probability)
        return output,metadata["identity"]
    return apply


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--path-map",action="append",default=[],help="Explicit local_prefix=execution_prefix")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--corruption",choices=("acoustic_style","generic"),default="acoustic_style")
    p.add_argument("--noise-profile",type=Path)
    a=p.parse_args()
    if a.steps<1:raise ValueError("positive training step count required")
    if a.corruption=="acoustic_style":
        if a.noise_profile is None:raise ValueError("source-shaped training requires an aggregate noise appearance profile")
        noise_profile=json.loads(a.noise_profile.read_text())
        corruption_fn=appearance_corruptor(noise_profile)
    else:
        noise_profile=None
        corruption_fn=corrupt
    mappings=[x.split("=",1) for x in a.path_map]
    if any(len(x)!=2 for x in mappings):raise ValueError("invalid explicit path mapping")
    mapping_json=json.dumps(mappings)
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    source=json.loads(a.manifest.read_text());rows=source["records"]
    split_audit=validate_splits(rows)
    train_rgb=[r for r in rows if r["split"]=="train" and r["include_in_learning"] and r["domain"]=="display_rgb"]
    train_aux=[r for r in rows if r["split"]=="train_aux" and r["include_in_learning"] and r["domain"]=="envelope"]
    validation=sorted([r for r in rows if r["split"]=="validation" and r["include_in_learning"]],key=lambda r:r["sample_id"])[::24][:24]
    if not train_rgb or not validation:raise ValueError("independent public training/validation groups required")
    snapshot=root/"source_snapshot";snapshot.mkdir()
    for name in ("TrainPublicSonarWavelet.py","PublicSonarWavelet.py","PublicSonarData.py","PublicSonarDisplayLuminance.py","AcousticStripeAugmentation.py","InspectPublicSonarSignals.py","InspectPublicSonarDatasets.py"):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    if noise_profile is not None:shutil.copy2(a.noise_profile,root/"acoustic_stripe_style_profile.json")
    torch.set_num_threads(2);torch.manual_seed(20261104)
    device=torch.device(a.device)
    if device.type=="cuda":
        torch.cuda.set_per_process_memory_fraction(.75,device)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.deterministic=True
    rng=random.Random(20261104)
    generator=torch.Generator(device=device).manual_seed(20261104)
    model=PublicSonarWavelet().to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
    report=dict(status="training_public_wavelet_from_scratch",steps_completed=0,steps_requested=a.steps,
        model_parameters=sum(p.numel() for p in model.parameters()),
        manifest=str(a.manifest),manifest_sha256=sha(a.manifest),split_audit=split_audit,
        source_sha256={f.name:sha(f) for f in snapshot.iterdir()},
        train_RGB_images=len(train_rgb),train_envelope_groups=len(train_aux),
        public_only=True,private_sonar_used=False,thermal_pretrained_weights_used=False,
        target_definition="original public measurements plus known artificial corruption; not clean truth",
        image_training_space="display luminance, not RGB color-map inversion or calibrated acoustic amplitude",
        chromaticity_not_used_as_network_shortcut=True,
        corruption_recipe=a.corruption,
        noise_profile_sha256=sha(a.noise_profile) if a.noise_profile else None,
        local_private_data_aggregate_style_parameters_used=noise_profile is not None,
        private_raw_data_uploaded=False,
        object_box_fidelity_weight=2.,wavelet_residual_loss_weight=.02,
        train_domain_probability_RGB=.8,identity_probability=.2,
        actual_training_source_ids=[],validation_history=[],checkpoints=[],quality_accepted=False,
        full_image_evaluation_complete=False,complex_domain_trained=False,
        torch=str(torch.__version__),cuda=torch.version.cuda,device=str(device))
    started=time.monotonic();used=set();rgb_queue=[];aux_queue=[]
    domain_steps={"display_rgb":0,"envelope":0}
    annotated_steps=0
    save_report(root,report)
    try:
        report["validation_history"].append(dict(step=0,**evaluate(model,validation,mapping_json,device,corruption_fn)))
        for step in range(1,a.steps+1):
            auxiliary=bool(train_aux) and rng.random()>=.8
            queue=aux_queue if auxiliary else rgb_queue
            pool=train_aux if auxiliary else train_rgb
            if not queue:queue.extend(rng.sample(pool,len(pool)))
            row=queue.pop()
            domain=1 if auxiliary else 0
            target,weight,boxes=crop(row,mapping_json,rng,device)
            damaged,identity=corruption_fn(target,domain,generator)
            optimizer.zero_grad(set_to_none=True)
            predicted=model(damaged,domain)
            loss=objective(predicted,target,weight)
            if not torch.isfinite(loss):raise ValueError("nonfinite public-domain training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();used.add(row["sample_id"])
            domain_steps[row["domain"]]+=1
            annotated_steps+=int(boxes>0)
            if step%50==0 or step==a.steps:
                report.update(steps_completed=step,loss=float(loss.detach()),elapsed_s=time.monotonic()-started,
                              actual_training_source_ids=sorted(used),training_steps_by_domain=domain_steps,
                              steps_with_original_object_annotations=annotated_steps,
                              every_training_record_seen=len(used)==len(train_rgb)+len(train_aux))
                save_report(root,report)
                print(f"[public wavelet] {step}/{a.steps}, loss {float(loss.detach()):.6f}, unique sources {len(used)}",flush=True)
            if step in {min(300,a.steps),min(600,a.steps),a.steps}:
                model.eval()
                metrics=evaluate(model,validation,mapping_json,device,corruption_fn)
                report["validation_history"].append(dict(step=step,**metrics))
                checkpoint=root/f"checkpoint_{step:06d}.pt"
                torch.save(dict(model=model.state_dict(),step=step,domain_names=model.domains,
                                manifest_sha256=report["manifest_sha256"],width=model.width),checkpoint)
                report["checkpoints"].append(dict(step=step,file=checkpoint.name,sha256=sha(checkpoint)))
                save_report(root,report);model.train()
                print(json.dumps(dict(step=step,**metrics)),flush=True)
        report["status"]="completed_public_training_full_image_and_real_noise_evaluation_pending"
    except Exception as exc:
        report.update(status="failed_public_training",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)
        save_report(root,report)


if __name__=="__main__":main()
