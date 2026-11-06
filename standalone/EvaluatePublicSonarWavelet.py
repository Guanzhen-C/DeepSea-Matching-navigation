"""Full-native public held-out images, known added stripes and identity tests.

No source resize, no test-set checkpoint selection, and no true-clean claim.
Every test image is processed at its complete original pixel dimensions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from PublicSonarData import load_display, validate_splits
from PublicSonarDisplayLuminance import display_luminance, restore_display
from PublicSonarPseudoColor import display_lookup,render_pseudocolor
from PublicSonarWavelet import PublicSonarWavelet
from TrainPublicSonarWavelet import corrupt, appearance_corruptor, evaluate, protection_boxes, resolve_path, sha


def select_checkpoint(training):
    choices=[]
    for record in training["checkpoints"]:
        matches=[m for m in training["validation_history"] if m["step"]==record["step"]]
        if len(matches)!=1:raise ValueError("checkpoint validation evidence missing or ambiguous")
        m=matches[0]
        score=.5*(m["known_injected_stripe_output_RMSE"]**2+m["extra_change_on_unmodified_measured_reference_RMSE"]**2)
        choices.append((score,record))
    if not choices:raise ValueError("no trained checkpoint available")
    return min(choices,key=lambda x:(x[0],x[1]["step"]))


def errors(target,damaged,predicted,identity,mask):
    n=int(mask.sum())*target.shape[0]
    if n==0:return None
    return dict(samples=n,input_SSE=float(((damaged-target)[:,mask]**2).sum(dtype=np.float64)),
        output_SSE=float(((predicted-target)[:,mask]**2).sum(dtype=np.float64)),
        identity_SSE=float(((identity-target)[:,mask]**2).sum(dtype=np.float64)))


def summarized(items):
    items=[r for r in items if r is not None]
    count=sum(r["samples"] for r in items)
    if not count:return None
    return dict(measured_channel_values=count,
        known_added_stripe_input_RMSE=math.sqrt(sum(r["input_SSE"] for r in items)/count),
        known_added_stripe_output_RMSE=math.sqrt(sum(r["output_SSE"] for r in items)/count),
        extra_change_on_unmodified_reference_RMSE=math.sqrt(sum(r["identity_SSE"] for r in items)/count))


def save(root,report):
    temporary=root/"run_summary.tmp"
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    temporary.replace(root/"run_summary.json")


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--path-map",action="append",default=[])
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--test-noise-profile",type=Path,
                   help="Evaluate all methods using the same calibrated appearance; checkpoint selection still uses validation only")
    a=p.parse_args()
    mappings=[x.split("=",1) for x in a.path_map]
    training=json.loads((a.training/"run_summary.json").read_text())
    if training["status"]!="completed_public_training_full_image_and_real_noise_evaluation_pending":
        raise ValueError("training is not complete")
    if training["manifest_sha256"]!=sha(a.manifest):raise ValueError("evaluation manifest differs from training")
    inference_names=("PublicSonarWavelet.py","PublicSonarData.py","PublicSonarDisplayLuminance.py",
                     "InspectPublicSonarSignals.py","InspectPublicSonarDatasets.py")
    for name in inference_names:
        if sha(Path(__file__).with_name(name))!=training["source_sha256"][name]:raise ValueError("frozen model/input interpretation changed before inference")
    manifest=json.loads(a.manifest.read_text())
    validate_splits(manifest["records"])
    rows=[r for r in manifest["records"] if r["split"] in ("test","external_test") and r["include_in_learning"] and r["domain"]=="display_rgb"]
    profile_path=a.test_noise_profile
    if profile_path is None and training.get("corruption_recipe")=="acoustic_style":
        profile_path=a.training/"acoustic_stripe_style_profile.json"
        if sha(profile_path)!=training["noise_profile_sha256"]:raise ValueError("training appearance profile changed")
        if sha(Path(__file__).with_name("AcousticStripeAugmentation.py"))!=training["source_sha256"]["AcousticStripeAugmentation.py"]:
            raise ValueError("training augmentation implementation changed; request an explicit new benchmark protocol")
    if profile_path is not None:
        profile=json.loads(profile_path.read_text())
        corruptor=appearance_corruptor(profile)
    else:
        corruptor=corrupt
    # A historical trainer script may differ without changing the frozen
    # forward graph. New test corruption is an explicit benchmark protocol,
    # not a claim that the old training has been reproduced under new code.
    revalidated=[]
    if a.test_noise_profile is not None:
        torch.set_num_threads(2)
        validation=sorted([r for r in manifest["records"] if r["split"]=="validation" and r["include_in_learning"]],key=lambda r:r["sample_id"])[::24][:24]
        for candidate in training["checkpoints"]:
            candidate_path=a.training/candidate["file"]
            if sha(candidate_path)!=candidate["sha256"]:raise ValueError("historical checkpoint hash mismatch")
            candidate_state=torch.load(candidate_path,map_location="cpu",weights_only=True)
            candidate_model=PublicSonarWavelet(width=candidate_state["width"]).to(a.device).eval()
            candidate_model.load_state_dict(candidate_state["model"],strict=True)
            metrics=evaluate(candidate_model,validation,json.dumps(mappings),torch.device(a.device),corruptor)
            revalidated.append(dict(step=candidate["step"],**metrics))
            del candidate_model
        selection_evidence=dict(training,validation_history=revalidated)
    else:
        selection_evidence=training
    score,choice=select_checkpoint(selection_evidence)
    checkpoint=a.training/choice["file"]
    if sha(checkpoint)!=choice["sha256"]:raise ValueError("checkpoint SHA differs from training ledger")
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if tuple(state["domain_names"])!=PublicSonarWavelet.domains:raise ValueError("checkpoint measurement-domain interpretation differs")
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    snapshot=root/"source_snapshot";snapshot.mkdir()
    for name in set(("EvaluatePublicSonarWavelet.py","TrainPublicSonarWavelet.py","AcousticStripeAugmentation.py","PublicSonarPseudoColor.py",*inference_names)):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    if profile_path is not None:shutil.copy2(profile_path,root/"test_noise_profile.json")
    torch.set_num_threads(2)
    device=torch.device(a.device)
    if device.type=="cuda":
        torch.cuda.set_per_process_memory_fraction(.75,device)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=PublicSonarWavelet(width=state["width"]).to(device).eval()
    model.load_state_dict(state["model"],strict=True)
    generator=torch.Generator(device=device).manual_seed(20261204)
    report=dict(status="evaluating_every_native_public_test_image",selected_checkpoint=choice,
        selection_rule="minimum equally weighted validation corrupt-reference MSE and identity-change MSE; no test selection",
        validation_selection_score=score,source_manifest_sha256=sha(a.manifest),cases=[],
        total_images=len(rows),completed_images=0,images_resized=False,
        private_data_used=False,real_clean_truth=False,quality_accepted=False,
        synthetic_generator_scope=("source-shaped irregular bands with native-width-normalized curvature" if profile_path is not None else "generic baseline formula; curvature extent grows with native width"),
        test_corruption_recipe="acoustic_style" if profile_path is not None else "generic",
        test_noise_profile_sha256=sha(profile_path) if profile_path is not None else None,
        checkpoint_validation_recomputed=revalidated,
        display_renderer="source pseudocolor lookup, no blue-chroma amplification",
        pending_domain="complex_field not trained in this first public model",
        source_sha256={f.name:sha(f) for f in snapshot.iterdir()})
    started=time.monotonic();save(root,report)
    try:
        for index,row in enumerate(rows):
            path=resolve_path(row["path"],mappings)
            if sha(path)!=row["source_sha256"]:raise ValueError("held-out public original changed")
            sample=load_display(path,sample_id=row["sample_id"],source=row["source"],group=row["group_id"],license=row["license"])
            q,_=display_luminance(sample)
            tensor=torch.from_numpy(q.copy())[None].to(device)
            damaged,_=corruptor(tensor,0,generator,identity_probability=0.)
            identity=model(tensor,0)[0].cpu().numpy()
            prediction=model(damaged,0)[0].cpu().numpy()
            noisy=damaged[0].cpu().numpy()
            h,w=q.shape[-2:]
            if identity.shape!=q.shape or prediction.shape!=q.shape:raise ValueError("native output grid changed")
            label=resolve_path(row["annotation_file"],mappings) if row.get("annotation_file") else None
            weights,boxes=protection_boxes(label,h,w)
            observed=sample.values.any(-1)
            folder=root/f"{index:03d}_{row['dataset']}";folder.mkdir()
            Image.fromarray(sample.values).save(folder/"original_native.png")
            restored_rgb=render_pseudocolor(sample,identity,display_lookup(sample))
            Image.fromarray(restored_rgb).save(folder/"model_on_original_unverified.png")
            np.save(folder/"predicted_original_luminance_3CHW.npy",identity,allow_pickle=False)
            # Full-resolution comparison. These are measured originals and
            # model changes, not a visual proof of recovered clean imagery.
            if row["dataset"]=="SubPipe" or index<4:
                Image.fromarray(np.concatenate((sample.values,restored_rgb),axis=1)).save(folder/"original_vs_model_full_native.png")
                np.save(folder/"known_added_stripe_prediction_3CHW.npy",prediction,allow_pickle=False)
            case=dict(sample_id=row["sample_id"],dataset=row["dataset"],output=str(folder),native_shape=[h,w],
                native_output_PNG_size=[w,h],object_boxes=boxes,source_sha256=row["source_sha256"],
                original_black_pixels_preserved=bool(np.array_equal(restored_rgb[~observed],sample.values[~observed])),
                known_augmented_input_sha256=hashlib.sha256(noisy.tobytes()).hexdigest(),
                all_pixels=errors(q,noisy,prediction,identity,np.ones((h,w),bool)),
                nonblack_observed_pixels=errors(q,noisy,prediction,identity,observed),
                object_box_pixels=errors(q,noisy,prediction,identity,weights>1),
                original_reference_is_clean_truth=False)
            report["cases"].append(case);report["completed_images"]+=1
            if index%5==0:save(root,report)
            print(f"[public full-native] {index+1}/{len(rows)}, {row['dataset']}, {w}x{h}",flush=True)
            del tensor,damaged
        report.update(status="completed_every_native_test_image_known_corruption_and_identity_evaluation",
            summary_all_pixels=summarized([r["all_pixels"] for r in report["cases"]]),
            summary_nonblack_observed=summarized([r["nonblack_observed_pixels"] for r in report["cases"]]),
            summary_object_boxes=summarized([r["object_box_pixels"] for r in report["cases"]]),
            summary_by_dataset={d:summarized([r["nonblack_observed_pixels"] for r in report["cases"] if r["dataset"]==d]) for d in sorted({r["dataset"] for r in report["cases"]})},
            completed_counts_by_dataset=dict(Counter(r["dataset"] for r in report["cases"])),
            real_interference_removal_still_requires_visual_and_task_specific_validation=True)
    except Exception as exc:
        report.update(status="failed_full_native_public_evaluation",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,
                      peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)
        save(root,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("cases","source_sha256")}),flush=True)


if __name__=="__main__":main()
