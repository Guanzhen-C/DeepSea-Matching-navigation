"""Frozen v8/v9 model diagnostics before/after verified complex preprocessing.

Predeclared complete native review regions have full true model context and
the actual complex predecessor. Whole-voyage amplitude/phase statistics are
fixed from the already independently replayed source run, not re-estimated
on favorable crops. This is NOT a complete-voyage neural inference or clean
quality acceptance. No checkpoints are trained or selected using these ROIs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import mmap
import shutil
import struct
import subprocess
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from BandConditionedSonarFeatures import context_features,mosaic_single_side
from CarrierRelativeSonarExperiment import CarrierRelativeSonarWavelet
from CarrierRelativeSonarFeatures import PROTOCOL
from PublicEnvelopeNativeTraining import context_region
from InferMeasuredPublicSonarWavelet import supported_delta
from NativeSonarPaletteStream import amplitude_indices,NativePaletteWriter,decoded_index_rows
from ReviewMeasuredSonarNativeDetails import REGIONS
from AuditMeasuredSonarDetailChanges import describe
from caris_orangecopper import caris_orangecopper


RESERVE=4*1024**3


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())


def read_exact(stream,count):
    pieces=[];remaining=count
    while remaining:
        piece=stream.read(min(remaining,1024**2))
        if not piece:raise EOFError('local context pipe ended prematurely')
        pieces.append(piece);remaining-=len(piece)
    return b''.join(pieces)


def local_context(python,source_run,side,context):
    # Do not resolve the venv executable symlink to /usr/bin/python: Python
    # discovers pyvenv.cfg from the requested executable path.
    executable=Path(python).absolute()
    command=[str(executable),str(Path(__file__).with_name('ExtractBandConditionedSonarContext.py')),
             '--source-run',str(source_run),'--side',str(side),'--context',*[str(v) for v in context]]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        length=struct.unpack('>I',read_exact(process.stdout,4))[0]
        if not 0<length<2*1024**2:raise ValueError('invalid local context metadata size')
        metadata=json.loads(read_exact(process.stdout,length));shape=tuple(metadata['native_shape'])
        if len(shape)!=3 or shape[0]!=8 or metadata['dtype']!='<c16' or np.prod(shape)*16>300*1024**2:
            raise ValueError('bounded lossless native complex128 context required')
        arrays=[]
        for size,digest in zip(metadata['array_bytes'],metadata['array_SHA256']):
            a=np.empty(shape,np.complex128);buffer=memoryview(a).cast('B')
            if len(buffer)!=size:raise ValueError('native byte count differs from context grid')
            cursor=0
            while cursor<len(buffer):
                n=process.stdout.readinto(buffer[cursor:cursor+4*1024**2])
                if not n:raise EOFError('incomplete native complex context payload')
                cursor+=n
            if hashlib.sha256(buffer).hexdigest()!=digest:raise ValueError('local native payload SHA differs')
            arrays.append(a)
        if len(arrays)!=2 or process.stdout.read(1):raise ValueError('original/subband payload coverage differs')
        error=process.stderr.read().decode('utf-8',errors='replace');status=process.wait()
        if status:raise RuntimeError('original context extractor failed: '+error[-2000:])
        if metadata['side']!=side or metadata['model_context']!=list(context):raise ValueError('received another native context')
        return arrays,metadata
    except Exception:
        # Closing pipes is not deletion/termination of unrelated processes.
        process.stdout.close()
        error=process.stderr.read().decode('utf-8',errors='replace');process.wait()
        if error:raise RuntimeError('local original-MAT extraction: '+error[-2000:])
        raise


def plan_region(region,shape,halo=160):
    _,row,col,height,width=region;count,bins=shape
    if col+width<=bins:side=0;radial=bins-col-width
    elif col>=bins:side=1;radial=col-bins
    else:raise ValueError('review region cannot straddle the two native sides')
    first=row//4-1;phase=row%4;records=(phase+height+3)//4+1
    if first<0 or first+records>count:raise ValueError('true preceding/current native records are missing')
    top,left,bottom,right,cy,cx,_,_=context_region(shape,first,radial,records,width,halo)
    return dict(side=side,first=first,records=records,phase=phase,radial=radial,
                context=[top,left,bottom,right],core_in_context=[cy,cx,records,width])


def load_models(roots,device):
    models={};selected={}
    for version,root in roots.items():
        report=read(root/'run_summary.json')
        if (report['status']!='completed_equal_input_phase_feature_ablation_real_denoising_unverified'
                or report['smoke'] or not report['all_training_sources_seen']
                or report['model_family']!=CarrierRelativeSonarWavelet.family
                or report['phase_feature_protocol']!=PROTOCOL):
            raise ValueError('completed frozen original public model family required')
        for name in ('CarrierRelativeSonarExperiment.py','CarrierRelativeSonarFeatures.py',
                     'PhaseAwareSonarWavelet.py','PublicSonarWavelet.py',
                     'PublicEnvelopeCoherentModel.py','PublicSonarEnvelopeModel.py'):
            if sha(Path(__file__).with_name(name))!=report['source_sha256'][name]:
                raise ValueError('frozen model/feature graph changed: '+name)
        for condition,use_phase in (('amplitude_only',False),('phase_context',True)):
            record=report['selected_by_synthetic_validation_before_private_evaluation'][condition]
            path=root/record['file']
            if sha(path)!=record['sha256']:raise ValueError('public-validation selected checkpoint changed')
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state['use_phase']!=use_phase or state['model_family']!=report['model_family']:
                raise ValueError('frozen checkpoint condition differs')
            model=CarrierRelativeSonarWavelet(width=state['width']).to(device)
            model.load_state_dict(state['model'],strict=True);model.eval()
            name=version+'_'+condition;models[name]=(model,use_phase);selected[name]=record
    return models,selected


def baseline_display_regions(source,palette,regions):
    report=read(source/'run_summary.json');arrays=[np.empty((r[3],r[4]),np.uint8) for r in regions]
    seen=[np.zeros(r[3],bool) for r in regions]
    for row,indices in enumerate(decoded_index_rows(Path(report['native_PNG_path']),palette)):
        for region,array,done in zip(regions,arrays,seen):
            _,top,left,height,width=region
            if top<=row<top+height:array[row-top]=indices[left:left+width];done[row-top]=True
    if not all(s.all() for s in seen):raise ValueError('full native baseline PNG lacks declared review regions')
    return arrays


def check_original_mosaic(actual,reference):
    # The h5py process uses NumPy 1.21 and the ML process NumPy 2.2. Their
    # complex abs/hypot may differ by an f64 ulp even when the complex input
    # bytes are identical. Keep the original established f64 reconstruction
    # tolerance AND require every displayed native index to be identical.
    np.testing.assert_allclose(actual,reference,rtol=2e-13,atol=1e-14)
    if not np.array_equal(amplitude_indices(actual),amplitude_indices(reference)):
        raise ValueError('native original display differs despite roundoff-tolerant f64 comparison')
    return float(np.max(np.abs(actual-reference)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','v8-training','v9-training','h5-python','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--disk-reserve-gib',type=float,default=4.)
    args=p.parse_args()
    if not np.isfinite(args.disk_reserve_gib) or args.disk_reserve_gib<3.:
        raise ValueError('at least three GiB of explicitly declared free disk reserve required')
    reserve_bytes=int(args.disk_reserve_gib*1024**3)
    source=read(args.source_run/'run_summary.json');source_audit=read(args.source_run/'independent_native_replay_audit.json')
    if (source_audit['status']!='passed_complete_original_MAT_replay_and_native_PNG_not_quality_acceptance'
            or source_audit['source_run_summary_SHA256']!=sha(args.source_run/'run_summary.json')):
        raise ValueError('complete independent all-native preprocessing audit required')
    shape=source['native_shape'][-2:];reference=np.load(source['reference_mosaic'],mmap_mode='r')
    limits=np.load(Path(source['envelope_cache'])/'bottom_first_ranges.npy')
    c=read(Path(source['envelope_cache'])/'run_summary.json')['parts'][0]['parameters']['c']
    fs=source['sampling_frequency_Hz'];palette=(caris_orangecopper(256)*255).astype(np.uint8)
    torch.set_num_threads(2);device=torch.device(args.device)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.4,device)
    roots={'v8':args.v8_training,'v9':args.v9_training};models,selected=load_models(roots,device)
    old_audit=read(args.v8_training/'independent_public_pair_audit.json')
    if not old_audit.get('selected_amplitude_control_tensors_exactly_match_v7'):
        raise ValueError('reuse of v7 amplitude field requires proven identical v8 tensors')
    project=Path(__file__).resolve().parents[1];real=project/'output/real_sonar'
    replay_roots=dict(v8_amplitude_only=real/'public_complex_phase_transfer_20261105_v7/amplitude_only',
        v8_phase_context=real/'public_carrier_relative_transfer_20261105_v8/phase_context',
        v9_amplitude_only=real/'public_irregular_complex_transfer_20261105_v9/amplitude_only',
        v9_phase_context=real/'public_irregular_complex_transfer_20261105_v9/phase_context')
    replay={}
    for name,path in replay_roots.items():
        checked=read(path/'independent_native_and_stripe_audit.json')
        field=path/'per_beam_supported_log_delta_float32.npy'
        if (checked['status']!='passed_full_native_numerical_integrity_not_real_denoising_acceptance'
                or sha(field)!=checked['checked_numeric_output_sha256'][field.name]):
            raise ValueError('complete old original-input field audit required')
        replay[name]=np.load(field,mmap_mode='r')
    root=args.output.resolve()
    canvas_shape=(100+len(REGIONS)*1094,4*1024+3*16)
    if np.prod(canvas_shape)+2*1024**2>24*1024**2:
        raise ValueError('diagnostic artifact must stay inside the declared 24 MiB budget')
    if shutil.disk_usage(root.parent).free<reserve_bytes+24*1024**2:
        raise ValueError('declared reserve plus bounded indexed detail display required')
    root.mkdir(parents=True,exist_ok=False)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for name in (Path(__file__).name,'ExtractBandConditionedSonarContext.py','BandConditionedSonarFeatures.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    report=dict(status='probing_frozen_models_before_after_whole_source_subband',regions=[],
        selected_public_checkpoints=selected,source_run_summary_SHA256=sha(args.source_run/'run_summary.json'),
        frozen_training_sources={key:dict(path=str(path.resolve()),report_SHA256=sha(path/'run_summary.json')) for key,path in roots.items()},
        ML_runtime=dict(torch=str(torch.__version__),numpy=np.__version__,device=str(device),
                        TF32=False,cudnn_deterministic=True),
        all_model_weights_frozen=True,private_training_or_checkpoint_selection=False,
        whole_source_statistics_fixed_not_estimated_on_ROIs=True,model_context_factor4_aligned=True,
        source_complex_read_via_local_RAM_pipe=True,private_raw_data_uploaded=False,
        diagnostic_is_not_complete_voyage_neural_inference=True,real_clean_truth=False,
        quality_accepted=False,source_code_SHA256={f.name:sha(f) for f in snapshot.iterdir()},
        model_feature_protocol=PROTOCOL,original_input_replay_tolerance_absolute=2e-4,
        declared_disk_reserve_bytes=reserve_bytes,diagnostic_artifact_budget_bytes=24*1024**2,
        native_display_pixels_resized=False,fixed_original_CARIS_limits=[0,1])
    def save():
        temp=root/'run_summary.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.replace(root/'run_summary.json')
    save();started=time.monotonic()
    native_baselines=baseline_display_regions(args.source_run,palette,REGIONS)
    canvas=np.zeros(canvas_shape,np.uint8)
    header=Image.fromarray(canvas,'P');header.putpalette(palette.reshape(-1).tolist())
    fontpath=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font=ImageFont.truetype(str(fontpath),30) if fontpath.exists() else ImageFont.load_default()
    draw=ImageDraw.Draw(header)
    for col,label in enumerate(('原始实测','仅频带预处理','预处理 + V8相位（未验收）','预处理 + V9相位（未验收）')):
        draw.text((col*1040+12,24),label,font=font,fill=255)
    try:
        for ordinal,region in enumerate(REGIONS):
            name,row,col,height,width=region;plan=plan_region(region,shape)
            arrays,metadata=local_context(args.h5_python,args.source_run.resolve(),plan['side'],plan['context'])
            if metadata['source_run_summary_SHA256']!=report['source_run_summary_SHA256']:
                raise ValueError('whole-source statistics or processing changed')
            cy,cx,n,w=plan['core_in_context'];extra=int(metadata['actual_left_predecessor_included'])
            cores=[abs(z[:,cy:cy+n,extra+cx:extra+cx+w]) for z in arrays]
            raw_mosaic=mosaic_single_side(cores[0],plan['side'],phase=plan['phase'],height=height)
            baseline=mosaic_single_side(cores[1],plan['side'],phase=plan['phase'],height=height)
            original=np.asarray(reference[row:row+height,col:col+width])
            raw_roundoff=check_original_mosaic(raw_mosaic,original)
            if not np.array_equal(amplitude_indices(baseline),native_baselines[ordinal]):
                raise ValueError('local preprocessing does not reproduce the independently audited full native PNG')
            record=dict(name=name,coordinates=list(region[1:]),plan=plan,extraction=metadata,
                        raw_mosaic_reproduces_all_original_pixels=True,
                        raw_mosaic_float64_max_absolute_roundoff=raw_roundoff,
                        raw_mosaic_float64_comparison_tolerance=dict(rtol=2e-13,atol=1e-14),
                        subband_mosaic_reproduces_all_full_native_display_indices=True,conditions={},feature_inputs={})
            shown={'original':original,'subband':baseline}
            for source_index,source_name,stats_key in ((0,'original','raw_complete_complex_statistics'),
                                                       (1,'subband','filtered_complete_complex_statistics')):
                features=[];masks=[]
                for beam in range(8):
                    stats=source[stats_key][plan['side']]['beams'][beam]
                    f,mask,_=context_features(arrays[source_index][beam],stats['mean_positive_log_amplitude'],
                        complex(*stats['adjacent_unit_phase_mean']),first_column=plan['context'][1])
                    features.append(f);masks.append(mask)
                record['feature_inputs'][source_name]=dict(
                    feature_SHA256=[hashlib.sha256(f.tobytes()).hexdigest() for f in features],
                    observation_mask_SHA256=[hashlib.sha256(m.tobytes()).hexdigest() for m in masks],
                    whole_source_log_origins=[b['mean_positive_log_amplitude'] for b in source[stats_key][plan['side']]['beams']],
                    whole_source_unit_phase_means=[b['adjacent_unit_phase_mean'] for b in source[stats_key][plan['side']]['beams']])
                for model_name,(model,use_phase) in models.items():
                    delta=np.empty((8,n,w),np.float32)
                    with torch.inference_mode():
                        for beam in range(8):
                            f=torch.from_numpy(features[beam])[None].to(device)
                            mask=torch.from_numpy(masks[beam])[None,None].to(device)
                            prediction=model.feature_residual(f,mask,use_phase=use_phase)[0,0].cpu().numpy()[cy:cy+n,cx:cx+w]
                            delta[beam]=supported_delta(cores[source_index][beam],prediction,
                                limits[plan['side'],plan['first']:plan['first']+n],
                                np.arange(plan['radial'],plan['radial']+w),c,fs)
                    item={}
                    if source_index==0:
                        previous=np.asarray(replay[model_name][plan['side'],:,plan['first']:plan['first']+n,
                                            plan['radial']:plan['radial']+w])
                        difference=float(np.max(abs(previous-delta)))
                        if difference>2e-4:raise ValueError('native original-input frozen model replay differs: '+model_name+': '+str(difference))
                        item['original_input_replay_max_absolute_log_delta_error']=difference
                        replay[model_name]._mmap.madvise(mmap.MADV_DONTNEED)
                    restored=mosaic_single_side(cores[source_index]*np.exp(delta.astype(np.float64)),
                        plan['side'],phase=plan['phase'],height=height)
                    base=raw_mosaic if source_index==0 else baseline
                    item.update(changes_vs_original_observation=describe(original,restored),
                                additional_changes_vs_input=describe(base,restored),
                                native_mosaic_SHA256=hashlib.sha256(restored.tobytes()).hexdigest(),
                                supported_per_beam_delta_SHA256=hashlib.sha256(delta.tobytes()).hexdigest())
                    record['conditions'][source_name+'_'+model_name]=item
                    if source_index==1 and model_name in ('v8_phase_context','v9_phase_context'):
                        shown[model_name]=restored.copy()
                del features,masks
            yy=100+ordinal*1094
            draw.text((12,yy+14),name+'（固定原尺寸诊断，不是全航段）',font=font,fill=255)
            for index,key in enumerate(('original','subband','v8_phase_context','v9_phase_context')):
                panel=Image.fromarray(amplitude_indices(shown[key][::-1]),'P');panel.putpalette(palette.reshape(-1).tolist())
                header.paste(panel,(index*1040,yy+70))
            report['regions'].append(record);report['elapsed_s']=time.monotonic()-started;save()
            print(json.dumps(dict(region=name,conditions=len(record['conditions']),elapsed_s=report['elapsed_s'])),flush=True)
            del arrays,cores,shown
        path=root/'固定原尺寸_原始_频带_V8_V9_冻结模型诊断_未验收.png'
        writer=NativePaletteWriter(path,canvas_shape[1],canvas_shape[0],palette,reserve_bytes=reserve_bytes)
        try:writer.append(np.asarray(header));encoding=writer.finish()
        finally:writer.close()
        report.update(status='completed_frozen_original_vs_subband_native_context_probe_not_voyage_acceptance',
                      figure=str(path),figure_SHA256=sha(path),figure_encoding=encoding)
    except Exception as exc:
        report.update(status='failed_preserved_band_conditioned_probe_no_automatic_restart',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_s']=time.monotonic()-started;save()
    print(json.dumps({k:report.get(k) for k in ('status','figure','elapsed_s','quality_accepted')}),flush=True)


if __name__=='__main__':main()
