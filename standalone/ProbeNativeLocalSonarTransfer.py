"""Fixed native diagnostics of the NEW same-device model on noisy Sept24.

No retraining or checkpoint selection uses these previously defined regions.
The model predicts scalar gains on observed complex context; original float64
amplitudes remain the residual base. Neither an unpaired processed image nor a
band-limited candidate is called clean truth. A full-voyage result is separate.
"""
from pathlib import Path
import argparse
import json
import mmap
import shutil
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from LocalRelativeSonarFeatures import local_features,PROTOCOL
from NativeLocalSonarWavelet import NativeLocalSonarWavelet
from TrainNativeLocalSonarWavelet import sha,array_sha
from ProbeBandConditionedSonarWavelet import local_context,plan_region,baseline_display_regions,check_original_mosaic
from BandConditionedSonarFeatures import mosaic_single_side
from InferMeasuredPublicSonarWavelet import supported_delta
from NativeSonarPaletteStream import amplitude_indices,NativePaletteWriter
from ReviewMeasuredSonarNativeDetails import REGIONS
from AuditMeasuredSonarDetailChanges import describe
from caris_orangecopper import caris_orangecopper


def load_selected(root,device):
    report=json.loads((root/'run_summary.json').read_text())
    if (report['status']!='completed_same_device_known_added_noise_pair_real_denoising_not_accepted'
            or report['smoke'] or report['model_family']!=NativeLocalSonarWavelet.family
            or report['feature_protocol']!=PROTOCOL or not report['all_train_side_beam_sources_seen']):
        raise ValueError('completed full native training/validation required before measured diagnostics')
    for name,digest in report['source_code_SHA256'].items():
        if sha(root/'source_snapshot'/name)!=digest or sha(Path(__file__).with_name(name))!=digest:
            raise ValueError('frozen model execution implementation differs: '+name)
    item=report['selected_by_full_native_known_noise_validation']['phase_context']
    path=root/item['file']
    if sha(path)!=item['SHA256']:raise ValueError('full-native-validation-selected checkpoint differs')
    state=torch.load(path,map_location=device,weights_only=True)
    if state['feature_protocol']!=PROTOCOL or state['use_phase'] is not True:
        raise ValueError('selected model input semantics differ')
    model=NativeLocalSonarWavelet(state['width']).to(device);model.load_state_dict(state['model'],strict=True)
    return model.eval(),item


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','source-run','h5-python','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    device=torch.device(args.device);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.4,device)
    model,selected=load_selected(args.training,device)
    source=json.loads((args.source_run/'run_summary.json').read_text())
    audit=json.loads((args.source_run/'independent_native_replay_audit.json').read_text())
    if (audit['status']!='passed_complete_original_MAT_replay_and_native_PNG_not_quality_acceptance'
            or audit['source_run_summary_SHA256']!=sha(args.source_run/'run_summary.json')):
        raise ValueError('complete independently replayed source preprocessing required')
    reference=np.load(source['reference_mosaic'],mmap_mode='r')
    cache=Path(source['envelope_cache']);limits=np.load(cache/'bottom_first_ranges.npy')
    c=json.loads((cache/'run_summary.json').read_text())['parts'][0]['parameters']['c']
    fs=source['sampling_frequency_Hz'];shape=source['native_shape'][-2:]
    palette=(caris_orangecopper(256)*255).astype(np.uint8)
    root=args.output.absolute();reserve=int(3.75*1024**3);budget=24*1024**2
    if shutil.disk_usage(root.parent).free<reserve+budget:raise ValueError('24MiB native diagnostic plus 3.75GiB disk reserve required')
    root.mkdir(parents=True,exist_ok=False)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    names=(Path(__file__).name,'LocalRelativeSonarFeatures.py','NativeLocalSonarWavelet.py',
        'ExtractBandConditionedSonarContext.py','ProbeBandConditionedSonarWavelet.py',
        'BandConditionedSonarFeatures.py','InferMeasuredPublicSonarWavelet.py')
    for name in names:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    report=dict(status='diagnosing_frozen_same_device_model_on_fixed_real_contexts',regions=[],
        selected_checkpoint=selected,training_report_SHA256=sha(args.training/'run_summary.json'),
        source_run_summary_SHA256=sha(args.source_run/'run_summary.json'),source_code_SHA256={f.name:sha(f) for f in snapshot.iterdir()},
        model_feature_protocol=PROTOCOL,real_native_inputs_explicitly_converted_complex64_for_features=True,
        original_float64_amplitudes_preserved_as_output_base=True,real_clean_truth=False,
        private_data_uploaded=False,real_ROIs_used_for_checkpoint_selection=False,
        diagnostic_is_not_full_voyage_neural_inference=True,native_pixels_resized=False,
        fixed_original_CARIS_limits=[0,1],disk_reserve_bytes=reserve,artifact_budget_bytes=budget,quality_accepted=False)
    start=time.monotonic()
    def save():
        report['elapsed_s']=time.monotonic()-start
        tmp=root/'run_summary.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');tmp.replace(root/'run_summary.json')
    save()
    baselines=baseline_display_regions(args.source_run,palette,REGIONS)
    canvas=Image.fromarray(np.zeros((100+len(REGIONS)*1094,4144),np.uint8),'P')
    canvas.putpalette(palette.reshape(-1).tolist());draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',25)
    labels=('原始观测（不是干净真值）','仅负频带候选','负频带＋同设备学习小波','原始复数＋同设备学习小波')
    for col,label in enumerate(labels):draw.text((col*1040+12,24),label,font=font,fill=255)
    try:
        for ordinal,region in enumerate(REGIONS):
            name,row,col,height,width=region;plan=plan_region(region,shape,halo=208)
            arrays,metadata=local_context(args.h5_python,args.source_run.absolute(),plan['side'],plan['context'])
            cy,cx,n,w=plan['core_in_context'];extra=int(metadata['actual_left_predecessor_included'])
            cores=[abs(z[:,cy:cy+n,extra+cx:extra+cx+w]) for z in arrays]
            original=np.asarray(reference[row:row+height,col:col+width])
            original_actual=mosaic_single_side(cores[0],plan['side'],phase=plan['phase'],height=height)
            roundoff=check_original_mosaic(original_actual,original)
            baseline=mosaic_single_side(cores[1],plan['side'],phase=plan['phase'],height=height)
            if not np.array_equal(amplitude_indices(baseline),baselines[ordinal]):
                raise ValueError('preprocessed context differs from full native display')
            record=dict(name=name,coordinates=list(region[1:]),plan=plan,extraction=metadata,
                original_float64_max_roundoff=roundoff,all_original_and_preprocessed_display_indices_checked=True,conditions={})
            shown=[original,baseline]
            outputs={}
            for which,label in ((0,'original'),(1,'subband')):
                deltas=np.empty((8,n,w),np.float32);feature_hashes=[]
                with torch.inference_mode():
                    for beam in range(8):
                        z=np.asarray(arrays[which][beam],np.complex64)
                        f,_,mask=local_features(z,first_column=plan['context'][1])
                        feature_hashes.append(array_sha(f))
                        prediction=model.feature_residual(torch.from_numpy(f[None]).to(device),
                            torch.from_numpy(mask[None,None]).to(device),use_phase=True)[0,0].cpu().numpy()
                        deltas[beam]=supported_delta(cores[which][beam],prediction[cy:cy+n,cx:cx+w],
                            limits[plan['side'],plan['first']:plan['first']+n],
                            np.arange(plan['radial'],plan['radial']+w),c,fs)
                restored=mosaic_single_side(cores[which]*np.exp(deltas.astype(np.float64)),
                    plan['side'],phase=plan['phase'],height=height)
                record['conditions'][label]=dict(per_beam_log_delta_SHA256=array_sha(deltas),
                    native_mosaic_SHA256=array_sha(restored),input_features_SHA256=feature_hashes,
                    changes_vs_original=describe(original,restored),
                    additional_changes_vs_input=describe(original if which==0 else baseline,restored))
                outputs[label]=restored
            shown += [outputs['subband'],outputs['original']]
            yy=100+ordinal*1094
            draw.text((12,yy+14),name+'（预先固定原尺寸诊断，不代表全航段）',font=font,fill=255)
            for column,a in enumerate(shown):
                panel=Image.fromarray(amplitude_indices(a[::-1]),'P');panel.putpalette(palette.reshape(-1).tolist())
                canvas.paste(panel,(column*1040,yy+70))
            report['regions'].append(record);save()
            print(json.dumps(dict(region=name,conditions=2,elapsed_s=report['elapsed_s'])),flush=True)
            reference._mmap.madvise(mmap.MADV_DONTNEED)
        path=root/'同设备学习小波_原始_频带_两种模型输入_固定原尺寸_未验收.png'
        writer=NativePaletteWriter(path,canvas.width,canvas.height,palette,reserve_bytes=reserve)
        try:writer.append(np.asarray(canvas));encoding=writer.finish()
        finally:writer.close()
        report.update(status='completed_same_device_wavelet_fixed_native_transfer_diagnostic_not_full_voyage_acceptance',
            figure=str(path),figure_SHA256=sha(path),figure_encoding=encoding)
    except BaseException as exc:
        report.update(status='failed_preserved_native_local_transfer_probe_no_automatic_restart',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()


if __name__=='__main__':main()
