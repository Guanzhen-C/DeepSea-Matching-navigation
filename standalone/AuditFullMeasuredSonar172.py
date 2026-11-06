"""Full original/field/PNG replay plus independent finite-context NN checks.

All native input bytes, gain values, numeric mosaic values and displayed pixels
are checked. The NN itself is independently recomputed in TWO predeclared cores
for every model/beam, not falsely claimed as a second full neural voyage run.
There is no paired clean truth and passing this audit is not quality acceptance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import torch

from InferFullMeasuredSonar172 import verify_metadata,verify_native
from RecoveredMatchedSonarModels import load_recovered
from RenderFullMeasuredSonar172 import numeric_blocks
from FullMeasuredSonarProtocol import supported_total_delta
from NativeLocalSonarWavelet import context_box
from MatchedSonarWavelet import context_prediction
from NativeSonarPaletteStream import amplitude_indices,decoded_index_rows
from TrainNativeLocalSonarWavelet import sha,array_sha
from caris_orangecopper import caris_orangecopper


CORES=((128,2048,64,256),(1800,9728,64,256))


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--voyage',type=Path,required=True)
    p.add_argument('--recovery',type=Path,required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    root=args.voyage.absolute();out=root/'full_frozen_inference';start_time=time.monotonic()
    path=out/'independent_native_audit.json'
    if path.exists():raise FileExistsError('preserve completed full native audit')
    producer=read(out/'run_summary.json');render=read(out/'render_summary.json')
    if (producer['status']!='completed_full_measured_frozen_inference_and_native_images_pending_independent_audit'
            or render['status']!='completed_all_seven_native_images_and_fixed_original_pixel_diagnostics_not_quality_acceptance'):
        raise ValueError('completed whole original native inference/render required')
    for name,digest in producer['new_execution_source_SHA256'].items():
        if sha(out/'source_snapshot'/name)!=digest or sha(Path(__file__).with_name(name))!=digest:
            raise ValueError('frozen executed code changed: '+name)
    manifest=verify_metadata(root,args.recovery.absolute());receipt=verify_native(root,manifest)
    a=np.load(root/'native_source/original_amplitude_float64.npy',mmap_mode='r')
    z=np.load(root/'native_source/measured_complex128.npy',mmap_mode='r')
    fields={name:np.load(out/f'{name}_supported_log_delta.npy',mmap_mode='r') for name in producer['models']}
    seen=set();count=0
    for index in range(producer['workers']):
        shard=read(out/f'worker_{index}.json')
        if shard['status']!='completed_all_assigned_native_measured_beams':raise ValueError('all actual shards completed required')
        for beam in shard['beams']:
            key=(beam['side'],beam['beam_0based'])
            if key in seen:raise ValueError('duplicated original beam evidence')
            seen.add(key)
            if array_sha(np.asarray(z[key],np.complex64))!=beam['source_complex64_SHA256']:
                raise ValueError('actual phase-feature source differs from original waveform')
            for name,field in fields.items():
                values=np.asarray(field[key])
                if (values.dtype!=np.float32 or values.shape!=z.shape[-2:] or not np.isfinite(values).all()
                        or array_sha(values)!=beam['models'][name]['applied_total_delta_SHA256']):
                    raise ValueError('saved complete per-model native gain differs from inferred values')
                count+=values.size
    if seen!={(s,b) for s in range(2) for b in range(8)} or count!=int(np.prod(z.shape))*6:
        raise ValueError('all six models and all sixteen full original beams required')
    palette=(caris_orangecopper(256)*255).astype(np.uint8)
    decoded={name:iter(decoded_index_rows(Path(item['file']),palette)) for name,item in render['images'].items()}
    digests={name:hashlib.sha256() for name in decoded};rows_seen=0
    for start,rows in numeric_blocks(root,producer['models']):
        if start!=rows_seen:raise ValueError('complete native row replay order required')
        for name,values in rows.items():
            digests[name].update(memoryview(np.ascontiguousarray(values)).cast('B'))
            for row in amplitude_indices(values):
                actual=next(decoded[name],None)
                if actual is None or not np.array_equal(actual,row):raise ValueError('saved native PNG pixel differs from numeric replay')
        rows_seen+=len(rows['original'])
        if rows_seen%1280==0:print(json.dumps(dict(audit_native_rows=rows_seen,total=manifest['native_mosaic_shape'][0])),flush=True)
    for name,iterator in decoded.items():
        item=render['images'][name]
        if (next(iterator,None) is not None or digests[name].hexdigest()!=item['float64_mosaic_stream_SHA256']
                or sha(Path(item['file']))!=item['file_SHA256']):raise ValueError('whole numeric/PNG file replay differs')
    if (rows_seen!=manifest['native_mosaic_shape'][0]
            or digests['original'].hexdigest()!=receipt['trailer']['original_forward_float64_mosaic_SHA256']):
        raise ValueError('complete original numeric mosaic does not reproduce local acquisition')
    device=torch.device(args.device);torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.6,device)
    models,bundle=load_recovered(args.recovery.absolute(),device)
    limits=np.load(root/'metadata/bottom_first_ranges.npy');ledger=read(root/'metadata/envelope_cache_summary.json')
    c=ledger['parts'][0]['parameters']['c'];fs=ledger['parts'][0]['parameters']['BeamSigFs'];checks=[]
    for side,beam in sorted(seen):
        for core in CORES:
            y,x,h,w=core;box,inner=context_box(z.shape[-2:],core,halo=(80,416));top,left,bottom,right=box
            original=np.asarray(z[side,beam,top:bottom,max(0,left-1):right],np.complex64)
            for name,(model,front) in models.items():
                with torch.inference_mode():
                    total,_=context_prediction(model,original,device,first_model_column=left,
                        full_range_bins=z.shape[-1],frontend=front,core=inner)
                expected=supported_total_delta(a[side,beam,y:y+h,x:x+w],total,
                    limits[side,y:y+h],np.arange(x,x+w),c,fs)
                actual=np.asarray(fields[name][side,beam,y:y+h,x:x+w])
                np.testing.assert_allclose(actual,expected,atol=5e-6,rtol=2e-5)
                checks.append(dict(side=side,beam=beam,model=name,native_core=list(core),
                                   maximum_absolute_delta_difference=float(np.max(abs(actual-expected)))))
        print(json.dumps(dict(neural_audit_side=side,beam=beam,all_models=True)),flush=True)
    result=dict(status='passed_full_native_measured_transport_gain_mosaic_PNG_and_predeclared_neural_core_audit',
                producer_report_SHA256=sha(out/'run_summary.json'),render_summary_SHA256=sha(out/'render_summary.json'),
                source_protocol_manifest_SHA256=sha(root/'metadata/measured_protocol_manifest.json'),
                auditor_SHA256=sha(Path(__file__)),original_complex128_values=int(np.prod(z.shape)),
                original_complex_and_amplitude_payload_bytes=receipt['payload_bytes'],
                complete_gain_values_checked=count,native_mosaic_values_per_image=rows_seen*manifest['native_mosaic_shape'][1],
                full_native_images_checked=len(decoded),full_native_PNG_pixels_checked=rows_seen*manifest['native_mosaic_shape'][1]*len(decoded),
                independent_neural_scope='two predeclared 64x256 native cores for every side/beam/model',
                independently_recomputed_neural_native_values=sum(64*256 for _ in checks),neural_checks=checks,
                full_neural_voyage_independently_recomputed=False,all_numeric_and_display_rows_independently_replayed=True,
                original_MAT_modified=False,source_resampled=False,real_clean_truth=False,quality_accepted=False,
                elapsed_s=time.monotonic()-start_time)
    with path.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='neural_checks'}),flush=True)


if __name__=='__main__':main()
