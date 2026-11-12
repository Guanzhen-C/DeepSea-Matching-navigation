"""Every original measured sample, six frozen models, independent GPU shards.

Uses the completed recovery certificate rather than rewriting an interrupted
training report. No crop-only acceptance, MAT write, resampling, new training,
gain floor or global brightness adjustment. All model gain fields are saved.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from RecoveredMatchedSonarModels import load_recovered,verified_bundle
from MatchedSonarWavelet import infer_group
from FullMeasuredSonarProtocol import supported_total_delta
from TrainNativeLocalSonarWavelet import sha,array_sha


def read(path):return json.loads(path.read_text())


def save(path,value):
    pending=path.with_suffix('.tmp');pending.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');pending.replace(path)


def verify_metadata(root,recovery):
    meta=root/'metadata';manifest=read(meta/'measured_protocol_manifest.json')
    if (manifest['authorized_host']!='[private-network]'
            or manifest['recovery_bundle_SHA256']!=sha(recovery/'bundle_manifest.json')):
        raise ValueError('authorized measured protocol and exact selected-weight bundle required')
    for name,digest in manifest['metadata_files_SHA256'].items():
        if sha(meta/name)!=digest:raise ValueError('measured metadata changed: '+name)
    return manifest


def verify_native(root,manifest):
    receipt=read(root/'native_source/transfer_receipt.json');header=receipt['header'];source=root/'native_source'
    if (receipt['status']!='completed_authorized_full_measured_complex128_transfer'
            or header['source_run_summary_SHA256']!=manifest['metadata_files_SHA256']['source_run_summary.json']
            or header['native_shape']!=manifest['native_shape']):raise ValueError('complete original transfer required')
    z=np.load(source/'measured_complex128.npy',mmap_mode='r');a=np.load(source/'original_amplitude_float64.npy',mmap_mode='r')
    if (list(z.shape)!=manifest['native_shape'] or a.shape!=z.shape
            or z.dtype!=np.complex128 or a.dtype!=np.float64):raise ValueError('all native shapes/precisions required')
    digest=hashlib.sha256();records=0
    for block in receipt['blocks']:
        start=block['start'];n=block['count']
        if start!=records:raise ValueError('all original records exactly once required')
        for values,key in ((z,'payload_SHA256'),(a,'amplitude_SHA256')):
            data=np.ascontiguousarray(values[:,:,start:start+n].transpose(0,2,1,3));view=memoryview(data).cast('B')
            if hashlib.sha256(view).hexdigest()!=block[key]:raise ValueError('original transported samples changed')
            digest.update(view)
        records+=n
    if records!=z.shape[2] or digest.hexdigest()!=receipt['trailer']['wire_payload_SHA256']:
        raise ValueError('whole original measured stream differs')
    return receipt


def worker(args):
    root=args.voyage.absolute();recovery=args.recovery.absolute();out=root/'full_frozen_inference'
    manifest=verify_metadata(root,recovery);device=torch.device(f'cuda:{args.worker_index}')
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.65,device)
    models,bundle=load_recovered(recovery,device);source=root/'native_source'
    z=np.load(source/'measured_complex128.npy',mmap_mode='r');a=np.load(source/'original_amplitude_float64.npy',mmap_mode='r')
    fields={name:np.load(out/f'{name}_supported_log_delta.npy',mmap_mode='r+') for name in models}
    limits=np.load(root/'metadata/bottom_first_ranges.npy');ledger=read(root/'metadata/envelope_cache_summary.json')
    c=ledger['parts'][0]['parameters']['c'];fs=ledger['parts'][0]['parameters']['BeamSigFs']
    shape=manifest['native_shape'];columns=np.arange(shape[-1]);start=time.monotonic()
    path=out/f'worker_{args.worker_index}.json'
    if path.exists():raise FileExistsError('preserve existing shard evidence')
    report=dict(status='running_complete_native_measured_shard',PID=os.getpid(),device=str(device),
                worker_index=args.worker_index,workers=args.workers,beams=[],source_code_SHA256=sha(Path(__file__)),
                runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)))
    save(path,report)
    try:
        for index in range(16):
            if index%args.workers!=args.worker_index:continue
            side,beam=divmod(index,8)
            values=np.asarray(z[side,beam],np.complex64)
            predicted,info=infer_group(models,values,device)
            original=np.asarray(a[side,beam]);case=dict(side=side,beam_0based=beam,
                native_values=original.size,source_complex64_SHA256=array_sha(values),models={})
            for name,total in predicted.items():
                delta=supported_total_delta(original,total,limits[side],columns,c,fs)
                fields[name][side,beam]=delta;fields[name].flush()
                case['models'][name]=dict(raw_total_delta_SHA256=array_sha(total),
                    applied_total_delta_SHA256=array_sha(delta),neural=info['neural'][name],
                    native_values=delta.size,all_applied_delta_finite=bool(np.isfinite(delta).all()))
            report['beams'].append(case);report['elapsed_s']=time.monotonic()-start;save(path,report)
            print(json.dumps(dict(worker=args.worker_index,side=side,beam=beam,native_values=original.size,
                                  completed=len(report['beams']),elapsed_s=report['elapsed_s'])),flush=True)
        report['status']='completed_all_assigned_native_measured_beams'
    except BaseException as exc:
        report.update(status='failed_preserved_measured_shard_no_automatic_restart',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['elapsed_s']=time.monotonic()-start;report['peak_CUDA_allocated_bytes']=torch.cuda.max_memory_allocated(device);save(path,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--voyage',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--worker-index',type=int,default=-1)
    args=p.parse_args()
    if not 1<=args.workers<=4:raise ValueError('one to four independent physical GPU shards required')
    if args.worker_index>=0:return worker(args)
    root=args.voyage.absolute();recovery=args.recovery.absolute();manifest=verify_metadata(root,recovery)
    bundle,_=verified_bundle(recovery);receipt=verify_native(root,manifest)
    if torch.cuda.device_count()<args.workers:raise ValueError('enough independent visible GPUs required')
    out=root/'full_frozen_inference';out.mkdir(exist_ok=False);start=time.monotonic()
    if shutil.disk_usage(out).free<48*1024**3:raise ValueError('48GiB free data disk preflight required')
    report=dict(status='starting_full_measured_native_frozen_model_shards',PID=os.getpid(),workers=args.workers,
                expected_native_shape=manifest['native_shape'],expected_native_source_values=int(np.prod(manifest['native_shape'])),
                original_source_receipt_SHA256=sha(root/'native_source/transfer_receipt.json'),
                protocol_manifest_SHA256=sha(root/'metadata/measured_protocol_manifest.json'),
                recovery_bundle_SHA256=sha(recovery/'bundle_manifest.json'),models=bundle['selected_weights'],
                full_native_grid_resampled=False,original_MAT_modified=False,models_retrained=False,
                final_observed_water_support_is_separate_from_network_training=True,
                original_phase_not_replaced_by_invented_phase=True,real_clean_truth=False,quality_accepted=False,
                gain_fields={},worker_PIDs=[])
    path=out/'run_summary.json';save(path,report);snapshot=out/'source_snapshot';snapshot.mkdir()
    names=('InferFullMeasuredSonar172.py','RecoveredMatchedSonarModels.py','FullMeasuredSonarProtocol.py',
           'RenderFullMeasuredSonar172.py','AuditFullMeasuredSonar172.py','NativeSonarPaletteStream.py')
    for name in names:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    report['new_execution_source_SHA256']={name:sha(snapshot/name) for name in names}
    for name in bundle['selected_weights']:
        target=out/f'{name}_supported_log_delta.npy'
        field=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=tuple(manifest['native_shape']))
        field.flush();del field;report['gain_fields'][name]=str(target)
    processes=[]
    try:
        for index in range(args.workers):
            command=[sys.executable,'-u',str(Path(__file__)), '--voyage',str(root),'--recovery',str(recovery),
                     '--workers',str(args.workers),'--worker-index',str(index)]
            process=subprocess.Popen(command);processes.append(process);report['worker_PIDs'].append(process.pid)
        report['status']='inferring_every_measured_native_sample_six_frozen_models';save(path,report)
        codes=[process.wait() for process in processes];report['worker_exit_codes']=codes
        if any(codes):raise RuntimeError('a measured shard failed; no automatic restart or partial acceptance')
        seen=set()
        for index in range(args.workers):
            shard=read(out/f'worker_{index}.json')
            if shard['status']!='completed_all_assigned_native_measured_beams':raise ValueError('complete shard evidence required')
            for beam in shard['beams']:
                key=(beam['side'],beam['beam_0based'])
                if key in seen or set(beam['models'])!=set(bundle['selected_weights']):raise ValueError('duplicate/incomplete original beam')
                seen.add(key)
        if seen!={(s,b) for s in range(2) for b in range(8)}:raise ValueError('all original sixteen beams required')
        report['status']='completed_all_native_measured_log_gain_fields_pending_render_and_audit';save(path,report)
        command=[sys.executable,'-u',str(Path(__file__).with_name('RenderFullMeasuredSonar172.py')),'--voyage',str(root)]
        subprocess.run(command,check=True)
        report['status']='completed_full_measured_frozen_inference_and_native_images_pending_independent_audit'
    except BaseException as exc:
        report.update(status='failed_preserved_full_measured_inference_no_automatic_restart',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['elapsed_s']=time.monotonic()-start;save(path,report)


if __name__=='__main__':main()
