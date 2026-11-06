"""Read-only full-voyage native phase-gradient extraction, no denoising.

Reads actual two-ULA complex sums from each original MAT. Every magnitude
is checked against the previously verified native envelope cache. Retains
all eight synchronous beams, every selected record and every range bin.
Stores derived float32 range-phase angles, not a fabricated clean phase.
This reusable local cache must never be uploaded with public training data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import mmap
import shutil
import time

import numpy as np

from BathymetrySideScanDrawing import PingFile


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def guard(path):
    stat=Path(path).stat();return dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def range_phase_angles(values):
    z=np.asarray(values)
    if z.ndim!=2 or not np.iscomplexobj(z) or not np.isfinite(z).all():
        raise ValueError('finite complex beam-by-range observations required')
    amplitude=np.abs(z);support=amplitude>0
    unit=z/np.where(support,amplitude,1.)
    adjacent=unit[:,1:]*np.conj(unit[:,:-1])
    valid=support[:,1:]&support[:,:-1]
    result=np.zeros(z.shape,np.float32)
    result[:,1:]=np.where(valid,np.angle(adjacent),0.).astype(np.float32)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('envelope-cache','replica-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();ledger=json.loads((args.envelope_cache/'run_summary.json').read_text())
    if not ledger['original_all_native_samples_reproduced_exactly']:raise ValueError('verified original envelope reconstruction required')
    guards=ledger['source_guards']
    if any(guard(path)!=expected for path,expected in guards.items()):raise ValueError('original MAT guards changed')
    source=args.envelope_cache/'原始MAT逐束包络_float64.npy';amplitudes=np.load(source,mmap_mode='r')
    if amplitudes.ndim!=4 or amplitudes.shape[:2]!=(2,8):raise ValueError('complete two-side/eight-beam native cache required')
    if sum(part['count'] for part in ledger['parts'])!=amplitudes.shape[2]:raise ValueError('physical record counts do not cover full cache')
    if shutil.disk_usage(args.envelope_cache).free<amplitudes.size*4+4*1024**3:raise ValueError('four GiB plus full derived cache disk headroom required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for name in (Path(__file__).name,'BathymetrySideScanDrawing.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    report=dict(status='extracting_full_native_measured_phase_gradients_not_denoising',
        source_envelope=str(source.resolve()),source_envelope_guard=guard(source),native_shape=list(amplitudes.shape),
        expected_values=int(amplitudes.size),processed_values=0,source_guards=guards,
        source_ledger_sha256=sha(args.envelope_cache/'run_summary.json'),
        source_sha256={path.name:sha(path) for path in snapshot.iterdir()},
        dtype='float32',native_samples_resized=False,source_MAT_written=False,
        private_raw_data_uploaded=False,clean_phase_recovered=False,quality_accepted=False,
        phase_angle_precision_is_derived_float32_not_lossless_complex128=True,
        first_native_range_column_has_no_phase_neighbour=True,records=[])
    def save():
        temp=root/'run_summary.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.replace(root/'run_summary.json')
    save();path=root/'原始MAT逐束相邻相位角_float32.npy'
    phases=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=amplitudes.shape)
    expected_digest=hashlib.sha256();totals=np.zeros((2,8),np.float64);counts=np.zeros((2,8),np.int64)
    try:
        report['source_envelope_sha256_before']=sha(source);save()
        for part in ledger['parts']:
            name=part['name'];offset=part['offset'];count=part['count']
            folder=args.replica_root/('下午返航_'+name)
            metadata=json.loads((folder/'run_summary.json').read_text())
            indices=np.load(folder/'SourceCellIndices1Based.npy')-1
            ids=np.load(folder/'PingNum_PingS_PingUs.npy')
            if len(indices)!=count or ids.shape!=(count,3):raise ValueError('selected record identity grid differs')
            for side,side_name in enumerate(('Port','Stbd')):
                original=Path(metadata['source'])/(side_name+'NormalModeBeamformingTotal.mat')
                if str(original) not in guards or guard(original)!=guards[str(original)]:raise ValueError('unverified source MAT')
                with PingFile(original) as reader:
                    for local,index in enumerate(indices):
                        if reader.identity(int(index))!=tuple(ids[local]):raise ValueError('original physical acoustic record identity differs')
                        z=reader.beam_sum(int(index),side_name);a=np.abs(z);position=offset+local
                        if not np.array_equal(a,amplitudes[side,:,position]):raise ValueError('native complex magnitude differs from verified original cache')
                        angles=range_phase_angles(z);phases[side,:,position]=angles
                        expected_digest.update(angles.tobytes())
                        if not np.array_equal(phases[side,:,position],angles):raise ValueError('derived phase cache write differs')
                        positive=a>0;logged=np.zeros(a.shape,np.float64);np.log(a,out=logged,where=positive)
                        totals[side]+=logged.sum(axis=1);counts[side]+=positive.sum(axis=1)
                        report['processed_values']+=a.size
                        if (local+1)%32==0:
                            phases.flush();phases._mmap.madvise(mmap.MADV_DONTNEED)
                            amplitudes._mmap.madvise(mmap.MADV_DONTNEED)
                        if (local+1)%200==0:
                            report['elapsed_s']=time.monotonic()-started;save()
                            print(json.dumps(dict(part=name,side=side_name,records=local+1,total=count,values=report['processed_values'])),flush=True)
                report['records'].append(dict(part=name,side=side_name,count=count,offset=offset,
                    first_identity=ids[0].tolist(),last_identity=ids[-1].tolist()))
                save()
        phases.flush()
        if report['processed_values']!=amplitudes.size:raise ValueError('full native phase extraction incomplete')
        actual_digest=hashlib.sha256()
        # Read ALL derived values back in original physical-record order.
        for part in ledger['parts']:
            for side in range(2):
                for position in range(part['offset'],part['offset']+part['count']):
                    actual_digest.update(np.ascontiguousarray(phases[side,:,position]).tobytes())
                    if position%32==0:phases._mmap.madvise(mmap.MADV_DONTNEED)
        if actual_digest.hexdigest()!=expected_digest.hexdigest():raise ValueError('complete derived cache readback differs from actual source-derived phase values')
        report['all_values_readback_verified_sha256']=actual_digest.hexdigest()
        report['source_envelope_sha256_after']=sha(source)
        if report['source_envelope_sha256_before']!=report['source_envelope_sha256_after']:raise ValueError('original envelope file changed')
        if any(guard(path)!=expected for path,expected in guards.items()):raise ValueError('original MAT guards changed during extraction')
        report.update(status='completed_full_native_phase_cache_not_denoising_or_clean_phase',
            output=str(path),output_sha256=sha(path),whole_beam_log_origins=(totals/np.maximum(counts,1)).tolist(),
            all_original_complex_magnitudes_match_verified_envelope_cache=True,
            all_original_phase_gradient_values_saved_and_readback_checked=True,original_sources_unchanged=True)
    except Exception as exc:
        report.update(status='failed_preserved_full_native_phase_extraction_no_automatic_restart',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['elapsed_s']=time.monotonic()-started;save()


if __name__=='__main__':main()
