"""Read-only complete native domain qualification before more training.

Operator-processed 9/25 observations are unpaired with the noisy 9/24 voyage,
NOT current clean ground truth. Every originally selected record/range bin is
streamed, respecting the existing explicit omission of malformed Ping 530.
No private arrays leave the computer; no MAT/previous outputs are overwritten.
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
from SonarComplexSourceStatistics import ComplexSourceStatistics


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def guard(path):
    s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns)


def verified_source_indices(reader,identities,index_path):
    identities=np.asarray(identities)
    if identities.ndim!=2 or identities.shape[1]!=3:raise ValueError('original saved ping identities required')
    index_path=Path(index_path)
    if index_path.exists():
        indices=np.load(index_path)
        if not np.issubdtype(indices.dtype,np.integer):raise ValueError('integer original cell-index ledger required')
        indices=indices-1;origin='saved_original_cell_indices'
    else:
        # Early exports predate the explicit cell-index ledger. Resolve each
        # saved physical identity uniquely in the actual source, not by an
        # assumption that PingNum or output ordinal equals a cell index.
        original=[reader.identity(index) for index in range(len(reader))]
        desired=[tuple(int(v) for v in identity) for identity in identities]
        first=[];cursor=0
        for identity in desired:
            while cursor<len(original) and original[cursor]!=identity:cursor+=1
            if cursor==len(original):raise ValueError('saved ping identity sequence not found in original MAT')
            first.append(cursor);cursor+=1
        last=[];cursor=len(original)-1
        for identity in reversed(desired):
            while cursor>=0 and original[cursor]!=identity:cursor-=1
            if cursor<0:raise ValueError('saved ping identity sequence not found in original MAT')
            last.append(cursor);cursor-=1
        if first!=last[::-1]:raise ValueError('saved ping identity sequence has no unique ordered original MAT selection')
        indices=np.asarray(first,np.int64);origin='unique_ordered_original_MAT_identity_sequence'
    if indices.shape!=(len(identities),) or (indices<0).any() or (indices>=len(reader)).any() or (np.diff(indices)<=0).any():
        raise ValueError('original source selection ordering/bounds differ')
    if any(reader.identity(int(index))!=tuple(identity) for index,identity in zip(indices,identities)):
        raise ValueError('saved selection does not reproduce original ping identities')
    return indices,dict(selection_index_origin=origin,
        zero_based_selected_indices_sha256=hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest(),
        original_saved_cell_indices_file_sha256=sha(index_path) if index_path.exists() else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('processed-replicas','phase-cache','public-manifest','synthetic-inputs','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume-completed-from',type=Path)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(status='streaming_complete_native_training_domain_qualification',cases=[],
        private_data_uploaded=False,original_MAT_written=False,neural_model_trained=False,quality_accepted=False,
        first_batch_processed_status_is_user_provided=True,processed_references_are_unpaired_not_current_clean_truth=True,
        native_axes_not_assumed_to_share_physical_units=True,
        descriptor_differences_do_not_identify_noise_physics=True)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for name in (Path(__file__).name,'SonarComplexSourceStatistics.py','BathymetrySideScanDrawing.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    report['source_sha256']={q.name:sha(q) for q in snapshot.iterdir()}
    reused={}
    if args.resume_completed_from is not None:
        parent=json.loads((args.resume_completed_from/'run_summary.json').read_text())
        if parent['source_sha256']['SonarComplexSourceStatistics.py']!=sha(Path(__file__).with_name('SonarComplexSourceStatistics.py')):
            raise ValueError('completed statistics used a different numeric descriptor implementation')
        if parent['status'] not in ('failed_preserved_native_domain_qualification_no_automatic_restart','completed_full_native_domain_statistics_not_noise_identification'):
            raise ValueError('only completed per-source results from a terminal prior run may be reused')
        reused={c['source']:c for c in parent['cases'] if c['domain']=='operator_processed_native_complex_beamformed_reference'}
        report['resumed_terminal_parent_report_sha256']=sha(args.resume_completed_from/'run_summary.json')
    def save():
        temp=root/'run_summary.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.replace(root/'run_summary.json')
    save()
    try:
        folders=sorted(args.processed_replicas.glob('*/run_summary.json'))
        if len(folders)!=4:raise ValueError('the four existing processed-reference part ledgers required')
        for manifest in folders:
            item=json.loads(manifest.read_text());folder=manifest.parent
            identities=np.load(folder/'PingNum_PingS_PingUs.npy')
            if identities.shape!=(item['processed_pings'],3):raise ValueError('original selection ledger mismatch')
            for side in ('Port','Stbd'):
                path=Path(item['source'])/(side+'NormalModeBeamformingTotal.mat');before=guard(path)
                if str(path) in reused:
                    old=reused[str(path)]
                    index_path=folder/'SourceCellIndices1Based.npy'
                    if old['source_guard']!=before or old['source_ping_identity_sha256']!=sha(folder/'PingNum_PingS_PingUs.npy') or old['original_saved_cell_indices_file_sha256']!=(sha(index_path) if index_path.exists() else None):
                        raise ValueError('completed source guards or original selection changed')
                    if old['native_records']!=len(identities) or old['native_values']!=8*len(identities)*old['native_range_bins'] or not old['all_selected_native_values_included']:
                        raise ValueError('reused source was not completely processed')
                    report['cases'].append(dict(old,reused_completed_source_from=str(args.resume_completed_from)))
                    save();print(json.dumps(dict(reused_complete_source=str(path)),ensure_ascii=False),flush=True);continue
                statistics=ComplexSourceStatistics(8)
                with PingFile(path) as reader:
                    indices,index_metadata=verified_source_indices(reader,identities,folder/'SourceCellIndices1Based.npy')
                    for ordinal,index in enumerate(indices):
                        if reader.identity(int(index))!=tuple(identities[ordinal]):raise ValueError('original selected ping identity changed')
                        statistics.update_complex(reader.beam_sum(int(index),side))
                        if (ordinal+1)%250==0:
                            report['current_source_progress']=dict(source=folder.name,side=side,records=ordinal+1,total=len(indices))
                            save();print(json.dumps(report['current_source_progress'],ensure_ascii=False),flush=True)
                if guard(path)!=before:raise ValueError('operator-processed original source changed')
                result=statistics.result()
                if result['native_records']!=len(indices) or result['native_values']!=8*len(indices)*result['native_range_bins']:
                    raise ValueError('some original processed observations not included')
                report['cases'].append(dict(domain='operator_processed_native_complex_beamformed_reference',
                    source=str(path),source_guard=before,**index_metadata,
                    source_ping_identity_sha256=sha(folder/'PingNum_PingS_PingUs.npy'),
                    parameters=item['parameters'],source_is_clean_truth=False,all_selected_native_values_included=True,
                    **result));save()
        cache=json.loads((args.phase_cache/'run_summary.json').read_text())
        if cache['status']!='completed_full_native_phase_cache_not_denoising_or_clean_phase':raise ValueError('complete native noisy phase cache required')
        source=Path(cache['source_envelope']);phase_source=Path(cache['output'])
        if sha(source)!=cache['source_envelope_sha256_after'] or sha(phase_source)!=cache['output_sha256']:
            raise ValueError('original noisy envelope/phase cache changed')
        a=np.load(source,mmap_mode='r');angles=np.load(phase_source,mmap_mode='r')
        for side in range(2):
            statistics=ComplexSourceStatistics(8)
            for first in range(0,a.shape[2],32):
                amplitudes=np.asarray(a[side,:,first:first+32]);phi=np.asarray(angles[side,:,first:first+32,1:])
                g=np.cos(phi)+1j*np.sin(phi)
                statistics.update_amplitude_gradient(amplitudes,g)
                a._mmap.madvise(mmap.MADV_DONTNEED);angles._mmap.madvise(mmap.MADV_DONTNEED)
            report['cases'].append(dict(domain='noisy_measured_native_complex_beamformed_observations',side=side,
                source_amplitude_sha256=cache['source_envelope_sha256_after'],source_gradient_sha256=cache['output_sha256'],
                absolute_phase_reconstructed=False,derived_gradient_angles_float32_precision=True,
                all_selected_native_values_included=True,**statistics.result()));save()
        public=json.loads(args.public_manifest.read_text())
        if public['completed_samples']!=129:raise ValueError('complete original public complex collection required')
        for row in public['records']:
            path=Path(row['path'])
            if sha(path)!=row['cache_sha256']:raise ValueError('original public complex chip changed')
            z=np.load(path);statistics=ComplexSourceStatistics(1);statistics.update_complex(z[None])
            report['cases'].append(dict(domain='public_SAS_complex_image_original_observation',sample_id=row['sample_id'],
                original_sha256=row['cache_sha256'],axes_are_image_pixels_not_physical_ping_and_range=True,
                all_selected_native_values_included=True,**statistics.result()))
        synthetic=json.loads((args.synthetic_inputs/'run_summary.json').read_text())
        if synthetic['status']!='completed_nine_full_native_joint_corruption_controls_not_denoising_or_training':
            raise ValueError('complete predeclared native complex synthetic inputs required')
        for case in synthetic['cases']:
            path=args.synthetic_inputs/f'{case["sample_id"]}_scale_{case["noise_scale"]:.2f}'/'synthetic_noisy_native_complex64.npy'
            if sha(path)!=case['noisy_input_sha256']:raise ValueError('saved known synthetic input changed')
            z=np.load(path);statistics=ComplexSourceStatistics(1);statistics.update_complex(z[None])
            report['cases'].append(dict(domain='public_SAS_known_added_complex_stripes',sample_id=case['sample_id'],
                hypothetical_noise_scale=case['noise_scale'],exact_v8_training_field_replay_claimed=False,
                all_selected_native_values_included=True,**statistics.result()))
        groups={}
        for case in report['cases']:groups.setdefault(case['domain'],[]).extend(case['beams'])
        report['domain_descriptor_quantiles']={name:{metric:np.quantile([b[metric] for b in group],[.05,.5,.95]).tolist()
            for metric in ('adjacent_unit_phase_coherence','adjacent_energy_weighted_covariance_magnitude',
                'phase_curvature_coherence','amplitude_normalized_fourth_moment','std_positive_log_amplitude')}
            for name,group in groups.items()}
        report.update(status='completed_full_native_domain_statistics_not_noise_identification',
            native_complex_observations_included=sum(case['native_values'] for case in report['cases']))
    except Exception as exc:
        report.update(status='failed_preserved_native_domain_qualification_no_automatic_restart',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['elapsed_s']=time.monotonic()-started;save()


if __name__=='__main__':main()
