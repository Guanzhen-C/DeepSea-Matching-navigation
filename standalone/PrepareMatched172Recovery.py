"""Immutable selected-weight/code bundle for the explicitly authorized 172.

The original local job completed training/selection but its native post-test
was killed by systemd-oomd. Never rewrite that report as a successful process.
This bundle starts validation ONLY; no optimizer or new training is exported.
"""
from pathlib import Path
import argparse
import json
import shutil

from MatchedSonarArtifacts import read,check_ledger
from TrainNativeLocalSonarWavelet import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','frozen-baseline','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();report=read(args.training/'run_summary.json');manifest=read(args.manifest)
    if (report['steps_completed']!=400 or report['smoke']
            or report['status']!='postselection_full_16_beams_both_noise_laws_all_controls'
            or Path('/proc',str(report['PID'])).exists()):
        raise ValueError('terminal interrupted post-test with completed 400-step training required')
    checked=check_ledger(args.training,report,manifest)
    root=args.output.absolute();root.mkdir(parents=True,exist_ok=False)
    code=root/'standalone';code.mkdir();weights=root/'selected_weights';weights.mkdir();records=root/'reports';records.mkdir()
    file_hashes={}
    def copy(source,target,expected=None):
        if expected is not None and sha(source)!=expected:raise ValueError('immutable source/checkpoint changed: '+str(source))
        shutil.copy2(source,target);file_hashes[str(target.relative_to(root))]=sha(target)
    for name,digest in report['source_code_SHA256'].items():
        copy(args.training/'source_snapshot'/name,code/name,digest)
    for name in ('ReceiveNativeSonarArray.py','RunMatched172Validation.py'):
        copy(Path(__file__).with_name(name),code/name)
    copy(args.training/'run_summary.json',records/'interrupted_local_training.json')
    copy(args.frozen_baseline/'run_summary.json',records/'frozen_prior_training.json')
    copy(args.manifest,records/'native_processed_manifest.json')
    copy(args.training/'shared_factorial_training_inputs.jsonl',records/'shared_factorial_training_inputs.jsonl',report['training_ledger_file_SHA256'])
    copy(args.training/'aggregate_appearance_profile.json',root/'aggregate_appearance_profile.json',report['noise_profile_SHA256'])
    chosen={}
    for name,item in report['selected_by_complete_common_validation'].items():
        target=weights/(name+'.pt');copy(args.training/item['file'],target,item['SHA256'])
        chosen[name]=dict(path=str(target.relative_to(root)),SHA256=item['SHA256'],step=item['step'],condition=report['conditions'][name])
    prior=read(args.frozen_baseline/'run_summary.json')
    item=prior['selected_by_complete_native_validation'];target=weights/'frozen_prior_anchored_raw.pt'
    copy(args.frozen_baseline/item['file'],target,item['SHA256'])
    chosen['frozen_prior_anchored_raw']=dict(path=str(target.relative_to(root)),SHA256=item['SHA256'],step=item['step'],condition=dict(frontend=False))
    selected_sources=[dict(source_index=i,side=s['side'],physical_part=s['physical_part'],
        shape=[8,s['native_records'],s['native_range_bins']],source_guard=s['source_guard'])
        for i,s in enumerate(manifest['sources']) if s['split']=='validation_known_extra_noise']
    bundle=dict(status='prepared_authorized_172_validation_only_bundle',**checked,
        files=file_hashes,selected_weights=chosen,native_sources=selected_sources,
        original_local_process_PID=report['PID'],original_process_exit_code=137,
        termination_evidence='systemd-oomd journal at 2026-11-05 12:58:01 reports this scope killed for memory pressure',
        source_training_report_SHA256=sha(args.training/'run_summary.json'),
        source_manifest_SHA256=sha(args.manifest),user_authorized_host='172.18.80.26',
        private_native_data_transfer_authorized_by_current_user=True,
        code_and_weights_only_here_native_arrays_transferred_separately=True,
        training_restarted=False,original_MAT_modified=False)
    (root/'bundle_manifest.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=bundle['status'],files=len(file_hashes),selected_models=len(chosen),
        bundle_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
