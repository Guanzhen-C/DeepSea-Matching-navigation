"""Strict frozen-model loading and evidence checks for the matched experiment."""
from pathlib import Path
from collections import Counter
import json

import torch

from MatchedSonarSignal import CONDITIONS,HALO
from MatchedSonarWavelet import MatchedSonarWavelet
from BandAnchoredNativeSonarFeatures import PROTOCOL
from AuditBandAnchoredNativeSonar import load_selected as load_prior
from TrainMatchedSonarFactorial import core_plan
from TrainNativeLocalSonarWavelet import sha


def read(path):return json.loads(Path(path).read_text())


def check_ledger(root,report,manifest):
    old=Path(report['sampling_plan_root']);previous=read(old/'run_summary.json')
    originals=[json.loads(s) for s in (old/'shared_training_inputs.jsonl').read_text().splitlines()]
    path=root/'shared_factorial_training_inputs.jsonl'
    if sha(path)!=report['training_ledger_file_SHA256']:raise ValueError('shared factorial training ledger changed')
    entries=[json.loads(s) for s in path.read_text().splitlines()]
    if len(entries)!=report['actual_training_examples_per_trained_condition'] or len(entries)!=800:
        raise ValueError('complete 800-example common training budget required')
    exposure=Counter();weak=identities=0
    for saved,actual in zip(originals,entries):
        for key in ('step','slot','source_index','beam','identity'):
            if saved[key]!=actual[key]:raise ValueError('common source/order/identity schedule changed')
        box,core=core_plan(saved,manifest['sources'][actual['source_index']])
        if actual['native_context']!=list(box) or actual['native_core_in_context']!=list(core):
            raise ValueError('actual original cores or compound native halo changed')
        if actual['source_index'] not in previous['train_source_indices'] or not 0<=actual['beam']<8:
            raise ValueError('reserved physical source leaked into training')
        entries_by_condition=actual['conditions']
        if set(entries_by_condition)!=set(CONDITIONS):raise ValueError('all five actual controls required')
        if entries_by_condition['mandatory_wide_raw']!=entries_by_condition['fusion_wide_raw']:
            raise ValueError('aggregation controls saw different actual inputs')
        if actual['identity']:
            identities+=1
            if (actual['noise_RMS_scale']!=0 or entries_by_condition['mandatory_wide_raw']['input_log_SHA256']!=actual['target_log_SHA256']
                    or entries_by_condition['fusion_limited_raw']!=entries_by_condition['fusion_wide_raw']
                    or entries_by_condition['fusion_limited_front']!=entries_by_condition['fusion_wide_front']):
                raise ValueError('identity examples were not the same unaugmented observations')
        else:
            scale=actual['noise_RMS_scale'];weak+=int(scale<=.20)
            for description in actual['actual_corrupted_field_RMS'].values():
                if abs(description['actual_added_complex_noise_RMS_over_source_RMS']-scale)>2e-6:
                    raise ValueError('the two known noise laws have different total RMS severity')
                if not description['missing_original_support_preserved']:raise ValueError('missing source support overwritten')
        exposure[f'{actual["source_index"]}:{actual["beam"]}']+=1
    if (dict(exposure)!=report['source_exposure_counts'] or len(exposure)!=48
            or identities!=report['identity_examples'] or weak!=report['weak_corrupted_examples']):
        raise ValueError('source coverage/identity/weak-noise exposures differ')
    return dict(common_training_examples_checked=len(entries),training_side_beam_sources_checked=48,
        weak_noise_examples_checked=weak,identity_examples_checked=identities,
        identity_record_bank_RMS_describes_unused_corrupted_variant_not_active_identity_input=True,
        original_training_feature_arrays_recomputed=False)


def check_full_cases(cases,previous,manifest):
    expected={(i,b,law) for i in previous['reserved_source_indices'] for b in range(8) for law in ('wide','limited')}
    actual={(c['source_index'],c['beam_0based'],c['test_noise_law']) for c in cases}
    if len(cases)!=32 or actual!=expected:raise ValueError('all sixteen complete beams under both noise laws required')
    for case in cases:
        source=manifest['sources'][case['source_index']]
        if (case['native_values']!=source['native_records']*source['native_range_bins']
                or not case['source_observation_is_not_clean_truth'] or not case['missing_support_delta_exact_zero']):
            raise ValueError('whole native field coverage or source-observation semantics changed')


def load_models(root,device):
    root=Path(root);report=read(root/'run_summary.json')
    if (report['status']!='completed_matched_processing_factorial_real_denoising_unaccepted'
            or report['smoke'] or report['steps_completed']!=400 or report['model_family']!=MatchedSonarWavelet.family
            or report['feature_protocol']!=PROTOCOL or report['conditions']!=CONDITIONS
            or report['training_core_halo']!=list(HALO)):
        raise ValueError('completed full matched-processing experiment required')
    for name,digest in report['source_code_SHA256'].items():
        if sha(root/'source_snapshot'/name)!=digest or sha(Path(__file__).with_name(name))!=digest:
            raise ValueError('exact frozen experimental implementation required: '+name)
    prior_root=Path(report['frozen_baseline_root'])
    if sha(prior_root/'run_summary.json')!=report['frozen_baseline_report_SHA256']:
        raise ValueError('frozen benchmark report changed')
    prior,prior_meta=load_prior(prior_root,device)
    models={'frozen_prior_anchored_raw':(prior,False)}
    for name,condition in CONDITIONS.items():
        chosen=report['selected_by_complete_common_validation'][name]
        best=min(report['validation_history'],key=lambda r:(r['metrics'][name]['selection_score'],r['step']))
        if chosen['step']!=best['step']:raise ValueError('common whole-native validation selection changed')
        for item in report['checkpoints'][name]:
            if sha(root/item['file'])!=item['SHA256']:raise ValueError('saved matched checkpoint changed')
        state=torch.load(root/chosen['file'],map_location=device,weights_only=True)
        if (state['model_family']!=report['model_family'] or state['condition']!=condition
                or state['condition_name']!=name or state['feature_protocol']!=PROTOCOL
                or state['source_code_SHA256']!=report['source_code_SHA256']):
            raise ValueError('condition preprocessing/aggregation/feature semantics changed')
        model=MatchedSonarWavelet(state['width'],aggregation=condition['aggregation']).to(device)
        model.load_state_dict(state['model'],strict=True);models[name]=(model.eval(),condition['frontend'])
    frozen=[r['metrics']['frozen_prior_anchored_raw'] for r in report['validation_history']]
    if any(a!=frozen[0] for a in frozen[1:]):raise ValueError('frozen prior outputs changed during new training')
    return models,report
