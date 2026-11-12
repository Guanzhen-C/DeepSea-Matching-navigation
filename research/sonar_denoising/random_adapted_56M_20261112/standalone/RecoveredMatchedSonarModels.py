"""Load frozen selected models through a separately verified recovery record.

Never relabel the locally SIGKILLed producer report as completed. The complete
172 validation and its independent evidence audit are the additional evidence.
"""
from pathlib import Path
import json

import torch

from MatchedSonarSignal import CONDITIONS
from MatchedSonarWavelet import MatchedSonarWavelet
from BandAnchoredNativeSonarWavelet import BandAnchoredNativeSonarWavelet
from BandAnchoredNativeSonarFeatures import PROTOCOL
from TrainNativeLocalSonarWavelet import sha
from AuditMatched172Validation import check_cases


def verified_bundle(root):
    root = Path(root);bundle = json.loads((root/'bundle_manifest.json').read_text())
    validation = json.loads((root/'remote_full_native_validation/run_summary.json').read_text())
    audit = json.loads((root/'remote_full_native_validation/independent_evidence_audit.json').read_text())
    if (bundle['user_authorized_host'] != '[private-network]'
            or audit['status'] != 'passed_full_172_transport_coverage_common_input_and_SSE_evidence_audit'
            or audit['source_report_SHA256'] != sha(root/'remote_full_native_validation/run_summary.json')
            or audit['source_bundle_manifest_SHA256'] != sha(root/'bundle_manifest.json')
            or audit['auditor_SHA256'] != sha(Path(__file__).with_name('AuditMatched172Validation.py'))
            or validation['source_bundle_manifest_SHA256'] != sha(root/'bundle_manifest.json')):
        raise ValueError('completed immutable recovery and full independent evidence audit required')
    check_cases(validation,bundle)
    for name,digest in bundle['files'].items():
        if sha(root/name) != digest:raise ValueError('preserved bundle file changed: '+name)
        if name.startswith('standalone/') and sha(Path(__file__).with_name(Path(name).name)) != digest:
            raise ValueError('actual imported frozen runtime differs: '+name)
    original = json.loads((root/'reports/interrupted_local_training.json').read_text())
    if sha(root/'reports/interrupted_local_training.json') != bundle['source_training_report_SHA256']:
        raise ValueError('original interrupted producer report changed')
    return bundle,original


def load_recovered(root,device):
    root = Path(root);bundle,original = verified_bundle(root);models = {}
    for name,item in bundle['selected_weights'].items():
        path = root/item['path']
        if sha(path) != item['SHA256']:raise ValueError('selected weights changed')
        state = torch.load(path,map_location=device,weights_only=True)
        if state['feature_protocol'] != PROTOCOL or state['step'] != item['step']:
            raise ValueError('selected step/feature interpretation differs')
        if name == 'frozen_prior_anchored_raw':
            if state['model_family'] != BandAnchoredNativeSonarWavelet.family:raise ValueError('prior family changed')
            model = BandAnchoredNativeSonarWavelet(state['width']).to(device);front = False
        else:
            condition = CONDITIONS[name];chosen = original['selected_by_complete_common_validation'][name]
            if (state['model_family'] != MatchedSonarWavelet.family or state['condition'] != condition
                    or state['condition_name'] != name or chosen['SHA256'] != item['SHA256']
                    or state['source_code_SHA256'] != original['source_code_SHA256']
                    or state['training_manifest_SHA256'] != bundle['source_manifest_SHA256']):
                raise ValueError('original trained condition/selection/source semantics changed')
            model = MatchedSonarWavelet(state['width'],aggregation=condition['aggregation']).to(device)
            front = condition['frontend']
        model.load_state_dict(state['model'],strict=True);models[name] = (model.eval(),front)
    return models,bundle
