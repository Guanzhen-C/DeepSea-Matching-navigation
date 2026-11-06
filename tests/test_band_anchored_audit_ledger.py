from pathlib import Path
import hashlib
import json
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from AuditBandAnchoredNativeSonar import check_matched_ledger


def test_only_declared_phase_features_can_change_in_complete_matched_ledger(tmp_path):
    old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    original=[dict(step=1,slot=i,seed=9163000+i,features_SHA256='a'*64,
        noisy_log_SHA256='b'*64,target_log_SHA256='c'*64,source_index=0,beam=i) for i in range(2)]
    encoded=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in original)
    (old/'shared_training_inputs.jsonl').write_text(encoded)
    previous=dict(training_input_ledger_file_SHA256=hashlib.sha256(encoded.encode()).hexdigest())
    rows=[dict(row,comparison_relative_features_SHA256='a'*64,features_SHA256='d'*64,
        amplitude_feature_SHA256='e'*64) for row in original]
    def save():
        path=new/'matched_training_inputs.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(actual_training_examples=2,matched_training_input_file_SHA256=save(),
        comparison_input_ledger_reconstructed_SHA256=previous['training_input_ledger_file_SHA256'])
    assert check_matched_ledger(report,previous,new,old)==2
    rows[0]['seed']+=1;report['matched_training_input_file_SHA256']=save()
    with pytest.raises(ValueError,match='differ'):check_matched_ledger(report,previous,new,old)
