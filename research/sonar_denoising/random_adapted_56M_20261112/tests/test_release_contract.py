from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
import torch
from NoFIRSonarWavelet import NoFIRSonarMWCNN


def test_formal_parameter_count_without_allocating_a_formal_model():
    with torch.device('meta'):
        model=NoFIRSonarMWCNN(96)
    assert sum(p.numel() for p in model.parameters())==56083513
    assert model.domain_affine.weight.numel()==24
    assert not hasattr(model,'consensus') and not hasattr(model,'local_gate')


def test_recipe_matches_completed_actual_stages_and_keeps_filter_distinction():
    root=Path(__file__).resolve().parents[1]
    recipe=json.loads((root/'configs/training_recipe.json').read_text())
    for phase,file in [('pretrain','pretrain_completed_summary.json'),('adapt','adapt_completed_summary.json')]:
        report=json.loads((root/'records'/file).read_text())
        assert report['steps_completed']==recipe[phase]['steps']
        assert report['actual_global_examples']==recipe[phase]['global_examples']
        assert report['selected']['SHA256']==recipe[phase]['selected_checkpoint_SHA256']
    assert recipe['received_signal_FIR_pre_or_post'] is False
    assert recipe['FIR_only_as_supervision_target'] is False
    assert recipe['synthetic_noise_colour_kernel_taps']==33
    assert recipe['feature_support_normalized_box_mean_shape']==[5,65]
    assert recipe['reference_is_clean_truth'] is False
