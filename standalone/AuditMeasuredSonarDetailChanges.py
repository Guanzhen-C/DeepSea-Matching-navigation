"""Describe observed bright-return changes, never claim a clean target error."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np

from ReviewMeasuredSonarNativeDetails import verified_numeric_candidate


def describe(original,changed):
    original=np.asarray(original,np.float64);changed=np.asarray(changed,np.float64)
    if original.shape!=changed.shape or not np.isfinite(original).all() or not np.isfinite(changed).all():
        raise ValueError('finite aligned native observation grids required')
    support=original>1e-12
    if not support.any():return dict(observed_positive_values=0,clean_truth=False)
    cutoff=float(np.quantile(original[support],.999))
    bright=support & (original>=cutoff)
    ratios=changed[support]/original[support]
    return dict(observed_positive_values=int(support.sum()),
        source_only_bright_return_quantile=.999,source_only_bright_amplitude_cutoff=cutoff,
        source_selected_bright_values=int(bright.sum()),
        bright_return_amplitude_ratio_q01_q50_q99=np.quantile(changed[bright]/original[bright],[.01,.5,.99]).tolist(),
        all_positive_amplitude_ratio_q01_q50_q99=np.quantile(ratios,[.01,.5,.99]).tolist(),
        mean_amplitude_ratio=float(changed[support].mean()/original[support].mean()),
        clean_truth=False,interpretation='Changes to observed returns, not recovered-clean accuracy; bright returns may include interference')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--original',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve an existing detail-change diagnostic')
    verified_numeric_candidate(args.baseline);verified_numeric_candidate(args.candidate)
    arrays=[np.load(path,mmap_mode='r') for path in (args.original,args.baseline,args.candidate)]
    protocol=json.loads(args.protocol.read_text())
    report=dict(status='completed_predeclared_native_observation_change_diagnostic_not_clean_accuracy',
        regions=[],quality_accepted=False,geological_feature_labels_verified=False,
        thresholds_used_only_for_describing_original_bright_returns_not_for_filtering=True)
    for name,row,col,h,w in protocol['regions']:
        a,b,c=[np.asarray(x[row:row+h,col:col+w]) for x in arrays]
        report['regions'].append(dict(name=name,coordinates=[row,col,h,w],
            baseline_changes=describe(a,b),candidate_changes=describe(a,c)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(__file__,args.output.parent/Path(__file__).name)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
