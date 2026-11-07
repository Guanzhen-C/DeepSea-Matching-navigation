from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone'))
from BuildEnhancedNavigationReport import section, change, write_report


def test_report_sections_replace_current_results_without_changing_other_sections():
    old = '# 报告\n\n## 一、范围\n保留\n\n## 二、结果\n旧指标\n### 1. 旧表\n旧值\n\n## 三、历史\n历史证据\n'
    new = section(old, '二、结果', '### 1. 新表\n\n真实新值')
    assert '旧指标' not in new and '旧值' not in new
    assert '真实新值' in new and '历史证据' in new and '保留' in new
    with pytest.raises(ValueError):
        section(old, '不存在', '不应写入')
    with pytest.raises(ValueError):
        section(old+'\n## 二、结果\n重复', '二、结果', '不应猜测')


def test_unfavourable_comparison_is_not_reworded_as_improvement():
    assert change(2., 1.) == '增加100.00%'
    assert change(1., 2.) == '降低50.00%'
    assert change(1., 1.) == '不变'


def test_running_or_prefix_comparison_cannot_produce_final_manuscript(tmp_path):
    (tmp_path/'data').mkdir()
    path = tmp_path/'data/verified_new_map_metrics.json'
    for d in ({'status': 'running', 'samples': 62458},
              {'status': 'completed_full_new_map_vs_unchanged_old_map_comparison', 'samples': 190}):
        path.write_text(json.dumps(d))
        with pytest.raises(ValueError):
            write_report(tmp_path)
        assert not (tmp_path/'海底地形地貌匹配算法仿真报告.md').exists()
