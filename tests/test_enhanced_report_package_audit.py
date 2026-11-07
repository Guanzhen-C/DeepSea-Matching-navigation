from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone'))
from AuditEnhancedNavigationReport import verify_outline, verify_no_raw_payloads
from BuildEnhancedNavigationReport import copy_referenced_images


def valid_outline():
    names = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二', '十三', '十四']
    return [(1, '海底地形地貌匹配算法仿真报告')] + [(2, s+'、章节') for s in names] + [(3, '1. 内容'), (4, '（1）子项')]


def test_user_heading_conventions_are_checked():
    verify_outline(valid_outline())


@pytest.mark.parametrize('index,text', [(0, '错误题目'), (1, '1. 第一章'), (15, '一、错误三级'), (16, '(1) 半角')])
def test_wrong_heading_styles_are_not_accepted(index, text):
    outline = valid_outline()
    outline[index] = (outline[index][0], text)
    with pytest.raises(ValueError):
        verify_outline(outline)


def test_unused_historical_figures_are_not_silently_packaged(tmp_path):
    source, target = tmp_path/'old', tmp_path/'new'
    (source/'figures').mkdir(parents=True)
    target.mkdir()
    (source/'figures/used.png').write_bytes(b'fixture image')
    (source/'figures/unused.png').write_bytes(b'unreferenced historical content')
    copied = copy_referenced_images('![使用](figures/used.png)', target, source)
    assert copied == ['figures/used.png']
    assert (target/'figures/used.png').read_bytes() == b'fixture image'
    assert not (target/'figures/unused.png').exists()
    with pytest.raises(ValueError):
        copy_referenced_images('![越界](../outside.png)', target, source)


def test_report_package_keeps_metrics_but_rejects_raw_payloads(tmp_path):
    (tmp_path/'metrics.json').write_text('{}')
    verify_no_raw_payloads(tmp_path)
    (tmp_path/'nested').mkdir()
    (tmp_path/'nested/private.mat').write_bytes(b'synthetic forbidden fixture')
    with pytest.raises(ValueError):
        verify_no_raw_payloads(tmp_path)
