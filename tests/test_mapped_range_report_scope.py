from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from BuildMappedRangeReportEvidence import require_report_scope


def valid(full=True):
    return dict(status=('completed_independent_full_identical_input_mapped_range_comparison' if full else
        'completed_independent_heldout_recording_mapped_range_comparison_NOT_full16km'),
        samples=62458 if full else 15152, source_seed=31 if full else 33,
        mapped_range_precision_policy='preserve_legacy',all_received_events_verified=True,
        DR_and_Local_bit_exact=True,full_original_input_file_SHA256_reverified=True,
        executed_dependency_audit=dict(core_files=4,prior_files=3),
        preserved_prefix_audit=dict(maximum_coordinate_difference_m=0.))


@pytest.mark.parametrize('full',[True,False])
def test_both_completed_scopes_are_required_without_substitution(full):
    require_report_scope(valid(full),full=full)
    with pytest.raises(ValueError): require_report_scope(valid(not full),full=full)


@pytest.mark.parametrize('change',[
    dict(status='running'),dict(samples=62457),dict(source_seed=32),
    dict(mapped_range_precision_policy='tempered_legacy'),dict(DR_and_Local_bit_exact=False),
    dict(all_received_events_verified=False),dict(full_original_input_file_SHA256_reverified=False),
    dict(executed_dependency_audit=dict(core_files=3,prior_files=3)),
    dict(preserved_prefix_audit=dict(maximum_coordinate_difference_m=.1)),
])
def test_partial_unverified_or_wrong_profile_cannot_be_reported_as_the_improved_full_run(change):
    d=valid(); d.update(change)
    with pytest.raises(ValueError): require_report_scope(d,full=True)


def test_new_report_title_is_preserved_in_editable_word_metadata_and_header(tmp_path):
    from docx import Document
    from BuildSeabedSimulationReportDocument import build
    from BuildMappedRangeNavigationReport import TITLE
    source=tmp_path/'report.md'; target=tmp_path/'report.docx'
    source.write_text('# '+TITLE+'\n\n## 一、结果\n\n完整结果。\n')
    build(source,target)
    document=Document(target)
    assert document.core_properties.title==TITLE
    assert document.sections[0].header.paragraphs[0].text==TITLE


def test_outline_requires_ordered_chinese_chapters_and_requested_subheading_style():
    from AuditMappedRangeNavigationReport import check_outline
    from BuildMappedRangeNavigationReport import TITLE
    numbers=['一','二','三','四','五','六','七','八','九','十','十一','十二','十三','十四']
    headings=[(1,TITLE)]+[(2,n+'、章节') for n in numbers]+[(3,'1. 方法'),(4,'（1）模型')]
    check_outline(headings)
    with pytest.raises(ValueError): check_outline(headings[:-1]+[(4,'1. 错误层级')])
