from pathlib import Path
import sys

from docx import Document
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from BuildSeabedSimulationReportDocument import build
from AuditSeabedSimulationReport import check_local_links


def test_word_preserves_all_heading_levels_tables_and_tall_figure_aspect(tmp_path):
    Image.new('RGB',(100,800),'orange').save(tmp_path/'figure.png')
    source=tmp_path/'report.md'
    source.write_text('# 题目\n\n编制日期：今天  \n项目：水下导航\n\n## 一、范围\n\n### 1. 指标\n\n#### （1）位置\n\n'
        '正文 **重点** 与 `code`，[来源](https://example.org)。\n\n'
        '| 项目 | 数值 |\n|---|---:|\n| RMSE | 1.234 |\n\n![图](figure.png)\n\n图1 原生比例\n')
    original=source.read_bytes(); target=tmp_path/'report.docx'
    result=build(source,target); doc=Document(target)
    assert source.read_bytes()==original
    assert [x['level'] for x in result['headings']]==[1,2,3,4]
    assert len(doc.tables)==1 and doc.tables[0].cell(1,1).text=='1.234'
    image=result['images'][0]
    assert image['placed_height_cm']<=19. and image['placed_width_cm']<=16.
    assert image['placed_width_cm']/image['placed_height_cm']==pytest.approx(1/8)
    assert result['editable_word_reopened']
    assert any(x.text=='编制日期：今天\n项目：水下导航' for x in doc.paragraphs)
    with pytest.raises(FileExistsError):build(source,target)


def test_word_does_not_embed_images_outside_report_directory(tmp_path):
    folder=tmp_path/'report'; folder.mkdir()
    Image.new('RGB',(2,2)).save(tmp_path/'outside.png')
    source=folder/'report.md';source.write_text('# 题目\n\n![图](../outside.png)\n')
    with pytest.raises(ValueError,match='image location'):build(source,folder/'out.docx')


def test_link_audit_allows_only_its_own_pending_output_and_rejects_other_missing_files(tmp_path):
    document=tmp_path/'README.md';pending=tmp_path/'audit.json'
    document.write_text('[审查](audit.json) [外部](https://example.org)\n')
    assert check_local_links([document],pending)==1
    document.write_text('[审查](audit.json) [错误](missing.json)\n')
    with pytest.raises(ValueError,match='missing.json'):check_local_links([document],pending)
