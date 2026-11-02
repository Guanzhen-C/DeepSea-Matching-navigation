#!/usr/bin/env python3
"""Check literature-only deliveries, optionally including a native Office render."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import build_literature_only_reports as report


def cjk_counter(text: str) -> Counter:
    return Counter(re.findall('[\u4e00-\u9fff]', text))


def check_recovery(expected: str, actual: str) -> float:
    wanted, found = cjk_counter(expected), cjk_counter(actual)
    recovery = sum((wanted & found).values()) / sum(wanted.values())
    assert recovery >= .99, recovery
    return round(recovery, 6)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def page_count(path: Path) -> int:
    info = subprocess.check_output(['pdfinfo', str(path)], text=True)
    return int(re.search(r'Pages:\s+(\d+)', info)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--office-render-dir', type=Path)
    parser.add_argument('--report-dir', type=Path, default=report.OUT)
    parser.add_argument('--desktop-dir', type=Path, default=report.DESKTOP)
    args = parser.parse_args()
    out, desktop = args.report_dir, args.desktop_dir
    manifest = json.loads((out / 'evidence/交付核查.json').read_text())
    results = []
    for entry in manifest['reports']:
        stem = entry['title']
        markdown = (out / (stem + '.md')).read_text()
        main_text = markdown.split('## 参考文献')[0]
        citation_numbers = [int(n) for n in re.findall(r'\[［(\d+)］\]\(https', main_text)]
        assert set(citation_numbers) == set(range(1, entry['references'] + 1))
        assert len(citation_numbers) == entry['citation_occurrences']
        assert len(entry['source_keys']) == len(set(entry['source_keys']))
        assert not re.search(r'\{@|TODO|TBD|OceanSim|73\.07|243569|72\.3\s*km|p0_shadow|RMSE\s*6\.3', markdown)
        assert not re.search(r'已完成.{0,15}(仿真|部署|海试)|我们.{0,15}(仿真|实验结果)', main_text)
        assert '文献证据与阅读范围' in markdown
        assert len(re.findall(r'^### ', main_text, re.M)) >= 35

        pdf_path = out / 'pdf' / (stem + '.pdf')
        pdf_text = subprocess.check_output(['pdftotext', str(pdf_path), '-'], text=True)
        recovery = check_recovery(markdown, pdf_text)
        pages = page_count(pdf_path)
        assert pages == entry['pdf_pages']
        links = subprocess.check_output(['pdfinfo', '-url', str(pdf_path)], text=True)
        assert len(re.findall(r'https?://', links)) >= entry['references']
        # Poppler's extracted text boxes must stay within the A4 page rectangle.
        bbox = subprocess.check_output(['pdftotext', '-bbox', str(pdf_path), '-'], text=True)
        bbox_root = ET.fromstring(bbox)
        xhtml = {'h': 'http://www.w3.org/1999/xhtml'}
        outside = []
        page_nodes = bbox_root.findall('.//h:page', xhtml)
        assert len(page_nodes) == pages
        for page_number, page in enumerate(page_nodes, 1):
            words = page.findall('.//h:word', xhtml)
            assert words
            width, height = float(page.attrib['width']), float(page.attrib['height'])
            for word in words:
                x0, y0, x1, y1 = [float(word.attrib[name]) for name in ('xMin', 'yMin', 'xMax', 'yMax')]
                if x0 < -1 or y0 < -1 or x1 > width + 1 or y1 > height + 1:
                    outside.append((page_number, word.text, (x0, y0, x1, y1)))
        assert not outside, outside

        docx_path = out / 'word' / (stem + '.docx')
        with ZipFile(docx_path) as archive:
            assert archive.testzip() is None
            root = ET.fromstring(archive.read('word/document.xml'))
            docx_text = ''.join(node.text or '' for node in root.findall('.//w:t', report.base.NS))
        docx_recovery = check_recovery(markdown, docx_text)
        for artifact in (pdf_path, docx_path):
            assert digest(artifact) == digest(desktop / artifact.name)
        assert digest(pdf_path) == entry['pdf_sha256']
        assert digest(docx_path) == entry['docx_sha256']

        record = dict(title=stem, pdf_pages=pages, references=entry['references'],
                      citations=len(citation_numbers), pdf_cjk_recovery=recovery,
                      docx_cjk_recovery=docx_recovery, page_bounds='pass', desktop_hashes='match')
        if args.office_render_dir:
            rendered = args.office_render_dir / (stem + '.pdf')
            assert rendered.exists(), rendered
            office_text = subprocess.check_output(['pdftotext', str(rendered), '-'], text=True)
            record.update(native_office_pages=page_count(rendered),
                          native_office_cjk_recovery=check_recovery(markdown, office_text))
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    assert len(results) == 4
    result = dict(status='pass', scope='four_separate_literature_reports_no_local_experiments',
                  unique_sources=manifest['unique_sources'], reports=results)
    (out / 'evidence/格式与范围复核.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
