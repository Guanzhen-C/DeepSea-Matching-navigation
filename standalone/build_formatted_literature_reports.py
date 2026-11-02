#!/usr/bin/env python3
"""Build the four literature reports with a strict four-level Chinese heading scheme."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import build_expanded_literature_reports as expanded
import build_literature_only_reports as report

ROOT = Path(__file__).resolve().parents[1]
SOURCE = expanded.OUT
OUT = ROOT / '研究报告_统一标题格式版_20261008'
DELIVERY = OUT / '交付文件'

CN_DIGITS = '零一二三四五六七八九'
CN_CLASS = '零〇一二三四五六七八九十百两'


def chinese_number(value: int) -> str:
    assert 1 <= value < 100
    if value < 10:
        return CN_DIGITS[value]
    tens, ones = divmod(value, 10)
    prefix = '' if tens == 1 else CN_DIGITS[tens]
    return prefix + '十' + (CN_DIGITS[ones] if ones else '')


def clean_heading(text: str, level: int) -> str:
    text = text.strip()
    if level == 2:
        patterns = (
            rf'^第[{CN_CLASS}0-9]+章\s*',
            rf'^专题研究[{CN_CLASS}0-9]+\s*',
            r'^专题补充\s*',
            rf'^深化研究[{CN_CLASS}0-9]+\s*',
            rf'^[{CN_CLASS}]+、\s*',
        )
    else:
        patterns = (
            r'^\d+(?:\.\d+)*(?:[.、]\s*|\s+)',
            rf'^[{CN_CLASS}]+、\s*',
            r'^（\d+）\s*',
        )
    for pattern in patterns:
        text = re.sub(pattern, '', text, count=1)
    assert text, (level, text)
    return text


def renumber_headings(text: str) -> str:
    lines = text.splitlines()
    second = third = fourth = 0
    formatted = []
    outline = []
    h1_count = 0
    for line in lines:
        match = re.fullmatch(r'(#{1,4})\s+(.+)', line)
        if not match:
            formatted.append(line)
            continue
        level, title = len(match[1]), match[2]
        if level == 1:
            h1_count += 1
            new_title = title.strip()
        elif level == 2:
            second += 1
            third = fourth = 0
            new_title = f'{chinese_number(second)}、{clean_heading(title, 2)}'
        elif level == 3:
            assert second > 0, title
            third += 1
            fourth = 0
            new_title = f'{third}. {clean_heading(title, 3)}'
        else:
            assert third > 0, title
            fourth += 1
            new_title = f'（{fourth}）{clean_heading(title, 4)}'
        formatted.append('#' * level + ' ' + new_title)
        outline.append((level, new_title))
    assert h1_count == 1, h1_count

    # Keep the contents compact: list H2 headings only. H3/H4 remain available
    # through Word's real outline levels and the document's visible hierarchy.
    directory = next(i for i, line in enumerate(formatted)
                     if re.fullmatch(r'##\s+.+、目录', line))
    end = next(i for i in range(directory + 1, len(formatted))
               if formatted[i].startswith('## '))
    entries = []
    for level, title in outline:
        plain_title = re.sub(rf'^[{CN_CLASS}]+、\s*', '', title)
        if level == 1 or (level == 2 and plain_title in
                          ('目录', '参考文献', '文献证据与阅读范围')):
            continue
        if level == 2:
            entries.extend((title, ''))
    formatted[directory + 1:end] = [''] + entries
    return '\n'.join(formatted).rstrip() + '\n'


def validate_headings(path: Path) -> dict:
    text = path.read_text(encoding='utf-8')
    headings = re.findall(r'^(#{1,4})\s+(.+)$', text, re.M)
    h1 = [t for h, t in headings if len(h) == 1]
    h2 = [t for h, t in headings if len(h) == 2]
    h3 = [t for h, t in headings if len(h) == 3]
    h4 = [t for h, t in headings if len(h) == 4]
    assert len(h1) == 1
    assert all(re.fullmatch(rf'[{CN_CLASS}]+、.+', value) for value in h2)
    # H3 numbering restarts under every H2; checking every literal form is enough here.
    assert all(re.fullmatch(r'\d+\.\s+.+', value) for value in h3)
    assert all(re.fullmatch(r'（\d+）.+', value) for value in h4)
    assert not re.search(r'^##\s+(?:第.+章|专题研究|专题补充|深化研究)', text, re.M)
    return {'title': h1[0], 'h1': 1, 'h2': len(h2), 'h3': len(h3), 'h4': len(h4)}


def package(manifest: dict, checks: list[dict]) -> Path:
    archive_path = OUT / '四份研究报告_统一标题格式版_20261008.zip'
    artifacts = []
    by_title = {item['title']: item for item in manifest['reports']}
    for title in sorted(by_title):
        for directory, suffix, hash_key in (('word', '.docx', 'docx_sha256'),
                                             ('pdf', '.pdf', 'pdf_sha256')):
            path = OUT / directory / (title + suffix)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == by_title[title][hash_key]
            artifacts.append((path, title[:3] + '/' + path.name, digest))
    note = ('四份研究报告统一标题格式版\n'
            '一级标题：报告题目。\n'
            '二级标题：一、二、三……。\n'
            '三级标题：1. 2. 3. ……，每个二级标题内重新编号。\n'
            '四级标题：（1）（2）（3）……，每个三级标题内重新编号。\n'
            '四份报告分别成册，每份均含可编辑Word和固定排版PDF。\n'
            '本版仅调整标题层级、编号、目录和对应版式，正文研究内容保持不变。\n')
    with ZipFile(archive_path, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('阅读说明.txt', note)
        for path, name, _ in artifacts:
            archive.write(path, name)
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 9
        for _, name, digest in artifacts:
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    evidence = {
        'archive': str(archive_path),
        'bytes': archive_path.stat().st_size,
        'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        'entries': 9,
        'report_artifacts': 8,
        'crc_check': 'pass',
        'heading_checks': checks,
    }
    (OUT / 'evidence/标题与压缩包核查.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return archive_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report.OUT, report.DESKTOP = OUT, DELIVERY
    report.SOURCES.update(expanded.ADDITIONAL_SOURCES)
    report.FORMULAS.update(expanded.ADDITIONAL_FORMULAS)
    shutil.copytree(SOURCE / 'manuscripts', OUT / 'manuscripts', dirs_exist_ok=True)
    shutil.copytree(SOURCE / 'extensions', OUT / 'extensions', dirs_exist_ok=True)

    original_compile = report.compile_report
    def compile_numbered(source: Path):
        path, order = original_compile(source)
        path.write_text(renumber_headings(path.read_text(encoding='utf-8')), encoding='utf-8')
        return path, order
    report.compile_report = compile_numbered
    report.main()

    checks = [validate_headings(OUT / (entry['title'] + '.md'))
              for entry in json.loads((OUT / 'evidence/交付核查.json').read_text())['reports']]
    manifest = json.loads((OUT / 'evidence/交付核查.json').read_text())
    archive = package(manifest, checks)
    provenance = {
        'source_edition': str(SOURCE),
        'content_change': 'heading hierarchy numbering contents and styles only',
        'output_directory': str(OUT),
        'desktop_copy': False,
        'archive': str(archive),
    }
    (OUT / 'evidence/版本说明.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'archive': str(archive),
                      'heading_checks': checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
