"""Build editable Word and PDF versions of the repository report source.

Scientific figures are embedded without geometric stretching. The source is
ordinary repository Markdown, and Word headings preserve its H1/H2/H3/H4
hierarchy. This utility never changes the source prose or experiment data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE
from PIL import Image


def set_font(style, name, east_asia, size):
    style.font.name = name; style.font.size = Pt(size)
    style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), east_asia)


def inline(paragraph, text):
    pattern = r'(\[[^\]]+\]\([^)]+\)|`[^`]+`|\*\*[^*]+\*\*)'
    for part in re.split(pattern, text):
        if not part: continue
        link = re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', part)
        if link:
            label, destination = link.groups()
            if destination.startswith('https://'):
                rid = paragraph.part.relate_to(destination, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
                node = OxmlElement('w:hyperlink'); node.set(qn('r:id'), rid)
                run = OxmlElement('w:r'); props = OxmlElement('w:rPr')
                color = OxmlElement('w:color'); color.set(qn('w:val'), '17638C'); props.append(color)
                run.append(props); value = OxmlElement('w:t'); value.text = label; run.append(value)
                node.append(run); paragraph._p.append(node)
            else: paragraph.add_run(label)
        elif part.startswith('`'):
            run = paragraph.add_run(part[1:-1]); run.font.name = 'Consolas'; run.font.size = Pt(9)
        elif part.startswith('**'): paragraph.add_run(part[2:-2]).bold = True
        else: paragraph.add_run(part)


def add_table(document, lines):
    rows = [[cell.strip() for cell in row.strip().strip('|').split('|')] for row in lines]
    rows = [row for row in rows if not all(re.fullmatch(r':?-+:?', cell) for cell in row)]
    if not rows or any(len(row) != len(rows[0]) for row in rows): raise ValueError('Malformed report table.')
    table = document.add_table(rows=0, cols=len(rows[0])); table.style = 'Table Grid'
    table.autofit = True
    for i, values in enumerate(rows):
        row = table.add_row()
        dontsplit = OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(dontsplit)
        if i == 0:
            repeat = OxmlElement('w:tblHeader'); row._tr.get_or_add_trPr().append(repeat)
        for cell, value in zip(row.cells, values):
            paragraph = cell.paragraphs[0]; inline(paragraph, value)
            paragraph.paragraph_format.space_after = Pt(4)
            for run in paragraph.runs:
                run.font.size = Pt(9); run.bold = i == 0
    document.add_paragraph()


def build(source, target):
    if target.exists(): raise FileExistsError('Preserve earlier Word documents.')
    doc = Document(); section = doc.sections[0]
    section.page_width = Cm(21); section.page_height = Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2.2)
    section.left_margin = section.right_margin = Cm(2.3)
    set_font(doc.styles['Normal'], 'Times New Roman', '宋体', 11)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.35
    doc.styles['Normal'].paragraph_format.space_after = Pt(6)
    for level, size in [(1, 22), (2, 16), (3, 13), (4, 11.5)]:
        style = doc.styles[f'Heading {level}']
        set_font(style, 'Arial', '黑体', size); style.font.color.rgb = RGBColor(25, 42, 55)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(13)
        style.paragraph_format.space_after = Pt(7)
        if level == 2: style.paragraph_format.page_break_before = True
    header = section.header.paragraphs[0]
    header.text = '海底地形地貌匹配算法仿真报告'; header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs: run.font.size = Pt(8)
    footer = section.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('— ')
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    footer.add_run(' —')
    lines = source.read_text().splitlines(); i = 0; images = []; headings = []; tables = 0
    while i < len(lines):
        text = lines[i].strip()
        if not text: i += 1; continue
        if text.startswith('```'):
            block = []; i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                block.append(lines[i]); i += 1
            paragraph = doc.add_paragraph()
            run = paragraph.add_run('\n'.join(block)); run.font.name = 'Consolas'; run.font.size = Pt(8)
            shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'F0F3F5')
            paragraph._p.get_or_add_pPr().append(shade); i += 1; continue
        if text.startswith('|'):
            block = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i]); i += 1
            add_table(doc, block); tables += 1; continue
        image = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', text)
        if image:
            path = (source.parent/image[2]).resolve()
            if not path.is_relative_to(source.parent.resolve()): raise ValueError('Unexpected report image location.')
            with Image.open(path) as im: width, height = im.size
            size = min(16.0/width, 19.0/height)
            paragraph = doc.add_paragraph(); paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.add_run().add_picture(str(path), width=Cm(width*size), height=Cm(height*size))
            images.append(dict(file=image[2], native_shape=[height,width],
                placed_width_cm=width*size, placed_height_cm=height*size, aspect_ratio_preserved=True))
            i += 1; continue
        heading = re.match(r'^(#{1,4})\s+(.+)$', text)
        if heading:
            level, title = len(heading[1]), heading[2]
            paragraph = doc.add_heading(title, level=level)
            if level == 1: paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            headings.append(dict(level=level, text=title)); i += 1; continue
        paragraph_text = text; hard_break = lines[i].endswith('  '); i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|```)', lines[i].strip()):
            paragraph_text += ('\n' if hard_break else ' ')+lines[i].strip()
            hard_break = lines[i].endswith('  '); i += 1
        paragraph = doc.add_paragraph(); inline(paragraph, paragraph_text)
        if text.startswith('图') and re.match(r'^图\d+', text):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs: run.font.size = Pt(9)
    doc.core_properties.title = '海底地形地貌匹配算法仿真报告'
    doc.core_properties.subject = '处理器在环与实测来源纯侧扫导航仿真'
    doc.core_properties.author = '项目研究组'
    doc.save(target)
    reopened = Document(target)
    if len(reopened.inline_shapes) != len(images) or len(reopened.tables) != tables:
        raise ValueError('Saved document did not retain every figure/table.')
    return dict(images=images, headings=headings, tables=tables, editable_word_reopened=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--pdf', action='store_true')
    args = p.parse_args(); source = args.source.resolve()
    target = source.with_suffix('.docx'); pdf = source.with_suffix('.pdf')
    if args.pdf and pdf.exists(): raise FileExistsError('Preserve prior PDF.')
    audit = build(source, target)
    if args.pdf:
        profile = Path(tempfile.mkdtemp(prefix='oceansim-report-lo-'))
        environment = {k:v for k,v in os.environ.items() if k not in
            ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH')}
        result = subprocess.run(['/usr/bin/libreoffice', '-env:UserInstallation='+profile.as_uri(),
            '--headless', '--convert-to', 'pdf', '--outdir', str(source.parent), str(target)],
            capture_output=True, text=True, env=environment)
        if result.returncode:
            raise RuntimeError('Isolated PDF conversion failed: '+result.stderr.strip())
        if not pdf.is_file(): raise RuntimeError('PDF conversion did not produce the requested output.')
        audit.update(pdf_conversion_completed=True, libreoffice_output=result.stdout.strip())
    (source.parent/'evidence/document_build.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(word=str(target), pdf=str(pdf) if args.pdf else None,
        embedded_figures=len(audit['images']), tables=audit['tables']), ensure_ascii=False))


if __name__ == '__main__': main()
