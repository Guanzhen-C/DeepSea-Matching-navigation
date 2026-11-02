#!/usr/bin/env python3
"""Build four independent research reports, with cover, contents and audits.

Existing report editions and simulation code are not changed. The DOCX files
remain editable; their native contents fields can be refreshed in Word.
"""
from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

from fontTools.ttLib import TTFont as FontToolsFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from PIL import Image as PILImage

import build_project_research_reports as base

ROOT = base.ROOT
OUT = ROOT / "docs/project_reports/2026阶段研究_研究报告详述版V3"
NAME = "ResearchCJKV3"
REPORTS = [
    "03月海底地形地貌匹配导航算法调研报告",
    "05月基于迁移学习的海底地形超分算法研究报告",
    "06月深海海底地形地貌匹配重定位算法研究报告",
    "07月深海海底地形地貌匹配声呐里程计定位算法研究报告",
]


class ResearchPDF(BaseDocTemplate):
    def __init__(self, target, title):
        super().__init__(str(target), pagesize=A4, leftMargin=22*mm,
                         rightMargin=22*mm, topMargin=24*mm, bottomMargin=23*mm,
                         title=title, author="OceanSim项目研究报告")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="Research", frames=frame,
                                          onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFont(NAME, 8)
        canvas.setFillColor(colors.HexColor("#657587"))
        if doc.page > 1:
            canvas.drawString(22*mm, 283*mm, "海底地形重建与高精度定位 · 研究报告")
            canvas.setStrokeColor(colors.HexColor("#cad3dc"))
            canvas.line(22*mm, 280*mm, 188*mm, 280*mm)
        canvas.drawString(22*mm, 13*mm, "V3研究详述版 · 资料整理：2026-10-02")
        canvas.drawRightString(188*mm, 13*mm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "SectionV3":
            title = flowable.getPlainText()
            key = f"chapter-{self.seq.nextf('chapter')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=0)
            self.notify("TOCEntry", (0, title, self.page, key))


def export_pdf(path, target):
    common = dict(fontName=NAME, wordWrap="CJK", splitLongWords=True,
                  allowWidows=0, allowOrphans=0)
    styles = {
        "text": ParagraphStyle("BodyV3", fontSize=10.5, leading=18.5,
                               spaceAfter=8, **common),
        "cover": ParagraphStyle("CoverV3", fontSize=24, leading=38,
                                alignment=TA_CENTER, textColor=colors.HexColor("#163955"),
                                spaceAfter=24, **common),
        "meta": ParagraphStyle("MetaV3", fontSize=11, leading=21,
                               alignment=TA_CENTER, spaceAfter=11, **common),
        "toc_title": ParagraphStyle("TOCTitleV3", fontSize=18, leading=27,
                                    alignment=TA_CENTER, spaceAfter=23, **common),
        "h2": ParagraphStyle("SectionV3", fontSize=13.5, leading=22,
                             spaceBefore=15, spaceAfter=9, keepWithNext=True,
                             textColor=colors.HexColor("#163955"), **common),
        "h3": ParagraphStyle("SubsectionV3", fontSize=11.5, leading=19,
                             spaceBefore=9, spaceAfter=7, keepWithNext=True, **common),
        "code": ParagraphStyle("CodeV3", fontSize=8.5, leading=14,
                               spaceAfter=3, backColor=colors.HexColor("#f3f5f7"),
                               borderPadding=5, **common),
        "cell": ParagraphStyle("CellV3", fontSize=8.5, leading=14, **common),
        "caption": ParagraphStyle("CaptionV3", fontSize=9, leading=15,
                                  alignment=TA_CENTER, spaceBefore=6, spaceAfter=11, **common),
    }
    parsed = list(base.blocks(path))
    title = parsed[0][1]
    month = int(path.name[:2])
    story = [Spacer(1, 29*mm), Paragraph(escape(title), styles["cover"]),
             Paragraph("基于迁移学习的海底地形重建<br/>与高精度定位技术研究", styles["meta"]),
             Spacer(1, 12*mm), Paragraph(f"任务归属：{month}月 · 四份独立研究报告之一", styles["meta"]),
             Paragraph("V3研究详述版", styles["meta"]),
             Paragraph("文献调研 · 理论分析 · 工程证据 · 实验方案", styles["meta"]),
             Spacer(1, 18*mm), Paragraph("资料检索与整理日期：2026年10月2日", styles["meta"]),
             Paragraph("依据现有资料补充研究依据，不追认未执行的训练或海试。", styles["caption"]),
             PageBreak(), Paragraph("目录", styles["toc_title"])]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("TOCLevelV3", fontSize=10.5, leading=22,
                                     spaceAfter=5, leftIndent=0, **common)]
    story.extend([toc, PageBreak()])
    for kind, value in parsed[1:]:
        if kind == "table":
            table = Table([[Paragraph(base.inline_pdf(c), styles["cell"]) for c in row]
                           for row in value], colWidths=[166*mm/len(value[0])]*len(value[0]),
                          repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8eff5")),
                ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#bdc9d3")),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING", (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ]))
            story.extend([table, Spacer(1, 10)])
        elif kind == "image":
            caption, image_path = value
            with PILImage.open(image_path) as im:
                width, height = im.size
            draw_width = min(166*mm, 110*mm*width/height)
            story.extend([Image(str(image_path), width=draw_width,
                                height=draw_width*height/width),
                          Paragraph(escape(caption), styles["caption"])])
        elif kind == "code":
            for line in value.splitlines():
                story.append(Paragraph(escape(base.pdf_font_safe(line)), styles["code"]))
            story.append(Spacer(1, 7))
        else:
            story.append(Paragraph(base.inline_pdf(value), styles.get(kind, styles["text"])))
    ResearchPDF(target, title).multiBuild(story)


def export_docx(path, target):
    base.export_docx([path], target)
    with ZipFile(target) as z:
        payload = {name: z.read(name) for name in z.namelist()}
    document = ET.fromstring(payload["word/document.xml"])
    body = document.find("w:body", base.NS)
    original_title = body[0]
    body.remove(original_title)
    parsed = list(base.blocks(path))
    month = int(path.name[:2])
    prefix = ET.Element("prefix")

    def paragraph(text="", style="Normal"):
        p = base.node(prefix, "w:p")
        base.node(base.node(p, "w:pPr"), "w:pStyle", {"w:val": style})
        base.node(base.node(p, "w:r"), "w:t", value=text)
        return p

    def pagebreak():
        p = paragraph()
        base.node(base.node(p, "w:r"), "w:br", {"w:type": "page"})

    paragraph("")
    paragraph("")
    prefix.append(original_title)
    paragraph("基于迁移学习的海底地形重建与高精度定位技术研究", "Caption")
    paragraph("")
    paragraph(f"任务归属：{month}月 · V3研究详述版", "Caption")
    paragraph("文献调研 · 理论分析 · 工程证据 · 实验方案", "Caption")
    paragraph("")
    paragraph("资料检索与整理日期：2026年10月2日", "Caption")
    paragraph("依据现有资料补充研究依据，不追认未执行的训练或海试。", "Caption")
    pagebreak()
    paragraph("目录", "Title")
    p = paragraph()
    base.node(base.node(p, "w:r"), "w:fldChar", {"w:fldCharType": "begin"})
    base.node(base.node(p, "w:r"), "w:instrText", value=' TOC \\o "1-1" \\h \\z \\u ')
    base.node(base.node(p, "w:r"), "w:fldChar", {"w:fldCharType": "separate"})
    for kind, value in parsed:
        if kind == "h2":
            paragraph(value)
    p = paragraph()
    base.node(base.node(p, "w:r"), "w:fldChar", {"w:fldCharType": "end"})
    paragraph("目录为可更新字段；在Word中可右击目录，更新页码与章节链接。", "Caption")
    pagebreak()
    for index, elem in enumerate(list(prefix)):
        body.insert(index, elem)
    payload["word/document.xml"] = base.xml(document)
    footer = payload["word/footer1.xml"].decode("utf-8")
    payload["word/footer1.xml"] = footer.replace("阶段研究材料", "研究报告 V3").encode("utf-8")
    styles = ET.fromstring(payload["word/styles.xml"])
    for prop in styles.findall(".//w:rFonts", base.NS):
        prop.set(f"{{{base.NS['w']}}}eastAsia", "文泉驿正黑")
        prop.set(f"{{{base.NS['w']}}}ascii", "DejaVu Sans")
        prop.set(f"{{{base.NS['w']}}}hAnsi", "DejaVu Sans")
    payload["word/styles.xml"] = base.xml(styles)
    with ZipFile(target, "w", ZIP_DEFLATED) as z:
        for name, data in payload.items():
            z.writestr(name, data)


def audit_source(path):
    text = path.read_text(encoding="utf-8")
    prefix = {"03":"M", "05":"S", "06":"G", "07":"O"}[path.name[:2]]
    main, refs = text.split("## 参考文献与资料", 1)
    ids = re.findall(rf"(?m)^{prefix}\d\d(?= )", refs)
    citations = set(re.findall(rf"\[({prefix}\d\d)\]\(", main))
    assert len(ids) == len(set(ids))
    assert citations <= set(ids), (path, citations-set(ids))
    assert set(ids) <= citations, (path, set(ids)-citations)
    assert not re.search(r"TODO|TBD|turn\d+(?:search|view)\d+", text)
    return {"file": path.name, "characters": len(text),
            "chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", text)),
            "chapters": len(re.findall(r"(?m)^## ", text)),
            "reference_entries": len(ids), "all_references_cited_in_body": True}


def main():
    for folder in ("word", "pdf", "evidence/render_checks"):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    base.verify_and_plot()
    pdfmetrics.registerFont(TTFont(NAME, str(base.FONT)))
    font = FontToolsFont(str(base.FONT), fontNumber=0)
    cmap = font.getBestCmap()
    reports, files = [], []
    for stem in REPORTS:
        path = OUT / f"{stem}.md"
        info = audit_source(path)
        pdf = OUT/"pdf"/f"{stem}_V3研究详述版.pdf"
        docx = OUT/"word"/f"{stem}_V3研究详述版.docx"
        export_pdf(path, pdf)
        export_docx(path, docx)
        with ZipFile(docx) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith((".xml", ".rels")):
                    ET.fromstring(z.read(name))
        pdf_info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        info["pdf_pages"] = int(next(line.split(":", 1)[1] for line in pdf_info.splitlines()
                                     if line.startswith("Pages:")))
        extracted = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
        assert "目录" in extracted and "参考文献与资料" in extracted
        assert "Placeholder" not in extracted
        missing = sorted({c for c in extracted if not c.isspace() and ord(c) not in cmap})
        assert not missing, (stem, missing)
        body_text = "\n".join(line for line in extracted.splitlines()
                              if not line.strip().startswith(("海底地形重建与高精度定位 · 研究报告",
                                                              "V3研究详述版 · 资料整理：")))
        compact_body = re.sub(r"\s", "", body_text)
        for kind, value in base.blocks(path):
            if kind in ("image", "table"):
                continue
            val = base.pdf_font_safe(base.LINK.sub(lambda m:m[1], value)).replace("`", "")
            for para in val.splitlines():
                if len(para) > 25:
                    assert re.sub(r"\s", "", para) in compact_body, (stem, para[:80])
        info["font_coverage"] = "passed"
        info["full_paragraph_pdf_presence"] = "passed"
        info["docx_xml_validation"] = "passed"
        reports.append(info)
        for p in (path, pdf, docx):
            files.append({"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size,
                          "sha256": base.digest(p)})
        print(json.dumps(info, ensure_ascii=False), flush=True)
    retrieval = json.loads((OUT/"evidence/new_literature/retrieval_manifest.json").read_text())
    manifest = {"built_on": "2026-10-02", "reports": reports,
                "total_main_report_characters": sum(r["characters"] for r in reports),
                "total_chinese_characters": sum(r["chinese_characters"] for r in reports),
                "total_pdf_pages": sum(r["pdf_pages"] for r in reports),
                "reference_entries_including_repetition": sum(r["reference_entries"] for r in reports),
                "new_local_retrieval_status": dict(Counter(r["status"] for r in retrieval["sources"])),
                "standalone_reports_only": True, "binder_created": False,
                "no_navigation_code_or_experiments_changed": True, "files": files}
    (OUT/"evidence/delivery_manifest_v3.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Validated four independent reports, {manifest['total_pdf_pages']} PDF pages.")


if __name__ == "__main__":
    main()
