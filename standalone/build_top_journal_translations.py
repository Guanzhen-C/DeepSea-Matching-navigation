#!/usr/bin/env python3
"""Export individually attributed Chinese paper excerpts, never a report binder."""
from html import escape
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

import build_project_research_reports as base

ROOT = base.ROOT
OUT = ROOT / "docs/literature_translations/20261002_top_journals"
DESKTOP = Path("/home/cgz/Desktop/领域权威期刊_三篇中文节译")


def equations():
    formulas = {
        "sr_2": r"x=\mathcal{D}(y;\delta)\qquad (2)",
        "sr_3": r"\hat y=\mathcal{M}(x;\theta)\qquad (3)",
        "sr_4": r"\hat\theta=\arg\min_{\theta}\mathcal{L}(\hat y,y)\qquad (4)",
        "sr_5": r"\mathrm{PSNR}(y,\hat y)=10\log_{10}\frac{L^2}{\frac{1}{N_y}\sum_{p\in\Omega_y}(y_p-\hat y_p)^2}\qquad (5)",
        "sr_22": r"\hat y=\mu_{\theta}(x)+\epsilon\,\sigma_{\theta}(x),\quad\epsilon\sim\mathcal{N}(0,I)\qquad (22)",
        "sr_23": r"\mathcal{L}_{\mathrm{ESU}}=\exp(-\ln\hat y_{\sigma})\,\|y-\hat y\|_1+2\ln\hat y_{\sigma}\qquad (23)",
        "sr_24": r"\mathcal{L}_{\mathrm{UDL}}=[\ln\hat y_{\sigma}-\min(\ln\hat y_{\sigma})]\,\|y-\hat y_{\mu}\|_1\qquad (24)",
        "sss_1": r"F(a)=b-a\qquad (1)",
        "sss_3": r"z_i=[r_i,0]^T=[\|\pi({}^{o}x)\|,(1,0,0)\,\pi({}^{o}x)]^T+\eta\qquad (3)",
        "sss_4": r"{}^{s}\bar x=({}^{b}T_s)^{-1}({}^{c}T_b)^{-1}({}^{o}T_c)^{-1}\,{}^{o}\bar x\qquad (4)",
        "sss_6": r"\Sigma_i=\mathrm{diag}(\sigma_r^2,r_i^2\alpha^2)\qquad (6)",
    }
    for name, formula in formulas.items():
        target = OUT / "assets" / f"{name}.png"
        fig = plt.figure(figsize=(9, .68), dpi=200)
        fig.text(.5, .5, "$" + formula + "$", ha="center", va="center", fontsize=16)
        fig.savefig(target, bbox_inches="tight", pad_inches=.10, facecolor="white")
        plt.close(fig)


def export_pdf(source, target):
    font = "TranslationCJK"
    pdfmetrics.registerFont(TTFont(font, str(base.FONT)))
    pdfmetrics.registerFont(TTFont("TranslationLatin", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    def inline(value):
        rendered = base.inline_pdf(value)
        return re.sub(r"w̄|h̄|ŷ|[ᵇᵒᶜ]", lambda match:
                      '<font name="TranslationLatin">' + match[0] + '</font>', rendered)
    common = dict(fontName=font, wordWrap="CJK", splitLongWords=True)
    styles = {
        "text": ParagraphStyle("Body", fontSize=10.5, leading=18, spaceAfter=9, **common),
        "h1": ParagraphStyle("Title", fontSize=18, leading=28, alignment=TA_CENTER,
                             spaceAfter=18, textColor=colors.HexColor("#163955"), **common),
        "h2": ParagraphStyle("Section", fontSize=13, leading=21, spaceBefore=14,
                             spaceAfter=8, keepWithNext=True, **common),
        "h3": ParagraphStyle("Subsection", fontSize=11.5, leading=19, spaceBefore=9,
                             spaceAfter=6, keepWithNext=True, **common),
        "caption": ParagraphStyle("Caption", fontSize=8.5, leading=14,
                                  alignment=TA_CENTER, spaceAfter=10, **common),
        "code": ParagraphStyle("EquationText", fontSize=9, leading=15, spaceAfter=6, **common),
    }
    story = []
    for kind, value in base.blocks(source):
        if kind == "image":
            caption, path = value
            with PILImage.open(path) as im:
                w, h = im.size
            width = min(166 * mm, w / 200 * 72)
            story += [Image(str(path), width=width, height=width * h / w),
                      Paragraph(escape(caption), styles["caption"])]
        elif kind == "code":
            for line in value.splitlines():
                story.append(Paragraph(escape(base.pdf_font_safe(line)), styles["code"]))
        elif kind == "table":
            raise ValueError("Translation files use paragraphs, not compressed summary tables")
        else:
            story.append(Paragraph(inline(value), styles.get(kind, styles["text"])))
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#657587"))
        canvas.drawString(22*mm, 285*mm, "领域权威期刊文献 · 非官方中文节译")
        canvas.drawString(22*mm, 13*mm, "译稿整理：2026-10-02 · 原文著作权归原作者")
        canvas.drawRightString(188*mm, 13*mm, str(doc.page))
        canvas.restoreState()
    SimpleDocTemplate(str(target), pagesize=A4, leftMargin=22*mm, rightMargin=22*mm,
                      topMargin=23*mm, bottomMargin=22*mm, title=source.stem,
                      author="非官方中文译稿整理").build(story, onFirstPage=decorate,
                                                        onLaterPages=decorate)


def export_docx(source, target):
    base.export_docx([source], target)
    with ZipFile(target) as z:
        payload = {name: z.read(name) for name in z.namelist()}
    payload["word/footer1.xml"] = payload["word/footer1.xml"].replace(
        "阶段研究材料".encode(), "非官方中文节译".encode())
    styles = ET.fromstring(payload["word/styles.xml"])
    for font in styles.findall(".//w:rFonts", base.NS):
        for attr in ("ascii", "hAnsi"):
            font.set(f"{{{base.NS['w']}}}{attr}", "DejaVu Sans")
        font.set(f"{{{base.NS['w']}}}eastAsia", "文泉驿正黑")
    payload["word/styles.xml"] = base.xml(styles)
    with ZipFile(target, "w", ZIP_DEFLATED) as z:
        for name, value in payload.items():
            z.writestr(name, value)


def main():
    equations()
    DESKTOP.mkdir(exist_ok=True)
    evidence = []
    for source in sorted(OUT.glob("0[123]_*.md")):
        body = source.read_text()
        assert "非官方中文节译" in body and "不是全文翻译" in body
        assert "https://creativecommons.org/licenses/by/4.0/" in body
        assert not re.search(r"\b(?:TODO|TBD)\b|turn\d+(?:view|search)", body)
        for suffix, exporter in (("pdf", export_pdf), ("docx", export_docx)):
            target = OUT / (source.stem + "." + suffix)
            exporter(source, target)
            if suffix == "docx":
                with ZipFile(target) as z:
                    assert z.testzip() is None
                    for name in z.namelist():
                        if name.endswith((".xml", ".rels")):
                            ET.fromstring(z.read(name))
            else:
                info = subprocess.check_output(["pdfinfo", str(target)], text=True)
                pages = int(re.search(r"Pages:\s+(\d+)", info)[1])
                assert pages >= 4
            shutil.copy2(target, DESKTOP / target.name)
            evidence.append({"file": target.name, "bytes": target.stat().st_size,
                             "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                             "source_characters": len(body), "scope": "selected sections, not full text"})
        print(source.stem, len(body), "characters")
    shutil.copy2(OUT / "重点文献选择说明.md", DESKTOP / "重点文献选择说明.md")
    (OUT / "validation.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(DESKTOP)


if __name__ == "__main__":
    main()
