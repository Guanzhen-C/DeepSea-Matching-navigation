#!/usr/bin/env python3
"""Build research-report figures, verify archived metrics and export DOCX/PDF.

No navigation code or archived experiment is changed. DOCX uses standard
OOXML so no document-authoring package is needed. Generated report artifacts
are reproducible from the Markdown sources and existing experiment files.
"""
from __future__ import annotations

import hashlib
import argparse
from html import escape
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/project_reports/2026阶段研究"
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
EXPERIMENTS = ROOT / "demo/large_scale_slam"
NEW = "p0_shadow_confirm_full_route_headless_20260917"
OLD = "strict_frontend_backend_fallback_full_route_20260917"
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verify_and_plot() -> None:
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
                         "axes.unicode_minus": False, "font.size": 11})
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    sources, loaded, metrics = [], {}, {}
    for name in (NEW, OLD):
        csv_path = EXPERIMENTS / f"{name}_position_errors.csv"
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        source = EXPERIMENTS / f"{name}.json"
        record = json.loads(source.read_text(encoding="utf-8"))
        stats = record["position_error_statistics"]
        assert len(data) == 243569
        assert np.all(np.isfinite(data["online_global_slam_3d_m"]))
        assert np.allclose(np.diff(data["timestamp_s"]), 0.2, atol=1e-6)
        evaluated = {}
        for col, key in (("dead_reckoning_3d_m", "dead_reckoning_imu_dvl_depth"),
                         ("local_acoustic_esikf_3d_m", "local_acoustic_esikf"),
                         ("online_global_slam_3d_m", "online_global_slam")):
            values = data[col]
            item = {"samples": len(values), "rmse_3d_m": float(np.sqrt(np.mean(values**2))),
                    "median_3d_m": float(np.median(values)), "p95_3d_m": float(np.percentile(values, 95)),
                    "max_3d_m": float(np.max(values)), "final_3d_m": float(values[-1])}
            for field in item:
                assert np.isclose(item[field], stats[key][field], rtol=1e-10, atol=1e-7), (name, key, field)
            evaluated[key] = item
        sources.extend([{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in (source, csv_path)])
        loaded[name] = data
        metrics[name] = {"position": evaluated, "se3_rpe_by_truth_distance": stats["se3_rpe_by_truth_distance"],
                         "survey_travelled_distance_m": record["survey_travelled_distance_m"]}
    new, old = loaded[NEW], loaded[OLD]
    for field in ("dead_reckoning_3d_m", "local_acoustic_esikf_3d_m"):
        assert np.array_equal(new[field], old[field]), field
    distance = (new["timestamp_s"] - new["timestamp_s"][0]) * 1.5 / 1000
    vals = new["online_global_slam_3d_m"]
    peak = int(np.argmax(vals))
    risk = (distance >= 59) & (distance <= 60.5)
    old_rmse = metrics[OLD]["position"]["online_global_slam"]["rmse_3d_m"]
    new_rmse = metrics[NEW]["position"]["online_global_slam"]["rmse_3d_m"]
    verification = {"verified_on": "2026-10-02", "sources": sources,
                    "metrics": metrics, "csv_matches_archived_json": True,
                    "identical_dr_and_local_between_runs": True,
                    "distance_axis": "elapsed simulated time times 1.5 m/s; differs from integrated truth distance by less than 0.1 m at endpoint",
                    "global_peak_distance_km": float(distance[peak]),
                    "global_peak_error_m": float(vals[peak]),
                    "risk_59_to_60_5km_max_error_m": float(np.max(vals[risk])),
                    "online_global_rmse_reduction_percent": 100 * (1-new_rmse/old_rmse)}
    (OUT / "evidence/long_route_metric_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
    axes[0].plot(distance, new["dead_reckoning_3d_m"], color="#667085", lw=1.1, label="纯航位推算 DR")
    axes[0].plot(distance, new["local_acoustic_esikf_3d_m"], color="#eb7c20", lw=1.05, label="局部里程计 Local")
    axes[0].plot(distance, vals, color="#007fac", lw=0.9, label="在线 Global 改进版")
    axes[0].set_title("73.07 km归档实验：三维位置误差（5 Hz，243569个样本）")
    axes[0].set_ylabel("误差 / m")
    axes[0].legend(loc="upper left", ncol=3)
    axes[1].plot(distance, old["online_global_slam_3d_m"], color="#b9656a", lw=0.7, alpha=0.7, label="改进前在线 Global")
    axes[1].plot(distance, vals, color="#007fac", lw=0.85, label="改进版在线 Global")
    axes[1].axhline(11.7064047368, color="#007fac", ls="--", lw=0.9, label="改进版 P95：11.71 m")
    axes[1].axvspan(59, 60.5, color="#f2d8b1", alpha=0.35)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_ylim(0, 600)
    axes[1].set_yticks([0,1,5,10,50,100,500], labels=["0","1","5","10","50","100","500"])
    axes[1].set_ylabel("在线误差 / m\n1 m以上对数刻度")
    axes[1].set_xlabel("从本次起航累计距离 / km（起航位置在主航线30 km处）")
    axes[1].legend(loc="upper left", ncol=3)
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.set_xlim(0, 73.08)
    fig.savefig(figures / "73km在线定位误差与对照.png", dpi=180)
    plt.close(fig)
    baseline = json.loads((OUT / "evidence/sr_interpolation_baseline.json").read_text())
    pooled = baseline["pooled_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = ["最近邻", "双线性", "双三次"]
    methods = ["nearest", "bilinear", "bicubic"]
    x = np.arange(3)
    for ax, key, unit in ((axes[0], "rmse_m", "高程RMSE / m"), (axes[1], "slope_rmse_deg", "坡度RMSE / °")):
        for scale, offset, color in ((2, -.18, "#007fac"), (4, .18, "#e5a13a")):
            values = [pooled[f"x{scale}_{m}"][key] for m in methods]
            ax.bar(x+offset, values, width=.34, color=color, label=f"{scale}倍退化后恢复")
        ax.set_xticks(x, labels=labels)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=.2)
        ax.legend()
    fig.suptitle("Jenkinson实测栅格的受控插值对照：12个完整有效块（未训练网络）")
    fig.savefig(figures / "地形超分插值基线.png", dpi=180)
    plt.close(fig)


def blocks(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            yield "code", "\n".join(code)
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            yield "table", rows
            continue
        elif line.startswith("!["):
            found = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if not found:
                raise ValueError(line)
            image_path = path.parent / found[2]
            assert image_path.exists(), image_path
            yield "image", (found[1], image_path)
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            yield f"h{level}", line[level:].strip()
        else:
            para = [line]
            while i+1 < len(lines) and lines[i+1].strip() and not lines[i+1].startswith(("#", "|", "```", "![")):
                i += 1
                para.append(lines[i].strip())
            yield "text", " ".join(para)
        i += 1


def pdf_font_safe(value: str) -> str:
    """Preserve mathematical meaning when the CJK font lacks a codepoint."""
    for source, replacement in (("Ĥ", "H_hat"), ("R̂", "R_hat"),
                                ("Ṙ", "dR/dt"), ("v̇", "dv/dt"), ("ṗ", "dp/dt"),
                                ("ḣ_D", "dh_D/dt"), ("ᵀ", "^T"),
                                ("⁻¹", "^(-1)"), ("⁽ⁿ⁾", "^(n)"),
                                ("₀", "_0"), ("−", "-")):
        value = value.replace(source, replacement)
    return value


def inline_pdf(value: str) -> str:
    value = pdf_font_safe(value)
    output, start = [], 0
    for match in LINK.finditer(value):
        output.append(escape(value[start:match.start()]))
        label, target = match.groups()
        output.append(f'<link href="{escape(target, quote=True)}" color="#126688">{escape(label)}</link>')
        start = match.end()
    output.append(escape(value[start:]))
    return "".join(output).replace("`", "")


def export_pdf(paths: list[Path], target: Path, binder=False) -> None:
    name = "ResearchCJK"
    pdfmetrics.registerFont(TTFont(name, str(FONT)))
    common = dict(fontName=name, wordWrap="CJK", splitLongWords=True)
    styles = {
        "text": ParagraphStyle("Body", fontSize=10.5, leading=18, spaceAfter=8, **common),
        "h1": ParagraphStyle("Title", fontSize=19, leading=29, alignment=TA_CENTER,
                             textColor=colors.HexColor("#163955"), spaceBefore=8, spaceAfter=20, **common),
        "h2": ParagraphStyle("Section", fontSize=13, leading=21, spaceBefore=13, spaceAfter=8,
                             keepWithNext=True, textColor=colors.HexColor("#163955"), **common),
        "h3": ParagraphStyle("Subsection", fontSize=11.5, leading=19, spaceBefore=8,
                             spaceAfter=6, keepWithNext=True, **common),
        "code": ParagraphStyle("Code", fontSize=8, leading=13, spaceAfter=8,
                               backColor=colors.HexColor("#f3f5f7"), borderPadding=7, **common),
        "cell": ParagraphStyle("Cell", fontSize=8.8, leading=14, **common),
        "caption": ParagraphStyle("Caption", fontSize=9, leading=15, alignment=TA_CENTER, spaceAfter=10, **common),
    }
    content = []
    if binder:
        content.extend([Spacer(1, 27*mm), Paragraph("基于迁移学习的海底地形重建<br/>与高精度定位技术研究", styles["h1"]),
                        Paragraph("1—9月阶段研究材料合订本", styles["h1"]),
                        Paragraph("2026年10月2日整理 · 技术报告与实施依据", styles["caption"]),
                        Spacer(1, 12*mm)])
        for path in paths:
            title = next(blocks(path))[1]
            content.append(Paragraph(escape(title), styles["text"]))
        content.append(PageBreak())
    for index, path in enumerate(paths):
        if index:
            content.append(PageBreak())
        for kind, val in blocks(path):
            if kind == "table":
                widths = [165*mm/len(val[0])] * len(val[0])
                table = Table([[Paragraph(inline_pdf(c), styles["cell"]) for c in row] for row in val],
                              colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0,0),(-1,0),colors.HexColor("#e8eff5")),
                    ("VALIGN", (0,0),(-1,-1),"TOP"),
                    ("GRID", (0,0),(-1,-1),.35,colors.HexColor("#bdc9d3")),
                    ("LEFTPADDING", (0,0),(-1,-1),6), ("RIGHTPADDING", (0,0),(-1,-1),6),
                    ("TOPPADDING", (0,0),(-1,-1),6), ("BOTTOMPADDING", (0,0),(-1,-1),6),
                ]))
                content.extend([table, Spacer(1, 9)])
            elif kind == "image":
                caption, image_path = val
                with PILImage.open(image_path) as im:
                    width, height = im.size
                draw_width = 165*mm
                content.extend([Image(str(image_path), width=draw_width, height=draw_width*height/width),
                                Paragraph(escape(caption), styles["caption"])])
            elif kind == "code":
                for line in val.splitlines():
                    content.append(Paragraph(escape(pdf_font_safe(line)), styles["code"]))
            else:
                style = styles.get(kind, styles["h3"] if kind.startswith("h") else styles["text"])
                content.append(Paragraph(inline_pdf(val), style))
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cad3dc"))
        canvas.line(22*mm, 282*mm, 187*mm, 282*mm)
        canvas.setFont(name, 8)
        canvas.setFillColor(colors.HexColor("#657587"))
        canvas.drawString(22*mm, 285*mm, "海底地形重建与高精度定位 · 阶段研究材料")
        canvas.drawString(22*mm, 14*mm, "整理日期：2026-10-02")
        canvas.drawRightString(187*mm, 14*mm, f"{doc.page}")
        canvas.restoreState()
    SimpleDocTemplate(str(target), pagesize=A4, leftMargin=22*mm, rightMargin=23*mm,
                      topMargin=22*mm, bottomMargin=22*mm,
                      title=target.stem, author="OceanSim项目研究材料").build(
        content, onFirstPage=decorate, onLaterPages=decorate)


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture"}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def node(parent, tag, attrs=None, value=None):
    prefix, local = tag.split(":")
    out = ET.SubElement(parent, f"{{{NS[prefix]}}}{local}")
    for key, val in (attrs or {}).items():
        if ":" in key:
            pre, loc = key.split(":")
            out.set(f"{{{NS[pre]}}}{loc}", str(val))
        else:
            out.set(key, str(val))
    if value is not None:
        out.text = value
    return out


def xml(element) -> bytes:
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def export_docx(paths: list[Path], target: Path, binder=False) -> None:
    document = ET.Element(f"{{{NS['w']}}}document")
    body = node(document, "w:body")
    relationships, media = [], []
    def relationship(kind, dest, external=False):
        rid = f"rId{len(relationships)+1}"
        relationships.append((rid, kind, dest, external))
        return rid
    relationship("styles", "styles.xml")
    footer_rid = relationship("footer", "footer1.xml")
    def run(parent, text, hyperlink=False):
        r = node(parent, "w:r")
        if hyperlink:
            rp = node(r, "w:rPr")
            node(rp, "w:color", {"w:val": "126688"})
            node(rp, "w:u", {"w:val": "single"})
        t = node(r, "w:t", value=text.replace("`", ""))
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    def paragraph(parent, value, style="Normal"):
        p = node(parent, "w:p")
        pp = node(p, "w:pPr")
        node(pp, "w:pStyle", {"w:val": style})
        start = 0
        for match in LINK.finditer(value):
            run(p, value[start:match.start()])
            rid = relationship("hyperlink", match[2], True)
            link = node(p, "w:hyperlink", {"r:id": rid})
            run(link, match[1], True)
            start = match.end()
        run(p, value[start:])
        return p
    def pagebreak():
        p = node(body, "w:p")
        node(node(p, "w:r"), "w:br", {"w:type": "page"})
    if binder:
        paragraph(body, "基于迁移学习的海底地形重建与高精度定位技术研究", "Title")
        paragraph(body, "1—9月阶段研究材料合订本", "Title")
        paragraph(body, "2026年10月2日整理 · 技术报告与实施依据", "Caption")
        for path in paths:
            paragraph(body, next(blocks(path))[1])
        pagebreak()
    for index, path in enumerate(paths):
        if index:
            pagebreak()
        for kind, val in blocks(path):
            if kind == "table":
                table = node(body, "w:tbl")
                properties = node(table, "w:tblPr")
                node(properties, "w:tblW", {"w:w": 9072, "w:type": "dxa"})
                node(properties, "w:tblLayout", {"w:type": "fixed"})
                borders = node(properties, "w:tblBorders")
                for border in ("top", "left", "bottom", "right", "insideH", "insideV"):
                    node(borders, f"w:{border}", {"w:val": "single", "w:sz": 4, "w:color": "BDC9D3"})
                grid = node(table, "w:tblGrid")
                cell_width = 9072//len(val[0])
                for _ in val[0]:
                    node(grid, "w:gridCol", {"w:w": cell_width})
                for row_index, row in enumerate(val):
                    tr = node(table, "w:tr")
                    if row_index == 0:
                        node(node(tr, "w:trPr"), "w:tblHeader")
                    for cell in row:
                        tc = node(tr, "w:tc")
                        prop = node(tc, "w:tcPr")
                        node(prop, "w:tcW", {"w:w": cell_width, "w:type": "dxa"})
                        if row_index == 0:
                            node(prop, "w:shd", {"w:fill": "E8EFF5"})
                        paragraph(tc, cell, "TableText")
                paragraph(body, "")
            elif kind == "image":
                caption, image_path = val
                number = len(media)+1
                filename = f"figure{number}.png"
                media.append((filename, image_path.read_bytes()))
                rid = relationship("image", f"media/{filename}")
                with PILImage.open(image_path) as im:
                    width, height = im.size
                cx, cy = 5760000, int(5760000*height/width)
                p = node(body, "w:p")
                node(node(p, "w:pPr"), "w:jc", {"w:val": "center"})
                drawing = node(node(p, "w:r"), "w:drawing")
                inline = node(drawing, "wp:inline")
                node(inline, "wp:extent", {"cx": cx, "cy": cy})
                node(inline, "wp:docPr", {"id": number, "name": caption})
                graphic = node(inline, "a:graphic")
                gd = node(graphic, "a:graphicData", {"uri": NS["pic"]})
                pic = node(gd, "pic:pic")
                nvp = node(pic, "pic:nvPicPr")
                node(nvp, "pic:cNvPr", {"id": number, "name": filename})
                node(nvp, "pic:cNvPicPr")
                fill = node(pic, "pic:blipFill")
                node(fill, "a:blip", {"r:embed": rid})
                node(node(fill, "a:stretch"), "a:fillRect")
                sp = node(pic, "pic:spPr")
                xfrm = node(sp, "a:xfrm")
                node(xfrm, "a:off", {"x": 0, "y": 0})
                node(xfrm, "a:ext", {"cx": cx, "cy": cy})
                node(node(sp, "a:prstGeom", {"prst": "rect"}), "a:avLst")
                paragraph(body, caption, "Caption")
            elif kind == "code":
                for line in val.splitlines():
                    paragraph(body, line, "Code")
            else:
                style = {"h1":"Title", "h2":"Heading1", "h3":"Heading2", "h4":"Heading3"}.get(kind, "Normal")
                paragraph(body, val, style)
    section = node(body, "w:sectPr")
    node(section, "w:footerReference", {"w:type": "default", "r:id": footer_rid})
    node(section, "w:pgSz", {"w:w": 11906, "w:h": 16838})
    node(section, "w:pgMar", {"w:top": 1417,"w:right": 1417,"w:bottom": 1417,"w:left":1417,"w:header":709,"w:footer":709})
    styles = ET.Element(f"{{{NS['w']}}}styles")
    for sid, size, color, bold in (("Normal",22,"202B35",False), ("Title",40,"163955",True),
                                    ("Heading1",28,"163955",True), ("Heading2",24,"202B35",True),
                                    ("Heading3",22,"34444D",True),
                                    ("TableText",19,"202B35",False), ("Code",17,"202B35",False),
                                    ("Caption",19,"657587",False)):
        s = node(styles, "w:style", {"w:type":"paragraph", "w:styleId":sid})
        if sid == "Normal":
            s.set(f"{{{NS['w']}}}default", "1")
        node(s, "w:name", {"w:val":sid})
        pp = node(s,"w:pPr")
        node(pp,"w:spacing",{"w:before":200 if sid.startswith("Heading") else 0,"w:after":120,"w:line":360,"w:lineRule":"auto"})
        if sid in ("Title","Caption"):
            node(pp,"w:jc",{"w:val":"center"})
        if sid.startswith("Heading"):
            node(pp,"w:keepNext")
            outline = {"Heading1": 0, "Heading2": 1, "Heading3": 2}[sid]
            node(pp,"w:outlineLvl",{"w:val":outline})
        rp=node(s,"w:rPr")
        node(rp,"w:rFonts",{"w:ascii":"Calibri","w:hAnsi":"Calibri","w:eastAsia":"黑体" if bold else "宋体"})
        node(rp,"w:sz",{"w:val":size})
        node(rp,"w:szCs",{"w:val":size})
        node(rp,"w:color",{"w:val":color})
        if bold:
            node(rp,"w:b")
    footer = ET.Element(f"{{{NS['w']}}}ftr")
    p = paragraph(footer, "阶段研究材料 · 2026-10-02 · ", "Caption")
    for typ in ("begin", "separate", "end"):
        if typ == "separate":
            node(node(p,"w:r"),"w:instrText",value=" PAGE ")
        node(node(p,"w:r"),"w:fldChar",{"w:fldCharType":typ})
        if typ == "separate":
            run(p,"1")
    rel_root=ET.Element("Relationships",xmlns="http://schemas.openxmlformats.org/package/2006/relationships")
    for rid,kind,dest,external in relationships:
        attrs={"Id":rid,"Type":f"{NS['r']}/{kind}","Target":dest}
        if external:
            attrs["TargetMode"]="External"
        ET.SubElement(rel_root,"Relationship",attrs)
    package=ET.Element("Relationships",xmlns="http://schemas.openxmlformats.org/package/2006/relationships")
    ET.SubElement(package,"Relationship",{"Id":"rId1","Type":f"{NS['r']}/officeDocument","Target":"word/document.xml"})
    ct=ET.Element("Types",xmlns="http://schemas.openxmlformats.org/package/2006/content-types")
    for ext,content in (("rels","application/vnd.openxmlformats-package.relationships+xml"),("xml","application/xml"),("png","image/png")):
        ET.SubElement(ct,"Default",{"Extension":ext,"ContentType":content})
    for part,kind in (("document.xml","document.main"),("styles.xml","styles"),("footer1.xml","footer")):
        ET.SubElement(ct,"Override",{"PartName":f"/word/{part}","ContentType":f"application/vnd.openxmlformats-officedocument.wordprocessingml.{kind}+xml"})
    with ZipFile(target,"w",ZIP_DEFLATED) as archive:
        for name,data in (("[Content_Types].xml",xml(ct)),("_rels/.rels",xml(package)),
                          ("word/document.xml",xml(document)),("word/styles.xml",xml(styles)),
                          ("word/footer1.xml",xml(footer)),("word/_rels/document.xml.rels",xml(rel_root))):
            archive.writestr(name,data)
        for name,data in media:
            archive.writestr(f"word/media/{name}",data)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    (OUT / "evidence").mkdir(parents=True, exist_ok=True)
    verify_and_plot()
    paths = sorted(OUT.glob("*.md"))
    outputs = []
    for folder in ("word", "pdf"):
        (OUT/folder).mkdir(exist_ok=True)
    for path in paths:
        docx, pdf = OUT/"word"/f"{path.stem}.docx", OUT/"pdf"/f"{path.stem}.pdf"
        export_docx([path], docx)
        export_pdf([path], pdf)
        outputs.extend([docx,pdf])
        print(f"exported {path.stem}")
    for suffix, exporter in (("docx",export_docx),("pdf",export_pdf)):
        folder="word" if suffix=="docx" else "pdf"
        target=OUT/folder/f"海底地形重建与高精度定位_阶段研究材料合订本.{suffix}"
        exporter(paths,target,binder=True)
        outputs.append(target)
    for path in outputs:
        if path.suffix=="docx":
            with ZipFile(path) as z:
                assert z.testzip() is None
                for name in z.namelist():
                    if name.endswith(".xml") or name.endswith(".rels"):
                        ET.fromstring(z.read(name))
    manifest={"created_on":"2026-10-02", "git_head_at_build":subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "files":[{"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":digest(p)}
                 for p in paths+outputs+sorted((OUT/"figures").glob("*.png"))
                 +[p for p in sorted((OUT/"evidence").iterdir()) if p.name!="delivery_manifest.json"]],
        "docx_xml_validation":"passed", "metric_checks":"passed"}
    (OUT/"evidence/delivery_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    archive_path=OUT/"阶段研究报告_可编辑版与PDF.zip"
    with ZipFile(archive_path,"w",ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            # The literature archive is retained alongside the reports, rather
            # than redistributing third-party PDFs in the deliverable ZIP.
            if path.is_file() and path!=archive_path and "literature" not in path.relative_to(OUT).parts:
                archive.write(path,path.relative_to(OUT))
    print(f"validated {len(outputs)} documents; saved {archive_path}")


if __name__=="__main__":
    main()
