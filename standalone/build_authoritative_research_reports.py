#!/usr/bin/env python3
"""Build four separate literature-based reports without editing SLAM code."""
from __future__ import annotations

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
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import build_project_research_reports as base

ROOT = base.ROOT
OUT = ROOT / "docs/project_reports/2026阶段研究_权威文献译读报告"
PARTS = OUT / "parts"
TRANSLATIONS = ROOT / "docs/literature_translations/20261002_top_journals"
V3 = ROOT / "docs/project_reports/2026阶段研究_研究报告详述版V3"
DESKTOP = Path("/home/cgz/Desktop/四份研究报告_权威文献译读详版")
RUN = ROOT / "demo/large_scale_slam/p0_shadow_confirm_full_route_headless_20260917"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def translated_body(name, selected=None):
    source = (TRANSLATIONS / name).read_text(encoding="utf-8")
    source = source[source.index("## 摘要"):]
    source = source.split("## 译者注", 1)[0].strip()
    if selected:
        chunks = re.split(r"(?=^## )", source, flags=re.M)
        source = "\n\n".join(chunk.strip() for chunk in chunks
                               if any(chunk.startswith("## " + s) for s in selected))
    source = re.sub(r"^## (.+)$", r"### 原文 \1", source, flags=re.M)
    source = re.sub(r"^####+ ", "### ", source, flags=re.M)
    return source


def expansion(name):
    source = (PARTS / name).read_text(encoding="utf-8")
    return re.sub(r"^## (.+)$", r"### \1", source, flags=re.M)


def equations():
    formulas = {
        "terrain_information": [
            r"r_i(q)=[T(q)p_i]_z-H([T(q)p_i]_{xy})",
            r"J_{i,xy}=-\nabla H_i^T,\qquad\mathcal{I}=J^T R^{-1}J",
        ],
        "sr_pixel": [
            r"\mathcal{L}_{L1}=\frac{1}{N_y}\sum_p|y_p-\hat y_p|\quad (19)",
            r"\mathcal{L}_{L2}=\frac{1}{N_y}\sum_p|y_p-\hat y_p|^2\quad (20)",
            r"\mathcal{L}_{\mathrm{Charbonnier}}=\frac{1}{N_y}\sum_p\sqrt{|y_p-\hat y_p|^2+\epsilon^2}\quad (21)",
        ],
        "sr_observation": [
            r"z=M\,\mathcal{D}(h)+\eta",
            r"\hat h=h_{\mathrm{base}}+f_\theta(z,M,\mathrm{quality})",
        ],
        "sr_loss": [
            r"\mathcal{L}=\lambda_h\|M_h(\hat h-h)\|_1",
            r"+\lambda_g\|M_g(\nabla\hat h-\nabla h)\|_1+\lambda_c\|M[\mathcal{D}(\hat h)-z]\|_1",
        ],
        "sr_uncertainty": [
            r"\mathcal{L}_{\mathrm{NLL}}=\frac{1}{2}\sum_i\left[\frac{(h_i-\mu_i)^2}{\sigma_i^2}+\ln\sigma_i^2\right]+\mathrm{const}",
            r"\sigma_{r,i}^2\approx\sigma_{\mathrm{sonar},i}^2+\sigma_{\mathrm{map},i}^2+\nabla H_i^T\Sigma_{xy,i}\nabla H_i",
        ],
        "pf_update": [
            r"p(q_k|z_{1:k})\propto p(z_k|q_k)\int p(q_k|q_{k-1},u_k)p(q_{k-1}|z_{1:k-1})\,dq_{k-1}",
            r"w_k^{(n)}\propto w_{k-1}^{(n)}\frac{p(z_k|q_k^{(n)})p(q_k^{(n)}|q_{k-1}^{(n)},u_k)}{g(q_k^{(n)}|q_{k-1}^{(n)},u_k,z_k)}",
            r"P(C_j|z_{1:k},H_1)=\sum_{n\in C_j}\bar w_k^{(n)}",
        ],
        "pf_likelihood": [
            r"p(r|q)=(1-\epsilon)\mathcal{N}(r;0,C_q)+\epsilon\,p_{\mathrm{out}}(r)",
            r"-\ln\mathcal{N}(r;0,C_q)=\frac{1}{2}r^TC_q^{-1}r+\frac{1}{2}\ln|C_q|+\mathrm{const}",
        ],
        "null_hypothesis": [
            r"P(C_j|Z)\propto p(Z|C_j)P(C_j),\qquad P(H_0|Z)\propto p(Z|H_0)P(H_0)",
            r"P(H_0|Z)+\sum_jP(C_j|Z)=1",
        ],
        "nuisance_information": [
            r"r\approx r_0+J_q\delta q+\mathbf{1}\delta b",
            r"\mathcal{I}_{q,\mathrm{eff}}=\mathcal{I}_{qq}-\mathcal{I}_{qb}\mathcal{I}_{bb}^{-1}\mathcal{I}_{bq}",
        ],
        "map_odom": [
            r"T_{\mathrm{map},b}(t)=T_{\mathrm{map},b}(t_r)[T_{\mathrm{odom},b}(t_r)]^{-1}T_{\mathrm{odom},b}(t)",
            r"T_{\mathrm{map},\mathrm{odom}}=T_{\mathrm{map},b}(t_r)[T_{\mathrm{odom},b}(t_r)]^{-1}",
        ],
        "sonar_residuals": [
            r"r_{\mathrm{MBES}}=n^T(Rp+t-q)",
            r"r_{\mathrm{SSS}}=[\|p_s\|-r,\ e_x^Tp_s]^T,\quad\Sigma_{\mathrm{SSS}}=\mathrm{diag}(\sigma_r^2,r^2\alpha^2)",
        ],
        "dvl_model": [
            r"v_D=R_{bd}^T\left(R_{wb}^T v_w+\omega_b\times l_{bd}\right)+\eta_D",
        ],
    }
    for name, lines in formulas.items():
        fig = plt.figure(figsize=(10.2, .54 * len(lines) + .24), dpi=220)
        for index, line in enumerate(lines):
            fontsize = 14 if len(line) > 130 else 16
            fig.text(.5, 1-(index+.65)/(len(lines)+.32), "$"+line+"$",
                     ha="center", va="center", fontsize=fontsize)
        fig.savefig(OUT / "assets" / (name + ".png"), bbox_inches="tight",
                    pad_inches=.13, facecolor="white")
        plt.close(fig)


def metrics():
    record = json.loads(RUN.with_suffix(".json").read_text(encoding="utf-8"))
    csv = RUN.parent / (RUN.name + "_position_errors.csv")
    values = np.genfromtxt(csv, delimiter=",", names=True)
    result = {}
    rows = ["| 输出 | 三维RMSE 米 | 中位数 米 | P95 米 | 最大误差 米 | 终点误差 米 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for label, field, key in [
        ("DR", "dead_reckoning_3d_m", "dead_reckoning_imu_dvl_depth"),
        ("Local", "local_acoustic_esikf_3d_m", "local_acoustic_esikf"),
        ("在线Global", "online_global_slam_3d_m", "online_global_slam"),
    ]:
        data = values[field]
        assert np.all(np.isfinite(data))
        item = dict(rmse_3d_m=float(np.sqrt(np.mean(data**2))),
                    median_3d_m=float(np.median(data)), p95_3d_m=float(np.percentile(data, 95)),
                    max_3d_m=float(np.max(data)), final_3d_m=float(data[-1]))
        for field_name, number in item.items():
            assert np.isclose(number, record["position_error_statistics"][key][field_name],
                              rtol=1e-10, atol=1e-7), (key, field_name)
        result[key] = item
        rows.append("| " + label + " | " + " | ".join(f"{v:.3f}" for v in item.values()) + " |")
    assert len(values) == 243569
    assert np.allclose(np.diff(values["timestamp_s"]), .2, atol=1e-6)
    evidence = dict(verified_on="2026-10-02", samples=len(values),
                    simulated_distance_m=record["survey_travelled_distance_m"],
                    statistics=result, source_json=str(RUN.with_suffix(".json").relative_to(ROOT)),
                    source_csv=str(csv.relative_to(ROOT)), csv_sha256=sha(csv),
                    json_sha256=sha(RUN.with_suffix(".json")), source="archived simulation, not sea trial")
    (OUT / "evidence" / "本次归档指标复核.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return "\n".join(rows) + "\n\n" + (
        "表中统计来自同一归档任务的同步位置误差，不是所译论文实验。"
        "完整误差曲线如下，包含失败峰值而不是只画正常时段。\n\n"
        "![项目归档实验 73公里在线位置误差与对照](assets/73km在线定位误差与对照.png)")


def baseline():
    path = V3 / "evidence/sr_interpolation_baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert not data["trained_model"] and not data["independent_survey"]
    assert len(data["selected_patches"]) == 12
    source = ROOT / data["source"]
    assert sha(source) == data["source_sha256"]
    rows = ["| 倍率与方法 | 高程RMSE 米 | MAE 米 | 坡度RMSE 度 | 回退一致性RMSE 米 |",
            "| --- | --- | --- | --- | --- |"]
    for name, item in data["pooled_metrics"].items():
        label = name.replace("nearest", "最近邻").replace("bilinear", "双线性").replace("bicubic", "双三次")
        vals = [item[k] for k in ("rmse_m", "mae_m", "slope_rmse_deg", "downsampling_consistency_rmse_m")]
        rows.append("| " + label + " | " + " | ".join(f"{v:.5f}" for v in vals) + " |")
    shutil.copy2(path, OUT / "evidence" / "已有插值基线_非学习模型.json")
    return "\n".join(rows) + "\n\n![受控插值实验 非学习型超分结果](assets/地形超分插值基线.png)"


def docx(source, target):
    base.export_docx([source], target)
    with ZipFile(target) as archive:
        payload = {name: archive.read(name) for name in archive.namelist()}
    styles = ET.fromstring(payload["word/styles.xml"])
    for font in styles.findall(".//w:rFonts", base.NS):
        font.set(f"{{{base.NS['w']}}}eastAsia", "文泉驿正黑")
        font.set(f"{{{base.NS['w']}}}ascii", "DejaVu Sans")
        font.set(f"{{{base.NS['w']}}}hAnsi", "DejaVu Sans")
    payload["word/styles.xml"] = base.xml(styles)
    payload["word/footer1.xml"] = payload["word/footer1.xml"].replace(
        "阶段研究材料".encode(), "权威文献译读研究报告".encode())
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)


def main():
    for folder in ("assets", "evidence", "word", "pdf"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    DESKTOP.mkdir(parents=True, exist_ok=True)
    for asset in (TRANSLATIONS / "assets").glob("*.png"):
        shutil.copy2(asset, OUT / "assets" / asset.name)
    for asset in (V3 / "figures").glob("*.png"):
        shutil.copy2(asset, OUT / "assets" / asset.name)
    equations()
    substitutions = {
        "SONAR_MAIN": translated_body("03_JOE2025_声呐深度学习鲁棒性综述_中文节译.md"),
        "SONAR_EXPANSION": expansion("sonar_expansion.md"),
        "SR_MAIN": translated_body("01_TPAMI2023_超分辨率综述_中文节译.md"),
        "SR_EXPANSION": expansion("sr_expansion.md"),
        "DENSE_MAIN": translated_body("02_JOE2025_稠密侧扫声呐SLAM_中文节译.md"),
        "SONAR_LOCALIZATION": translated_body(
            "03_JOE2025_声呐深度学习鲁棒性综述_中文节译.md", ["III-C", "IV-F"]),
        "METRICS": metrics(), "SR_BASELINE": baseline(),
    }
    old_inline = base.inline_pdf
    pdfmetrics.registerFont(TTFont("ReportLatin", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    def safe_inline(value):
        output = old_inline(value)
        return re.sub(r"w̄|h̄|ŷ|[ᵇᵒᶜ]", lambda m: '<font name="ReportLatin">'+m[0]+'</font>', output)
    base.inline_pdf = safe_inline
    outputs = []
    for template in sorted(PARTS.glob("*月报告.md")):
        text = template.read_text(encoding="utf-8")
        for key, replacement in substitutions.items():
            text = text.replace("{{"+key+"}}", replacement)
        assert not re.search(r"\{\{[A-Z_]+\}\}", text)
        assert not re.search(r"\b(?:TODO|TBD)\b", text)
        assert "非官方" in text and "CC BY 4.0" in text
        heads = re.findall(r"^## (.+)$", text, re.M)
        navigation = "## 阅读导航\n\n" + "；".join(heads) + "。\n\n"
        marker = text.index("## 一 ")
        text = text[:marker]+navigation+text[marker:]
        title = text.splitlines()[0].removeprefix("# ")
        source = OUT / (template.stem[:3]+title+".md")
        source.write_text(text, encoding="utf-8")
        pdf = OUT / "pdf" / (source.stem+".pdf")
        word = OUT / "word" / (source.stem+".docx")
        base.export_pdf([source], pdf)
        docx(source, word)
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        pages = int(re.search(r"Pages:\s+(\d+)", info)[1])
        extracted = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        source_cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        extracted_cjk = len(re.findall(r"[\u4e00-\u9fff]", extracted))
        assert "参考文献" in extracted and extracted_cjk >= .95 * source_cjk
        with ZipFile(word) as archive:
            assert archive.testzip() is None
            for name in archive.namelist():
                if name.endswith((".xml", ".rels")):
                    ET.fromstring(archive.read(name))
        print(f"{source.stem}: {len(text)} characters, {pages} PDF pages", flush=True)
        for file in (pdf, word):
            destination = DESKTOP / file.name
            shutil.copy2(file, destination)
            assert sha(destination) == sha(file)
        outputs.append(dict(report=title, source=str(source.relative_to(OUT)),
                            characters=len(text), pdf_pages=pages,
                            pdf=pdf.name, docx=word.name,
                            pdf_sha256=sha(pdf), docx_sha256=sha(word)))
    assert len(outputs) == 4
    manifest = dict(created_on="2026-10-02", separate_reports=True, binder=False,
                    reports=outputs, total_pdf_pages=sum(item["pdf_pages"] for item in outputs),
                    markdown_characters=sum(item["characters"] for item in outputs),
                    interpretation="selected licensed translations plus separately labeled original project analysis",
                    experiments="archived simulation and interpolation only; no new trained model or sea trial claimed",
                    pdf_text_and_docx_xml_validation="passed", desktop_copy_hash_validation="passed")
    evidence = OUT / "evidence/文档交付与验证.json"
    evidence.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    shutil.copy2(evidence, DESKTOP / evidence.name)
    originals = DESKTOP / "英文原文"
    originals.mkdir(exist_ok=True)
    for original in (TRANSLATIONS / "originals").glob("*.pdf"):
        shutil.copy2(original, originals / original.name)
    for evidence in (OUT / "evidence").glob("*.json"):
        shutil.copy2(evidence, DESKTOP / evidence.name)
    shutil.copy2(OUT / "交付说明.md", DESKTOP / "交付说明.md")
    print(json.dumps({"reports":len(outputs),"PDF_pages":manifest["total_pdf_pages"],
                      "desktop":str(DESKTOP)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
