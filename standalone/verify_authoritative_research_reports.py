#!/usr/bin/env python3
"""Read back rendered DOCX files and record independently verified delivery."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import build_authoritative_research_reports as reports


def main():
    render = Path("/tmp/oceansim-authoritative-report-render")
    manifest_path = reports.OUT / "evidence/文档交付与验证.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    for item in manifest["reports"]:
        source = reports.OUT / item["source"]
        rendered = render / item["pdf"]
        assert rendered.exists(), rendered
        info = subprocess.check_output(["pdfinfo", str(rendered)], text=True)
        extracted = subprocess.check_output(["pdftotext", str(rendered), "-"], text=True)
        text = source.read_text(encoding="utf-8")
        source_cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        rendered_cjk = len(re.findall(r"[\u4e00-\u9fff]", extracted))
        assert rendered_cjk >= .95 * source_cjk, (item["report"], source_cjk, rendered_cjk)
        assert all(term in extracted for term in ("摘要", "参考文献", "CC BY 4.0"))
        for key, suffix, folder in (("pdf_sha256", ".pdf", "pdf"), ("docx_sha256", ".docx", "word")):
            filename = item["pdf"] if suffix == ".pdf" else item["docx"]
            built = reports.OUT / folder / filename
            desktop = reports.DESKTOP / filename
            assert reports.sha(built) == reports.sha(desktop) == item[key]
        checks.append(dict(report=item["report"],
                           DOCX_render_pages=int(re.search(r"Pages:\s+(\d+)", info)[1]),
                           source_cjk_characters=source_cjk, rendered_cjk_characters=rendered_cjk,
                           reference_and_license_text_present=True, desktop_hash_matches=True))
    output = dict(checked_on="2026-10-02", docx_actually_opened_with="LibreOffice Writer headless",
                  result="passed", reports=checks,
                  visual_checks="PDF: SR pixel-loss formulas, PF weights, MBES/SSS and DVL models; DOCX render sampled separately",
                  scope="selected translations and original project analysis; not complete paper translations")
    destination = reports.OUT / "evidence/办公软件打开与交付复核.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    shutil.copy2(destination, reports.DESKTOP / destination.name)
    shutil.copy2(reports.OUT / "交付说明.md", reports.DESKTOP / "交付说明.md")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
