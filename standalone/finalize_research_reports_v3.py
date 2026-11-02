#!/usr/bin/env python3
"""Check independent Word rendering and copy only owned V3 deliverables."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from build_research_reports_v3 import OUT, REPORTS
import build_project_research_reports as base

DESKTOP = Path("/home/cgz/Desktop/海底地形研究_四份独立研究报告V3")


def main():
    checks = []
    for stem in REPORTS:
        source = OUT/"word"/f"{stem}_V3研究详述版.docx"
        rendered = OUT/"evidence/render_checks"/f"{stem}_V3研究详述版.pdf"
        assert rendered.is_file(), rendered
        assert rendered.stat().st_mtime > source.stat().st_mtime
        content = subprocess.check_output(["pdftotext", "-layout", str(rendered), "-"], text=True)
        assert "目录" in content and "研究报告" in content and "参考文献与资料" in content
        with ZipFile(source) as z:
            root = ET.fromstring(z.read("word/document.xml"))
            assert any("TOC" in (e.text or "") for e in root.findall(".//w:instrText", base.NS))
        info = subprocess.check_output(["pdfinfo", str(rendered)], text=True)
        pages = int(next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:")))
        checks.append({"source_docx": source.name, "source_sha256": base.digest(source),
                       "libreoffice_rendered_pdf": rendered.name, "pages": pages,
                       "rendered_pdf_sha256": base.digest(rendered), "validation": "passed"})
    literature = json.loads((OUT/"evidence/new_literature/retrieval_manifest.json").read_text())
    valid_pdfs = [s for s in literature["sources"] if s["status"] == "retrieved" and s["requested_type"] == "pdf"]
    for entry in valid_pdfs:
        path = OUT/"evidence/new_literature"/entry["saved"]
        subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True)
        assert base.digest(path) == entry["sha256"]
    with (OUT/"evidence/文献来源与核读范围.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 76 and len({r["来源标识"] for r in rows}) == 62
    DESKTOP.mkdir(parents=True, exist_ok=True)
    files = []
    for folder in ("word", "pdf"):
        for path in sorted((OUT/folder).iterdir()):
            assert path.suffix in (".docx", ".pdf")
            files.append(path)
    assert len(files) == 8
    files.extend([OUT/"阅读说明.txt", OUT/"evidence/文献来源与核读范围.csv"])
    deliveries = []
    for source in files:
        dest = DESKTOP/source.name
        if dest.exists():
            # Avoid overwriting a file externally edited after handoff.
            assert base.digest(dest) == base.digest(source), f"Existing desktop file differs: {dest}"
        else:
            shutil.copy2(source, dest)
        assert base.digest(dest) == base.digest(source)
        deliveries.append({"file": dest.name, "bytes": dest.stat().st_size,
                           "sha256": base.digest(dest)})
    final = {"checked_on": "2026-10-02", "word_rendering": checks,
             "additional_valid_pdf_archives": len(valid_pdfs),
             "references": 76, "distinct_source_identifiers": 62,
             "desktop_folder": str(DESKTOP), "copied_files": deliveries,
             "binder_created": False, "old_editions_preserved": True,
             "wechat_delivery_of_v3": "not_sent_in_this_revision"}
    (OUT/"evidence/final_delivery_validation.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"desktop_folder": str(DESKTOP), "files": len(deliveries),
                      "valid_new_pdf_archives": len(valid_pdfs), "word_rendering": "passed"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
