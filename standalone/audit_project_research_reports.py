#!/usr/bin/env python3
"""Check document packages, local references, source counts and archived hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from fontTools.ttLib import TTFont
from build_project_research_reports import pdf_font_safe


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=root/"docs/project_reports/2026阶段研究_系统调研详版")
    args=parser.parse_args()
    out=args.output_dir.resolve()
    reports=sorted(out.glob("*.md"))
    contents={p.name:p.read_text(encoding="utf-8") for p in reports}
    pdfs=[]
    for path in sorted((out/"pdf").glob("*.pdf")):
        info=subprocess.check_output(["pdfinfo",str(path)],text=True)
        pages=int(next(s.split(":",1)[1] for s in info.splitlines() if s.startswith("Pages:")))
        extracted=subprocess.check_output(["pdftotext",str(path),"-"],text=True)
        assert pages>0 and "海底" in extracted
        pdfs.append({"file":path.name,"pages":pages,"text_characters":len(extracted)})
    for path in sorted((out/"word").glob("*.docx")):
        with ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith((".xml",".rels")):
                    ET.fromstring(z.read(name))
    for name,content in contents.items():
        for label,target in re.findall(r"!?\[([^\]]+)\]\(([^)]+)\)",content):
            if not target.startswith(("http://","https://")):
                assert (out/target).exists(),(name,label,target)
    ledger=contents["00系统调研方法与文献证据矩阵.md"]
    ids=re.findall(r"^\| ([NROVD]\d{2})\b",ledger,re.M)
    assert len(ids)==len(set(ids))==62,ids
    manifest=json.loads((out/"literature/retrieval_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["papers"]:
        if entry["status"]=="downloaded_and_text_extracted":
            assert sha(out/"literature"/entry["pdf"])==entry["sha256"]
    cmap=TTFont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",fontNumber=0).getBestCmap()
    alltext="\n".join(contents.values())
    missing=sorted({ch for ch in pdf_font_safe(alltext) if ord(ch)>=32 and ord(ch) not in cmap})
    result={"checked_on":"2026-10-02","markdown_files":len(reports),"source_entries":len(ids),
            "primary_pdfs_retrieved":sum(p["status"]=="downloaded_and_text_extracted" for p in manifest["papers"]),
            "official_repositories":len(manifest["repositories"]),"pdfs":pdfs,
            "chinese_characters":len(re.findall(r"[\u4e00-\u9fff]",alltext)),
            "markdown_characters":len(alltext),"font_missing_characters":missing,
            "local_links":"passed","docx_xml":"passed","literature_hashes":"passed"}
    target=out/"evidence/report_audit.json"
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
