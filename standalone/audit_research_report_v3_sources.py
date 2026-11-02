#!/usr/bin/env python3
"""Record source identity, actual reading scope and public DOI metadata."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re
from urllib.parse import quote
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/project_reports/2026阶段研究_研究报告详述版V3"
OLD = ROOT / "docs/project_reports/2026阶段研究_系统调研详版"

# Reading scope is separate from download status and reproduction. These
# entries reflect reviewed source passages, not an automated full-review claim.
SCOPES = {
    "M": ["摘要与可见章节", "指南相关章节", "公开正文方法与实验", "正文方法与条件性证书",
          "官方摘要与仓库说明", "官方摘要", "公开摘要", "机构摘要", "正文方法与实验",
          "机构元数据与选定代码", "正文方法与实验", "v2正文方法与实验及正式书目",
          "正文方法与实验", "正文模型与实验", "公开正文方法与实验", "正文及表II",
          "官方数据发布与本地地图", "测量规范相关章节"],
    "S": ["公开正文方法与实验", "正文及表II", "出版摘要与元数据", "出版摘要与元数据",
          "期刊摘要与元数据", "网页可见方法章节与官方代码说明", "出版页可见方法与实验片段",
          "公开章节片段与元数据", "公开方法章节", "正文模型与实验", "公开方法章节",
          "官方数据发布与本地栅格", "官方数据发布与本地栅格", "官方数据发布与本地栅格",
          "官方产品说明", "机构元数据与摘要", "正文模型与实验", "正文与官方仓库说明",
          "正文理论与实验", "正文方法与实验", "机构摘要与正式书目"],
    "G": ["机构项目说明", "官方说明与选定源文件", "官方摘要", "公开摘要", "机构摘要",
          "正文方法与条件性证书", "官方摘要与仓库说明", "正文方法与实验", "官方技术说明",
          "指南相关章节", "正文方法与实测条件", "原始论文方法", "机构元数据与选定代码",
          "正文方法与实验", "正文方法与实验"],
    "O": ["正文模型与实验", "正文模型与实验", "正文方法与实验", "v2正文方法与实验及正式书目",
          "正文方法与实验", "正文方法与实验", "正文方法与实验", "正文方法与实验",
          "正文方法与退化分析", "官方维护仓库说明", "正文模型与真实试验条件",
          "正文几何推导与Monte Carlo条件", "正文方法与实验", "旋转与误差状态相关章节",
          "正文可观性与试验", "正文方法与实验", "正文方法与实验", "正文方法与实验",
          "公开方法章节", "正文模型与实测条件", "正文模型与掉锁条件",
          "正文模型、两段400秒试验与协方差讨论"],
}


def canonical(line, urls):
    doi = re.search(r"10\.\d{4,9}/[^\s、，；;。)]+", line)
    if doi:
        return "doi:" + doi[0].rstrip(".").lower()
    for url in urls:
        arxiv = re.search(r"arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})", url)
        if arxiv:
            return "arxiv:" + arxiv[1]
    return urls[0].rstrip("/")


def metadata(key):
    doi = key.removeprefix("doi:")
    rec = {"doi": doi, "checked_on": "2026-10-02"}
    for registry, url in (("Crossref", "https://api.crossref.org/works/"+quote(doi, safe="")),
                          ("DataCite", "https://api.datacite.org/dois/"+quote(doi, safe=""))):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OceanSim-research-evidence/3.0"})
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.load(response)
            if registry == "Crossref":
                message = data["message"]
                rec.update(status="metadata_retrieved", registry=registry,
                           title=message.get("title"), authors=message.get("author"),
                           published=message.get("published"), container=message.get("container-title"),
                           source_url=url)
            else:
                attrs = data["data"]["attributes"]
                rec.update(status="metadata_retrieved", registry=registry,
                           title=attrs.get("titles"), authors=attrs.get("creators"),
                           published=attrs.get("publicationYear"), source_url=url)
            return rec
        except Exception as exc:
            rec.setdefault("access_failures", []).append({"registry": registry, "reason": str(exc)})
    rec["status"] = "metadata_not_retrieved"
    return rec


def main():
    manifests = [json.loads((OUT/"evidence/new_literature/retrieval_manifest.json").read_text()),
                 json.loads((OLD/"literature/retrieval_manifest.json").read_text())]
    archived = []
    for data in manifests:
        # The previous edition uses a different record container.
        for item in data.get("sources", data.get("papers", data.get("records", []))):
            archived.append(item)
    rows = []
    for path in sorted(OUT.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        refs = text.split("## 参考文献与资料", 1)[1]
        for line in refs.splitlines():
            found = re.match(r"([MSGO])(\d\d) (.+)", line)
            if not found:
                continue
            family, index, description = found.groups()
            urls = re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", description)
            key = canonical(description, urls)
            archive_hits = []
            for item in archived:
                source_url = item.get("url", item.get("requested_url", item.get("source_url", "")))
                arxiv_key = key.removeprefix("arxiv:")
                same_arxiv = key.startswith("arxiv:") and arxiv_key in source_url
                if same_arxiv or source_url in urls:
                    archive_hits.append({"url": source_url, "status": item.get("status"),
                                         "saved": item.get("saved", item.get("pdf", item.get("path")))})
            rows.append({"报告": path.stem, "编号": family+index, "来源标识": key,
                         "书目": description, "核读范围": SCOPES[family][int(index)-1],
                         "公开入口": " | ".join(urls),
                         "本地归档记录": json.dumps(archive_hits, ensure_ascii=False),
                         "复现状态": "未在本次任务中复现该论文；项目既有实验另列证据"})
    target = OUT/"evidence/文献来源与核读范围.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = sorted({r["来源标识"] for r in rows})
    previous = OUT/"evidence/source_identity_and_doi_validation.json"
    cached = {}
    if previous.exists():
        cached = {r["doi"]: r for r in json.loads(previous.read_text())["doi_metadata"]
                  if r["status"] == "metadata_retrieved" and r["checked_on"] == "2026-10-02"}
    def from_cache_or_registry(key):
        return cached.get(key.removeprefix("doi:")) or metadata(key)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(from_cache_or_registry, [key for key in keys if key.startswith("doi:")]))
    result = {"created_on": "2026-10-02", "reference_entries_including_repetition": len(rows),
              "distinct_primary_source_identities": len(keys), "canonical_source_keys": keys,
              "doi_metadata": records,
              "interpretation": "Source identities include papers, data, standards and official software; they are not all independent peer-reviewed papers. Reading scope and retrieval status are separate. Metadata retrieval is not a full-paper review."}
    (OUT/"evidence/source_identity_and_doi_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Recorded {len(rows)} reference entries, {len(keys)} distinct source identities, "
          f"{sum(r['status']=='metadata_retrieved' for r in records)}/{len(records)} DOI metadata records.")
    for rec in records:
        print(json.dumps({k: rec.get(k) for k in ("doi", "status", "title")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
