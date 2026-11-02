#!/usr/bin/env python3
"""Collect additional open primary evidence without altering V1/V2 artifacts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/project_reports/2026阶段研究_研究报告详述版V3/evidence/new_literature"
SOURCES = [
    ("sss_geometry_matching_2025", "pdf", "https://arxiv.org/pdf/2509.11255v3"),
    ("sss_two_view_geometry_2026", "pdf", "https://arxiv.org/pdf/2609.25271v1"),
    ("sss_opti_acoustic_2026", "pdf", "https://arxiv.org/pdf/2608.23479v1"),
    ("sss_landmark_navigation_2025", "pdf", "https://arxiv.org/pdf/2503.07900v1"),
    ("deep_ins_dvl_correlation_2025", "pdf", "https://arxiv.org/pdf/2503.21727"),
    ("partial_dvl_2024", "pdf", "https://arxiv.org/pdf/2404.13742"),
    ("dvl_outage_transformer_2025", "pdf", "https://arxiv.org/pdf/2504.07697"),
    ("insso3d_2026", "pdf", "https://arxiv.org/pdf/2601.05805v1"),
    ("barkby_gp_2011", "pdf", "https://www-personal.acfr.usyd.edu.au/sbarkby/2011_iros_bpslam.pdf"),
    ("fox_kld_2001", "pdf", "https://papers.neurips.cc/paper/1998-kld-sampling-adaptive-particle-filters.pdf"),
    ("max_mixture_2012", "pdf", "https://april.eecs.umich.edu/media/pdfs/olson2012rss.pdf"),
    ("eskf_rotation_2017", "pdf", "https://arxiv.org/pdf/1711.02508"),
    ("teaser_2020", "pdf", "https://arxiv.org/pdf/2001.07715"),
    ("lightglue_2023", "pdf", "https://arxiv.org/pdf/2306.13643"),
    ("loftr_2021", "pdf", "https://arxiv.org/pdf/2104.00680"),
    ("global_utan_2026", "html", "https://pure.kaist.ac.kr/en/publications/terrain-based-global-tracking-without-prior-pose-information-usin/"),
    ("adaptive_sr_2025", "html", "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024EA003610"),
    ("plos_bathymetry_sr_2020", "html", "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0235487"),
    ("probabilistic_multibeam_slam_2016", "html", "https://www.mdpi.com/1424-8220/16/4/560"),
    ("ctasr_2026", "html", "https://www.sciencedirect.com/science/article/pii/S0924271626003291"),
    ("terrain_cross_attention_2026", "html", "https://www.tandfonline.com/doi/full/10.1080/10095020.2026.2683151"),
    ("trn_mbari", "html", "https://www.mbari.org/project/terrain-relative-navigation/"),
    ("adaptive_sr_code", "html", "https://github.com/matsuokadaisuke/adaptive-sr"),
    ("mid_ocean_ridge_transfer_2024", "html", "https://portal.research.lu.se/en/publications/transfer-learning-reconstructs-submarine-topography-for-global-mi/"),
    ("probabilistic_multibeam_slam_pmc", "html", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4851074/"),
]


def collect(item):
    key, kind, url = item
    record = {"key": key, "url": url, "accessed_on": "2026-10-02", "requested_type": kind}
    path = OUT / f"{key}.{kind}"
    try:
        if path.exists():
            data = path.read_bytes()
            final_url = url
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "OceanSim-research-review/3.0"})
            with urllib.request.urlopen(req, timeout=25) as response:
                limit = 96 * 1024 * 1024
                data = response.read(limit + 1)
                final_url = response.geturl()
                if len(data) > limit:
                    raise ValueError("Response exceeds the 96 MiB archive limit; no truncated PDF is saved")
            if kind == "pdf" and not data.startswith(b"%PDF"):
                raise ValueError("Response is not PDF")
            if kind == "html" and b"<" not in data[:2048]:
                raise ValueError("Response is not HTML")
            path.write_bytes(data)
        record.update(final_url=final_url, bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest(), saved=path.name)
        if kind == "pdf":
            subprocess.run(["pdftotext", "-layout", str(path), str(path.with_suffix(".txt"))],
                           check=True, capture_output=True)
            info = subprocess.check_output(["pdfinfo", str(path)], text=True)
            record["pages"] = int(next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:")))
        else:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(data, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            path.with_suffix(".txt").write_text(soup.get_text("\n", strip=True), encoding="utf-8")
        record["status"] = "retrieved"
    except Exception as exc:
        record.update(status="not_retrieved", reason=str(exc))
        if kind == "pdf" and path.exists():
            invalid = path.with_suffix(".invalid_download")
            path.replace(invalid)
            record["saved"] = invalid.name
    print(f"{key}: {record['status']}", flush=True)
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(collect, SOURCES))
    (OUT / "retrieval_manifest.json").write_text(json.dumps({
        "created_on": "2026-10-02", "sources": records,
        "interpretation": "Retrieval does not mean reproduction or full-paper review. Failed publisher access is not bypassed. Web tool access and local archive access can differ. Reading scope is documented separately."
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
