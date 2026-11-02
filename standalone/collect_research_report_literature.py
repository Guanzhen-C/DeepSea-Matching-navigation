#!/usr/bin/env python3
"""Archive openly accessible primary papers and selected official source files.

Downloads are evidence for research, not experiments or algorithm installs.
No publisher authentication or access restriction is bypassed. Each response
is checked before it is labeled a PDF; failures remain explicit in the log.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/project_reports/2026阶段研究_系统调研详版/literature"
PAPERS = [
    ("side_scan_sparse", "https://arxiv.org/pdf/2304.01854"),
    ("side_scan_dense", "https://arxiv.org/pdf/2312.13802v2"),
    ("neurrss", "https://arxiv.org/pdf/2405.05807"),
    ("sonar_keyframes", "https://arxiv.org/pdf/2106.16032"),
    ("pointnetkl", "https://arxiv.org/pdf/2003.10931"),
    ("bathymetry_loop_retrieval", "https://arxiv.org/pdf/2209.08578"),
    ("fast_livo2", "https://arxiv.org/pdf/2408.14035"),
    ("fast_lio2", "https://arxiv.org/pdf/2107.06829"),
    ("underwater_doppler_calibration", "https://arxiv.org/pdf/1509.02054"),
    ("sonar_depth_uncertainty", "https://arxiv.org/pdf/2206.07810"),
    ("sonar_normals_reconstruction", "https://arxiv.org/pdf/2206.07819"),
    ("sonar_differentiable_rendering", "https://arxiv.org/pdf/2206.07821"),
    ("edsr", "https://arxiv.org/pdf/1707.02921"),
    ("swinir", "https://arxiv.org/pdf/2108.10257"),
    ("liif", "https://arxiv.org/pdf/2012.09161"),
    ("deep_ensembles", "https://arxiv.org/pdf/1612.01474"),
    ("perception_distortion", "https://arxiv.org/pdf/1711.06077"),
    ("bathymetry_transfer_2022", "https://par.nsf.gov/servlets/purl/10400900"),
    ("iho_s44_6_2", "https://iho.int/uploads/user/pubs/standards/s-44/S-44_Edition_6.2.0_adopted.pdf"),
    ("spatial_cross_validation", "https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf"),
    ("geohab_backscatter", "https://geohab.org/wp-content/uploads/2018/09/BSWG-Draft-Report-Geohab2015.pdf"),
]


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "OceanSim-literature-review/2.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        data = response.read(50 * 1024 * 1024)
        return data, response.geturl(), response.headers.get("Content-Type", "")


def paper(item):
    key, url = item
    result = {"key": key, "requested_url": url, "retrieved_on": "2026-10-02"}
    path = OUT / f"{key}.pdf"
    try:
        if path.exists() and path.read_bytes().startswith(b"%PDF"):
            data = path.read_bytes()
            final, mime = url, "application/pdf (cached)"
        else:
            data, final, mime = get(url)
            if not data.startswith(b"%PDF"):
                raise ValueError(f"response is not PDF: {mime}")
            path.write_bytes(data)
        text_path = path.with_suffix(".txt")
        subprocess.run(["pdftotext", "-layout", str(path), str(text_path)], check=True,
                       capture_output=True)
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        result.update({"status": "downloaded_and_text_extracted", "final_url": final,
                       "content_type": mime, "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest(),
                       "pages": int(next(line.split(":",1)[1] for line in info.splitlines() if line.startswith("Pages:"))),
                       "pdf": path.name, "text": text_path.name})
    except Exception as exc:
        result.update({"status": "not_retrieved", "reason": str(exc)})
    print(f"{key}: {result['status']}", flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(paper, PAPERS))
    repositories = []
    for repo, files in [
        ("dwcaress/MB-System", ["src/mbtrnav/terrain-nav/TerrainNav.cpp", "src/mbtrnav/terrain-nav/TNavPointMassFilter.cpp", "src/mbtrnav/terrain-nav/TNavParticleFilter.cpp", "LICENSE"]),
        ("halajun/acoustic_slam", ["README.md", "LICENSE"]),
        ("halajun/diasss", ["README.md", "LICENSE"]),
        ("ignaciotb/UWExploration", ["README.md", "LICENSE"]),
        ("big-data-lab-umbc/bathymetry_super_resolution", ["README.md", "LICENSE"]),
    ]:
        record = {"repository": repo, "checked_on": "2026-10-02", "files": []}
        try:
            meta = json.loads(get(f"https://api.github.com/repos/{repo}")[0])
            branch = meta["default_branch"]
            commit = json.loads(get(f"https://api.github.com/repos/{repo}/commits/{branch}")[0])["sha"]
            record.update({"commit": commit, "default_branch": branch})
            folder = OUT / "official_code" / repo.replace("/", "_")
            folder.mkdir(parents=True, exist_ok=True)
            for file in files:
                item = {"path": file, "url": f"https://raw.githubusercontent.com/{repo}/{commit}/{file}"}
                try:
                    data = get(item["url"])[0]
                    destination = folder / file.replace("/", "__")
                    destination.write_bytes(data)
                    item.update({"status": "retrieved", "sha256": hashlib.sha256(data).hexdigest(), "saved": str(destination.relative_to(OUT))})
                except Exception as exc:
                    item.update({"status": "not_retrieved", "reason": str(exc)})
                record["files"].append(item)
        except Exception as exc:
            record["metadata_error"] = str(exc)
        repositories.append(record)
        print(f"repository {repo}: {record.get('commit', 'unavailable')}", flush=True)
    manifest = {"retrieval_date": "2026-10-02", "papers": records, "repositories": repositories,
                "note": "Retrieval is not a claim of line-by-line full-paper review or reproduction. Study level is documented in the source ledger."}
    (OUT / "retrieval_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"retrieved {sum(r['status']=='downloaded_and_text_extracted' for r in records)}/{len(records)} primary PDFs")


if __name__ == "__main__":
    main()
