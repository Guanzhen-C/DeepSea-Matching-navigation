"""Verify the report delivery from exported full data and rendered documents."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read_csv(path):
    opener = gzip.open if path.suffix == '.gz' else Path.open
    with opener(path, mode='rt' if path.suffix == '.gz' else 'r', newline='') as stream:
        reader = csv.reader(stream); names = next(reader)
        values = np.array([[float(v) for v in row] for row in reader])
    if not np.isfinite(values).all() or len(names) != values.shape[1]: raise ValueError('Invalid numeric CSV.')
    return {name:values[:, i] for i,name in enumerate(names)}


def check_local_links(documents, pending_output):
    pending_output=Path(pending_output).resolve(); checked=0
    for document in documents:
        for destination in re.findall(r'\]\(([^)]+)\)',document.read_text()):
            if destination.startswith(('https://','http://','#')): continue
            target=(document.parent/destination.split('#',1)[0]).resolve()
            # The audit can link to the exact file it will create on success.
            # No other nonexistent target is accepted.
            if not target.exists() and target!=pending_output:
                raise ValueError('Missing local link: '+str(document)+' -> '+destination)
            checked+=1
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args(); root = args.report.resolve()
    if args.out.exists(): raise FileExistsError('Preserve prior audit.')
    manifest = json.loads((root/'evidence/export_manifest.json').read_text())
    for name, expected in manifest['exported_files'].items():
        path = root/name
        if sha(path) != expected['SHA256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('Exported artifact changed: '+name)
    metrics = json.loads((root/'data/verified_metrics.json').read_text())
    checked = {}; total = 0
    for prefix, key in [('lake_hil','lake_hil'), ('sea_hil','sea_hil'), ('qiandao_side_scan','qiandao_side_scan')]:
        data = read_csv(root/f'data/{prefix}_errors_full.csv')
        count = len(data['timestamp_s']); expected = metrics[key]
        if count != expected['samples'] or np.any(np.diff(data['timestamp_s']) <= 0):
            raise ValueError('Incomplete time coverage: '+prefix)
        np.testing.assert_equal(data['distance_m'][-1], expected['distance_m'])
        for channel, value in expected['metrics'].items():
            e = data[channel+'_error_m']
            calculated = dict(rmse_m=np.sqrt(np.dot(e,e)/len(e)), median_m=np.median(e),
                p95_m=np.quantile(e,.95),p99_m=np.quantile(e,.99),max_m=e.max(),final_m=e[-1])
            for name in calculated: np.testing.assert_allclose(calculated[name], value[name], rtol=0, atol=1e-9)
        xyz = read_csv(root/f'data/{prefix}_trajectories_full.csv.gz')
        for name in ('timestamp_s','distance_m'): np.testing.assert_array_equal(data[name],xyz[name])
        truth = np.column_stack([xyz['truth_'+c+'_m'] for c in 'xyz'])
        for channel in ('dr','local','global'):
            pos = np.column_stack([xyz[channel+'_'+c+'_m'] for c in 'xyz'])
            error_key = ('projected_loops' if channel=='global' else channel) if prefix=='qiandao_side_scan' else 'board_'+channel
            np.testing.assert_allclose(np.linalg.norm(pos-truth,axis=1),data[error_key+'_error_m'],rtol=0,atol=1e-8)
        checked[prefix] = dict(samples=count, full_errors_recomputed=True, all_lossless_XYZ_checked=True)
        total += count

    report_path = root/'海底地形地貌匹配算法仿真报告.md'
    text = report_path.read_text()
    headings = re.findall(r'^(#{1,4})\s+(.+)$', text, re.M)
    if [title for marker,title in headings if marker=='#'] != ['海底地形地貌匹配算法仿真报告']:
        raise ValueError('Report title hierarchy changed.')
    for marker,title in headings:
        if marker=='##' and not re.match(r'^[一二三四五六七八九十]+、',title): raise ValueError('Invalid second-level heading.')
        if marker=='###' and not re.match(r'^\d+\.',title): raise ValueError('Invalid third-level heading.')
        if marker=='####' and not re.match(r'^（\d+）',title): raise ValueError('Invalid fourth-level heading.')
    links=check_local_links([report_path,root/'README.md',root/'数据字典.md',root.parents[2]/'README.md'],args.out)
    from docx import Document
    word=report_path.with_suffix('.docx'); pdf=report_path.with_suffix('.pdf')
    with zipfile.ZipFile(word) as archive:
        if archive.testzip() is not None: raise ValueError('Corrupt Word archive.')
    doc=Document(word)
    actual_headings=[(int(x.style.name.split()[-1]),x.text) for x in doc.paragraphs if x.style.name.startswith('Heading ')]
    if actual_headings != [(len(m),title) for m,title in headings]: raise ValueError('Word heading hierarchy changed.')
    figures=len(re.findall(r'^!\[',text,re.M))
    if len(doc.inline_shapes)!=figures or figures!=10: raise ValueError('Missing embedded report figures.')
    pdfmeta=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    pages=int(re.search(r'^Pages:\s+(\d+)',pdfmeta,re.M)[1])
    extracted=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
    compact=re.sub(r'\s+','',extracted)
    for token in ['海底地形地貌匹配算法仿真报告','71.44%','6.829','6.830','10.800','图10']:
        if token not in compact: raise ValueError('Rendered PDF is missing report content: '+token)
    if pages<15: raise ValueError('Unexpectedly short PDF.')
    oversized=[]; secrets=[]; patterns=[r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',
        r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        r'(?im)(?:password|passwd)\s*[:=]\s*[\"\'][^\"\'\n]{4,}[\"\']']
    for path in root.rglob('*'):
        if not path.is_file(): continue
        if path.stat().st_size>=100*1024**2: oversized.append(str(path.relative_to(root)))
        if path.suffix in ('.md','.json','.txt','.py'):
            content=path.read_text()
            if any(re.search(pattern,content) for pattern in patterns): secrets.append(str(path.relative_to(root)))
    if oversized or secrets: raise ValueError('Unsafe delivery artifacts: '+json.dumps(dict(oversized=oversized,secrets=secrets)))
    result=dict(status='completed_independent_full_data_and_document_delivery_audit',datasets=checked,
        total_native_error_rows=total,exported_artifact_hashes_verified=len(manifest['exported_files']),
        local_links_checked=links,word_heading_levels_preserved=True,word_figures=figures,word_tables=len(doc.tables),
        pdf_pages=pages,pdf_text_and_metadata_readable=True,scientific_fidelity_requires_report_scope=True,
        no_secret_patterns_detected_in_report_package=True,no_single_file_over_100MiB=True,
        source_script_SHA256=sha(Path(__file__)),documents_SHA256={p.name:sha(p) for p in
            [report_path,word,pdf,root/'README.md',root/'数据字典.md']})
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='documents_SHA256'},ensure_ascii=False))


if __name__=='__main__': main()
