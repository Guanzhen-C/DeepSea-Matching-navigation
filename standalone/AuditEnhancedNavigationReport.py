"""Independent full numeric and document checks for the new LOCAL report.

Returns a pending-visual-review status; it cannot stand in for inspection of
rendered pages or for the full experiment itself. No upload operation exists.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from docx import Document

from AnalyzeSideScanLoopReplay import sha


def verify_outline(headings):
    titles = [text for level, text in headings if level == 1]
    if titles != ['海底地形地貌匹配算法仿真报告']:
        raise ValueError('Keep the exact single report title')
    chapters = [text for level, text in headings if level == 2]
    prefixes = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二', '十三', '十四']
    if len(chapters) != 14 or any(not text.startswith(prefix+'、') for prefix, text in zip(prefixes, chapters)):
        raise ValueError('Second-level headings must use the complete ordered Chinese chapter numbers')
    for level, text in headings:
        if level == 3 and re.match(r'^\d+\. ', text) is None:
            raise ValueError('Third-level headings require the 1. style')
        if level == 4 and re.match(r'^（\d+）', text) is None:
            raise ValueError('Fourth-level headings require full-width parenthesized numbers')


def verify_no_raw_payloads(directory):
    forbidden = {'.mat', '.npy', '.npz', '.h5', '.hdf5', '.xtf', '.s7k', '.bag', '.mcap', '.db3'}
    if any(p.suffix.lower() in forbidden for p in Path(directory).rglob('*') if p.is_file()):
        raise ValueError('Do not copy private raw/recorded/cache payloads into the report package')


def numeric_check(directory):
    directory = Path(directory)
    d = json.loads((directory / 'data/verified_new_map_metrics.json').read_text())
    with gzip.open(directory / 'data/full_trajectories_new_vs_old.csv.gz', 'rt') as stream:
        points = list(csv.DictReader(stream))
    with (directory / 'data/full_errors_new_vs_old.csv').open() as stream:
        errors = list(csv.DictReader(stream))
    n = d['samples']
    if n != 62458 or len(points) != n or len(errors) != n:
        raise ValueError('Report must retain all62458 samples in both full exports')
    timestamps = np.array([float(r['timestamp_s']) for r in points])
    np.testing.assert_array_equal(timestamps, [float(r['timestamp_s']) for r in errors])
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Missing or nonfinite chronological samples')
    truth = np.array([[float(r['truth_'+c+'_m']) for c in 'xyz'] for r in points])
    source_path = Path(d['source_run']) / 'navigation_states.npz'
    old_path = Path(d['original_source']) / 'navigation_states.npz'
    replay_path = Path(d['original_replay']) / 'estimated_states.npz'
    if (sha(source_path) != d['new_sources']['executed_states_SHA256']
            or sha(old_path) != d['baseline_sources']['original_states_SHA256']
            or sha(replay_path) != d['baseline_sources']['replay_outputs_SHA256']['estimated_states.npz']):
        raise ValueError('Underlying executed new/old states changed')
    source = np.load(source_path)
    old = np.load(old_path)
    replay = np.load(replay_path)
    np.testing.assert_array_equal(timestamps, source['timestamps'])
    np.testing.assert_array_equal(timestamps, old['timestamps'])
    np.testing.assert_array_equal(timestamps, replay['timestamps'])
    np.testing.assert_array_equal(truth, source['positions'][:, 0])
    for name in ('DR', 'old_Local', 'old_Global', 'new_Local', 'new_Global'):
        p = np.array([[float(r[name+'_'+c+'_m']) for c in 'xyz'] for r in points])
        e = np.linalg.norm(p-truth, axis=1)
        np.testing.assert_allclose(e, [float(r[name+'_error_m']) for r in errors], rtol=0, atol=1e-10)
        stats = dict(rmse_m=float(np.sqrt(np.mean(e**2))), median_m=float(np.median(e)),
                     p95_m=float(np.percentile(e, 95)), p99_m=float(np.percentile(e, 99)),
                     max_m=float(e.max()), final_m=float(e[-1]))
        for k, value in stats.items():
            np.testing.assert_allclose(value, d['metrics'][name][k], rtol=0, atol=1e-9)
        if name in ('DR', 'new_Local', 'new_Global'):
            index = {'DR': 1, 'new_Local': 2, 'new_Global': 3}[name]
            np.testing.assert_array_equal(p, source['positions'][:, index])
        else:
            index = {'old_Local': 1, 'old_Global': 2}[name]
            np.testing.assert_array_equal(p, replay['positions'][:, index])
    return d


def audit(directory):
    directory = Path(directory).resolve()
    verify_no_raw_payloads(directory)
    d = numeric_check(directory)
    source = directory / '海底地形地貌匹配算法仿真报告.md'
    word, pdf = source.with_suffix('.docx'), source.with_suffix('.pdf')
    text = source.read_text()
    images = re.findall(r'^!\[[^\]]*\]\(([^)]+)\)', text, re.M)
    if any(not (directory / p).is_file() for p in images):
        raise ValueError('Broken local report illustration')
    expected_headings = [(len(m[1]), m[2]) for m in re.finditer(r'^(#{1,4}) (.+)$', text, re.M)]
    doc = Document(word)
    actual = [(int(p.style.name.split()[-1]), p.text) for p in doc.paragraphs if p.style.name.startswith('Heading ')]
    if expected_headings != actual or len(doc.inline_shapes) != len(images):
        raise ValueError('Word omitted or changed headings/images')
    verify_outline(actual)
    content = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
    if '海底地形地貌匹配算法仿真报告' not in content or f'{d["metrics"]["new_Global"]["rmse_m"]:.3f}' not in content:
        raise ValueError('PDF lacks the correct report identity or actual new metric')
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])
    if pages < 14:
        raise ValueError('Full multi-section PDF appears incomplete')
    result = dict(status='passed_full_data_and_document_checks_pending_visual_review',
                  samples=d['samples'], complete_error_and_coordinate_exports_recomputed=True,
                  word_heading_hierarchy_and_images_verified=True, embedded_figures=len(images),
                  pdf_pages=pages, visual_review_required=True, raw_data_uploaded=False,
                  raw_payloads_not_copied_into_report=True,
                  complete_new_and_old_coordinate_groups_bound_to_executed_sources=True,
                  outputs={str(p.relative_to(directory)): dict(bytes=p.stat().st_size, SHA256=sha(p))
                           for p in directory.rglob('*') if p.is_file()})
    target = directory / 'evidence/final_numeric_document_audit.json'
    if target.exists():
        raise FileExistsError('Preserve previous audit')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'samples', 'pdf_pages', 'embedded_figures')}), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    audit(a.report)
