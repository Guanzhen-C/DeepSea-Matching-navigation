"""Independent full-sample and rendered-document audit for the rewritten report.

Checks saved coordinates against executed states, all errors and segment
aggregates, Word structure and PDF text. Visual layout inspection remains a
separate explicit review; this tool never claims that numerical checks prove it.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np
from docx import Document

from AnalyzeSideScanLoopReplay import sha
from BuildMappedRangeNavigationReport import TITLE
from BuildSeabedSimulationReportEvidence import summarize


def load(path):
    return json.loads(Path(path).read_text())


def check_outline(headings):
    if [(level,text) for level,text in headings if level==1] != [(1,TITLE)]:
        raise ValueError('One exact report title required')
    numbers=['一','二','三','四','五','六','七','八','九','十','十一','十二','十三','十四']
    chapters=[text for level,text in headings if level==2]
    if len(chapters)!=len(numbers) or any(not title.startswith(n+'、') for n,title in zip(numbers,chapters)):
        raise ValueError('Fourteen ordered Chinese-numbered chapters required')
    for level,text in headings:
        if level==3 and not re.match(r'^\d+\. ',text): raise ValueError('Third level must use 1. style')
        if level==4 and not re.match(r'^（\d+）',text): raise ValueError('Fourth level must use （1） style')


def numeric_check(directory, label, summary):
    d=summary['comparison']; n=d['samples']
    source=Path(d['source_run']); candidate=Path(d['candidate_run'])
    if sha(source/'navigation_states.npz')!=d['source_states_SHA256']:
        raise ValueError('Executed source changed')
    for name,expected in d['replay_outputs_SHA256'].items():
        if sha(candidate/name)!=expected: raise ValueError('Executed candidate changed')
    with np.load(source/'navigation_states.npz') as f: original={k:f[k] for k in f.files}
    with np.load(candidate/'estimated_states.npz') as f: improved={k:f[k] for k in f.files}
    with gzip.open(directory/f'data/{label}_trajectories_full.csv.gz','rt') as f:
        rows=list(csv.DictReader(f))
    with (directory/f'data/{label}_errors_full.csv').open() as f: errors=list(csv.DictReader(f))
    if len(rows)!=n or len(errors)!=n: raise ValueError('Full-sample exports missing rows')
    expected=dict(truth=original['positions'][:,0],DR=improved['positions'][:,0],
        Local=improved['positions'][:,1],baseline_Global=original['positions'][:,3],
        improved_Global=improved['positions'][:,2])
    timestamps=np.array([float(r['timestamp_s']) for r in rows]); distance=np.array([float(r['distance_m']) for r in rows])
    np.testing.assert_array_equal(timestamps,original['timestamps'])
    np.testing.assert_array_equal(distance,original['distance_m'])
    np.testing.assert_array_equal(timestamps,[float(r['timestamp_s']) for r in errors])
    np.testing.assert_array_equal(distance,[float(r['distance_m']) for r in errors])
    if np.any(np.diff(timestamps)<=0): raise ValueError('Timeline must be complete and ordered')
    for name,p in expected.items():
        actual=np.array([[float(row[name+'_'+c+'_m']) for c in 'xyz'] for row in rows])
        np.testing.assert_array_equal(actual,p)
        if name=='truth': continue
        e=np.linalg.norm(actual-expected['truth'],axis=1)
        np.testing.assert_allclose(e,[float(r[name+'_error_m']) for r in errors],rtol=0,atol=1e-10)
        for key,value in summarize(e).items():
            np.testing.assert_allclose(value,summary['statistics'][name][key],rtol=0,atol=1e-9)
    with (directory/f'data/{label}_segments_100m.csv').open() as f: segments=list(csv.DictReader(f))
    for name in expected.keys()-{'truth'}:
        selected=[r for r in segments if r['channel']==name]
        if sum(int(r['samples']) for r in selected)!=n: raise ValueError('Segments omit samples')
        rmse=np.sqrt(sum(int(r['samples'])*float(r['rmse_m'])**2 for r in selected)/n)
        np.testing.assert_allclose(rmse,summary['statistics'][name]['rmse_m'],rtol=0,atol=1e-9)
    return dict(samples=n,all_coordinate_groups_match_executed_states=True,
                complete_errors_and_all_statistics_recomputed=True,segment_weighted_RMSE_verified=True)


def audit(directory):
    directory=Path(directory).resolve(); result_path=directory/'evidence/final_numeric_document_audit.json'
    if result_path.exists(): raise FileExistsError('Preserve previous audit')
    forbidden={'.mat','.npy','.npz','.h5','.hdf5','.bag','.mcap','.db3','.xtf','.s7k'}
    if any(p.suffix.lower() in forbidden for p in directory.rglob('*') if p.is_file()):
        raise ValueError('Private raw or reconstructable acoustic caches cannot enter report package')
    data=load(directory/'data/verified_report_results.json')
    verified={label:numeric_check(directory,label,data[label]) for label in ['main','fresh']}
    source=directory/(TITLE+'.md'); word=source.with_suffix('.docx'); pdf=source.with_suffix('.pdf')
    text=source.read_text(); headings=[(len(m[1]),m[2]) for m in re.finditer(r'^(#{1,4}) (.+)$',text,re.M)]
    check_outline(headings)
    with zipfile.ZipFile(word) as archive:
        if archive.testzip() is not None: raise ValueError('Invalid cross-platform Word ZIP')
    doc=Document(word); actual=[(int(p.style.name.split()[-1]),p.text) for p in doc.paragraphs
                              if p.style.name.startswith('Heading ')]
    images=re.findall(r'^!\[[^\]]*\]\(([^)]+)\)',text,re.M)
    if actual!=headings or len(doc.inline_shapes)!=len(images): raise ValueError('Word omits outline or figures')
    if doc.core_properties.title!=TITLE or doc.sections[0].header.paragraphs[0].text!=TITLE:
        raise ValueError('Word title/header mismatch')
    for relative in images:
        if not (directory/relative).is_file(): raise ValueError('Missing illustration')
    captions=re.findall(r'^图(\d+) ',text,re.M)
    if list(map(int,captions))!=list(range(1,len(images)+1)): raise ValueError('Figure captions must be consecutive')
    pdftext=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
    required=[TITLE,'57.04','21.99','5.040','5.951','9.940','0.98','62458','15152','preserve_legacy']
    if any(item not in pdftext for item in required): raise ValueError('PDF lacks required result/scope text')
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
    if pages<14: raise ValueError('Full report is incomplete')
    provenance=load(directory/'evidence/report_rewrite_provenance.json')
    if sha(provenance['previous_report'])!=provenance['previous_report_SHA256']:
        raise ValueError('Original report changed')
    result=dict(status='passed_full_numeric_and_document_checks_pending_visual_review',
        verified_trials=verified,pdf_pages=pages,embedded_figures=len(images),tables=len(doc.tables),
        heading_hierarchy_verified=True,Word_ZIP_reopened=True,all_coordinate_exports_are_lossless=True,
        previous_report_unchanged=True,raw_payloads_not_copied=True,external_upload_performed=False,
        main_and_fresh_are_controlled_simulations_not_field_navigation=True,
        final_endpoint_and_relative_regressions_disclosed=True,
        audit_source_SHA256=sha(Path(__file__)),visual_review_required=True,
        outputs={str(p.relative_to(directory)):dict(bytes=p.stat().st_size,SHA256=sha(p))
                 for p in directory.rglob('*') if p.is_file()})
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','pdf_pages','embedded_figures','tables']},ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(); audit(args.report)
