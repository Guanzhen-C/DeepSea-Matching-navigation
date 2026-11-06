"""Stage a separate, hashed local-source closure for a reproducible long run.

Only Python sources are copied, never raw data, weights, outputs or secrets
from other folders. This allows later workspace edits without altering the
already-running experiment. The frozen estimator bundle stays external.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def source_closure(source, entries):
    source = Path(source); pending = list(entries); found = set()
    while pending:
        name = pending.pop()
        if name in found: continue
        path = source/name
        if not path.is_file(): raise FileNotFoundError(path)
        found.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
            for module in modules:
                candidate = module.split('.')[0]+'.py'
                if (source/candidate).is_file() and candidate not in found: pending.append(candidate)
    return sorted(found)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError('Never overwrite an existing staged runtime.')
    source = Path(__file__).resolve().parent
    entries = ['RunPureSideScanFeatureShort.py', 'AnalyzePureSideScanShort.py', 'FinalizePureSideScanExperiment.py']
    names = source_closure(source, entries)
    target = a.out/'standalone'; target.mkdir(parents=True)
    hashes = {}
    for name in names:
        shutil.copy2(source/name, target/name)
        expected = hashlib.sha256((source/name).read_bytes()).hexdigest()
        if hashlib.sha256((target/name).read_bytes()).hexdigest() != expected: raise RuntimeError('Copy changed source.')
        hashes[name] = expected
    report = dict(status='complete_separate_local_source_runtime', files_SHA256=hashes,
        source_directory=str(source), runtime_directory=str(target.resolve()), entrypoints=entries,
        required_PYTHONPATH=str(source.parent),
        source_bytes=sum((target/n).stat().st_size for n in names),
        data_or_weights_copied=False, frozen_estimator_bundle_modified=False)
    (a.out/'runtime_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
