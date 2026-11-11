"""Prepare an allowlisted public research bundle; never push or read raw sonar.

Only derived navigation trajectories, verified statistics, selected JSON evidence,
the completed report and report figures are copied. Original files stay intact.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TITLE = '海底地形地貌匹配导航算法仿真与Jetson硬件在环验证报告'
EXPECTED = {'sea': 243569, 'lake': 157056, 'qiandao': 62458}
EVIDENCE = (
    'sea_AGX/completion_and_integrity.json', 'sea_AGX/relative_motion_audit.json',
    'sea_AGX/timing_and_risk.json', 'sea_AGX/map_metadata.json',
    'sea_AGX/sender_summary.json', 'sea_AGX/common_truth_metrics.json',
    'lake_AGX/completion_and_integrity.json', 'lake_AGX/relative_motion_audit.json',
    'lake_AGX/map_metadata.json', 'lake_AGX/sender_summary.json',
    'lake_AGX/common_truth_metrics.json',
    'qiandao_NX/independent_full_audit.json', 'qiandao_NX/source_summary.json',
    'qiandao_NX/receiver_summary.json', 'qiandao_NX/scene_report.json',
    'qiandao_NX/deployment_manifest.json',
    'denoising/FIR_numeric_audit.json', 'denoising/FIR_summary.json',
    'denoising/NX_model_summary.json', 'native_preview_audit.json',
    'AGX_final_scope_review.json', 'document_build.json',
    'final_document_audit.json',
)
SENSITIVE_KEY = re.compile(
    r'password|passwd|secret|access_token|api_key|private_key|authorization|wifi_psk'
    r'|^(?:hostname|username|mac_address|serial_number|board_ip|host_ip)$', re.I)
LOCAL_PATH = re.compile(r'/(?:home|media|mnt)/[^\s\"\'<>;,，；）)]+')
IPV4 = re.compile(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def clean_text(value):
    def network(match):
        try:
            addr = ipaddress.ip_address(match[0])
        except ValueError:
            return match[0]
        return '[private-network]' if addr.is_private else match[0]
    value = IPV4.sub(network, value)
    return LOCAL_PATH.sub(lambda m: '[local-path]/' + Path(m[0]).name, value)


def clean(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            public_key = clean_text(key)
            if public_key in result:
                raise ValueError('Sanitized key collision; review evidence manually.')
            result[public_key] = '[redacted]' if SENSITIVE_KEY.search(key) else clean(item)
        return result
    if isinstance(value, list):
        return [clean(item) for item in value]
    return clean_text(value) if isinstance(value, str) else value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def export_frozen_deployment(deployment, out):
    """Export only hash-matched Python algorithms, never the deployed map."""
    deployment, out = Path(deployment).resolve(), Path(out).resolve()
    manifest = json.loads((deployment / 'deployment_manifest.json').read_text())
    exported = []
    for relative, entry in manifest['files'].items():
        if not relative.endswith('.py') or not relative.startswith((
                'runtime/standalone/', 'runtime/core/oceansim/slam/')):
            continue
        src = (deployment / relative).resolve()
        if not src.is_relative_to(deployment) or sha(src) != entry['SHA256']:
            raise ValueError('Frozen deployed code hash mismatch.')
        text = src.read_text()
        if clean_text(text) != text or re.search(r'(?:password|passwd|api_key)\s*=', text, re.I):
            raise ValueError('Review source code before public export.')
        dst = out / 'algorithm_snapshot/qiandao_NX' / relative.removeprefix('runtime/')
        if dst.exists():
            raise FileExistsError('Preserve code snapshot.')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        exported.append({'path': str(dst.relative_to(out)), 'SHA256': sha(dst),
                         'matches_deployed_manifest': True})
    dump(out / 'evidence/frozen_NX_algorithm_sources.json', {
        'status': 'hash_matched_frozen_deployment_sources_for_reading',
        'deployment_manifest_original_SHA256': sha(deployment / 'deployment_manifest.json'),
        'not_a_complete_standalone_replay_distribution': True,
        'no_deployed_map_arrays_or_weights_exported': True,
        'files': exported})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deployment', type=Path)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError('Preserve existing export; choose a new output directory.')
    if not out.is_relative_to(ROOT / 'docs/reports'):
        raise ValueError('Publication output must stay within project report directories.')
    report = (source / (TITLE + '.md')).read_text()
    if clean_text(report) != report:
        raise ValueError('Report prose requires a public-copy privacy review.')
    out.mkdir(parents=True)
    copied = []

    def copy(relative):
        src, dst = source / relative, out / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({'public_path': relative, 'source_SHA256': sha(src),
                       'export_SHA256': sha(dst), 'byte_identical': True})

    for suffix in ('.md', '.docx', '.pdf'):
        copy(TITLE + suffix)
    for rel in re.findall(r'!\[[^\]]*\]\((figures/[^)]+)\)', report):
        copy(rel)
    for scene in ('sea', 'lake'):
        copy(f'data/{scene}_hil_trajectories_full.csv.gz')
    copy('data/qiandao_NX_navigation_states.npz')
    metrics = json.loads((source / 'data/verified_metrics.json').read_text())
    dump(out / 'data/verified_metrics.json', clean(metrics))
    for scene, count in EXPECTED.items():
        src = source / f'data/{scene}_complete_error_samples.csv'
        with src.open('rb') as incoming, (out / f'data/{scene}_complete_error_samples.csv.gz').open('wb') as target:
            with gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0) as zipped:
                shutil.copyfileobj(incoming, zipped)
        a = np.loadtxt(src, delimiter=',', skiprows=1)
        if len(a) != count:
            raise ValueError('Incomplete error data.')
        for j, channel in enumerate(('DR', 'Local', 'Global')):
            error = a[:, j + 2]
            stats = {'rmse_m': np.sqrt(np.mean(error ** 2)),
                     'p95_m': np.percentile(error, 95), 'max_m': error.max(),
                     'final_m': error[-1]}
            for key, value in stats.items():
                if not np.isclose(value, metrics[scene]['metrics'][channel][key], rtol=1e-12, atol=1e-12):
                    raise ValueError('Reported metric differs from full error CSV.')
        ix = np.unique(np.linspace(0, len(a) - 1, 2000, dtype=int))
        np.savetxt(out / f'data/{scene}_error_preview_only.csv', a[ix], delimiter=',',
                   header='timestamp_s,distance_m,DR_error_m,Local_error_m,Global_error_m', comments='')
    sanitized = []
    for rel in EVIDENCE:
        src = source / 'evidence' / rel
        original = json.loads(src.read_text())
        public = clean(original)
        target = out / 'evidence' / rel
        dump(target, public)
        sanitized.append({'public_path': 'evidence/' + rel,
                          'private_original_SHA256': sha(src),
                          'export_SHA256': sha(target),
                          'privacy_redacted': original != public})
    manifest = json.loads((source / 'evidence/source_manifest.json').read_text())
    dump(out / 'evidence/source_fingerprints.json', {
        'note': 'Fingerprints identify local evidence; absent raw inputs are not uploaded.',
        'sources': [{'source_id': f'source_{i:02d}', 'basename': Path(name).name,
                     **entry} for i, (name, entry) in enumerate(manifest['source_files'].items(), 1)]})
    dump(out / 'evidence/public_export_audit.json', {
        'status': 'allowlisted_derived_navigation_report_export',
        'raw_MAT_uploaded': False, 'per_ping_sonar_uploaded': False,
        'complex_signals_uploaded': False, 'models_or_engines_uploaded': False,
        'original_local_report_modified': False, 'simulation_started': False,
        'copied_unchanged': copied, 'sanitized_evidence': sanitized,
        'complete_samples': EXPECTED,
        'preview_only_max_samples': 2000,
        'summary_metrics_computed_from_full_data_not_previews': True,
        'source_script_SHA256': sha(__file__),
    })
    if args.deployment:
        export_frozen_deployment(args.deployment, out)
    for path in out.rglob('*'):
        if path.is_file() and path.stat().st_size >= 95 * 1024 ** 2:
            raise ValueError('File exceeds publication size budget.')
        if path.suffix in ('.json', '.md', '.csv', '.txt'):
            text = path.read_text()
            if clean_text(text) != text:
                raise ValueError('Unexpected private path/network in public text.')
    print(json.dumps({'out': str(out), 'files': sum(p.is_file() for p in out.rglob('*')),
                      'bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
