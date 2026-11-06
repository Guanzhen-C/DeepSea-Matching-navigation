"""Freeze a small, explicit, source-complete denoising checkpoint for Git.

No raw acoustic/GPS payloads, credentials, large images or unrelated worktree
changes are collected. Existing scientific source and model bytes are never
edited. The release contains references to repository sources plus two small
selected checkpoints; Git itself stores the immutable source snapshot.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil


ROOTS = '''BathymetrySideScanDrawing Bathymetry_SonarDefaultColormap3D
StitchBathymetrySideScanDrawing PrepareMeasuredSonarPhaseCache
ExportMeasuredSonarVoyage172 ReceiveMeasuredSonarVoyage172
RecoveredMatchedSonarModels TrainMatchedSonarFactorial PrepareMatched172Recovery
RunMatched172Validation AuditMatched172Validation InferFullMeasuredSonar172
RenderFullMeasuredSonar172 AuditFullMeasuredSonar172 RangeGainContinuityRepair
TrialRangeGainContinuity172 PackageBoundaryCandidatePortraitPDF
ExportRangeGainContinuityReview ExportPinnedFullSamplePortrait
FreezeSonarDenoisingRelease'''.split()
TESTS = '''test_band_anchored_native_features.py test_band_anchored_native_wavelet.py
test_band_anchored_input_replay.py test_band_anchored_audit_ledger.py
test_matched_sonar_signal.py test_matched_sonar_wavelet.py
test_matched_sonar_artifacts.py test_matched_172_evidence.py test_matched_172_transfer.py
test_full_measured_172_protocol.py test_native_sonar_palette_stream.py
test_range_gain_continuity_repair.py test_boundary_candidate_portrait_pdf.py
test_pinned_full_sample_portrait.py test_matlab_sonar_replica.py
test_matlab_sonar_stitch.py test_public_sonar_wavelet.py test_phase_aware_sonar_wavelet.py
test_local_relative_sonar_features.py'''.split()
WEIGHTS = {
    'fusion_wide_front': 'faece15dfd9c9a4b028a409b6d8ee583c93287e1b6ec81afd0c274c4686073e0',
    'fusion_limited_front': '1bdf60020552169378efd9079526b3d50f543a892eb0efd497b00535c6b88f1a',
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda: f.read(1024*1024), b''):
            h.update(x)
    return h.hexdigest()


def dependencies(tree, base):
    for node in ast.walk(tree):
        names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                 [n.name for n in node.names] if isinstance(node, ast.Import) else [])
        for name in names:
            top = name.split('.')[0]
            if (base/(top+'.py')).is_file():
                yield top
        # Include source scripts invoked by subprocess or copied into a
        # scientific source snapshot, not just Python import statements.
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.endswith('.py') and len(node.value) < 128
                and not any(c.isspace() for c in node.value) and '/' not in node.value
                and (base/node.value).is_file()):
            yield node.value[:-3]


def freeze(project, out):
    project, out = Path(project).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Keep earlier releases, including incomplete ones.')
    base = project/'standalone'
    todo, seen = list(ROOTS), set()
    tests = [project/'tests'/n for n in TESTS]
    for p in tests:
        todo.extend(dependencies(ast.parse(p.read_text()), base))
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        p = base/(n+'.py')
        tree = ast.parse(p.read_text())
        seen.add(n)
        todo.extend(dependencies(tree, base))
    paths = sorted([base/(n+'.py') for n in seen]+tests)
    files = {str(p.relative_to(project)): dict(SHA256=sha(p), bytes=p.stat().st_size) for p in paths}
    source = project/'output/real_sonar/固定去噪基线_20261106/weights'
    if any(sha(source/(n+'.pt')) != digest for n, digest in WEIGHTS.items()):
        raise ValueError('Exact two user-pinned selected checkpoints required.')
    out.mkdir(parents=True)
    (out/'weights').mkdir()
    checkpoints = {}
    for n, digest in WEIGHTS.items():
        p = out/'weights'/(n+'.pt')
        shutil.copy2(source/(n+'.pt'), p)
        if sha(p) != digest:
            raise ValueError('Checkpoint copy differs.')
        checkpoints[n] = dict(file=str(p.relative_to(project)), SHA256=digest, bytes=p.stat().st_size)
    report = dict(release='sonar-denoise-stop-point-20261106',
        baseline='fusion_wide_front', comparison_only='fusion_limited_front',
        last_reviewed_edge_variant='gain_and_commonmode',
        automatic_replacement_of_original_baseline=False,
        denoising_development_stopped_by_user=True,
        edge_processing='RangeGainContinuityRepair.py',
        code_files=files, source_count=len(seen), test_file_count=len(tests),
        selected_checkpoints=checkpoints,
        complete_local_import_and_literal_script_dependency_closure=True,
        source_shape=[2, 8, 2020, 12189], original_parts=[1033, 987],
        front=dict(fs_Hz=37650, reference_band_Hz=[-15000, -3000], taps=257,
                   recorded_band_is_hypothesis_not_verified_both_side_RF_calibration=True),
        edge=dict(context=1024, patch=512, modified_valid_columns=384, ramp_columns=256,
                  sigma_physical_records=8, statistical_radiometric_correction=True),
        audit_references=dict(
            original_native_PNG_SHA256='510621757a2323767974f19b6a1188169ff11df1f4044febb8ad415c3233d2fc',
            edge_native_PNG_SHA256='b48ba90c5656e51279c34591d7fba719fe25a22e28a000d1b5cfa3d5009d97be',
            edge_completed_report_SHA256='f8ca174a4279e524dbd41ba2151e339f72c198661af691a104ad186fd18c8142',
            edge_native_body_pixels_unchanged_across_two_candidates=377587472),
        private_MAT_GPS_payloads_or_images_included=False,
        native_data_and_original_recovery_evidence_required_separately=True,
        clean_ground_truth_available=False, guaranteed_lossless_denoising=False,
        display_colour_map_used_for_navigation_measurements=False)
    manifest = out/'release_manifest.json'
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def verify(project, release):
    project, release = Path(project).resolve(), Path(release).resolve()
    report = json.loads((release/'release_manifest.json').read_text())
    checks = {**report['code_files'],
              **{x['file']: x for x in report['selected_checkpoints'].values()}}
    for rel, record in checks.items():
        p = project/rel
        if p.stat().st_size != record['bytes'] or sha(p) != record['SHA256']:
            raise ValueError('Release file changed: '+rel)
    return dict(status='all_frozen_source_and_selected_weight_hashes_verified',
                files=len(checks), source=report['source_count'], tests=report['test_file_count'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    if a.verify:
        print(json.dumps(verify(a.project, a.release)))
    else:
        r = freeze(a.project, a.release)
        print(json.dumps(dict(source=r['source_count'], tests=r['test_file_count'], weights=len(r['selected_checkpoints']))))
