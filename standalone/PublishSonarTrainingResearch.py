"""Export code/configuration/training evidence only; never upload raw signals.

The export does not train, infer, connect to a board, or change frozen sources.
Machine-specific locations in public copies are redacted and explicitly logged.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil

from standalone.PublishJetsonHILResearchBundle import clean, clean_text, dump, sha

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT/'output/real_sonar/random_ripple_multisource_20261108_v2'
SEEDS = ('NoFIRSonarWavelet', 'TrainNoFIRSonarDistributed', 'RunRandomRippleSonarTrial',
         'RandomRippleEvaluation', 'EvaluateRandomRippleSonar', 'QualifyRandomRippleTraining',
         'AuditCompletedRandomRippleTrial', 'RunRippleTrainingRegression',
         'AuditRandomRippleLaunch', 'PackageAspectPreservingSonarPDF')
TESTS = ('test_no_fir_sonar_trial.py', 'test_random_triggered_sonar_noise.py',
         'test_random_ripple_progress_audit.py', 'test_random_ripple_evaluation_flow.py')
EVIDENCE = ('completed_run_summary.json', 'pretrain_completed_summary.json',
            'adapt_completed_summary.json', 'audits/pretrain_completed_24000_inputs.json',
            'audits/adapt_completed_8000_inputs.json',
            'audits/completed_full_execution_audit.json',
            'canonical_native_support_independent_audit.json',
            'all64_canonical_native_context_qualification.json')


def closure(source, seeds):
    pending = list(seeds); graph = {}
    while pending:
        name = pending.pop()
        if name in graph:
            continue
        path = source/(name+'.py')
        tree = ast.parse(path.read_text()); dependencies = set()
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split('.')[0]]
            elif isinstance(node, ast.Import):
                names = [n.name.split('.')[0] for n in node.names]
            dependencies.update(n for n in names if (source/(n+'.py')).is_file())
        graph[name] = sorted(dependencies)
        pending.extend(dependencies-set(graph))
    return graph


def normalized_ast(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = clean_text(node.value)
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(); out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh publication directory required.')
    if not out.is_relative_to(ROOT/'research'):
        raise ValueError('Unexpected publication target.')
    completed = json.loads((TRAINING/'completed_run_summary.json').read_text())
    for name, expected in completed['source_code_SHA256'].items():
        if sha(ROOT/'standalone'/name) != expected:
            raise ValueError('Historical training source changed: '+name)
    graph = closure(ROOT/'standalone', (*SEEDS, *(Path(n).stem for n in completed['source_code_SHA256'])))
    out.mkdir(parents=True); copied = []

    def public_code(src, relative):
        target = out/relative; target.parent.mkdir(parents=True, exist_ok=True)
        original = src.read_text(); exported = clean_text(original)
        if normalized_ast(original) != normalized_ast(exported):
            raise ValueError('Public-copy cleanup changed executable structure.')
        if exported == original:
            shutil.copy2(src, target)
        else:
            target.write_text(exported)
        copied.append(dict(path=relative, original_SHA256=sha(src), export_SHA256=sha(target),
            machine_location_redacted=original!=exported,
            same_executable_structure_except_location_strings=True,
            historical_training_manifest_SHA256=completed['source_code_SHA256'].get(src.name)))

    for name in sorted(graph):
        public_code(ROOT/'standalone'/(name+'.py'), 'standalone/'+name+'.py')
    for name in TESTS:
        public_code(ROOT/'tests'/name, 'tests/'+name)
    evidence=[]
    for name in EVIDENCE:
        source=TRAINING/name; value=json.loads(source.read_text())
        target=out/'records'/name; dump(target, clean(value))
        evidence.append(dict(path=str(target.relative_to(out)), original_SHA256=sha(source),
                             export_SHA256=sha(target), machine_locations_redacted=value!=clean(value)))
    profile=TRAINING/'random_ripple_profile.json'
    dump(out/'configs/random_ripple_profile.json', clean(json.loads(profile.read_text())))
    manifest=json.loads((TRAINING/'admission_manifest.json').read_text())
    public_sources=[]
    for i,row in enumerate(manifest['records']):
        keep={k:row[k] for k in ('id','domain','role','shape','beam','source','license',
            'unknown_original_noise','file_SHA256','source_payload_SHA256') if k in row}
        keep['local_path_not_published']=True
        keep['source_index']=i
        public_sources.append(keep)
    dump(out/'records/training_sources_without_paths.json',dict(
        original_admission_manifest_SHA256=sha(TRAINING/'admission_manifest.json'),
        counts=manifest['counts'], records=public_sources,
        source_arrays_and_observed_pixels_not_included=True,
        upstream_acquisition_processing_NOT_reversible=True))
    dump(out/'SOURCE_PROVENANCE.json',dict(
        status='code_configs_and_training_records_only',
        original_sources_modified=False,training_or_inference_started=False,
        model_weights_uploaded=False,raw_measurements_uploaded=False,
        code_dependency_graph=graph,source_files=copied,evidence_files=evidence,
        completed_training_manifest_source_files_verified=len(completed['source_code_SHA256']),
        other_dependencies_have_current_file_fingerprints_not_independent_historical_training_hashes=True))
    print(json.dumps(dict(output=str(out),source_modules=len(graph),
                          historical_source_matches=len(completed['source_code_SHA256'])),ensure_ascii=False))


if __name__=='__main__':main()
