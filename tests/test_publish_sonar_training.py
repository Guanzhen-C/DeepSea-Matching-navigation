from standalone.PublishSonarTrainingResearch import closure,normalized_ast


def test_dependency_closure_handles_cycles_and_excludes_external_modules(tmp_path):
    (tmp_path/'A.py').write_text('import numpy\nfrom B import f\n')
    (tmp_path/'B.py').write_text('import A\ndef f(): return 1\n')
    assert closure(tmp_path,['A'])=={'A':['B'],'B':['A']}


def test_machine_location_cleanup_does_not_allow_changed_math():
    before="ROOT='/home/test/private/file.npy'\ndef f(x): return x*2\n"
    public="ROOT='[local-path]/file.npy'\ndef f(x): return x*2\n"
    assert normalized_ast(before)==normalized_ast(public)
    assert normalized_ast(before)!=normalized_ast(public.replace('x*2','x*3'))
