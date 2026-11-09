"""Only traceback-proven ARPACK import aliases may move to SciPy's public API."""
from copy import deepcopy
import os
from pathlib import Path

import pytest

from src import source_repair as repair
from src.method_adapters import digest
from test_source_repair_auto import enable_auto
from test_source_repair_service import CompatibilityLLM, repair_case


LEGACY = 'scipy.sparse.linalg.eigen.arpack'
PUBLIC = 'scipy.sparse.linalg'
TERMINAL = ("ModuleNotFoundError: No module named 'scipy.sparse.linalg.eigen.arpack'; "
            "'scipy.sparse.linalg.eigen' is not a package")
SOURCE = f'from {LEGACY} import eigsh as spectral\nprint(spectral.__name__)\n'


def scipy_failure(tmp_path, source=SOURCE, *, line=1, terminal=TERMINAL, path=None, source_name='model.py'):
    target = tmp_path / source_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding='utf-8')
    frame = path or target
    diagnostic = (f'Traceback (most recent call last):\n  File "{frame}", line {line}, in <module>\n'
                  f'    {source.splitlines()[line - 1].strip()}\n{terminal}\n')
    return ({'executed': True, 'success': False,
             'final': {'exit_code': 1, 'stderr': diagnostic}},
            {'files': {source_name: digest(target)}})


def validate_migration(tmp_path, source, diagnostic, *, source_name='model.py'):
    repair.validate_compatibility_proposal(tmp_path, [{'path': source_name, 'content': source}], diagnostic)


@pytest.mark.parametrize('source,line,symbols', [
    (SOURCE, 1, ['eigsh']),
    (f'from {LEGACY} import eigs\n', 1, ['eigs']),
    (f'from {LEGACY} import eigsh as sym, eigs as nonsym\n', 1, ['eigsh', 'eigs']),
    (f'from {LEGACY} import (\n    eigsh as sym,\n    eigs as nonsym,\n)\n', 2, ['eigsh', 'eigs']),
])
def test_scipy_exact_exports_aliases_and_multiline_imports_are_traceback_bound(tmp_path, source, line, symbols):
    execution, snapshot = scipy_failure(tmp_path, source, line=line)
    selected = repair._automatic_repair_sources(execution, tmp_path, snapshot)
    assert selected['selected'] == ['model.py']
    assert selected['compatibility_kind'] == 'import_location'
    assert set(selected['missing_names']) == set(symbols)
    validate_migration(tmp_path, source.replace(LEGACY, PUBLIC), execution['final']['stderr'])


@pytest.mark.parametrize('terminal', [
    "ModuleNotFoundError: No module named 'scipy.sparse.linalg.eigen.arpack'",
    'ModuleNotFoundError: No module named "scipy.sparse.linalg.eigen.arpack"',
])
def test_exact_missing_legacy_module_can_omit_parent_not_package_suffix(tmp_path, terminal):
    execution, snapshot = scipy_failure(tmp_path, terminal=terminal)
    assert repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected'] == ['model.py']
    validate_migration(tmp_path, SOURCE.replace(LEGACY, PUBLIC), execution['final']['stderr'])


@pytest.mark.parametrize('source', [
    f'from {LEGACY} import *\n',
    f'from {LEGACY} import eigsh, ArpackError\n',
    f'from {LEGACY} import _arpack\n',
    f'from .{LEGACY} import eigsh\n',
    f'import {LEGACY} as arpack\n',
    'from scipy.sparse.linalg.eigen import eigsh\n',
    'from scipy.sparse.linalg.eigen.arpack_extra import eigsh\n',
    f'# from {LEGACY} import eigsh\n',
])
def test_scipy_migration_does_not_generalize_to_modules_private_or_mixed_exports(tmp_path, source):
    execution, snapshot = scipy_failure(tmp_path, source)
    assert not repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected']
    if not source.startswith('#'):
        with pytest.raises(ValueError):
            changed = source.replace(LEGACY, PUBLIC).replace('scipy.sparse.linalg.eigen', PUBLIC)
            validate_migration(tmp_path, changed, execution['final']['stderr'])


@pytest.mark.parametrize('terminal', [
    "ModuleNotFoundError: No module named 'scipy'",
    "ModuleNotFoundError: No module named 'scipy.sparse.linalg.eigen.arpack_extra'",
    "ImportError: cannot import name 'eigsh' from 'scipy.sparse.linalg.eigen.arpack'",
    "AttributeError: 'dict' has no attribute 'iteritems'",
])
def test_unrelated_terminal_error_cannot_authorize_scipy_import_migration(tmp_path, terminal):
    execution, snapshot = scipy_failure(tmp_path, terminal=terminal)
    assert not repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected']
    with pytest.raises(ValueError):
        validate_migration(tmp_path, SOURCE.replace(LEGACY, PUBLIC), execution['final']['stderr'])


@pytest.mark.parametrize('kind', ['missing_traceback', 'trailing_output', 'historical_print', 'historical_traceback'])
def test_scipy_import_fix_requires_actual_complete_terminal_traceback(tmp_path, kind):
    execution, snapshot = scipy_failure(tmp_path)
    diagnostic = execution['final']['stderr']
    if kind == 'missing_traceback':
        diagnostic = TERMINAL
    elif kind == 'trailing_output':
        diagnostic += 'continuing unrelated work\n'
    else:
        prefix = diagnostic if kind == 'historical_traceback' else f'Historical example: {TERMINAL}\n'
        diagnostic = prefix + 'Traceback (most recent call last):\n'
        diagnostic += f'  File "{tmp_path / "model.py"}", line 2, in <module>\n'
        diagnostic += 'RuntimeError: actual unrelated failure\n'
    execution['final']['stderr'] = diagnostic
    assert not repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected']
    with pytest.raises(ValueError):
        validate_migration(tmp_path, SOURCE.replace(LEGACY, PUBLIC), diagnostic)


@pytest.mark.parametrize('mutation', ['changed_hash', 'wrong_line', 'outside_traceback'])
def test_scipy_automatic_scope_requires_same_frozen_source_and_real_frame(tmp_path, mutation):
    source = SOURCE + "raise RuntimeError('different statement')\n"
    execution, snapshot = scipy_failure(tmp_path, source)
    if mutation == 'changed_hash':
        (tmp_path / 'model.py').write_text(source + '# changed\n', encoding='utf-8')
    elif mutation == 'wrong_line':
        execution, snapshot = scipy_failure(tmp_path, source, line=3)
    else:
        execution, snapshot = scipy_failure(tmp_path, source, path=tmp_path.parent / 'outside.py')
    assert not repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected']
    if mutation != 'changed_hash':
        with pytest.raises(ValueError):
            validate_migration(tmp_path, source.replace(LEGACY, PUBLIC), execution['final']['stderr'])


@pytest.mark.parametrize('shadow,source_name', [
    ('scipy.py', 'model.py'), ('scipy/__init__.py', 'model.py'), ('utils/scipy.py', 'utils/model.py'),
])
def test_repository_scipy_shadow_never_authorizes_public_dependency_substitution(tmp_path, shadow, source_name):
    execution, snapshot = scipy_failure(tmp_path, source_name=source_name)
    target = tmp_path / shadow
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('LOCAL = True\n', encoding='utf-8')
    snapshot['files'][shadow] = digest(target)
    assert not repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected']
    with pytest.raises(ValueError):
        validate_migration(tmp_path, SOURCE.replace(LEGACY, PUBLIC), execution['final']['stderr'], source_name=source_name)


def test_only_the_traceback_import_in_one_file_can_move(tmp_path):
    source = f'from {LEGACY} import eigsh\nfrom {LEGACY} import eigs\n'
    execution, snapshot = scipy_failure(tmp_path, source)
    assert repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected'] == ['model.py']
    validate_migration(tmp_path, source.replace(LEGACY, PUBLIC, 1), execution['final']['stderr'])
    with pytest.raises(ValueError):
        validate_migration(tmp_path, source.replace(LEGACY, PUBLIC), execution['final']['stderr'])


def test_same_physical_line_multiple_imports_are_ambiguous_and_never_migrated(tmp_path):
    source = f'from {LEGACY} import eigsh; from {LEGACY} import eigs\n'
    execution, snapshot = scipy_failure(tmp_path, source)
    assert not repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected']
    with pytest.raises(ValueError):
        validate_migration(tmp_path, source.replace(LEGACY, PUBLIC), execution['final']['stderr'])


def test_migration_cannot_reorder_dual_symbol_bindings(tmp_path):
    source = f'from {LEGACY} import eigsh as symmetric, eigs as general\n'
    execution, _ = scipy_failure(tmp_path, source)
    changed = f'from {PUBLIC} import eigs as general, eigsh as symmetric\n'
    with pytest.raises(ValueError):
        validate_migration(tmp_path, changed, execution['final']['stderr'])


@pytest.mark.parametrize('changed', [
    SOURCE.replace(LEGACY, PUBLIC).replace('eigsh as spectral', 'eigs as spectral'),
    SOURCE.replace(LEGACY, PUBLIC).replace('as spectral', 'as other'),
    SOURCE.replace(LEGACY, 'numpy.linalg'),
    SOURCE.replace(LEGACY, PUBLIC).replace('spectral.__name__', '"fabricated result"'),
    SOURCE.replace(LEGACY, PUBLIC) + "print('fabricated metric')\n",
])
def test_scipy_migration_preserves_exact_symbol_binding_and_computation(tmp_path, changed):
    execution, _ = scipy_failure(tmp_path)
    with pytest.raises(ValueError):
        validate_migration(tmp_path, changed, execution['final']['stderr'])


def test_scipy_call_parameters_and_training_schedule_cannot_change(tmp_path):
    source = (f'from {LEGACY} import eigsh as spectral\n'
              'epochs = 10000\nseed = 42\nresult = spectral(matrix, k=6, tol=0)\n')
    execution, _ = scipy_failure(tmp_path, source)
    migrated = source.replace(LEGACY, PUBLIC)
    for replacement in (migrated.replace('k=6', 'k=2'), migrated.replace('tol=0', 'tol=1e-3'),
                        migrated.replace('epochs = 10000', 'epochs = 10'),
                        migrated.replace('seed = 42', 'seed = 7')):
        with pytest.raises(ValueError):
            validate_migration(tmp_path, replacement, execution['final']['stderr'])


def set_external_scipy_fixture(case, tmp_path, source):
    """A synthetic external dependency tests repair transport, not SciPy numerics."""
    external = tmp_path / 'external_dependency'
    for name, content in {
        'scipy/__init__.py': '', 'scipy/sparse/__init__.py': '',
        'scipy/sparse/linalg/__init__.py': 'def eigsh():\n    pass\ndef eigs():\n    pass\n',
        'scipy/sparse/linalg/eigen.py': '# The former package is now a module.\n',
    }.items():
        path = external / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    case['runner'].executor._exec_env.return_value = {
        **os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONPATH': str(external)}
    (case['workspace'] / 'model.py').write_text(source, encoding='utf-8')
    (case['workspace'] / 'train.py').write_text('import model\n', encoding='utf-8')
    for name in case['snapshot']['files']:
        case['snapshot']['files'][name] = digest(case['workspace'] / name)
    initial = case['runner'].run(case['workspace'], case['data']['execution_plan'], {})
    assert not initial['success']
    case['initial'].clear()
    case['initial'].update(initial)
    enable_auto(case)
    return external


def assert_original_experiment_preserved(case, result, original_plan, runtime, data_sha):
    assert result['success'], result
    assert case['data']['original_execution_plan'] == original_plan
    assert case['data']['execution_plan']['steps'] == original_plan['steps']
    assert all(result['environment'][key] == value for key, value in runtime.items())
    copied = Path(case['data']['source_repair']['workspace'])
    assert digest(copied / 'input.json') == data_sha
    case['verify']()
    verified = repair.verify_repaired_execution(case['plan'], case['data'], result)
    assert verified['source_modified'] and verified['is_reproduced'] is None
    assert verified['paper_metrics_verified'] is False


def test_real_scipy_import_failure_retries_original_argv_runtime_and_frozen_data(repair_case, tmp_path):
    case = repair_case
    source = (f'from {LEGACY} import eigsh as spectral\n'
              "import json\nwith open('input.json') as stream:\n    records = json.load(stream)\n"
              "print(spectral.__name__, sum(records['values']))\n")
    external = set_external_scipy_fixture(case, tmp_path, source)
    assert LEGACY in case['initial']['final']['stderr']
    original_plan = deepcopy(case['data']['execution_plan'])
    runtime = {key: case['initial']['environment'][key] for key in ('executable', 'runtime', 'requirements_txt')}
    data_sha = digest(case['workspace'] / 'input.json')
    dependency_sha = digest(external / 'scipy/sparse/linalg/__init__.py')
    llm = CompatibilityLLM(lambda *_: {'model.py': source.replace(LEGACY, PUBLIC)})
    result = case['run'](llm)
    assert result['final']['stdout'].strip() == 'eigsh 6'
    assert len(llm.calls) == 1
    assert digest(external / 'scipy/sparse/linalg/__init__.py') == dependency_sha
    assert_original_experiment_preserved(case, result, original_plan, runtime, data_sha)


def test_collections_then_scipy_cumulative_repairs_keep_both_observed_import_migrations(repair_case, tmp_path):
    case = repair_case
    source = ('from collections import Mapping as Map\n'
              f'from {LEGACY} import eigsh as spectral\n'
              'print(isinstance({}, Map), spectral.__name__)\n')
    set_external_scipy_fixture(case, tmp_path, source)
    assert 'ImportError' in case['initial']['final']['stderr']
    original_plan = deepcopy(case['data']['execution_plan'])
    runtime = {key: case['initial']['environment'][key] for key in ('executable', 'runtime', 'requirements_txt')}
    data_sha = digest(case['workspace'] / 'input.json')

    def propose(context, number):
        fixed = source.replace('from collections import', 'from collections.abc import')
        if number == 2:
            assert LEGACY in context['diagnostic']
            assert 'collections' in context['initial_diagnostic']
            fixed = fixed.replace(LEGACY, PUBLIC)
        return {'model.py': fixed}

    llm = CompatibilityLLM(propose)
    result = case['run'](llm)
    assert len(llm.calls) == 2
    assert result['final']['stdout'].strip() == 'True eigsh'
    assert [item['status'] for item in result['repair_attempts']] == ['failed', 'succeeded']
    assert_original_experiment_preserved(case, result, original_plan, runtime, data_sha)
