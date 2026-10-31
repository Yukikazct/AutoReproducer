"""Repair real removed stdlib aliases without changing imports arbitrarily."""
from copy import deepcopy
import collections

import pytest

from src import source_repair as repair
from src.method_adapters import digest
from test_source_repair_service import CompatibilityLLM, repair_case
from test_source_repair_auto import enable_auto


SOURCE = 'from collections import Mapping as Map\nprint(isinstance({}, Map))\n'
FIXED = SOURCE.replace('from collections import', 'from collections.abc import')


def import_failure(tmp_path, source=SOURCE, symbol='Mapping', module='collections', line=1):
    path = tmp_path / 'model.py'
    path.write_text(source, encoding='utf-8')
    failure = {'executed': True, 'success': False, 'final': {'exit_code': 1,
        'stderr': f'Traceback (most recent call last):\n  File "{path}", line {line}, in <module>\n'
                  f"ImportError: cannot import name '{symbol}' from '{module}' ({collections.__file__})\n"}}
    return failure, {'files': {'model.py': digest(path)}}


def test_real_import_error_enters_ai_repair_and_reexecutes_original_command(repair_case):
    case = repair_case
    (case['workspace'] / 'model.py').write_text(SOURCE, encoding='utf-8')
    (case['workspace'] / 'train.py').write_text('import model\n', encoding='utf-8')
    for name in case['snapshot']['files']:
        case['snapshot']['files'][name] = digest(case['workspace'] / name)
    execution = case['runner'].run(case['workspace'], case['data']['execution_plan'], {})
    assert not execution['success'] and 'ImportError' in execution['final']['stderr']
    case['initial'].clear()
    case['initial'].update(execution)
    enable_auto(case)
    original_plan = deepcopy(case['data']['execution_plan'])
    llm = CompatibilityLLM(lambda *_: {'model.py': FIXED})
    result = case['run'](llm)
    assert result['success'] and result['final']['stdout'].strip() == 'True'
    assert len(llm.calls) == 1
    assert list(llm.calls[0]['sources']) == ['model.py']
    state = case['data']['source_repair']
    assert state['auto_selection']['compatibility_kind'] == 'import_location'
    assert state['auto_selection']['missing_names'] == ['Mapping']
    assert case['data']['original_execution_plan'] == original_plan
    assert case['data']['execution_plan']['steps'] == original_plan['steps']
    case['verify']()
    verified = repair.verify_repaired_execution(case['plan'], case['data'], result)
    assert verified['source_modified'] and verified['is_reproduced'] is None


@pytest.mark.parametrize('source,symbol', [
    ('from collections import Mapping, MutableMapping\n', 'Mapping'),
    ('from collections import (\n    Mapping as Map,\n)\n', 'Mapping'),
    ('from collections import Iterable\n', 'Iterable'),
])
def test_only_known_exact_stdlib_aliases_are_selected_and_accepted(tmp_path, source, symbol):
    execution, snapshot = import_failure(tmp_path, source, symbol)
    selected = repair._automatic_repair_sources(execution, tmp_path, snapshot)
    assert selected['selected'] == ['model.py']
    repair.validate_compatibility_proposal(tmp_path, [{'path': 'model.py',
        'content': source.replace('from collections import', 'from collections.abc import')}],
        execution['final']['stderr'])


@pytest.mark.parametrize('source,symbol,module,line', [
    ('from collections import Missing\n', 'Missing', 'collections', 1),
    ('from thirdparty import Mapping\n', 'Mapping', 'thirdparty', 1),
    ('from .collections import Mapping\n', 'Mapping', 'collections', 1),
    ('from collections import *\n', 'Mapping', 'collections', 1),
    ('from collections import Mapping, Counter\n', 'Mapping', 'collections', 1),
    ('# from collections import Mapping\n', 'Mapping', 'collections', 1),
    ('from collections import Mapping\nraise ImportError()\n', 'Mapping', 'collections', 2),
])
def test_unsupported_imports_or_unrelated_traceback_line_never_enter_model_scope(
        tmp_path, source, symbol, module, line):
    execution, snapshot = import_failure(tmp_path, source, symbol, module, line)
    assert not repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected']


@pytest.mark.parametrize('changed', [
    FIXED.replace('collections.abc', 'thirdparty'),
    FIXED.replace('Mapping as Map', 'MutableMapping as Map'),
    FIXED.replace('Mapping as Map', 'Mapping as Other'),
    FIXED.replace('{}, Map', '[], Map'),
    FIXED + "print('fabricated metric')\n",
])
def test_import_migration_keeps_object_alias_computation_and_protocol(tmp_path, changed):
    execution, _ = import_failure(tmp_path)
    with pytest.raises(ValueError):
        repair.validate_compatibility_proposal(tmp_path, [{'path': 'model.py', 'content': changed}],
                                              execution['final']['stderr'])


def test_import_migration_requires_matching_actual_failure(tmp_path):
    import_failure(tmp_path)
    with pytest.raises(ValueError, match='实际报错'):
        repair.validate_compatibility_proposal(tmp_path, [{'path': 'model.py', 'content': FIXED}],
                                              "AttributeError: 'dict' has no attribute 'iteritems'")


def test_import_migration_cannot_be_reversed_without_evidence(tmp_path):
    import_failure(tmp_path, FIXED)
    with pytest.raises(ValueError, match='实际报错'):
        repair.validate_compatibility_proposal(tmp_path, [{'path': 'model.py', 'content': SOURCE}],
            f"ImportError: cannot import name 'Mapping' from 'collections' ({collections.__file__})")


def test_repository_shadow_of_collections_is_not_treated_as_stdlib(tmp_path):
    execution, snapshot = import_failure(tmp_path)
    shadow = tmp_path / 'lib/collections/__init__.py'
    shadow.parent.mkdir(parents=True)
    shadow.write_text('', encoding='utf-8')
    diagnostic = execution['final']['stderr'].replace(str(collections.__file__), str(shadow))
    execution['final']['stderr'] = diagnostic
    assert not repair._automatic_repair_sources(execution, tmp_path, snapshot)['selected']
    with pytest.raises(ValueError, match='实际报错'):
        repair.validate_compatibility_proposal(tmp_path, [{'path': 'model.py', 'content': FIXED}], diagnostic)
