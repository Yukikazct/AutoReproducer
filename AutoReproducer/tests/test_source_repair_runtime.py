"""Runtime recovery evidence cannot be rebound to a different computation."""
import ast
from copy import deepcopy

import pytest

from src.method_adapters import digest
from src.source_repair_runtime import rebase_compatibility


def pair(tmp_path, source, candidate=None):
    original, repaired = tmp_path / 'original', tmp_path / 'repaired'
    original.mkdir()
    repaired.mkdir()
    (original / 'model.py').write_text(source, encoding='utf-8')
    (repaired / 'model.py').write_text(candidate if candidate is not None else '\n\n' + source,
                                       encoding='utf-8')
    return original, repaired, {'model.py': digest(repaired / 'model.py')}


@pytest.mark.parametrize('kind,source,expression', [
    ('numpy_builtin_alias', 'import numpy as np\nvalue = np.float(1)\n', 'np.float'),
    ('numpy_float_reciprocal', 'import numpy as np\nvalue = np.power(values, -1)\n', 'np.power(values, -1)'),
    ('temporary_dataset_root', "load_dataset(root='/tmp/data')\n", "'/tmp/data'"),
])
def test_unchanged_expression_retains_identity_after_line_shift(tmp_path, kind, source, expression):
    original, repaired, effective = pair(tmp_path, source)
    node = next(node for node in ast.walk(ast.parse(source)) if ast.get_source_segment(source, node) == expression)
    fix = {'kind': kind, 'path': 'model.py', 'source_sha256': digest(original / 'model.py'),
           'line': node.lineno, 'column': node.col_offset, 'original_expression': expression}
    field = 'path_compatibility' if kind == 'temporary_dataset_root' else 'numeric_compatibility'
    config = {field: [fix]}
    before = deepcopy(config)
    rebased = rebase_compatibility(config, original, repaired, effective)[field][0]
    assert rebased['line'] == node.lineno + 2
    assert rebased['original_expression'] == expression
    assert rebased['source_sha256'] == effective['model.py']
    assert config == before


@pytest.mark.parametrize('replacement', ['np.float64(1)', 'np.float(2)'])
def test_prior_failure_cannot_authorize_a_different_recorded_expression(tmp_path, replacement):
    source = 'import numpy as np\nvalue = np.float(1)\n'
    original, repaired, effective = pair(tmp_path, source, source.replace('np.float(1)', replacement))
    config = {'numeric_compatibility': [{'kind': 'numpy_float_reciprocal', 'path': 'model.py',
        'source_sha256': digest(original / 'model.py'), 'line': 2, 'column': 8,
        'original_expression': 'np.float(1)'}]}
    with pytest.raises(ValueError, match='表达式'):
        rebase_compatibility(config, original, repaired, effective)


def test_duplicate_expressions_are_relocated_by_original_ast_position(tmp_path):
    source = 'import numpy as np\nfirst = np.float(1)\nsecond = np.float(2)\n'
    original, repaired, effective = pair(tmp_path, source)
    config = {'numeric_compatibility': [{'kind': 'numpy_builtin_alias', 'path': 'model.py',
        'source_sha256': digest(original / 'model.py'), 'line': 3, 'column': 9,
        'original_expression': 'np.float'}]}
    rebased = rebase_compatibility(config, original, repaired, effective)['numeric_compatibility'][0]
    assert rebased['line'] == 5 and rebased['column'] == 9


def test_ragged_reductions_keep_each_recorded_call_and_schema(tmp_path):
    source = 'import numpy as np\nmean = np.mean(values, axis=0)\nvariance = np.var(values, axis=0)\n'
    original, repaired, effective = pair(tmp_path, source)
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    config = {'numeric_compatibility': [{'kind': 'numpy_ragged_summary', 'path': 'model.py',
        'source_sha256': digest(original / 'model.py'), 'schema': {'rows': 2, 'columns': [{'kind': 'scalar'}]},
        'calls': [{'line': node.lineno, 'column': node.col_offset,
                   'original_expression': ast.get_source_segment(source, node), 'operation': node.func.attr}
                  for node in calls]}]}
    rebased = rebase_compatibility(config, original, repaired, effective)['numeric_compatibility'][0]
    assert [call['line'] for call in rebased['calls']] == [4, 5]
    assert rebased['schema'] == config['numeric_compatibility'][0]['schema']


@pytest.mark.parametrize('tamper', ['original', 'candidate'])
def test_relocation_requires_both_original_and_candidate_hashes(tmp_path, tamper):
    source = 'import numpy as np\nvalue = np.float(1)\n'
    original, repaired, effective = pair(tmp_path, source)
    config = {'numeric_compatibility': [{'kind': 'numpy_builtin_alias', 'path': 'model.py',
        'source_sha256': digest(original / 'model.py'), 'line': 2, 'column': 8,
        'original_expression': 'np.float'}]}
    root = original if tamper == 'original' else repaired
    (root / 'model.py').write_text(source + '# changed\n', encoding='utf-8')
    with pytest.raises(ValueError, match='哈希'):
        rebase_compatibility(config, original, repaired, effective)
