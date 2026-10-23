"""Fixed mixed-storage paths around invented native runs, not paper acceptance."""
from copy import deepcopy
import ast
import hashlib

import pytest

from src.discovered_repository_plan import build_evidence_packet
from src.discovered_repository_runtime import validate_capture, verify_capture
from src.native_loop import (discover_native_selections, resolve_function_selection,
                             resolve_module_loop, selector_expression)
from test_discovered_module_loop import use_function_loop
from test_discovered_repository_series import run_series, series_fixture


def mixed(name, *steps):
    return {'name': name, 'path': list(steps)}


def use_mixed_paths(fixture):
    use_function_loop(fixture)
    path = fixture['workspace'] / 'train.py'
    source = path.read_text('utf-8')
    source = source.replace('        self.features = x\n        self.labels = y\n        self.indices = ids',
        "        self.graph = {'node_feat': x, 'targets': y, 'indices': ids}")
    source = source.replace('nothing = None', """assert bundle.graph['indices'].ndim == 1
class AuthorMetrics:
    def __init__(self, value):
        self.values = {'validation': value}
nothing = None""")
    source = source.replace('model(features, nothing)', "model(bundle.graph['node_feat'], nothing)")
    source = source.replace('labels[:4]', "bundle.graph['targets'][:4]")
    source = source.replace('labels[idx_val]', "bundle.graph['targets'][idx_val]")
    source = source.replace('labels[idx_test]', "bundle.graph['targets'][bundle.graph['indices']]")
    source = source.replace('            if val_loss < best_val_loss:',
        "            metrics = AuthorMetrics(val_loss)\n            if metrics.values['validation'] < best_val_loss:")
    path.write_text(source, encoding='utf-8')
    fixture['snapshot']['files']['train.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
    next(c for c in fixture['proposal']['citations'] if c.get('path') == 'train.py')['quote'] = source
    capture = fixture['proposal']['capture']
    capture.update(
        forward_args=[mixed('bundle', {'attribute': 'graph'}, {'key': 'node_feat'}), 'nothing'],
        labels=mixed('bundle', {'attribute': 'graph'}, {'key': 'targets'}),
        test_indices=mixed('bundle', {'attribute': 'graph'}, {'key': 'indices'}))
    capture['scope']['selection']['validation_metric'] = mixed(
        'metrics', {'attribute': 'values'}, {'key': 'validation'})


def test_mixed_capture_paths_bind_to_the_complete_original_ast_chain(tmp_path):
    fixture = series_fixture(tmp_path)
    use_mixed_paths(fixture)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    capture = fixture['proposal']['capture']
    original = deepcopy(capture)
    assert validate_capture(capture, packet, fixture['workspace']) == capture
    assert capture == original
    # Each token occurs elsewhere, but this assembled path never occurs.
    invalid = deepcopy(capture)
    invalid['labels'] = mixed('metrics', {'attribute': 'graph'}, {'key': 'targets'})
    with pytest.raises(ValueError, match='ordered storage path is absent'):
        validate_capture(invalid, packet, fixture['workspace'])
    invalid = deepcopy(capture)
    invalid['labels'] = mixed('bundle', {'key': 'targets'}, {'attribute': 'graph'})
    with pytest.raises(ValueError, match='ordered storage path is absent'):
        validate_capture(invalid, packet, fixture['workspace'])


def test_mixed_native_selection_is_discovered_and_matches_exact_original_branch(tmp_path):
    fixture = series_fixture(tmp_path)
    use_mixed_paths(fixture)
    source = (fixture['workspace'] / 'train.py').read_text('utf-8')
    scope = fixture['proposal']['capture']['scope']
    resolved = resolve_module_loop(source, scope)
    found = discover_native_selections({'train.py': {'text': source, 'sha256': 'a' * 64}})
    assert any(option['scope']['selection']['validation_metric'] == scope['selection']['validation_metric']
               and option['selection_start'] == resolved['selection_start'] for option in found)
    function_scope = deepcopy(scope)
    function_scope.pop('loop_counter')
    assert resolve_function_selection(source, function_scope)['selection_start'] == resolved['selection_start']
    assert ast.unparse(selector_expression(scope['selection']['validation_metric'])) == "metrics.values['validation']"
    for steps in ([{'attribute': 'values'}, {'key': 'test'}],
                  [{'key': 'validation'}, {'attribute': 'values'}]):
        changed = deepcopy(scope)
        changed['selection']['validation_metric'] = mixed('metrics', *steps)
        with pytest.raises(ValueError, match='original source comparison'):
            resolve_module_loop(source, changed)


def test_mixed_paths_capture_all_native_runs_and_independent_selected_predictions(tmp_path):
    fixture, plan, spec_hash, execution = run_series(tmp_path, 'cpu', use_mixed_paths)
    result = verify_capture(plan, fixture['workspace'], execution, spec_hash)
    assert result['is_reproduced'] and result['independent_metrics_pass']
    assert [run['optimizer_steps'] for run in result['capture']['runs']] == [8, 10, 12]
    assert [run['accuracy'] for run in result['independent_metrics']['runs']] == [1., 1., 1.]
    assert all(run['validation_selection']['optimizer_steps'] < run['optimizer_steps']
               for run in result['capture']['runs'])
