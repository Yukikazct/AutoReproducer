"""Advertise real source-resolved snapshot options, without assuming completion."""
from copy import deepcopy
import hashlib

import pytest

from src.native_loop import discover_native_selections, resolve_module_loop
from src.discovered_repository_runtime import _source_epoch_offset
import ast
from test_discovered_module_loop import SOURCE, SCOPE, use_function_loop
from test_discovered_repository_series import series_fixture
from test_graph_author_capture import use_pyg_function_loop


def files(source):
    return {'train.py': {'text': source, 'sha256': hashlib.sha256(source.encode()).hexdigest()}}


def test_module_loop_exposes_original_selection_without_requiring_author_checkpoint():
    options = discover_native_selections(files(SOURCE))
    assert len(options) == 1
    option = options[0]
    scope = deepcopy(option['scope'])
    scope['selection']['result_unit'] = 'fraction'
    assert scope == SCOPE
    assert option['author_checkpoint_required'] is False
    assert option['source_sha256'] == files(SOURCE)['train.py']['sha256']
    assert option['validation_branch'].startswith('if val_loss < best_val_loss:')
    assert option['scope']['selection']['result'] == 'selected_test'
    assert option['selection_start'] == resolve_module_loop(SOURCE, SCOPE)['selection_start']
    assert 'independent predictions at runtime' in option['remaining_checks']


def test_function_dictionary_and_underscore_options_are_exact_source_results(tmp_path):
    fixture = series_fixture(tmp_path)
    use_pyg_function_loop(fixture)
    source = (fixture['workspace'] / 'train.py').read_text('utf-8')
    options = discover_native_selections(files(source))
    scope = next(option['scope'] for option in options if 'loop_counter' in option['scope'])
    assert scope['function'] == 'run_all' and scope['loop_counter'] == '_'
    assert scope['selection']['validation_metric'] == {'name': 'eval_info', 'keys': ['val_loss']}
    assert 'result_unit' not in scope['selection']  # Units remain an evidence judgement.


@pytest.mark.parametrize('mutation', [
    lambda s: s.replace('for idx in range(3):', 'while True:'),
    lambda s: s.replace('if val_loss < best_val_loss:', 'if compute_metric() < best_val_loss:'),
    lambda s: s.replace('    results.append(selected_test)',
                       '    if val_loss < best_val_loss:\n        selected_test = 0.0\n    results.append(selected_test)'),
])
def test_unsupported_or_ambiguous_source_does_not_advertise_capture(mutation):
    assert discover_native_selections(files(mutation(SOURCE))) == []


def test_a_nested_function_guard_is_not_a_selection_in_the_outer_loop():
    source = '''for idx in range(3):
    def nested():
        if val_loss < best_val_loss:
            selected_test = output
    nested()
'''
    assert discover_native_selections(files(source)) == []
    with pytest.raises(ValueError, match='one original source comparison'):
        resolve_module_loop(source, SCOPE)


@pytest.mark.parametrize('iterator,offset', [('range(epochs)', 1), ('range(0, epochs)', 1),
    ('range(1, epochs + 1)', 0), ('range(1, epochs + 1, 1)', 0),
    ('range(start, epochs)', None), ('range(1, epochs, 2)', None), ('custom(epochs)', None)])
def test_epoch_count_offset_comes_from_original_range(iterator, offset):
    tree = ast.parse('for epoch in ' + iterator + ':\n    train()\n')
    assert _source_epoch_offset(tree, 'epoch') == offset
    assert _source_epoch_offset(tree, 'missing') is None


def test_ambiguous_or_nested_function_epoch_counters_are_not_guessed():
    assert _source_epoch_offset(ast.parse('for epoch in range(epochs):\n    pass\n'
                                         'for epoch in range(epochs):\n    pass\n'), 'epoch') is None
    assert _source_epoch_offset(ast.parse('def nested():\n    for epoch in range(epochs):\n        pass\n'), 'epoch') is None


def test_protocol_view_recomputes_options_instead_of_trusting_input_flags(tmp_path):
    from src.discovered_repository_plan import _planning_view, build_evidence_packet
    fixture = series_fixture(tmp_path)
    use_function_loop(fixture)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['program_detected_native_selections'] = [{'invented': True}]
    view = _planning_view(packet)
    assert any('loop_counter' in item['scope'] for item in view['program_detected_native_selections'])
    assert all('invented' not in item for item in view['program_detected_native_selections'])
    assert packet['program_detected_native_selections'] == [{'invented': True}]
