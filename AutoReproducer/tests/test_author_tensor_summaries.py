"""Source-bound scalar summaries work across author tasks and device reprs."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.author_unlabelled_metrics import (
    interpret_numeric_summaries,
    numeric_summaries,
    validate_summary_labels,
)


CLASSIFICATION_SOURCE = '''import torch
accs = []
for trial in range(50):
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print(acc)
accs = torch.stack(accs)
print(accs.mean())
print(accs.std())
'''
PRINTS = 'print(accs.mean())\nprint(accs.std())'
OUTPUT = "tensor(82.2380, device='cuda:0')\ntensor(0.1451, device='cuda:0')"


def _case(tmp_path, *, source=CLASSIFICATION_SOURCE, output=OUTPUT,
          metric='accuracy', unit='percent', metric_quote=None, unit_quote=None):
    path = tmp_path / 'experiment.py'
    path.write_text(source, encoding='utf-8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    execution = {'success': True, 'steps': [{'id': 'author', 'success': True,
        'exit_code': 0, 'stdout': 'Training completed\n' + output + '\n', 'stderr': ''}]}
    candidates = numeric_summaries(execution)
    assert len(candidates) == 1
    packet = {'pdf': {'pages': []}, 'file_inventory': {}, 'repository': {
        'revision': 'd' * 40, 'files': {'experiment.py': {'text': source, 'sha256': digest}}}}
    citation = {'id': 'source', 'source': 'repository', 'path': 'experiment.py',
        'sha256': digest, 'revision': 'd' * 40, 'quote': source}
    item = {'candidate_id': candidates[0]['id'], 'column': 0, 'std_column': 1,
        'name': metric, 'unit': unit, 'split': 'test', 'aggregation': 'mean', 'runs': None,
        'citations': ['source'], 'print_quote': PRINTS, 'order_quote': PRINTS,
        'metric_quote': metric_quote or
            'acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]',
        'aggregation_quote': PRINTS}
    if unit_quote is not None or metric == 'accuracy':
        item['unit_quote'] = unit_quote if unit_quote is not None else 'accs.append(acc * 100)'
    return {'payload': {'metrics': [item], 'citations': [citation]},
        'candidates': candidates, 'packet': packet, 'root': tmp_path, 'execution': execution}


def _validate(case):
    return validate_summary_labels(case['payload'], case['candidates'], case['packet'], case['root'])


def _replace_source(case, old, new):
    path = case['root'] / 'experiment.py'
    text = path.read_text(encoding='utf-8').replace(old, new)
    path.write_text(text, encoding='utf-8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    case['packet']['repository']['files']['experiment.py'] = {'text': text, 'sha256': digest}
    case['payload']['citations'][0].update(quote=text, sha256=digest)
    for key, value in list(case['payload']['metrics'][0].items()):
        if isinstance(value, str):
            case['payload']['metrics'][0][key] = value.replace(old, new)


def test_percentage_classification_uses_original_append_stack_and_both_output_lines(tmp_path):
    case = _case(tmp_path)
    before = deepcopy((case['payload'], case['execution'], case['candidates']))
    row, = _validate(case)
    assert (row['name'], row['value'], row['std'], row['unit']) == ('accuracy', 82.238, 0.1451, 'percent')
    assert row['runs'] is None and row['runs_source'] == {'kind': 'unknown'}
    assert row['source_quote'] == OUTPUT
    assert row['output_sources'] == [
        {'source': 'author:stdout:2', 'source_quote': OUTPUT.splitlines()[0]},
        {'source': 'author:stdout:3', 'source_quote': OUTPUT.splitlines()[1]}]
    assert row['summary_source']['unit_source']['expression'] == 'acc * 100'
    assert row['summary_source']['std_print_line'] == row['summary_source']['print_line'] + 1
    assert (case['payload'], case['execution'], case['candidates']) == before


def test_final_tensor_summary_takes_priority_over_earlier_bare_training_numbers():
    execution = {'success': True, 'steps': [{'id': 'author', 'success': True,
        'exit_code': 0, 'stdout': '1.0 2.0\nTraining done\n' + OUTPUT}]}
    candidate, = numeric_summaries(execution)
    assert candidate['summary_format'] == 'tensor_pair'
    assert candidate['values'] == [82.238, 0.1451]


def test_same_variable_name_does_not_join_statistics_from_different_array_versions(tmp_path):
    case = _case(tmp_path)
    _replace_source(case, PRINTS,
        'mean = accs.mean()\naccs = other\nstd = accs.std()\nprint(mean)\nprint(std)')
    case['payload']['metrics'][0]['print_quote'] = 'print(mean)\nprint(std)'
    with pytest.raises(ValueError, match='directly'):
        _validate(case)


def test_indented_multiline_metric_quote_is_parsed_without_rewriting_original_evidence(tmp_path):
    case = _case(tmp_path)
    quote = ('acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]\n'
             '    accs.append(acc * 100)')
    case['payload']['metrics'][0]['metric_quote'] = quote
    row, = _validate(case)
    original = (tmp_path / 'experiment.py').read_bytes().decode('utf-8')
    expected = quote.replace('\n', '\r\n') if '\r\n' in original else quote
    assert expected in original
    assert row['interpretation_evidence']['metric_quote'] == expected
    assert row['interpretation_locations']['metric_quote']['quote'] == expected
    assert row['value'] == 82.238


def test_regression_scalar_mean_std_use_same_generic_source_binding(tmp_path):
    expression = 'mse = ((predictions - test_targets) ** 2).mean()'
    source = CLASSIFICATION_SOURCE.replace(
        'acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]', expression
    ).replace('accs.append(acc * 100)', 'accs.append(mse)').replace('print(acc)', 'print(mse)')
    case = _case(tmp_path, source=source, metric='mse', unit='scalar', metric_quote=expression,
        output='tensor(1.234e-02, dtype=torch.float64)\ntensor(0.001, dtype=torch.float64)')
    row, = _validate(case)
    assert (row['name'], row['value'], row['std'], row['unit']) == ('mse', 0.01234, 0.001, 'scalar')
    assert row['runs'] is None
    assert len(row['output_sources']) == 2


@pytest.mark.parametrize('metadata', [
    '', ", device='cpu'", ', dtype=torch.float64', ", device='cuda:12', dtype=torch.float32",
    ', requires_grad=True, grad_fn=<MeanBackward0>',
])
def test_scalar_device_and_dtype_numbers_never_become_measurement_columns(metadata):
    output = f'tensor(82.2380{metadata})\ntensor(0.1451{metadata})'
    execution = {'success': True, 'steps': [{'id': 'x', 'success': True, 'exit_code': 0, 'stdout': output}]}
    candidate, = numeric_summaries(execution)
    assert candidate['values'] == [82.238, 0.1451]
    assert len(candidate['output_sources']) == 2


@pytest.mark.parametrize('output', [
    'tensor([82.2380])\ntensor(0.1451)',
    'tensor(82.2380)\ntensor([0.1451])',
    'tensor(82.2380, 0.1451)\ntensor(0.1451)',
    'tensor(nan)\ntensor(0.1451)',
    'tensor(82.2380)\ntensor(inf)',
    'tensor(82.2380) trailing\ntensor(0.1451)',
    'tensor(82.2380)\nnot a scalar\ntensor(0.1451)',
    'tensor(82.2380)\n',
    'tensor(82.2380, device=\'cuda:0")\ntensor(0.1451, device=\'cuda:0")',
    'tensor(82.2380, device="cuda:0\')\ntensor(0.1451, device="cuda:0\')',
])
def test_non_scalar_damaged_or_nonconsecutive_output_is_not_a_candidate(output):
    execution = {'success': True, 'steps': [{'id': 'x', 'success': True, 'exit_code': 0, 'stdout': output}]}
    assert numeric_summaries(execution) == []


@pytest.mark.parametrize('suffix', [
    '\ntensor(nan)', '\ntensor(inf)', '\ntensor([0.2])', '\nRuntimeError: failed',
])
def test_incomplete_or_failed_final_output_cannot_reuse_an_earlier_valid_pair(suffix):
    execution = {'success': True, 'steps': [{'id': 'x', 'success': True, 'exit_code': 0,
        'stdout': OUTPUT + suffix}]}
    assert numeric_summaries(execution) == []


@pytest.mark.parametrize('change', [
    {'runs': 50}, {'runs': False}, {'column': 1, 'std_column': 0}, {'std_column': None},
    {'aggregation': 'single'}, {'aggregation': 'best'}, {'value': 99.0}, {'name': 'loss'},
    {'print_quote': 'print(accs.mean())'},
])
def test_model_cannot_invent_runs_values_order_or_metric(tmp_path, change):
    case = _case(tmp_path)
    case['payload']['metrics'][0].update(change)
    with pytest.raises(ValueError):
        _validate(case)


@pytest.mark.parametrize('old,new', [
    ('print(accs.std())', 'print(other.std())'),
    ('print(accs.std())', 'print(accs.mean())'),
    (PRINTS, 'print(accs.std())\nprint(accs.mean())'),
    (PRINTS, 'print(accs.mean())\naccs = other\nprint(accs.std())'),
    (PRINTS, 'for epoch in range(10):\n    print(accs.mean())\n    print(accs.std())'),
])
def test_reductions_must_be_consecutive_complete_same_values_outside_training_loop(tmp_path, old, new):
    case = _case(tmp_path)
    _replace_source(case, old, new)
    with pytest.raises(ValueError):
        _validate(case)


def test_standard_module_main_guard_keeps_complete_source_bound_statistics(tmp_path):
    case = _case(tmp_path)
    _replace_source(case, PRINTS,
        "if __name__ == '__main__':\n    print(accs.mean())\n    print(accs.std())")
    row, = _validate(case)
    assert (row['value'], row['std'], row['runs']) == (82.238, 0.1451, None)
    assert row['unit'] in {'percent', ''}
    assert row['source_quote'] == OUTPUT
    assert len(row['output_sources']) == 2
    assert row['summary_source']['std_print_line'] == row['summary_source']['print_line'] + 1


@pytest.mark.parametrize('branch', [
    'if ready:\n    print(accs.mean())\n    print(accs.std())',
    'if ready:\n    pass\nelse:\n    print(accs.mean())\n    print(accs.std())',
    "if __name__ == '__main__':\n    pass\nelse:\n    print(accs.mean())\n    print(accs.std())",
])
def test_ordinary_conditional_or_main_guard_else_cannot_prove_terminal_statistics(tmp_path, branch):
    case = _case(tmp_path)
    _replace_source(case, PRINTS, branch)
    with pytest.raises(ValueError):
        _validate(case)


@pytest.mark.parametrize('old,new', [
    ('accs.append(acc * 100)', 'unrelated = acc * 100\n    accs.append(acc)'),
    ('accs = torch.stack(accs)', 'accs = torch.stack(accs)\naccs /= 100'),
    ('accs = torch.stack(accs)', 'accs = torch.stack(accs)\naccs.mul_(0.01)'),
    ('accs = torch.stack(accs)', 'accs = torch.stack(accs)\nmutate(accs)'),
    ('accs = torch.stack(accs)', 'accs = torch.stack(accs, dim=0)'),
])
def test_unconnected_scaling_or_mutation_keeps_measured_values_with_unknown_unit(tmp_path, old, new):
    case = _case(tmp_path)
    _replace_source(case, old, new)
    if new.startswith('unrelated'):
        case['payload']['metrics'][0]['unit_quote'] = 'unrelated = acc * 100'
    row, = _validate(case)
    assert (row['value'], row['std'], row['unit']) == (82.238, 0.1451, '')
    assert row['summary_source']['unit_source']['kind'] == 'unknown'


@pytest.mark.parametrize('unit_evidence', ['scale_quote', 'omitted', 'unrelated'])
def test_small_percentages_cannot_be_relabelled_as_fraction_without_unit_source(tmp_path, unit_evidence):
    case = _case(tmp_path, output='tensor(0.8238)\ntensor(0.001)', unit='fraction')
    if unit_evidence == 'omitted':
        case['payload']['metrics'][0].pop('unit_quote')
    elif unit_evidence == 'unrelated':
        case['payload']['metrics'][0]['unit_quote'] = 'accs = torch.stack(accs)'
    try:
        rows = _validate(case)
    except ValueError:
        return
    assert rows[0]['unit'] == ''
    assert rows[0]['value'] == 0.8238


def test_rebound_array_library_cannot_prove_percentage(tmp_path):
    case = _case(tmp_path)
    _replace_source(case, 'accs = torch.stack(accs)', 'torch = unknown_library\naccs = torch.stack(accs)')
    row, = _validate(case)
    assert row['unit'] == ''
    assert row['summary_source']['unit_source']['kind'] == 'unknown'


def test_known_owned_observer_tail_preserves_original_scalar_lines(tmp_path):
    case = _case(tmp_path)
    receipt = {'author_command_finished': 'author', 'optimizer_steps': 5352}
    case['execution']['steps'][0]['stdout'] += json.dumps(receipt) + '\n'
    candidates = numeric_summaries(case['execution'])
    assert candidates == case['candidates']
    case['candidates'] = candidates
    assert _validate(case)[0]['source_quote'] == OUTPUT


@pytest.mark.parametrize('receipt', [
    {'author_command_finished': 'other', 'optimizer_steps': 5352},
    {'author_command_finished': 'author', 'optimizer_steps': True},
    {'author_command_finished': 'author', 'optimizer_steps': -1},
    {'author_command_finished': 'author', 'optimizer_steps': 5352, 'success': True},
])
def test_unknown_or_mismatched_json_tail_does_not_hide_later_output(receipt):
    execution = {'success': True, 'steps': [{'id': 'author', 'success': True, 'exit_code': 0,
        'stdout': OUTPUT + '\n' + json.dumps(receipt)}]}
    assert numeric_summaries(execution) == []


@pytest.mark.parametrize('change', ['failed', 'nonzero', 'metrics', 'metric_records', 'marker'])
def test_failed_execution_or_authoritative_metrics_disable_fallback(change):
    step = {'id': 'x', 'success': True, 'exit_code': 0, 'stdout': OUTPUT}
    execution = {'success': True, 'steps': [step]}
    if change == 'failed':
        execution['success'] = False
    elif change == 'nonzero':
        step['exit_code'] = 1
    elif change == 'marker':
        step['stdout'] += '\nAUTOREPRO_METRICS:{"accuracy": 0.8}\n'
    else:
        step[change] = []
    assert numeric_summaries(execution) == []


def test_stdout_source_does_not_establish_stderr_and_changed_source(tmp_path):
    case = _case(tmp_path)
    case['candidates'][0]['stream'] = 'stderr'
    with pytest.raises(ValueError, match='stream'):
        _validate(case)
    case['candidates'][0]['stream'] = 'stdout'
    (tmp_path / 'experiment.py').write_text('print("changed")', encoding='utf-8')
    with pytest.raises(ValueError):
        _validate(case)


def test_production_interpreter_retains_source_bound_pair_without_rewriting_logs(tmp_path):
    case = _case(tmp_path)

    class SourceLLM:
        mock_mode = False

        def chat(self, prompt, **kwargs):
            assert kwargs['task'] == 'author_numeric_summary'
            assert json.loads(prompt)['outputs'][0]['source_quote'] == OUTPUT
            return json.dumps(case['payload'])

    before = deepcopy(case['execution'])
    records, audit = interpret_numeric_summaries(SourceLLM(), case['packet'], {'steps': []},
        case['root'], case['execution'])
    assert audit['status'] == 'completed'
    assert records[0]['value'] == 82.238
    assert case['execution'] == before
