import hashlib
import json
from copy import deepcopy
import pytest

from src.author_unlabelled_metrics import interpret_numeric_summaries, numeric_summaries, validate_summary_labels


@pytest.fixture
def labelled(tmp_path):
    text = ('def run():\n    return valid_score, test_score\n'
            'def repeat():\n    return scores.mean(axis=0), scores.std(axis=0)\n'
            'def accuracy(pred, y):\n    return (pred == y).sum() / len(y)\n'
            'print(" ".join(str(s) for s in means + stds))\n')
    (tmp_path / 'train.py').write_text(text, encoding='utf-8')
    sha = hashlib.sha256((tmp_path / 'train.py').read_bytes()).hexdigest()
    packet = {'pdf': {'pages': []}, 'file_inventory': {},
              'repository': {'revision': 'a' * 40, 'files': {'train.py': {'text': text, 'sha256': sha}}}}
    citation = {'id': 'source', 'source': 'repository', 'path': 'train.py', 'sha256': sha,
                'revision': 'a' * 40, 'quote': text}
    execution = {'success': True, 'steps': [{'id': 'student', 'success': True, 'exit_code': 0,
                 'stdout': 'training done\n0.8600 0.8130 0.0200 0.0100\n'}]}
    candidates = numeric_summaries(execution)
    item = {'candidate_id': candidates[0]['id'], 'column': 1, 'std_column': 3,
            'name': 'accuracy', 'unit': 'fraction', 'split': 'test', 'aggregation': 'mean', 'runs': 10,
            'citations': ['source'], 'print_quote': 'print(" ".join(str(s) for s in means + stds))',
            'order_quote': 'return valid_score, test_score',
            'metric_quote': 'return (pred == y).sum() / len(y)',
            'aggregation_quote': 'return scores.mean(axis=0), scores.std(axis=0)'}
    return {'metrics': [item], 'citations': [citation]}, candidates, packet, tmp_path


def test_summary_values_and_std_are_selected_from_actual_output(labelled):
    records = validate_summary_labels(*labelled)
    assert records[0]['value'] == 0.813 and records[0]['std'] == 0.01
    assert records[0]['split'] == 'test'
    assert records[0]['stage'] == 'test'
    assert records[0]['interpretation_source'] == 'ai_with_frozen_source_citations'


@pytest.mark.parametrize('change', [
    {'value': 0.99}, {'column': 999}, {'column': True}, {'std_column': 1},
    {'std_column': -1}, {'metric_quote': 'invented metric implementation'},
    {'runs': False}, {'citations': ['made_up']}, {'name': 'made_up_score'},
])
def test_model_cannot_invent_measurements_or_source_quotes(labelled, change):
    payload, candidates, packet, root = labelled
    payload['metrics'][0].update(change)
    with pytest.raises(ValueError):
        validate_summary_labels(payload, candidates, packet, root)


def test_changed_author_source_invalidates_summary_mapping(labelled):
    payload, candidates, packet, root = labelled
    (root / 'train.py').write_text('print("different")')
    with pytest.raises(ValueError):
        validate_summary_labels(payload, candidates, packet, root)


def test_source_quote_recovers_model_escaped_newlines_and_wrong_excerpt_of_same_file(labelled):
    payload, candidates, packet, root = labelled
    payload['citations'][0]['quote'] = 'def run():\n    return valid_score, test_score'
    payload['metrics'][0]['metric_quote'] = 'def accuracy(pred, y):\\n        return (pred == y).sum() / len(y)'
    records = validate_summary_labels(payload, candidates, packet, root)
    assert records[0]['value'] == 0.813
    assert records[0]['interpretation_locations']['metric_quote']['path'] == 'train.py'
    assert '\\n' not in records[0]['interpretation_evidence']['metric_quote']


def test_failed_or_interrupted_run_never_supplies_numeric_summary():
    execution = {'success': False, 'steps': [{'id': 'x', 'success': True, 'exit_code': 0, 'stdout': '0.8 0.1'}]}
    assert numeric_summaries(execution) == []
    execution.update(success=True)
    execution['steps'][0].update(success=False, exit_code=1)
    assert numeric_summaries(execution) == []


def test_original_indentation_disambiguates_repeated_metric_implementation(tmp_path):
    from src.author_unlabelled_metrics import _original_quote
    text = ('def outer():\n    def metric(x, y):\n        return x.eq(y).float().mean()\n'
            'def metric(x, y):\n    return x.eq(y).float().mean()\n')
    (tmp_path / 'metrics.py').write_text(text, encoding='utf-8')
    sources = [{'path': 'metrics.py', 'sha256': hashlib.sha256((tmp_path / 'metrics.py').read_bytes()).hexdigest()}]
    quote = 'def metric(x, y):\\n    return x.eq(y).float().mean()'
    location = _original_quote(quote, sources, tmp_path)
    assert location['line'] == 4
    assert location['quote'] in (tmp_path / 'metrics.py').read_bytes().decode('utf-8')
    # Whitespace recovery must not guess between indistinguishable locations.
    assert _original_quote('def metric(x, y):\\n  return x.eq(y).float().mean()', sources, tmp_path) is None


FINAL_SOURCE = '''import numpy as np
def evaluate(pred, labels):
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc
def train(args, split):
    return evaluate(pred, labels)
n_splits = 10
best_results = []
for split in range(n_splits):
    best_results.append(train(args, split))
best_results = np.array(best_results)
mean_acc = np.mean(best_results)
std = np.std(best_results)

log = 'Final test results -- mean: {:.4f}, std: {:.4f}'
print(log.format(mean_acc,std))
'''
FINAL_OUTPUT = 'Final test results -- mean: 0.8549, std: 0.0374'


@pytest.fixture
def final_summary(tmp_path):
    source = tmp_path / 'train.py'
    source.write_text(FINAL_SOURCE, encoding='utf-8')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    packet = {'pdf': {'pages': []}, 'file_inventory': {},
              'repository': {'revision': 'b' * 40,
                             'files': {'train.py': {'text': FINAL_SOURCE, 'sha256': sha}}}}
    citation = {'id': 'source', 'source': 'repository', 'path': 'train.py',
                'sha256': sha, 'revision': 'b' * 40, 'quote': FINAL_SOURCE}
    execution = {'success': True, 'steps': [{'id': 'author', 'success': True, 'exit_code': 0,
                 'stdout': 'Split: 9, Epoch: 198, Train: 1.0, Val: 0.8, Test: 0.84\n' + FINAL_OUTPUT + '\n'}]}
    candidates = numeric_summaries(execution)
    assert len(candidates) == 1
    item = {'candidate_id': candidates[0]['id'], 'column': 0, 'std_column': 1,
            'name': 'accuracy', 'unit': 'fraction', 'split': 'test', 'aggregation': 'mean',
            'runs': None, 'citations': ['source'],
            'print_quote': "log = 'Final test results -- mean: {:.4f}, std: {:.4f}'\nprint(log.format(mean_acc,std))",
            'order_quote': 'n_splits = 10\nbest_results = []\nfor split in range(n_splits):\n    best_results.append(train(args, split))',
            'metric_quote': 'acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()',
            'aggregation_quote': 'mean_acc = np.mean(best_results)\nstd = np.std(best_results)'}
    return {'metrics': [item], 'citations': [citation]}, candidates, packet, tmp_path


def test_final_named_statistics_preserve_observed_values_and_unknown_runs(final_summary):
    records = validate_summary_labels(*final_summary)
    record = records[0]
    assert (record['name'], record['value'], record['std'], record['aggregation']) == (
        'accuracy', 0.8549, 0.0374, 'mean')
    assert record['runs'] is None
    assert record['source_quote'] == FINAL_OUTPUT
    assert record['interpretation_source'] == 'ai_with_frozen_source_citations'
    assert record['summary_source']['source_sha256'] == final_summary[0]['citations'][0]['sha256']


@pytest.mark.parametrize('change', [
    {'name': 'loss'}, {'column': 1, 'std_column': 0}, {'std_column': None},
    {'split': 'validation'}, {'aggregation': 'single'}, {'aggregation': 'best'},
    {'runs': 10}, {'value': 0.99}, {'print_quote': 'return acc'},
    {'aggregation_quote': 'best_results = np.array(best_results)'},
])
def test_final_statistics_cannot_be_relabelled_or_gain_unreported_runs(final_summary, change):
    payload, candidates, packet, root = final_summary
    payload['metrics'][0].update(change)
    with pytest.raises(ValueError):
        validate_summary_labels(payload, candidates, packet, root)


@pytest.mark.parametrize('output', [
    'Final test results -- mean: nan, std: 0.0374',
    'Final test results -- mean: 0.8549, std: inf',
    'Final test results -- mean: 0.8549, std: -0.0374',
    'Final test results -- mean: 0.8549, std: 0.0374 truncated',
    'Epoch 10 Final test results -- mean: 0.8549, std: 0.0374',
    FINAL_OUTPUT + '\nFinal test results -- mean: 0.9000, std: 0.0100',
])
def test_damaged_intermediate_or_ambiguous_final_statistics_are_not_candidates(output):
    execution = {'success': True, 'steps': [{'id': 'x', 'success': True, 'exit_code': 0, 'stdout': output}]}
    assert numeric_summaries(execution) == []


def test_validation_summary_cannot_be_claimed_as_test(final_summary):
    payload, _, packet, root = final_summary
    output = FINAL_OUTPUT.replace('test', 'validation')
    execution = {'success': True, 'steps': [{'id': 'author', 'success': True, 'exit_code': 0, 'stdout': output}]}
    candidates = numeric_summaries(execution)
    assert len(candidates) == 1
    payload['metrics'][0]['candidate_id'] = candidates[0]['id']
    with pytest.raises(ValueError, match='split'):
        validate_summary_labels(payload, candidates, packet, root)


def test_same_final_test_format_does_not_turn_loss_into_accuracy(final_summary):
    payload, candidates, packet, root = final_summary
    text = FINAL_SOURCE.replace('acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()',
                                'loss = loss_function(pred, labels)').replace('return acc', 'return loss')
    path = root / 'train.py'
    path.write_text(text, encoding='utf-8')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    packet['repository']['files']['train.py'] = {'text': text, 'sha256': sha}
    payload['citations'][0].update(quote=text, sha256=sha)
    payload['metrics'][0]['metric_quote'] = 'loss = loss_function(pred, labels)'
    with pytest.raises(ValueError, match='metric'):
        validate_summary_labels(payload, candidates, packet, root)
    payload['metrics'][0].update(name='loss', unit='scalar')
    assert validate_summary_labels(payload, candidates, packet, root)[0]['name'] == 'loss'


@pytest.mark.parametrize('old,new', [
    ('np.std(best_results)', 'np.mean(best_results)'),
    ('np.std(best_results)', 'np.std(other_results)'),
    ('print(log.format(mean_acc,std))', 'print(log.format(std,mean_acc))'),
    ('std = np.std(best_results)', 'std = np.std(best_results)\nstd = np.std(best_results)'),
])
def test_matching_output_needs_the_original_unambiguous_mean_std_data_flow(final_summary, old, new):
    payload, candidates, packet, root = final_summary
    text = FINAL_SOURCE.replace(old, new)
    source = root / 'train.py'
    source.write_text(text, encoding='utf-8')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    packet['repository']['files']['train.py'] = {'text': text, 'sha256': sha}
    payload['citations'][0].update(quote=text, sha256=sha)
    for field in ('print_quote', 'aggregation_quote', 'order_quote', 'metric_quote'):
        payload['metrics'][0][field] = payload['metrics'][0][field].replace(old, new)
    with pytest.raises(ValueError):
        validate_summary_labels(payload, candidates, packet, root)


def test_final_summary_uses_the_production_source_bound_interpreter_without_changing_output(final_summary):
    payload, candidates, packet, root = final_summary

    class SourceLLM:
        mock_mode = False

        def chat(self, prompt, **kwargs):
            assert kwargs['task'] == 'author_numeric_summary'
            observed = json.loads(prompt)['outputs'][0]
            assert observed['values'] == [0.8549, 0.0374]
            assert observed['labelled_statistics']['runs'] is None
            return json.dumps(payload)

    execution = {'success': True, 'steps': [{'id': 'author', 'success': True, 'exit_code': 0,
                 'stdout': FINAL_OUTPUT + '\n'}]}
    payload['metrics'][0]['candidate_id'] = numeric_summaries(execution)[0]['id']
    before = deepcopy(execution)
    records, audit = interpret_numeric_summaries(SourceLLM(), packet, {'steps': []}, root, execution)
    assert audit['status'] == 'completed' and records[0]['value'] == 0.8549
    assert execution == before


def test_stdout_print_cannot_prove_a_same_shaped_stderr_summary(final_summary):
    payload, _, packet, root = final_summary
    execution = {'success': True, 'steps': [{'id': 'author', 'success': True, 'exit_code': 0,
                 'stdout': '', 'stderr': FINAL_OUTPUT}]}
    candidates = numeric_summaries(execution)
    payload['metrics'][0]['candidate_id'] = candidates[0]['id']
    with pytest.raises(ValueError, match='stream'):
        validate_summary_labels(payload, candidates, packet, root)


LOGGER_SOURCE = '''import torch
def eval_acc(pred, labels):
    correct = pred == labels
    return correct.sum() / len(labels)
def print_statistics(self):
    result = 100 * torch.tensor(self.results)
    best_results = []
    for r in result:
        test = r[r[:, 1].argmax(), 2].item()
        best_results.append((test, test))
    best_result = torch.tensor(best_results)
    r = best_result[:, 1]
    print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
'''
LOGGER_OUTPUT = ("Namespace(dataset='invented', runs=2, rocauc=False)\n"
                 'Epoch: 475, Loss: 0.0000, Train: 100.00%, Valid: 48.75%, Test: 99.99%\n'
                 'Run 01:\nFinal Test: 51.95\nRun 02:\nFinal Test: 63.85\n'
                 'All runs:\nHighest Train: 100.00 ± 0.00\nHighest Valid: 99.99 ± 1.00\n'
                 '  Final Train: 84.58 ± 16.72\n   Final Test: 57.90 ± 8.41\n')


@pytest.fixture
def logger_summary(tmp_path):
    (tmp_path / 'logger.py').write_text(LOGGER_SOURCE, encoding='utf-8')
    sha = hashlib.sha256((tmp_path / 'logger.py').read_bytes()).hexdigest()
    packet = {'pdf': {'pages': []}, 'file_inventory': {}, 'repository': {'revision': 'c' * 40,
              'files': {'logger.py': {'text': LOGGER_SOURCE, 'sha256': sha}}}}
    execution = {'success': True, 'steps': [{'id': 'author', 'success': True, 'exit_code': 0,
                                          'stdout': LOGGER_OUTPUT}]}
    candidates = numeric_summaries(execution)
    assert len(candidates) == 1
    item = {'candidate_id': candidates[0]['id'], 'column': 0, 'std_column': 1,
            'name': 'accuracy', 'unit': 'percent', 'split': 'test', 'aggregation': 'mean', 'runs': 2,
            'citations': ['source'], 'print_quote': "print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')",
            'order_quote': 'test = r[r[:, 1].argmax(), 2].item()',
            'metric_quote': 'def eval_acc(pred, labels):\n    correct = pred == labels\n    return correct.sum() / len(labels)',
            'aggregation_quote': "r = best_result[:, 1]\n    print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')",
            'unit_quote': 'result = 100 * torch.tensor(self.results)'}
    citation = {'id': 'source', 'source': 'repository', 'path': 'logger.py', 'sha256': sha,
                'revision': 'c' * 40, 'quote': LOGGER_SOURCE}
    return {'metrics': [item], 'citations': [citation]}, candidates, packet, tmp_path


def test_all_runs_summary_selects_actual_final_aggregate_with_source_bound_units(logger_summary):
    payload, candidates, packet, root = logger_summary
    before = deepcopy((payload, candidates))
    row = validate_summary_labels(payload, candidates, packet, root)[0]
    assert (row['name'], row['value'], row['std'], row['unit'], row['runs']) == (
        'accuracy', 57.90, 8.41, 'percent', 2)
    assert row['runs_source']['kind'] == 'completed_logger_runs'
    assert row['summary_source']['unit_source']['expression'] == '100 * torch.tensor(self.results)'
    assert (payload, candidates) == before


@pytest.mark.parametrize('printed', ['57.90% ± 8.41', '57.90 ± 8.41%', '57.90% ± 8.41%'])
def test_logger_summary_accepts_explicit_percent_on_either_statistic(printed):
    execution = {'success': True, 'steps': [{'id': 'x', 'success': True, 'exit_code': 0,
        'stdout': LOGGER_OUTPUT.replace('57.90 ± 8.41', printed)}]}
    candidate = numeric_summaries(execution)[0]
    assert candidate['values'] == [57.9, 8.41]
    assert candidate['labelled_statistics']['unit'] == 'percent'


@pytest.mark.parametrize('change', [
    {'unit': 'fraction'},
    {'runs': 10}, {'aggregation': 'best'}, {'name': 'loss'}, {'std_column': None},
])
def test_logger_summary_cannot_invent_units_statistics_or_metric_identity(logger_summary, change):
    payload, candidates, packet, root = logger_summary
    payload['metrics'][0].update(change)
    with pytest.raises(ValueError):
        validate_summary_labels(payload, candidates, packet, root)


@pytest.mark.parametrize('old,new', [
    ('r.std()', 'other.std()'), ('r.std()', 'r.mean()'),
    ('{r.mean():.2f}', '{r.mean():.{precision}f}'),
])
def test_logger_mean_std_and_scaling_must_feed_the_same_original_values(logger_summary, old, new):
    payload, candidates, packet, root = logger_summary
    text = LOGGER_SOURCE.replace(old, new)
    (root / 'logger.py').write_text(text, encoding='utf-8')
    sha = hashlib.sha256((root / 'logger.py').read_bytes()).hexdigest()
    packet['repository']['files']['logger.py'] = {'text': text, 'sha256': sha}
    payload['citations'][0].update(quote=text, sha256=sha)
    for field in ('print_quote', 'order_quote', 'metric_quote', 'aggregation_quote', 'unit_quote'):
        payload['metrics'][0][field] = payload['metrics'][0][field].replace(old, new)
    with pytest.raises(ValueError):
        validate_summary_labels(payload, candidates, packet, root)


@pytest.mark.parametrize('change', [{'unit_quote': 'r = best_result[:, 1]'}, {'unit_quote': ''},
                                  {'unit': '', 'unit_quote': ''}])
def test_unproved_logger_unit_retains_the_final_measured_values(logger_summary, change):
    payload, candidates, packet, root = logger_summary
    payload['metrics'][0].update(change)
    row = validate_summary_labels(payload, candidates, packet, root)[0]
    assert row['unit'] == '' and row['value'] == 57.9 and row['std'] == 8.41
    assert row['summary_source']['unit_source']['kind'] == 'unknown'


@pytest.mark.parametrize('old,new', [
    ('best_results.append((test, test))', 'best_results.append((test, loss))'),
    ('result = 100 * torch.tensor(self.results)',
     'unrelated = 100 * torch.tensor(self.results)\n    result = torch.tensor(self.results)'),
    ('result = 100 * torch.tensor(self.results)',
     'result = torch.tensor(self.results)\n    if flag:\n        result = 100 * torch.tensor(self.results)'),
    ('best_results = []', 'result /= 100\n    best_results = []'),
    ('best_results = []', 'result[0] = 0\n    best_results = []'),
    ('best_results = []', 'result.mul_(.01)\n    best_results = []'),
    ('best_results = []', 'mutate(result)\n    best_results = []'),
])
def test_unknown_mutation_or_conditional_scaling_cannot_prove_percentage(logger_summary, old, new):
    payload, candidates, packet, root = logger_summary
    text = LOGGER_SOURCE.replace(old, new)
    (root / 'logger.py').write_text(text, encoding='utf-8')
    sha = hashlib.sha256((root / 'logger.py').read_bytes()).hexdigest()
    packet['repository']['files']['logger.py'] = {'text': text, 'sha256': sha}
    payload['citations'][0].update(quote=text, sha256=sha)
    if 'unrelated =' in new:
        payload['metrics'][0]['unit_quote'] = 'unrelated = 100 * torch.tensor(self.results)'
    row = validate_summary_labels(payload, candidates, packet, root)[0]
    assert row['unit'] == '' and row['value'] == 57.9 and row['std'] == 8.41


def test_identical_scaling_text_at_an_unrelated_source_location_is_not_unit_proof(logger_summary):
    payload, candidates, packet, root = logger_summary
    text = LOGGER_SOURCE.replace('result = 100 * torch.tensor(self.results)',
        'unrelated = 100 * torch.tensor(self.results)\n    result = 100 * torch.tensor(self.results)')
    (root / 'logger.py').write_text(text, encoding='utf-8')
    sha = hashlib.sha256((root / 'logger.py').read_bytes()).hexdigest()
    packet['repository']['files']['logger.py'] = {'text': text, 'sha256': sha}
    payload['citations'][0].update(quote=text, sha256=sha)
    payload['metrics'][0]['unit_quote'] = 'unrelated = 100 * torch.tensor(self.results)'
    row = validate_summary_labels(payload, candidates, packet, root)[0]
    assert row['unit'] == ''


@pytest.mark.parametrize('change', [
    lambda text: text.replace('All runs:', 'Run 03:'),
    lambda text: text + 'All runs:\nFinal Test: 90.00 ± 1.00\n',
    lambda text: text + 'Final Test: 90.00 ± 1.00\n',
    lambda text: text.replace('57.90 ± 8.41', 'nan ± 8.41'),
    lambda text: text.replace('57.90 ± 8.41', '57.90 ± -1.00'),
    lambda text: text.replace('57.90 ± 8.41', '57.90 ± 8.41 (95% CI)'),
])
def test_single_trials_or_ambiguous_damaged_logger_aggregates_are_not_candidates(change):
    execution = {'success': True, 'steps': [{'id': 'x', 'success': True, 'exit_code': 0,
                                          'stdout': change(LOGGER_OUTPUT)}]}
    assert numeric_summaries(execution) == []


@pytest.mark.parametrize('replacement', ['', 'Run 01:', 'Run 03:'])
def test_missing_duplicate_or_noncontiguous_run_blocks_do_not_prove_runs(replacement):
    output = LOGGER_OUTPUT.replace('Run 02:', replacement)
    execution = {'success': True, 'steps': [{'id': 'x', 'success': True, 'exit_code': 0, 'stdout': output}]}
    candidate = numeric_summaries(execution)[0]
    assert candidate['values'] == [57.9, 8.41]
    assert candidate['labelled_statistics']['runs'] is None


@pytest.mark.parametrize('field', ['metrics', 'metric_records', 'marker'])
def test_authoritative_structured_output_does_not_supply_numeric_fallback(field):
    step = {'id': 'x', 'success': True, 'exit_code': 0, 'stdout': LOGGER_OUTPUT, 'stderr': LOGGER_OUTPUT}
    if field == 'marker':
        step['stdout'] += 'AUTOREPRO_METRICS:{"accuracy": 0.8}\n'
    else:
        step[field] = []
    assert numeric_summaries({'success': True, 'steps': [step]}) == []


def test_selected_candidates_only_are_sent_and_validated_by_interpreter(logger_summary):
    payload, candidates, packet, root = logger_summary

    class SourceLLM:
        mock_mode = False

        def __init__(self):
            self.calls = 0

        def chat(self, prompt, **kwargs):
            self.calls += 1
            assert kwargs['task'] == 'author_numeric_summary'
            assert json.loads(prompt)['outputs'] == candidates
            return json.dumps(payload)

    execution = {'success': False, 'steps': []}  # Explicit candidates are the caller's filtered evidence.
    llm = SourceLLM()
    rows, audit = interpret_numeric_summaries(llm, packet, {'steps': []}, root, execution, candidates=candidates)
    assert audit['status'] == 'completed' and rows[0]['value'] == 57.9 and llm.calls == 1
    assert interpret_numeric_summaries(llm, packet, {'steps': []}, root, execution, candidates=[])[0] == []
    assert llm.calls == 1
