import hashlib
from copy import deepcopy
import pytest

from src.author_unlabelled_metrics import numeric_summaries, validate_summary_labels


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
