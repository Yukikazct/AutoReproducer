"""A parsed training loss must not hide an unparsed final result."""
from copy import deepcopy
import json

import pytest

from discovered_fixtures import make_experiment
from src import author_unlabelled_metrics as summaries
from src.author_claim_comparison import collect_author_metrics, compare_author_claims
from src.discovered_repository_plan import build_evidence_packet


class NoClaimLLM:
    mock_mode = False
    api_key = ''

    def __init__(self):
        self.calls = []

    def chat(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        assert kwargs['task'] == 'author_claim_comparison'
        return json.dumps({'comparisons': [], 'reason': 'No reference supplied by this fixture.'})


def compare(tmp_path, monkeypatch, execution, interpret):
    fixture = make_experiment(tmp_path)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    monkeypatch.setattr(summaries, 'interpret_numeric_summaries', interpret)
    llm = NoClaimLLM()
    original = deepcopy((execution, fixture['proposal']))
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], execution)
    assert (execution, fixture['proposal']) == original
    return result, llm


def completed(stdout, **extra):
    return {'success': True, 'steps': [{'id': 'train', 'success': True, 'exit_code': 0,
                                      'stdout': stdout, 'stderr': '', **extra}]}


def test_final_numeric_summary_supplements_loss_without_changing_it(tmp_path, monkeypatch):
    execution = completed('Loss: 0.0\n0.8549 0.0374\n')
    original, _ = collect_author_metrics(execution)
    calls = []

    def interpret(llm, packet, plan, workspace, saved, *, candidates):
        calls.append(candidates)
        assert saved == execution
        assert len(candidates) == 1 and candidates[0]['source_quote'] == '0.8549 0.0374'
        source = candidates[0]['id']
        return [{'id': source + ':column:0', 'source': source, 'step_id': 'train',
                 'source_quote': candidates[0]['source_quote'], 'name': 'accuracy',
                 'value': 0.8549, 'std': 0.0374, 'unit': 'fraction', 'split': 'test',
                 'aggregation': 'mean', 'runs': 10}], {'status': 'completed'}

    result, llm = compare(tmp_path, monkeypatch, execution, interpret)
    assert len(calls) == 1
    assert result['metric_records'][:len(original)] == original
    assert [(row['name'], row['value']) for row in result['metric_records']] == [
        ('loss', 0.0), ('accuracy', 0.8549)]
    assert result['numeric_summary_interpretation']['status'] == 'completed'
    observed = json.loads(llm.calls[0][0].split('\nObserved metrics:\n', 1)[1].split(
        '\nOriginal sources:\n', 1)[0])
    assert observed == result['metric_records']


@pytest.mark.parametrize('structured', [[], {'accuracy': 0.9}, 'malformed'])
def test_explicit_metrics_stay_authoritative_over_numeric_logs(tmp_path, monkeypatch, structured):
    execution = completed('Loss: 0.0\n0.8549 0.0374\n', metric_records=structured)
    original, _ = collect_author_metrics(execution)

    def interpret(*args, **kwargs):
        pytest.fail('Explicit structured metrics must not be supplemented from raw logs')

    result, _ = compare(tmp_path, monkeypatch, execution, interpret)
    assert result['metric_records'] == original


@pytest.mark.parametrize('payload', ['[]', '{"accuracy": 0.9}', '{broken'])
def test_final_metric_marker_stays_authoritative(tmp_path, monkeypatch, payload):
    execution = completed('0.8549 0.0374\nAUTOREPRO_METRICS: ' + payload)
    original, _ = collect_author_metrics(execution)

    def interpret(*args, **kwargs):
        pytest.fail('A final structured marker must not fall back to numeric logs')

    result, _ = compare(tmp_path, monkeypatch, execution, interpret)
    assert result['metric_records'] == original


def test_already_parsed_source_is_not_interpreted_again(tmp_path, monkeypatch):
    execution = completed('Test Accuracy: 0.8549')
    original, _ = collect_author_metrics(execution)
    monkeypatch.setattr(summaries, 'numeric_summaries', lambda saved: [{
        'id': original[0]['source'], 'step_id': 'train', 'stream': 'stdout',
        'source_quote': 'Test Accuracy: 0.8549', 'values': [0.8549, 0.0374]}])

    def interpret(*args, **kwargs):
        pytest.fail('An already parsed log source must not be interpreted twice')

    result, _ = compare(tmp_path, monkeypatch, execution, interpret)
    assert result['metric_records'] == original


def test_unresolved_summary_preserves_existing_measurement(tmp_path, monkeypatch):
    execution = completed('Loss: 0.0\n0.8549 0.0374\n')
    original, _ = collect_author_metrics(execution)

    def interpret(*args, **kwargs):
        return [], {'status': 'unavailable', 'attempts': ['unresolved metric meaning']}

    result, _ = compare(tmp_path, monkeypatch, execution, interpret)
    assert result['metric_records'] == original
    assert result['numeric_summary_interpretation']['status'] == 'unavailable'
