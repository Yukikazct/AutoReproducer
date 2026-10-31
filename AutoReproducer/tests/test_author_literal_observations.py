"""Literal author numbers remain distinct from metrics and paper acceptance."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from discovered_fixtures import make_experiment
from src.author_claim_comparison import collect_author_metrics, compare_author_claims
from src.author_literal_observations import MAX_DIAGNOSTICS, MAX_OBSERVATIONS, collect_author_observations
from src.discovered_repository_plan import build_evidence_packet
from src.execution_completion import completed_execution
from src.reproduction_assessment import assess_reproduction, apply_reproduction_assessment


PAIRS = ('orig ROC:0.9124, precision @ rank n:0.4595\n'
         'new ROC:0.9263, precision @ rank n:0.5405\n'
         'orig moa ROC:0.9164, precision @ rank n:0.4595\n'
         'new moa ROC:0.9307, precision @ rank n:0.5405')


def execution(text=PAIRS):
    return {'mode': 'repository', 'executed': True, 'success': True,
            'steps': [{'id': 'author_demo', 'executed': True, 'success': True, 'exit_code': 0,
                       'stdout': text, 'stderr': ''}]}


def test_paired_original_and_new_scopes_preserve_all_eight_literal_values():
    run = execution()
    original = deepcopy(run)
    observations, errors = collect_author_observations(run)
    assert not errors and len(observations) == 8
    assert [(item['row_label'], item['metric_label'], item['value']) for item in observations] == [
        ('orig', 'ROC', .9124), ('orig', 'precision @ rank n', .4595),
        ('new', 'ROC', .9263), ('new', 'precision @ rank n', .5405),
        ('orig moa', 'ROC', .9164), ('orig moa', 'precision @ rank n', .4595),
        ('new moa', 'ROC', .9307), ('new moa', 'precision @ rank n', .5405)]
    assert len({item['id'] for item in observations}) == 8
    for index, item in enumerate(observations):
        assert item['source'] == f'author_demo:stdout:{index // 2 + 1}'
        assert item['source_quote'] == PAIRS.splitlines()[index // 2]
        assert item['unit'] == '' and item['interpretation_status'] == 'unresolved'
        assert not {'name', 'method', 'dataset', 'aggregation', 'runs', 'split', 'stage'} & item.keys()
    assert run == original
    assert collect_author_metrics(run) == ([], [])


def test_timing_labels_and_exact_numeric_tokens_do_not_imply_seconds_or_speedup():
    text = 'Fit time: 33.71004581451416\nOrig Fit time: 32.558557748794556\nApproximation time: 68.27312922477722'
    observations, errors = collect_author_observations(execution(text))
    assert not errors
    assert [item['metric_label'] for item in observations] == ['Fit time', 'Orig Fit time', 'Approximation time']
    assert [item['value_text'] for item in observations] == ['33.71004581451416', '32.558557748794556', '68.27312922477722']
    assert all(item['unit'] == '' and item['row_label'] == '' for item in observations)


@pytest.mark.parametrize('prefix', ['Paper ', 'Reference ', 'Target ', 'Declared '])
def test_reference_and_target_labels_are_not_observations(prefix):
    contaminated = '\n'.join(prefix + line for line in PAIRS.splitlines())
    assert collect_author_observations(execution(contaminated)) == ([], [])


@pytest.mark.parametrize('change', [{'success': False}, {'exit_code': 1}, {'cancelled': True},
                                   {'timed_out': True}, {'interrupted': True}, {'cleanup_unconfirmed': True}])
def test_failed_attempt_output_is_never_used(change):
    run = execution('')
    failed = {**deepcopy(execution()['steps'][0]), **change, 'id': 'failed_attempt'}
    run['steps'].insert(0, failed)
    run['historical_attempts'] = [execution()]
    assert collect_author_observations(run) == ([], [])
    assert collect_author_observations({**execution(), 'success': False}) == ([], [])


@pytest.mark.parametrize('line', ['orig ROC:0.9, precision @ rank n:nan', 'orig ROC:inf, precision @ rank n:0.5',
                                 'orig ROC:0.9garbage, precision @ rank n:0.5',
                                 'orig ROC:0.9, precision @ rank n:0.5, other:1',
                                 'orig ROC:0.9', 'orig ROC:0.9, precision:0.5'])
def test_malformed_or_nonfinite_pairs_are_not_partially_promoted(line):
    observations, _ = collect_author_observations(execution(line))
    assert not observations


@pytest.mark.parametrize('configuration', [
    {'metric_records': []}, {'metrics': {'accuracy': .8}}, {'metric_records': None, 'metrics': {'accuracy': .8}},
    {'stdout': 'AUTOREPRO_METRICS: {"accuracy":0.8}\n' + PAIRS},
    {'stderr': 'AUTOREPRO_METRICS: malformed'},
])
def test_any_structured_metric_protocol_suppresses_raw_fallback(configuration):
    run = execution()
    run['steps'][0].update(configuration)
    assert collect_author_observations(run) == ([], [])


def test_recognized_metrics_and_stdout_stderr_provenance_remain_distinct():
    run = execution('Test accuracy:0.81\n' + PAIRS.splitlines()[0])
    run['steps'][0]['stderr'] = PAIRS.splitlines()[1]
    metrics, errors = collect_author_metrics(run)
    assert not errors and [(item['name'], item['value']) for item in metrics] == [('accuracy', .81)]
    observations, errors = collect_author_observations(run)
    assert not errors and len(observations) == 4
    assert {item['source'] for item in observations} == {'author_demo:stdout:2', 'author_demo:stderr:1'}
    assert collect_author_metrics(run) == (metrics, [])


def test_repeated_scope_labels_are_not_collapsed_or_given_a_final_run_meaning():
    observations, _ = collect_author_observations(execution(PAIRS + '\n' + PAIRS))
    assert len(observations) == 16 and len({item['id'] for item in observations}) == 16
    assert observations[0]['source'] != observations[8]['source']


def test_observation_and_error_limits_are_bounded_without_splitting_a_pair():
    observations, errors = collect_author_observations(execution('orig ROC:0.9, precision @ rank n:0.5\n' * 100))
    assert len(observations) == MAX_OBSERVATIONS and len(observations) % 2 == 0
    assert len(errors) == 1 and '保留上限' in errors[0]
    observations, errors = collect_author_observations(execution('Fit time: nan\n' * 1000 + PAIRS))
    assert len(observations) == 8 and len(errors) == MAX_DIAGNOSTICS and '省略' in errors[-1]


def test_raw_values_do_not_create_paper_comparisons_or_upgrade_execution(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    run = execution()
    llm = Mock(mock_mode=False)
    comparison = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], run)
    assert len(comparison['raw_observations']) == 8 and not comparison['metric_records']
    assert comparison['status'] == 'unavailable' and not comparison['comparisons']
    assert '已保留' in comparison['reason'] and '尚未核验' in comparison['reason']
    validation = completed_execution(run)
    assessment = assess_reproduction(llm, comparison, run, validation=validation)
    apply_reproduction_assessment(validation, assessment)
    assert validation['status'] == 'author_code_completed' and validation['is_reproduced'] is None
    assert validation['paper_metrics_verified'] is False
    assert assessment['verdict'] == 'insufficient_evidence'
    llm.chat.assert_not_called()


def test_raw_early_return_redacts_exact_and_generic_credentials(tmp_path):
    fixture = make_experiment(tmp_path)
    secret = 'private-client-credential'
    generic = 'sk-' + 'x' * 32
    run = execution(f'{secret} ROC:0.9, precision @ rank n:0.5\n{generic} ROC:0.8, precision @ rank n:0.4\nFit time: nan')
    run['steps'][0]['id'] = secret + '-' + generic
    llm = Mock(mock_mode=True, api_key=secret)
    result = compare_author_claims(llm, {}, fixture['proposal'], fixture['workspace'], run)
    assert len(result['raw_observations']) == 4 and '[REDACTED]' in json.dumps(result)
    assert result['raw_observation_errors'] and '[REDACTED]' in result['raw_observation_errors'][0]
    assert secret not in json.dumps(result) and generic not in json.dumps(result)
