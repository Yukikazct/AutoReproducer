"""Author summaries outrank individual trials and keep their own statistics."""
import json

import pytest

from src.author_claim_comparison import collect_author_metrics
from src.runtime_metrics import METRICS_MARKER, parse_runtime_metrics


DGC_OUTPUT = """Test accuracy: 0.8320, train time: 0.0572s
Test accuracy: 0.8330, train time: 0.0572s
Dataset: cora Test accuracy of 10 runs: mean 0.83280, std 0.00040
"""


def test_dgc_final_ten_trial_summary_replaces_last_trial():
    execution = {'steps': [{'id': 'train', 'success': True, 'exit_code': 0,
                            'stdout': DGC_OUTPUT, 'stderr': ''}]}
    records, errors = collect_author_metrics(execution)
    assert not errors and len(records) == 1
    row = records[0]
    assert row['name'] == 'accuracy' and row['value'] == 0.8328
    assert row['aggregation'] == 'mean' and row['runs'] == 10 and row['std'] == 0.0004
    assert row['source'] == 'train:stdout:3' and row['source_quote'] == DGC_OUTPUT.splitlines()[-1]


def test_digcn_saved_output_keeps_test_mean_separate_from_validation():
    line = 'Val Acc: 0.8055, Val Loss: 0.6815, Test Accuracy: 0.7719 ± 0.0044, Duration: 2.4038'
    records, errors = parse_runtime_metrics(line, preserve_contexts=True)
    assert not errors and len(records) == 3
    test = next(row for row in records if row['split'] == 'test')
    assert test['value'] == 0.7719 and test['std'] == 0.0044 and test['aggregation'] == 'mean'
    assert 'runs' not in test
    assert all('std' not in row for row in records if row['split'] == 'validation')


@pytest.mark.parametrize('line,expected', [
    ('Test accuracy: 0.7719 ± 0.0044', (0.7719, 0.0044, '', None)),
    ('Test Accuracy: 77.19 ± 0.44%', (77.19, 0.44, 'percent', None)),
    ('Test accuracy: 77.19% +/- 0.44%', (77.19, 0.44, 'percent', None)),
    ('Mean Test accuracy: 0.82, std: 0.02, runs=10', (0.82, 0.02, '', 10)),
    ('Test accuracy over 20 seeds: mean=0.82, std=0.02', (0.82, 0.02, '', 20)),
])
def test_explicit_summary_statistics_and_units(line, expected):
    records, errors = parse_runtime_metrics(line, preserve_contexts=True)
    assert not errors and len(records) == 1
    row = records[0]
    assert (row['value'], row['std'], row['unit'], row.get('runs')) == expected
    assert row['aggregation'] == 'mean'


@pytest.mark.parametrize('aggregation', ['mean', 'median', 'best'])
def test_reported_summary_is_not_overwritten_by_a_later_trial(aggregation):
    records, errors = parse_runtime_metrics(
        f'Test accuracy of 10 runs: {aggregation} 0.82\nTest accuracy: 0.91', preserve_contexts=True)
    assert not errors and len(records) == 1
    assert records[0]['aggregation'] == aggregation and records[0]['value'] == 0.82


def test_plain_measurement_does_not_invent_repetitions_or_statistics():
    records, errors = parse_runtime_metrics('Test accuracy: 0.82', preserve_contexts=True)
    assert not errors
    assert not any(key in records[0] for key in ('aggregation', 'runs', 'std'))


def test_unrelated_summary_does_not_abort_final_metric_extraction():
    records, errors = parse_runtime_metrics(
        'latency mean 2.1; Test accuracy: 0.8', preserve_contexts=True)
    assert not errors and len(records) == 1
    assert records[0]['name'] == 'accuracy' and records[0]['value'] == 0.8
    assert 'aggregation' not in records[0]


def test_explicitly_requested_custom_summary_remains_supported():
    records, errors = parse_runtime_metrics('latency mean 2.1', expected_names=['latency'])
    assert not errors and len(records) == 1
    assert records[0]['name'] == 'latency' and records[0]['value'] == 2.1
    assert records[0]['aggregation'] == 'mean'


def test_confidence_interval_is_not_a_standard_deviation():
    records, errors = parse_runtime_metrics('Test accuracy: 0.82 ± 0.02 (95% CI)', preserve_contexts=True)
    assert not errors and 'std' not in records[0] and 'aggregation' not in records[0]


def test_two_metrics_on_a_line_cannot_borrow_statistics():
    records, errors = parse_runtime_metrics(
        'Mean Validation accuracy: 0.88, std: 0.01; Test accuracy: 0.81', preserve_contexts=True)
    assert not errors and len(records) == 2
    validation, test = records
    assert validation['aggregation'] == 'mean' and validation['std'] == 0.01
    assert test['split'] == 'test' and 'aggregation' not in test and 'std' not in test


def test_paper_reference_summary_never_overwrites_actual_output():
    records, errors = parse_runtime_metrics(
        DGC_OUTPUT + 'Paper test accuracy of 10 runs: mean 0.99, std 0.01', preserve_contexts=True)
    assert not errors and records[0]['value'] == 0.8328


@pytest.mark.parametrize('extra', [
    {'runs': True}, {'runs': 0}, {'runs': 1.5}, {'std': -0.01},
    {'std': 'nan'}, {'aggregation': []}, {'aggregation': 'invented'},
])
def test_invalid_structured_statistics_remain_an_error_without_text_fallback(extra):
    records, errors = parse_runtime_metrics(DGC_OUTPUT, structured=[
        {'name': 'accuracy', 'value': 0.5, 'split': 'test', **extra}])
    assert errors and records[0]['value'] == 0.5


def test_structured_summary_remains_authoritative_and_keeps_provenance():
    payload = {'metrics': [{'name': 'accuracy', 'value': 0.9, 'unit': 'fraction',
                           'aggregation': 'mean', 'runs': 20, 'std': 0.01}]}
    records, errors = parse_runtime_metrics(METRICS_MARKER + json.dumps(payload) + '\n' + DGC_OUTPUT)
    assert not errors
    assert records[0]['value'] == 0.9 and records[0]['runs'] == 20
    assert records[0]['source'] == 'stdout:1'
