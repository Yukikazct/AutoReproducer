"""Comparisons are evidence bound and never replace code completion checks."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from discovered_fixtures import make_experiment
from src.author_claim_comparison import collect_author_metrics, compare_author_claims, validate_comparisons
from src.discovered_repository_plan import build_evidence_packet
from src.agents.report_generator import ReportGeneratorAgent
from src.runtime_metrics import parse_runtime_metrics


def actual_execution(value=1.0):
    return {'success': True, 'steps': [{'id': 'train', 'success': True, 'exit_code': 0,
        'stdout': f'Validation Accuracy: 0.88 Test Accuracy: {value}', 'stderr': ''}]}


@pytest.fixture
def sources(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    records, _ = collect_author_metrics(actual_execution())
    test = next(record for record in records if record['split'] == 'test')
    reference = {'name': 'accuracy', 'value': 100.0, 'unit': 'percent', 'method': 'Signal Linear',
        'dataset': 'SignGrid', 'split': 'test', 'aggregation': 'single', 'runs': 1,
        'value_quote': 'Signal Linear SignGrid test 100.0', 'citations': ['paper_results']}
    actual = {'unit': 'fraction', 'method': 'Signal Linear', 'dataset': 'SignGrid', 'split': 'test',
        'aggregation': 'single', 'runs': 1, 'citations': ['readme', 'script']}
    payload = {'citations': deepcopy(fixture['proposal']['citations']), 'comparisons': [
        {'actual_id': test['id'], 'reference': reference, 'actual_context': actual, 'direction': 'higher',
         'settings': {'status': 'matched', 'reason': 'The fixed complete test split and full command agree.',
                      'citations': ['paper_results', 'readme', 'script']}}]}
    return fixture, packet, records, payload


def test_separate_validation_and_test_on_one_line_and_keep_metric_source():
    records, errors = parse_runtime_metrics('Validation Accuracy: 0.88 Test Accuracy: 0.81', preserve_contexts=True)
    assert not errors
    assert [(r['split'], r['value']) for r in records] == [('validation', 0.88), ('test', 0.81)]
    assert all(r['source'] == 'stdout:1' for r in records)
    records, _ = parse_runtime_metrics('Train Accuracy: 0.9 Test Accuracy: 0.7')
    assert records[0]['split'] == 'test' and records[0]['value'] == 0.7


def test_qualified_metric_labels_are_parsed_for_author_output():
    records, _ = parse_runtime_metrics('val_acc: 0.88\ntest_acc: 0.81', preserve_contexts=True)
    assert [(r['name'], r['split']) for r in records] == [('accuracy', 'validation'), ('accuracy', 'test')]


def test_multiple_paper_numbers_on_one_line_cannot_be_used_as_measurements():
    records, errors = parse_runtime_metrics('Paper accuracy:0.99 f1:0.90\nTest accuracy:0.81', preserve_contexts=True)
    assert not errors
    assert [(r['name'], r['value']) for r in records] == [('accuracy', 0.81)]


def test_fraction_conversion_and_percent_point_difference(sources):
    fixture, packet, records, payload = sources
    next(r for r in records if r['split'] == 'test')['value'] = 0.94
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    row = rows[0]
    assert row['status'] == 'comparable'
    assert row['difference'] == pytest.approx(-6.0)
    assert row['relative_difference'] == pytest.approx(-0.06)
    assert row['difference_unit'] == 'percentage_points'
    assert row['meets_reference'] is False and row['within_tolerance'] is False


def test_method_label_in_name_is_repaired_only_with_cited_metric_header(sources):
    fixture, packet, records, payload = sources
    payload['comparisons'][0]['reference']['name'] = 'Signal Linear'
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert rows[0]['reference']['name'] == 'accuracy'
    assert rows[0]['reference']['proposed_metric_name'] == 'Signal Linear'
    payload['comparisons'][0]['reference']['name'] = 'mse'
    with pytest.raises(ValueError, match='Reference name must'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_single_run_and_paper_mean_show_difference_without_claiming_acceptance(sources):
    fixture, packet, records, payload = sources
    payload['comparisons'][0]['reference'].update(aggregation='mean', runs=40)
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert rows[0]['status'] == 'reference_only'
    assert rows[0]['difference'] == 0
    assert rows[0]['meets_reference'] is None
    assert any('统计方式' in item for item in rows[0]['limitations'])


@pytest.mark.parametrize('change', [
    lambda row: row['actual_context'].update(dataset=''),
    lambda row: row['actual_context'].update(method=''),
])
def test_incompatible_or_unconfirmed_context_is_visible_without_numeric_verdict(sources, change):
    fixture, packet, records, payload = sources
    change(payload['comparisons'][0])
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert rows[0]['status'] == 'not_comparable' and rows[0]['difference'] is None


def test_settings_differences_show_numeric_gap_without_acceptance(sources):
    fixture, packet, records, payload = sources
    payload['comparisons'][0]['settings']['status'] = 'different'
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert rows[0]['status'] == 'reference_only' and rows[0]['difference'] == 0
    assert rows[0]['meets_reference'] is None


def test_validation_metric_cannot_stand_in_for_paper_test(sources):
    fixture, packet, records, payload = sources
    row = payload['comparisons'][0]
    row['actual_id'] = records[0]['id']
    row['actual_context']['split'] = 'validation'
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert rows[0]['status'] == 'not_comparable' and rows[0]['difference'] is None


@pytest.mark.parametrize('change', [
    lambda row: row['reference'].update(value=91.3),
    lambda row: row['reference'].update(value=98.0, value_quote='Baseline SignGrid train 98.0'),
    lambda row: row.update(actual_id='invented'),
    lambda row: row['actual_context'].update(split='validation'),
])
def test_invented_values_baseline_rows_and_relabelled_output_are_rejected(sources, change):
    fixture, packet, records, payload = sources
    change(payload['comparisons'][0])
    with pytest.raises(ValueError):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_explicit_output_unit_cannot_be_rewritten(sources):
    fixture, packet, records, payload = sources
    next(r for r in records if r['split'] == 'test')['unit'] = 'percent'
    with pytest.raises(ValueError, match='explicit output unit'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_api_failure_keeps_actual_measurements_and_successful_execution(sources):
    fixture, packet, _, _ = sources
    execution = actual_execution()
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = RuntimeError('API unavailable')
    comparison = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], execution)
    assert comparison['status'] == 'unavailable'
    assert len(comparison['metric_records']) == 2
    assert execution['success'] is True


def test_reference_absence_preserves_real_metrics(sources):
    fixture, packet, _, _ = sources
    llm = Mock(mock_mode=False)
    llm.chat.return_value = json.dumps({'comparisons': [], 'reason': 'No corresponding paper claim'})
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'unavailable' and result['metric_records'][-1]['value'] == 1.0


def test_report_exposes_gap_scope_and_sources_without_reproduction_success(sources):
    fixture, packet, records, payload = sources
    rows, citations = validate_comparisons(payload, packet, fixture['workspace'], records)
    report = ReportGeneratorAgent(logger=Mock())._build_report({'execution': {'mode': 'repository'},
        'validation': {'status': 'author_code_completed', 'is_reproduced': None,
            'author_claim_comparison': {'comparisons': rows, 'citations': citations,
                'relative_tolerance': 0.05, 'reason': 'Observed author output only.'}}})
    assert '作者输出与论文宣称对比' in report and '个百分点' in report
    assert 'PDF 第 3 页' in report and '尚未独立复算' in report
    assert '整篇论文数值复现通过' not in report
