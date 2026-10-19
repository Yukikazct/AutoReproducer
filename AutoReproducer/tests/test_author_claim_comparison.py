"""Comparisons are evidence bound and never replace code completion checks."""
from copy import deepcopy
import hashlib
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


@pytest.mark.parametrize('field,wrong', [('aggregation', 'single'), ('runs', 1), ('std', 0.03), ('runs', True)])
def test_llm_cannot_rewrite_observed_summary_statistics(sources, field, wrong):
    fixture, packet, records, payload = sources
    record = next(row for row in records if row['split'] == 'test')
    record.update(aggregation='mean', runs=10, std=0.01)
    payload['comparisons'][0]['actual_context'].update(aggregation='mean', runs=10, std=0.01)
    payload['comparisons'][0]['actual_context'][field] = wrong
    with pytest.raises(ValueError, match='observed|Invalid actual'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_program_summary_fills_unknown_fields_and_report_keeps_std(sources):
    fixture, packet, records, payload = sources
    record = next(row for row in records if row['split'] == 'test')
    record.update(aggregation='mean', runs=10, std=0.01)
    payload['comparisons'][0]['actual_context'].update(aggregation='unknown', runs=None)
    rows, citations = validate_comparisons(payload, packet, fixture['workspace'], records)
    actual = rows[0]['actual']
    assert actual['runs'] == 10 and actual['aggregation'] == 'mean' and actual['std'] == 0.01
    assert rows[0]['status'] == 'reference_only'
    report = ReportGeneratorAgent(logger=Mock())._build_report({'execution': {'mode': 'repository'},
        'validation': {'status': 'author_code_completed', 'is_reproduced': None,
            'author_claim_comparison': {'comparisons': rows, 'citations': citations}}})
    assert '± 0.01' in report and '10 次' in report and '仅供参考' in report


def _confidence_record(records):
    record = next(row for row in records if row['split'] == 'test')
    record.update(reported_uncertainty=0.015536, uncertainty_kind='bootstrap_confidence_interval',
                  confidence_level=95, uncertainty_source={'path': 'train.py',
                  'source_sha256': 'a' * 64, 'definition_line': 5, 'print_line': 6})
    return record


def test_observed_confidence_half_width_survives_comparison_and_report(sources):
    fixture, packet, records, payload = sources
    original = _confidence_record(records)
    rows, citations = validate_comparisons(payload, packet, fixture['workspace'], records)
    actual = rows[0]['actual']
    assert actual['reported_uncertainty'] == original['reported_uncertainty']
    assert actual['uncertainty_kind'] == 'bootstrap_confidence_interval'
    assert actual['confidence_level'] == 95
    assert actual['uncertainty_source'] == original['uncertainty_source']
    assert actual.get('std') is None
    report = ReportGeneratorAgent(logger=Mock())._build_report({'execution': {'mode': 'repository'},
        'validation': {'status': 'author_code_completed', 'is_reproduced': None,
            'author_claim_comparison': {'comparisons': rows, 'citations': citations}}})
    assert '± 0.015536' in report and '95% bootstrap 置信区间半宽' in report


@pytest.mark.parametrize('field,value', [('reported_uncertainty', 0.1),
    ('uncertainty_kind', 'standard_deviation'), ('confidence_level', 99),
    ('uncertainty_source', {'path': 'invented.py'})])
def test_model_cannot_change_observed_uncertainty_context(sources, field, value):
    fixture, packet, records, payload = sources
    _confidence_record(records)
    payload['comparisons'][0]['actual_context'][field] = value
    with pytest.raises(ValueError, match='observed'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_model_cannot_invent_unobserved_confidence_context(sources):
    fixture, packet, records, payload = sources
    payload['comparisons'][0]['actual_context'].update(reported_uncertainty=0.1,
        uncertainty_kind='bootstrap_confidence_interval', confidence_level=95)
    with pytest.raises(ValueError, match='unobserved'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_unpaired_actual_evidence_table_keeps_confidence_half_width(sources):
    _, _, records, _ = sources
    _confidence_record(records)
    report = ReportGeneratorAgent(logger=Mock())._build_report({'execution': {'mode': 'repository'},
        'validation': {'status': 'author_code_completed', 'is_reproduced': None, 'metric_records': records,
            'author_claim_comparison': {'status': 'unavailable', 'metric_records': records}}})
    assert '± 0.015536' in report and '95% bootstrap 置信区间半宽' in report


def _two_step_proposal(payload):
    execution = actual_execution()
    execution['steps'].append({'id': 'another', 'success': True, 'exit_code': 0,
                               'stdout': 'Test accuracy: 0.9', 'stderr': ''})
    records, _ = collect_author_metrics(execution)
    second = deepcopy(payload['comparisons'][0])
    second['actual_id'] = next(row['id'] for row in records if row['step_id'] == 'another')
    return execution, second


def test_bad_row_does_not_discard_valid_row_and_correction_merges_results(sources):
    fixture, packet, _, payload = sources
    execution, second = _two_step_proposal(payload)
    first = deepcopy(payload)
    broken = deepcopy(second); broken['reference']['value'] = 91.3
    first['comparisons'].append(broken)
    corrected = deepcopy(payload); corrected['comparisons'] = [second]
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = [json.dumps(first), json.dumps(corrected)]
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], execution)
    assert llm.chat.call_count == 2 and len(result['comparisons']) == 2
    assert result['status'] == 'compared' and not result['validation_errors']
    assert [row['actual']['value'] for row in result['comparisons']] == [1.0, 0.9]
    assert result['extraction_attempts'][0]['validation_errors']
    assert first['comparisons'][1]['reference']['value'] == 91.3


@pytest.mark.parametrize('response', ['empty', 'api_failure', 'bad_citation', 'bad_identifier'])
def test_unresolved_row_is_reported_while_valid_rows_survive(sources, response):
    fixture, packet, _, payload = sources
    execution, second = _two_step_proposal(payload)
    first = deepcopy(payload)
    second['reference']['value'] = 91.3
    if response == 'bad_citation':
        second['reference']['citations'] = ['does_not_exist']
    if response == 'bad_identifier':
        second['actual_id'] = ['not', 'a', 'string']
    first['comparisons'].insert(0, second)
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = [json.dumps(first), RuntimeError('API unavailable') if response == 'api_failure'
                           else json.dumps({'comparisons': []})]
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], execution)
    assert llm.chat.call_count == 2
    assert result['status'] == 'partial' and len(result['comparisons']) == 1
    assert result['comparisons'][0]['actual']['value'] == 1.0 and result['validation_errors']
    if response == 'api_failure':
        assert 'API unavailable' in result['diagnostic']
    report = ReportGeneratorAgent(logger=Mock())._build_report({'execution': {'mode': 'repository'},
        'validation': {'status': 'author_code_completed', 'author_claim_comparison': result}})
    assert '未完成的配对' in report and '作者输出与论文宣称对比' in report


def _flat_table(sources, annotation=None):
    import hashlib
    fixture, packet, records, payload = sources
    packet = deepcopy(packet)
    paper = next(ref for ref in payload['citations'] if ref['id'] == 'paper_results')
    header = 'Method SignGrid OtherGrid'
    method = 'Signal Linear' + (f' ({annotation})' if annotation else '')
    row = method + ' test 100.0 98.0'
    text = 'Test accuracy (%)\n' + header + '\n' + row
    page = next(page for page in packet['pdf']['pages'] if page['page'] == paper['page'])
    page['text'] += '\n' + text; page['sha256'] = hashlib.sha256(page['text'].encode()).hexdigest()
    paper['quote'] = text
    reference = payload['comparisons'][0]['reference']
    reference.update(value_quote=row, column_quote=header, method=method)
    return fixture, packet, records, payload


def test_flat_table_value_must_belong_to_the_correct_dataset_column(sources):
    fixture, packet, records, payload = _flat_table(sources)
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert rows[0]['reference']['column_verified'] is True
    payload['comparisons'][0]['reference']['value'] = 98.0
    with pytest.raises(ValueError, match='dataset column'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


@pytest.mark.parametrize('separator', ['±', '+/-', '\x06'])
def test_reference_deviation_cannot_come_from_another_dataset_column(sources, separator):
    import hashlib
    fixture, packet, records, payload = _flat_table(sources)
    reference = payload['comparisons'][0]['reference']
    old = reference['value_quote']
    row = f'Signal Linear test 100.0 {separator} 0.1 98.0 {separator} 0.2'
    paper = next(ref for ref in payload['citations'] if ref['id'] == 'paper_results')
    page = next(page for page in packet['pdf']['pages'] if page['page'] == paper['page'])
    page['text'] = page['text'].replace(old, row)
    page['sha256'] = hashlib.sha256(page['text'].encode()).hexdigest()
    paper['quote'] = paper['quote'].replace(old, row)
    reference.update(value_quote=row, std=0.1)
    accepted, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert accepted[0]['reference']['std'] == 0.1
    reference['std'] = 0.2
    with pytest.raises(ValueError, match='standard deviation.*dataset column'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


@pytest.mark.parametrize('header', [None, 'Method OtherGrid'])
def test_multi_dataset_row_requires_its_exact_identifying_header(sources, header):
    fixture, packet, records, payload = _flat_table(sources)
    payload['comparisons'][0]['reference']['column_quote'] = header
    with pytest.raises(ValueError, match='column_quote'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def _spaced_method_table(sources, label='SIG NAL LIN EAR', first='100.0 ±0.1'):
    import hashlib
    fixture, packet, records, payload = _flat_table(sources)
    paper = next(ref for ref in payload['citations'] if ref['id'] == 'paper_results')
    page = next(page for page in packet['pdf']['pages'] if page['page'] == paper['page'])
    header = 'Type Model Non-linearity PDE-solver Input-G SignGrid OtherGrid'
    row = f'{label} - - - {first} 98.0 ±0.2'
    text = 'Test accuracy (%)\n' + header + '\n' + row
    page['text'] += '\n' + text
    page['sha256'] = hashlib.sha256(page['text'].encode()).hexdigest()
    paper['quote'] = text
    payload['comparisons'][0]['reference'].update(value_quote=row, column_quote=header, std=0.1)
    return fixture, packet, records, payload


def test_pdf_method_kerning_and_metadata_placeholders_keep_exact_column_evidence(sources):
    fixture, packet, records, payload = _spaced_method_table(sources)
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    ref = rows[0]['reference']
    assert ref['method'] == 'SIG NAL LIN EAR' and ref['proposed_method_label'] == 'Signal Linear'
    assert ref['column_verified'] is True and ref['value'] == 100 and ref['std'] == 0.1
    assert payload['comparisons'][0]['reference']['method'] == 'Signal Linear'
    payload['comparisons'][0]['reference']['std'] = 0.2
    with pytest.raises(ValueError, match='standard deviation.*dataset column'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


@pytest.mark.parametrize('label', ['SIG NAL LIN EAR (PPR)', 'SIG NAL LIN EAR-s', 'OTHER METHOD'])
def test_pdf_typography_does_not_drop_variant_suffixes_or_change_methods(sources, label):
    fixture, packet, records, payload = _spaced_method_table(sources, label=label)
    with pytest.raises(ValueError, match='method row'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_dataset_column_placeholder_is_not_a_neighboring_dataset_result(sources):
    fixture, packet, records, payload = _spaced_method_table(sources, first='-')
    payload['comparisons'][0]['reference'].update(value=98.0, std=0.2)
    with pytest.raises(ValueError, match='column.*(?:value|reported)'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_separator_spelling_does_not_merge_distinct_variants():
    from src.author_claim_comparison import _context
    assert _context('Signal_Linear') == _context('Signal-Linear')
    assert _context('DGC_Euler') != _context('DGC-PPR')


@pytest.mark.parametrize('annotation', ['ours', 'our method', 'this work'])
def test_author_table_annotation_is_not_an_algorithm_variant(sources, annotation):
    fixture, packet, records, payload = _flat_table(sources, annotation)
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert rows[0]['status'] == 'comparable' and rows[0]['difference'] == 0
    assert rows[0]['reference']['method'].endswith(f'({annotation})')


@pytest.mark.parametrize('annotation', ['Euler', 'PPR', 'GAT'])
def test_real_parenthetical_variants_are_not_stripped(sources, annotation):
    fixture, packet, records, payload = _flat_table(sources, annotation)
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert rows[0]['status'] == 'not_comparable' and rows[0]['difference'] is None


def test_method_row_name_correction_handles_only_cosmetic_author_annotations(sources):
    fixture, packet, records, payload = _flat_table(sources, 'ours')
    reference = payload['comparisons'][0]['reference']
    reference.update(name='Signal Linear (ours)', method='Signal Linear')
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert rows[0]['metric'] == 'accuracy' and rows[0]['status'] == 'comparable'
    assert rows[0]['reference']['proposed_metric_name'] == 'Signal Linear (ours)'
    reference['name'] = 'Signal Linear (Euler)'
    with pytest.raises(ValueError, match='Reference name'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def _paired_metric_table(sources, selected='f1_macro'):
    import hashlib
    fixture, packet, records, payload = _flat_table(sources)
    paper = next(ref for ref in payload['citations'] if ref['id'] == 'paper_results')
    page = next(page for page in packet['pdf']['pages'] if page['page'] == paper['page'])
    header = 'Method SignGrid OtherGrid'
    caption = 'Test results (AUC, Macro-F1) in percent.'
    row = 'Signal Linear (87.5 ±0.9, 68.9 ±0.8) (90.2 ±0.4, 70.1 ±1.2)'
    text = caption + '\n' + header + '\n' + row
    page['text'] += '\n' + text
    page['sha256'] = hashlib.sha256(page['text'].encode()).hexdigest()
    paper['quote'] = text
    record = next(r for r in records if r['split'] == 'test')
    record.update(name=selected, value=0.6859)
    payload['comparisons'][0]['reference'].update(name=selected, value=68.9, std=0.8,
        value_quote=row, column_quote=header, metric_quote=caption)
    return fixture, packet, records, payload


def test_binary_f1_cannot_stand_in_for_paper_macro_f1_even_if_model_copies_runtime_name(sources):
    fixture, packet, records, payload = _paired_metric_table(sources, selected='f1_score')
    with pytest.raises(ValueError, match='metric.*evidence'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_tuple_table_pairs_the_dataset_and_metric_columns(sources):
    fixture, packet, records, payload = _paired_metric_table(sources)
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records)
    assert rows[0]['reference']['column_verified'] is True
    assert rows[0]['difference'] == pytest.approx(-0.31)
    payload['comparisons'][0]['reference'].update(value=87.5, std=0.9)
    with pytest.raises(ValueError, match='dataset column'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_tuple_metric_order_requires_exact_same_page_evidence(sources):
    fixture, packet, records, payload = _paired_metric_table(sources)
    payload['comparisons'][0]['reference']['metric_quote'] = 'Test results (Macro-F1, AUC) in percent.'
    with pytest.raises(ValueError, match='metric_quote'):
        validate_comparisons(payload, packet, fixture['workspace'], records)


def test_exact_selected_evidence_id_can_recover_an_omitted_citation_declaration(sources):
    fixture, packet, _, payload = sources
    from src.discovered_repository_plan import _planning_view
    excerpts = _planning_view(packet)['evidence_excerpts']
    identifier = next(key for key, ref in excerpts.items() if ref['source'] == 'pdf' and ref['page'] == 3)
    payload['citations'] = [ref for ref in payload['citations'] if ref['id'] != 'paper_results']
    for field in ('reference', 'settings'):
        node = payload['comparisons'][0][field]
        node['citations'] = [identifier if key == 'paper_results' else key for key in node['citations']]
    llm = Mock(mock_mode=False)
    llm.chat.return_value = json.dumps(payload)
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'compared' and llm.chat.call_count == 1
    assert result['extraction_attempts'][0]['citation_repairs'] == [
        {'id': identifier, 'evidence_id': identifier, 'reason': 'Selected exact evidence ID omitted from citation declarations'}]
    assert all(ref['id'] != identifier for ref in payload['citations'])


def test_malformed_comparison_response_is_retained_and_corrected_with_current_feedback(sources):
    fixture, packet, _, payload = sources
    malformed = '{"comparisons": [BROKEN_CURRENT_RESPONSE]}'
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = [malformed, json.dumps(payload)]
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    attempts = result['extraction_attempts']
    assert result['status'] == 'compared' and llm.chat.call_count == 2
    assert len(attempts) == 2 and [item['attempt'] for item in attempts] == [1, 2]
    assert attempts[0]['status'] == 'invalid_response' and 'error' in attempts[0]
    assert 'proposal' not in attempts[0]
    assert attempts[0]['response_sha256'] == hashlib.sha256(malformed.encode()).hexdigest()
    assert attempts[0]['response_excerpt'] == malformed
    assert attempts[0]['json_error']['line'] == 1 and attempts[0]['json_error']['column'] > 1
    feedback = llm.chat.call_args_list[1].args[0].split('\n\nCorrection feedback', 1)[1]
    assert 'BROKEN_CURRENT_RESPONSE' in feedback and 'line 1' in feedback
    assert 'valid JSON object' in feedback and len(feedback) <= 12000
    assert attempts[1]['status'] == 'validated' and attempts[1]['proposal'] == payload


def test_each_malformed_response_has_its_own_attempt_and_finite_retry(sources):
    fixture, packet, _, _ = sources
    responses = ['first invalid response', 'second invalid response']
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = responses
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'unavailable' and llm.chat.call_count == 2
    assert len(result['extraction_attempts']) == 2
    assert [item['response_excerpt'] for item in result['extraction_attempts']] == responses
    assert all(item['status'] == 'invalid_response' and item['error'] for item in result['extraction_attempts'])
    assert 'not valid JSON' in result['diagnostic']


def test_api_exception_is_retained_without_inventing_response_or_retrying(sources):
    fixture, packet, _, _ = sources
    exact_secret = 'private-client-credential'
    generic_secret = 'sk-' + 'a' * 32
    llm = Mock(mock_mode=False, api_key=exact_secret)
    llm.chat.side_effect = RuntimeError(f'API unavailable {exact_secret}; Bearer generic-value; {generic_secret}')
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'unavailable' and llm.chat.call_count == 1
    attempt, = result['extraction_attempts']
    assert attempt['status'] == 'api_error' and attempt['error_type'] == 'RuntimeError'
    assert 'API unavailable' in attempt['error'] and 'response_excerpt' not in attempt
    persisted = json.dumps(result)
    assert all(secret not in persisted for secret in (exact_secret, generic_secret, 'generic-value'))


def test_parse_failure_does_not_overwrite_previous_proposal_or_valid_rows(sources):
    fixture, packet, _, payload = sources
    execution, second = _two_step_proposal(payload)
    first = deepcopy(payload)
    second['reference']['value'] = 91.3
    first['comparisons'].append(second)
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = [json.dumps(first), 'latest malformed response']
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], execution)
    assert result['status'] == 'partial' and len(result['comparisons']) == 1
    assert result['comparisons'][0]['actual']['value'] == 1.0 and result['validation_errors']
    attempts = result['extraction_attempts']
    assert len(attempts) == 2
    assert attempts[0]['proposal'] == first and attempts[0]['validation_errors']
    assert 'error' not in attempts[0]
    assert attempts[1]['status'] == 'invalid_response' and 'proposal' not in attempts[1]
    assert attempts[1]['response_excerpt'] == 'latest malformed response'


def test_api_exception_does_not_overwrite_previous_validation_attempt(sources):
    fixture, packet, _, payload = sources
    execution, second = _two_step_proposal(payload)
    first = deepcopy(payload)
    second['reference']['value'] = 91.3
    first['comparisons'].append(second)
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = [json.dumps(first), RuntimeError('second request unavailable')]
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], execution)
    assert result['status'] == 'partial' and len(result['comparisons']) == 1
    assert len(result['extraction_attempts']) == 2 and llm.chat.call_count == 2
    assert result['extraction_attempts'][0]['validation_errors']
    assert 'error' not in result['extraction_attempts'][0]
    assert result['extraction_attempts'][1]['status'] == 'api_error'


def test_long_malformed_response_is_bounded_redacted_and_hashed_before_redaction(sources):
    fixture, packet, _, payload = sources
    exact_secret = 'private-client-credential'
    generic_secret = 'sk-' + 'b' * 32
    response = (f'{{"api_key": "labelled-secret", "message": "{exact_secret} {generic_secret}"}}\n'
                + 'x' * 30000 + '\nBearer trailing-secret')
    llm = Mock(mock_mode=False, api_key=exact_secret)
    llm.chat.side_effect = [response, json.dumps(payload)]
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    attempt = result['extraction_attempts'][0]
    assert attempt['response_sha256'] == hashlib.sha256(response.encode()).hexdigest()
    assert attempt['response_chars'] == len(response) and attempt['response_truncated'] is True
    assert len(attempt['response_excerpt']) <= 6000
    feedback = llm.chat.call_args_list[1].args[0].split('\n\nCorrection feedback', 1)[1]
    assert len(feedback) <= 12000
    persisted = json.dumps(result) + feedback
    assert all(secret not in persisted for secret in
               (exact_secret, generic_secret, 'labelled-secret', 'trailing-secret'))


def test_nontext_response_is_recorded_without_serializing_arbitrary_objects(sources):
    fixture, packet, _, payload = sources
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = [None, json.dumps(payload)]
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'compared' and len(result['extraction_attempts']) == 2
    attempt = result['extraction_attempts'][0]
    assert attempt['status'] == 'invalid_response' and attempt['response_type'] == 'NoneType'
    assert 'response_excerpt' not in attempt and 'response_sha256' not in attempt


def test_invalid_structure_feedback_uses_only_current_parsed_proposal(sources):
    fixture, packet, _, payload = sources
    invalid = {'reason': 'CURRENT_SCHEMA_FAILURE'}
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = [json.dumps(invalid), json.dumps(payload)]
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'compared' and len(result['extraction_attempts']) == 2
    attempt = result['extraction_attempts'][0]
    assert attempt['status'] == 'validation_error' and attempt['proposal'] == invalid
    assert 'Expected a JSON object with comparisons' in attempt['error']
    feedback = llm.chat.call_args_list[1].args[0].split('\n\nCorrection feedback', 1)[1]
    assert 'CURRENT_SCHEMA_FAILURE' in feedback


def test_valid_parsed_proposals_and_diagnostics_also_redact_client_credentials(sources):
    fixture, packet, _, payload = sources
    exact_secret = 'private-client-credential'
    payload['comparisons'][0]['settings']['reason'] += ' ' + exact_secret
    llm = Mock(mock_mode=False, api_key=exact_secret)
    llm.chat.return_value = json.dumps(payload)
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'compared' and len(result['comparisons']) == 1
    assert exact_secret not in json.dumps(result)
    assert payload['comparisons'][0]['settings']['reason'].endswith(exact_secret)
    assert result['comparisons'][0]['reference']['value'] == 100


def test_oversized_parsed_proposal_is_bounded_without_bypassing_evidence_validation(sources):
    fixture, packet, _, payload = sources
    payload['reason'] = 'x' * 50000
    llm = Mock(mock_mode=False)
    response = json.dumps(payload)
    llm.chat.return_value = response
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'compared' and llm.chat.call_count == 1
    attempt, = result['extraction_attempts']
    assert 'proposal' not in attempt and attempt['proposal_truncated'] is True
    assert len(attempt['proposal_excerpt']) <= 6000 and len(attempt['response_excerpt']) <= 6000
    assert attempt['response_sha256'] == hashlib.sha256(response.encode()).hexdigest()
    assert result['comparisons'][0]['actual']['value'] == 1.0


def test_json_escaped_client_key_is_redacted_in_malformed_response_and_feedback(sources):
    fixture, packet, _, payload = sources
    credential = 'opaque_"client_雪_value'
    response = json.dumps({'reason': credential}) + ' malformed tail'
    llm = Mock(mock_mode=False, api_key=credential)
    llm.chat.side_effect = [response, json.dumps(payload)]
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    persisted = json.dumps(result, ensure_ascii=False) + llm.chat.call_args_list[1].args[0]
    assert credential not in persisted and json.dumps(credential)[1:-1] not in persisted
    assert '[REDACTED]' in result['extraction_attempts'][0]['response_excerpt']


@pytest.mark.parametrize('invalid_number', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_rejected_row_does_not_preempt_other_valid_comparisons(sources, invalid_number):
    fixture, packet, _, payload = sources
    execution, second = _two_step_proposal(payload)
    first = deepcopy(payload)
    second['reference']['value'] = invalid_number
    first['comparisons'].append(second)
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = [json.dumps(first), json.dumps({'comparisons': []})]
    result = compare_author_claims(llm, packet, fixture['proposal'], fixture['workspace'], execution)
    assert result['status'] == 'partial' and len(result['comparisons']) == 1
    assert result['comparisons'][0]['actual']['value'] == 1.0
    attempt = result['extraction_attempts'][0]
    assert attempt['status'] == 'validation_failed' and attempt['validation_errors']
    assert attempt['proposal']['comparisons'][1]['reference']['value'] == str(invalid_number)
    json.dumps(result, allow_nan=False)
