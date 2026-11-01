"""Useful numerical references preserve measurements and their meanings."""
from copy import deepcopy
import hashlib

import pytest

from discovered_fixtures import make_experiment
from src.author_claim_comparison import collect_author_metrics, validate_comparisons
from src.discovered_repository_plan import build_evidence_packet


@pytest.fixture
def practical_sources(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    records, errors = collect_author_metrics({'success': True, 'steps': [
        {'id': 'train', 'success': True, 'exit_code': 0, 'stdout': 'Test Accuracy: 0.94', 'stderr': ''}]})
    assert not errors
    payload = {'citations': deepcopy(fixture['proposal']['citations']), 'comparisons': [{
        'actual_id': records[0]['id'],
        'reference': {'name': 'accuracy', 'value': 100., 'unit': 'percent', 'dataset': 'SignGrid',
                      'method': 'Signal Linear', 'split': 'test', 'aggregation': 'single', 'runs': 1,
                      'value_quote': 'Signal Linear SignGrid test 100.0', 'citations': ['paper_results']},
        'actual_context': {'unit': 'fraction', 'dataset': 'SignGrid', 'method': 'Signal Linear',
                           'split': 'test', 'aggregation': 'single', 'runs': 1, 'citations': ['readme', 'script']},
        'settings': {'status': 'matched', 'reason': 'Synthetic fixed experiment.',
                     'citations': ['paper_results', 'readme', 'script']}, 'direction': 'higher'}]}
    return fixture, packet, records, payload


def _run(sources, **kwargs):
    fixture, packet, records, payload = sources
    rows, _ = validate_comparisons(payload, packet, fixture['workspace'], records,
                                   allow_reference_only=True, **kwargs)
    return rows[0]


def _replace_table(sources, text):
    _, packet, _, payload = sources
    citation = next(ref for ref in payload['citations'] if ref['id'] == 'paper_results')
    page = next(item for item in packet['pdf']['pages'] if item['page'] == citation['page'])
    page['text'] = text
    page['sha256'] = hashlib.sha256(text.encode()).hexdigest()
    citation['quote'] = text


def _no_verdict(row):
    assert row['within_tolerance'] is None
    assert row['meets_reference'] is None
    assert row.get('independently_verified') is not True


@pytest.mark.parametrize('side,field,value', [
    ('actual_context', 'method', 'Wrapper-Backbone'),
    ('reference', 'method', 'PaperMethodAlias'),
    ('actual_context', 'dataset', 'OtherDataset'),
    ('actual_context', 'dataset', None),
    ('actual_context', 'method', None),
    ('reference', 'dataset', None),
    ('reference', 'method', None),
    ('reference', 'split', None),
])
def test_names_and_missing_metadata_do_not_gate_numerical_reference(practical_sources, side, field, value):
    payload = practical_sources[-1]
    payload['comparisons'][0][side][field] = value
    before = deepcopy((practical_sources[1], practical_sources[2], payload))
    row = _run(practical_sources)
    assert row['status'] == 'reference_only'
    assert row['difference'] == pytest.approx(-6.)
    assert row['difference_unit'] == 'percentage_points'
    assert row['actual']['value'] == .94 and row['reference']['value'] == 100.
    assert row['actual' if side == 'actual_context' else 'reference'][field] == value
    _no_verdict(row)
    assert (practical_sources[1], practical_sources[2], payload) == before


def test_different_datasets_and_splits_keep_honest_labels(practical_sources):
    _, _, records, payload = practical_sources
    record = records[0]
    record.update(split='train', stage='train', source_quote='Train Accuracy: 0.94')
    payload['comparisons'][0]['actual_context'].update(dataset='OtherDataset', split='train')
    row = _run(practical_sources)
    assert row['status'] == 'reference_only' and row['difference'] == pytest.approx(-6.)
    assert row['reference']['split'] == 'test' and row['actual']['split'] == 'train'
    assert row['reference']['dataset'] == 'SignGrid' and row['actual']['dataset'] == 'OtherDataset'
    assert any('dataset' in note for note in row['limitations'])
    assert any('跨划分' in note for note in row['limitations'])
    _no_verdict(row)


@pytest.mark.parametrize('paper_metric,actual_metric,paper_label', [
    ('f1_score', 'accuracy', 'F1'), ('accuracy', 'f1_score', 'Accuracy'),
    ('mse', 'rmse', 'MSE'), ('rmse', 'mse', 'RMSE'),
    ('f1_macro', 'f1_micro', 'Macro-F1'),
])
def test_distinct_paper_and_author_metrics_keep_both_values_without_delta(
        practical_sources, paper_metric, actual_metric, paper_label):
    _, _, records, payload = practical_sources
    records[0].update(name=actual_metric, unit='scalar')
    row = payload['comparisons'][0]
    row['reference'].update(name=paper_metric, unit='scalar')
    row['actual_context'].update(unit='scalar', citations=['readme'])
    _replace_table(practical_sources, f'Method Dataset Split {paper_label}\nSignal Linear SignGrid test 100.0')
    result = _run(practical_sources)
    assert result['status'] == 'reference_only'
    assert result['reference']['name'] == paper_metric
    assert result['actual']['name'] == actual_metric and result['metric'] == actual_metric
    assert result['reference']['value'] == 100. and result['actual']['value'] == .94
    assert result['difference'] is None and result['relative_difference'] is None
    assert any('指标语义不一致' in note for note in result['limitations'])
    _no_verdict(result)


@pytest.mark.parametrize('paper_label,actual_metric,source_metric', [
    ('F1 (percent)', 'accuracy', 'f1_score'),
    ('RMSE', 'mse', 'rmse'), ('MSE', 'rmse', 'mse'),
])
def test_model_cannot_relabel_a_different_selected_table_metric(
        practical_sources, paper_label, actual_metric, source_metric):
    _, _, records, payload = practical_sources
    records[0].update(name=actual_metric, unit='scalar')
    payload['comparisons'][0]['reference'].update(name=actual_metric, unit='scalar')
    payload['comparisons'][0]['actual_context'].update(unit='scalar', citations=['readme'])
    _replace_table(practical_sources, f'Method Dataset Split {paper_label}\nSignal Linear SignGrid test 100.0')
    result = _run(practical_sources)
    assert result['reference']['name'] == source_metric
    assert result['reference']['proposed_metric_name'] == actual_metric
    assert result['reference']['metric_evidence']['status'] == 'contradicted'
    assert result['actual']['name'] == actual_metric and result['metric'] == actual_metric
    assert result['reference']['value_quote'] == 'Signal Linear SignGrid test 100.0'
    assert result['actual']['value'] == .94 and result['reference']['value'] == 100.
    assert result['difference'] is None
    _no_verdict(result)


def test_original_table_recovers_a_misnamed_matching_metric_and_its_difference(practical_sources):
    payload = practical_sources[-1]
    payload['comparisons'][0]['reference']['name'] = 'mse'
    _replace_table(practical_sources,
        'Method Dataset Split Accuracy (percent)\nSignal Linear SignGrid test 100.0')
    before = deepcopy(payload)
    result = _run(practical_sources)
    assert result['reference']['name'] == result['actual']['name'] == 'accuracy'
    assert result['reference']['proposed_metric_name'] == 'mse'
    assert result['reference']['metric_evidence']['status'] == 'contradicted'
    assert result['difference'] == pytest.approx(-6.)
    assert result['status'] == 'reference_only'
    assert any('原文更正为 accuracy' in note for note in result['limitations'])
    assert not any('指标语义不一致' in note for note in result['limitations'])
    assert payload == before
    _no_verdict(result)


def test_another_table_or_figure_on_the_same_page_does_not_change_the_metric(practical_sources):
    _replace_table(practical_sources,
        'Table 1: F1 results\nMethod Dataset F1\nOther Model OtherGrid 50.0\n'
        'Figure 1: MSE and RMSE curves\nTable 2: Accuracy results in percent\n'
        'Method Dataset Split Accuracy (percent)\nSignal Linear SignGrid test 100.0')
    result = _run(practical_sources)
    assert result['reference']['metric_evidence']['value'] == 'accuracy'
    assert result['difference'] == pytest.approx(-6.)


def test_wrong_metric_quote_from_another_table_does_not_supply_metric_semantics(practical_sources):
    _replace_table(practical_sources,
        'Table 1: Accuracy results in percent\nOther Model OtherGrid 50.0\n'
        'Table 2: F1 results in percent\nMethod Dataset Split F1 (percent)\nSignal Linear SignGrid test 100.0')
    practical_sources[-1]['comparisons'][0]['reference']['metric_quote'] = 'Table 1: Accuracy results in percent'
    result = _run(practical_sources)
    assert result['reference']['name'] == 'f1_score'
    assert result['difference'] is None


@pytest.mark.parametrize('missing', ['paper', 'actual', 'both'])
def test_empty_units_preserve_raw_values_without_guessing_conversion(practical_sources, missing):
    _, _, records, payload = practical_sources
    records[0]['unit'] = ''
    row = payload['comparisons'][0]
    row['reference']['unit'] = '' if missing in {'paper', 'both'} else 'percent'
    row['actual_context'].update(unit='' if missing in {'actual', 'both'} else 'fraction', citations=['readme'])
    _replace_table(practical_sources, 'Method Dataset Split Accuracy\nSignal Linear SignGrid test 100.0')
    result = _run(practical_sources)
    assert result['status'] == 'reference_only'
    assert result['reference']['value'] == 100. and result['actual']['value'] == .94
    assert result['difference'] is None and result['relative_difference'] is None
    assert result['difference_unit'] == ''
    assert any('单位未明确' in note for note in result['limitations'])
    _no_verdict(result)


def test_nonempty_unverified_source_interpretation_can_compute_a_labelled_reference(practical_sources):
    _, _, records, payload = practical_sources
    records[0]['unit'] = ''
    payload['comparisons'][0]['actual_context']['citations'] = ['readme']
    _replace_table(practical_sources, 'Method Dataset Split Accuracy\nSignal Linear SignGrid test 100.0')
    result = _run(practical_sources)
    assert result['status'] == 'reference_only' and result['difference'] == pytest.approx(-6.)
    assert result['reference']['field_evidence']['unit']['status'] == 'unknown'
    assert result['actual']['field_evidence']['unit']['status'] == 'unknown'
    assert any('按所列单位计算' in note for note in result['limitations'])
    _no_verdict(result)


def test_explicitly_contradicted_unit_never_allows_a_delta(practical_sources):
    practical_sources[-1]['comparisons'][0]['reference']['unit'] = 'fraction'
    result = _run(practical_sources)
    assert result['reference']['field_evidence']['unit']['status'] == 'contradicted'
    assert result['reference']['value'] == 100. and result['actual']['value'] == .94
    assert result['difference'] is None
    _no_verdict(result)


@pytest.mark.parametrize('field,value', [
    ('value', 99.), ('value_quote', 'Signal Linear SignGrid test 99.0'),
])
def test_fake_paper_number_or_quotation_is_still_rejected(practical_sources, field, value):
    practical_sources[-1]['comparisons'][0]['reference'][field] = value
    with pytest.raises(ValueError, match='Reference number'):
        _run(practical_sources)


@pytest.mark.parametrize('field,value', [
    ('unit', 'percent'), ('split', 'validation'), ('value', .1), ('name', 'f1_score'),
    ('std', .1), ('runs', True),
])
def test_model_cannot_override_observed_author_fields(practical_sources, field, value):
    if field == 'unit':
        practical_sources[2][0]['unit'] = 'fraction'
    practical_sources[-1]['comparisons'][0]['actual_context'][field] = value
    with pytest.raises(ValueError, match='observed|explicit output|unobserved|Invalid actual'):
        _run(practical_sources)


def test_optional_bad_context_quote_is_quarantined_without_rewriting_it(practical_sources):
    proposed = 'This invented split explanation does not occur in the PDF.'
    practical_sources[-1]['comparisons'][0]['reference']['context_quotes'] = {'split': proposed}
    result = _run(practical_sources)
    rejected, = result['reference']['rejected_context_quotes']
    assert rejected['proposed_quote'] == proposed
    assert 'split' not in result['reference']['context_quotes']
    assert result['status'] == 'reference_only' and result['difference'] == pytest.approx(-6.)
    _no_verdict(result)


def test_optional_static_method_alignment_is_never_a_gate(practical_sources, monkeypatch):
    import src.author_method_alignment as alignment
    monkeypatch.setattr(alignment, 'align_method_components', lambda *args: {'status': 'unknown', 'reason': 'Unsupported syntax'})
    practical_sources[-1]['comparisons'][0]['actual_context']['method'] = 'Wrapper-Backbone'
    plan = {'steps': [{'id': 'train'}]}
    result = _run(practical_sources, plan=plan)
    assert result['method_alignment']['status'] == 'unknown'
    assert result['status'] == 'reference_only' and result['difference'] == pytest.approx(-6.)
    _no_verdict(result)
