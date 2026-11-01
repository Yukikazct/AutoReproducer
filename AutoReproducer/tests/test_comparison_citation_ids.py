"""Invalid citation aliases need complete source-bound correction feedback."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from src.author_claim_comparison import compare_author_claims, _exposed_validation_retry, _validated_rows
from src.author_comparison_context import comparison_view, prepare_comparison_packet
from test_author_claim_comparison import actual_execution, sources


def unknown_proposal(payload):
    result = deepcopy(payload)
    for index, item in enumerate(result['citations']):
        if item['id'] in {'paper_results', 'script'}:
            result['citations'][index] = {'id': item['id'],
                'evidence_id': 'paper_setup' if item['id'] == 'paper_results' else 'invented_script'}
    return result


def test_all_selected_unknown_ids_are_reported_together_and_corrected_by_the_model(sources):
    fixture, packet, records, payload = sources
    rejected = unknown_proposal(payload)
    before = deepcopy((packet, rejected, payload))
    model = Mock(mock_mode=False)
    model.chat.side_effect = [json.dumps(rejected), json.dumps(payload)]

    result = compare_author_claims(model, packet, fixture['proposal'], fixture['workspace'], actual_execution())

    assert result['status'] == 'compared' and model.chat.call_count == 2
    first = result['extraction_attempts'][0]
    errors = first['validation_errors']
    assert {item['rejected_evidence_id'] for item in errors} == {'paper_setup', 'invented_script'}
    assert {item['citation_id'] for item in errors} == {'paper_results', 'script'}
    assert {item['field'] for item in errors} == {
        'citations.paper_results.evidence_id', 'citations.script.evidence_id'}
    assert first['proposal'] == rejected and first['citation_repairs'] == []
    feedback = model.chat.call_args_list[1].args[0].split('\n\nCorrection feedback', 1)[1]
    assert 'paper_setup' in feedback and 'invented_script' in feedback
    assert 'Change each rejected citations entry evidence_id itself' in feedback
    assert len(feedback) <= 12000
    view = comparison_view(prepare_comparison_packet(packet, fixture['proposal'], fixture['workspace']),
                           fixture['proposal'], records)
    assert first['citation_id_catalog']
    assert all(item['evidence_id'] in view['evidence_excerpts'] for item in first['citation_id_catalog'])
    assert {item['source'] for item in first['citation_id_catalog']} >= {'pdf', 'repository'}
    assert (packet, rejected, payload) == before
    assert result['extraction_attempts'][1]['proposal'] == payload


def test_unknown_ids_are_never_guessed_when_the_model_repeats_its_rejected_proposal(sources):
    fixture, packet, _, payload = sources
    rejected = unknown_proposal(payload)
    model = Mock(mock_mode=False)
    model.chat.return_value = json.dumps(rejected)
    result = compare_author_claims(model, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'unavailable' and result['comparisons'] == []
    assert model.chat.call_count == 2
    assert len(result['validation_errors']) == 2
    assert all(item['proposal'] == rejected and item['citation_repairs'] == []
               for item in result['extraction_attempts'])


def test_unselected_unknown_declaration_does_not_invalidate_a_verified_row(sources):
    fixture, packet, _, payload = sources
    payload['citations'].append({'id': 'unused', 'evidence_id': 'not_a_source'})
    model = Mock(mock_mode=False)
    model.chat.return_value = json.dumps(payload)
    result = compare_author_claims(model, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'compared' and model.chat.call_count == 1
    assert 'citation_id_catalog' not in result['extraction_attempts'][0]


def nonliteral_method(payload):
    result = deepcopy(payload)
    result['comparisons'][0]['actual_context']['method'] = 'Signal Linear (inferred implementation)'
    return result


def assert_warned_method_reference(result):
    assert result['status'] == 'partial' and not result['validation_errors']
    row, = result['comparisons']
    assert row['status'] == 'reference_only'
    assert row['reference']['method'] == 'Signal Linear'
    assert row['actual']['method'] == 'Signal Linear (inferred implementation)'
    assert row['reference']['value_quote'] == 'Signal Linear SignGrid test 100.0'
    assert row['reference']['value'] == 100.0 and row['actual']['value'] == 1.0
    assert row['difference'] == pytest.approx(0.)
    assert any('actual_context.method' in note and '未确认逐字对应' in note
               for note in row['limitations'])
    assert row['within_tolerance'] is None and row['meets_reference'] is None


def test_resolved_ids_keep_nonliteral_method_as_reference_without_a_third_request(sources):
    fixture, packet, _, payload = sources
    first, second = unknown_proposal(payload), nonliteral_method(payload)
    before = deepcopy((packet, first, second, payload))
    model = Mock(mock_mode=False)
    model.chat.side_effect = [json.dumps(first), json.dumps(second), json.dumps(payload)]

    result = compare_author_claims(model, packet, fixture['proposal'], fixture['workspace'], actual_execution())

    assert_warned_method_reference(result)
    assert model.chat.call_count == 2
    attempts = result['extraction_attempts']
    assert [item['proposal'] for item in attempts] == [first, second]
    assert all('additional_correction_reason' not in item for item in attempts)
    assert all(not item['citation_repairs'] for item in attempts)
    assert (packet, first, second, payload) == before


@pytest.mark.parametrize('correction', ['same_label', 'invented_value'])
def test_resolved_ids_keep_value_validation_and_do_not_extend_the_budget(sources, correction):
    fixture, packet, _, payload = sources
    first = unknown_proposal(payload)
    second = nonliteral_method(payload) if correction == 'same_label' else deepcopy(payload)
    if correction == 'invented_value':
        second['comparisons'][0]['reference']['value'] = 91.3
    before = deepcopy((first, second, payload))
    model = Mock(mock_mode=False)
    model.chat.side_effect = [json.dumps(first), json.dumps(second), json.dumps(payload)]

    result = compare_author_claims(model, packet, fixture['proposal'], fixture['workspace'], actual_execution())

    if correction == 'same_label':
        assert_warned_method_reference(result)
    else:
        assert result['status'] == 'unavailable' and not result['comparisons']
        assert any('Reference number' in item['error'] for item in result['validation_errors'])
        assert next(record for record in result['metric_records'] if record['split'] == 'test')['value'] == 1.0
    assert model.chat.call_count == 2
    assert [item['proposal'] for item in result['extraction_attempts']] == [first, second]
    assert all('additional_correction_reason' not in item for item in result['extraction_attempts'])
    assert all(not item['citation_repairs'] for item in result['extraction_attempts'])
    assert (first, second, payload) == before


@pytest.mark.parametrize('proposal_kind', ['nonliteral_method', 'invented_value'])
def test_only_real_validation_failure_uses_two_requests_without_blocked_source_ids(sources, proposal_kind):
    fixture, packet, _, payload = sources
    rejected = nonliteral_method(payload) if proposal_kind == 'nonliteral_method' else deepcopy(payload)
    if proposal_kind == 'invented_value':
        rejected['comparisons'][0]['reference']['value'] = 91.3
    model = Mock(mock_mode=False)
    model.chat.return_value = json.dumps(rejected)
    result = compare_author_claims(model, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    if proposal_kind == 'nonliteral_method':
        assert_warned_method_reference(result)
        assert model.chat.call_count == 1
    else:
        assert result['status'] == 'unavailable' and not result['comparisons']
        assert model.chat.call_count == 2
        assert any('Reference number' in item['error'] for item in result['validation_errors'])
    assert all('additional_correction_reason' not in item for item in result['extraction_attempts'])


def test_final_correction_cannot_add_another_observed_row(sources):
    fixture, packet, records, payload = sources
    final = deepcopy(payload)
    extra = deepcopy(payload['comparisons'][0])
    extra['actual_id'] = records[0]['id']
    extra['actual_context']['split'] = 'validation'
    final['comparisons'].append(extra)
    before = deepcopy(final)
    model = Mock(mock_mode=False)
    model.chat.side_effect = [json.dumps(unknown_proposal(payload)),
                             json.dumps(nonliteral_method(payload)), json.dumps(final)]

    result = compare_author_claims(model, packet, fixture['proposal'], fixture['workspace'], actual_execution())

    assert_warned_method_reference(result)
    assert model.chat.call_count == 2
    assert [row['actual']['id'] for row in result['comparisons']] == [payload['comparisons'][0]['actual_id']]
    assert all(item['proposal'] != final for item in result['extraction_attempts'])
    assert all('additional_correction_actual_ids' not in item for item in result['extraction_attempts'])

    # A harmless inferred label no longer earns the final request. Check the
    # actual correction-scope validator independently so its new-row guard
    # remains covered when a final correction is explicitly authorized.
    accepted, citations = {}, {}
    errors = _validated_rows(final, packet, fixture['workspace'], records, accepted, citations,
        allowed_actual_ids={payload['comparisons'][0]['actual_id']},
        plan=fixture['proposal'], allow_reference_only=True)
    assert list(accepted) == [payload['comparisons'][0]['actual_id']]
    assert len(errors) == 1 and errors[0]['field'] == 'correction_scope'
    assert errors[0]['actual_id'] == records[0]['id']
    assert before == final


@pytest.mark.parametrize('first_row_kind', ['nonliteral_method', 'invented_value'])
def test_duplicate_rows_preserve_accepted_reference_and_need_no_third_request(sources, first_row_kind):
    fixture, packet, _, payload = sources
    second = nonliteral_method(payload) if first_row_kind == 'nonliteral_method' else deepcopy(payload)
    if first_row_kind == 'invented_value':
        second['comparisons'][0]['reference']['value'] = 91.3
    second['comparisons'].append(deepcopy(payload['comparisons'][0]))
    model = Mock(mock_mode=False)
    model.chat.side_effect = [json.dumps(unknown_proposal(payload)), json.dumps(second)]
    result = compare_author_claims(model, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    if first_row_kind == 'nonliteral_method':
        assert_warned_method_reference(result)
    else:
        assert result['status'] == 'compared' and not result['validation_errors']
        assert result['comparisons'][0]['actual']['method'] == 'Signal Linear'
        assert result['comparisons'][0]['reference']['value'] == 100.0
    assert model.chat.call_count == 2
    assert 'additional_correction_reason' not in result['extraction_attempts'][-1]


def test_citation_validation_failure_does_not_unlock_literal_label_correction(sources):
    fixture, packet, _, payload = sources
    malformed = deepcopy(payload)
    next(item for item in malformed['citations'] if item['id'] == 'script')['evidence_id'] = None
    model = Mock(mock_mode=False)
    model.chat.side_effect = [json.dumps(unknown_proposal(payload)), json.dumps(malformed)]
    result = compare_author_claims(model, packet, fixture['proposal'], fixture['workspace'], actual_execution())
    assert result['status'] == 'unavailable' and model.chat.call_count == 2
    assert result['validation_errors']
    assert all('validation_stage' not in item for item in result['validation_errors'])


@pytest.mark.parametrize('current', [
    [],
    [{'actual_id': 'other', 'error': 'actual_context.method rejected'}],
    [{'actual_id': None, 'error': 'malformed row'}],
    [{'actual_id': [], 'error': 'malformed row'}],
    [{'actual_id': 'train:stdout:0', 'rejected_evidence_id': 'still_unknown'}],
    [{'actual_id': 'train:stdout:0', 'error': 'Source hash mismatch'}],
    [{'actual_id': 'train:stdout:0', 'validation_stage': 'context_quotes'}],
])
def test_unrelated_or_unresolved_rows_do_not_extend_the_budget(current):
    attempts = [{'validation_errors': [{'actual_id': 'train:stdout:0',
        'rejected_evidence_id': 'unknown'}]}, {}]
    assert not _exposed_validation_retry(attempts, current)
    assert not _exposed_validation_retry([*attempts, {}], current)
