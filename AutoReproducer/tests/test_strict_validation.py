"""Strict mode must preserve source bindings and never promote author output."""
from copy import deepcopy

import pytest

from discovered_fixtures import PlannedLLM, accepted_review, fixture_protocol, make_experiment
from src import discovered_repository_plan as planner
from src.agents.experiment_protocol import validate_experiment_protocol
from src.repository_reproduction import spec_digest
from src.strict_validation import (
    assert_strict_plan, finalize_strict_validation, normalize_strict_validation_request,
    verify_strict_execution,
)


@pytest.fixture
def strict_plan(tmp_path):
    fixture = make_experiment(tmp_path)
    packet = planner.build_evidence_packet(fixture['pdf'], fixture['workspace'], fixture['snapshot'])
    packet['author_execution_only'] = False
    raw = fixture_protocol(fixture['proposal'], planner._planning_view(packet))
    packet['experiment_protocol'] = validate_experiment_protocol(raw, packet, fixture['workspace'])
    plan = planner.propose_plan(PlannedLLM(fixture['proposal']), packet, fixture['workspace'])
    return fixture, packet, plan


def rehash(plan):
    body = {key: value for key, value in plan.items() if key != 'plan_sha256'}
    plan['plan_sha256'] = planner._digest(planner._canonical(body).encode('utf-8'))


def complete_result(plan, matched=True):
    return {'status': 'reproduced' if matched else 'reference_not_met',
            'is_reproduced': matched, 'protocol_pass': True, 'independent_metrics_pass': True,
            'comparisons': [{**metric, 'pass': matched,
                'actual': metric['reference'] if matched else metric['reference'] * 0.8,
                'absolute_tolerance': abs(metric['reference']) * metric['tolerance_relative']}
                for metric in plan['metrics']],
            'completion_checks': {key: True for key in (
                'required_processes_exited', 'full_training_and_test_capture',
                'all_native_runs_evaluated', 'fresh_model_predictions_verified',
                'author_metric_reconciled')} | {'paper_reference_met': matched}}


def test_mode_is_explicit_boolean_idempotent_and_default_stays_author_execution():
    default = normalize_strict_validation_request({})
    assert default['strict_validation'] is False
    assert default['author_execution_only'] is True
    strict = normalize_strict_validation_request({'strict_validation': True,
        'author_execution_only': True, 'use_llm_review': False})
    assert strict['author_execution_only'] is False
    assert strict['use_llm_review'] is False
    assert strict['allow_result_summary_review'] is False
    assert normalize_strict_validation_request(strict) == strict


@pytest.mark.parametrize('value', [None, 'true', 1, 0, [], {}])
def test_malformed_strict_mode_never_silently_changes_scope(value):
    with pytest.raises(ValueError, match='explicit boolean'):
        normalize_strict_validation_request({'strict_validation': value})


@pytest.mark.parametrize('field,value', [('mock_mode', True), ('code', 'print(1)'),
    ('corpus_paper', 'fixture'), ('experiment_profile', 'preset')])
def test_unsupported_strict_entry_cannot_silently_select_another_workflow(field, value):
    with pytest.raises(ValueError, match='generic repository'):
        normalize_strict_validation_request({'strict_validation': True, field: value})


def test_valid_review_and_protocol_are_rechecked_without_model_calls(strict_plan):
    fixture, packet, plan = strict_plan
    original = deepcopy((packet, plan))
    receipt = assert_strict_plan(plan, packet, fixture['workspace'])
    assert receipt['status'] == 'verified'
    assert receipt['protocol_sha256'] == packet['experiment_protocol']['protocol_sha256']
    assert receipt['reviewed_plan_sha256'] == plan['semantic_review']['reviewed_plan_sha256']
    assert (packet, plan) == original


@pytest.mark.parametrize('mutation,match', [
    (lambda p: p.pop('semantic_review'), 'plan_sha256'),
    (lambda p: p['semantic_review'].update(reviewed_plan_sha256='0' * 64), 'does not bind'),
    (lambda p: p['semantic_review']['checks'].pop('same_experiment'), 'required protocol check'),
    (lambda p: p['semantic_review']['checks']['capture_uses_test_data'].update(passed=False), 'did not establish'),
    (lambda p: p['semantic_review'].update(evidence_sha256='0' * 64), 'source-bound semantic review'),
    (lambda p: p['capture'].update(expected_optimizer_steps=1), 'does not bind'),
    (lambda p: p['metrics'][0].update(reference=1), 'does not bind'),
])
def test_missing_or_tampered_review_cannot_pass_even_with_rehashed_plan(strict_plan, mutation, match):
    fixture, packet, plan = strict_plan
    mutation(plan)
    if 'semantic_review' in plan:
        rehash(plan)
    with pytest.raises(planner.PlanEvidenceError, match=match):
        assert_strict_plan(plan, packet, fixture['workspace'])


def test_source_change_is_fatal_even_if_old_runtime_artifacts_exist(strict_plan):
    fixture, packet, plan = strict_plan
    (fixture['workspace'] / 'train.py').write_text('print("forged")\n', encoding='utf-8')
    with pytest.raises(planner.PlanEvidenceError, match='changed|hash'):
        assert_strict_plan(plan, packet, fixture['workspace'])


def test_protocol_and_packet_changes_cannot_borrow_previous_review(strict_plan):
    fixture, packet, plan = strict_plan
    packet['experiment_protocol']['model_selection']['evaluation_weights'] = 'Use a different model'
    plan['experiment_protocol'] = deepcopy(packet['experiment_protocol'])
    rehash(plan)
    with pytest.raises(planner.PlanEvidenceError, match='protocol_sha256'):
        assert_strict_plan(plan, packet, fixture['workspace'])


def test_original_ids_added_by_review_retain_exact_candidate_binding(strict_plan):
    fixture, packet, plan = strict_plan
    candidate = deepcopy(plan)
    candidate.pop('semantic_review')
    candidate['citations'] = candidate['citations'][:]
    rehash(candidate)
    original_id = next(key for key, ref in planner._planning_excerpts(packet).items()
                       if ref['source'] == 'pdf')
    review = accepted_review()
    review['checks']['same_experiment']['citations'].append(original_id)
    reviewed = planner.review_plan(PlannedLLM(fixture['proposal'], review), candidate, packet,
                                   fixture['workspace'])
    assert len(reviewed['citations']) > len(candidate['citations'])
    assert assert_strict_plan(reviewed, packet, fixture['workspace'])['status'] == 'verified'


@pytest.mark.parametrize('matched', [True, False])
def test_verified_metrics_differ_from_meeting_the_paper_reference(strict_plan, matched):
    fixture, packet, plan = strict_plan
    result = complete_result(plan, matched)
    verified = finalize_strict_validation(result, plan, packet, fixture['workspace'])
    assert verified['paper_metrics_verified'] is True
    assert verified['is_reproduced'] is matched
    assert verified['reproduction_scope'] == 'selected_paper_experiment'
    assert verified['pre_execution_review'] == 'verified'
    assert 'paper_metrics_verified' not in result


@pytest.mark.parametrize('gate', ['required_processes_exited', 'full_training_and_test_capture',
    'all_native_runs_evaluated', 'fresh_model_predictions_verified', 'author_metric_reconciled'])
def test_incomplete_execution_cannot_be_promoted(strict_plan, gate):
    fixture, packet, plan = strict_plan
    result = complete_result(plan)
    result['completion_checks'][gate] = False
    with pytest.raises(planner.PlanEvidenceError, match='complete independent execution'):
        finalize_strict_validation(result, plan, packet, fixture['workspace'])


def test_wrapper_rejects_wrong_runtime_spec_before_reading_runtime_artifacts(strict_plan):
    fixture, packet, plan = strict_plan
    with pytest.raises(planner.PlanEvidenceError, match='different frozen specification'):
        verify_strict_execution(plan, packet, fixture['workspace'], {}, '0' * 64)


def test_dependency_warning_is_deferred_only_until_real_independent_execution(strict_plan):
    fixture, packet, plan = strict_plan
    candidate = deepcopy(plan)
    candidate.pop('semantic_review')
    rehash(candidate)
    review = accepted_review()
    review['checks']['dependency_compatibility'].update(passed=False, severity='warning')
    reviewed = planner.review_plan(PlannedLLM(fixture['proposal'], review), candidate, packet,
                                   fixture['workspace'])
    receipt = assert_strict_plan(reviewed, packet, fixture['workspace'])
    assert receipt['deferred_checks'] == ['dependency_compatibility']
    result = complete_result(reviewed)
    assert finalize_strict_validation(result, reviewed, packet,
        fixture['workspace'])['paper_metrics_verified'] is True
    result['independent_metrics_pass'] = False
    with pytest.raises(planner.PlanEvidenceError, match='complete independent execution'):
        finalize_strict_validation(result, reviewed, packet, fixture['workspace'])


@pytest.mark.parametrize('field,value', [('actual', float('nan')), ('actual', 0),
    ('reference', 0), ('unit', 'fraction'), ('absolute_tolerance', 100)])
def test_forged_comparison_flags_cannot_replace_frozen_numerical_evidence(strict_plan, field, value):
    fixture, packet, plan = strict_plan
    result = complete_result(plan)
    result['comparisons'][0][field] = value
    with pytest.raises(planner.PlanEvidenceError, match='Strict comparison'):
        finalize_strict_validation(result, plan, packet, fixture['workspace'])


def mock_runtime_result(monkeypatch, plan, capture_pid=101, evaluation_pid=202, linked_pid=101):
    """Only exercise wrapper process binding; native tests run real processes."""
    from unittest.mock import Mock
    result = complete_result(plan)
    result['capture'] = {'process_pid': capture_pid}
    result['independent_metrics'] = {'process_pid': evaluation_pid, 'capture_process_pid': linked_pid}
    verify = Mock(return_value=result)
    monkeypatch.setattr('src.discovered_repository_runtime.verify_capture', verify)
    return result, verify


@pytest.mark.parametrize('capture_pid,evaluation_pid,linked_pid', [
    (None, 202, 101), ('101', 202, 101), (True, 202, 1), (0, 202, 0), (-101, 202, -101),
    (101, None, 101), (101, '202', 101), (101, True, 101), (101, 0, 101), (101, -202, 101),
    (101, 101, 101), (101, 202, None), (101, 202, '101'), (101, 202, 999),
    (101, 202, 101.0), (1, 202, True),
])
def test_strict_wrapper_rejects_malformed_same_or_mismatched_process_evidence(
        strict_plan, monkeypatch, capture_pid, evaluation_pid, linked_pid):
    fixture, packet, plan = strict_plan
    _, verify = mock_runtime_result(monkeypatch, plan, capture_pid, evaluation_pid, linked_pid)
    execution, spec_hash = {'owned_execution': 'fixture'}, spec_digest(plan)
    with pytest.raises(planner.PlanEvidenceError, match='separate recorded process'):
        verify_strict_execution(plan, packet, fixture['workspace'], execution, spec_hash)
    verify.assert_called_once_with(plan, fixture['workspace'], execution, spec_hash)


@pytest.mark.parametrize('record,field', [('capture', 'process_pid'),
    ('independent_metrics', 'process_pid'), ('independent_metrics', 'capture_process_pid')])
def test_strict_wrapper_rejects_missing_process_receipt_fields(strict_plan, monkeypatch, record, field):
    fixture, packet, plan = strict_plan
    result, _ = mock_runtime_result(monkeypatch, plan)
    result[record].pop(field)
    with pytest.raises(planner.PlanEvidenceError, match='separate recorded process'):
        verify_strict_execution(plan, packet, fixture['workspace'], {}, spec_digest(plan))


@pytest.mark.parametrize('matched', [True, False])
def test_distinct_recorded_processes_keep_verified_and_reference_outcomes_separate(
        strict_plan, monkeypatch, matched):
    fixture, packet, plan = strict_plan
    result, verify = mock_runtime_result(monkeypatch, plan)
    measured = complete_result(plan, matched)
    result.update(measured)
    verified = verify_strict_execution(plan, packet, fixture['workspace'], {}, spec_digest(plan))
    assert verified['paper_metrics_verified'] is True
    assert verified['is_reproduced'] is matched
    assert verified['capture']['process_pid'] == 101
    assert verified['independent_metrics']['process_pid'] == 202
    verify.assert_called_once()
