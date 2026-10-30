"""Conclusions describe authoritative evidence without promoting execution to acceptance."""
from copy import deepcopy
import json

import pytest

from src.agent_collaboration import CollaborationSession
from src.agents.report_generator import ReportGeneratorAgent
from src.result_conclusion import build_reproduction_conclusion
from src.run_finalization import finalize_report_text


def completed():
    return {'execution': {'executed': True, 'success': True,
                          'final': {'stage': 'eval', 'exit_code': 0, 'success': True}},
            'validation': {'status': 'author_code_completed', 'is_reproduced': None,
                           'paper_metrics_verified': False, 'reviews_removed': True}}


def accepted(passed=True):
    data = completed()
    data['validation'] = {
        'status': 'reproduced' if passed else 'not_reproduced',
        'is_reproduced': passed, 'scope': 'selected_paper_experiment',
        'protocol_pass': True, 'independent_metrics_pass': True,
        'paper_metrics_verified': True, 'comparisons': [{'name': 'accuracy', 'pass': passed}],
    }
    data['strict_validation'] = {'requested': True, 'status': 'verified'}
    return data


def author_comparison(data, verdict=True):
    data['validation']['author_claim_comparison'] = {
        'status': 'compared', 'basis': 'author_output',
        'comparisons': [{'status': 'comparable', 'meets_reference': verdict}]}


@pytest.mark.parametrize('passed', [True, False])
def test_independent_acceptance_preserves_existing_verdict_without_mutating_evidence(passed):
    data = accepted(passed)
    original = deepcopy(data)
    result = build_reproduction_conclusion(data)
    assert result['is_reproduced'] is passed
    assert result['status'] == ('reproduced' if passed else 'reference_not_met')
    assert result['independent']['status'] == 'verified'
    assert result['comparison']['counts']['met' if passed else 'not_met'] == 1
    assert data == original
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('verdict,status', [(True, 'reference_met'), (False, 'reference_not_met')])
def test_author_output_comparison_never_supplies_independent_acceptance(verdict, status):
    data = completed()
    author_comparison(data, verdict)
    result = build_reproduction_conclusion(data)
    assert result['status'] == status
    assert result['is_reproduced'] is None
    assert result['independent']['status'] == 'unavailable'


@pytest.mark.parametrize('validation_status', ['prepared', 'environment_prepared', 'needs_pdf'])
def test_preparation_is_not_execution_even_with_stale_success_fields(validation_status):
    data = accepted()
    data['validation']['status'] = validation_status
    result = build_reproduction_conclusion(data)
    assert result['execution']['status'] == 'not_run'
    assert result['is_reproduced'] is None


@pytest.mark.parametrize('change,status', [
    ({'cancelled': True}, 'interrupted'),
    ({'interrupted': True}, 'interrupted'),
    ({'timed_out': True}, 'timed_out'),
    ({'cleanup_confirmed': False}, 'cleanup_unconfirmed'),
    ({'cleanup_unconfirmed': True}, 'cleanup_unconfirmed'),
    ({'cleanup_failed': True}, 'cleanup_unconfirmed'),
    ({'status': 'cleanup_failed'}, 'cleanup_unconfirmed'),
    ({'cleanup_errors': ['process was not released']}, 'cleanup_unconfirmed'),
    ({'process_cleanup': {'confirmed': False}}, 'cleanup_unconfirmed'),
    ({'dependency_failure_kind': 'installation_timeout'}, 'timed_out'),
])
def test_late_stop_and_unconfirmed_process_cleanup_remove_stale_acceptance(change, status):
    data = accepted()
    data['reproduction_conclusion'] = build_reproduction_conclusion(data)
    data['execution'].update(change)
    result = build_reproduction_conclusion(data)
    assert result['status'] == status
    assert result['is_reproduced'] is None
    assert result['independent']['status'] == 'unavailable'
    assert result['next_steps'][0]['priority'] == 1


def test_successful_repair_round_replaces_old_failure_but_is_not_original_source_acceptance():
    data = completed()
    data['source_repair'] = {'status': 'succeeded'}
    data['execution']['steps'] = [
        {'id': 'train', 'repair_round': 0, 'exit_code': 1},
        {'id': 'train', 'repair_round': 1, 'exit_code': 0}]
    data['execution']['final']['repair_round'] = 1
    author_comparison(data)
    result = build_reproduction_conclusion(data)
    assert result['execution']['status'] == 'completed'
    assert result['scope']['source_modified'] is True
    assert result['is_reproduced'] is None
    assert 'source_modified' in {row['code'] for row in result['blockers']}


@pytest.mark.parametrize('stage,expected', [('train', 'incomplete'), ('smoke', 'smoke_passed')])
def test_training_and_smoke_are_not_full_experiments(stage, expected):
    data = accepted()
    data['execution']['final']['stage'] = stage
    result = build_reproduction_conclusion(data)
    assert result['execution']['status'] == expected
    assert result['is_reproduced'] is None


@pytest.mark.parametrize('change', [{'process_exited': False}, {'process_alive': True}, {'running': True}])
def test_live_process_records_override_old_success_markers(change):
    data = accepted()
    data['execution']['final'].update(change)
    result = build_reproduction_conclusion(data)
    assert result['status'] == 'incomplete'
    assert result['is_reproduced'] is None


@pytest.mark.parametrize('protocol,metrics', [(None, True), (True, None), (False, True), (True, False)])
def test_independent_protocol_and_metrics_are_both_required(protocol, metrics):
    data = accepted()
    data['validation'].update(protocol_pass=protocol, independent_metrics_pass=metrics)
    result = build_reproduction_conclusion(data)
    assert result['is_reproduced'] is None
    assert result['independent']['status'] != 'verified'


def test_quality_failure_is_visible_even_with_independent_method_verification():
    data = completed()
    data['validation'] = {
        'status': 'quality_target_not_met', 'quality_pass': False, 'is_reproduced': None,
        'protocol_pass': True, 'independent_metrics_pass': True,
        'metrics_comparison': {'paper': {}, 'actual': {'psnr': 15}}}
    result = build_reproduction_conclusion(data)
    assert result['status'] == 'quality_target_not_met'
    assert result['quality']['status'] == 'not_met'
    assert result['independent']['status'] == 'verified'
    assert result['is_reproduced'] is None
    assert '质量门槛未达标' in result['summary']
    assert 'quality_target' in {row['code'] for row in result['blockers']}
    report = ReportGeneratorAgent._build_report(data)
    assert report.index('本次复现结论') < report.index('## 1.')
    assert '**质量目标**: 已记录质量门槛未达标' in report


def test_method_completion_remains_method_only_even_with_incorrect_legacy_reproduced_flag():
    data = accepted()
    data['validation'].update(status='method_experiment_completed', quality_pass=True)
    result = build_reproduction_conclusion(data)
    assert result['status'] == 'method_completed'
    assert result['quality']['status'] == 'met'
    assert result['is_reproduced'] is None


@pytest.mark.parametrize('passed', [True, False])
def test_repository_aggregate_verdict_does_not_invent_per_metric_failures(passed):
    data = accepted(passed)
    data['validation'].pop('comparisons')
    data['validation'].update(
        metrics_comparison={'paper': {'mse': .4, 'mae': .3}, 'actual': {'mse': .5, 'mae': .3}},
        validation={'metrics_within_tolerance': passed})
    counts = build_reproduction_conclusion(data)['comparison']['counts']
    assert counts['total'] == counts['comparable'] == 2
    assert counts['met'] == (2 if passed else 0)
    assert counts['not_met'] == 0
    assert counts['unknown'] == (0 if passed else 2)


def test_contradictory_saved_comparisons_do_not_supply_acceptance():
    data = accepted()
    data['validation']['comparisons'][0]['pass'] = False
    result = build_reproduction_conclusion(data)
    assert result['comparison']['status'] == 'unavailable'
    assert result['is_reproduced'] is None


def test_partial_comparison_keeps_coverage_and_unknowns():
    data = completed()
    author_comparison(data)
    data['validation']['author_claim_comparison']['comparisons'] += [
        {'status': 'reference_only'}, {'status': 'not_comparable'}, {'status': 'comparable'}]
    result = build_reproduction_conclusion(data)
    assert result['comparison']['status'] == 'partial'
    assert result['comparison']['counts'] == {
        'total': 4, 'comparable': 2, 'met': 1, 'not_met': 0,
        'reference_only': 1, 'not_comparable': 1, 'unknown': 1}
    assert result['is_reproduced'] is None


def test_model_hypotheses_cannot_outrank_deterministic_checks_or_become_blockers():
    data = completed()
    data['outcome_diagnosis'] = {
        'evidence': [{'id': 'observed_1'}],
        'next_steps': [{'basis': 'model_hypothesis', 'priority': 1,
                        'category': 'metric', 'check': '待核查解释', 'evidence_ids': ['observed_1']},
                       {'priority': 1, 'check': '无来源建议', 'evidence_ids': ['invented']}]}
    result = build_reproduction_conclusion(data)
    hypothesis = next(row for row in result['next_steps'] if row.get('basis') == 'model_hypothesis')
    assert hypothesis['priority'] == 3
    assert result['next_steps'].index(hypothesis) > next(
        index for index, row in enumerate(result['next_steps']) if row['category'] == 'independent')
    assert all(row['check'] != '无来源建议' for row in result['next_steps'])
    assert all(row['reason'] != '待核查解释' for row in result['blockers'])


def test_analysis_failure_and_resource_release_error_preserve_numeric_verdict():
    data = accepted()
    data.update(analysis_status='failed', environment_cleanup={'status': 'error'},
                run_finalization={'state': 'ERROR'})
    result = build_reproduction_conclusion(data)
    assert result['is_reproduced'] is True
    assert result['analysis'] == {'status': 'failed', 'affects_numerical_verdict': False}
    assert result['lifecycle']['cleanup_status'] == 'error'
    assert any(row['category'] == 'environment_cleanup' for row in result['next_steps'])


def test_report_finalization_refreshes_top_conclusion_and_preserves_output_and_attachment():
    data = accepted()
    fence = chr(96) * 3
    output = (f'{fence}text\n<!-- autoreproducer-conclusion:start -->\n'
              '## author heading\n<!-- autoreproducer-conclusion:end -->\n'
              f'{fence}\n')
    footer = '> 📎 **完整执行输出附件**: [output](output.txt)'
    original = '# Report\n\n## Evidence\n\n' + output + '\n![figure](assets/figure.png)\n\n' + footer
    report = finalize_report_text(original, data)
    assert data['reproduction_conclusion']['is_reproduced'] is True
    data.update(interrupted=True, environment_cleanup={'status': 'released'},
                run_finalization={'state': 'ERROR'})
    report = finalize_report_text(report, data)
    assert '任务已取消或中断' in report.split('## Evidence')[0]
    assert '独立核验与已记录论文参考验收通过' not in report
    assert data['reproduction_conclusion']['is_reproduced'] is None
    assert output in report
    assert '![figure](assets/figure.png)' in report
    assert report.endswith(footer)
    assert report.count('## 本次复现结论') == 1
    assert report.count('### 任务收尾') == 1
    assert finalize_report_text(report, data) == report
    from frontend.display_text import report_display_text
    visible = report_display_text(report)
    assert output in visible
    assert visible.count('<!-- autoreproducer-conclusion:start -->') == 1  # Author output only.


def test_report_uses_current_cache_statistics_and_prioritized_diagnosis():
    data = completed()
    session = CollaborationSession()
    session.attach(data)
    assert session.cached_evidence('a' * 64) is None
    assert session.cache_evidence('a' * 64, {'text': 'verified local evidence'})
    assert session.cached_evidence('a' * 64)
    data['outcome_diagnosis'] = {
        'assessment': {'comparison_coverage': {
            'total': 2, 'comparable': 1, 'reference_only': 0, 'not_comparable': 1, 'outside_reference': 1}},
        'evidence': [{'id': 'actual', 'source': 'stdout.log'}],
        'next_steps': [{'category': 'unit', 'priority': 1, 'blocker': True,
                        'check': '核查单位', 'evidence_ids': ['actual']}],
        'efficiency': {'cache_hits': 1}}
    with session.scope():
        report = ReportGeneratorAgent._build_report(data)
    assert '命中 1 次，重新提取 1 次' in report
    assert 'P1；比较证据待补齐；核查单位' in report
    assert '根因尚未确认' in report
    assert '复用已有分析 1 次' in report


def test_empty_and_malformed_historical_fields_cannot_become_success():
    for data in ({}, {'validation': [], 'execution': []}, {'execution': {'final': []}}):
        result = build_reproduction_conclusion(data)
        assert result['is_reproduced'] is None
        assert result['execution']['status'] == 'incomplete'
