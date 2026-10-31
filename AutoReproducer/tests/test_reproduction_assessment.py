"""Ordinary AI assessment uses measured evidence and explained tolerances."""
from copy import deepcopy
import json

import pytest

from src.reproduction_assessment import assess_reproduction, apply_reproduction_assessment


class Model:
    mock_mode = False
    api_key = 'synthetic-assessment-credential'

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return json.dumps(response, ensure_ascii=False) if isinstance(response, dict) else response


def execution():
    return {'success': True, 'executed': True,
            'final': {'success': True, 'executed': True, 'exit_code': 0},
            'steps': [{'id': 'train', 'success': True, 'executed': True, 'exit_code': 0,
                       'argv': ['python', 'citation.py', '--dataset', 'cora'],
                       'stdout': 'Test Accuracy: 0.81', 'stderr': ''}]}


def comparison():
    return {'status': 'partial', 'validation_errors': [], 'relative_tolerance': 0.05,
            'comparisons': [{
                'metric': 'accuracy', 'status': 'reference_only', 'direction': 'higher',
                'reference': {'name': 'accuracy', 'value': 83.0, 'unit': 'percent',
                              'method': 'SGC', 'dataset': 'Cora', 'split': 'test',
                              'aggregation': 'mean', 'runs': 10, 'std': 0.2},
                'actual': {'id': 'train:stdout:1', 'value': 0.81, 'unit': 'fraction',
                           'method': 'SGC', 'dataset': 'Cora', 'split': 'test',
                           'aggregation': 'single', 'runs': 1,
                           'source': 'train:stdout:1', 'source_quote': 'Test Accuracy: 0.81'},
                'difference': -2.0, 'relative_difference': -2 / 83,
                'difference_unit': 'percentage_points',
                'within_tolerance': None, 'meets_reference': None,
                'settings': {'status': 'unknown', 'reason': '硬件与随机种子未独立复核'},
                'limitations': ['论文为 10 次均值，本次为单次结果。']} ]}


def proposal(verdict='reproduced'):
    return {'verdict': verdict, 'reason': '已完成作者实验，指标差异在本次任务可接受范围内。',
            'confidence': 0.86, 'limitations': ['未进行独立复算；本次使用单次结果。'],
            'metric_assessments': [{'actual_id': 'train:stdout:1', 'verdict': verdict,
                'reason': '实测 81%，论文均值 83%，相差 2 个百分点。',
                'tolerance': {'absolute': 3.0, 'relative': None,
                              'reason': '根据小数据集波动和实验设置允许 3 个百分点差异。'}}]}


def test_ai_can_accept_single_run_reference_only_without_independent_strict_verification():
    source, run = comparison(), execution()
    originals = deepcopy((source, run))
    model = Model(proposal())
    result = assess_reproduction(model, source, run, validation={'paper_metrics_verified': False})
    assert result['status'] == 'completed' and result['verdict'] == 'reproduced'
    assert result['source'] == 'ai' and result['scope'] == 'executed_experiment'
    metric = result['metric_assessments'][0]
    assert metric['actual'] == source['comparisons'][0]['actual']
    assert metric['reference'] == source['comparisons'][0]['reference']
    assert metric['difference'] == -2.0 and metric['difference_unit'] == 'percentage_points'
    assert metric['metric'] == 'accuracy'
    assert '论文为 10 次均值，本次为单次结果。' in result['limitations']
    assert any('本次实际执行' in item for item in result['limitations'])
    assert (source, run) == originals
    assert model.calls[0][1]['task'] == 'reproduction_assessment'
    assert len(model.calls) == 1


def test_no_fixed_five_percent_gate_overrides_ai_tolerance():
    source = comparison()
    source['comparisons'][0].update(difference=-9.0, relative_difference=-0.1,
                                    within_tolerance=False, meets_reference=False)
    source['comparisons'][0]['reference']['value'] = 90.0
    response = proposal()
    response['metric_assessments'][0]['tolerance'].update(absolute=None, relative=0.12)
    result = assess_reproduction(Model(response), source, execution())
    assert result['status'] == 'completed' and result['verdict'] == 'reproduced'


def test_complete_owned_steps_do_not_require_redundant_top_level_execution_summary():
    run = execution()
    run.pop('executed')
    run.pop('final')
    result = assess_reproduction(Model(proposal()), comparison(), run)
    assert result['status'] == 'completed' and result['verdict'] == 'reproduced'


def test_ai_receives_compatibility_patch_and_runtime_uncertainty_context():
    source = comparison()
    source.update(basis='patched_author_output', source_modified=True,
                  source_repair_manifest_sha256='a' * 64)
    model = Model(proposal())
    result = assess_reproduction(model, source, execution(), validation={
        'source_modified': True, 'warnings': ['已使用标准库兼容补丁'],
        'dataset_observation_verified': False, 'paper_metrics_verified': False})
    assert result['status'] == 'completed'
    evidence = json.loads(model.calls[0][0].split('已验证证据：\n')[1])
    assert evidence['output_provenance']['basis'] == 'patched_author_output'
    assert evidence['output_provenance']['source_modified'] is True
    assert evidence['execution_context']['warnings'] == ['已使用标准库兼容补丁']
    assert evidence['execution_context']['dataset_observation_verified'] is False


def test_missing_summary_without_step_execution_evidence_stays_blocked():
    run = execution()
    run.pop('executed')
    run.pop('final')
    run['steps'][0].pop('executed')
    model = Model(proposal())
    result = assess_reproduction(model, comparison(), run)
    assert result['status'] == 'blocked' and model.calls == []


@pytest.mark.parametrize('direction,delta', [('higher', 15), ('lower', -15)])
def test_directional_improvement_can_exceed_tolerance(direction, delta):
    source = comparison()
    source['comparisons'][0].update(direction=direction, difference=delta, relative_difference=delta / 83)
    result = assess_reproduction(Model(proposal()), source, execution())
    assert result['verdict'] == 'reproduced' and result['status'] == 'completed'


def test_reproduced_must_agree_with_the_models_own_declared_tolerance():
    response = proposal()
    response['metric_assessments'][0]['tolerance']['absolute'] = 0.1
    model = Model(response, response)
    result = assess_reproduction(model, comparison(), execution())
    assert result['status'] == 'unavailable' and len(model.calls) == 2
    assert '超过' in result['attempts'][0]['error']


def test_float_roundoff_at_ai_tolerance_boundary_does_not_change_verdict():
    source = comparison()
    source['comparisons'][0].update(difference=-2.0000000000000004)
    response = proposal()
    response['metric_assessments'][0]['tolerance']['absolute'] = 2.0
    result = assess_reproduction(Model(response), source, execution())
    assert result['status'] == 'completed' and result['verdict'] == 'reproduced'


@pytest.mark.parametrize('mutate', [
    lambda run: run.update(executed=False),
    lambda run: run.update(success=False),
    lambda run: run.update(mock_mode=True),
    lambda run: run['final'].update(source='mock_demo'),
    lambda run: run['steps'][0].update(exit_code=1),
    lambda run: run.update(cancelled=True),
    lambda run: run['final'].update(timed_out=True),
    lambda run: run['steps'][0].update(interrupted=True),
    lambda run: run.update(stages=[{'environment_cleanup': {'process_cleanup_confirmed': False}}]),
    lambda run: run.update(stages={'train': {'cleanup_receipt': {'state': 'cleanup_unconfirmed'}}}),
    lambda run: run['final'].update(cleanup_receipt={'processes_alive': [123]}),
    lambda run: run.update(environment_cleanup={'cleanup_errors': ['descendant still alive']}),
    lambda run: run.update(stages=[{'exit_code': 1}]),
])
def test_incomplete_mock_cancelled_or_unclean_execution_cannot_pass(mutate):
    run = execution()
    mutate(run)
    model = Model(proposal())
    result = assess_reproduction(model, comparison(), run)
    assert result['status'] == 'blocked' and result['verdict'] == 'insufficient_evidence'
    assert not model.calls


@pytest.mark.parametrize('field,value', [('method', 'SGC+'), ('dataset', 'CiteSeer'),
                                         ('split', 'validation'), ('method', ''), ('dataset', None)])
def test_incompatible_identity_cannot_pass_even_with_comparable_label(field, value):
    source = comparison()
    source['comparisons'][0]['status'] = 'comparable'
    source['comparisons'][0]['actual'][field] = value
    model = Model(proposal())
    result = assess_reproduction(model, source, execution())
    assert result['status'] == 'blocked' and not model.calls


@pytest.mark.parametrize('mutate', [
    lambda source: source.update(comparisons=[]),
    lambda source: source['comparisons'][0].update(status='not_comparable'),
    lambda source: source['comparisons'][0]['actual'].update(value=float('nan')),
    lambda source: source['comparisons'][0]['reference'].update(value=float('inf')),
    lambda source: source['comparisons'][0].update(difference=None),
    lambda source: source['comparisons'][0]['actual'].pop('id'),
    lambda source: source['comparisons'].append(deepcopy(source['comparisons'][0])),
])
def test_absent_unpairable_nonfinite_or_ambiguous_measurements_do_not_call_model(mutate):
    source = comparison()
    mutate(source)
    model = Model(proposal())
    result = assess_reproduction(model, source, execution())
    assert result['status'] == 'blocked' and not model.calls


def test_noncomparable_row_cannot_be_successful_alongside_a_pairable_row():
    source = comparison()
    second = deepcopy(source['comparisons'][0])
    second['actual']['id'] = 'train:stdout:2'
    second.update(status='not_comparable', difference=None, relative_difference=None)
    source['comparisons'].append(second)
    bad, good = proposal(), proposal('partially_reproduced')
    bad['metric_assessments'].append({**deepcopy(bad['metric_assessments'][0]), 'actual_id': 'train:stdout:2'})
    good['metric_assessments'][0]['verdict'] = 'reproduced'
    good['metric_assessments'].append({'actual_id': 'train:stdout:2', 'verdict': 'insufficient_evidence',
        'reason': '来源未完成匹配，不应判断成功。',
        'tolerance': {'absolute': None, 'relative': None, 'reason': '暂无可信可配对数值。'}})
    model = Model(bad, good)
    result = assess_reproduction(model, source, execution())
    assert result['status'] == 'completed' and result['verdict'] == 'partially_reproduced'
    assert len(model.calls) == 2
    assert result['attempts'][0]['status'] == 'validation_error'


@pytest.mark.parametrize('mutate', [
    lambda response: response.update(verdict='strict_success'),
    lambda response: response.update(verdict={}),
    lambda response: response.update(reason=''),
    lambda response: response.update(confidence=True),
    lambda response: response.update(confidence=float('nan')),
    lambda response: response.update(confidence=1.1),
    lambda response: response.update(limitations='unknown'),
    lambda response: response.update(metric_assessments=[]),
    lambda response: response['metric_assessments'][0].update(actual_id='invented'),
    lambda response: response['metric_assessments'][0].update(verdict=[]),
    lambda response: response['metric_assessments'][0].update(reason=''),
    lambda response: response['metric_assessments'][0].update(tolerance={}),
    lambda response: response['metric_assessments'][0]['tolerance'].update(relative=-0.1),
    lambda response: response['metric_assessments'][0]['tolerance'].update(absolute=float('inf')),
    lambda response: response['metric_assessments'][0]['tolerance'].update(absolute=None),
    lambda response: response['metric_assessments'][0].update(actual={'value': 0.99}),
    lambda response: response['metric_assessments'][0].update(difference=0),
    lambda response: response['metric_assessments'][0].update(verdict='not_reproduced'),
    lambda response: response.update(verdict='not_reproduced'),
    lambda response: response.update(verdict='insufficient_evidence'),
])
def test_bad_structure_invention_or_contradiction_gets_one_correction(mutate):
    response = proposal()
    mutate(response)
    model = Model(response, proposal())
    result = assess_reproduction(model, comparison(), execution())
    assert result['status'] == 'completed' and len(model.calls) == 2
    assert result['attempts'][0]['status'] == 'validation_error'
    assert result['attempts'][1]['status'] == 'validated'


def test_invalid_json_is_corrected_with_bounded_redacted_feedback_and_audit():
    credential = Model.api_key
    huge_invalid = '{"reason":"' + credential + ' sk-abcdefghijklmnop Bearer abcdefg ' + ('X' * 50000)
    model = Model(huge_invalid, proposal())
    source = comparison()
    source['comparisons'][0]['actual']['source_quote'] += ' Authorization: Bearer abcdefg ' + credential
    result = assess_reproduction(model, source, execution())
    assert result['status'] == 'completed' and len(model.calls) == 2
    first = result['attempts'][0]
    assert first['status'] == 'invalid_response' and first['response_truncated']
    assert first['json_error']['line'] == 1
    assert len(first['response_excerpt']) <= 6000
    assert len(model.calls[1][0]) - len(model.calls[0][0]) <= 10000
    recorded = json.dumps({'result': result, 'calls': model.calls}, ensure_ascii=False)
    assert credential not in recorded and 'sk-abcdefghijklmnop' not in recorded and 'Bearer abcdefg' not in recorded
    assert '[REDACTED]' in recorded


def test_api_configuration_errors_are_unavailable_without_useless_second_call():
    model = Model(RuntimeError('invalid api_key=' + Model.api_key), proposal())
    result = assess_reproduction(model, comparison(), execution())
    assert result['status'] == 'unavailable' and len(model.calls) == 1
    assert result['attempts'][0]['status'] == 'api_error'
    assert Model.api_key not in repr(result)


def test_two_invalid_responses_stop_and_leave_no_partial_unvalidated_verdict():
    model = Model('invalid', 'still invalid', proposal())
    result = assess_reproduction(model, comparison(), execution())
    assert result['status'] == 'unavailable' and result['verdict'] == 'insufficient_evidence'
    assert not result['metric_assessments'] and len(model.calls) == 2


@pytest.mark.parametrize('model', [None, type('MockModel', (), {'mock_mode': True})()])
def test_no_real_model_is_unavailable(model):
    assert assess_reproduction(model, comparison(), execution())['status'] == 'unavailable'


@pytest.mark.parametrize('verdict,expected,reproduced', [
    ('reproduced', 'ai_reproduced', True),
    ('partially_reproduced', 'ai_partially_reproduced', None),
    ('not_reproduced', 'ai_not_reproduced', False),
    ('insufficient_evidence', 'ai_insufficient_evidence', None),
])
def test_apply_updates_ai_result_without_overwriting_execution_or_strict_evidence(verdict, expected, reproduced):
    strict = {'status': 'incomplete', 'reason': '未独立复算'}
    validation = {'status': 'author_code_completed', 'execution_status': 'author_code_completed',
                  'result_level': 'author_code_completed', 'is_reproduced': None,
                  'paper_metrics_verified': False, 'strict_validation': strict,
                  'strict_reproduction_status': 'pending', 'completion_checks': {'owned_processes_finished': True}}
    assessment = {'status': 'completed', 'verdict': verdict, 'reason': '本次实验的 AI 结论'}
    returned = apply_reproduction_assessment(validation, assessment)
    assert returned is validation
    assert validation['status'] == validation['result_level'] == expected
    assert validation['is_reproduced'] is reproduced and validation['verdict_source'] == 'ai'
    assert validation['reason'] == assessment['reason']
    assert validation['execution_status'] == 'author_code_completed'
    assert validation['strict_validation'] is strict
    assert validation['strict_reproduction_status'] == 'pending'
    assert validation['paper_metrics_verified'] is False
    assert validation['completion_checks'] == {'owned_processes_finished': True}
    assessment['reason'] = 'later changed'
    assert validation['reproduction_assessment']['reason'] == '本次实验的 AI 结论'


@pytest.mark.parametrize('status', ['unavailable', 'blocked'])
def test_unavailable_or_blocked_assessment_only_attaches_its_diagnostics(status):
    validation = {'status': 'author_code_completed', 'reason': '作者命令结束',
                  'execution_status': 'author_code_completed', 'is_reproduced': None,
                  'paper_metrics_verified': False}
    original = deepcopy(validation)
    assessment = {'status': status, 'verdict': 'insufficient_evidence', 'reason': '暂无 AI 判断'}
    apply_reproduction_assessment(validation, assessment)
    assert {key: value for key, value in validation.items() if key != 'reproduction_assessment'} == original
