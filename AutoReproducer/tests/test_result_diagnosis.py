"""Numerical diagnosis observes frozen evidence and never changes experiment state."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import Mock

import pytest

from src import result_diagnosis as diagnosis
from src.agents.report_generator import ReportGeneratorAgent, _outcome_diagnosis_lines


class Model:
    def __init__(self, response=None):
        self.calls = []
        self.api_key = 'fixture-private-diagnosis-key'
        self.response = response if response is not None else {'hypotheses': [{
            'category': 'statistics', 'message': '汇总方式可能影响差值，需核对原始汇总记录。',
            'evidence_ids': ['comparison_1', 'frozen_plan']}]}

    def chat(self, prompt, **kwargs):
        self.calls.append((json.loads(prompt), kwargs))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


@pytest.fixture
def case(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    source = workspace / 'train.py'
    source.write_text("print('Test Accuracy: 0.7')\n", encoding='utf-8')
    plan = {'scope': 'author_code_execution', 'steps': [{'id': 'train', 'kind': 'train',
            'argv': ['python', 'train.py', '--seed', '42'],
            'entrypoint': {'source_path': 'train.py'}}],
            'capture': {'device': 'cpu'}, 'requirements': {'author': ['numpy==1.26.4']}}
    packet = {'file_inventory': {'train.py': {'text_available': True,
              'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}}
    execution = {'success': True, 'steps': [{'id': 'train', 'executed': True, 'success': True,
        'exit_code': 0, 'stdout': 'Test Accuracy: 0.7', 'stderr': ''}]}
    row = {'metric': 'accuracy', 'status': 'comparable', 'direction': 'higher',
        'reference': {'value': .9, 'unit': 'fraction', 'dataset': 'Example', 'method': 'Model',
                      'value_quote': 'Model Example Test Accuracy: 0.9',
                      'split': 'test', 'aggregation': 'mean', 'runs': 20},
        'actual': {'value': .7, 'unit': 'fraction', 'dataset': 'Example', 'method': 'Model',
                   'split': 'test', 'aggregation': 'mean', 'runs': 20, 'source': 'train:stdout:1'},
        'settings': {'status': 'matched'}, 'difference': -.2, 'relative_difference': -.2 / .9,
        'within_tolerance': False, 'meets_reference': False}
    validation = {'status': 'author_code_completed', 'is_reproduced': None,
        'paper_metrics_verified': False, 'dataset_provenance_verified': False,
        'author_claim_comparison': {'status': 'compared', 'relative_tolerance': .05,
                                    'comparisons': [row]}}
    data = {'execution': execution, 'validation': validation}
    run_dir = tmp_path / 'run'
    return {'workspace': workspace, 'plan': plan, 'packet': packet, 'execution': execution,
            'validation': validation, 'data': data, 'run_dir': run_dir, 'row': row, 'events': []}


def run(case, llm=None):
    return diagnosis.diagnose_experiment_result(llm, case['plan'], case['packet'], case['workspace'],
        case['execution'], case['validation'], case['run_dir'], case['data'], case['events'].append)


def test_deviation_records_frozen_evidence_and_one_qualified_model_analysis(case):
    before = deepcopy({key: case[key] for key in ('plan', 'packet', 'execution', 'validation')})
    original_source = (case['workspace'] / 'train.py').read_bytes()
    llm = Model()
    result = run(case, llm)
    assert result['status'] == 'diagnosed' and result['trigger'] == ['reference_not_met']
    assert len(llm.calls) == 1 and llm.calls[0][1]['task'] == 'result_diagnosis'
    assert result['model_analysis']['status'] == 'completed'
    assert result['hypotheses'][0]['certainty'] == 'hypothesis'
    assert all(fact['certainty'] == 'observed' for fact in result['facts'])
    assert result['actions'] == {'mutated': False, 'retrained': False}
    evidence = {item['id']: item for item in result['evidence']}
    assert evidence['source_1']['value']['sha256'] == case['packet']['file_inventory']['train.py']['sha256']
    assert evidence['frozen_plan']['value']['settings']['steps'][0]['argv'][-2:] == ['--seed', '42']
    assert evidence['execution']['value']['steps'][0]['exit_code'] == 0
    assert all(set(item['evidence_ids']) <= set(evidence) for item in result['facts'] + result['hypotheses'])
    assert json.loads((case['run_dir'] / 'outcome_diagnosis.json').read_text('utf-8')) == result
    assert case['data']['outcome_diagnosis'] == result
    assert {key: case[key] for key in before} == before
    assert (case['workspace'] / 'train.py').read_bytes() == original_source
    assert [event['log']['status'] for event in case['events']] == ['RUNNING', 'SUCCESS']
    assert all(event['type'] == 'log' and event['log']['action'] == 'diagnose_result' for event in case['events'])
    assert run(case, llm) is result and len(llm.calls) == 1


@pytest.mark.parametrize('mode', ['within_reference', 'reference_only_small_gap', 'better_than_reference',
                                  'failed', 'cancelled', 'no_comparison'])
def test_no_unnecessary_diagnosis_or_retraining(case, mode):
    if mode == 'failed':
        case['execution']['success'] = False
    elif mode == 'cancelled':
        case['execution']['cancelled'] = True
    elif mode == 'no_comparison':
        case['validation'].pop('author_claim_comparison')
    else:
        case['row'].update(meets_reference=True, within_tolerance=True, relative_difference=.01)
        if mode == 'reference_only_small_gap':
            case['row'].update(status='reference_only', meets_reference=None, within_tolerance=None)
        if mode == 'better_than_reference':
            case['row'].update(relative_difference=.4, within_tolerance=False)
    llm = Model()
    assert run(case, llm) is None
    assert not llm.calls and not case['run_dir'].exists()
    assert 'outcome_diagnosis' not in case['data'] and not case['events']


def test_incomparable_fields_produce_evidence_gaps_and_concrete_checks(case):
    row = case['row']
    row.update(status='not_comparable', meets_reference=None, within_tolerance=None, relative_difference=None)
    row['actual'].update(unit=None, dataset='Other', method='Variant', split='validation', runs=1, aggregation='single')
    row['settings'] = {'status': 'different', 'reason': 'frozen configuration differs'}
    result = run(case)
    assert result['status'] == 'evidence_limited' and result['trigger'] == ['not_comparable']
    categories = {item['category'] for item in result['facts']}
    assert {'unit', 'method', 'data', 'statistics', 'protocol', 'evidence'} <= categories
    assert not result['hypotheses'] and result['model_analysis']['llm_calls'] == 0
    assert all(item['check'] and item['evidence_ids'] for item in result['next_steps'])


def test_reference_only_large_gap_is_diagnosed_without_claiming_acceptance(case):
    case['row'].update(status='reference_only', meets_reference=None, within_tolerance=None)
    result = run(case)
    assert result['status'] == 'evidence_limited' and result['trigger'] == ['reference_not_met']
    assert case['validation']['is_reproduced'] is None
    assert case['validation']['paper_metrics_verified'] is False


def test_unavailable_comparison_preserves_metrics_and_execution(case):
    case['validation']['metric_records'] = [{'name': 'accuracy', 'value': .7, 'source': 'train:stdout:1'}]
    case['validation']['author_claim_comparison'] = {'status': 'unavailable', 'comparisons': [],
                                                   'reason': 'No corresponding claim'}
    result = run(case)
    assert result['trigger'] == ['comparison_unavailable'] and result['status'] == 'evidence_limited'
    assert case['data']['execution']['success'] is True
    assert case['validation']['metric_records'][0]['value'] == .7
    assert any(item['id'] == 'comparison' for item in result['evidence'])


def test_strict_failed_numerical_acceptance_has_distinct_diagnosis(case):
    case['validation'].pop('author_claim_comparison')
    case['validation'].update(status='reference_not_met', is_reproduced=False, protocol_pass=True,
                              independent_metrics_pass=True, paper_metrics_verified=True,
                              comparisons=[{'name': 'accuracy', 'actual': .7, 'reference': .9, 'pass': False}])
    result = run(case)
    assert result['trigger'] == ['strict_not_reproduced'] and result['status'] == 'diagnosed'
    assert any('独立指标核验已通过' in item['message'] for item in result['facts'])
    assert case['validation']['is_reproduced'] is False


def ai_assessment(verdict):
    return {'status': 'completed', 'source': 'ai', 'scope': 'executed_experiment',
            'verdict': verdict, 'reason': '已结合论文结果和本次实际设置综合判断。',
            'confidence': .8, 'limitations': ['本次单次运行，论文报告多次均值。'],
            'metric_assessments': [{'actual_id': 'train:stdout:0', 'verdict': verdict,
                'reason': '按实际误差及论文统计口径判断。',
                'tolerance': {'absolute': .03, 'relative': None,
                              'reason': '容许随机种子变化导致的绝对误差。'}}]}


@pytest.mark.parametrize('row_status', ['comparable', 'reference_only'])
def test_ai_reproduced_is_not_overruled_by_old_reference_threshold(case, row_status):
    case['row'].update(status=row_status)
    case['row']['actual'].update(aggregation='single', runs=1)
    case['validation'].update(status='ai_reproduced', is_reproduced=True,
                              reproduction_assessment=ai_assessment('reproduced'))
    before = deepcopy(case['validation'])
    llm = Model()
    assert run(case, llm) is None
    assert not llm.calls and not case['run_dir'].exists()
    assert case['validation'] == before


@pytest.mark.parametrize('verdict', ['partially_reproduced', 'not_reproduced', 'insufficient_evidence'])
def test_ai_unresolved_verdict_drives_diagnosis_and_preserves_error_basis(case, verdict):
    # The AI decision can need diagnosis even when the old 5% comparison passes.
    case['row'].update(meets_reference=True, within_tolerance=True, relative_difference=.01)
    case['row']['actual']['id'] = 'train:stdout:0'
    assessment = ai_assessment(verdict)
    case['validation'].update(status='ai_' + verdict,
                              reproduction_assessment=assessment,
                              is_reproduced=False if verdict == 'not_reproduced' else None)
    before = deepcopy(case['validation'])
    llm = Model({'hypotheses': [{'category': 'statistics', 'message': '重复统计差异可能影响误差解释。',
                               'evidence_ids': ['reproduction_assessment', 'comparison_1']}]})
    result = run(case, llm)
    assert result['trigger'] == ['ai_' + verdict]
    evidence = {item['id']: item for item in result['evidence']}
    assert evidence['reproduction_assessment']['value'] == assessment
    messages = ' '.join(item['message'] for item in result['facts'])
    assert '不再使用固定阈值重新否决' in messages
    assert '容许随机种子变化导致的绝对误差' in messages
    assert '已记录数值偏差未满足当前参考规则' not in messages
    assert '可选核验与普通 AI 复现判断分别保留' in messages
    assert case['validation'] == before
    assert result['model_analysis']['status'] == 'completed'
    assert json.loads((case['run_dir'] / 'outcome_diagnosis.json').read_text('utf-8')) == result
    if verdict == 'insufficient_evidence':
        assert result['status'] == 'evidence_limited'


def test_ai_insufficient_evidence_without_comparison_still_records_assessment(case):
    case['validation']['reproduction_assessment'] = ai_assessment('insufficient_evidence')
    case['validation']['author_claim_comparison']['comparisons'] = []
    result = run(case)
    assert result['trigger'] == ['ai_insufficient_evidence']
    assert any(item['id'] == 'reproduction_assessment' for item in result['evidence'])
    assert result['status'] == 'evidence_limited'


@pytest.mark.parametrize('patch', [{'status': 'unavailable'}, {'status': 'blocked'},
                                  {'source': 'unknown'}, {'verdict': 'unknown'}])
def test_incomplete_or_invalid_ai_assessment_keeps_prior_reference_diagnosis(case, patch):
    case['validation']['reproduction_assessment'] = {**ai_assessment('reproduced'), **patch}
    result = run(case)
    assert result['trigger'] == ['reference_not_met']


def test_ai_assessment_cannot_erase_independent_strict_validation_failure(case):
    case['validation'].pop('author_claim_comparison')
    case['validation'].update(status='reference_not_met', is_reproduced=False, protocol_pass=True,
                              independent_metrics_pass=True, paper_metrics_verified=True,
                              reproduction_assessment=ai_assessment('reproduced'))
    result = run(case)
    assert result['trigger'] == ['strict_not_reproduced']
    assert any('独立指标核验已通过' in item['message'] for item in result['facts'])
    assert not any('固定阈值重新否决' in item['message'] for item in result['facts'])


def test_ai_assessment_diagnostic_evidence_redacts_model_credential(case):
    assessment = ai_assessment('not_reproduced')
    assessment['reason'] += ' fixture-private-diagnosis-key'
    case['validation']['reproduction_assessment'] = assessment
    llm = Model({'hypotheses': []})
    result = run(case, llm)
    combined = json.dumps(result) + json.dumps(llm.calls[0][0])
    assert 'fixture-private-diagnosis-key' not in combined
    assert '[REDACTED]' in combined
    assert 'fixture-private-diagnosis-key' in case['validation']['reproduction_assessment']['reason']


@pytest.mark.parametrize('response', [
    'not json', RuntimeError('model unavailable'),
    {'hypotheses': [{'category': 'data', 'message': 'unknown', 'evidence_ids': ['invented_evidence']}]},
    {'hypotheses': [], 'action': 'retrain'},
    {'hypotheses': [{'category': 'metric', 'message': 'unknown', 'evidence_ids': []}]},
    {'hypotheses': [{'category': 'metric', 'message': 'unknown', 'evidence_ids': ['frozen_plan'], 'patch': 'x'}]},
])
def test_model_failures_cannot_replace_deterministic_observations_or_original_result(case, response):
    before = deepcopy(case['validation'])
    result = run(case, Model(response))
    assert result['model_analysis']['status'] == 'failed'
    assert result['model_analysis']['llm_calls'] == 1
    assert result['facts'] and not result['hypotheses'] and result['next_steps']
    assert case['validation'] == before and case['execution']['success'] is True


def test_model_context_and_saved_diagnosis_redact_fake_credentials(case):
    llm = Model({'hypotheses': [{'category': 'metric', 'message': 'fixture-private-diagnosis-key',
                               'evidence_ids': ['comparison_1']}]})
    case['execution']['steps'][0]['stdout'] += '\nfixture-private-diagnosis-key api_key=abc-local-secret'
    case['plan']['capture']['token'] = 'another-private-value'
    result = run(case, llm)
    combined = json.dumps(result) + json.dumps(llm.calls[0][0]) + (case['run_dir'] / 'outcome_diagnosis.json').read_text('utf-8')
    for secret in ('fixture-private-diagnosis-key', 'abc-local-secret', 'another-private-value'):
        assert secret not in combined
    assert '[REDACTED]' in combined
    assert 'fixture-private-diagnosis-key' in case['execution']['steps'][0]['stdout']


def test_oversized_model_response_does_not_replace_observations(case):
    result = run(case, Model('x' * 24001))
    assert result['model_analysis']['status'] == 'failed'
    assert result['facts'] and result['model_analysis']['llm_calls'] == 1


def test_oversized_context_keeps_diagnosis_without_calling_model(case):
    case['plan']['large_fixture'] = {str(index): 'x' * 3000 for index in range(40)}
    llm = Model()
    result = run(case, llm)
    assert not llm.calls and result['facts']
    assert result['model_analysis']['status'] == 'not_requested'
    assert '预算' in result['model_analysis']['reason']


def test_mutated_source_is_excluded_from_diagnostic_context(case):
    (case['workspace'] / 'train.py').write_text('changed source', encoding='utf-8')
    result = run(case, Model())
    assert not any(item['kind'] == 'source_excerpt' for item in result['evidence'])


def test_model_cancellation_retains_diagnosis_without_swallowing_interruption(case):
    with pytest.raises(KeyboardInterrupt, match='user stopped'):
        run(case, Model(KeyboardInterrupt('user stopped')))
    saved = json.loads((case['run_dir'] / 'outcome_diagnosis.json').read_text('utf-8'))
    assert saved['model_analysis']['status'] == 'interrupted' and saved['facts']
    assert case['execution']['success'] and case['validation']['is_reproduced'] is None


def test_storage_failure_preserves_in_memory_diagnosis_and_validation(case, monkeypatch):
    monkeypatch.setattr(diagnosis, 'write_json', Mock(side_effect=OSError('storage unavailable')))
    result = run(case)
    assert result['storage_error'] == 'storage unavailable' and result['facts']
    assert case['data']['outcome_diagnosis'] is result and case['execution']['success']


def test_report_separates_facts_hypotheses_checks_and_keeps_runtime_repair_section(case):
    run(case, Model())
    case['data']['self_repair'] = {'status': 'resolved', 'reason': 'Original import restored', 'attempts': []}
    report = ReportGeneratorAgent._build_report(case['data'])
    assert '### 数值结果与证据自动诊断' in report
    assert '**已观察事实与缺失证据**' in report and '**待核实假设**' in report
    assert '**下一步具体核查**' in report and '**诊断证据索引**' in report
    assert 'outcome\\_diagnosis.json' in report and '### 自动诊断与修复' in report
    assert '未修改实验参数、源码、参考值或重新训练' in report
    assert _outcome_diagnosis_lines({}) == []


def test_report_discloses_model_failure_and_escapes_generated_markdown(case):
    run(case, Model(RuntimeError('failed **untrusted** token=private-value')))
    report = '\n'.join(_outcome_diagnosis_lines(case['data']))
    assert '**模型补充分析**' in report and 'failed \\*\\*untrusted\\*\\*' in report
    assert 'private-value' not in report
