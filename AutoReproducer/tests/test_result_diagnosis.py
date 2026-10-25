"""Numerical diagnosis observes frozen evidence and never changes experiment state."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import Mock

import pytest

from src import result_diagnosis as diagnosis
from src.agent_collaboration import CollaborationSession
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


def test_blocking_checks_are_prioritized_and_merge_all_related_metric_evidence(case):
    second = deepcopy(case['row'])
    second.update(metric='f1', status='not_comparable')
    second['actual'].update(unit=None, dataset='Other', runs=None)
    case['row']['actual']['dataset'] = 'Other'
    case['validation']['author_claim_comparison']['comparisons'].append(second)
    llm = Model({'hypotheses': [{'category': 'execution', 'message': '可进一步核对自然结束范围。',
                               'evidence_ids': ['execution']}]})
    result = run(case, llm)
    checks = result['next_steps']
    assert [item['rank'] for item in checks] == list(range(1, len(checks) + 1))
    assert [item['priority'] for item in checks] == sorted(item['priority'] for item in checks)
    assert checks[0]['category'] == 'evidence' and checks[0]['priority'] == 1
    data_check = next(item for item in checks if item['category'] == 'data')
    assert set(data_check['evidence_ids']) == {'comparison_1', 'comparison_2', 'validation'}
    model_check = next(item for item in checks if item['category'] == 'execution')
    assert model_check['basis'] == 'model_hypothesis' and model_check['priority'] == 3
    assert model_check['blocker'] is False
    known = {item['id'] for item in result['evidence']}
    assert all(set(item['evidence_ids']) <= known for item in checks)
    assert result['assessment']['comparison_coverage'] == {
        'total': 2, 'comparable': 1, 'reference_only': 0, 'not_comparable': 1, 'outside_reference': 2}
    assert result['assessment']['evidence_status'] == 'limited'
    assert result['assessment']['cause_status'] == 'unconfirmed'
    assert case['validation']['is_reproduced'] is None


def test_completed_execution_does_not_create_a_redundant_execution_check(case):
    result = run(case)
    assert any(item['category'] == 'execution' for item in result['facts'])
    assert not any(item['category'] == 'execution' for item in result['next_steps'])
    metric = result['assessment']['metric_outcomes'][0]
    assert metric['evidence_id'] == 'comparison_1'
    assert metric['difference'] == case['row']['difference']
    assert metric['reference_rule'] == 'not_met'


def test_explicit_unit_conversion_is_recorded_without_asking_to_repeat_it(case):
    case['row']['reference'].update(value=90.0, unit='percent')
    case['row']['difference_unit'] = 'percentage_points'
    result = run(case)
    assert not any(item['category'] == 'unit' for item in result['next_steps'])
    assert any('已按显式单位换算' in item['message'] for item in result['facts'])


def test_incomplete_field_evidence_stays_a_gap_instead_of_aborting_diagnosis(case):
    case['row']['reference']['field_evidence'] = {'unit': None}
    result = run(case)
    assert any(item['category'] == 'unit' and item['blocker'] for item in result['next_steps'])


def test_identical_verified_inputs_reuse_analysis_and_persist_reuse_count(case):
    llm = Model()
    with CollaborationSession().scope():
        first = run(case, llm)
        second = run(case, llm)
    assert first is second and len(llm.calls) == 1
    assert second['efficiency']['cache_hits'] == 1
    saved = json.loads((case['run_dir'] / 'outcome_diagnosis.json').read_text('utf-8'))
    assert saved == second


@pytest.mark.parametrize('change', ['stdout_tail', 'validation_tail', 'source_hash', 'run_dir', 'workspace', 'model'])
def test_reuse_identity_covers_complete_inputs_and_run_bindings(case, change):
    llm = Model()
    if change == 'stdout_tail':
        case['execution']['steps'][0]['stdout'] = 'x' * 5000 + 'before'
    elif change == 'validation_tail':
        case['validation']['details'] = 'x' * 5000 + 'before'
    elif change == 'source_hash':
        source = case['workspace'] / 'train.py'
        source.write_text('x' * 5000 + 'before', encoding='utf-8')
        case['packet']['file_inventory']['train.py']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    first = run(case, llm)
    if change == 'stdout_tail':
        case['execution']['steps'][0]['stdout'] = 'x' * 5000 + 'after'
    elif change == 'validation_tail':
        case['validation']['details'] = 'x' * 5000 + 'after'
    elif change == 'source_hash':
        source.write_text('x' * 5000 + 'after', encoding='utf-8')
        case['packet']['file_inventory']['train.py']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    elif change == 'run_dir':
        case['run_dir'] = case['run_dir'].parent / 'another-run'
    elif change == 'workspace':
        original = (case['workspace'] / 'train.py').read_bytes()
        case['workspace'] = case['workspace'].parent / 'another-workspace'
        case['workspace'].mkdir()
        (case['workspace'] / 'train.py').write_bytes(original)
    else:
        llm.model = 'another-model'
    second = run(case, llm)
    assert second is not first and second['fingerprint'] != first['fingerprint']
    assert len(llm.calls) == 2
    assert second['path'] == str(case['run_dir'] / 'outcome_diagnosis.json')


def test_diagnosis_is_not_reused_across_collaboration_runs(case):
    llm = Model()
    with CollaborationSession().scope():
        first = run(case, llm)
    with CollaborationSession().scope():
        second = run(case, llm)
    assert len(llm.calls) == 2 and first['fingerprint'] != second['fingerprint']


def test_edited_diagnosis_is_not_reused_as_model_or_observed_evidence(case):
    llm = Model()
    first = run(case, llm)
    first['facts'][0]['message'] = 'untrusted edited conclusion'
    second = run(case, llm)
    assert len(llm.calls) == 2 and second is not first
    assert 'untrusted edited conclusion' not in json.dumps(llm.calls[1][0])


def test_diagnosis_consumes_navigation_and_hands_off_report_without_new_authority(case):
    session, llm = CollaborationSession(), Model()
    with session.scope():
        earlier = session.send('FailureAnalyst', 'RepositoryRepairer', 'failure_findings',
            {'reason': 'Earlier dependency failure was already resolved.'})
        direct = session.send('PlanValidator', 'ResultDiagnosis', 'feedback',
            {'reason': 'Check the frozen protocol only.'})
        result = run(case, llm)
        reused = run(case, llm)
    context = llm.calls[0][0]
    assert {earlier['id'], direct['id']} <= {item['id'] for item in context['agent_messages']}
    assert all(item['id'] not in {earlier['id'], direct['id']} for item in result['evidence'])
    assert 'never message IDs' in llm.calls[0][1]['system_prompt']
    messages = session.snapshot()['messages']
    reports = [item for item in messages if item['sender'] == 'ResultDiagnosis'
               and item['recipient'] == 'ReportGenerator']
    assert len(reports) == 2 and reports[-1]['content']['reused'] is True
    assert reports[-1]['content']['cause_status'] == 'unconfirmed'
    assert reports[-1]['content']['artifact'] == 'outcome_diagnosis.json'
    assert reused is result and len(llm.calls) == 1


def test_model_cannot_cite_upstream_message_as_verified_result_evidence(case):
    session = CollaborationSession()
    with session.scope():
        message = session.send('FailureAnalyst', 'ResultDiagnosis', 'feedback', {'reason': 'A hypothesis'})
        llm = Model({'hypotheses': [{'category': 'metric', 'message': '尚需核查。',
                                   'evidence_ids': [message['id']]}]})
        result = run(case, llm)
    assert result['model_analysis']['status'] == 'failed' and not result['hypotheses']


@pytest.mark.parametrize('marker', [
    {'timed_out': True}, {'cleanup_unconfirmed': True}, {'cleanup_confirmed': False},
    {'final': {'exit_code': 130}}, {'process_cleanup': {'confirmed': False}},
])
def test_stopped_or_unclean_execution_never_spends_diagnostic_model_calls(case, marker):
    case['execution'].update(marker)
    llm, session = Model(), CollaborationSession()
    with session.scope():
        assert run(case, llm) is None
    assert not llm.calls and not session.snapshot()['messages']
    assert 'outcome_diagnosis' not in case['data']


def test_interrupted_diagnosis_is_handed_off_as_interrupted_without_findings(case):
    session = CollaborationSession()
    with session.scope(), pytest.raises(KeyboardInterrupt):
        run(case, Model(KeyboardInterrupt()))
    responses = [item for item in session.snapshot()['messages'] if item['sender'] == 'ResultDiagnosis']
    assert responses and all(item['content']['status'] == 'interrupted' for item in responses)
    assert any(item['kind'] == 'analysis_interrupted' for item in responses)
    assert case['data']['outcome_diagnosis']['hypotheses'] == []


@pytest.mark.parametrize('status', ['cancelled', 'timed_out', 'cleanup_failed', 'cleanup_unconfirmed'])
def test_terminal_validation_stop_status_also_blocks_model_analysis(case, status):
    case['validation']['status'] = status
    llm = Model()
    assert run(case, llm) is None and not llm.calls


def test_interrupted_model_response_is_not_reused_as_a_finished_diagnosis(case):
    llm = Model(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run(case, llm)
    llm.response = {'hypotheses': []}
    result = run(case, llm)
    assert len(llm.calls) == 2 and result['model_analysis']['status'] == 'completed'


def test_unsupported_plan_values_are_not_formatted_or_sent_to_the_model(case):
    class PrivateValue:
        def __str__(self):
            raise AssertionError('must not inspect private value')

        __repr__ = __str__
    case['plan']['untrusted_extra'] = PrivateValue()
    result = run(case, Model())
    assert result['model_analysis']['status'] == 'completed'
    assert '[unsupported evidence value]' in json.dumps(result)
