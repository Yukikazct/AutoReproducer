"""Investigation changes actions without acquiring execution/metric authority."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from src.intelligent_recovery import (RecoverySession, RecoveryBudgetExceeded, InvestigationStopped,
    current_investigation, normalize_intelligent_recovery_request)
from src.investigation_tools import InvestigationTools


class ScriptedModel:
    mock_mode = False
    api_key = 'test-credential-value'

    def __init__(self, respond):
        self.respond, self.calls = respond, []

    def chat(self, prompt, **kwargs):
        context = json.loads(prompt)
        self.calls.append((context, kwargs))
        return json.dumps(self.respond(context, kwargs, len(self.calls)))


@pytest.fixture
def experiment(tmp_path):
    workspace = tmp_path / 'author'
    workspace.mkdir()
    (workspace / 'model.py').write_text('class Model:\n    def value(self):\n        return 7\n    def other(self):\n        return 8\n', encoding='utf-8')
    (workspace / 'train.py').write_text('from model import Model\nprint(Model().get_value())\n', encoding='utf-8')
    inventory = {path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'text_available': True}
                 for path in workspace.iterdir()}
    packet = {'file_inventory': inventory, 'pdf': {'sha256': 'a' * 64}}
    plan = {'repository': {'url': 'https://github.com/example/method', 'revision': 'b' * 40},
            'steps': [{'id': 'train', 'kind': 'train', 'argv': ['python', 'train.py'], 'cwd': '.',
                       'entrypoint': {'kind': 'script', 'source_path': 'train.py'}}],
            'requirements': {'author': [], 'compatibility': []}}
    failure = {'success': False, 'final': {'exit_code': 1, 'stderr': "AttributeError: 'Model' object has no attribute 'get_value'"}}
    return workspace, packet, plan, failure


def test_probe_evidence_drives_next_decision_and_reviewer_veto(tmp_path, experiment):
    workspace, packet, plan, failure = experiment
    def respond(context, kwargs, call):
        if kwargs['task'] == 'recovery_counter_review':
            return {'action': 'reject', 'reason': 'The caller contract is not established.', 'evidence_ids': ['failure']}
        if not context['probe_results']:
            return {'action': 'probe', 'reason': 'Read the existing implementation.', 'evidence_ids': ['failure'],
                    'tool': 'source_context', 'arguments': {'paths': ['model.py']},
                    'hypotheses': [{'message': 'The original method may have another name.', 'evidence_ids': ['failure']}]}
        result = context['probe_results'][0]
        assert 'def value' in result['evidence'][0]['value']['text']
        return {'action': 'propose_repair', 'reason': 'Use the observed API.',
                'evidence_ids': [result['evidence'][0]['id']]}
    session = RecoverySession(ScriptedModel(respond))
    data, original = {}, deepcopy(plan)
    with session.scope():
        session.attach(data, tmp_path)
        result = session.investigate(trigger='execution_failure', plan=plan, packet=packet,
            workspace=workspace, environment={}, execution=failure)
        assert result['status'] == 'needs_human'
        assert result['stop_reason'] == 'review_rejected'
        assert result['hypotheses'][0]['status'] == 'unknown'
        assert session.state['usage']['probes'] == 1
        assert session.investigate(trigger='execution_failure', plan=plan, packet=packet,
            workspace=workspace, environment={}, execution=failure) is None
    assert current_investigation() is None and plan == original
    assert data['intelligent_recovery']['requires_human'] is True


def test_default_model_use_has_no_eight_request_or_64k_total_cutoff(tmp_path):
    llm = ScriptedModel(lambda *_: {'observed': True})
    session = RecoverySession(llm)
    session.attach({}, tmp_path)
    with session.scope():
        for _ in range(12):
            session.wrap_llm().chat(json.dumps({'context': 'x' * 6000}), task='recovery_investigation')
    assert len(llm.calls) == 12
    assert session.state['budget']['max_model_requests'] is None
    assert session.state['budget']['max_tokens'] is None
    assert session.state['usage']['total_tokens'] is None
    assert session.state['usage']['estimated_tokens'] > 64000
    assert session.state['usage']['monetary_cost'] is None


def test_explicit_ledger_is_shared_across_roles_and_errors(tmp_path):
    llm = ScriptedModel(lambda *_: {'observed': True})
    session = RecoverySession(llm, budget={'max_model_requests': 2})
    session.attach({}, tmp_path)
    session.wrap_llm().chat('{}', task='author_failure_analysis')
    session.wrap_llm().chat('{}', task='repository_repair')
    with pytest.raises(RecoveryBudgetExceeded):
        session.wrap_llm().chat('{}', task='recovery_counter_review')
    assert len(llm.calls) == 2 and session.state['requires_human'] is True


@pytest.mark.parametrize('flag', [{'cancelled': True}, {'timed_out': True}, {'cleanup_confirmed': False},
                               {'final': {'exit_code': 130}}, {'cleanup_unconfirmed': True}])
def test_stopped_execution_never_calls_model_or_probe(tmp_path, experiment, flag):
    workspace, packet, plan, failure = experiment
    llm = ScriptedModel(lambda *_: pytest.fail('stopped investigation called model'))
    session = RecoverySession(llm)
    session.attach({}, tmp_path)
    assert session.investigate(trigger='execution_failure', plan=plan, packet=packet,
        workspace=workspace, environment={}, execution={**failure, **flag}) is None
    with pytest.raises(InvestigationStopped):
        session.ensure_running()
    if flag.get('cleanup_confirmed') is False or flag.get('cleanup_unconfirmed'):
        with pytest.raises(InvestigationStopped) as error:
            session.ensure_running()
        assert error.value.process_cleanup_confirmed is False
        assert error.value.cleanup_receipt['status'] == 'unconfirmed'


def test_unconfirmed_probe_cleanup_blocks_further_recovery(tmp_path, experiment):
    workspace, packet, plan, failure = experiment
    llm = ScriptedModel(lambda *_: {'action': 'probe', 'reason': 'Observe.', 'evidence_ids': ['failure'],
                                    'tool': 'runtime_metadata', 'arguments': {'packages': ['numpy']}})
    toolset = Mock()
    toolset.execute.return_value = {'status': 'stopped', 'cleanup_confirmed': False, 'evidence': []}
    session = RecoverySession(llm)
    session.attach({}, tmp_path)
    session.investigate(trigger='execution_failure', plan=plan, packet=packet, workspace=workspace,
                        environment={}, execution=failure, tools=toolset)
    with pytest.raises(InvestigationStopped) as error:
        session.ensure_running()
    assert error.value.process_cleanup_confirmed is False
    assert len(llm.calls) == 1


@pytest.mark.parametrize('receipt', [{'cleanup': {'confirmed': False}},
    {'process_cleanup': {'status': 'unconfirmed'}}, {'cleanup_receipt': {'status': 'failed'}},
    {'cleanup_receipt': {'cleanup_confirmed': False}},
    {'cleanup_receipt': {'process_cleanup': {'status': 'unconfirmed'}}}])
@pytest.mark.parametrize('location', ['execution', 'environment', 'effective_env_config'])
def test_all_cleanup_receipts_remain_blocking_and_keep_false_on_the_exception(tmp_path, experiment, receipt, location):
    workspace, packet, plan, failure = experiment
    environment = {}
    if location == 'execution':
        failure.update(receipt)
    elif location == 'effective_env_config':
        failure['effective_env_config'] = receipt
    else:
        environment = receipt
    session = RecoverySession(ScriptedModel(lambda *_: pytest.fail('unconfirmed cleanup called a model')))
    session.attach({}, tmp_path)
    assert session.investigate(trigger='execution_failure', plan=plan, packet=packet,
        workspace=workspace, environment=environment, execution=failure) is None
    with pytest.raises(InvestigationStopped) as error:
        session.ensure_running()
    assert error.value.process_cleanup_confirmed is False
    assert error.value.cleanup_receipt['process_cleanup_confirmed'] is False


def test_repeated_probe_stops_without_resetting_shared_allowance(tmp_path, experiment):
    workspace, packet, plan, failure = experiment
    llm = ScriptedModel(lambda *_: {'action': 'probe', 'reason': 'Observe.', 'evidence_ids': ['failure'],
                                    'tool': 'source_context', 'arguments': {'paths': ['model.py']}})
    session = RecoverySession(llm, mode='single')
    session.attach({}, tmp_path)
    task = session.investigate(trigger='execution_failure', plan=plan, packet=packet,
        workspace=workspace, environment={}, execution=failure)
    assert task['status'] == 'needs_human' and len(llm.calls) == 2
    assert session.state['usage']['probes'] == 1


def test_invented_evidence_cannot_authorize_recovery(tmp_path, experiment):
    workspace, packet, plan, failure = experiment
    llm = ScriptedModel(lambda *_: {'action': 'propose_repair', 'reason': 'invented', 'evidence_ids': ['fake']})
    session = RecoverySession(llm)
    session.attach({}, tmp_path)
    task = session.investigate(trigger='execution_failure', plan=plan, packet=packet,
        workspace=workspace, environment={}, execution=failure)
    assert task['status'] == 'needs_human'
    assert session.state['requires_human'] is True


@pytest.mark.parametrize('value', ['yes', 1, None, {}, []])
def test_request_boolean_is_explicit(value):
    with pytest.raises(ValueError):
        normalize_intelligent_recovery_request({'intelligent_recovery': value})


def test_patch_veto_feeds_existing_rollback_then_real_original_command(tmp_path, experiment):
    from src.agents.code_executor import CodeExecutorAgent
    from src.repository_runner import RepositoryRunner
    import os
    workspace, packet, plan, failure = experiment
    calls = {'patch': 0}
    def respond(context, kwargs, call):
        if kwargs['task'] == 'recovery_investigation':
            return {'action': 'propose_repair', 'reason': 'Check original API.', 'evidence_ids': ['failure']}
        if kwargs['task'] == 'repository_repair':
            calls['patch'] += 1
            if calls['patch'] == 2:
                assert 'Independent patch review rejected' in context['previous_attempts'][0]['error']
                assert 'get_value' in (workspace / 'train.py').read_text()
            method = 'other' if calls['patch'] == 1 else 'value'
            return {'reason': 'Restore original API contract.', 'patches': [{'path': 'train.py',
                'base_sha256': context['sources']['train.py']['base_sha256'],
                'content': 'from model import Model\nprint(Model().' + method + '())\n'}]}
        candidate = context['candidate_decision']
        if candidate['action'] == 'apply_compatibility_patch':
            patch = candidate['patches']['patches'][0]['content']
            return {'action': 'reject' if '.other()' in patch else 'accept',
                    'reason': 'The other method returns another value.' if '.other()' in patch else 'No counterexample found.',
                    'evidence_ids': ['patch_sources']}
        return {'action': 'accept', 'reason': 'Original evidence supports investigation.', 'evidence_ids': ['failure']}
    llm = ScriptedModel(respond)
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._exec_env = Mock(return_value={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    executor._ensure_local_deps = Mock(return_value=None)
    runner = RepositoryRunner(executor=executor)
    session = RecoverySession(llm)
    session.attach({}, tmp_path)
    commands = deepcopy(plan['steps'])
    with session.scope():
        session.investigate(trigger='execution_failure', plan=plan, packet=packet,
            workspace=workspace, environment={'python_executable': sys.executable}, execution=failure)
        result = runner.run_with_repairs(workspace, commands, {}, llm=llm, editable=['train.py'])
        session.record_execution(result)
    assert result['success'] is True and result['final']['stdout'].strip() == '7'
    assert [row['status'] for row in result['repair_attempts']] == ['rejected', 'succeeded']
    assert commands == plan['steps']
    assert session.state['tasks'][0]['numerical_equivalence'] == 'unknown'
    assert session.state['tasks'][0]['status'] == 'command_recovered'
