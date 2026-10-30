"""Failure handoffs drive bounded repair prompts without changing frozen plans."""
from copy import deepcopy
import hashlib
import json
import os
from unittest.mock import Mock

import pytest

from src.agent_collaboration import CollaborationSession
from src.agents.code_executor import CodeExecutorAgent
from src.author_failure_analysis import analyze_author_failure
from src.author_self_repair import failure_diagnostic
from src.repository_repairer import RepositoryRepairer
from src.repository_runner import RepositoryRunner
from src.safety.patch_policy import PatchPolicy
from src.safety.workspace_snapshot import snapshot_workspace
from src.source_repair import validate_compatibility_proposal


class ScriptedLLM:
    mock_mode = False

    def __init__(self, propose):
        self.propose, self.calls = propose, []

    def chat(self, prompt, **kwargs):
        context = json.loads(prompt)
        self.calls.append((context, kwargs))
        return json.dumps(self.propose(context, len(self.calls)))


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'model.py').write_bytes(b'class Model:\n    def value(self):\n        return 7\n')
    (root / 'train.py').write_bytes(b'from model import Model\nprint(Model().get_value())\n')
    (root / 'data').mkdir()
    (root / 'data' / 'split.csv').write_bytes(b'frozen split\n')
    return root


@pytest.fixture
def runner():
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._exec_env = Mock(return_value={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    executor._ensure_local_deps = Mock(return_value=None)
    return RepositoryRunner(executor=executor)


def steps():
    return [{'id': 'train', 'argv': ['python', 'train.py'], 'cwd': '.', 'kind': 'train'}]


def patch(context, changes):
    return {'reason': 'Align the original caller with its existing model API.',
            'patches': [{'path': name, 'base_sha256': context['sources'][name]['base_sha256'],
                         'content': content} for name, content in changes.items()]}


def valid_patch(context, call):
    return patch(context, {'train.py': 'from model import Model\nprint(Model().value())\n'})


def metadata(root):
    packet = {'file_inventory': {name: {'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest(),
                                      'text_available': True} for name in ('model.py', 'train.py')}}
    plan = {'repository': {'url': 'https://github.com/example/method'},
            'steps': [{**steps()[0], 'entrypoint': {'kind': 'script', 'source_path': 'train.py'}}],
            'requirements': {'author': [], 'compatibility': []}}
    return packet, plan


def test_analyst_findings_are_consumed_by_real_repair_prompt(repository, runner, tmp_path):
    initial = runner.run(repository, steps(), {})
    assert not initial['success']
    packet, plan = metadata(repository)
    analyst = ScriptedLLM(lambda *_: {'action': 'blocked',
        'reason': 'The caller requests the absent get_value API; inspect its original value method.'})
    llm = ScriptedLLM(valid_patch)
    session = CollaborationSession()
    command = deepcopy(steps())
    with session.scope():
        assert analyze_author_failure(analyst, initial, plan, packet, repository,
                                      {}, tmp_path, {}) is None
        result = runner.run_with_repairs(repository, command, {}, llm=llm,
            editable=['train.py'], initial_execution=initial)
    assert result['success'], result
    assert result['final']['stdout'].strip() == '7'
    assert command == steps()
    received = llm.calls[0][0]['agent_messages']
    findings = next(message for message in received if message['sender'] == 'FailureAnalyst')
    assert 'original value method' in findings['content']['reason']
    assert 'hypothesis' in findings['content']['authority']
    assert session.snapshot()['messages'][0]['recipient'] == 'FailureAnalyst'
    assert any(item['kind'] == 'failure_findings' and item['delivered']
               for item in session.snapshot()['messages'])
    assert (repository / 'data' / 'split.csv').read_bytes() == b'frozen split\n'


def test_validator_rejection_returns_to_second_prompt_after_full_rollback(repository, runner):
    original = snapshot_workspace(repository)

    def propose(context, call):
        if call == 1:
            return patch(context, {
                'model.py': 'class Model:\n    def value(self):\n        return 8\n',
                'train.py': 'from model import Model\nprint(Model().value())\n'})
        assert snapshot_workspace(repository) == original
        assert '兼容补丁改变' in context['previous_attempts'][0]['error']
        rejection = next(item for item in context['agent_messages']
                         if item['sender'] == 'PatchValidator' and item['content']['status'] == 'rejected')
        assert rejection['reply_to']
        assert '兼容补丁改变' in rejection['content']['reason']
        return valid_patch(context, call)

    def validate(root, files, failed):
        validate_compatibility_proposal(root, files, failure_diagnostic(failed))

    session, llm = CollaborationSession(), ScriptedLLM(propose)
    with session.scope():
        result = runner.run_with_repairs(repository, steps(), {}, llm=llm,
            editable=['model.py', 'train.py'], validate_patches=validate)
    assert result['success'] and len(llm.calls) == 2
    assert [item['status'] for item in result['repair_attempts']] == ['rejected', 'succeeded']
    assert (repository / 'model.py').read_bytes() == original['model.py']['content']
    assert (repository / 'data' / 'split.csv').read_bytes() == original['data/split.csv']['content']


@pytest.mark.parametrize('full_plan', [False, True])
@pytest.mark.parametrize('reuse_failure', [False, True])
def test_repair_handoff_accepts_both_runner_plan_forms_without_changing_commands(
        repository, runner, monkeypatch, full_plan, reuse_failure):
    command = {'version': 1, 'mode': 'repository', 'steps': steps()} if full_plan else steps()
    original = deepcopy(command)
    initial = runner.run(repository, command, {}) if reuse_failure else None
    run = Mock(wraps=runner.run)
    monkeypatch.setattr(runner, 'run', run)
    session = CollaborationSession()
    with session.scope():
        result = runner.run_with_repairs(repository, command, {}, llm=ScriptedLLM(valid_patch),
            editable=['train.py'], initial_execution=initial)
    assert result['success'], result
    assert result['final']['stdout'].strip() == '7'
    assert command == original
    assert run.call_count == (1 if reuse_failure else 2)
    assert all(call.args[1] == original for call in run.call_args_list)
    messages = session.snapshot()['messages']
    request = next(item for item in messages if item['kind'] == 'rerun_original_plan')
    assert request['content']['step_ids'] == ['train']
    outcome = next(item for item in messages if item['reply_to'] == request['id'])
    assert outcome['kind'] == 'execution_result' and outcome['content']['success'] is True


def test_failed_rerun_is_sent_back_as_concrete_execution_feedback(repository, runner):
    initial_source = (repository / 'train.py').read_bytes()

    def propose(context, call):
        if call == 1:
            return patch(context, {'train.py': 'from model import Model\nprint(Model().still_missing())\n'})
        assert (repository / 'train.py').read_bytes() == initial_source
        outcome = next(item for item in context['agent_messages']
                       if item['sender'] == 'CodeExecutor' and item['kind'] == 'execution_result')
        assert outcome['content']['success'] is False and outcome['reply_to']
        assert 'still_missing' in outcome['content']['diagnostic']
        assert 'get_value' in context['initial_diagnostic']
        assert context['candidate_workspace_state'] == 'original_baseline_after_rollback'
        return valid_patch(context, call)

    session, llm = CollaborationSession(), ScriptedLLM(propose)
    with session.scope():
        result = runner.run_with_repairs(repository, steps(), {}, llm=llm, editable=['train.py'])
    assert result['success'] and len(llm.calls) == 2
    assert [item['status'] for item in result['repair_attempts']] == ['failed', 'succeeded']


def test_rejected_collaboration_does_not_extend_two_round_budget(repository, runner):
    before = snapshot_workspace(repository)
    llm = ScriptedLLM(lambda context, call: patch(context, {'train.py': 'raise RuntimeError("changed")\n'}))

    def reject(*_):
        raise ValueError('The frozen training and evaluation protocol cannot change.')

    with CollaborationSession().scope():
        result = runner.run_with_repairs(repository, steps(), {}, llm=llm,
            editable=['train.py'], validate_patches=reject)
    assert not result['success'] and len(llm.calls) == 2
    assert len(result['repair_attempts']) == 2
    assert snapshot_workspace(repository) == before


@pytest.mark.parametrize('markers', [
    {'cancelled': True}, {'timed_out': True}, {'interrupted': True},
    {'cleanup_unconfirmed': True}, {'cleanup_failed': True},
    {'cleanup_confirmed': False},
    {'status': 'cleanup_unconfirmed'}, {'process_cleanup': {'confirmed': False}},
    {'effective_env_config': {'dependency_failure_kind': 'installation_timeout'}},
    {'final': {'exit_code': 1, 'interrupted': True}},
    {'final': {'exit_code': 1, 'stderr': 'process-tree cleanup failed: access denied'}},
])
def test_cancelled_or_unclean_execution_never_requests_analysis_or_source_repair(repository, tmp_path, markers):
    execution = {'executed': True, 'success': False,
                 'final': {'exit_code': 1, 'stderr': 'AttributeError: absent API'}, **markers}
    packet, plan = metadata(repository)
    analyst, llm = ScriptedLLM(lambda *_: {}), ScriptedLLM(valid_patch)
    fake_runner = Mock()
    fake_runner.run.return_value = execution
    before = snapshot_workspace(repository)
    session = CollaborationSession()
    with session.scope():
        assert analyze_author_failure(analyst, execution, plan, packet, repository,
                                      {}, tmp_path, {}) is None
        result = RepositoryRepairer(fake_runner, llm, PatchPolicy(editable=['train.py'])).run(
            repository, steps(), {})
    assert result['repair_attempts'] == [] and result['repair_llm_calls'] == 0
    assert analyst.calls == llm.calls == []
    assert session.snapshot()['messages'] == []
    assert snapshot_workspace(repository) == before


@pytest.mark.parametrize('markers', [
    {'timed_out': True}, {'interrupted': True}, {'cleanup_failed': True},
    {'cleanup_confirmed': False}, {'final': {'exit_code': 130}},
])
def test_success_flag_cannot_commit_stopped_candidate(repository, markers):
    before = snapshot_workspace(repository)
    initial = {'executed': True, 'success': False, 'final': {'exit_code': 1}}
    stopped = {'executed': True, 'success': True, 'final': {'exit_code': 0}, **markers}
    fake_runner = Mock()
    fake_runner.run.side_effect = [initial, stopped]
    llm = ScriptedLLM(valid_patch)
    with CollaborationSession().scope():
        result = RepositoryRepairer(fake_runner, llm, PatchPolicy(editable=['train.py'])).run(
            repository, steps(), {})
    assert result['success'] is False and len(llm.calls) == 1
    assert fake_runner.run.call_count == 2
    from src.execution_safety import cleanup_is_unconfirmed
    if cleanup_is_unconfirmed(stopped):
        # A still-live candidate may be writing its workspace. Preserve the
        # evidence and block reuse; rollback itself must wait for cleanup.
        assert result['repair_status'] == 'cleanup_unconfirmed'
        assert result['process_cleanup_confirmed'] is False
        assert result['repair_workspace_retained'] is True
        attempt = result['repair_attempts'][0]
        assert attempt['rollback_deferred'] and not attempt.get('changed_files')
        assert snapshot_workspace(repository) != before
    else:
        assert snapshot_workspace(repository) == before


def test_environment_timeout_guard_does_not_require_duplicate_result_flag(repository, tmp_path):
    packet, plan = metadata(repository)
    llm = ScriptedLLM(lambda *_: {})
    result = {'final': {'exit_code': 1, 'stderr': 'installer was stopped'}}
    assert analyze_author_failure(llm, result, plan, packet, repository,
        {'dependency_failure_kind': 'installation_timeout'}, tmp_path, {}) is None
    assert llm.calls == []


@pytest.mark.parametrize('interrupt', [KeyboardInterrupt(), SystemExit(130)])
def test_interrupted_analysis_is_persisted_without_publishing_repair_findings(repository, tmp_path, interrupt):
    packet, plan = metadata(repository)
    llm = Mock(mock_mode=False)
    llm.chat.side_effect = interrupt
    data, session = {}, CollaborationSession()
    with session.scope(), pytest.raises(type(interrupt)):
        analyze_author_failure(llm, {'final': {'exit_code': 1, 'stderr': 'AttributeError: absent API'}},
                               plan, packet, repository, {}, tmp_path, data)
    record = json.loads((tmp_path / 'failure_analyses.json').read_text('utf-8'))[0]
    assert record['action'] == 'interrupted'
    messages = session.snapshot()['messages']
    assert any(message['kind'] == 'analysis_interrupted' for message in messages)
    assert not any(message['recipient'] in {'RepositoryRepairer', 'EnvBuilder'} for message in messages)
