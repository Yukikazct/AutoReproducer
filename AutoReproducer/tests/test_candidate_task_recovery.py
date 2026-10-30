"""Implementation fallback cannot obtain a fresh recovery allowance."""
from copy import deepcopy
import shutil

from src import author_execution_plan, discovered_repository_reproduction as service
from src.discovered_repository_plan import PlanEvidenceError
from src.intelligent_recovery import RecoverySession
from test_discovered_repository_service import setup_service


def linked_service(tmp_path, monkeypatch, *, failure='planning'):
    fixture, runner, request, llm = setup_service(tmp_path, monkeypatch)
    original = fixture['snapshot']['url']
    linked = {'url': 'https://github.com/example/linked-training',
              'source': 'readme_linked_training', 'context': 'Author README training implementation'}
    exports = []
    monkeypatch.setattr(service, 'resolve_repository_revision',
        lambda *args, **kwargs: (fixture['snapshot']['revision'], {'source': 'frozen_fixture'}))
    monkeypatch.setattr(service, 'linked_training_repositories', lambda packet: [deepcopy(linked)])

    def export(root, profile, workspace, **kwargs):
        exports.append(profile['repository']['url'])
        shutil.copytree(fixture['workspace'], workspace)
        snapshot = {**deepcopy(fixture['snapshot']), **profile['repository'],
                    'resolved_sha': profile['repository']['revision']}
        llm.proposal['repository'] = deepcopy(profile['repository'])
        return snapshot

    monkeypatch.setattr(service, 'export_repository', export)
    propose = author_execution_plan.propose_author_execution_plan

    def plan(llm, packet, workspace):
        if failure == 'planning' and packet['repository']['url'] == original:
            raise PlanEvidenceError('Original repository contains no supported training command',
                                    category='missing_information')
        return propose(llm, packet, workspace)

    monkeypatch.setattr(author_execution_plan, 'propose_author_execution_plan', plan)
    execute = runner._execute_frozen

    def execute_candidate(*args, **kwargs):
        if failure == 'environment' and args[2]['repository']['url'] == original:
            error = ValueError('Original implementation requires an unavailable author runtime')
            error.reproduction_phase = 'prepare_environment'
            error.category = 'unsupported_runtime'
            raise error
        return execute(*args, **kwargs)

    monkeypatch.setattr(runner, '_execute_frozen', execute_candidate)
    request.update(prepare_only=True, intelligent_recovery=True)
    return runner, request, exports, original, linked


def test_author_linked_replanning_is_verified_and_charged_once(tmp_path, monkeypatch):
    runner, request, exports, original, linked = linked_service(tmp_path, monkeypatch)
    result = runner.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    assert exports == [original, linked['url']]
    recovery = result['data']['intelligent_recovery']
    action, = recovery['actions']
    assert action['kind'] == 'replan_author_implementation'
    assert action['status'] == 'verified'
    assert action['verification']['evidence']['repository']['url'] == linked['url']
    assert not action['execution'].get('original_command_rerun')
    assert recovery['usage']['repair_actions'] == 1
    assert result['data']['validation']['status'] == 'prepared'
    assert result['data']['repository_executed'] is False


def test_exhausted_shared_budget_stops_before_linked_export(tmp_path, monkeypatch):
    runner, request, exports, original, _ = linked_service(tmp_path, monkeypatch)
    with RecoverySession(runner.llm).scope() as session:
        for number in range(8):
            session.run_action('earlier_correction', lambda: True, evidence={'number': number},
                               verify=lambda value: {'success': value})
        result = runner.run(request)
        assert result['state'] == 'ERROR'
        assert exports == [original]
        assert session.state['usage']['repair_actions'] == 8
        assert session.state['status'] == 'stopped'
        assert session.state['actions'][-1]['status'] == 'rejected'
        assert result['data']['repository_executed'] is False


def test_environment_fallback_replanning_uses_the_same_action_budget(tmp_path, monkeypatch):
    runner, request, exports, original, linked = linked_service(tmp_path, monkeypatch, failure='environment')
    result = runner.run(request)
    assert result['state'] == 'COMPLETED', result['error']
    assert exports == [original, linked['url']]
    recovery = result['data']['intelligent_recovery']
    action, = recovery['actions']
    assert 'unavailable author runtime' in action['problem']
    assert action['verification']['success'] is True
    assert recovery['usage']['repair_actions'] == 1
    assert result['data']['validation']['status'] == 'prepared'
