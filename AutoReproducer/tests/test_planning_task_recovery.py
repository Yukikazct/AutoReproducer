"""Real pinned planning corrections share the task recovery ledger and stops."""
from copy import deepcopy

import pytest

from src import author_execution_plan as author
from src import author_python
from src.discovered_repository_reproduction import environment_for_plan
from src.intelligent_recovery import RecoverySession, InvestigationStopped, RecoveryBudgetExceeded
from src.planning_recovery import validate_plan_consistency
from test_author_plan_collaboration import Planner, setup_plan


def test_actual_plan_correction_is_verified_on_shared_action_ledger(tmp_path, monkeypatch):
    fixture, packet, proposal = setup_plan(tmp_path, monkeypatch)
    invalid = deepcopy(proposal)
    invalid['repository']['revision'] = 'b' * 40
    llm = Planner([invalid, proposal])
    with RecoverySession(llm).scope() as session:
        session.attach({}, tmp_path / 'run')
        session.freeze_goal({'experiment_request': {'method': 'original'}})
        plan = author.propose_author_execution_plan(llm, packet, fixture['workspace'])
        action, = session.state['actions']
        assert action['phase'] == 'entrypoint_planning'
        assert action['kind'] == 'replan_original_command'
        assert action['status'] == 'verified'
        assert action['verification']['success'] is True
        assert session.state['usage']['repair_actions'] == 1
        assert session.state['usage']['model_requests'] == 2
        assert 'original' in llm.calls[1]
        assert plan['repository'] == proposal['repository']
        assert all(item['status'] == 'resolved' for item in session.state['unresolved_issues'])


def test_same_replan_cannot_receive_new_allowance_in_another_service_phase(tmp_path, monkeypatch):
    fixture, packet, proposal = setup_plan(tmp_path, monkeypatch)
    invalid = deepcopy(proposal)
    invalid['repository']['revision'] = 'b' * 40
    llm = Planner([invalid, proposal, invalid, proposal])
    with RecoverySession(llm).scope() as session:
        author.propose_author_execution_plan(llm, deepcopy(packet), fixture['workspace'])
        with session.phase('another_service'), pytest.raises(InvestigationStopped, match='Unchanged recovery action'):
            author.propose_author_execution_plan(llm, deepcopy(packet), fixture['workspace'])
        assert len(llm.calls) == 3  # Duplicate correction never makes the fourth model request.
        assert session.state['usage']['repair_actions'] == 1


def test_final_complete_plan_resolves_earlier_distinct_structural_errors(tmp_path, monkeypatch):
    fixture, packet, proposal = setup_plan(tmp_path, monkeypatch)
    wrong_repository = deepcopy(proposal)
    wrong_repository['repository']['revision'] = 'b' * 40
    wrong_entrypoint = deepcopy(proposal)
    wrong_entrypoint['steps'][0]['argv'] = ['python', 'missing_original.py']
    llm = Planner([wrong_repository, wrong_entrypoint, proposal])
    with RecoverySession(llm).scope() as session:
        plan = author.propose_author_execution_plan(llm, packet, fixture['workspace'])
        assert plan['steps'][0]['argv'] == proposal['steps'][0]['argv']
        assert len(session.state['actions']) == 2
        assert session.state['actions'][0]['status'] == 'failed'
        assert session.state['actions'][1]['status'] == 'verified'
        assert len(session.state['unresolved_issues']) == 2
        assert all(issue['status'] == 'resolved' for issue in session.state['unresolved_issues'])


def test_planner_source_read_uses_remaining_global_probe_allowance(tmp_path, monkeypatch):
    fixture, packet, proposal = setup_plan(tmp_path, monkeypatch)
    packet['repository']['files'].pop('author_model.py')
    llm = Planner([{'request_files': ['author_model.py']}, proposal])
    with RecoverySession(llm).scope() as session:
        for number in range(4):
            session.consume_probe(tool='runtime_metadata', arguments={'observation': number})
        with pytest.raises(RecoveryBudgetExceeded, match='observation budget'):
            author.propose_author_execution_plan(llm, packet, fixture['workspace'])
        assert 'author_model.py' not in packet['repository']['files']
        assert len(llm.calls) == 1
        assert session.state['usage']['probes'] == 4


def test_handoff_consistency_checks_source_and_author_environment(tmp_path, monkeypatch):
    fixture, packet, proposal = setup_plan(tmp_path, monkeypatch)
    plan = author.validate_author_execution_plan(proposal, packet, fixture['workspace'])
    environment = environment_for_plan(plan)
    assert validate_plan_consistency(plan, packet, fixture['workspace'], environment)['status'] == 'verified'
    wrong = {**environment, 'author_requirements': ['unrequested-package']}
    with pytest.raises(ValueError, match='author declarations'):
        validate_plan_consistency(plan, packet, fixture['workspace'], wrong)
    source = fixture['workspace'] / plan['steps'][0]['entrypoint']['source_path']
    source.write_bytes(source.read_bytes() + b'\n# source changed\n')
    with pytest.raises(ValueError, match='differs from pinned source'):
        validate_plan_consistency(plan, packet, fixture['workspace'], environment)


def test_cancelled_python_install_does_not_start_another_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(author_python, 'select_author_python', lambda *args, **kwargs:
                        pytest.fail('Cancelled installation cannot trigger Python selection'))
    preparation = {'success': False, 'cancelled': True, 'cleanup_confirmed': False,
        'effective_env_config': {'dependency_failure_kind': 'python_incompatible'}}
    with RecoverySession(None).scope() as session:
        with pytest.raises(InvestigationStopped):
            author_python.retry_requested_python_conflict({}, tmp_path, {}, {'python_version': '3.12'},
                preparation, lambda _: pytest.fail('Must not rerun after cancellation'))
        assert session.state['cleanup_confirmed'] is False
        assert not session.state['actions']


@pytest.mark.parametrize('executed', [True, False])
def test_python_switch_ledger_records_actual_original_preflight_outcome(tmp_path, monkeypatch, executed):
    monkeypatch.setattr('src.agents.code_executor._requested_python_conflict',
        lambda diagnostic, requirements: {'requires_python': '<3.12', 'package': 'original-dependency'})
    monkeypatch.setattr(author_python, 'select_author_python', lambda *args, **kwargs:
        {'python_version': '3.11.9', 'python_executable': '/managed/python'})
    environment = {'python_version': '3.12.1', 'requirements_txt': 'original-dependency==1.0'}
    failed = {'success': False, 'cleanup_confirmed': True, 'effective_env_config': {
        **environment, 'dependency_failure_kind': 'python_incompatible',
        'dependency_install_attempts': [{'failure_kind': 'python_incompatible', 'diagnostic': 'Requires-Python <3.12'}]}}
    executions = []
    def preflight(effective):
        executions.append(deepcopy(effective))
        return {'success': True, 'executed': executed, 'run_dir': str(tmp_path / 'preflight'), 'cleanup_confirmed': True}
    with RecoverySession(None).scope() as session:
        result = author_python.retry_requested_python_conflict({}, tmp_path, environment,
            {'python_version': '3.12.1'}, failed, preflight)
        assert result['success']
        action, = session.state['actions']
        assert action['kind'] == 'select_compatible_author_python'
        assert action['phase'] == 'environment_preparation'
        assert action['status'] == ('verified' if executed else 'executed')
        assert action['verification']['recovered_phase'] == ('environment_preparation' if executed else None)
        assert action['execution'].get('original_command_rerun') is not True
        assert session.state['usage']['repair_actions'] == 1
        assert executions[0]['requirements_txt'] == 'original-dependency==1.0'
