"""Task-wide recovery retains evidence and cannot reset action allowances."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event
from types import SimpleNamespace

import pytest

from src.agent_collaboration import CollaborationSession
from src.intelligent_recovery import (InvestigationStopped, RecoveryBudgetExceeded,
    RecoverySession, current_investigation, investigating_run)


def test_nested_service_reuses_budget_and_keeps_primary_attachments(tmp_path):
    parent, child = {}, {}
    (tmp_path / 'parent').mkdir()
    (tmp_path / 'child').mkdir()
    service = SimpleNamespace(llm=None)

    @investigating_run
    def nested(self, request):
        assert current_investigation() is session
        session.attach(child, tmp_path / 'child')
        return session.run_action('replan', lambda: {'valid': True},
            problem='entrypoint missing', evidence={'source': 'b' * 64},
            verify=lambda result: {'success': result['valid']})

    with RecoverySession(None, budget={'max_repair_actions': 1}).scope() as session:
        session.attach(parent, tmp_path / 'parent')
        nested(service, {'intelligent_recovery': False})
        assert session.run_dir == tmp_path / 'parent'
        assert parent['intelligent_recovery'] is child['intelligent_recovery'] is session.state
        assert child['recovery_summary']['remaining_budget']['repair_actions'] == 0
        with pytest.raises(RecoveryBudgetExceeded):
            session.run_action('environment', lambda: pytest.fail('budget reset'), evidence={'new': True})
    saved = json.loads((tmp_path / 'parent' / 'intelligent_recovery.json').read_text())
    assert saved['usage']['repair_actions'] == 1
    assert saved['actions'][0]['verification']['success'] is True


def test_duplicate_action_ignores_roles_phases_and_execution_directory(tmp_path):
    with RecoverySession(None).scope() as session:
        session.attach({}, tmp_path)
        with session.phase('planning'):
            session.run_action('replan', lambda: False, problem='No train entrypoint',
                evidence={'source_sha256': 'a' * 64, 'run_dir': 'first'}, role='Planner',
                verify=lambda value: {'success': value})
        with session.phase('environment'):
            with pytest.raises(InvestigationStopped, match='cannot be repeated'):
                session.run_action('replan', lambda: pytest.fail('unchanged retry'),
                    problem='No train entrypoint', evidence={'source_sha256': 'a' * 64,
                    'run_dir': 'second'}, role='Reviewer')
        assert session.state['stop_reason'] == 'no_progress'
        assert session.state['usage']['repair_actions'] == 1


def test_new_source_evidence_permits_next_action_without_new_allowance():
    with RecoverySession(None).scope() as session:
        for source_hash in ('a' * 64, 'b' * 64):
            session.run_action('replan', lambda: True, problem='Invalid import',
                evidence={'source_sha256': source_hash}, verify=lambda value: {'success': value})
        assert session.state['usage']['repair_actions'] == 2
        assert session.recovery_summary()['actions_verified'] == 2


def test_reviewer_veto_does_not_run_and_unknown_effect_is_not_verified():
    with RecoverySession(None).scope() as session:
        with pytest.raises(ValueError, match='rejected'):
            session.run_action('patch', lambda: pytest.fail('rejected patch ran'),
                review={'status': 'rejected', 'reason': 'Changes protected data split'})
        session.run_action('observe_result', lambda: {'success': True})
        assert session.state['usage']['repair_actions'] == 1
        assert session.recovery_summary()['actions_verified'] == 0
        assert session.state['actions'][0]['execution']['status'] == 'not_executed'
        assert session.state['actions'][1]['verification']['success'] is None


def test_cleanup_failure_retains_receipt_and_prevents_next_action():
    with RecoverySession(None).scope() as session:
        with pytest.raises(InvestigationStopped):
            session.run_action('environment', lambda: {'success': False,
                'cleanup_confirmed': False}, verify=lambda value: {'success': True})
        assert session.state['cleanup_confirmed'] is False
        assert session.state['actions'][-1]['status'] == 'stopped'
        with pytest.raises(InvestigationStopped):
            session.run_action('replan', lambda: pytest.fail('cleanup failure ignored'))


def test_task_facts_and_rejections_survive_mailbox_window():
    with RecoverySession(None).scope() as session, CollaborationSession().scope() as collaboration:
        session.freeze_goal({'dataset': 'original', 'epochs': 100})
        session.remember_fact('entrypoint', {'path': 'train.py'}, evidence_ids=['source:1'])
        session.record_issue('Original data unavailable', evidence={'expected_hash': 'c' * 64})
        with pytest.raises(ValueError):
            session.run_action('replace_dataset', lambda: None,
                review={'status': 'rejected', 'reason': 'Changes frozen experiment'})
        for index in range(20):
            collaboration.send('Reader', 'Planner', 'handoff', {'index': index})
        assert len(collaboration.context('Planner')) == 6
        context = collaboration.task_context()
        assert context['goals']['epochs'] == 100
        assert context['facts']['entrypoint']['evidence_ids'] == ['source:1']
        assert context['unresolved_issues'][0]['problem'] == 'Original data unavailable'
        assert context['rejected_actions'][0]['kind'] == 'replace_dataset'
        with pytest.raises(ValueError, match='Frozen experiment'):
            session.freeze_goal({'epochs': 1})


def test_mutations_from_concurrent_roles_are_serialized():
    session, entered, release, second_entered = RecoverySession(None), Event(), Event(), Event()

    def first():
        entered.set()
        assert release.wait(3)
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        one = executor.submit(session.run_action, 'prepare', first)
        assert entered.wait(3)
        two = executor.submit(session.run_action, 'patch', lambda: second_entered.set())
        assert not second_entered.wait(.05)
        release.set()
        assert one.result(3) is True
        two.result(3)
    assert second_entered.is_set()
    assert session.state['usage']['repair_actions'] == 2


def test_probe_budget_is_shared_with_preplanning_source_reads():
    with RecoverySession(None, budget={'max_probes': 1}).scope() as session:
        session.consume_probe(arguments={'paths': ['train.py'], 'sha256': 'a' * 64})
        with pytest.raises(RecoveryBudgetExceeded):
            session.consume_probe(tool='runtime_metadata', arguments={'packages': ['numpy']})
        assert session.state['usage']['probes'] == 1


def test_actual_original_command_rerun_verifies_legacy_repair():
    with RecoverySession(None).scope() as session:
        session.record_external_repair('install_missing_dependency', problem='No module named x',
            evidence={'requirement': 'x==1'}, reason='Actual missing import')
        session.verify_external_repair({'success': True, 'executed': True,
                                       'run_dir': 'rerun', 'final': {'exit_code': 0}})
        action = session.state['actions'][0]
        assert action['execution']['original_command_rerun'] is True
        assert action['verification']['success'] is True
        assert session.recovery_summary()['actions_executed'] == 1


def test_successful_installation_without_author_execution_cannot_verify_repair():
    with RecoverySession(None).scope() as session:
        session.record_external_repair('install_dependency', problem='No module named x')
        session.verify_external_repair({'success': True, 'executed': False,
                                       'environment_ready': True})
        action = session.state['actions'][0]
        assert action['verification']['success'] is None
        assert action['execution'].get('original_command_rerun') is not True
        assert session.recovery_summary()['actions_verified'] == 0


def test_same_diagnostic_on_different_pinned_sources_does_not_resolve_other_issue():
    with RecoverySession(None).scope() as session:
        old = session.record_issue('Missing data', evidence={'source_sha256': 'a' * 64})
        session.run_action('correct_path', lambda: True, problem='Missing data',
            evidence={'source_sha256': 'b' * 64}, verify=lambda value: {'success': value})
        assert old['status'] == 'unresolved'
        assert session.state['unresolved_issues'][1]['status'] == 'resolved'


def test_interruption_preserves_original_cancellation_and_stops_actions():
    with RecoverySession(None).scope() as session:
        def interrupt():
            raise KeyboardInterrupt('cancelled')
        with pytest.raises(KeyboardInterrupt):
            session.run_action('replan', interrupt)
        assert session.state['actions'][-1]['status'] == 'stopped'
        with pytest.raises(InvestigationStopped):
            session.ensure_running()


def test_later_cancellation_cannot_downgrade_cleanup_failure_or_human_requirement():
    session = RecoverySession(None)
    session.stop('probe_cleanup_unconfirmed', cleanup_confirmed=False, requires_human=True)
    session.stop('run_interrupted', requires_human=False, cleanup_confirmed=True)
    assert session.state['stop_reason'] == 'probe_cleanup_unconfirmed'
    assert session.state['cleanup_confirmed'] is False
    assert session.state['requires_human'] is True


def test_external_blocker_human_requirement_survives_first_cancellation_stop():
    session = RecoverySession(None)
    session.record_issue('Original dataset unavailable', evidence={'source': 'author README'})
    session.stop('run_interrupted', requires_human=False)
    assert session.state['requires_human'] is True


def test_late_execution_receipt_cannot_resurrect_a_stopped_recovery():
    session = RecoverySession(None)
    action = session.begin_action('patch', problem='Old API incompatible')
    session._active_task = {'id': 'investigation_1', 'trigger': 'execution_failure', 'status': 'repair_proposed'}
    session.state['tasks'].append(session._active_task)
    session.stop('process_cleanup_unconfirmed', cleanup_confirmed=False)
    session.finish_action(action, result={'success': True, 'executed': True},
                          verification={'success': True})
    session.record_execution({'success': True, 'executed': True})
    assert action['status'] == 'stopped'
    assert action['verification']['success'] is False
    assert session._active_task['status'] == 'repair_proposed'
    assert session.state['requires_human'] is True


def test_owned_services_meter_read_identity_and_planning_once_and_restore_clients():
    class MeteredModel:
        api_key = ''
        max_tokens = 100

        def __init__(self):
            self.calls, self.last_usage = [], {}

        def chat(self, prompt, **kwargs):
            self.calls.append(kwargs['task'])
            self.last_usage = {'total_tokens': 7, 'monetary_cost': .01}
            return '{}'

    llm = MeteredModel()
    parent, child = SimpleNamespace(llm=llm), SimpleNamespace(llm=llm)
    receipts = []

    @investigating_run
    def nested(self, request):
        session = current_investigation()
        assert self.llm is not llm
        self.llm.chat('{}', task='repository_identity')
        session.wrap_llm(self.llm).chat('{}', task='author_execution_plan')
        return session.state

    @investigating_run
    def outer(self, request):
        session = current_investigation()
        self.llm.chat('{}', task='paper_reading')
        receipts.append(session)
        return nested(child, {'intelligent_recovery': session.state})

    state = outer(parent, {'intelligent_recovery': True})
    assert parent.llm is child.llm is llm
    assert current_investigation() is None
    assert llm.calls == ['paper_reading', 'repository_identity', 'author_execution_plan']
    assert state is receipts[0].state
    assert state['tasks'] == []  # Preparation metering does not fabricate investigation tasks.
    assert state['usage']['model_requests'] == 3
    assert state['usage']['known_tokens'] == 21
    assert state['usage']['monetary_cost'] == pytest.approx(.03)
    assert {role: usage['model_requests'] for role, usage in state['usage']['roles'].items()} == {
        'paper_reading': 1, 'repository_identity': 1, 'author_execution_plan': 1}


def test_disabled_api_keeps_raw_model_and_owned_failure_restores_it():
    llm = object()
    service = SimpleNamespace(llm=llm)

    @investigating_run
    def run(self, request):
        if request['intelligent_recovery']:
            assert self.llm is not llm
            raise ValueError('Actual planning failure')
        assert self.llm is llm
        return True

    assert run(service, {'intelligent_recovery': False}) is True
    assert service.llm is llm
    with pytest.raises(ValueError, match='planning failure'):
        run(service, {'intelligent_recovery': True})
    assert service.llm is llm
    assert current_investigation() is None
