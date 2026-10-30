"""Preparation recovery cannot become an author-command rerun claim."""
import pytest

from src.author_self_repair import record_repair, verify_repair_result
from src.intelligent_recovery import RecoverySession


@pytest.mark.parametrize('phase,step,status', [
    ('environment_preparation', 'native_preflight', 'environment_recovered'),
    ('dataset_preparation', 'prepare_author_datasets', 'dataset_recovered'),
    ('author_execution', 'train', 'command_recovered'),
])
def test_repair_ledger_legacy_record_and_investigation_keep_the_verified_scope(tmp_path, phase, step, status):
    failed = {'success': False, 'executed': True, 'final': {'id': step, 'stderr': 'No module named helper'}}
    recovered = {'success': True, 'executed': True, 'run_dir': 'actual-rerun',
                 'final': {'id': step, 'exit_code': 0}}
    data = {}
    with RecoverySession(None).scope() as session, session.phase(phase):
        session._active_task = {'id': 'task_1', 'trigger': 'execution_failure', 'phase': phase,
                                'status': 'repair_proposed'}
        session.state['tasks'].append(session._active_task)
        record_repair(tmp_path, data, lambda _: None, failed, 'restore_import', 'Restore existing import root')
        verify_repair_result(tmp_path, data, lambda _: None, recovered)
        session.record_execution(recovered)
        task, action = session._active_task, session.state['actions'][0]
        assert task['status'] == status
        assert task['recovered_phase'] == phase
        assert task['phase_recovered'] is True
        assert task['execution_recovered'] is (phase == 'author_execution')
        assert action['status'] == 'verified'
        assert action['verification']['recovered_phase'] == phase
        assert action['execution'].get('original_command_rerun', False) is (phase == 'author_execution')
        attempt = data['self_repair']['attempts'][0]
        assert attempt['status'] == 'resolved'
        assert attempt['recovered_phase'] == data['self_repair']['recovered_phase'] == phase
        assert attempt['original_command_rerun'] is (phase == 'author_execution')
        if phase != 'author_execution':
            assert '作者实验尚未核验' in data['self_repair']['reason']
            assert 'Original author command rerun' not in action['verification']['reason']


def test_environment_preflight_cannot_resolve_an_existing_author_execution_failure():
    with RecoverySession(None).scope() as session:
        task = {'id': 'author_failure', 'trigger': 'execution_failure', 'phase': 'author_execution',
                'status': 'repair_proposed'}
        session._active_task = task
        session.state['tasks'].append(task)
        issue = session.record_issue('Author training failed', evidence={'source_sha256': 'a' * 64})
        with session.phase('environment_preparation'):
            session.record_execution({'success': True, 'executed': True, 'final': {'id': 'native_preflight'}})
        assert task['status'] == 'repair_proposed'
        assert issue['status'] == 'unresolved'
        assert task['phase_checks']['environment_preparation']['success'] is True
        with session.phase('author_execution'):
            session.record_execution({'success': True, 'executed': True, 'final': {'id': 'train'}})
        assert task['status'] == 'command_recovered'
        assert issue['status'] == 'resolved'


def test_unscoped_legacy_receipts_identify_owned_preflight_but_default_to_author_commands(tmp_path):
    with RecoverySession(None).scope() as session:
        assert session.execution_scope({'final': {'id': 'native_preflight'}}) == 'environment_preparation'
        assert session.execution_scope({'steps': [{'id': 'prepare_author_datasets'}]}) == 'dataset_preparation'
        assert session.execution_scope({'executed': True, 'final': {'id': 'train'}}) == 'author_execution'
        assert session.execution_scope({'final': {'id': 'native_preflight'}},
                                       'author_execution') == 'environment_preparation'
        assert session.execution_scope({'steps': [{'id': 'native_preflight'}, {'id': 'train'}],
            'final': {'id': 'train'}}, 'author_execution') == 'author_execution'
    # Older independent callers need no live session to avoid a preflight claim.
    data = {}
    failed = {'success': False, 'executed': True, 'final': {'id': 'native_preflight', 'stderr': 'missing dependency'}}
    record_repair(tmp_path, data, lambda _: None, failed, 'install_dependency', 'Actual missing import')
    verify_repair_result(tmp_path, data, lambda _: None,
        {'success': True, 'executed': True, 'final': {'id': 'native_preflight'}})
    assert data['self_repair']['attempts'][0]['original_command_rerun'] is False
    assert data['self_repair']['recovered_phase'] == 'environment_preparation'


def test_installation_success_without_execution_proof_remains_unverified(tmp_path):
    data = {}
    record_repair(tmp_path, data, lambda _: None,
        {'success': False, 'final': {'stderr': 'Missing dependency'}}, 'install_dependency', 'Actual missing import')
    verify_repair_result(tmp_path, data, lambda _: None, {'success': True, 'executed': False})
    assert data['self_repair']['attempts'][0]['status'] == 'unverified'
    assert data['self_repair']['attempts'][0]['original_command_rerun'] is False


def test_concrete_preflight_receipt_narrows_a_broad_author_phase_in_both_ledgers(tmp_path):
    data = {}
    with RecoverySession(None).scope() as session, session.phase('author_execution'):
        record_repair(tmp_path, data, lambda _: None, {'success': False, 'executed': True,
            'final': {'id': 'native_preflight', 'stderr': 'Missing dependency'}}, 'install_dependency', 'Missing import')
        verify_repair_result(tmp_path, data, lambda _: None,
            {'success': True, 'executed': True, 'final': {'id': 'native_preflight', 'exit_code': 0}})
        action = session.state['actions'][0]
        assert action['phase'] == action['recovered_phase'] == 'environment_preparation'
        assert action['execution'].get('original_command_rerun') is not True
        assert data['self_repair']['attempts'][0]['original_command_rerun'] is False
