"""A repair must not restart work or claim success before lifecycle completion."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from src.author_runtime_dependencies import run_with_dependency_repair


MISSING_IMPORT = "ModuleNotFoundError: No module named 'einops'"


def _plan():
    return {'requirements': {'author': [], 'compatibility': [], 'repository_namespaces': []},
            'capture': {'device': 'cpu'}}


def _result(success=False):
    return {'success': success, 'executed': True, 'run_dir': '/evidence/failed-run',
            'final': {'success': success, 'exit_code': 0 if success else 1,
                      'stderr': MISSING_IMPORT, 'stdout': ''}}


def _put_evidence(result, location, evidence):
    if location == 'top':
        result.update(evidence)
    elif location == 'final':
        result['final'].update(evidence)
    elif location in {'step', 'stage'}:
        # An earlier step can own remaining processes even when final succeeds.
        result['steps' if location == 'step' else 'stages'] = [{'id': 'earlier', **evidence}]
    else:
        result[location] = deepcopy(evidence)


@pytest.mark.parametrize('location', ['top', 'final', 'step', 'stage'])
@pytest.mark.parametrize('flag,status', [
    ('cancelled', 'interrupted'), ('interrupted', 'interrupted'),
    ('timed_out', 'timed_out'), ('cleanup_unconfirmed', 'cleanup_unconfirmed'),
    ('processes_alive', 'cleanup_unconfirmed'),
])
@pytest.mark.parametrize('claimed_success', [False, True])
def test_terminal_lifecycle_prevents_retry_even_with_stale_success(
        tmp_path, location, flag, status, claimed_success):
    result = _result(claimed_success)
    _put_evidence(result, location, {flag: True})
    runner = Mock()
    runner.run.return_value = result
    analyze, numeric, path, pyg = (Mock() for _ in range(4))
    data, events = {}, []
    outcome = run_with_dependency_repair(runner, tmp_path, [], {}, _plan(), {'file_inventory': {}},
        tmp_path, data, events.append, lambda: None, repair_numeric=numeric, repair_path=path,
        analyze_failure=analyze, repair_pyg=pyg)
    assert outcome['success'] is False
    assert outcome['final']['stderr'] == MISSING_IMPORT
    assert runner.run.call_count == 1
    assert not data['dependency_repair_attempts']
    for callback in (analyze, numeric, path, pyg):
        callback.assert_not_called()
    saved = json.loads((tmp_path / 'self_repair.json').read_text('utf-8'))
    assert saved['status'] == status and saved['reason']
    assert all(event.get('log', {}).get('status') != 'SUCCESS' for event in events)


@pytest.mark.parametrize('location', ['top', 'final', 'step', 'cleanup_receipt', 'environment_cleanup'])
@pytest.mark.parametrize('evidence', [
    {'process_cleanup_confirmed': False}, {'cleanup_confirmed': False},
    {'cleanup_errors': ['descendant remains']}, {'status': 'cleanup_unconfirmed'},
])
def test_cleanup_receipts_override_zero_exit_and_stop_before_dependency_install(
        tmp_path, location, evidence):
    result = _result(True)
    _put_evidence(result, location, evidence)
    runner = Mock()
    runner.run.return_value = result
    data = {}
    outcome = run_with_dependency_repair(runner, tmp_path, [], {}, _plan(), {'file_inventory': {}},
        tmp_path, data, lambda _: None, lambda: None)
    assert outcome['success'] is False and outcome['final']['exit_code'] == 0
    assert data['self_repair']['status'] == 'cleanup_unconfirmed'
    assert runner.run.call_count == 1
    assert not data['dependency_repair_attempts']


@pytest.mark.parametrize('evidence,status', [
    ({'cleanup_receipt': {'process_cleanup_confirmed': False}}, 'cleanup_unconfirmed'),
    ({'cancelled': True}, 'interrupted'), ({'interrupted': True}, 'interrupted'),
    ({'timed_out': True}, 'timed_out'),
])
def test_pending_repair_is_not_resolved_by_unsafe_success(tmp_path, evidence, status):
    failure, unsafe = _result(), _result(True)
    unsafe.update(evidence, run_dir='/evidence/retry')
    runner = Mock()
    runner.run.side_effect = [failure, unsafe]
    data, events = {}, []
    steps = [{'argv': ['python', 'train.py', '--epochs', '200'], 'cwd': '.'}]
    original_steps = deepcopy(steps)
    outcome = run_with_dependency_repair(runner, tmp_path, steps, {}, _plan(), {'file_inventory': {}},
        tmp_path, data, events.append, lambda: None)
    assert outcome['success'] is False
    assert runner.run.call_count == 2 and len(data['dependency_repair_attempts']) == 1
    assert steps == original_steps
    state = json.loads((tmp_path / 'self_repair.json').read_text('utf-8'))
    assert state['status'] == status
    assert state['attempts'][0]['status'] == status
    assert state['attempts'][0]['verification_execution_dir'] == '/evidence/retry'
    assert all(event.get('log', {}).get('status') != 'SUCCESS' for event in events)


def test_installation_timeout_cannot_be_hidden_by_success(tmp_path):
    result = _result(True)
    result['effective_env_config'] = {'dependency_failure_kind': 'installation_timeout'}
    runner = Mock()
    runner.run.return_value = result
    data = {}
    outcome = run_with_dependency_repair(runner, tmp_path, [], {}, _plan(), {'file_inventory': {}},
        tmp_path, data, lambda _: None, lambda: None)
    assert outcome['success'] is False
    assert data['self_repair']['status'] == 'timed_out'
    assert runner.run.call_count == 1
