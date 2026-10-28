"""Process cleanup uncertainty must survive runner, service and environment exit."""
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from src import environment_lifecycle as lifecycle
from src.agents import code_executor
from src.repository_runner import RepositoryRunner
import src.repository_runner as repository


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / 'deps'
    monkeypatch.setenv('AUTOREPRO_DEPS_ROOT', str(root))
    monkeypatch.setenv('AUTOREPRO_DATA_ROOT', str(tmp_path / 'data'))
    monkeypatch.setattr(code_executor, 'DEPS_CACHE_ROOT', root)
    monkeypatch.setattr(repository, 'DEPS_CACHE_ROOT', root)
    from src import runtime_preparation
    monkeypatch.setattr(runtime_preparation, 'release_runtime_downloads', lambda: {
        'status': 'released', 'removed': [], 'errors': [], 'freed_bytes': 0})
    return root


def test_actual_runner_uncertainty_preserves_packages_and_blocks_next_owner(root, tmp_path, monkeypatch):
    executor = code_executor.CodeExecutorAgent(None, logger=Mock())
    executor._ensure_local_deps = Mock(return_value=None)
    executor._exec_env = Mock(return_value={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    executor._dependency_housekeeping = Mock()
    executor._discard_failed_dependency_installs = Mock()
    runner = RepositoryRunner(executor=executor)
    actual_kill = runner._kill_group

    def uncertain_cleanup(process):
        actual_kill(process)
        return 'fixture remaining descendant cannot be ruled out'

    monkeypatch.setattr(runner, '_kill_group', uncertain_cleanup)
    workspace, run_dir = tmp_path / 'repo', tmp_path / 'evidence'
    workspace.mkdir()
    run_dir.mkdir()
    with lifecycle.managed_environment(root) as lease:
        lifecycle.bind_environment_run(run_dir)
        installed = root / 'installed-package.py'
        installed.write_text('VALUE = 17\n', encoding='utf-8')
        result = runner.run(workspace, [{'id': 'author', 'argv': ['python', '-c', "print('DONE')"]}], {})
        lease.result = {'state': 'ERROR', 'data': {'run_dir': str(run_dir), 'execution': result}}
    assert installed.read_text('utf-8') == 'VALUE = 17\n'
    assert result['cleanup_unconfirmed'] and not result['success']
    assert executor.env_config['process_cleanup_confirmed'] is False
    executor._discard_failed_dependency_installs.assert_not_called()
    assert [call.kwargs['phase'] for call in executor._dependency_housekeeping.call_args_list] == ['scope_started']
    receipt = lease.result['data']['environment_cleanup']
    assert receipt['process_cleanup_confirmed'] is False and not receipt['removed']
    assert json.loads((run_dir / 'environment_cleanup.json').read_text('utf-8')) == receipt
    with pytest.raises(lifecycle.EnvironmentCleanupError) as blocked:
        with lifecycle.managed_environment(root):
            pytest.fail('unconfirmed root must not be reused')
    assert blocked.value.process_cleanup_confirmed is False
    from frontend.task_control import register_task
    with pytest.raises(RuntimeError, match='清理|持有'):
        register_task(tmp_path / 'another' / 'progress.json')
    assert installed.is_file()


@pytest.mark.parametrize('exception_type', [OSError, KeyboardInterrupt])
def test_exception_cleanup_receipt_prevents_dependency_release(root, exception_type):
    failure = exception_type('original failure')
    failure.process_cleanup_confirmed = False
    failure.cleanup_receipt = {'status': 'error', 'process_cleanup_confirmed': False,
                               'errors': ['tree termination unconfirmed']}
    with pytest.raises(exception_type) as raised:
        with lifecycle.managed_environment(root):
            (root / 'retained.bin').write_bytes(b'live process may need this')
            raise failure
    assert raised.value is failure and (root / 'retained.bin').is_file()
    with pytest.raises(lifecycle.EnvironmentCleanupError):
        with lifecycle.managed_environment(root):
            pytest.fail('must preserve the unresolved root')


def test_service_persists_cleanup_exception_and_retains_environment(root, tmp_path, monkeypatch):
    from test_discovered_repository_service import setup_service
    _, service, request, _ = setup_service(tmp_path, monkeypatch)
    receipt = {'status': 'error', 'process_cleanup_confirmed': False,
               'errors': ['fixture process cleanup failed']}
    failure = OSError('cleanup exception after author command')
    failure.process_cleanup_confirmed = False
    failure.cleanup_receipt = receipt
    failure.execution = {'success': False, 'executed': True, 'cleanup_unconfirmed': True,
                         'process_cleanup_confirmed': False, 'cleanup_receipt': receipt,
                         'final': {'stderr': 'ORIGINAL AUTHOR FAILURE', 'exit_code': 7}}

    def raise_uncertain(*args, **kwargs):
        (root / 'keep-installed.py').write_text('VALUE = 1', encoding='utf-8')
        raise failure

    service.runner.run = Mock(side_effect=raise_uncertain)
    result = service.run(request)
    assert result['state'] == 'ERROR'
    data = result['data']
    assert data['validation']['failure_kind'] == 'cleanup_unconfirmed'
    assert data['environment_preparation']['final']['stderr'] == 'ORIGINAL AUTHOR FAILURE'
    assert data['environment_cleanup']['process_cleanup_confirmed'] is False
    assert (root / 'keep-installed.py').is_file()
    service.runner.run.assert_called_once()
