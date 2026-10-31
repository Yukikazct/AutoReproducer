"""Real command results retain process cleanup failures and block every retry."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.author_runtime_dependencies import run_with_dependency_repair
from src.repository_repairer import RepositoryRepairer
from src.repository_runner import RepositoryRunner
import src.repository_runner as runner_module


@pytest.fixture
def runner():
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._exec_env = Mock(return_value={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    executor._ensure_local_deps = Mock(return_value=None)
    return RepositoryRunner(executor=executor)


def _step(name, source, *, required=True):
    return {'id': name, 'argv': ['python', '-u', '-c', source], 'required': required}


@pytest.mark.parametrize('required', [True, False])
@pytest.mark.parametrize('exit_code', [0, 7])
def test_cleanup_diagnostic_is_structured_and_halts_even_optional_steps(
        runner, tmp_path, monkeypatch, required, exit_code):
    actual_kill = runner._kill_group

    def failed_cleanup(process):
        actual_kill(process)
        return 'fixture descendant termination could not be confirmed'

    monkeypatch.setattr(runner, '_kill_group', failed_cleanup)
    events = []
    result = runner.run(tmp_path, [
        _step('author', f"import sys; print('ORIGINAL OUTPUT'); print('ORIGINAL ERROR',file=sys.stderr); sys.exit({exit_code})",
              required=required),
        _step('forbidden', "from pathlib import Path; Path('continued').touch()"),
    ], {}, on_event=events.append)
    assert not result['success'] and result['cleanup_unconfirmed']
    assert result['process_cleanup_confirmed'] is False
    assert len(result['steps']) == 1 and not (tmp_path / 'continued').exists()
    final = result['final']
    assert final['process_exit_code'] == exit_code
    assert final['exit_code'] == (exit_code or -2)
    assert 'ORIGINAL OUTPUT' in final['stdout'] and 'ORIGINAL ERROR' in final['stderr']
    assert final['cleanup_receipt']['process_cleanup_confirmed'] is False
    assert final['cleanup_errors'] == ['fixture descendant termination could not be confirmed']
    directory = Path(result['run_dir'])
    assert json.loads((directory / 'author.json').read_text('utf-8'))['cleanup_unconfirmed']
    assert json.loads((directory / 'execution.json').read_text('utf-8'))['cleanup_unconfirmed']
    assert events[-1]['type'] == 'execution_run' and events[-1]['status'] == 'error'
    assert not RepositoryRepairer._repairable(result)


@pytest.mark.parametrize('exception_type', [OSError, RuntimeError])
def test_job_cleanup_exception_keeps_identity_receipt_and_original_output(
        runner, tmp_path, monkeypatch, exception_type):
    actual_job = runner_module.ProcessJob
    failure = exception_type('fixture job cleanup failed')

    class FailingJob:
        def __init__(self):
            self.job = actual_job()

        def assign(self, process):
            return self.job.assign(process)

        def resume(self, process):
            return self.job.resume(process)

        def close(self):
            self.job.close()
            raise failure

    monkeypatch.setattr(runner_module, 'ProcessJob', FailingJob)
    events = []
    with pytest.raises(exception_type) as raised:
        runner.run(tmp_path, [_step('author', "print('PRESERVED OUTPUT')")], {}, on_event=events.append)
    assert raised.value is failure
    assert failure.process_cleanup_confirmed is False
    assert failure.cleanup_receipt['process_cleanup_confirmed'] is False
    result = failure.execution
    assert not result['success'] and result['cleanup_unconfirmed']
    assert 'PRESERVED OUTPUT' in result['final']['stdout']
    assert result['final']['process_exit_code'] == 0
    assert len(result['steps']) == 1
    stored = json.loads((Path(result['run_dir']) / 'execution.json').read_text('utf-8'))
    assert stored['cleanup_unconfirmed'] and 'PRESERVED OUTPUT' in stored['final']['stdout']
    assert events[-1]['status'] == 'error'


def test_real_cleanup_failure_blocks_missing_dependency_install_and_model(
        runner, tmp_path, monkeypatch):
    actual_kill = runner._kill_group

    def failed_cleanup(process):
        actual_kill(process)
        return 'fixture process-tree cleanup failed'

    monkeypatch.setattr(runner, '_kill_group', failed_cleanup)
    plan = {'requirements': {'author': [], 'compatibility': [], 'repository_namespaces': []},
            'capture': {'device': 'cpu'}}
    analyze = Mock()
    data = {}
    result = run_with_dependency_repair(runner, tmp_path,
        [_step('author', "raise ModuleNotFoundError(\"No module named 'einops'\")")],
        {}, plan, {'file_inventory': {}}, tmp_path, data, lambda _: None, lambda: None,
        analyze_failure=analyze)
    assert not result['success'] and data['self_repair']['status'] == 'cleanup_unconfirmed'
    assert not data['dependency_repair_attempts']
    analyze.assert_not_called()
    runner.executor._ensure_local_deps.assert_called_once()


def test_cancellation_remains_primary_when_job_cleanup_also_fails(runner, tmp_path, monkeypatch):
    actual_job, actual_kill = runner_module.ProcessJob, runner._kill_group
    cancellation = KeyboardInterrupt('fixture cancellation')

    class FailingJob:
        def __init__(self):
            self.job = actual_job()

        def assign(self, process):
            return self.job.assign(process)

        def resume(self, process):
            return self.job.resume(process)

        def close(self):
            self.job.close()
            raise OSError('fixture cleanup after cancellation failed')

    def cancel_after_exit(process):
        actual_kill(process)
        raise cancellation

    monkeypatch.setattr(runner_module, 'ProcessJob', FailingJob)
    monkeypatch.setattr(runner, '_kill_group', cancel_after_exit)
    with pytest.raises(KeyboardInterrupt) as raised:
        runner.run(tmp_path, [_step('author', "print('BEFORE CANCEL')")], {})
    assert raised.value is cancellation
    assert cancellation.process_cleanup_confirmed is False
    result = cancellation.execution
    assert result['cancelled'] and result['cleanup_unconfirmed']
    assert result['final']['cancelled'] and 'BEFORE CANCEL' in result['final']['stdout']
    assert 'cleanup after cancellation failed' in result['final']['stderr']


def test_dead_group_leader_does_not_prove_cleanup_after_permission_denied(monkeypatch):
    process = SimpleNamespace(pid=12345, poll=lambda: 0, kill=Mock())
    killpg = Mock(side_effect=PermissionError('denied'))
    monkeypatch.setattr(runner_module, 'os', SimpleNamespace(name='posix', killpg=killpg))
    monkeypatch.setattr(runner_module, 'signal', SimpleNamespace(SIGKILL=9))
    diagnostic = RepositoryRunner._kill_group(process)
    assert 'unconfirmed' in diagnostic
    process.kill.assert_not_called()
