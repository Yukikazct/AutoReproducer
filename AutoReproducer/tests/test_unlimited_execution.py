"""No elapsed-time cut-off may end a running experiment on current entrypoints."""
import os
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.repository_runner import RepositoryRunner
from src.runtime_preparation import run_owned_process
from src.sandbox_timeout import main


def test_owned_worker_waits_for_unlimited_process_to_exit():
    observations = []
    started = time.monotonic()
    result = run_owned_process(
        [sys.executable, '-I', '-c', "import time;time.sleep(.4);print('NATURAL_EXIT')"],
        timeout_s=None, on_poll=lambda process: observations.append(process.poll()))
    assert time.monotonic() - started >= .4
    assert observations and all(state is None for state in observations)
    assert result.returncode == 0 and result.stdout.strip() == 'NATURAL_EXIT'


def test_repository_unlimited_step_has_no_early_completion(tmp_path):
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._exec_env = Mock(return_value={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    executor._ensure_local_deps = Mock(return_value=None)
    events = []
    def observe(event):
        if event.get('type') == 'repository_output' and 'STILL_RUNNING' in event.get('text', ''):
            assert not any(item.get('status') in {'success', 'completed'} for item in events)
        events.append(event)
    result = RepositoryRunner(executor=executor).run(tmp_path, [{
        'id': 'train', 'kind': 'train', 'timeout_s': None,
        'argv': ['python', '-u', '-c',
                 "import time;print('STILL_RUNNING',flush=True);time.sleep(.4);print('NATURAL_EXIT')"],
    }], {}, on_event=observe)
    assert result['success']
    assert result['final']['timeout_s'] is None
    assert not result['final']['timed_out']
    assert result['final']['exit_code'] == 0
    assert 'NATURAL_EXIT' in result['final']['stdout']
    assert [event['status'] for event in events if event['type'] == 'repository_step'] == ['running', 'success']


def test_standalone_supervisor_accepts_unlimited_execution(monkeypatch):
    monkeypatch.delenv('AUTOREPRO_PARENT_PID', raising=False)
    assert main(['execution', 'none', sys.executable, '-I', '-c',
                 'import time;time.sleep(.3)']) == 0


def test_generic_local_keeps_all_output_after_owned_process_migration(tmp_path):
    script = tmp_path / 'run.py'
    size = 2 * 1024 * 1024 + 100
    script.write_text(f"print('x' * {size})", encoding='utf-8')
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._exec_env = Mock(return_value={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    result = executor._run_local_script(str(script), str(tmp_path), None)
    assert result['success'] and result['stdout'].rstrip() == 'x' * size


@pytest.mark.parametrize('stage', ['smoke', 'full'])
def test_generic_local_ignores_legacy_execution_limits(tmp_path, monkeypatch, stage):
    import src.agents.code_executor as module
    monkeypatch.setattr(module, 'LOCAL_TIMEOUT_SMOKE', .01)
    monkeypatch.setattr(module, 'LOCAL_TIMEOUT_FULL', .01)
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._ensure_local_deps = Mock(return_value=None)
    executor._exec_env = Mock(return_value={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    executor._collect_disk_usage = Mock(return_value={})
    result = executor._execute_code_local("import time;time.sleep(.2);print('COMPLETE')", stage, str(tmp_path))
    assert result['success'] and result['exit_code'] == 0
    assert 'COMPLETE' in result['stdout']


def test_generic_docker_cancel_removes_only_owned_container(tmp_path, monkeypatch):
    executor = CodeExecutorAgent(None, logger=Mock(), use_docker=True)
    monkeypatch.setattr(executor, '_resolve_docker_cmd', lambda: 'docker')
    from src.base_agent import BaseAgent
    monkeypatch.setattr(BaseAgent, 'docker_engine_available', staticmethod(lambda *_: (True, None)))
    launched = []
    def cancel(base, image, runner, timeout):
        launched.append(base)
        assert timeout is None and runner[4].lower() == 'none'
        raise KeyboardInterrupt('user stop')
    monkeypatch.setattr(executor, '_run_docker_cmd_with_sandbox', cancel)
    cleanup = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, 'run', cleanup)
    executor._collect_disk_usage = Mock(return_value={})
    with pytest.raises(KeyboardInterrupt, match='user stop'):
        executor._execute_code_docker('print(1)', 'full', str(tmp_path))
    name = launched[0][launched[0].index('--name') + 1]
    assert cleanup.call_args.args[0] == ['docker', 'rm', '-f', name]
