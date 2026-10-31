"""Real timeout ownership at the preset dependency boundary; no pip/network."""
import subprocess
import os
from pathlib import Path
import sys
import time
from unittest.mock import Mock

import pytest

import src.agents.code_executor as ce
import src.process_lifecycle as lifecycle
import src.runtime_preparation as runtime
from src.resource_events import ResourceEventLogger


def executor(tmp_path):
    instance = ce.CodeExecutorAgent(None, logger=Mock())
    instance.env_config = {"auto_prepare": True, "dependency_health_check": True}
    instance.deps_cache_root = tmp_path / 'deps'
    instance.resource_events = ResourceEventLogger(tmp_path / "events.jsonl")
    return instance


@pytest.mark.parametrize("boundary", ["installer", "health_probe"])
def test_dependency_timeout_stops_descendant_before_retry_and_keeps_logs(monkeypatch, tmp_path, boundary):
    started_file = tmp_path / "descendant_started.txt"
    sentinel = tmp_path / "surviving_installer.txt"
    descendant = ("import time;from pathlib import Path;"
                  f"Path({str(started_file)!r}).write_text('started');"
                  f"time.sleep(1.6);Path({str(sentinel)!r}).write_text('orphan')")
    program = ("import subprocess,sys,time;"
               f"subprocess.Popen([sys.executable,'-I','-c',{descendant!r}]);"
               "print('partial target copy',flush=True);"
               "print('native wheel extraction',file=sys.stderr,flush=True);time.sleep(10)")
    instance = executor(tmp_path)
    began = time.monotonic()
    if boundary == "installer":
        command = [sys.executable, "-I", "-c", program]
        with pytest.raises(subprocess.TimeoutExpired) as captured:
            instance._run_dependency_command(command, timeout=.8, env=instance._pip_env())
        error = captured.value
        assert error.cmd == command and error.timeout == .8
        assert "partial target copy" in error.stdout
        assert "native wheel extraction" in error.stderr
        assert isinstance(error.__cause__, runtime.RuntimePreparationTimeout)
    else:
        deps = tmp_path / "deps" / "cache"
        deps.mkdir(parents=True)
        monkeypatch.setattr(ce, "DEPENDENCY_HEALTH_PROBE", program)
        monkeypatch.setattr(ce, "LOCAL_DEPENDENCY_HEALTH_TIMEOUT", .8)
        diagnostic = instance._check_local_dependency_health(deps, "native-demo==1.0", "installed")
        assert "健康检查超时(0.8s)" in diagnostic
        assert "partial target copy" in diagnostic and "native wheel extraction" in diagnostic
        record, = instance.env_config["dependency_health_attempts"]
        assert record["timed_out"] and not record["success"]
    assert time.monotonic() - began < 4, "timeout did not finish promptly"
    assert started_file.exists(), "the descendant must have started before its owner timed out"
    # Simulate a retry immediately reusing the same destination. A redirector
    # that survived subprocess.run's kill used to write into this new target.
    sentinel.write_text("retry target", encoding="utf-8")
    time.sleep(1.9)
    assert sentinel.read_text(encoding="utf-8") == "retry target"


def test_store_guard_rejects_owned_dependency_spawn_before_process_start(monkeypatch, tmp_path):
    instance = executor(tmp_path)
    monkeypatch.setattr(lifecycle, "validate_windows_venv", Mock(side_effect=ValueError("Store-backed venv")))
    spawn = Mock(side_effect=AssertionError("unsafe interpreter must not start"))
    monkeypatch.setattr(runtime, "run_owned_process", spawn)
    with pytest.raises(ValueError, match="Store-backed"):
        instance._run_dependency_command(["python", "-m", "pip"], timeout=300, env={})
    spawn.assert_not_called()


def test_owned_dependency_cancellation_propagates(monkeypatch, tmp_path):
    instance = executor(tmp_path)
    monkeypatch.setattr(lifecycle, "validate_windows_venv", lambda: None)
    monkeypatch.setattr(runtime, "run_owned_process", Mock(side_effect=KeyboardInterrupt("cancel")))
    with pytest.raises(KeyboardInterrupt, match="cancel"):
        instance._run_dependency_command(["python", "-m", "pip"], timeout=300, env={})


def test_owned_timeout_output_retains_only_bounded_stream_tails():
    program = ("import sys,time;print('x'*10000+'STDOUT_TAIL',flush=True);"
               "print('y'*10000+'STDERR_TAIL',file=sys.stderr,flush=True);time.sleep(10)")
    with pytest.raises(runtime.RuntimePreparationTimeout) as captured:
        runtime.run_owned_process([sys.executable, "-I", "-c", program], timeout_s=.6,
                                  max_output_bytes=128)
    error = captured.value
    for output, marker in ((error.stdout, "STDOUT_TAIL"), (error.stderr, "STDERR_TAIL")):
        assert output.startswith("[earlier output omitted]\n") and marker in output
        assert len(output) <= 128 + len("[earlier output omitted]\n")


def test_non_pip_dependency_command_preserves_existing_runner(monkeypatch, tmp_path):
    instance = executor(tmp_path)
    instance.env_config["auto_prepare"] = False
    command = ["python", "-I", "probe.py"]
    result = subprocess.CompletedProcess(command, 0, "done", "")
    runner = Mock(return_value=result)
    monkeypatch.setattr(ce.subprocess, "run", runner)
    assert instance._run_dependency_command(command, timeout=300, env={"PIP_USER": "0"}) is result
    runner.assert_called_once_with(command, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=300, env={"PIP_USER": "0"})


def test_pip_owns_processes_and_temporaries_without_auto_prepare(monkeypatch, tmp_path):
    instance = executor(tmp_path)
    instance.env_config['auto_prepare'] = False
    command = ['python', '-m', 'pip']
    result = subprocess.CompletedProcess(command, 0, 'done', '')
    run = Mock(return_value=result)
    monkeypatch.setattr(runtime, 'run_owned_process', run)
    monkeypatch.setattr(lifecycle, 'validate_windows_venv', lambda: None)
    assert instance._run_dependency_command(command, timeout=300, env={'PIP_USER': '0'}) is result
    env = run.call_args.kwargs['env']
    assert env['PIP_USER'] == '0' and env['TMPDIR'] == env['TEMP'] == env['TMP']
    assert not Path(env['TMPDIR']).exists()
    run.assert_called_once_with(command, timeout_s=300, env=env)


@pytest.mark.skipif(os.name != 'nt', reason='Windows pip target compatibility bootstrap')
@pytest.mark.parametrize('outcome', ['success', 'timeout', 'cancel'])
def test_target_bootstrap_retains_owned_runner_env_cleanup_and_original_argv(monkeypatch, tmp_path, outcome):
    instance = executor(tmp_path)
    instance.env_config['auto_prepare'] = False
    monkeypatch.setenv('AUTOREPRO_INSTALLER_INHERITED_TEST', 'inherited value')
    monkeypatch.setenv('LLM_API_KEY', 'must be excluded')
    monkeypatch.setattr(lifecycle, 'validate_windows_venv', lambda: None)
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--target', str(tmp_path / 'target'), 'fixture.whl']
    calls = []

    def owned(actual, **kwargs):
        calls.append((actual, kwargs))
        assert actual[:4] == [sys.executable, '-I', '-X', 'utf8']
        assert Path(actual[4]).name == 'pip_windows_target.py'
        assert actual[5:] == command[3:] + ['--log', str(instance._dependency_install_monitor.log_path), '--progress-bar', 'off']
        environment = kwargs['env']
        assert environment['AUTOREPRO_INSTALLER_INHERITED_TEST'] == 'inherited value'
        assert environment['SYSTEMROOT'] == os.environ['SYSTEMROOT']
        assert environment['PATH'] == os.environ['PATH'] and 'LLM_API_KEY' not in environment
        assert environment['TEMP'].startswith('\\\\?\\')
        assert environment['TEMP'] == environment['TMP'] == environment['TMPDIR']
        if outcome == 'timeout':
            raise runtime.RuntimePreparationTimeout('owned deadline', stdout='partial output', stderr='partial error')
        if outcome == 'cancel':
            raise KeyboardInterrupt('owned cancellation')
        return subprocess.CompletedProcess(actual, 0, 'done', '')

    monkeypatch.setattr(runtime, 'run_owned_process', owned)
    if outcome == 'timeout':
        with pytest.raises(subprocess.TimeoutExpired) as captured:
            instance._run_dependency_command(command, timeout=3, env=instance._pip_env())
        assert captured.value.cmd[:len(command)] == command and captured.value.stdout == 'partial output'
    elif outcome == 'cancel':
        with pytest.raises(KeyboardInterrupt, match='owned cancellation'):
            instance._run_dependency_command(command, timeout=3, env=instance._pip_env())
    else:
        result = instance._run_dependency_command(command, timeout=3, env=instance._pip_env())
        assert result.args[:len(command)] == command and result.stdout == 'done'
    assert len(calls) == 1
    receipt, = instance.env_config['dependency_temporary_files']
    assert receipt['status'] == 'removed' and not Path(receipt['path']).exists()


def test_real_pip_timeout_reaps_descendants_before_scratch_cleanup(monkeypatch, tmp_path):
    instance = executor(tmp_path)
    instance.env_config['auto_prepare'] = False
    scratch_file = tmp_path / 'scratch-path.txt'
    started_file = tmp_path / 'child-started.txt'
    survivor = tmp_path / 'surviving-child.txt'
    child = (
        'import os,time;from pathlib import Path;'
        f'Path({str(started_file)!r}).write_text("started");'
        'time.sleep(1.6);'
        'work=Path(os.environ["TMPDIR"]);work.mkdir(parents=True,exist_ok=True);'
        '(work/"late-copy").write_text("orphan");'
        f'Path({str(survivor)!r}).write_text("orphan")')
    program = (
        'import os,subprocess,sys,time;from pathlib import Path;'
        'work=Path(os.environ["TMPDIR"]);'
        f'Path({str(scratch_file)!r}).write_text(str(work));'
        '(work/"unpacked-wheel").write_bytes(b"x"*3000);'
        f'subprocess.Popen([sys.executable,"-I","-c",{child!r}]);'
        'print("unpacking",flush=True);time.sleep(10)')
    real_owned = runtime.run_owned_process
    monkeypatch.setattr(runtime, 'run_owned_process', lambda argv, **kwargs:
                        real_owned([sys.executable, '-I', '-c', program], **kwargs))
    with pytest.raises(subprocess.TimeoutExpired) as captured:
        instance._run_dependency_command([sys.executable, '-m', 'pip', 'install', 'fixture'],
                                         timeout=.8, env=instance._pip_env())
    assert 'unpacking' in captured.value.stdout
    assert started_file.exists()
    scratch = Path(scratch_file.read_text('utf-8'))
    assert not scratch.exists()
    receipt, = instance.env_config['dependency_temporary_files']
    assert receipt['status'] == 'removed' and receipt['freed_bytes'] >= 3000
    time.sleep(1.9)
    assert not scratch.exists() and not survivor.exists()
