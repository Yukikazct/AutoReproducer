"""Real timeout ownership at the preset dependency boundary; no pip/network."""
import subprocess
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


def test_generic_dependency_command_preserves_existing_runner(monkeypatch, tmp_path):
    instance = executor(tmp_path)
    instance.env_config["auto_prepare"] = False
    command = ["python", "-m", "pip"]
    result = subprocess.CompletedProcess(command, 0, "done", "")
    runner = Mock(return_value=result)
    monkeypatch.setattr(ce.subprocess, "run", runner)
    assert instance._run_dependency_command(command, timeout=300, env={"PIP_USER": "0"}) is result
    runner.assert_called_once_with(command, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=300, env={"PIP_USER": "0"})
