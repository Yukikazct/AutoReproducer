"""Small real processes exercise cancellation on macOS, Linux, and Windows."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from filelock import FileLock

from src.agents.code_executor import CodeExecutorAgent
from src.dependency_cache import cache_guard
from src.method_adapters import read_json, write_json
from src.method_profiles import method_profile
from src.method_reproduction import MethodReproduction
from src.repository_runner import RepositoryRunner
from src.run_lifecycle import recover_run


def wait_file(path, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(.05)
    raise AssertionError(f"process did not produce {path.name}")


def test_cancellation_kills_descendants_retains_logs_and_releases_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr("src.repository_runner.DEPS_CACHE_ROOT", cache)
    repo = tmp_path / "中文 repo"; repo.mkdir()
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._exec_env = Mock(return_value=dict(os.environ))
    executor._ensure_local_deps = Mock(return_value=None)
    child = "import time;from pathlib import Path;time.sleep(1);Path('escaped').touch()"
    program = ("import subprocess,sys,time;" f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
               "print('READY',flush=True);time.sleep(10)")
    def cancel(event):
        if event.get("type") == "repository_output" and "READY" in event.get("text", ""):
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as interrupted:
        RepositoryRunner(executor=executor).run(repo, [
            {"id": "train", "argv": ["python", "-c", program], "timeout_s": 10},
            {"id": "later", "argv": ["python", "-c", "raise AssertionError('must not run')"]},
        ], {"deadline_monotonic": time.monotonic()+20}, on_event=cancel)
    result = interrupted.value.execution
    assert result["cancelled"] and not result["success"]
    assert result["final"]["exit_code"] == 130 and not result["final"]["timed_out"]
    assert "READY" in result["final"]["stdout"]
    assert result["skipped"][0]["id"] == "later"
    saved = read_json(Path(result["run_dir"]) / "train.json")
    assert saved["cancelled"] and saved["executed"]
    with cache_guard(cache, cleanup=True):
        pass
    time.sleep(1.1)
    assert not (repo / "escaped").exists()


def test_forced_owner_exit_stops_orphan_experiment_tree(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    ready = tmp_path / "ready"
    escaped = repo / "escaped"
    grandchild = "import time;from pathlib import Path;time.sleep(2);Path('escaped').touch()"
    program = ("import subprocess,sys,time;" f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]);"
               "print('READY',flush=True);time.sleep(20)")
    driver = """
import os,sys
from pathlib import Path
from unittest.mock import Mock
from src.agents.code_executor import CodeExecutorAgent
from src.repository_runner import RepositoryRunner
executor=CodeExecutorAgent(None,logger=Mock())
executor._exec_env=Mock(return_value=dict(os.environ))
executor._ensure_local_deps=Mock(return_value=None)
def event(item):
    if item.get('type')=='repository_output' and 'READY' in item.get('text',''):
        Path(sys.argv[2]).touch()
RepositoryRunner(executor=executor).run(sys.argv[1],[{'id':'run','argv':['python','-c',sys.argv[3]],'timeout_s':30}],{},on_event=event)
"""
    owner = subprocess.Popen([sys.executable, "-c", driver, str(repo), str(ready), program],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_file(ready)
        owner.kill()
        owner.wait(timeout=5)
        time.sleep(2.2)
        assert not escaped.exists()
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=5)


def test_os_cancellation_signal_keeps_supervisor_available_for_cleanup(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    ready, final = tmp_path / "ready", tmp_path / "cancelled.json"
    child = "import time;from pathlib import Path;time.sleep(1.5);Path('escaped').touch()"
    command = ("import subprocess,sys,time;" f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
               "print('READY',flush=True);time.sleep(20)")
    driver = """
import json,os,sys
from pathlib import Path
from unittest.mock import Mock
from src.agents.code_executor import CodeExecutorAgent
from src.dependency_cache import cache_guard
from src.repository_runner import RepositoryRunner
from src.run_lifecycle import cancellation_signals
executor=CodeExecutorAgent(None,logger=Mock())
executor._exec_env=Mock(return_value=dict(os.environ))
executor._ensure_local_deps=Mock(return_value=None)
def event(item):
    if item.get('type')=='repository_output' and 'READY' in item.get('text',''):
        Path(sys.argv[2]).touch()
try:
    with cancellation_signals():
        RepositoryRunner(executor=executor).run(sys.argv[1],[{'id':'run','argv':['python','-c',sys.argv[4]],'timeout_s':30}],{},on_event=event)
except KeyboardInterrupt as exc:
    with cache_guard(os.environ['AUTOREPRO_DEPS_ROOT'],cleanup=True):
        Path(sys.argv[3]).write_text(json.dumps(exc.execution),encoding='utf-8')
"""
    owner = subprocess.Popen([sys.executable, "-c", driver, str(repo), str(ready), str(final), command],
                             env={**os.environ, "AUTOREPRO_DEPS_ROOT": str(tmp_path / "cache")},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=os.name == "posix",
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    try:
        wait_file(ready)
        owner.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        owner.wait(timeout=8)
        saved = read_json(final)
        assert saved["cancelled"] and saved["final"]["exit_code"] == 130
        assert not saved["final"]["timed_out"]
        time.sleep(1.6)
        assert not (repo / "escaped").exists()
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=5)


def test_launcher_exit_cancels_its_worker(tmp_path):
    ready, cancelled = tmp_path / "ready", tmp_path / "cancelled"
    worker = """
import sys,time
from pathlib import Path
from src.run_lifecycle import cancellation_signals
try:
    with cancellation_signals():
        Path(sys.argv[1]).touch()
        while True:
            time.sleep(.05)
except KeyboardInterrupt:
    Path(sys.argv[2]).touch()
"""
    launcher = """
import subprocess,sys,time
from pathlib import Path
process=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
while not Path(sys.argv[2]).exists():
    if process.poll() is not None: raise SystemExit(1)
    time.sleep(.05)
sys.stdin.readline()
"""
    process = subprocess.Popen([sys.executable, "-c", launcher, worker, str(ready), str(cancelled)],
                               stdin=subprocess.PIPE)
    try:
        wait_file(ready)
        process.communicate(input=b"exit\n", timeout=5)
        wait_file(cancelled)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_recovery_refuses_live_owner_and_preserves_finished_trials(tmp_path):
    write_json(tmp_path / "run_status.json", {"status": "running", "pid": 123})
    write_json(tmp_path / "optimization.json", {"status": "running", "optimized": False,
               "trials": [{"status": "completed", "metrics": {"mae": .3}}]})
    for label, status in [("done", "completed"), ("active", "running")]:
        directory = tmp_path / "trials" / label; directory.mkdir(parents=True)
        write_json(directory / "trial.json", {"status": status})
    active = tmp_path / "trials" / "active" / "trial.json"
    original = active.read_bytes()
    with FileLock(str(tmp_path / ".run.lock")):
        assert recover_run(tmp_path)["status"] == "running"
        assert active.read_bytes() == original
    result = recover_run(tmp_path)
    assert result["status"] == "interrupted" and result["recovered"]
    assert read_json(active)["status"] == "interrupted"
    assert read_json(tmp_path / "trials/done/trial.json")["status"] == "completed"
    optimization = read_json(tmp_path / "optimization.json")
    assert optimization["status"] == "interrupted" and not optimization["optimized"]
    assert optimization["trials"][0]["metrics"] == {"mae": .3}
    assert (tmp_path / "recovery_originals/trials/active/trial.json").read_bytes() == original
    assert not (tmp_path / "result.json").exists()
    assert recover_run(tmp_path) == result


def test_recovery_does_not_rewrite_unowned_legacy_run(tmp_path):
    write_json(tmp_path / "optimization.json", {"status": "budget_exhausted"})
    before = (tmp_path / "optimization.json").read_bytes()
    assert recover_run(tmp_path)["status"] == "unknown_owner"
    assert (tmp_path / "optimization.json").read_bytes() == before


def test_recovery_preserves_a_final_result_written_before_owner_exit(tmp_path):
    write_json(tmp_path / "run_status.json", {"status": "running"})
    write_json(tmp_path / "result.json", {"data": {"validation": {"status": "completed"}}, "error": None})
    before = (tmp_path / "result.json").read_bytes()
    assert recover_run(tmp_path)["status"] == "completed"
    assert (tmp_path / "result.json").read_bytes() == before


def test_atomic_json_write_preserves_previous_value_on_serialization_failure(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"status": "running"})
    with pytest.raises(ValueError):
        write_json(path, {"metric": float("nan")})
    assert read_json(path) == {"status": "running"}
    assert list(tmp_path.iterdir()) == [path]


def test_method_cancellation_writes_final_result_and_releases_owner_lock(tmp_path, monkeypatch):
    from src import repository_reproduction
    def export(root, profile, workspace, **kwargs):
        workspace.mkdir()
        return {"path": str(workspace), "url": "https://example.org/official", "files": {}}
    monkeypatch.setattr(repository_reproduction, "export_repository", export)
    adapter = Mock()
    adapter.prepare_dataset.return_value = {}
    adapter.materialize.return_value = {"files": {}}
    adapter.public_sources.return_value = []
    adapter.steps.return_value = [{"id": "train", "argv": ["python", "-c", "print(1)"]}]
    monkeypatch.setattr("src.method_reproduction.get_adapter", lambda p: adapter)
    monkeypatch.setattr("src.agents.report_generator.ReportGeneratorAgent.run", lambda *a, **kw: {"report": "interrupted"})
    logger = Mock(); logger.get_stats.return_value = {}
    runner = Mock(); runner.run.side_effect = KeyboardInterrupt()
    result = MethodReproduction(tmp_path, logger, runner=runner).run({"experiment_profile": "neural_ode_spiral"})
    directory = Path(result["data"]["run_dir"])
    assert result["state"] == "ERROR" and result["data"]["interrupted"]
    assert read_json(directory / "result.json")["data"]["validation"]["status"] == "interrupted"
    assert read_json(directory / "run_status.json")["status"] == "interrupted"
    with FileLock(str(directory / ".run.lock"), timeout=0):
        pass
