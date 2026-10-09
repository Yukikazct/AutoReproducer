"""Repository execution contracts, using only tiny standard-library programs."""
import json
import os
import sys
import time
import subprocess
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.agents.code_executor import CodeExecutorAgent
from src.repository_runner import RepositoryRunner
import src.repository_runner as runner_module


@pytest.fixture
def runner():
    executor = CodeExecutorAgent(None, logger=Mock())
    # Keep these execution tests independent of optional plotting imports and
    # machine-installed libraries. Production still calls the same env helper.
    executor._exec_env = Mock(return_value={**os.environ, "PYTHONIOENCODING": "utf-8"})
    executor._ensure_local_deps = Mock(return_value=None)
    return RepositoryRunner(executor=executor)


def step(identifier, *argv, cwd=".", timeout=3, env=None):
    return {"id": identifier, "argv": ["python", *argv], "cwd": cwd,
            "timeout_s": timeout, "env": env or {}}


def test_multiple_files_import_and_nested_cwd_are_preserved(runner, tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "experiment"
    nested.mkdir(parents=True)
    (nested / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")
    original = "from helper import VALUE\nfrom pathlib import Path\nprint(Path.cwd().name, VALUE)\n"
    (nested / "train.py").write_text(original, encoding="utf-8")
    (repo / "requirements.txt").write_text("author-original==1\n", encoding="utf-8")
    events = []
    result = runner.run(repo, [step("train", "-u", "train.py", cwd="experiment")], {},
                        on_event=events.append)
    assert result["success"] is True
    assert result["mode"] == "repository"
    assert result["executed"] is True
    assert result["llm_calls"] == 0
    assert result["final"]["stage"] == "full"
    assert "experiment 42" in result["final"]["stdout"]
    assert result["final"]["argv"][0] == sys.executable
    assert result["final"]["cwd"] == str(nested)
    assert (nested / "train.py").read_text(encoding="utf-8") == original
    assert (repo / "requirements.txt").read_text(encoding="utf-8") == "author-original==1\n"
    assert not (repo / "run.py").exists()
    assert not (repo / ".autorepro_plot").exists()
    assert Path(result["final"]["stdout_path"]).read_text(encoding="utf-8") == result["final"]["stdout"]
    record = json.loads((Path(result["run_dir"]) / "train.json").read_text(encoding="utf-8"))
    assert record["exit_code"] == 0
    assert record["elapsed_s"] > 0
    assert [e["status"] for e in events if e["type"] == "repository_step"] == ["running", "success"]
    assert any(e["type"] == "repository_output" and "42" in e["text"] for e in events)
    prepared = Path(runner.executor._ensure_local_deps.call_args.args[0])
    assert prepared.is_relative_to(Path(result["run_dir"]))
    assert prepared != repo


def test_failure_stops_later_steps_and_retains_full_logs(runner, tmp_path):
    result = runner.run(tmp_path, [
        step("prepare", "-c", "print('READY')"),
        step("failed", "-c", "import sys; print('X'*70000); print('DETAIL',file=sys.stderr);sys.exit(7)"),
        step("forbidden", "-c", "from pathlib import Path;Path('should_not_exist').touch()"),
    ], {})
    assert result["success"] is False
    assert result["executed"] is True
    assert [s["id"] for s in result["attempts"]] == ["prepare", "failed"]
    assert result["final"]["exit_code"] == 7
    assert len(result["final"]["stdout"]) == 70001
    assert "DETAIL" in result["final"]["stderr"]
    assert not (tmp_path / "should_not_exist").exists()
    assert Path(result["final"]["stdout_path"]).read_text(encoding="utf-8") == result["final"]["stdout"]


def test_nested_entry_point_can_import_repository_root_packages(runner, tmp_path):
    package = tmp_path / "repository_local_model"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 123\n", encoding="utf-8")
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "demo.py").write_text("from repository_local_model import VALUE\nprint(VALUE)\n", encoding="utf-8")
    result = runner.run(tmp_path, [step("nested_entry", "examples/demo.py")], {})
    assert result["success"] is True
    assert result["final"]["stdout"] == "123\n"


@pytest.mark.parametrize("cwd", [
    "..", "../outside", "/tmp", r"\tmp", "C:relative", r"C:\tmp",
    r"\\server\share", r"nested\..\outside",
])
def test_cwd_escape_is_rejected_before_any_command(runner, tmp_path, cwd):
    result = runner.run(tmp_path, [step("escape", "-c", "print('BAD')", cwd=cwd)], {})
    assert result["success"] is False
    assert result["executed"] is False
    assert result["attempts"] == []
    assert "relative path" in result["reason"]


@pytest.mark.parametrize("entry", [
    "/absolute.py", r"\absolute.py", "C:relative.py", r"C:\absolute.py",
    r"\\server\share\script.py", r"nested\..\outside.py",
])
@pytest.mark.parametrize("python_script", [True, False])
def test_entry_path_rejected_before_preparation_or_process(runner, tmp_path, monkeypatch, entry, python_script):
    launch = Mock(side_effect=AssertionError("Unsafe entry must not launch a process"))
    monkeypatch.setattr(runner_module.subprocess, "Popen", launch)
    plan = {"id": "unsafe", "argv": ["python", entry] if python_script else [entry]}
    result = runner.run(tmp_path, [plan], {})
    assert result["executed"] is False
    assert "relative path" in result["reason"]
    runner.executor._ensure_local_deps.assert_not_called()
    launch.assert_not_called()


def test_script_escape_and_external_symlink_are_rejected(runner, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    script = tmp_path / "outside.py"
    script.write_text("print('BAD')", encoding="utf-8")
    result = runner.run(repo, [step("escape", "../outside.py")], {})
    assert result["not_runnable"] is True
    (repo / "linked.py").symlink_to(script)
    linked = runner.run(repo, [step("link", "linked.py")], {})
    assert linked["not_runnable"] is True
    assert "symlink" in linked["reason"]
    assert linked["executed"] is False


def test_new_symlink_from_first_step_blocks_later_execution(runner, tmp_path):
    result = runner.run(tmp_path, [
        step("make_link", "-c", "from pathlib import Path;Path('link').symlink_to('/tmp')"),
        step("blocked", "-c", "print('BAD')"),
    ], {})
    assert result["success"] is False
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["success"] is True
    assert result["final"]["executed"] is False
    assert "symlink" in result["final"]["stderr"]


def test_timeout_keeps_partial_logs_and_stops_descendants(runner, tmp_path):
    # The grandchild would write after its parent's deadline if only the direct
    # Python child were killed by the standalone supervisor.
    descendant = "import time;from pathlib import Path;time.sleep(1);Path('escaped_child').touch()"
    code = ("import subprocess,sys,time;"
            f"subprocess.Popen([sys.executable,'-c',{descendant!r}]);"
            "print('PARTIAL',flush=True);time.sleep(5)")
    started = time.monotonic()
    result = runner.run(tmp_path, [step("timeout", "-c", code, timeout=0.3)], {})
    assert result["success"] is False
    assert result["final"]["timed_out"] is True
    assert result["final"]["exit_code"] == 124
    assert "PARTIAL" in result["final"]["stdout"]
    assert "timed out" in result["final"]["stderr"]
    assert time.monotonic() - started < 3
    time.sleep(1)
    assert not (tmp_path / "escaped_child").exists()


def test_step_environment_and_arguments_are_literal(runner, tmp_path):
    code = "import os,sys;print(os.environ['RUN_VALUE']);print(sys.argv[1])"
    literal = "$(touch do_not_create); spaces"
    result = runner.run(tmp_path, [step("env", "-c", code, literal,
                                      env={"RUN_VALUE": "中文 value"})], {})
    assert result["success"] is True
    assert result["final"]["stdout"] == "中文 value\n" + literal + "\n"
    assert not (tmp_path / "do_not_create").exists()
    assert os.environ.get("RUN_VALUE") != "中文 value"


def test_dependency_failure_does_not_launch_repository(runner, tmp_path):
    runner.executor._ensure_local_deps.return_value = "dependency install failed"
    result = runner.run(tmp_path, [step("blocked", "-c", "print('BAD')")],
                        {"requirements_txt": "placeholder==1"})
    assert result["success"] is False
    assert result["executed"] is False
    assert result["final"]["exit_code"] == -4
    assert "dependency install failed" in result["final"]["stderr"]
    runner.executor._exec_env.assert_not_called()


def test_docker_request_is_explicitly_unsupported(runner, tmp_path):
    result = runner.run(tmp_path, [step("no_fallback", "-c", "print('BAD')")], {}, use_docker=True)
    assert result["success"] is False
    assert result["executed"] is False
    assert "unsupported" in result["reason"]
    runner.executor._ensure_local_deps.assert_not_called()


@pytest.mark.parametrize("steps", [[], [step("bad", "-c", "print(1)", timeout=0)],
                                  [step("same", "-c", "print(1)"), step("same", "-c", "print(2)")]])
def test_invalid_plans_are_rejected_before_environment_preparation(runner, tmp_path, steps):
    result = runner.run(tmp_path, steps, {})
    assert result["not_runnable"] is True
    assert result["executed"] is False
    runner.executor._ensure_local_deps.assert_not_called()


def test_progress_observer_failure_does_not_change_verdict(runner, tmp_path):
    def broken_callback(event):
        raise RuntimeError("observer offline")
    result = runner.run(tmp_path, [step("run", "-c", "print('OK')")], {}, on_event=broken_callback)
    assert result["success"] is True
    assert any("observer offline" in warning for warning in result["artifact_warnings"])


def test_windows_timeout_uses_process_tree_termination(monkeypatch):
    monkeypatch.setattr(runner_module, "os", SimpleNamespace(name="nt"))
    process = Mock(pid=1234)
    process.poll.return_value = None
    termination = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    monkeypatch.setattr(runner_module.subprocess, "run", termination)
    assert RepositoryRunner._kill_group(process) is None
    assert termination.call_args.args[0] == ["taskkill", "/PID", "1234", "/T", "/F"]
    assert termination.call_args.kwargs["shell"] is False
    process.kill.assert_not_called()


@pytest.mark.skipif(os.name!='nt',reason='Windows Store AppExecutionAlias venvs')
def test_store_alias_venv_is_rejected_before_environment_or_training(runner,tmp_path,monkeypatch):
    import src.process_lifecycle as lifecycle
    environment=tmp_path/'venv'
    executable=environment/'Scripts'/'python.exe'
    executable.parent.mkdir(parents=True)
    home=tmp_path/'store-alias'
    (environment/'pyvenv.cfg').write_text(f'home = {home}\n',encoding='utf-8')
    monkeypatch.setattr(lifecycle,'sys',SimpleNamespace(executable=str(executable)))
    monkeypatch.setattr(lifecycle,'_is_app_execution_alias',lambda path:path==home/'python.exe')
    launch=Mock(side_effect=AssertionError('unsupported interpreter must not launch'))
    monkeypatch.setattr(runner_module.subprocess,'Popen',launch)
    result=runner.run(tmp_path,[step('blocked','-c',"print('BAD')")],{})
    assert result['not_runnable'] and not result['executed']
    assert 'Windows Store' in result['reason'] and 'python.org' in result['reason']
    runner.executor._ensure_local_deps.assert_not_called()
    launch.assert_not_called()


@pytest.mark.parametrize('exception',[KeyboardInterrupt,SystemExit])
def test_cancellation_stops_descendants_persists_logs_and_releases_cache(runner,tmp_path,exception):
    from src.dependency_cache import cache_guard
    descendant="import time;from pathlib import Path;time.sleep(1);Path('escaped_child').touch()"
    code=("import subprocess,sys,time;"
          f"subprocess.Popen([sys.executable,'-c',{descendant!r}]);"
          "print('CANCEL_READY',flush=True);time.sleep(30)")
    def cancel(event):
        if event.get('type')=='repository_output' and 'CANCEL_READY' in event.get('text',''):
            raise exception('user cancellation')
    with pytest.raises(exception) as caught:
        runner.run(tmp_path,[step('cancel','-c',code,timeout=30)],{},on_event=cancel)
    execution=caught.value.execution
    assert execution['status']=='interrupted' and execution['cancelled']
    assert execution['final']['exit_code']==130
    assert execution['final']['timed_out'] is False
    assert 'CANCEL_READY' in execution['final']['stdout']
    saved=json.loads((Path(execution['run_dir'])/'cancel.json').read_text(encoding='utf-8'))
    assert saved['cancelled'] and not saved['success']
    with cache_guard(runner.executor.deps_lock_root,cleanup=True,timeout=0):
        pass
    time.sleep(1.1)
    assert not (tmp_path/'escaped_child').exists()


def test_actual_dependency_and_environment_helpers_need_no_pip_for_empty_plan_deps(tmp_path):
    runner = RepositoryRunner(executor=CodeExecutorAgent(None, logger=Mock()))
    result = runner.run(tmp_path, [step("actual_env", "-c", "print('REAL_ENV_OK')", timeout=5)], {})
    assert result["success"] is True
    assert result["final"]["stdout"] == "REAL_ENV_OK\n"
    assert not (tmp_path / "requirements.txt").exists()
    assert not (tmp_path / ".autorepro_plot").exists()
    assert (Path(result["run_dir"]) / ".autorepro_plot" / "sitecustomize.py").is_file()
