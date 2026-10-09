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


def step(identifier, *argv, cwd=".", timeout=3, env=None, **declared):
    """Author one reviewed step; extra keywords carry plan fields (depends_on, ...)."""
    return {"id": identifier, "argv": ["python", *argv], "cwd": cwd,
            "timeout_s": timeout, "env": env or {}, **declared}


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


def test_actual_dependency_and_environment_helpers_need_no_pip_for_empty_plan_deps(tmp_path):
    runner = RepositoryRunner(executor=CodeExecutorAgent(None, logger=Mock()))
    result = runner.run(tmp_path, [step("actual_env", "-c", "print('REAL_ENV_OK')", timeout=5)], {})
    assert result["success"] is True
    assert result["final"]["stdout"] == "REAL_ENV_OK\n"
    assert not (tmp_path / "requirements.txt").exists()
    assert not (tmp_path / ".autorepro_plot").exists()
    assert (Path(result["run_dir"]) / ".autorepro_plot" / "sitecustomize.py").is_file()


def test_multi_file_fixture_survives_spaces_and_chinese_in_paths(runner, tmp_path):
    # argv is a literal list with shell=False, so a workspace that needs quoting in a
    # shell must still import, read relative config and produce evidence unchanged.
    repo = tmp_path / "我的 repo" / "实验 run"
    package = repo / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from .config import LOAD\n", encoding="utf-8")
    (package / "config.py").write_text("LOAD = 42\n", encoding="utf-8")
    (repo / "settings.json").write_text('{"split": "测试切分"}', encoding="utf-8")
    entry = repo / "main.py"
    entry.write_text("import json\nfrom pkg import LOAD\n"
                     "cfg = json.load(open('settings.json', encoding='utf-8'))\n"
                     "print(LOAD, cfg['split'])\n", encoding="utf-8")
    result = runner.run(repo, [step("entry", "main.py",
                                    artifacts=["settings.json"])], {})
    assert result["success"] is True
    assert result["final"]["stdout"] == "42 测试切分\n"
    captured = result["attempts"][0]["artifacts"][0]
    assert captured["matched"] and Path(captured["files"][0]["captured_to"]).is_file()


def test_runner_accepts_a_validated_plan_dict_and_runs_it_in_authored_order(runner, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    probe = "import json;from pkg import VALUE;print(json.dumps({'value': VALUE}))"
    plan = {"version": 1, "mode": "repository", "profile": "dlinear",
            "spec_sha256": "0" * 64, "workspace": str(tmp_path),
            "steps": [step("import_check", "-c", probe),
                      step("train", "-c", "print('TRAINING')")]}
    events = []
    result = runner.run(tmp_path, plan, {}, on_event=events.append)
    assert result["success"] is True
    assert [a["id"] for a in result["attempts"]] == ["import_check", "train"]
    assert json.loads(result["attempts"][0]["stdout"]) == {"value": 7}
    assert [e["step_id"] for e in events if e["type"] == "repository_step"
            and e["status"] == "running"] == ["import_check", "train"]


def test_failed_prerequisite_blocks_downstream_steps_and_is_a_failed_verdict(runner, tmp_path):
    events = []
    result = runner.run(tmp_path, [
        step("train", "-c", "raise SystemExit(3)"),
        step("eval", "-c", "print('MUST_NOT_RUN')", depends_on=["train"]),
    ], {}, on_event=events.append)
    assert result["success"] is False
    assert [a["id"] for a in result["attempts"]] == ["train"]
    assert result["skipped"] == [{"id": "eval", "not_run": "prerequisite_failed",
                                  "blocked_by": "train", "required": True}]
    assert "MUST_NOT_RUN" not in json.dumps(result["attempts"])
    blocked = [e for e in events if e.get("status") == "skipped"]
    assert blocked == [{"type": "repository_step", "step_id": "eval", "status": "skipped",
                        "not_run": "prerequisite_failed", "blocked_by": "train"}]
    # The halted run is still a failure of the whole verdict, and the real log of
    # the step that did run survives on disk.
    assert result["final"]["exit_code"] == 3
    assert Path(result["attempts"][0]["stderr_path"]).is_file()


def test_a_blocked_step_is_never_reported_as_optional_skipped(runner, tmp_path):
    # optional_required=False only excuses a step's own failure from halting later
    # work; it never hides the fact that an earlier failure stopped this step.
    result = runner.run(tmp_path, [
        step("probe", "-c", "raise SystemExit(9)", required=False),
        step("train", "-c", "print('TRAINING')", depends_on=["probe"]),
    ], {})
    assert result["success"] is True
    assert result["skipped"] == []
    assert [a["id"] for a in result["attempts"]] == ["probe", "train"]


def test_missing_prerequisite_product_blocks_the_consumer_even_after_success(runner, tmp_path):
    result = runner.run(tmp_path, [
        step("train", "-c", "print('TRAINING')"),
        step("eval", "-c", "print('MUST_NOT_RUN')", depends_on=["train"],
             requires=["artifacts/checkpoint.pt"]),
    ], {})
    assert result["success"] is False
    assert [a["id"] for a in result["attempts"]] == ["train"]
    assert result["skipped"][0]["not_run"] == "missing_requirement"
    assert result["skipped"][0]["requirement"] == "artifacts/checkpoint.pt"
    assert "MUST_NOT_RUN" not in result["final"]["stdout"]


def test_present_prerequisite_product_lets_the_evaluator_run(runner, tmp_path):
    write = "from pathlib import Path;Path('artifacts').mkdir(exist_ok=True);" \
            "Path('artifacts/checkpoint.pt').write_bytes(b'weights')"
    result = runner.run(tmp_path, [
        step("train", "-c", write),
        step("eval", "-c", "print('EVAL_OK')", depends_on=["train"],
             requires=["artifacts/checkpoint.pt"]),
    ], {})
    assert result["success"] is True
    assert result["skipped"] == []
    assert result["final"]["stdout"] == "EVAL_OK\n"


def test_declared_artifacts_are_hashed_and_kept_when_a_later_step_fails(runner, tmp_path):
    write = ("from pathlib import Path;"
             "Path('results/run1').mkdir(parents=True);"
             "Path('results/run1/checkpoint.pth').write_bytes(b'WEIGHTS');"
             "Path('results/run1/pred.npy').write_bytes(b'PRED')")
    events = []
    result = runner.run(tmp_path, [
        step("train", "-c", write,
             artifacts=["results/*/checkpoint.pth", "results/*/pred.npy"]),
        step("eval", "-c", "raise SystemExit(4)", depends_on=["train"]),
    ], {}, on_event=events.append)
    assert result["success"] is False
    captured = result["attempts"][0]["artifacts"]
    assert [entry["path"] for entry in captured] == ["results/*/checkpoint.pth",
                                                     "results/*/pred.npy"]
    assert all(entry["matched"] for entry in captured)
    import hashlib
    assert captured[0]["files"][0]["sha256"] == hashlib.sha256(b"WEIGHTS").hexdigest()
    kept = Path(captured[0]["files"][0]["captured_to"])
    assert kept.is_relative_to(Path(result["run_dir"]))
    assert kept.read_bytes() == b"WEIGHTS"
    assert any(e["type"] == "repository_artifact" for e in events)
    # Evidence of what genuinely ran is retained even though the verdict failed.
    assert Path(result["attempts"][1]["stderr_path"]).is_file()


def test_a_mistyped_artifact_glob_cannot_turn_a_ran_step_into_a_failure(runner, tmp_path):
    result = runner.run(tmp_path, [step("train", "-c", "print('OK')",
                                        artifacts=["nowhere/*/absent.pth"])], {})
    assert result["success"] is True
    entry = result["attempts"][0]["artifacts"][0]
    assert entry["matched"] is False and entry["reason"] == "not found"


def test_timeout_halts_dependent_steps(runner, tmp_path):
    result = runner.run(tmp_path, [
        step("slow", "-c", "import time;time.sleep(5)", timeout=0.3),
        step("eval", "-c", "print('MUST_NOT_RUN')", depends_on=["slow"]),
    ], {})
    assert result["success"] is False
    assert result["attempts"][0]["timed_out"] is True
    assert result["skipped"][0]["blocked_by"] == "slow"


def test_running_order_follows_declared_steps_not_an_invented_sort(runner, tmp_path):
    # Reading order is running order: a step may only look backwards, so the
    # authored list needs no separate topological sort to be executable.
    result = runner.run(tmp_path, [
        step("first", "-c", "print('ONE')"),
        step("second", "-c", "print('TWO')", depends_on=["first"]),
        step("third", "-c", "print('THREE')", depends_on=["first", "second"]),
    ], {})
    assert result["success"] is True
    assert [a["stdout"].strip() for a in result["attempts"]] == ["ONE", "TWO", "THREE"]


def test_plan_is_rejected_before_any_process_when_dependencies_are_forward(runner, tmp_path):
    result = runner.run(tmp_path, [
        step("a", "-c", "print('A')", depends_on=["b"]),
        step("b", "-c", "print('B')"),
    ], {})
    assert result["not_runnable"] is True
    assert result["executed"] is False
    runner.executor._ensure_local_deps.assert_not_called()
