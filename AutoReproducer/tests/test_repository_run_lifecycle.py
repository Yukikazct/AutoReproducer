"""Invocation completion is independent of repository step completion."""
from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.agents.code_executor as executor_module
import src.repository_runner as runner_module
from src.agents.code_executor import CodeExecutorAgent
from src.repository_runner import RepositoryRunner


def plan(*, required=True, artifacts=None):
    return [{"id": "work", "argv": ["python", "-c", "print('OK')"],
             "required": required, "artifacts": artifacts or []}]


def run_events(events):
    return [event for event in events if event["type"] == "execution_run"]


@pytest.fixture
def invocation_runner(monkeypatch):
    monkeypatch.setattr(executor_module, "ResourceEventLogger", Mock)
    executor = CodeExecutorAgent(None, logger=Mock())
    executor._ensure_local_deps = Mock(return_value=None)
    executor._exec_env = Mock(return_value=dict(os.environ))
    runner, order = RepositoryRunner(executor=executor), []

    @contextmanager
    def dependency_scope(**kwargs):
        order.append("dependency_enter")
        try:
            yield
        finally:
            order.append("dependency_exit")

    executor.dependency_scope = dependency_scope

    def execute(root, step, run_dir, emit):
        record = {"id": step["id"], "execution_id": run_dir.name,
                  "step_index": step["step_index"], "step_count": step["step_count"],
                  "success": True, "executed": True, "exit_code": 0,
                  "required": step["required"], "stage": "full",
                  "stdout": "OK\n", "stderr": "", "artifacts": []}
        emit({"type": "repository_step", "step_id": step["id"],
              "status": "running", **record})
        order.append("execute")
        emit({"type": "repository_step", "step_id": step["id"],
              "status": "success", **record})
        return record

    runner._execute = Mock(side_effect=execute)

    def collect(*args):
        order.append("collect_images")
        return {"artifacts": [{"path": "result.png"}], "artifact_warnings": []}

    monkeypatch.setattr(runner_module, "collect_images", Mock(side_effect=collect))
    return runner, order


def test_terminal_follows_collection_result_computation_and_dependency_exit(invocation_runner, tmp_path):
    runner, order = invocation_runner
    events = []

    def observe(event):
        events.append(event)
        order.append((event["type"], event.get("status")))

    result = runner.run(tmp_path, plan(), {}, on_event=observe)
    assert result["success"]
    assert result["final"]["artifacts"] == [{"path": "result.png"}]
    identity = Path(result["run_dir"]).name
    assert run_events(events) == [
        {"type": "execution_run", "execution_id": identity, "status": "running"},
        {"type": "execution_run", "execution_id": identity, "status": "success"}]
    assert order.index(("repository_step", "success")) < order.index("collect_images")
    assert order.index("collect_images") < order.index("dependency_exit") < len(order) - 1
    assert order[-1] == ("execution_run", "success")


def test_optional_final_failure_uses_the_whole_run_verdict(invocation_runner, tmp_path):
    runner, _ = invocation_runner
    runner._execute.side_effect = None
    runner._execute.return_value = {"id": "work", "success": False, "executed": True,
                                    "required": False, "exit_code": 7, "artifacts": []}
    events = []
    result = runner.run(tmp_path, plan(required=False), {}, on_event=events.append)
    assert result["success"] is True and result["final"]["exit_code"] == 7
    assert [event["status"] for event in run_events(events)] == ["running", "success"]


def test_dependency_failure_closes_run_after_releasing_scope(invocation_runner, tmp_path):
    runner, order = invocation_runner
    runner.executor._ensure_local_deps.return_value = "dependency failure"
    events = []

    def observe(event):
        events.append(event)
        order.append((event["type"], event.get("status")))

    result = runner.run(tmp_path, plan(), {}, on_event=observe)
    assert not result["success"] and not result["executed"]
    assert result["final"]["exit_code"] == -4
    runner._execute.assert_not_called()
    assert [event["status"] for event in run_events(events)] == ["running", "error"]
    assert order[-2:] == ["dependency_exit", ("execution_run", "error")]


@pytest.mark.parametrize("steps,use_docker", [(plan(), True), ([], False)])
def test_preflight_rejection_emits_no_invocation(invocation_runner, tmp_path, steps, use_docker):
    runner, _ = invocation_runner
    events = []
    result = runner.run(tmp_path, steps, {}, use_docker=use_docker, on_event=events.append)
    assert result["not_runnable"] and not result["executed"]
    assert events == []
    runner.executor._ensure_local_deps.assert_not_called()


def test_post_step_bookkeeping_error_closes_invocation_with_error(invocation_runner, tmp_path, monkeypatch):
    runner, order = invocation_runner
    original = RuntimeError("image bookkeeping failed")
    monkeypatch.setattr(runner_module, "collect_images", Mock(side_effect=original))
    events = []

    def observe(event):
        events.append(event)
        order.append((event["type"], event.get("status")))

    with pytest.raises(RuntimeError) as caught:
        runner.run(tmp_path, plan(), {}, on_event=observe)
    assert caught.value is original
    assert ("repository_step", "success") in order
    assert order[-2:] == ["dependency_exit", ("execution_run", "error")]


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("observer_error", [KeyboardInterrupt, SystemExit])
def test_terminal_observer_cannot_replace_original_failure(invocation_runner, tmp_path, error, observer_error):
    runner, order = invocation_runner
    original = error("original failure")
    runner._execute.side_effect = original
    events = []

    def observe(event):
        events.append(event)
        if event["type"] == "execution_run" and event["status"] != "running":
            assert order[-1] == "dependency_exit"
            raise observer_error("terminal observer failed")

    with pytest.raises(error) as caught:
        runner.run(tmp_path, plan(), {}, on_event=observe)
    assert caught.value is original
    status = "error" if error is RuntimeError else "interrupted"
    assert [event["status"] for event in run_events(events)] == ["running", status]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("at_status", ["running", "success"])
def test_observer_cancellation_still_propagates(invocation_runner, tmp_path, interruption, at_status):
    runner, order = invocation_runner
    original = interruption("observer cancellation")
    events = []

    def observe(event):
        events.append(event)
        if event["type"] == "execution_run" and event["status"] == at_status:
            if at_status == "success":
                assert order[-1] == "dependency_exit"
            raise original

    with pytest.raises(interruption) as caught:
        runner.run(tmp_path, plan(), {}, on_event=observe)
    assert caught.value is original
    if at_status == "running":
        runner.executor._ensure_local_deps.assert_not_called()
        assert [event["status"] for event in run_events(events)] == ["running", "interrupted"]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interruption_identity_survives_persistence_failure(invocation_runner, tmp_path, monkeypatch, interruption):
    runner, order = invocation_runner
    original = interruption("dependency preparation cancelled")
    runner.executor._ensure_local_deps.side_effect = original
    write_text = Path.write_text

    def failing_write(path, value, *args, **kwargs):
        if path.name in {"execution.json", "environment.json"} and "interrupted" in value:
            raise OSError("interrupted metadata cannot be saved")
        return write_text(path, value, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write)
    events = []
    with pytest.raises(interruption) as caught:
        runner.run(tmp_path, plan(), {}, on_event=events.append)
    assert caught.value is original
    assert caught.value.execution["cancelled"]
    assert order[-1] == "dependency_exit"
    assert [event["status"] for event in run_events(events)] == ["running", "interrupted"]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_cancelled_step_keeps_original_when_artifact_observer_also_cancels(invocation_runner, tmp_path, interruption):
    runner, _ = invocation_runner
    runner._execute = RepositoryRunner._execute.__get__(runner)
    (tmp_path / "evidence.txt").write_text("evidence", encoding="utf-8")
    original = interruption("cancel before launch")
    events = []

    def observe(event):
        events.append(event)
        if event["type"] == "repository_step" and event.get("status") == "running":
            raise original
        if event["type"] == "repository_artifact":
            raise SystemExit("artifact observer also cancelled")

    with pytest.raises(interruption) as caught:
        runner.run(tmp_path, plan(artifacts=["evidence.txt"]), {}, on_event=observe)
    assert caught.value is original
    assert caught.value.execution["final"]["exit_code"] == 130
    assert any(event["type"] == "repository_artifact" for event in events)
    assert [event["status"] for event in run_events(events)] == ["running", "interrupted"]
