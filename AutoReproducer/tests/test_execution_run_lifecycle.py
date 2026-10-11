"""Invocation lifecycle encloses steps, cleanup, and public result bookkeeping."""
from contextlib import contextmanager
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.agents.code_executor as executor_module
from src.agents.code_executor import CodeExecutorAgent
from src.llm.llm_client import LLMClient


SUCCESS = {"success": True, "exit_code": 0, "stdout": "result\n", "stderr": ""}


@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setattr(executor_module, "ResourceEventLogger", Mock)
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(side_effect=AssertionError("Lifecycle tests must not call a model"))
    agent = CodeExecutorAgent(llm, logger=Mock(), mock_mode=True)
    agent._execute_code_local = Mock(return_value=dict(SUCCESS))
    agent._execute_code_docker = Mock(return_value=dict(SUCCESS))
    return agent


def run_events(events):
    return [event for event in events if event["type"] == "execution_run"]


def step_events(events):
    return [event for event in events if event["type"] == "execution_step"]


def assert_run_pair(events, status):
    runs = run_events(events)
    assert [event["status"] for event in runs] == ["running", status]
    assert isinstance(runs[0]["execution_id"], str) and runs[0]["execution_id"]
    assert len({event["execution_id"] for event in events}) == 1
    assert events[0] is runs[0] and events[-1] is runs[-1]


def assert_idle(executor):
    assert executor._execution_id is None
    assert executor._execution_step_index == 0
    assert executor._execution_run_state is None
    executor.llm.chat.assert_not_called()


@pytest.mark.parametrize("use_docker", [False, True])
@pytest.mark.parametrize("repair", [False, True])
def test_smoke_full_and_repair_keep_one_invocation_open(executor, use_docker, repair):
    events, between_steps = [], []

    def observe(event):
        events.append(event)
        if event["type"] == "execution_step" and event["status"] != "running":
            between_steps.append([item["status"] for item in run_events(events)])

    executor.use_docker, executor.on_event = use_docker, observe
    runner = executor._execute_code_docker if use_docker else executor._execute_code_local
    if repair:
        runner.side_effect = [
            {"success": False, "exit_code": 1, "stdout": "", "stderr": "NameError"},
            dict(SUCCESS), dict(SUCCESS),
        ]
        executor._repair_after_execution = Mock(return_value=(
            "print('fixed')", {"repairable": True}, "print('fixed')"))
    result = executor.run({"code": "print('result')"})

    assert result["success"] is True
    assert_run_pair(events, "success")
    assert between_steps == [["running"]] * (3 if repair else 2)
    assert [call.args[1] for call in runner.call_args_list] == (
        ["smoke", "smoke", "full"] if repair else ["smoke", "full"])
    assert_idle(executor)


def test_reused_scopes_and_nested_steps_only_close_at_owner_exit(executor):
    events = []
    executor.on_event = events.append
    with executor._execution_scope() as outer:
        with executor._execution_scope(reuse=True) as inner:
            assert inner is outer
            first = executor._execute_code("print('first')", "smoke")
            assert [event["status"] for event in run_events(events)] == ["running"]
        assert [event["status"] for event in run_events(events)] == ["running"]
        second = executor._execute_code("print('second')", "full")
        outer["result"] = {"success": True, "final": second, "stages": [first, second]}
        assert [event["status"] for event in run_events(events)] == ["running"]

    assert_run_pair(events, "success")
    assert [event["step_index"] for event in step_events(events)] == [1, 1, 2, 2]
    assert_idle(executor)


def test_public_and_standalone_calls_get_fresh_invocations(executor):
    events, identities = [], []
    executor.on_event = events.append
    for invoke in (
        lambda: executor.run({"code": "print('result')"}),
        lambda: executor.execute_in_workspace("print('result')", "unused-workspace"),
        lambda: executor._execute_code("print('result')", "full"),
        lambda: executor._execute_code("print('result')", "full"),
    ):
        events.clear()
        invoke()
        assert_run_pair(events, "success")
        identities.append(run_events(events)[0]["execution_id"])
        assert step_events(events)[0]["step_index"] == 1
        assert_idle(executor)
    assert len(set(identities)) == len(identities)


@pytest.mark.parametrize("preflight", ["syntax", "danger", "generated_syntax", "zero_calls"])
def test_preflight_with_no_execute_code_call_emits_no_execution_events(executor, preflight):
    events = []
    executor.on_event = events.append
    if preflight == "syntax":
        input_data = {"code": "print("}
    elif preflight == "danger":
        input_data = {"code": "import subprocess\nprint('blocked')"}
    elif preflight == "generated_syntax":
        executor._produce_code = Mock(return_value=(
            "print(", {"prose_dropped": 0, "code_dropped": 0}, False))
        input_data = {}
    else:
        executor._run = Mock(return_value={"success": False, "not_runnable": True})
        input_data = {}

    assert executor.run(input_data)["not_runnable"] is True
    assert events == []
    executor._execute_code_local.assert_not_called()
    executor._execute_code_docker.assert_not_called()
    assert_idle(executor)


@pytest.mark.parametrize("bookkeeping", ["aggregate", "ledger", "success_log", "call_count"])
def test_public_bookkeeping_failure_after_successful_steps_closes_run_as_error(executor, bookkeeping):
    events = []
    executor.on_event = events.append
    original = RuntimeError("public result bookkeeping failed")
    if bookkeeping == "aggregate":
        execute = executor._execute_with_repair

        def fail_aggregation(*args):
            execute(*args)
            raise original

        executor._execute_with_repair = fail_aggregation
    elif bookkeeping == "ledger":
        executor.log_experiment = Mock(side_effect=original)
    elif bookkeeping == "success_log":
        log = executor.log

        def fail_success_log(action, status, *args, **kwargs):
            if action == "execute_code" and status == "SUCCESS":
                raise original
            return log(action, status, *args, **kwargs)

        executor.log = fail_success_log
    else:
        executor._delta_llm_calls = Mock(side_effect=original)

    with pytest.raises(RuntimeError) as caught:
        executor.run({"code": "print('result')"})
    assert caught.value is original
    assert [event["status"] for event in step_events(events)] == [
        "running", "success", "running", "success"]
    assert_run_pair(events, "error")
    assert_idle(executor)


def test_terminal_waits_for_real_workspace_cleanup_and_public_accounting(executor, monkeypatch, tmp_path):
    events, timeline, workspaces, terminal_snapshots = [], [], [], []

    def workspace(**kwargs):
        path = tmp_path / f"temporary_execution_{len(workspaces)}"
        path.mkdir()
        workspaces.append(path)
        return str(path)

    def remove(path, **kwargs):
        shutil.rmtree(path, **kwargs)
        timeline.append("workspace_removed")

    @contextmanager
    def dependency_scope(**kwargs):
        try:
            yield
        finally:
            timeline.append("dependencies_released")

    def collect(*args):
        timeline.append("artifacts_collected")
        return {"artifacts": [], "artifact_warnings": []}

    def measure(*args):
        timeline.append("disk_measured")
        return {"status": "measured", "bytes": 0}

    def observe(event):
        events.append(event)
        if event["type"] == "execution_run" and event["status"] != "running":
            terminal_snapshots.append((list(timeline), [path.exists() for path in workspaces]))
            timeline.append("run_terminal")

    executor.on_event = observe
    executor._execute_code_local = CodeExecutorAgent._execute_code_local.__get__(executor)
    executor._ensure_local_deps = Mock(return_value=None)
    executor._run_local_script = Mock(side_effect=lambda *args: dict(SUCCESS))
    executor._collect_disk_usage = measure
    executor.dependency_scope = dependency_scope
    executor.log_experiment = Mock(side_effect=lambda *args, **kwargs: timeline.append("public_ledger"))
    executor.llm.get_call_count = Mock(side_effect=lambda: timeline.append("public_call_count") or 0)
    monkeypatch.setattr(executor_module, "tempfile", SimpleNamespace(mkdtemp=workspace))
    monkeypatch.setattr(executor_module, "shutil", SimpleNamespace(rmtree=remove))
    monkeypatch.setattr(executor_module, "collect_images", collect)

    result = executor.run({"code": "print('result')"})

    assert result["success"] is True
    assert_run_pair(events, "success")
    assert len(workspaces) == 2 and terminal_snapshots[0][1] == [False, False]
    before_terminal = terminal_snapshots[0][0]
    assert before_terminal == [
        "artifacts_collected", "disk_measured", "workspace_removed", "dependencies_released",
        "artifacts_collected", "disk_measured", "workspace_removed", "dependencies_released",
        "public_ledger", "public_call_count",
    ]
    assert timeline[-1] == "run_terminal"
    assert_idle(executor)


@pytest.mark.parametrize("outcome,status", [
    ({"success": True, "exit_code": 0}, "success"),
    ({"success": False, "exit_code": 1}, "error"),
    ({"success": True, "exit_code": 1}, "error"),
    ({"success": True, "exit_code": 0, "timed_out": True}, "error"),
    ({"success": False, "exit_code": 130, "cancelled": True}, "interrupted"),
    ({"success": False, "exit_code": 130, "interrupted": True}, "interrupted"),
])
def test_workspace_result_controls_run_terminal_status(executor, outcome, status):
    events = []
    executor.on_event = events.append
    result = {**outcome, "stdout": "Training completed successfully", "stderr": ""}
    executor._execute_code_local.return_value = result

    assert executor.execute_in_workspace("print('result')", "unused-workspace") is result
    assert_run_pair(events, status)
    assert_idle(executor)


@pytest.mark.parametrize("runner_error", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("observer_error", [KeyboardInterrupt, SystemExit])
def test_terminal_observer_cannot_replace_original_runner_failure(executor, runner_error, observer_error):
    events = []
    original = runner_error("original runner failure")
    executor._execute_code_local.side_effect = original

    def observe(event):
        events.append(event)
        if event.get("status") in {"error", "interrupted"}:
            raise observer_error("terminal observer failed")

    executor.on_event = observe
    with pytest.raises(runner_error) as caught:
        executor.execute_in_workspace("print('result')", "unused-workspace")
    assert caught.value is original
    assert_run_pair(events, "error" if runner_error is RuntimeError else "interrupted")
    assert_idle(executor)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("event_type,status,runner_calls,terminal", [
    ("execution_run", "running", 0, "interrupted"),
    ("execution_step", "running", 0, "interrupted"),
    ("execution_output", None, 1, "interrupted"),
    ("execution_step", "success", 1, "interrupted"),
    ("execution_run", "success", 2, "success"),
])
def test_observer_cancellation_propagates_original_object(executor, interruption, event_type,
                                                         status, runner_calls, terminal):
    events = []
    original = interruption("original observer cancellation")

    def observe(event):
        events.append(event)
        if event["type"] == event_type and (status is None or event.get("status") == status):
            raise original

    executor.on_event = observe
    with pytest.raises(interruption) as caught:
        executor.run({"code": "print('result')"})
    assert caught.value is original
    assert executor._execute_code_local.call_count == runner_calls
    assert_run_pair(events, terminal)
    assert_idle(executor)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("terminal_error", [KeyboardInterrupt, SystemExit])
def test_terminal_observer_cannot_replace_prior_observer_cancellation(executor, interruption, terminal_error):
    events = []
    original = interruption("original output observer cancellation")

    def observe(event):
        events.append(event)
        if event["type"] == "execution_output":
            raise original
        if event["type"] == "execution_run" and event["status"] == "interrupted":
            raise terminal_error("terminal notification cancellation")

    executor.on_event = observe
    with pytest.raises(interruption) as caught:
        executor.run({"code": "print('result')"})
    assert caught.value is original
    assert_run_pair(events, "interrupted")
    assert_idle(executor)


def test_ordinary_observer_errors_leave_run_result_unchanged(executor):
    events = []

    def observe(event):
        events.append(event)
        raise RuntimeError("ordinary observer failure")

    executor.on_event = observe
    assert executor.run({"code": "print('result')"})["success"] is True
    assert_run_pair(events, "success")
    assert_idle(executor)
