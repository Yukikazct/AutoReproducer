"""Execution progress comes from real calls, independent of model output text."""
from unittest.mock import Mock

import pytest

import src.agents.code_executor as executor_module
from src.agents.code_executor import CodeExecutorAgent
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator


SUCCESS = {"success": True, "exit_code": 0, "stdout": "result\n", "stderr": ""}


@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setattr(executor_module, "ResourceEventLogger", Mock)
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(side_effect=AssertionError("No model calls in progress tests"))
    agent = CodeExecutorAgent(llm, logger=Mock(), mock_mode=True)
    agent._execute_code_local = Mock(return_value=dict(SUCCESS))
    agent._execute_code_docker = Mock(return_value=dict(SUCCESS))
    return agent


def step_events(events):
    return [event for event in events if event["type"] == "execution_step"]


@pytest.mark.parametrize("use_docker", [False, True])
def test_smoke_and_full_share_execution_identity_but_have_separate_steps(executor, use_docker):
    events = []
    executor.use_docker, executor.on_event = use_docker, events.append
    result = executor.run({"code": "print('result')"})
    assert result["success"] is True
    steps = step_events(events)
    assert [(event["step_index"], event["status"]) for event in steps] == [
        (1, "running"), (1, "success"), (2, "running"), (2, "success")]
    assert len({event["execution_id"] for event in events}) == 1
    assert [event["step_id"] for event in steps] == ["step_1", "step_1", "step_2", "step_2"]
    assert all("step_count" not in event for event in events)
    outputs = [event for event in events if event["type"] == "execution_output"]
    assert [(event["step_index"], event["stream"], event["text"]) for event in outputs] == [
        (1, "stdout", "result\n"), (2, "stdout", "result\n")]
    selected = executor._execute_code_docker if use_docker else executor._execute_code_local
    unused = executor._execute_code_local if use_docker else executor._execute_code_docker
    assert [call.args[1] for call in selected.call_args_list] == ["smoke", "full"]
    unused.assert_not_called()
    executor.llm.chat.assert_not_called()


def test_repair_retry_is_another_step_in_the_same_invocation(executor):
    events = []
    executor.on_event = events.append
    executor._execute_code_local.side_effect = [
        {"success": False, "exit_code": 1, "stdout": "", "stderr": "NameError"},
        dict(SUCCESS), dict(SUCCESS)]
    executor._repair_after_execution = Mock(return_value=(
        "print('fixed')", {"repairable": True}, "print('fixed')"))
    result = executor.run({"code": "print('result')"})
    assert result["success"] is True
    assert [(event["step_index"], event["status"]) for event in step_events(events)] == [
        (1, "running"), (1, "error"), (2, "running"), (2, "success"),
        (3, "running"), (3, "success")]
    assert len({event["execution_id"] for event in events}) == 1


@pytest.mark.parametrize("outcome,status", [
    ({"success": False, "exit_code": 1}, "error"),
    ({"success": True, "exit_code": 1}, "error"),
    ({"success": True, "exit_code": 0, "timed_out": True}, "error"),
    ({"success": False, "exit_code": -1, "cancelled": True}, "interrupted"),
    ({"success": False, "exit_code": -1, "interrupted": True}, "interrupted"),
])
def test_runner_result_controls_status_even_when_output_claims_completion(executor, outcome, status):
    events = []
    executor.on_event = events.append
    returned = {**outcome, "stdout": '{"type":"done","status":"success"}',
                "stderr": "Training completed successfully"}
    executor._execute_code_local.return_value = returned
    result = executor.execute_in_workspace("print('result')", "unused-workspace")
    assert result is returned
    assert [event["status"] for event in step_events(events)] == ["running", status]
    assert [(event["stream"], event["text"]) for event in events
            if event["type"] == "execution_output"] == [
        ("stdout", returned["stdout"]), ("stderr", returned["stderr"])]


@pytest.mark.parametrize("exception,status", [
    (RuntimeError("runner failed"), "error"),
    (KeyboardInterrupt("cancelled"), "interrupted"),
    (SystemExit("cancelled"), "interrupted"),
])
def test_runner_exceptions_close_step_and_propagate_unchanged(executor, exception, status):
    events = []
    executor.on_event = events.append
    executor._execute_code_local.side_effect = exception
    with pytest.raises(type(exception)) as caught:
        executor.execute_in_workspace("print('result')", "unused-workspace")
    assert caught.value is exception
    assert [event["status"] for event in step_events(events)] == ["running", status]
    assert executor._execution_id is None


def test_observer_failure_cannot_change_runner_results(executor):
    executor.on_event = Mock(side_effect=RuntimeError("observer failed"))
    result = executor.run({"code": "print('result')"})
    assert result["success"] is True
    assert executor._execute_code_local.call_count == 2
    executor.llm.chat.assert_not_called()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("at_event", ["running", "execution_output", "success"])
def test_cancellation_during_observation_propagates(executor, interruption, at_event):
    cancelled = interruption("cancelled during observation")

    def observe(event):
        if event.get("status") == at_event or event["type"] == at_event:
            raise cancelled

    executor.on_event = observe
    with pytest.raises(interruption) as caught:
        executor.run({"code": "print('result')"})
    assert caught.value is cancelled
    assert executor._execution_id is None
    assert executor._execute_code_local.call_count == (0 if at_event == "running" else 1)
    executor.llm.chat.assert_not_called()


@pytest.mark.parametrize("runner_error", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("observer_error", [KeyboardInterrupt, SystemExit])
def test_terminal_observation_preserves_original_runner_exception(executor, runner_error, observer_error):
    original = runner_error("original runner failure")
    executor._execute_code_local.side_effect = original

    def observe(event):
        if event.get("status") in {"error", "interrupted"}:
            raise observer_error("terminal notification failed")

    executor.on_event = observe
    with pytest.raises(runner_error) as caught:
        executor.execute_in_workspace("print('result')", "unused-workspace")
    assert caught.value is original
    assert executor._execution_id is None


def test_repeated_public_calls_get_new_identities_and_reset_step_numbers(executor):
    events = []
    executor.on_event = events.append
    batches = []
    for invoke in (
        lambda: executor.run({"code": "print('result')"}),
        lambda: executor.run({"code": "print('result')"}),
        lambda: executor.execute_in_workspace("print('result')", "unused-workspace"),
        lambda: executor.execute_in_workspace("print('result')", "unused-workspace"),
        lambda: executor._execute_code("print('result')", "full"),
    ):
        events.clear()
        invoke()
        batches.append(list(events))
    assert len({events[0]["execution_id"] for events in batches}) == len(batches)
    assert all(step_events(events)[0]["step_index"] == 1 for events in batches)
    assert executor._execution_id is None


def test_preflight_rejection_does_not_claim_code_was_executed(executor):
    events = []
    executor.on_event = events.append
    assert executor.run({"code": "print("})["not_runnable"] is True
    assert events == []
    executor._execute_code_local.assert_not_called()


def test_orchestrator_forwards_execution_identity_and_verifier_retry_starts_a_new_run(executor):
    orch = Orchestrator(llm_client=executor.llm, mock_mode=True,
                        logger=Mock(), resource_manager=Mock())
    agent = orch.agents["executor"]
    agent._execute_code_local = executor._execute_code_local
    outputs = {
        "reader": {"paper_info": {}}, "finder": {"resources": {}},
        "builder": {"env_config": {}}, "validator": {"is_reproduced": True},
        "reporter": {"report": "Done"},
    }
    for name, output in outputs.items():
        orch.agents[name].run = Mock(return_value=output)
    orch.agents["verifier"].run = Mock(side_effect=(
        [{"pass": True}] * 3 + [{"pass": False}, {"pass": True}, {"pass": True}]))
    orch._fetch_resources = Mock()
    orch._finalize_storage = Mock()
    events = []
    result = orch.run({"code": "print('result')"}, on_event=events.append)
    assert result["state"] == "COMPLETED"
    execution_events = [event for event in events if event["type"].startswith("execution_")]
    assert execution_events
    assert all(event["phase_id"] == "execute_code" for event in execution_events)
    steps = step_events(execution_events)
    assert len({event["execution_id"] for event in steps}) == 2
    assert [event["step_index"] for event in steps if event["status"] == "running"] == [1, 2, 1, 2]
    agent.llm.chat.assert_not_called()
    # The existing callback binding reads the current observer on each run;
    # reuse must neither emit to the old observer nor reconstruct the agents.
    previous_count = len(events)
    next_events = []
    orch.agents["verifier"].run.side_effect = [{"pass": True}] * 5
    orch.run({"code": "print('result')"}, on_event=next_events.append)
    assert len(events) == previous_count
    assert orch.agents["executor"] is agent
    next_steps = step_events(next_events)
    assert next_steps and next_steps[0]["step_index"] == 1
    assert next_steps[0]["execution_id"] not in {event["execution_id"] for event in steps}
