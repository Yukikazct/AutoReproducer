"""Verifier retries carry the actual failure without leaking across stages."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.orchestrator as orchestration
from src.orchestrator import Orchestrator


def make_orchestrator(reviews):
    orch = Orchestrator.__new__(Orchestrator)
    events = []
    orch.on_event = events.append
    orch.logger = Mock()
    orch.data = {"paper_info": {"title": "ReZero"}, "total_llm_calls": 0}
    orch.agents = {"verifier": SimpleNamespace(run=Mock(side_effect=reviews))}
    return orch, events


def rejection(issue="Code did not execute", suggestion="Check the failing call"):
    return {"pass": False, "issues": [issue], "fix_suggestions": [suggestion],
            "llm_calls": 1}


def test_retry_receives_actual_failure_and_review_without_previous_code():
    review = rejection()
    orch, events = make_orchestrator([review, {"pass": True, "llm_calls": 1}])
    previous = {"success": False, "reason": "eval precheck rejected model.eval()",
                "code": "PREVIOUS CODE MUST NOT ENTER FEEDBACK",
                "final": {"stderr": "less useful error", "exit_code": -5}}
    fixed = {"success": True, "final": {"success": True, "exit_code": 0},
             "llm_calls": 2}
    agent = SimpleNamespace(name="CodeExecutor", run=Mock(return_value=fixed))
    original_input = orch.data
    original_paper_info = orch.data["paper_info"]

    orch._verify_step("EXECUTE_CODE", agent, previous)

    received = agent.run.call_args.args[0]
    assert received is not original_input
    assert received["paper_info"] is original_paper_info
    assert received["retry_feedback"] == {
        "agent": "CodeExecutor", "state": "EXECUTE_CODE", "attempt": 2,
        "issues": ["Code did not execute"],
        "fix_suggestions": ["Check the failing call"],
        "previous_failure": "eval precheck rejected model.eval()",
    }
    assert received["retry_feedback"]["issues"] is not review["issues"]
    assert received["retry_feedback"]["fix_suggestions"] is not review["fix_suggestions"]
    assert "PREVIOUS CODE" not in str(received["retry_feedback"])
    assert "retry_feedback" not in original_input
    assert orch.data["execution"] is fixed
    assert orch.data["total_llm_calls"] == 4
    assert [(e["agent"], e["status"], e["attempt"]) for e in events] == [
        ("Verifier", "running", 1), ("Verifier", "error", 1),
        ("CodeExecutor", "running", 2), ("CodeExecutor", "success", 2),
        ("Verifier", "running", 2), ("Verifier", "success", 2),
    ]


@pytest.mark.parametrize("previous,expected", [
    ({"final": {"stderr": "dataset unavailable"}}, "dataset unavailable"),
    ({"reason": "primary error", "final": {"stderr": "secondary error"}},
     "primary error"),
    ({"reason": "x" * 3000}, "x" * 2000),
    ({}, ""),
])
def test_feedback_uses_only_bounded_reason_or_final_stderr(previous, expected):
    orch, _ = make_orchestrator([rejection(), {"pass": True}])
    agent = SimpleNamespace(name="CodeExecutor", run=Mock(return_value={"success": True}))

    orch._verify_step("EXECUTE_CODE", agent, previous)

    assert agent.run.call_args.args[0]["retry_feedback"]["previous_failure"] == expected


def test_feedback_is_specific_to_stage_and_does_not_pollute_shared_input():
    orch, _ = make_orchestrator([
        rejection("execution error", "execution repair"), {"pass": True},
        rejection("environment error", "environment repair"), {"pass": True},
    ])
    executor = SimpleNamespace(name="CodeExecutor", run=Mock(return_value={"success": True}))
    builder = SimpleNamespace(name="EnvBuilder", run=Mock(return_value={"env_config": {}}))

    orch._verify_step("EXECUTE_CODE", executor, {"reason": "execution rejected"})
    assert "retry_feedback" not in orch.data
    orch._verify_step("BUILD_ENV", builder, {"reason": "environment incomplete"})

    execution_feedback = executor.run.call_args.args[0]["retry_feedback"]
    environment_feedback = builder.run.call_args.args[0]["retry_feedback"]
    assert execution_feedback["state"] == "EXECUTE_CODE"
    assert environment_feedback == {
        "agent": "EnvBuilder", "state": "BUILD_ENV", "attempt": 2,
        "issues": ["environment error"], "fix_suggestions": ["environment repair"],
        "previous_failure": "environment incomplete",
    }
    assert "retry_feedback" not in orch.data


@pytest.mark.parametrize("retry_limit", [0, 1, 2])
def test_retry_budget_still_bounds_calls_and_uses_latest_failed_output(monkeypatch, retry_limit):
    monkeypatch.setattr(orchestration, "MAX_FIX_RETRIES", retry_limit)
    reviews = [rejection(f"issue {i}", f"repair {i}") for i in range(retry_limit + 1)]
    orch, events = make_orchestrator(reviews)
    outputs = [{"success": False, "reason": f"failure {i + 1}", "llm_calls": 2}
               for i in range(retry_limit)]
    agent = SimpleNamespace(name="CodeExecutor", run=Mock(side_effect=outputs))

    orch._verify_step("EXECUTE_CODE", agent, {"reason": "failure 0"})

    assert agent.run.call_count == retry_limit
    assert orch.agents["verifier"].run.call_count == retry_limit + 1
    assert orch.data["total_llm_calls"] == (retry_limit + 1) + retry_limit * 2
    assert "retry_feedback" not in orch.data
    for i, call in enumerate(agent.run.call_args_list):
        feedback = call.args[0]["retry_feedback"]
        assert feedback["attempt"] == i + 2
        assert feedback["previous_failure"] == f"failure {i}"
        assert feedback["issues"] == [f"issue {i}"]
        assert feedback["fix_suggestions"] == [f"repair {i}"]
    assert all(e["attempt"] <= retry_limit + 1 for e in events)


def test_accepted_output_does_not_retry_or_create_feedback():
    orch, _ = make_orchestrator([{"pass": True}])
    agent = SimpleNamespace(name="CodeExecutor", run=Mock())

    orch._verify_step("EXECUTE_CODE", agent, {"success": True})

    agent.run.assert_not_called()
    assert "retry_feedback" not in orch.data


def test_string_review_feedback_stays_a_complete_message():
    orch, _ = make_orchestrator([
        {"pass": False, "issues": "dataset unavailable",
         "fix_suggestions": "read the local dataset"},
        {"pass": True},
    ])
    agent = SimpleNamespace(name="CodeExecutor", run=Mock(return_value={"success": True}))

    orch._verify_step("EXECUTE_CODE", agent, {"reason": "local data missing"})

    feedback = agent.run.call_args.args[0]["retry_feedback"]
    assert feedback["issues"] == ["dataset unavailable"]
    assert feedback["fix_suggestions"] == ["read the local dataset"]
