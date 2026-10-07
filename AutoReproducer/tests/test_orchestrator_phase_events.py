"""Generic progress tracks generation, review, retries, and numeric gates separately."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from frontend.backend_pipeline import ProgressStore
from src.audit.audit_logger import AuditLogger
from src.llm.llm_client import LLMClient
from src.orchestrator import Orchestrator
from src.resource_manager import ResourceManager


MAIN_PHASES = ["generate_reader", "find_resources", "build_environment",
               "execute_code", "validate_result"]
REVIEW_PHASES = ["verify_reader", "verify_finder", "verify_builder",
                 "verify_executor", "verify_validator"]
ACCEPTED = {"pass": True, "issues": [], "fix_suggestions": []}
REJECTED = {"pass": False, "issues": ["Source evidence is incomplete"],
            "fix_suggestions": ["Add the source evidence"]}


@pytest.fixture
def pipeline(tmp_path):
    llm = LLMClient(mock_mode=True)
    llm.chat = Mock(side_effect=AssertionError("Progress tests must not call a model"))
    orch = Orchestrator(
        llm_client=llm, mock_mode=True,
        logger=AuditLogger(log_dir=str(tmp_path / "logs")),
        resource_manager=ResourceManager(data_root=str(tmp_path / "data")),
    )
    outputs = {
        "reader": {"paper_info": {"title": "Public paper", "method": "DLinear"}},
        "finder": {"resources": {"code": "https://github.com/example/repo"}},
        "builder": {"env_config": {"python_version": "3.11"}},
        "executor": {"executed": True, "final": {"success": True, "exit_code": 0}},
        "validator": {"is_reproduced": True, "status": "reproduced"},
        "reporter": {"report": "Independent phase results"},
    }
    for name, output in outputs.items():
        orch.agents[name].run = Mock(return_value=deepcopy(output))
    orch.agents["verifier"].run = Mock(return_value=deepcopy(ACCEPTED))
    orch.agents["optimizer"].run = Mock(side_effect=AssertionError("Reserved optimization must not run"))
    orch._fetch_resources = Mock()
    orch._finalize_storage = Mock()
    return orch


def run_with_snapshot(orch, tmp_path):
    events = []
    store = ProgressStore(str(tmp_path / "progress.jsonl"))

    def emit(event):
        events.append(event)
        store.emit(event)

    result = orch.run({"paper_title": "Public paper"}, on_event=emit)
    store.emit({"type": "done", "result": result})
    snapshot = ProgressStore.read_snapshot(str(store.path))
    rows = {row["id"]: row for row in snapshot["pipeline_stages"]}
    orch.llm.chat.assert_not_called()
    orch.agents["optimizer"].run.assert_not_called()
    return result, events, snapshot, rows


def phase_events(events, phase):
    return [event for event in events if event.get("phase_id") == phase]


def test_each_main_phase_finishes_before_its_independent_review(pipeline, tmp_path):
    result, events, snapshot, rows = run_with_snapshot(pipeline, tmp_path)
    plan = events[0]
    expected_ids = [phase for pair in zip(MAIN_PHASES, REVIEW_PHASES) for phase in pair]
    expected_ids += ["reserve_optimization", "generate_report"]
    assert plan["type"] == "pipeline_plan"
    assert plan["pipeline"] == "generic"
    assert [stage["id"] for stage in plan["stages"]] == expected_ids
    assert list(rows) == expected_ids
    assert len(rows) == 12
    assert result["state"] == "COMPLETED"
    assert snapshot["done"] and not snapshot["running"]
    assert pipeline.agents["verifier"].run.call_count == 5
    for main, review in zip(MAIN_PHASES, REVIEW_PHASES):
        main_events = phase_events(events, main)
        review_events = phase_events(events, review)
        assert [event["status"] for event in main_events] == ["running", "success"]
        assert [event["status"] for event in review_events] == ["running", "success"]
        assert events.index(main_events[-1]) < events.index(review_events[0])
        assert rows[main]["status"] == rows[review]["status"] == "success"
    assert rows["reserve_optimization"]["status"] == "skipped"
    assert rows["generate_report"]["status"] == "success"


def test_rejected_review_then_correction_has_two_attempts_without_stale_errors(pipeline, tmp_path):
    pipeline.agents["verifier"].run.side_effect = [REJECTED] + [ACCEPTED] * 5
    result, events, _, rows = run_with_snapshot(pipeline, tmp_path)
    assert result["state"] == "COMPLETED"
    assert pipeline.agents["reader"].run.call_count == 2
    assert [(event["attempt"], event["status"]) for event in phase_events(events, "generate_reader")] == [
        (1, "running"), (1, "success"), (2, "running"), (2, "success")]
    reviews = phase_events(events, "verify_reader")
    assert [(event["attempt"], event["status"]) for event in reviews] == [
        (1, "running"), (1, "error"), (2, "running"), (2, "success")]
    assert reviews[1]["reason"] == "Source evidence is incomplete"
    assert reviews[1]["outcome"] == "rejected"
    assert rows["verify_reader"]["attempt"] == 2
    assert rows["verify_reader"]["status"] == "success"
    assert "reason" not in rows["verify_reader"]
    assert rows["verify_reader"]["outcome"] == "accepted"
    assert len(result["data"]["fix_records"]) == 1


def test_final_quality_rejection_remains_error_with_existing_flow_preserved(pipeline, tmp_path):
    pipeline.agents["verifier"].run.side_effect = [REJECTED, REJECTED] + [ACCEPTED] * 4
    result, events, _, rows = run_with_snapshot(pipeline, tmp_path)
    assert result["state"] == "COMPLETED"
    assert pipeline.agents["reader"].run.call_count == 2
    assert [event["status"] for event in phase_events(events, "verify_reader")] == [
        "running", "error", "running", "error"]
    assert rows["verify_reader"]["status"] == "error"
    assert rows["verify_reader"]["attempt"] == 2
    assert rows["verify_reader"]["outcome"] == "rejected"
    assert rows["generate_reader"]["status"] == "success"
    assert rows["generate_report"]["status"] == "success"


@pytest.mark.parametrize("failure_attempt", [1, 2])
def test_verifier_exception_preserves_completed_main_and_blocks_pending_phases(pipeline, tmp_path, failure_attempt):
    failure = RuntimeError("Review transport failed")
    pipeline.agents["verifier"].run.side_effect = [failure] if failure_attempt == 1 else [REJECTED, failure]
    result, events, _, rows = run_with_snapshot(pipeline, tmp_path)
    assert result["state"] == "ERROR"
    assert result["error"] == "Review transport failed"
    assert rows["generate_reader"]["status"] == "success"
    assert rows["generate_reader"]["attempt"] == failure_attempt
    assert rows["verify_reader"]["status"] == "error"
    assert rows["verify_reader"]["attempt"] == failure_attempt
    assert rows["verify_reader"]["outcome"] == "exception"
    assert rows["verify_reader"]["reason"] == "Review transport failed"
    assert rows["find_resources"]["status"] == "blocked"
    assert rows["verify_finder"]["status"] == "blocked"
    assert rows["generate_report"]["status"] == "blocked"
    assert rows["reserve_optimization"]["status"] == "skipped"
    assert not any(event["status"] == "error" for event in phase_events(events, "generate_reader"))
    pipeline.agents["finder"].run.assert_not_called()


def test_main_correction_exception_is_attributed_to_second_main_attempt(pipeline, tmp_path):
    reader_output = pipeline.agents["reader"].run.return_value
    pipeline.agents["reader"].run.side_effect = [reader_output, RuntimeError("Correction failed")]
    pipeline.agents["verifier"].run.return_value = REJECTED
    result, events, _, rows = run_with_snapshot(pipeline, tmp_path)
    assert result["state"] == "ERROR"
    assert rows["generate_reader"]["status"] == "error"
    assert rows["generate_reader"]["attempt"] == 2
    assert rows["generate_reader"]["reason"] == "Correction failed"
    assert rows["generate_reader"]["outcome"] == "exception"
    assert rows["verify_reader"]["status"] == "error"
    assert rows["verify_reader"]["attempt"] == 1
    assert rows["verify_reader"]["outcome"] == "rejected"
    assert len(phase_events(events, "generate_reader")) == 4
    assert rows["find_resources"]["status"] == "blocked"


@pytest.mark.parametrize("executor_output", [
    {"executed": True, "final": {"success": False, "exit_code": 1, "stderr": "Training failed"}},
    {"not_runnable": True, "reason": "No runnable code"},
])
def test_execution_failure_is_independent_of_quality_review(pipeline, tmp_path, executor_output):
    pipeline.agents["executor"].run.return_value = executor_output
    pipeline.agents["validator"].run.return_value = {
        "is_reproduced": False, "status": "execution_failed", "reason": "Execution evidence failed"}
    result, _, _, rows = run_with_snapshot(pipeline, tmp_path)
    assert result["data"]["execution"] == executor_output
    assert result["data"]["validation"]["is_reproduced"] is False
    assert rows["execute_code"]["status"] == "error"
    assert rows["execute_code"]["outcome"] == "execution_failed"
    assert rows["verify_executor"]["status"] == "success"
    assert rows["validate_result"]["status"] == "error"


@pytest.mark.parametrize("outcome,reproduced,level", [
    ("not_reproduced", False, "experiment_completed"),
    ("inconclusive", None, "inconclusive"),
    ("smoke_passed", None, "smoke_passed"),
])
def test_completed_validation_preserves_its_conclusion_independently_of_quality_review(
        pipeline, tmp_path, outcome, reproduced, level):
    validation = {"is_reproduced": reproduced, "status": outcome, "result_level": level,
                  "reason": "Numerical validation conclusion"}
    pipeline.agents["validator"].run.return_value = validation
    result, events, _, rows = run_with_snapshot(pipeline, tmp_path)
    assert result["data"]["validation"] == validation
    assert [event["status"] for event in phase_events(events, "validate_result")] == ["running", "success"]
    assert rows["validate_result"]["status"] == "success"
    assert rows["validate_result"]["outcome"] == outcome
    assert rows["validate_result"]["reason"] == validation["reason"]
    assert rows["verify_validator"]["status"] == "success"
    assert result["data"]["optimization"]["optimized"] is False


def test_invalid_metric_evidence_remains_a_failed_validation_stage(pipeline, tmp_path):
    validation = {"is_reproduced": False, "status": "not_reproduced", "result_level": "failed",
                  "reason": "Final evaluation metric is missing"}
    pipeline.agents["validator"].run.return_value = validation
    result, _, _, rows = run_with_snapshot(pipeline, tmp_path)
    assert result["data"]["validation"] == validation
    assert rows["validate_result"]["status"] == "error"
    assert rows["validate_result"]["outcome"] == "not_reproduced"
    assert rows["verify_validator"]["status"] == "success"


@pytest.mark.parametrize("bad_review", [None, {"pass": True, "llm_calls": "invalid"}])
def test_review_output_processing_exception_emits_immediate_review_error(pipeline, tmp_path, bad_review):
    pipeline.agents["verifier"].run.return_value = bad_review
    result, events, _, rows = run_with_snapshot(pipeline, tmp_path)
    assert result["state"] == "ERROR"
    assert [event["status"] for event in phase_events(events, "verify_reader")] == ["running", "error"]
    assert rows["verify_reader"]["outcome"] == "exception"
    assert rows["verify_reader"]["reason"] == result["error"]
    assert rows["generate_reader"]["status"] == "success"


@pytest.mark.parametrize("bad_correction", [None, {"paper_info": {}, "llm_calls": "invalid"}])
def test_correction_output_processing_exception_closes_the_main_retry(pipeline, tmp_path, bad_correction):
    reader_output = pipeline.agents["reader"].run.return_value
    pipeline.agents["reader"].run.side_effect = [reader_output, bad_correction]
    pipeline.agents["verifier"].run.return_value = REJECTED
    result, events, _, rows = run_with_snapshot(pipeline, tmp_path)
    assert result["state"] == "ERROR"
    assert [event["status"] for event in phase_events(events, "generate_reader")] == [
        "running", "success", "running", "error"]
    assert rows["generate_reader"]["attempt"] == 2
    assert rows["generate_reader"]["outcome"] == "exception"
    assert rows["generate_reader"]["reason"] == result["error"]
    assert rows["verify_reader"]["status"] == "error"
    assert rows["verify_reader"]["outcome"] == "rejected"
