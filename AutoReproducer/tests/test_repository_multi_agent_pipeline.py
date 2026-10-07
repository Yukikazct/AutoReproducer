"""Service-level analysis gates/accounting; no HTTP, packages, or training.

The service still uses its frozen commands, final-test metric parser,
deterministic verdict, audit logger, report generator and JSON persistence.
Public source loading and model analyses are injected at their public methods;
their client counter advances as actual attempted calls would.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import execution_artifacts, repository_reproduction as reproduction
from src import repository_validation
from src.audit.audit_logger import AuditLogger
from src.repository_analysis import RepositoryAnalysis, RepositoryAnalysisError, STAGES
from src.repository_profiles import get_profile
from src.repository_public_sources import RepositoryPublicSources


class CountedClient:
    """Existing client history must not count toward this service invocation."""

    mock_mode = False
    base_url = "https://public-analysis-provider.invalid/v1"
    model = "service-counter-fixture"

    def __init__(self, existing_calls=23):
        self.initial_calls = existing_calls
        self.call_count = existing_calls

    def get_call_count(self):
        return self.call_count

    def attempted_calls(self, count):
        self.call_count += count

    def chat(self, *args, **kwargs):
        raise AssertionError("These service tests must never open an API transport")


def accepted_analysis(packet, names, calls):
    return {"version": 1, "status": "accepted", "source": "real_api",
            "input_scope": "public_sources_only", "model": CountedClient.model,
            "calls": calls, "gate": {"pass": True}, "analyses": {},
            "sources": [{key: source[key] for key in ("source_id", "url", "locator")}
                        for source in packet["sources"]],
            "stages": [{"name": name, "attempted": True, "completed": True,
                        "accepted": True, "calls": 1, "usage": {}}
                       for name in names]}


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    profile = get_profile("dlinear_etth1_reference")
    client = CountedClient()
    logger = AuditLogger(log_dir=str(tmp_path / "logs"), ledger_dir=str(tmp_path / "ledger"))
    timeline = []
    packet = {"version": 1, "repository": deepcopy(profile["repository"]), "sources": [
        {"source_id": "paper_table2", "url": profile["paper"]["reference_source"],
         "locator": "#S5.T2", "text": "Public Table 2 fixture: DLinear ETTh1 96 MSE .375 MAE .399"},
    ]}

    def export(data_root, spec, destination, *, offline):
        assert offline is True
        destination.mkdir()
        return {**spec["repository"], "resolved_sha": spec["repository"]["revision"],
                "path": str(destination.resolve()), "files": {}}

    def dataset(data_root, spec, workspace, *, offline):
        assert offline is True
        return {**spec["dataset"], "path": str(Path(workspace) / spec["dataset"]["target"]),
                "verified": True, "preview": "LOCAL_DATA_CONTENTS_SENTINEL"}

    def public_sources(workspace, snapshot, spec, *, offline):
        assert offline is True
        assert spec == profile
        assert workspace == snapshot["path"]
        timeline.append("public_sources")
        return deepcopy(packet)

    monkeypatch.setattr(reproduction, "export_repository", export)
    monkeypatch.setattr(reproduction, "prepare_dataset", dataset)
    # This verifies orchestration independently of the test host's interpreter.
    monkeypatch.setattr(reproduction, "sys", SimpleNamespace(
        version_info=(3, 12), executable=sys.executable))
    source_mock = Mock(side_effect=public_sources)
    monkeypatch.setattr(RepositoryPublicSources, "build_packet", source_mock)

    private_log = "LOCAL_TRAINING_LOG_SENTINEL"
    private_path = str(tmp_path / "LOCAL_WORKSPACE_SENTINEL")
    final = {"id": "train_and_eval", "stage": "full", "success": True,
             "executed": True, "exit_code": 0, "timed_out": False,
             "stdout": (f"{private_log} {private_path}\n"
                        "Epoch: 7, Steps: 256 | Train Loss: .4 Vali Loss: .4 Test Loss: .389\n"
                        "Early stopping\n>>>>>>>testing : current_run<<<<<<<<\n"
                        "test 2785\nmse:0.389, mae:0.407\n"),
             "stderr": "LOCAL_STDERR_SENTINEL", "stdout_path": private_path + "/train.stdout.log"}
    completed = {"mode": "repository", "success": True, "executed": True,
                 "steps": [{"id": "import_check", "success": True, "exit_code": 0}, final],
                 "final": final, "artifacts": [], "llm_calls": 0,
                 "environment": {"dependencies_path": private_path + "/deps"}}

    def execute(workspace, steps, environment, *, use_docker, on_event):
        assert timeline[-1] == "analysis:verifier:accepted", timeline
        # Commands remain the production profile, with no generated replacement.
        assert steps == profile["steps"]
        assert environment["requirements_txt"] == profile["environment"]["requirements_txt"]
        assert use_docker is False
        timeline.append("runner")
        return deepcopy(completed)

    runner = Mock()
    runner.run.side_effect = execute
    protocol = Mock(return_value={"pass": True, "epochs_completed": 7,
                                  "reason": "LOCAL_PROTOCOL_DETAILS_SENTINEL"})
    monkeypatch.setattr(repository_validation, "verify_dlinear_protocol", protocol)
    recompute = Mock(return_value={"pass": True, "metrics": {"mse": .389, "mae": .407},
                                  "artifact_dir": private_path + "/artifacts",
                                  "source": private_path + "/results/pred.npy"})
    monkeypatch.setattr(reproduction, "recompute_dlinear_metrics", recompute)
    monkeypatch.setattr(execution_artifacts, "collect_images", Mock(return_value={"artifacts": []}))

    def analyze(analysis, actual_packet, actual_profile, on_stage=None):
        assert analysis.llm is client
        assert actual_packet == packet and actual_profile == profile
        assert timeline == ["public_sources"]
        for name in STAGES:
            on_stage(name, {"status": "started"})
            client.attempted_calls(1)
            timeline.append(f"analysis:{name}:accepted")
            on_stage(name, {"status": "accepted"})
        return accepted_analysis(packet, STAGES, calls=4)

    analysis_mock = Mock()

    def analysis_entry(analysis, *args, **kwargs):
        analysis_mock(*args, **kwargs)
        return analyze(analysis, *args, **kwargs)

    monkeypatch.setattr(RepositoryAnalysis, "run", analysis_entry)
    review_mock = Mock(side_effect=AssertionError("Result summary requires explicit authorization"))

    def review_entry(analysis, *args, **kwargs):
        return review_mock(*args, **kwargs)

    monkeypatch.setattr(RepositoryAnalysis, "review_result_summary", review_entry)
    service = reproduction.RepositoryReproduction(tmp_path / "data", logger, runner=runner, llm=client)
    return SimpleNamespace(service=service, profile=profile, client=client, logger=logger,
                           runner=runner, source_mock=source_mock, analysis_mock=analysis_mock,
                           review_mock=review_mock, protocol=protocol, recompute=recompute,
                           packet=packet, timeline=timeline, monkeypatch=monkeypatch,
                           private_log=private_log, private_path=private_path)


def invoke(pipeline, *, authorize_result_review, events=None, **overrides):
    return pipeline.service.run({"experiment_profile": "dlinear_etth1_reference",
                                 "offline": True, "use_llm_review": True,
                                 "analysis_mode": "multi_agent",
                                 "allow_result_summary_review": authorize_result_review,
                                 "local_notes": "LOCAL_USER_NOTES_SENTINEL", **overrides},
                                on_event=events.append if events is not None else None)


def assert_call_delta(pipeline, result, expected):
    assert pipeline.client.get_call_count() == pipeline.client.initial_calls + expected
    assert result["data"]["total_llm_calls"] == expected
    assert pipeline.logger.llm_calls == expected
    assert result["data"]["audit_stats"]["llm_calls"] == expected
    saved = json.loads((Path(result["data"]["run_dir"]) / "result.json").read_text())
    assert saved["data"]["total_llm_calls"] == expected


def test_preanalysis_gates_execution_and_authorized_result_review_shares_only_small_summary(pipeline):
    def review(summary, packet, profile, on_stage=None):
        assert pipeline.timeline[-1] == "runner"
        assert summary == {"metrics": {"mse": .389, "mae": .407}, "epochs_completed": 7,
                           "protocol_pass": True, "independent_metrics_pass": True}
        assert packet == pipeline.packet and profile == pipeline.profile
        rendered = json.dumps(summary)
        for private in (pipeline.private_log, pipeline.private_path, "LOCAL_STDERR_SENTINEL",
                        "LOCAL_DATA_CONTENTS_SENTINEL", "LOCAL_PROTOCOL_DETAILS_SENTINEL",
                        "LOCAL_USER_NOTES_SENTINEL", "stdout", "pred.npy"):
            assert private not in rendered
        on_stage("result_validator", {"status": "started"})
        pipeline.client.attempted_calls(1)
        on_stage("result_validator", {"status": "accepted"})
        return accepted_analysis(packet, ["result_validator"], calls=1)

    pipeline.review_mock.side_effect = review
    result = invoke(pipeline, authorize_result_review=True)
    assert result["state"] == "COMPLETED", result["error"]
    assert result["data"]["validation"]["status"] == "reproduced"
    assert result["data"]["analysis_status"] == "completed"
    pipeline.source_mock.assert_called_once()
    pipeline.analysis_mock.assert_called_once()
    pipeline.runner.run.assert_called_once()
    pipeline.protocol.assert_called_once()
    pipeline.recompute.assert_called_once()
    pipeline.review_mock.assert_called_once()
    assert_call_delta(pipeline, result, 5)
    run_dir = Path(result["data"]["run_dir"])
    assert json.loads((run_dir / "repository_analysis.json").read_text())["status"] == "accepted"
    assert json.loads((run_dir / "result_analysis.json").read_text())["status"] == "accepted"


def test_readiness_rejection_preserves_partial_analysis_and_never_executes(pipeline):
    partial = {"status": "failed", "calls": 999, "analyses": {"reader": {"accepted": True}},
               "stages": [{"name": "reader", "attempted": True, "completed": True,
                           "accepted": True, "calls": 1},
                          {"name": "finder", "attempted": True, "completed": True,
                           "accepted": False, "calls": 1}],
               "gate": {"pass": False, "reason": "Public source mapping rejected"}}

    def reject(analysis, packet, profile, on_stage=None):
        pipeline.client.attempted_calls(2)
        on_stage("finder", {"status": "started"})
        on_stage("finder", {"status": "rejected"})
        raise RepositoryAnalysisError("Public source mapping rejected", partial)

    pipeline.monkeypatch.setattr(RepositoryAnalysis, "run", reject)
    result = invoke(pipeline, authorize_result_review=True)
    assert result["state"] == "ERROR"
    assert "Public source mapping rejected" in result["error"]
    assert result["data"]["execution"]["not_runnable"] is True
    assert result["data"]["execution"]["executed"] is False
    assert result["data"]["validation"]["status"] == "analysis_failed"
    assert result["data"]["repository_analysis"] == partial
    pipeline.runner.run.assert_not_called()
    pipeline.protocol.assert_not_called()
    pipeline.recompute.assert_not_called()
    pipeline.review_mock.assert_not_called()
    assert_call_delta(pipeline, result, 2)  # Claimed stage metadata is not a counter.
    run_dir = Path(result["data"]["run_dir"])
    assert json.loads((run_dir / "repository_analysis.json").read_text()) == partial
    assert (run_dir / "report.md").is_file()


@pytest.mark.parametrize("attempted_result_calls", [0, 1])
def test_failed_result_explanation_preserves_deterministic_reproduction_and_counts_only_attempts(
        pipeline, attempted_result_calls):
    partial = {"status": "failed", "calls": 999, "analyses": {}, "stages": [],
               "gate": {"pass": False, "reason": "Result explanation rejected"}}

    def reject_summary(summary, packet, profile, on_stage=None):
        pipeline.client.attempted_calls(attempted_result_calls)
        raise RepositoryAnalysisError("Result explanation rejected", partial)

    pipeline.review_mock.side_effect = reject_summary
    result = invoke(pipeline, authorize_result_review=True)
    assert result["state"] == "COMPLETED", result["error"]
    assert result["error"] is None
    assert result["data"]["execution"]["executed"] is True
    verdict = result["data"]["validation"]
    assert verdict["status"] == "reproduced" and verdict["is_reproduced"] is True
    assert verdict["validation"]["verdict_source"] == "deterministic"
    assert result["data"]["analysis_status"] == "result_analysis_failed"
    assert result["data"]["analysis_error"] == "Result explanation rejected"
    assert result["data"]["result_analysis"] == partial
    pipeline.runner.run.assert_called_once()
    pipeline.review_mock.assert_called_once()
    assert_call_delta(pipeline, result, 4 + attempted_result_calls)
    run_dir = Path(result["data"]["run_dir"])
    assert json.loads((run_dir / "result_analysis.json").read_text()) == partial
    assert json.loads((run_dir / "result.json").read_text())["data"]["validation"]["status"] == "reproduced"


def test_without_result_authorization_only_four_public_readiness_calls_are_made(pipeline):
    result = invoke(pipeline, authorize_result_review=False)
    assert result["state"] == "COMPLETED", result["error"]
    assert result["data"]["validation"]["status"] == "reproduced"
    assert result["data"]["analysis_status"] == "public_readiness_accepted"
    assert "result_analysis" not in result["data"]
    pipeline.runner.run.assert_called_once()
    pipeline.review_mock.assert_not_called()
    assert_call_delta(pipeline, result, 4)
    assert not (Path(result["data"]["run_dir"]) / "result_analysis.json").exists()


def test_real_phase_order_separates_readiness_from_local_verification(pipeline):
    events = []
    result = invoke(pipeline, authorize_result_review=False, events=events)
    assert result["state"] == "COMPLETED", result["error"]
    plan = next(event["stages"] for event in events if event["type"] == "pipeline_plan")
    assert [stage["id"] for stage in plan] == [
        "select_experiment", "prepare_repository", "prepare_environment_plan", "load_public_sources",
        "analyze_reader", "analyze_finder", "analyze_builder", "review_readiness", "execute_repository",
        "verify_protocol", "validate_metrics", "review_result_summary", "generate_report"]
    assert next(stage for stage in plan if stage["id"] == "review_result_summary")["status"] == "skipped"
    states = [event for event in events if event["type"] == "state" and event.get("phase_id")]
    transitions = [(event["phase_id"], event["status"]) for event in states]
    assert transitions.index(("review_readiness", "success")) < transitions.index(("execute_repository", "running"))
    assert transitions.index(("execute_repository", "success")) < transitions.index(("verify_protocol", "running"))
    assert transitions.index(("verify_protocol", "success")) < transitions.index(("validate_metrics", "running"))
    assert next(event for event in states if event["phase_id"] == "verify_protocol" and event["status"] == "success")["outcome"] == "pass"
    assert next(event for event in states if event["phase_id"] == "validate_metrics" and event["status"] == "success")["outcome"] == "reproduced"
    assert_call_delta(pipeline, result, 4)


def test_bounded_analysis_correction_keeps_phase_id_and_attempt_metadata(pipeline):
    def correct(analysis, packet, profile, on_stage):
        on_stage("reader", {"status": "started", "attempt": 1, "calls": 0})
        pipeline.client.attempted_calls(1)
        on_stage("reader", {"status": "failed", "attempt": 1, "calls": 1, "reason": "Citation mismatch"})
        on_stage("reader", {"status": "started", "attempt": 2, "calls": 0})
        pipeline.client.attempted_calls(1)
        on_stage("reader", {"status": "accepted", "attempt": 2, "calls": 1})
        for name in STAGES[1:]:
            on_stage(name, {"status": "started", "attempt": 1, "calls": 0})
            pipeline.client.attempted_calls(1)
            pipeline.timeline.append(f"analysis:{name}:accepted")
            on_stage(name, {"status": "accepted", "attempt": 1, "calls": 1})
        return accepted_analysis(packet, STAGES, calls=5)

    pipeline.monkeypatch.setattr(RepositoryAnalysis, "run", correct)
    events = []
    result = invoke(pipeline, authorize_result_review=False, events=events)
    assert result["state"] == "COMPLETED", result["error"]
    reader = [event for event in events if event.get("phase_id") == "analyze_reader"]
    assert [(event["attempt"], event["status"]) for event in reader] == [
        (1, "running"), (1, "error"), (2, "running"), (2, "success")]
    assert reader[1]["reason"] == "Citation mismatch" and reader[-1]["calls"] == 1
    assert_call_delta(pipeline, result, 5)


def test_final_verification_exception_marks_its_own_phase_and_validate_state(pipeline):
    pipeline.protocol.side_effect = RuntimeError("local protocol checker failed")
    events = []
    result = invoke(pipeline, authorize_result_review=False, events=events)
    assert result["state"] == "ERROR"
    assert result["error"].startswith("VALIDATE 阶段失败:")
    verification = [event for event in events if event.get("phase_id") == "verify_protocol"]
    assert verification[0]["status"] == "running"
    assert verification[-1]["status"] == "error"
    assert verification[-1]["reason"] == "local protocol checker failed"
    assert any(event.get("phase_id") == "execute_repository" and event["status"] == "success" for event in events)
    pipeline.recompute.assert_not_called()
    assert (Path(result["data"]["run_dir"]) / "report.md").is_file()


def test_completed_validation_keeps_inconclusive_outcome_distinct_from_success(pipeline):
    pipeline.protocol.return_value = {"pass": False, "reason": "Full protocol was not verified"}
    events = []
    result = invoke(pipeline, authorize_result_review=False, events=events)
    assert result["data"]["validation"]["status"] == "inconclusive"
    assert result["data"]["validation"]["is_reproduced"] is None
    local_check = [event for event in events if event.get("phase_id") == "verify_protocol"][-1]
    metric_validation = [event for event in events if event.get("phase_id") == "validate_metrics"][-1]
    assert local_check["status"] == "error" and local_check["outcome"] == "fail"
    assert metric_validation["status"] == "success" and metric_validation["outcome"] == "inconclusive"


def test_result_explanation_failure_does_not_overwrite_metric_phase(pipeline):
    partial = {"status": "failed", "stages": [], "calls": 0,
               "gate": {"pass": False, "reason": "summary rejected"}}
    pipeline.review_mock.side_effect = RepositoryAnalysisError("summary rejected", partial)
    events = []
    result = invoke(pipeline, authorize_result_review=True, events=events)
    assert result["state"] == "COMPLETED"
    assert result["data"]["validation"]["status"] == "reproduced"
    metric_phase = [event for event in events if event.get("phase_id") == "validate_metrics"][-1]
    review_phase = [event for event in events if event.get("phase_id") == "review_result_summary"][-1]
    assert metric_phase["status"] == "success" and metric_phase["outcome"] == "reproduced"
    assert review_phase["status"] == "error" and review_phase["reason"] == "summary rejected"


@pytest.mark.parametrize("analysis_enabled", [False, True])
def test_prepare_only_plan_skips_every_unexecuted_review_and_experiment(pipeline, analysis_enabled):
    events = []
    result = invoke(pipeline, authorize_result_review=True, events=events,
                    prepare_only=True, use_llm_review=analysis_enabled)
    assert result["data"]["validation"]["status"] == "prepared"
    plan = next(event["stages"] for event in events if event["type"] == "pipeline_plan")
    enabled = [stage["id"] for stage in plan if stage["status"] != "skipped"]
    assert enabled == ["select_experiment", "prepare_repository", "prepare_environment_plan", "generate_report"]
    pipeline.runner.run.assert_not_called()
    pipeline.analysis_mock.assert_not_called()
    pipeline.review_mock.assert_not_called()
    assert_call_delta(pipeline, result, 0)


def test_no_analysis_plan_and_reserved_optimization_do_not_execute_model_or_optimizer(pipeline):
    pipeline.runner.run.side_effect = None
    pipeline.runner.run.return_value = {"success": False, "executed": False,
                                       "final": {"stderr": "Execution unavailable", "exit_code": -2}}
    events = []
    result = invoke(pipeline, authorize_result_review=True, events=events,
                    use_llm_review=False, enable_optimization=True)
    plan = next(event["stages"] for event in events if event["type"] == "pipeline_plan")
    enabled = [stage["id"] for stage in plan if stage["status"] != "skipped"]
    assert enabled == ["select_experiment", "prepare_repository", "prepare_environment_plan", "execute_repository",
                       "verify_protocol", "validate_metrics", "generate_report"]
    optimization = result["data"]["optimization"]
    assert optimization["requested"] is True and optimization["available"] is False
    assert optimization["optimized"] is False and optimization["status"] == "not_implemented"
    assert any(event.get("phase_id") == "verify_protocol" and event["status"] == "blocked" for event in events)
    assert all(stage["agent"] != "Optimizer" for stage in plan)
    pipeline.analysis_mock.assert_not_called()
    pipeline.review_mock.assert_not_called()
    assert_call_delta(pipeline, result, 0)
