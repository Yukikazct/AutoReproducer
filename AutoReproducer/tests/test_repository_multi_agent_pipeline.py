"""Preset execution and persistence without legacy model review gates."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import repository_reproduction as reproduction
from src import repository_validation
from src.audit.audit_logger import AuditLogger
from src.repository_analysis import RepositoryAnalysis
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
                 "steps": [final],
                 "final": final, "artifacts": [], "llm_calls": 0,
                 "environment": {"dependencies_path": private_path + "/deps"}}

    def execute(workspace, steps, environment, *, use_docker, on_event):
        assert not timeline, "Review stages must not precede execution"
        # Commands remain the production profile, with no generated replacement.
        assert steps == [{**step, "depends_on": [], "timeout_s": None}
                         for step in profile["steps"] if step.get("kind") != "check"]
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

    analysis_mock = Mock(side_effect=AssertionError("Deleted readiness review must not run"))
    review_mock = Mock(side_effect=AssertionError("Deleted result review must not run"))
    monkeypatch.setattr(RepositoryAnalysis, "run", analysis_mock)
    monkeypatch.setattr(RepositoryAnalysis, "review_result_summary", review_mock)
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
    saved = json.loads((Path(result["data"]["run_dir"]) / "result.json").read_text(encoding="utf-8"))
    assert saved["data"]["total_llm_calls"] == expected


def assert_no_reviews(pipeline, result):
    for function in (pipeline.source_mock, pipeline.analysis_mock, pipeline.review_mock,
                     pipeline.protocol, pipeline.recompute):
        function.assert_not_called()
    assert_call_delta(pipeline, result, 0)
    for name in ("repository_analysis.json", "result_analysis.json"):
        assert not (Path(result["data"]["run_dir"]) / name).exists()


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("result_review", [False, True])
def test_legacy_flags_cannot_restore_reviews_or_claim_numeric_reproduction(pipeline, review, result_review):
    result = invoke(pipeline, use_llm_review=review, authorize_result_review=result_review)
    assert result["state"] == "COMPLETED", result["error"]
    assert result["data"]["validation"]["status"] == "author_code_completed"
    assert result["data"]["validation"]["is_reproduced"] is None
    records = result["data"]["validation"]["metric_records"]
    assert {record["name"]: record["value"] for record in records} == {"mse": .389, "mae": .407}
    pipeline.runner.run.assert_called_once()
    assert_no_reviews(pipeline, result)


def test_execution_proceeds_to_report_with_no_review_phases(pipeline):
    events = []
    result = invoke(pipeline, events=events, authorize_result_review=True)
    assert result["state"] == "COMPLETED", result["error"]
    plan = next(event["stages"] for event in events if event["type"] == "pipeline_plan")
    assert [stage["id"] for stage in plan] == ["select_experiment", "prepare_repository",
        "prepare_environment_plan", "prepare_dependencies", "execute_repository", "generate_report"]
    states = [(event["phase_id"], event["status"]) for event in events
              if event.get("type") == "state" and event.get("phase_id")]
    assert states.index(("execute_repository", "success")) < states.index(("generate_report", "running"))
    assert_no_reviews(pipeline, result)


@pytest.mark.parametrize("failure", ["nonzero", "cancelled", "timed_out", "unfinished"])
def test_actual_execution_failures_still_persist_error_and_report(pipeline, failure):
    final = {"success": True, "exit_code": 0, "stdout": "", "stderr": ""}
    if failure == "nonzero":
        final.update(exit_code=1, success=False, stderr="Actual author error")
    elif failure == "unfinished":
        final["exit_code"] = None
    else:
        final[failure] = True
    pipeline.runner.run.side_effect = None
    pipeline.runner.run.return_value = {"executed": True, "success": final["success"],
                                       "final": final, "steps": [final], "artifacts": []}
    events = []
    result = invoke(pipeline, events=events, authorize_result_review=True)
    assert result["state"] == "ERROR"
    assert result["error"].startswith("EXECUTE_CODE 阶段失败:")
    assert result["data"]["validation"]["status"] == "execution_failed"
    assert any(event.get("phase_id") == "execute_repository" and event["status"] == "error" for event in events)
    assert (Path(result["data"]["run_dir"]) / "report.md").is_file()
    assert_no_reviews(pipeline, result)


@pytest.mark.parametrize("review", [False, True])
def test_prepare_only_does_not_run_author_commands_or_reviewers(pipeline, review):
    events = []
    result = invoke(pipeline, events=events, prepare_only=True,
                    authorize_result_review=True, use_llm_review=review)
    assert result["data"]["validation"]["status"] == "prepared"
    plan = next(event["stages"] for event in events if event["type"] == "pipeline_plan")
    assert [stage["id"] for stage in plan if stage["status"] != "skipped"] == [
        "select_experiment", "prepare_repository", "prepare_environment_plan", "generate_report"]
    pipeline.runner.run.assert_not_called()
    assert_no_reviews(pipeline, result)


def test_reserved_optimization_never_adds_execution_or_model_calls(pipeline):
    result = invoke(pipeline, authorize_result_review=True, enable_optimization=True)
    assert result["state"] == "COMPLETED", result["error"]
    optimization = result["data"]["optimization"]
    assert optimization["requested"] is True and optimization["available"] is False
    assert optimization["optimized"] is False and optimization["status"] == "not_implemented"
    pipeline.runner.run.assert_called_once()
    assert_no_reviews(pipeline, result)
