"""Online review failures belong to the role that actually produced them."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontend.backend_pipeline import ProgressStore
from src.method_reproduction import MethodReproduction


ROLES = ("reader", "finder", "builder", "verifier")
SOURCE = {"source_id": "author", "content": "official model and optimizer",
          "url": "https://example.org/pinned/author.py", "locator": "author.py"}


@pytest.fixture
def method_run(tmp_path, monkeypatch):
    adapter = SimpleNamespace(prepare_dataset=Mock(return_value={}), materialize=Mock(return_value={}),
        public_sources=Mock(return_value=[SOURCE]), steps=Mock(return_value=[]),
        verify=Mock(return_value={"status": "method_experiment_completed", "is_reproduced": None}))
    monkeypatch.setattr("src.method_reproduction.get_adapter", lambda profile: adapter)
    def export(root, profile, workspace, **kwargs):
        workspace.mkdir()
        return {"url": "https://example.org/repository", "path": str(workspace)}
    monkeypatch.setattr("src.repository_reproduction.export_repository", export)
    monkeypatch.setattr("src.agents.report_generator.ReportGeneratorAgent.run",
                        lambda *args, **kwargs: {"report": "saved review report"})
    runner = SimpleNamespace(run=Mock(return_value={"success": True, "artifacts": [], "final": {}}))
    logger = Mock()
    logger.get_stats.return_value = {}
    llm = SimpleNamespace(mock_mode=False, base_url="https://example.org", model="test-model",
                          get_call_count=lambda: 0)
    service = MethodReproduction(tmp_path, logger, runner=runner, llm=llm)
    progress_path = tmp_path / "progress.jsonl"
    progress = ProgressStore(str(progress_path))
    def run():
        result = service.run({"experiment_profile": "neural_ode_spiral", "use_llm_review": True,
                              "offline": True}, on_event=progress.emit)
        progress.emit({"type": "done", "result": result})
        rows = {row["id"]: row for row in ProgressStore.read_snapshot(str(progress_path))["pipeline_stages"]}
        return result, rows
    return run, runner


def accepted_response():
    return {"response": json.dumps({"status": "accepted", "summary": "The supplied code supports this review.",
            "evidence": [{"source_id": SOURCE["source_id"], "quote": SOURCE["content"]}]})}


@pytest.mark.parametrize("failed_role", ROLES)
def test_each_review_rejection_preserves_prior_success_and_blocks_unstarted_roles(method_run, monkeypatch, failed_role):
    run, runner = method_run
    index = ROLES.index(failed_role)
    rejected = {"response": json.dumps({"status": "insufficient_evidence",
                "summary": "Missing source mapping for " + failed_role, "evidence": []})}
    request = Mock(side_effect=[accepted_response() for _ in ROLES[:index]] + [rejected])
    monkeypatch.setattr("src.method_advice.request_text", request)
    result, rows = run()
    assert result["state"] == "ERROR"
    assert [identifier for identifier, row in rows.items() if row["status"] == "error"] == ["review_" + failed_role]
    for role in ROLES[:index]:
        assert rows["review_" + role]["status"] == "success"
    for role in ROLES[index + 1:]:
        assert rows["review_" + role]["status"] == "blocked"
        assert "未执行" in rows["review_" + role]["reason"]
    assert rows["execute_repository"]["status"] == "blocked"
    assert rows["generate_report"]["status"] == "success"
    assert failed_role in rows["review_" + failed_role]["reason"]
    runner.run.assert_not_called()
    assert request.call_count == index + 1
    data = result["data"]
    assert data["analysis_status"] == "public_readiness_rejected"
    assert data["validation"]["status"] == "analysis_failed"
    assert data["validation"]["failure_phase"] == "review_" + failed_role
    saved = json.loads((Path(data["run_dir"]) / "method_analysis.json").read_text(encoding="utf-8"))
    assert saved == data["method_analysis"]
    assert saved["failed_role"] == failed_role
    assert [review["role"] for review in saved["reviews"]] == list(ROLES[:index])
    assert saved["attempts"][-1]["raw_response"] == rejected["response"]


def test_successful_reviews_are_also_saved_before_training(method_run, monkeypatch):
    run, runner = method_run
    monkeypatch.setattr("src.method_advice.request_text", Mock(side_effect=[accepted_response() for _ in ROLES]))
    result, rows = run()
    assert result["state"] == "COMPLETED"
    assert all(rows["review_" + role]["status"] == "success" for role in ROLES)
    runner.run.assert_called_once()
    data = result["data"]
    saved = json.loads((Path(data["run_dir"]) / "method_analysis.json").read_text(encoding="utf-8"))
    assert saved == data["method_analysis"]
    assert saved["status"] == "accepted" and len(saved["reviews"]) == 4
