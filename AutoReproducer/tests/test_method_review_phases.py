"""Legacy method review settings cannot add blocking phases or model calls."""
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
        public_sources=Mock(return_value=[SOURCE]),
        steps=Mock(side_effect=lambda profile, train=True: [
            {"id": "train" if train else "import_check", "kind": "train" if train else "check",
             "argv": ["python", "-c", "print(1)"]}]),
        verify_environment=Mock(),
        verify=Mock(return_value={"status": "method_experiment_completed", "is_reproduced": None}))
    monkeypatch.setattr("src.method_reproduction.get_adapter", lambda profile: adapter)
    def export(root, profile, workspace, **kwargs):
        workspace.mkdir()
        return {"url": "https://example.org/repository", "path": str(workspace)}
    monkeypatch.setattr("src.repository_reproduction.export_repository", export)
    monkeypatch.setattr("src.agents.report_generator.ReportGeneratorAgent.run",
                        lambda *args, **kwargs: {"report": "saved execution report"})
    runner = SimpleNamespace(run=Mock(return_value={"success": True, "executed": True,
        "artifacts": [], "final": {"success": True, "exit_code": 0, "stdout": ""}}))
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


@pytest.mark.parametrize("legacy_response", ["accepted", "insufficient_evidence", "invalid JSON"])
def test_legacy_reviewer_responses_do_not_change_execution(method_run, monkeypatch, legacy_response):
    run, runner = method_run
    request = Mock(return_value={"response": legacy_response})
    monkeypatch.setattr("src.method_advice.request_text", request)
    result, rows = run()
    assert result["state"] == "COMPLETED", result["error"]
    assert not any(identifier.startswith("review_") for identifier in rows)
    assert rows["execute_repository"]["status"] == "success"
    assert rows["generate_report"]["status"] == "success"
    assert runner.run.call_count == 2
    assert [step["id"] for step in runner.run.call_args_list[0].args[1]] == ["prepare_dependencies"]
    assert [step["id"] for step in runner.run.call_args_list[1].args[1]] == ["train"]
    request.assert_not_called()
    data = result["data"]
    assert data["validation"]["status"] == "author_code_completed"
    assert data["validation"]["is_reproduced"] is None
    assert not (Path(data["run_dir"]) / "method_analysis.json").exists()
    saved = json.loads((Path(data["run_dir"]) / "result.json").read_text(encoding="utf-8"))
    assert saved["data"]["validation"] == data["validation"]
