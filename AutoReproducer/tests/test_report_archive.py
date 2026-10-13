"""Historical reports retain local rendering input after progress is cleaned."""
import json
from pathlib import Path
from types import SimpleNamespace

from frontend import backend_pipeline as pipeline
from frontend.history_manager import get_session_report_result


def test_backend_archives_report_without_request_credentials(tmp_path, monkeypatch):
    secret = "private-request-fixture"
    data = {
        "paper_info": {"title": "Archived experiment"},
        "report": "# Legacy report",
        "execution": {"final": {"exit_code": 0, "stdout": "accuracy=0.845"}},
        "validation": {"status": "author_code_completed", "is_reproduced": None,
                       "metric_records": [{"name": "accuracy", "value": 0.845}]},
        "model_settings": {"api_key": secret},
    }
    monkeypatch.setattr(pipeline, "get_data_root", lambda: tmp_path)
    monkeypatch.setattr(pipeline, "LLMClient", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "_create_orchestrator", lambda **kwargs: SimpleNamespace(
        run=lambda *args, **kwargs: {"state": "COMPLETED", "data": data, "error": None}))
    progress = tmp_path / "progress.jsonl"
    result = pipeline.run_pipeline_core(str(progress), paper_title="Archived experiment",
                                        mock_mode=True, api_key=secret)
    report = Path(result["report_path"])
    archive = report.with_suffix(".result.json")
    persisted = json.loads(archive.read_text(encoding="utf-8"))
    assert secret not in archive.read_text(encoding="utf-8")
    assert persisted["state"] == result["state"] == "COMPLETED"
    assert persisted["data"]["execution"] == result["data"]["execution"]
    assert persisted["data"]["validation"] == result["data"]["validation"]
    assert persisted["data"]["report"] == report.read_text(encoding="utf-8")
    progress.unlink()
    restored = get_session_report_result({"state": "COMPLETED", "report_path": str(report)})
    assert restored["data"]["report"] == persisted["data"]["report"]
    assert restored["data"]["validation"]["metric_records"][0]["value"] == 0.845
