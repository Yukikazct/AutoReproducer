"""Historical reports retain local rendering input after progress is cleaned."""
import json
import io
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
import zipfile

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
    assert persisted["data"]["reproduction_conclusion"] == result["data"]["reproduction_conclusion"]
    assert persisted["data"]["reproduction_conclusion"]["is_reproduced"] is None
    progress.unlink()
    restored = get_session_report_result({"state": "COMPLETED", "report_path": str(report)})
    assert restored["data"]["report"] == persisted["data"]["report"]
    assert restored["data"]["validation"]["metric_records"][0]["value"] == 0.845


@pytest.mark.parametrize("with_image", [False, True])
def test_final_web_report_updates_run_copy_and_portable_attachments(tmp_path, monkeypatch, with_image):
    """Late runtime details and the output footer must survive every saved copy."""
    from PIL import Image
    from src import execution_artifacts

    run_dir = tmp_path / "runs" / "fixture"
    run_dir.mkdir(parents=True)
    artifacts = []
    if with_image:
        monkeypatch.setattr(execution_artifacts, "ARTIFACT_ROOT", tmp_path / "artifacts")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        Image.new("RGB", (8, 8), "blue").save(workspace / "plot.png")
        artifacts = execution_artifacts.collect_images(str(workspace), "full")["artifacts"]
    data = {
        "paper_title": "Publication fixture",
        "run_dir": str(run_dir), "report_path": str(run_dir / "report.md"),
        "report": "# Preliminary author report",
        "execution": {"final": {"exit_code": 0, "stdout": "mean accuracy: 0.845",
                                 "stderr": "", "artifacts": artifacts}},
        "validation": {"status": "author_code_completed", "paper_metrics_verified": False,
                       "is_reproduced": None},
    }
    (run_dir / "report.md").write_text(data["report"], encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(
        {"state": "COMPLETED", "data": data, "error": None}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "get_data_root", lambda: tmp_path)
    monkeypatch.setattr(pipeline, "LLMClient", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "_create_orchestrator", lambda **kwargs: SimpleNamespace(
        run=lambda *args, **kwargs: {"state": "COMPLETED", "data": data, "error": None}))
    result = pipeline.run_pipeline_core(
        str(tmp_path / "progress.jsonl"), mock_mode=True,
        _runtime_preparation={"preset": "fixture-standard-cpython"})
    report = Path(result["report_path"])
    text = report.read_text("utf-8")
    assert (run_dir / "report.md").read_text("utf-8") == text
    assert result["data"]["report"] == text
    assert result["data"]["report_length"] == len(text)
    for path in (run_dir / "result.json", report.with_suffix(".result.json")):
        saved = json.loads(path.read_text("utf-8"))
        assert saved["data"]["report"] == text
        assert saved["data"]["runtime_preparation"] == result["data"]["runtime_preparation"]
        assert saved["data"]["validation"] == result["data"]["validation"]
        assert saved["data"]["execution"] == result["data"]["execution"]
        assert saved["data"]["environment_cleanup"] == result["data"]["environment_cleanup"]
    attachment = report.with_name(report.stem + "_execution.txt")
    assert (run_dir / attachment.name).read_bytes() == attachment.read_bytes()
    assert "mean accuracy: 0.845" in attachment.read_text("utf-8")
    if with_image:
        target = re.search(r"!\[运行结果图 1\]\(([^)]+)\)", text).group(1)
        assert (run_dir / unquote(target)).read_bytes() == (report.parent / unquote(target)).read_bytes()
    from frontend.report_renderer import build_report_bundle
    with zipfile.ZipFile(io.BytesIO(build_report_bundle(text, report))) as archive:
        assert archive.read(report.name).decode("utf-8") == text
        assert archive.read(attachment.name) == attachment.read_bytes()


def test_report_copy_failure_keeps_training_evidence_but_fails_publication(tmp_path, monkeypatch):
    from src.audit.audit_logger import AuditLogger

    logger = AuditLogger(log_dir=str(tmp_path / "audit"))
    monkeypatch.setattr(pipeline, "AuditLogger", lambda: logger)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    validation = {"status": "author_code_completed", "paper_metrics_verified": False,
                  "is_reproduced": None}
    data = {"run_dir": str(run_dir), "report": "# Completed author report",
            "validation": validation, "execution": {"success": True}}
    (run_dir / "result.json").write_text(json.dumps(
        {"state": "COMPLETED", "data": data, "error": None}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "get_data_root", lambda: tmp_path)
    monkeypatch.setattr(pipeline, "LLMClient", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "_create_orchestrator", lambda **kwargs: SimpleNamespace(
        run=lambda *args, **kwargs: {"state": "COMPLETED", "data": data, "error": None}))

    def locked_report(result):
        raise OSError("report destination is locked")

    monkeypatch.setattr(pipeline, "_sync_run_report", locked_report)
    progress = tmp_path / "progress.jsonl"
    result = pipeline.run_pipeline_core(str(progress), mock_mode=True)
    assert result["state"] == "ERROR" and "locked" in result["error"]
    assert result["data"]["validation"] == validation
    assert result["data"]["execution"]["success"] is True
    assert result["data"]["environment_cleanup"]["status"] == "released"
    saved = json.loads((run_dir / "result.json").read_text("utf-8"))
    assert saved["state"] == "ERROR" and "locked" in saved["error"]
    snapshot = pipeline.ProgressStore.read_snapshot(str(progress))
    assert snapshot["done"] is True and snapshot["result"]["state"] == "ERROR"
    assert logger.replay()[-1]["result"]["state"] == "ERROR"
    archived = get_session_report_result({"state": "ERROR", "report_path": result["report_path"]})
    assert archived["state"] == "ERROR" and archived["data"]["validation"] == validation
