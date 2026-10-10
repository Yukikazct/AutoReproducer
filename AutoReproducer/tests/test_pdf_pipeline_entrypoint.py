"""PDF uploads exercise the actual web/backend entrypoints without training."""
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

import frontend.backend_pipeline as backend
import frontend.history_manager as history
import src.local_llm_settings as local_settings
import src.pdf_input as pdf_input
import src.runtime_preparation as runtime
from src.audit.audit_logger import AuditLogger
from src.repository_reproduction import RepositoryReproduction
from test_pdf_input_routing import TITLE, write_pdf


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def safe_handoff(monkeypatch, tmp_path):
    prepared = runtime.RuntimePreparation(str(tmp_path / "safe-python.exe"),
                                          True, False, "Store host", "cpython-312")
    requirement = Mock(return_value="Switch Store host to safe Python")
    prepare = Mock(return_value=prepared)
    monkeypatch.setattr(runtime, "runtime_requirement", requirement)
    monkeypatch.setattr(runtime, "prepare_runtime", prepare)
    host_factory = Mock(side_effect=AssertionError("Host must not construct an orchestrator"))
    monkeypatch.setattr(backend, "_create_orchestrator", host_factory)
    requests = []

    def worker(command, **kwargs):
        request = json.loads(kwargs["input"])
        requests.append(request)
        assert command[0] == prepared.executable
        assert "LLM_API_KEY" not in kwargs["env"]
        child = backend.ProgressStore(request["progress_path"], reset=False)
        child.emit({"type": "done", "result": {
            "state": "COMPLETED", "error": None,
            "data": {"pdf_resolution": request.get("pdf_resolution")},
        }})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime, "run_owned_process", worker)
    return requests, requirement, prepare, host_factory


@pytest.mark.parametrize("review", [True, False])
def test_backend_pdf_is_identified_before_store_handoff_and_preserves_review_choice(tmp_path, safe_handoff, review):
    requests, requirement, prepare, host_factory = safe_handoff
    source = write_pdf(tmp_path / "paper.pdf", [TITLE, "Authors", "Abstract"])
    progress = tmp_path / "progress.jsonl"
    result = backend.run_pipeline_core(str(progress), pdf_path=str(source),
                                       use_llm_review=review, allow_result_summary_review=False)
    request, = requests
    assert request["experiment_profile"] == "neural_ode_spiral"
    assert request["pdf_resolution"]["title"] == TITLE
    assert request["pdf_resolution"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert request["use_llm_review"] is review
    assert request["allow_result_summary_review"] is False
    assert request["optimization_mode"] == "off"
    assert request["_managed_runtime"] is True
    assert result["data"]["pdf_resolution"] == request["pdf_resolution"]
    requirement.assert_called_once()
    prepare.assert_called_once()
    host_factory.assert_not_called()
    events = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "pdf_resolution"
    assert backend.ProgressStore.read_snapshot(str(progress))["done"] is True


@pytest.mark.parametrize("kind", ["missing", "invalid", "blank"])
def test_invalid_backend_pdf_rejected_before_constructing_llm_or_orchestrator(monkeypatch, tmp_path, kind):
    source = tmp_path / "paper.pdf"
    if kind == "invalid":
        source.write_text("not PDF content", encoding="utf-8")
    elif kind == "blank":
        write_pdf(source)
    llm = Mock(side_effect=AssertionError("Rejected PDF must not construct LLM"))
    host = Mock(side_effect=AssertionError("Rejected PDF must not construct orchestrator"))
    prepare = Mock(side_effect=AssertionError("Invalid bytes do not require training runtime"))
    monkeypatch.setattr(backend, "LLMClient", llm)
    monkeypatch.setattr(backend, "_create_orchestrator", host)
    monkeypatch.setattr(runtime, "prepare_runtime", prepare)
    progress = tmp_path / "progress.jsonl"
    result = backend.run_pipeline_core(str(progress), pdf_path=str(source), paper_title=TITLE)
    assert result["state"] == "ERROR"
    assert result["data"]["pdf_input"]["status"] == "rejected"
    assert "execution" not in result["data"]
    assert backend.ProgressStore.read_snapshot(str(progress))["running"] is False
    llm.assert_not_called()
    host.assert_not_called()
    prepare.assert_not_called()


def test_missing_pdf_parser_recovers_before_host_agents_and_resolves_again_in_managed_worker(monkeypatch, tmp_path):
    source = write_pdf(tmp_path / "paper.pdf", [TITLE, "Authors", "Abstract"])
    progress = tmp_path / "progress.jsonl"
    real_resolve = pdf_input.resolve_pdf_request
    resolver = Mock(side_effect=[pdf_input.PDFParserUnavailable("parser missing"),
                                real_resolve({"pdf_path": str(source)})])
    monkeypatch.setattr(pdf_input, "resolve_pdf_request", resolver)
    prepared = runtime.RuntimePreparation(str(tmp_path / "safe-python.exe"), True, False,
                                          "PDF parser recovery", "cpython-312")
    prepare = Mock(return_value=prepared)
    monkeypatch.setattr(runtime, "prepare_runtime", prepare)
    requirement = Mock(side_effect=AssertionError("Known missing parser already requires recovery"))
    monkeypatch.setattr(runtime, "runtime_requirement", requirement)
    logger = Mock(session_id="pdf-fixture")
    logger.get_summary.return_value = []
    logger.get_stats.return_value = {"duration_sec": 0, "llm_calls": 0}
    monkeypatch.setattr(backend, "AuditLogger", Mock(return_value=logger))
    managed = {"active": False}
    requests = []

    class ReviewedFixture:
        def run(self, request, on_event):
            requests.append(request)
            return {"state": "COMPLETED", "error": None,
                    "data": {"pdf_resolution": request["pdf_resolution"]}}

    def factory(**kwargs):
        assert managed["active"], "Host agents ran before parser recovery"
        return ReviewedFixture()

    monkeypatch.setattr(backend, "_create_orchestrator", factory)

    def worker(command, **kwargs):
        payload = json.loads(kwargs["input"])
        assert payload["experiment_profile"] is None
        assert payload["pdf_path"] == str(source)
        managed["active"] = True
        try:
            result = backend.run_pipeline_core(**payload)
        finally:
            managed["active"] = False
        assert result["state"] == "COMPLETED"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime, "run_owned_process", worker)
    result = backend.run_pipeline_core(str(progress), pdf_path=str(source), use_llm_review=False)
    assert result["state"] == "COMPLETED", result.get("error")
    prepare.assert_called_once()
    requirement.assert_not_called()
    assert resolver.call_count == 2
    request, = requests
    assert request["experiment_profile"] == "neural_ode_spiral"
    assert result["data"]["pdf_resolution"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize("title", [TITLE, "Are Transformers Effective for Time Series Forecasting?"])
def test_pdf_provenance_survives_author_source_failure_in_both_reviewed_pipeline_types(monkeypatch, tmp_path, title):
    import src.repository_reproduction as repository
    source = write_pdf(tmp_path / "paper.pdf", [title, "Authors", "Abstract"])
    request = pdf_input.resolve_pdf_request({"pdf_path": str(source), "use_llm_review": False})
    unavailable = Mock(side_effect=RuntimeError("author source unavailable fixture"))
    monkeypatch.setattr(repository, "export_repository", unavailable)
    logger = AuditLogger(log_dir=str(tmp_path / "logs"))
    result = RepositoryReproduction(tmp_path / "data", logger).run(request)
    unavailable.assert_called_once()
    assert "author source unavailable fixture" in result["error"]
    assert result["data"]["pdf_resolution"] == request["pdf_resolution"]
    assert result["data"]["pdf_resolution"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize("review", [None, False])
@pytest.mark.parametrize("stale_action", ["仅准备源码和命令", "准备实验环境"])
def test_actual_upload_start_verifies_bytes_and_ignores_stale_preset_controls(monkeypatch, tmp_path, safe_handoff, review, stale_action):
    requests, requirement, prepare, host_factory = safe_handoff
    source = write_pdf(tmp_path / "real-upload.pdf", [TITLE, "Authors", "Abstract", "Actual upload text"])
    payload = source.read_bytes()
    upload = UploadedFile(UploadedFileRec("fixture-file", "unrelated-name.pdf", "application/pdf", payload), FileURLs())
    monkeypatch.delenv("AUTOREPRO_RESUME_PROGRESS", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.invalid")
    monkeypatch.setenv("LLM_MODEL", "fixture-model")
    monkeypatch.setenv("LLM_API_KEY", "fixture-key")
    monkeypatch.setattr(local_settings, "load_local_llm_settings", Mock())
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)

    def uploaded_file(*args, **kwargs):
        st.session_state[kwargs["key"]] = upload
        return upload

    monkeypatch.setattr(st, "file_uploader", uploaded_file)
    threads, progress_paths = [], []
    real_start = backend.run_pipeline_background

    def start(path, **kwargs):
        assert not Path(path).exists(), "Do not overwrite an existing user run"
        progress_paths.append(Path(path))
        thread = real_start(path, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(backend, "run_pipeline_background", start)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["input_mode"] = "上传PDF"
    app.session_state["mock_mode"] = False
    app.session_state["docker_probe"] = (False, "test uses local execution")
    try:
        app.run()
        assert not app.exception
        assert any("rtqichen/torchdiffeq" in notice.value for notice in app.success)
        assert app.sidebar.checkbox(key="pdf_llm_review").value is True
        if review is not None:
            app.sidebar.checkbox(key="pdf_llm_review").set_value(review).run()
            assert not app.exception
        app.session_state["experiment_profile"] = "siren_camera_quick"
        app.session_state["method_action"] = stale_action
        app.session_state["method_optimization"] = "validate"
        app.session_state["repository_prepare_only"] = True
        app.session_state["repository_result_review"] = True
        next(button for button in app.sidebar.button if "开始复现" in button.label).click().run()
        assert not app.exception
        assert len(threads) == 1
        threads[0].join(timeout=5)
        assert not threads[0].is_alive()
        request, = requests
        assert request["experiment_profile"] == "neural_ode_spiral"
        assert request["pdf_resolution"]["sha256"] == hashlib.sha256(payload).hexdigest()
        assert request["use_llm_review"] is (True if review is None else review)
        assert request["prepare_only"] is False and request["prepare_environment"] is False
        assert request["optimization_mode"] == "off" and request["enable_optimization"] is False
        assert request["allow_result_summary_review"] is False
        assert request["pdf_path"] != str(source)
        assert not Path(request["pdf_path"]).exists(), "Background owner must clean its upload copy"
        host_factory.assert_not_called()
        prepare.assert_called_once()
        requirement.assert_called_once()
        app.run()
        assert not app.exception
        assert app.session_state["running"] is False
    finally:
        for thread in threads:
            thread.join(timeout=5)
        for path in progress_paths:
            path.unlink(missing_ok=True)
