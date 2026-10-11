"""User URL fallback retains source provenance and all experiment gates."""
from copy import deepcopy
import hashlib
from unittest.mock import Mock

import pytest

from discovered_fixtures import identity_review, write_pdf
from frontend.repository_fallback import fallback_for_pdf, manual_repository, repository_widget_key
from src.agents.repository_verifier import identity_view, validate_identity_review
from src.discovered_repository_reproduction import author_repository_input
from src.discovered_repository_plan import build_evidence_packet
from test_discovered_repository_service import (setup_service, use_preinstalled_cache,
                                              cpu_dependency_cache)
from test_pdf_pipeline_entrypoint import safe_handoff


def test_fallback_is_available_only_for_same_failed_pdf():
    result = {"data": {"repository_fallback": {"available": True, "pdf_sha256": "a" * 64, "reason": "Not confirmed"}}}
    assert fallback_for_pdf(result, "a" * 64)["reason"] == "Not confirmed"
    assert fallback_for_pdf(result, "b" * 64) is None
    assert fallback_for_pdf({}, "a" * 64) is None
    assert repository_widget_key("a" * 64) != repository_widget_key("b" * 64)


@pytest.mark.parametrize("value", ["https://example.com/owner/repo", "file:///C:/data", "https://github.com/search", "not a URL"])
def test_manual_url_cannot_bypass_github_url_validation(value):
    url, error = manual_repository(value)
    assert not url and error


def test_manual_url_normalizes_repository_root():
    assert manual_repository("https://github.com/fixture-lab/model/blob/main/train.py") == ("https://github.com/fixture-lab/model", "")
    assert manual_repository("") == ("", "")


def test_failed_auto_confirmation_exposes_fallback_and_preserves_original_run(tmp_path, monkeypatch):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    write_pdf(fixture["pdf"], [["Separating Signed Signals", "Abstract", "No repository link was printed."]])
    runner.runner.run = Mock(side_effect=AssertionError("Unconfirmed lookup cannot execute"))
    result = runner.run(request)
    fallback = result["data"]["repository_fallback"]
    assert result["state"] == "ERROR" and result["data"]["validation"]["is_reproduced"] is None
    assert fallback["available"] and fallback["pdf_sha256"] == hashlib.sha256(fixture["pdf"].read_bytes()).hexdigest()
    assert result["data"]["run_dir"]
    runner.runner.run.assert_not_called()


def test_user_url_is_candidate_provenance_and_does_not_claim_pdf_authorship(tmp_path, monkeypatch):
    fixture, _, request, _ = setup_service(tmp_path, monkeypatch)
    write_pdf(fixture["pdf"], [["Separating Signed Signals", "Abstract", "No repository link was printed."]])
    request["preferred_repo_url"] = fixture["snapshot"]["url"]
    document, url, evidence = author_repository_input(request, require_author=False)
    assert evidence["source"] == "user_supplied" and evidence["is_author_code"] is False and evidence["page"] is None
    with pytest.raises(ValueError):
        author_repository_input(request)
    packet = build_evidence_packet(fixture["pdf"], fixture["workspace"], fixture["snapshot"], for_identity_review=True)
    with pytest.raises(ValueError):
        identity_view(packet)
    packet["repository_input"] = {"source": "user_supplied", "url": url}
    view = identity_view(packet)
    # Use source IDs as the real model does, with honest unconfirmed ownership.
    model_view = deepcopy(view)
    for page in model_view["pdf"]["pages"]:
        page.pop("text")
    review = validate_identity_review(packet, identity_review(model_view))
    assert review["accepted"] and not review["author_identity_confirmed"]
    assert review["selection_source"] == "user_supplied"
    packet.pop("repository_input")
    with pytest.raises(ValueError):
        validate_identity_review(packet, review)


def test_user_provided_repository_still_completes_native_execution_and_independent_metrics(
        tmp_path, monkeypatch, cpu_dependency_cache):
    fixture, runner, request, _ = setup_service(tmp_path, monkeypatch)
    request["preferred_repo_url"] = fixture["snapshot"]["url"]
    use_preinstalled_cache(monkeypatch, runner, cpu_dependency_cache)
    result = runner.run(request)
    assert result["state"] == "COMPLETED", result["error"]
    data = result["data"]
    assert data["execution_source"] == "user_supplied_repository" and data["repository_executed"] is True
    identity = data["resources"]["repository_identity"]
    assert identity["status"] == "user_supplied_verified" and identity["author_identity_confirmed"] is False
    assert data["validation"]["is_reproduced"] and data["validation"]["independent_metrics_pass"]
    assert "用户在自动确认失败后手动提供" in data["report"]
    assert "未确认；仅确认论文与实现内容匹配" in data["report"]


def test_manual_url_survives_managed_runtime_handoff(tmp_path, safe_handoff):
    import frontend.backend_pipeline as backend
    from test_pdf_input_routing import TITLE, write_pdf as routed_pdf
    source = routed_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", "Paper content"])
    requests, _, _, host = safe_handoff
    backend.run_pipeline_core(str(tmp_path / "manual.jsonl"), pdf_path=str(source),
                               preferred_repo_url="https://github.com/fixture-lab/provided")
    request, = requests
    assert request["preferred_repo_url"] == "https://github.com/fixture-lab/provided"
    assert request["experiment_profile"] is None
    assert request["_managed_runtime"]
    host.assert_not_called()


def test_app_shows_last_resort_only_after_failure_and_submits_user_url(tmp_path, monkeypatch):
    from pathlib import Path
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
    from streamlit.proto.Common_pb2 import FileURLs
    import frontend.repository_entrypoint as entrypoint
    import frontend.history_manager as history
    import src.local_llm_settings as settings
    from test_pdf_input_routing import TITLE, write_pdf as routed_pdf
    source = routed_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", "Paper content"])
    payload = source.read_bytes()
    pdf_hash = hashlib.sha256(payload).hexdigest()
    upload = UploadedFile(UploadedFileRec("manual-fixture", "paper.pdf", "application/pdf", payload), FileURLs())

    def uploaded_file(*args, **kwargs):
        st.session_state[kwargs["key"]] = upload
        return upload

    monkeypatch.setattr(st, "file_uploader", uploaded_file)
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    monkeypatch.setattr(settings, "load_local_llm_settings", Mock())
    monkeypatch.delenv("AUTOREPRO_RESUME_PROGRESS", raising=False)
    capture = Mock()
    monkeypatch.setattr(entrypoint, "background_entrypoint", lambda previous: capture)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    app.session_state["input_mode"] = "上传PDF"
    app.session_state["mock_mode"] = False
    app.session_state["docker_probe"] = (False, "test local runtime")
    app.run()
    assert not app.exception
    assert not any(item.label == "补充 GitHub 仓库" for item in app.text_input)
    old_failure = tmp_path / "original_failure.json"
    old_failure.write_text("Original failed run evidence", encoding="utf-8")
    app.session_state["result"] = {"state": "ERROR", "error": "Not confirmed", "data": {
        "repository_fallback": {"available": True, "pdf_sha256": pdf_hash, "reason": "Insufficient repository evidence"}}}
    app.run()
    assert not app.exception
    key = repository_widget_key(pdf_hash)
    app.text_input(key=key).set_value("https://github.com/fixture-lab/provided").run()
    assert not app.exception
    next(button for button in app.button if button.label == "开始复现").click().run()
    assert not app.exception
    request = capture.call_args.kwargs
    assert request["preferred_repo_url"] == "https://github.com/fixture-lab/provided"
    assert request["experiment_profile"] is None and request["pdf_path"]
    assert old_failure.read_text(encoding="utf-8") == "Original failed run evidence"
    assert Path(request["pdf_path"]).read_bytes() == payload
    Path(request["pdf_path"]).unlink()
