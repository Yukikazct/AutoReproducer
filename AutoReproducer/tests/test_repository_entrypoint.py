"""New repository requests use fresh capabilities without changing active owners."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

import frontend.backend_pipeline as old_backend
import frontend.history_manager as history
import frontend.repository_entrypoint as entrypoint
import src.local_llm_settings as settings
from src.llm.llm_client import LLMClient
from src.repository_routing import (REZERO_DISCOVERY_REPOSITORY as AUTHOR,
                                    REZERO_PROFILE_ID as PROFILE,
                                    REZERO_TITLE as TITLE,
                                    REZERO_TRAINING_REPOSITORY as TRAINING)
from test_pdf_input_routing import write_pdf


CLAIM = "2Code for ReZero applied to various neural architectures: "
APP = Path(__file__).resolve().parents[1] / "app.py"


def test_stale_pdf_parser_and_loader_can_upgrade_without_mutating_running_globals(monkeypatch, tmp_path):
    legacy = ModuleType("src.pdf_input")
    legacy.PDF_INPUT_API_VERSION = 2
    legacy.owner = {"running": True}
    exec("def extract_pdf_input(path):\n    return owner\n", vars(legacy))
    legacy.resolve_pdf_request = Mock()
    old_extract = legacy.extract_pdf_input
    monkeypatch.setitem(sys.modules, "src.pdf_input", legacy)
    current = entrypoint.load_pdf_input()
    assert current is not legacy and current.PDF_INPUT_API_VERSION == 3
    assert sys.modules["src.pdf_input"] is legacy
    assert old_extract.__globals__ is vars(legacy) and old_extract(None) is legacy.owner
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", CLAIM + AUTHOR])
    assert current.resolve_pdf_request({"pdf_path": str(path)})["experiment_profile"] == PROFILE
    legacy.resolve_pdf_request.assert_not_called()


def test_same_api_with_old_source_generation_gets_current_hyphenation_fix(monkeypatch, tmp_path):
    legacy = ModuleType("src.pdf_input")
    legacy.PDF_INPUT_API_VERSION = 3
    legacy.PDF_SOURCE_FINGERPRINT = "old-source-with-same-api"
    legacy.owner = {"running": True}
    exec("def extract_pdf_input(path):\n    return owner\n", vars(legacy))
    legacy.resolve_pdf_request = Mock()
    monkeypatch.setitem(sys.modules, "src.pdf_input", legacy)
    # A cached routing module may have bound an older extractor too.
    import src.repository_routing as routing
    old = Mock(side_effect=AssertionError("Stale extractor was called"))
    monkeypatch.setattr(routing, "extract_current_repository_links", old)
    current = entrypoint.load_pdf_input()
    assert current is not legacy
    path = write_pdf(tmp_path / "wrapped.pdf", [
        "A Separate Paper", "Abstract", "Code is avail-",
        "able at: https://github.com/fixture-lab/wrapped-method",
    ])
    assert current.extract_pdf_input(path).repository_links[0]["is_author_code"] is True
    assert legacy.extract_pdf_input(None) is legacy.owner
    assert sys.modules["src.pdf_input"] is legacy
    old.assert_not_called()


def test_unregistered_pdf_gets_new_owned_backend_without_a_named_preset(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / "new.pdf", [
        "A Previously Unknown Paper", "Abstract", "Our code is at https://github.com/fixture-lab/new-method",
    ])
    previous = Mock(side_effect=AssertionError("Stale generic backend called"))
    latest = SimpleNamespace(run_pipeline_background=Mock(return_value="new-owner"))
    monkeypatch.setattr(entrypoint, "load_repository_backend", Mock(return_value=latest))
    assert entrypoint.background_entrypoint(previous)("progress", pdf_path=str(path)) == "new-owner"
    request = latest.run_pipeline_background.call_args.kwargs
    assert request["pdf_path"] == str(path) and not request.get("experiment_profile")


def test_stale_profile_registry_retains_existing_active_factory_and_uses_private_upgrade(monkeypatch):
    legacy = ModuleType("src.repository_profiles")
    legacy.PROFILE_LABELS = {"old_profile": "Old profile"}
    legacy.owner = {"running": True}
    exec("def get_profile(profile):\n    return owner\n", vars(legacy))
    old_factory = legacy.get_profile
    upgraded = ModuleType("src._reviewed_profiles")
    upgraded.PROFILE_LABELS = {"old_profile": "Old profile", PROFILE: "Reviewed ReZero"}
    upgraded.get_profile = Mock(return_value={"id": PROFILE})
    monkeypatch.setitem(sys.modules, "src.repository_profiles", legacy)
    loader = Mock(return_value=upgraded)
    monkeypatch.setattr(entrypoint, "_isolated_source", loader)
    assert PROFILE in entrypoint.profile_labels()
    assert entrypoint.get_profile(PROFILE) == {"id": PROFILE}
    assert entrypoint.get_profile("old_profile") is legacy.owner
    assert sys.modules["src.repository_profiles"] is legacy
    assert old_factory.__globals__ is vars(legacy) and legacy.owner == {"running": True}


def test_stale_registry_loads_real_registered_rezero_profile_without_replacing_existing_owner(monkeypatch):
    legacy = ModuleType("src.repository_profiles")
    legacy.PROFILE_LABELS = {"old_profile": "Old profile"}
    legacy.get_profile = Mock(return_value={"id": "old_profile"})
    monkeypatch.setitem(sys.modules, "src.repository_profiles", legacy)
    assert PROFILE in entrypoint.profile_labels()
    selected = entrypoint.get_profile(PROFILE)
    assert selected["id"] == PROFILE and selected["adapter_id"] == "rezero"
    assert selected["repository"]["url"] == TRAINING
    assert selected["parameters"]["epochs"] == 45
    assert selected["parameters"]["batch_size"] == 512
    assert sys.modules["src.repository_profiles"] is legacy
    legacy.get_profile.assert_not_called()


def test_new_repository_backend_delegates_to_owned_worker_without_touching_cached_backend(monkeypatch, tmp_path):
    old_core, old_globals = old_backend.run_pipeline_core, old_backend.run_pipeline_core.__globals__
    backend = entrypoint.load_repository_backend()
    assert backend is not old_backend
    prepared = Mock(return_value={"state": "COMPLETED", "data": {}, "error": None})
    monkeypatch.setattr(backend, "_run_prepared_pipeline", prepared)
    factory = Mock(side_effect=AssertionError("The web host must not load cached adapters"))
    monkeypatch.setattr(backend, "_create_orchestrator", factory)
    path = tmp_path / "progress.jsonl"
    result = backend.run_pipeline_core(str(path), experiment_profile=PROFILE,
                                       use_llm_review=False, offline=True)
    assert result["state"] == "COMPLETED"
    request = prepared.call_args.args[1]
    assert request["experiment_profile"] == PROFILE and request["offline"] is True
    assert request["use_llm_review"] is False
    assert request["progress_path"] == str(path.resolve())
    assert old_backend.run_pipeline_core is old_core and old_core.__globals__ is old_globals
    factory.assert_not_called()


def test_managed_worker_does_not_delegate_recursively(monkeypatch, tmp_path):
    backend = entrypoint.load_repository_backend()
    prepared = Mock(side_effect=AssertionError("An owned worker must not restart itself"))
    monkeypatch.setattr(backend, "_run_prepared_pipeline", prepared)
    orchestrator = SimpleNamespace(run=Mock(return_value={"state": "COMPLETED", "error": None, "data": {}}))
    monkeypatch.setattr(backend, "_create_orchestrator", Mock(return_value=orchestrator))
    result = backend.run_pipeline_core(str(tmp_path / "progress.jsonl"),
                                       experiment_profile=PROFILE, _managed_runtime=True)
    assert result["state"] == "COMPLETED"
    assert orchestrator.run.call_args.args[0]["experiment_profile"] == PROFILE
    prepared.assert_not_called()


def test_owned_worker_rejects_replaced_pdf_before_creating_orchestrator(monkeypatch, tmp_path):
    backend = entrypoint.load_repository_backend()
    path = write_pdf(tmp_path / "replaced.pdf", ["Unrelated replacement paper", "Abstract", CLAIM + AUTHOR])
    factory = Mock(side_effect=AssertionError("Unproven PDF route must not start reproduction"))
    monkeypatch.setattr(backend, "_create_orchestrator", factory)
    result = backend.run_pipeline_core(str(tmp_path / "progress.jsonl"), pdf_path=str(path),
                                       experiment_profile=PROFILE, _managed_runtime=True)
    assert result["state"] == "ERROR" and "未共同确认" in result["error"]
    factory.assert_not_called()


@pytest.mark.parametrize("explicit", [{}, {"experiment_profile": PROFILE}])
def test_background_pdf_route_uses_latest_repository_backend_without_starting_a_process(monkeypatch, tmp_path, explicit):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", CLAIM + AUTHOR])
    previous = Mock()
    latest = SimpleNamespace(run_pipeline_background=Mock(return_value="owned-worker-placeholder"))
    monkeypatch.setattr(entrypoint, "load_repository_backend", Mock(return_value=latest))
    dispatch = entrypoint.background_entrypoint(previous)
    assert dispatch("progress", pdf_path=str(path), mock_mode=False, **explicit) == "owned-worker-placeholder"
    request = latest.run_pipeline_background.call_args.kwargs
    assert request["experiment_profile"] == PROFILE and request["pdf_path"] == str(path)
    if not explicit:
        assert request["pdf_resolution"]["evidence"]["url"] == AUTHOR
    previous.assert_not_called()


@pytest.mark.parametrize("explicit", [{"experiment_profile": "dlinear_etth1_smoke"},
                                       {"mock_mode": True},
                                       {"experiment_profile": PROFILE, "mock_mode": True}])
def test_background_explicit_presets_and_mock_keep_the_previous_owner(monkeypatch, tmp_path, explicit):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", CLAIM + AUTHOR])
    previous = Mock(return_value="existing-owner")
    latest = Mock(side_effect=AssertionError("Existing route must keep its owner"))
    monkeypatch.setattr(entrypoint, "load_repository_backend", latest)
    assert entrypoint.background_entrypoint(previous)("progress", pdf_path=str(path), **explicit) == "existing-owner"
    assert previous.call_args.kwargs == {"pdf_path": str(path), **explicit}
    latest.assert_not_called()


def test_app_upload_rezero_displays_grounded_training_route_and_runs_full_experiment_without_stale_preset_options(monkeypatch, tmp_path):
    path = write_pdf(tmp_path / "paper.pdf", [TITLE, "Abstract", CLAIM + AUTHOR])
    upload = UploadedFile(UploadedFileRec("rezero-routed-preview", "paper.pdf", "application/pdf",
                                         path.read_bytes()), FileURLs())

    def uploaded_file(*args, **kwargs):
        st.session_state[kwargs["key"]] = upload
        return upload

    capture = Mock()
    monkeypatch.setattr(st, "file_uploader", uploaded_file)
    monkeypatch.setattr(entrypoint, "background_entrypoint", lambda previous: capture)
    monkeypatch.setattr(settings, "load_local_llm_settings", Mock())
    monkeypatch.setattr(LLMClient, "chat", Mock(side_effect=AssertionError("Preview cannot call a model")))
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    monkeypatch.delenv("AUTOREPRO_RESUME_PROGRESS", raising=False)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["input_mode"] = "上传PDF"
    app.session_state["mock_mode"] = False
    app.session_state["pdf_llm_review"] = False
    app.session_state["repository_prepare_only"] = True
    app.session_state["repository_result_review"] = True
    app.session_state["method_action"] = "仅准备源码和命令"
    app.session_state["docker_probe"] = (False, "test-local-only")
    app.run()
    assert not app.exception
    assert any("PDF 原文作者代码声明已匹配实验" in item.value and AUTHOR in item.value for item in app.success)
    assert any("作者主仓库与实际训练仓库分别记录" in item.value and TRAINING in item.value for item in app.caption)
    assert not app.session_state["running"]
    assert capture.call_count == 0
    start = next(button for button in app.button if button.label == "🚀 开始复现")
    start.click().run()
    assert not app.exception
    request = capture.call_args.kwargs
    assert request["experiment_profile"] == PROFILE
    assert request["prepare_only"] is False and request["prepare_environment"] is False
    assert request["allow_result_summary_review"] is False and request["use_llm_review"] is False
    assert request["optimization_mode"] == "off"
    assert Path(request["pdf_path"]).read_bytes() == path.read_bytes()
    # The fake background owner does not clean its temporary PDF; this test does.
    Path(request["pdf_path"]).unlink()
    LLMClient.chat.assert_not_called()
