"""Uploaded bytes expose repository evidence before any model or search call."""
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

import frontend.history_manager as history
import frontend.pdf_entrypoint as entrypoint
import src.local_llm_settings as settings
import src.pdf_input as current_pdf
from src.llm.llm_client import LLMClient
from test_pdf_repository_pipeline import write_repository_pdf


APP = Path(__file__).resolve().parents[1] / "app.py"
URL = "https://github.com/independent-lab/small-solver"


@pytest.fixture
def isolated_upgrade(monkeypatch):
    previous = sys.modules.pop(entrypoint._UPGRADED_MODULE, None)
    yield
    sys.modules.pop(entrypoint._UPGRADED_MODULE, None)
    if previous is not None:
        sys.modules[entrypoint._UPGRADED_MODULE] = previous


def test_current_pdf_entrypoint_preserves_current_parser_and_mocks(monkeypatch, isolated_upgrade):
    parser = Mock()
    monkeypatch.setattr(current_pdf, "extract_pdf_input", parser)
    assert entrypoint.load_pdf_input() is current_pdf
    assert entrypoint.load_pdf_input().extract_pdf_input is parser


def test_stale_pdf_module_upgrade_preserves_active_old_reader_globals(monkeypatch, tmp_path, isolated_upgrade):
    legacy = ModuleType("src.pdf_input")
    legacy.owner_state = {"running": True}
    exec("def extract_pdf_input(path):\n    return owner_state\n", vars(legacy))
    old_extract = legacy.extract_pdf_input
    monkeypatch.setitem(sys.modules, "src.pdf_input", legacy)
    current = entrypoint.load_pdf_input()
    assert current is not legacy and current.PDF_INPUT_API_VERSION == entrypoint.PDF_INPUT_API_VERSION
    assert sys.modules["src.pdf_input"] is legacy
    assert old_extract.__globals__ is vars(legacy) and old_extract("anything") is legacy.owner_state
    assert legacy.owner_state == {"running": True}
    paper = write_repository_pdf(tmp_path / "paper.pdf", [
        ["Independent paper", "Abstract"],
        ["Our source code is published at https://github.", "com/independent-lab/small-solver"],
    ])
    evidence = current.extract_pdf_input(paper).repository_links
    assert evidence[0]["url"] == URL and evidence[0]["page"] == 2
    assert evidence[0]["is_author_code"]


def test_binary_upload_preview_displays_later_page_author_link_without_model(monkeypatch, tmp_path):
    paper = write_repository_pdf(tmp_path / "paper.pdf", [
        ["Independent paper", "Abstract", "No repository on this page."],
        ["Our source code is published at https://github.", "com/independent-lab/small-solver"],
    ])
    upload = UploadedFile(UploadedFileRec("repository-preview", "input.pdf", "application/pdf",
                                         paper.read_bytes()), FileURLs())

    def uploaded_file(*args, **kwargs):
        st.session_state[kwargs["key"]] = upload
        return upload

    monkeypatch.setattr(st, "file_uploader", uploaded_file)
    monkeypatch.setattr(settings, "load_local_llm_settings", Mock())
    monkeypatch.setattr(LLMClient, "chat", Mock(side_effect=AssertionError("Preview must not call a model")))
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    monkeypatch.delenv("AUTOREPRO_RESUME_PROGRESS", raising=False)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["input_mode"] = "上传PDF"
    app.session_state["mock_mode"] = False
    app.session_state["docker_probe"] = (False, "local preview test")
    app.run()
    assert not app.exception
    assert any("PDF 原文代码声明链接" in item.value and URL in item.value for item in app.success)
    assert any("第 2 页" in item.value for item in app.caption)
    assert any("代码公开声明" in item.value for item in app.markdown)
    assert not app.session_state["running"] and app.session_state["progress_file"] is None
    LLMClient.chat.assert_not_called()
