"""连接按钮反馈与论文输入的交互回归测试。"""
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import frontend.backend_pipeline as pipeline
import frontend.history_manager as history
import frontend.llm_config as llm_config


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"),
                           default_timeout=30)
    at.session_state["docker_probe"] = (True, None)
    return at.run()


@pytest.mark.parametrize("ok,message", [
    (True, "连接成功：OK"),
    (False, "[LLM API Error: HTTP 401 invalid key]"),
    (False, "[LLM API Error: timed out]"),
])
def test_connection_button_shows_result_in_sidebar(app, monkeypatch, ok, message):
    probe = Mock(return_value=(ok, message))
    monkeypatch.setattr(llm_config, "test_llm_connection", probe)
    app.sidebar.toggle[0].set_value(False).run()
    app.sidebar.text_input(key="llm_base_url").set_value("http://localhost:1234").run()
    app.sidebar.text_input(key="llm_api_key").set_value("test-key").run()
    app.sidebar.text_input(key="llm_model").set_value("test-model").run()
    app.sidebar.button(key="test_llm_connection").click().run()

    assert not app.exception
    probe.assert_called_once_with("http://localhost:1234", "test-key", "test-model")
    feedback = app.sidebar.success if ok else app.sidebar.error
    assert any(message in item.value for item in feedback)
    # 输入改动后，旧连接结果应失效。
    app.sidebar.text_input(key="llm_model").set_value("another-model").run()
    assert not app.sidebar.success
    assert not app.sidebar.error


def test_pdf_mode_renders_uploader_and_switches_back(app):
    app.sidebar.text_input(key="paper_title_input").set_value("My paper").run()
    app.sidebar.radio(key="input_mode").set_value("上传PDF").run()
    assert not app.exception
    uploader = app.sidebar.get("file_uploader")
    assert len(uploader) == 1
    assert uploader[0].key == "pdf_uploader"
    assert "paper_title_input" not in [item.key for item in app.sidebar.text_input]
    app.sidebar.radio(key="input_mode").set_value("论文标题").run()
    assert not app.exception
    assert app.sidebar.text_input(key="paper_title_input").value == "My paper"
    assert not app.sidebar.get("file_uploader")


def test_pdf_mode_requires_file_even_if_title_was_entered(app, monkeypatch):
    start = Mock()
    monkeypatch.setattr(pipeline, "run_pipeline_background", start)
    app.sidebar.text_input(key="paper_title_input").set_value("Previous title").run()
    app.sidebar.radio(key="input_mode").set_value("上传PDF").run()
    next(b for b in app.sidebar.button if "开始复现" in b.label).click().run()
    assert not app.exception
    start.assert_not_called()
    assert any("请先上传PDF文件" in error.value for error in app.sidebar.error)


@pytest.mark.parametrize("terminal", ["COMPLETED", "ERROR"])
def test_terminal_progress_updates_controls_in_same_render(app, tmp_path, terminal):
    progress = tmp_path / "progress.jsonl"
    store = pipeline.ProgressStore(str(progress))
    if terminal == "COMPLETED":
        store.emit({"type": "done", "result": {"state": "COMPLETED", "data": {}}})
    else:
        store.emit({"type": "error", "error": "test failure"})
    app.session_state["running"] = True
    app.session_state["current_state"] = "VALIDATE"
    app.session_state["progress_file"] = str(progress)
    app.run()
    assert not app.exception
    assert app.session_state["running"] is False
    assert app.session_state["current_state"] == terminal
    assert next(b for b in app.sidebar.button if "开始复现" in b.label).disabled is False
    assert terminal in " ".join(m.value for m in app.sidebar.markdown)
