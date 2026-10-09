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


def test_repository_preset_requires_real_mode(app, monkeypatch):
    start = Mock()
    monkeypatch.setattr(pipeline, "run_pipeline_background", start)
    app.sidebar.toggle[0].set_value(True).run()
    app.sidebar.radio(key="input_mode").set_value("官方仓库预设").run()
    next(b for b in app.sidebar.button if "开始复现" in b.label).click().run()
    assert not app.exception
    start.assert_not_called()
    assert any("关闭 Mock" in error.value for error in app.sidebar.error)


def test_repository_preset_starts_without_llm_configuration(app, monkeypatch):
    start = Mock()
    monkeypatch.setattr(pipeline, "run_pipeline_background", start)
    app.sidebar.toggle[0].set_value(False).run()
    app.sidebar.radio(key="input_mode").set_value("官方仓库预设").run()
    app.sidebar.text_input(key="llm_base_url").set_value("").run()
    app.sidebar.text_input(key="llm_model").set_value("").run()
    app.sidebar.checkbox(key="repository_prepare_only").set_value(True).run()
    next(b for b in app.sidebar.button if "开始复现" in b.label).click().run()
    assert not app.exception
    start.assert_called_once()
    kwargs = start.call_args.kwargs
    assert kwargs["experiment_profile"] == "dlinear_etth1_reference"
    assert kwargs["prepare_only"] is True
    assert kwargs["mock_mode"] is False
    assert kwargs["use_docker"] is False
    assert not app.error


def test_method_environment_preparation_does_not_require_api(app, monkeypatch):
    start = Mock()
    monkeypatch.setattr(pipeline, "run_pipeline_background", start)
    app.sidebar.toggle[0].set_value(False).run()
    app.sidebar.radio(key="input_mode").set_value("官方仓库预设").run()
    app.sidebar.selectbox(key="experiment_profile").set_value("siren_camera_quick").run()
    app.sidebar.selectbox(key="method_action").set_value("准备实验环境").run()
    app.sidebar.text_input(key="llm_base_url").set_value("").run()
    app.sidebar.text_input(key="llm_model").set_value("").run()
    next(b for b in app.sidebar.button if "开始复现" in b.label).click().run()
    assert not app.exception
    start.assert_called_once()
    kwargs = start.call_args.kwargs
    assert kwargs["experiment_profile"] == "siren_camera_quick"
    assert kwargs["prepare_environment"] is True
    assert kwargs["prepare_only"] is False
    assert kwargs["use_llm_review"] is False


def test_method_success_is_not_paper_table_reproduction(app):
    app.session_state["result"] = {"state": "COMPLETED", "data": {"validation": {
        "status": "method_experiment_completed", "is_reproduced": None, "metric_records": [
            {"name": "psnr", "value": 37.36, "unit": "dB", "split": "fit"}]}}}
    app.run()
    assert not app.exception
    assert any("官方方法实验已完成" in item.value for item in app.success)
    assert not any("论文数值验收通过" in item.value for item in app.success)


def test_repository_preset_docker_is_explicitly_rejected(app, monkeypatch):
    start = Mock()
    monkeypatch.setattr(pipeline, "run_pipeline_background", start)
    app.sidebar.toggle[0].set_value(False).run()
    app.sidebar.radio(key="input_mode").set_value("官方仓库预设").run()
    app.sidebar.toggle[1].set_value(True).run()
    next(b for b in app.sidebar.button if "开始复现" in b.label).click().run()
    assert not app.exception
    start.assert_not_called()
    assert any("使用本地执行" in error.value for error in app.sidebar.error)


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


def test_numerical_mismatch_is_not_presented_as_reproduction_success(app, tmp_path):
    progress = tmp_path / "progress.jsonl"
    pipeline.ProgressStore(str(progress)).emit({"type": "done", "result": {
        "state": "COMPLETED", "data": {"validation": {
            "status": "not_reproduced", "result_level": "experiment_completed",
            "reason": "MSE超出固定容差", "is_reproduced": False,
        }},
    }})
    app.session_state["running"] = True
    app.session_state["progress_file"] = str(progress)
    app.run()
    assert not app.exception
    assert any("论文数值验收未通过" in notice.value for notice in app.warning)
    assert not any("复现流程成功完成" in notice.value for notice in app.success)


def test_running_task_rejects_reset_even_for_a_stale_button_event(app, tmp_path, monkeypatch):
    start = Mock()
    monkeypatch.setattr(pipeline, "run_pipeline_background", start)
    progress = tmp_path / "active.jsonl"
    pipeline.ProgressStore(str(progress)).emit({
        "type": "state", "state": "EXECUTE_CODE", "agent": "CodeExecutor", "status": "running"})
    app.session_state["progress_file"] = str(progress)
    app.run()
    assert app.session_state["running"] is True
    assert next(b for b in app.sidebar.button if "开始复现" in b.label).disabled
    reset = next(b for b in app.sidebar.button if "重置" in b.label)
    assert reset.disabled
    assert any("任务结束后可重置" in caption.value for caption in app.sidebar.caption)
    # Inject an already-delivered event; recent AppTest versions block disabled clicks.
    import streamlit as st
    original_button = st.button

    def stale_button(label, *args, **kwargs):
        clicked = original_button(label, *args, **kwargs)
        return True if "重置" in label else clicked

    monkeypatch.setattr(st, "button", stale_button)
    app.run()
    assert not app.exception
    assert app.session_state["running"] is True
    assert app.session_state["progress_file"] == str(progress)
    assert app.session_state["current_state"] == "EXECUTE_CODE"
    start.assert_not_called()


@pytest.mark.parametrize("terminal", ["COMPLETED", "ERROR"])
def test_terminal_task_can_reset_and_start_again(app, tmp_path, monkeypatch, terminal):
    start = Mock()
    monkeypatch.setattr(pipeline, "run_pipeline_background", start)
    progress = tmp_path / "finished.jsonl"
    pipeline.ProgressStore(str(progress)).emit({
        "type": "done", "result": {"state": terminal, "data": {}}})
    app.session_state["progress_file"] = str(progress)
    app.session_state["running"] = True
    app.run()
    reset = next(b for b in app.sidebar.button if "重置" in b.label)
    assert not reset.disabled
    reset.click().run()
    assert app.session_state["progress_file"] is None
    assert app.session_state["result"] is None
    assert app.session_state["current_state"] == "INIT"
    app.sidebar.toggle[0].set_value(True).run()
    app.sidebar.text_input(key="paper_title_input").set_value("Restart test").run()
    next(b for b in app.sidebar.button if "开始复现" in b.label).click().run()
    assert not app.exception
    start.assert_called_once()
