"""Current web routes stay on local author execution, including old sessions."""
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from src.base_agent import BaseAgent
from frontend import backend_pipeline, history_manager

APP_PATH = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture
def app_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(history_manager, "get_project_data_dir", lambda: tmp_path)
    probe = Mock(return_value=(True, None))
    monkeypatch.setattr(BaseAgent, "docker_engine_available", staticmethod(probe))

    def make(input_mode="论文标题", *, mock_mode=False):
        app = AppTest.from_file(APP_PATH, default_timeout=30)
        app.session_state["input_mode"] = input_mode
        app.session_state["mock_mode"] = mock_mode
        # Sessions created before the route change may still request Docker.
        app.session_state["use_docker"] = True
        app.session_state["docker_probe"] = (True, None)
        return app.run()

    return make, probe


def docker_toggle(app):
    return next(toggle for toggle in app.sidebar.toggle if "Docker" in toggle.label)


@pytest.mark.parametrize("input_mode", ["论文标题", "上传PDF", "官方仓库预设"])
def test_author_inputs_disable_docker_even_when_engine_is_available(app_factory, input_mode):
    make, probe = app_factory
    app = make(input_mode)
    assert not app.exception, app.exception
    assert docker_toggle(app).disabled is True
    assert docker_toggle(app).value is False
    assert app.session_state["use_docker"] is False
    assert "本地受控进程" in " ".join(item.value for item in app.sidebar.caption)
    probe.assert_not_called()


def test_mock_mode_clears_stale_docker_selection_without_probe(app_factory):
    make, probe = app_factory
    app = make(mock_mode=True)
    assert not app.exception, app.exception
    assert docker_toggle(app).disabled is True
    assert app.session_state["use_docker"] is False
    assert any("Mock 模式不执行真实代码" in item.value for item in app.sidebar.caption)
    probe.assert_not_called()


def test_title_start_keeps_local_execution_after_stale_docker_selection(app_factory, monkeypatch):
    make, probe = app_factory
    app = make()
    start = Mock()
    monkeypatch.setattr(backend_pipeline, "run_pipeline_background", start)
    app.sidebar.text_input(key="paper_title_input").set_value("A new paper title").run()
    app.session_state["use_docker"] = True
    next(button for button in app.sidebar.button if "开始复现" in button.label).click().run()
    assert not app.exception, app.exception
    start.assert_called_once()
    assert start.call_args.kwargs["use_docker"] is False
    assert start.call_args.kwargs["author_execution_only"] is True
    probe.assert_not_called()


def test_unmatched_title_explains_pdf_requirement_and_report_action(app_factory):
    make, _ = app_factory
    app = make()
    assert any("运行作者代码需上传 PDF" in item.value for item in app.sidebar.caption)
    action = "请上传该论文的原始 PDF，再运行发现的作者仓库。"
    app.session_state["result"] = {"state": "ERROR", "error": "缺少论文正文", "data": {
        "source_request": {"status": "needs_pdf", "next_action": action},
    }}
    app.run()
    assert not app.exception, app.exception
    assert any(item.value == action for item in app.info)
