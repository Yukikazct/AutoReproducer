"""Impossible optimization budgets are rejected before launching a task."""
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from frontend import backend_pipeline, history_manager


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(history_manager, "get_project_data_dir", lambda: tmp_path)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    app.session_state["docker_probe"] = (True, None)
    app.run()
    app.sidebar.radio(key="input_mode").set_value("官方仓库预设").run()
    app.sidebar.selectbox(key="experiment_profile").set_value("siren_camera_quick").run()
    app.sidebar.selectbox(key="method_optimization").set_value("validate").run()
    return app


def start_button(app):
    return next(button for button in app.sidebar.button if "开始复现" in button.label)


def test_forty_minute_budget_is_explained_and_never_starts(app, monkeypatch):
    start = Mock()
    monkeypatch.setattr(backend_pipeline, "run_pipeline_background", start)
    app.sidebar.number_input(key="method_budget_minutes").set_value(40).run()
    assert any("40" in notice.value and "120" in notice.value for notice in app.sidebar.warning)
    start_button(app).click().run()
    assert not app.exception
    start.assert_not_called()
    assert app.session_state["progress_file"] is None
    assert app.session_state["running"] is False
    assert app.sidebar.error
    # The application explains the rejected limit without silently increasing it.
    assert app.sidebar.number_input(key="method_budget_minutes").value == 40


@pytest.mark.parametrize("action,minutes", [("运行实验", 120), ("准备实验环境", 40)])
def test_valid_budget_or_environment_preparation_can_start(app, monkeypatch, action, minutes):
    start = Mock()
    monkeypatch.setattr(backend_pipeline, "run_pipeline_background", start)
    app.sidebar.selectbox(key="method_action").set_value(action).run()
    app.sidebar.number_input(key="method_budget_minutes").set_value(minutes).run()
    start_button(app).click().run()
    assert not app.exception
    start.assert_called_once()
    assert start.call_args.kwargs["budget_seconds"] == minutes * 60
    assert start.call_args.kwargs["prepare_environment"] is (action == "准备实验环境")


def test_old_budget_limited_result_distinguishes_baseline_from_optimization(app):
    app.session_state["result"] = {"state": "COMPLETED", "data": {
        "validation": {"status": "method_experiment_completed", "metric_records": [
            {"name": "psnr", "value": 37.362052, "unit": "dB", "split": "fit"}]},
        "optimization": {"mode": "validate", "status": "budget_insufficient", "optimized": False,
                         "reason": "剩余预算不足以预留40分钟确认阶段", "trials": [], "confirmation": []},
    }}
    app.run()
    assert not app.exception
    assert any("基线方法实验已完成" in notice.value for notice in app.success)
    assert any("参数优化验证未完成" in notice.value for notice in app.warning)
    assert any("已尝试 0 个" in item.value and "0/2" in item.value for item in app.caption)
    assert not any("已通过两个随机种子" in notice.value for notice in app.success)
