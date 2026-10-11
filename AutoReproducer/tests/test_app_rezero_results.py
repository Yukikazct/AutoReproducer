"""Render controlled result fixtures through the actual Streamlit application."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend import history_manager


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(history_manager, "get_project_data_dir", lambda: tmp_path)
    monkeypatch.setattr("src.local_llm_settings.load_local_llm_settings", lambda: None)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    app.session_state["docker_probe"] = (True, None)
    app.session_state["progress_file"] = None
    return app


def result(accuracy):
    passed = accuracy >= 94.0
    return {"state": "COMPLETED", "data": {
        "experiment_spec": {"adapter_id": "rezero"},
        "validation": {"status": "reproduced" if passed else "reference_not_met",
                       "is_reproduced": passed, "result_level": "reproduced" if passed else "experiment_completed",
                       "metrics_comparison": {"actual": {"top1_accuracy_pct": accuracy}}}}}


def test_below_reference_is_explicit_numerical_nonpass_without_execution_error(app):
    app.session_state["result"] = result(93.91)
    app.run()
    assert not app.exception
    assert any("独立评估数值未达到固定参考" in notice.value for notice in app.warning)
    assert any("93.91%" in caption.value and "94.00%" in caption.value for caption in app.caption)
    assert any("不等于代码执行失败" in caption.value for caption in app.caption)
    assert not any("验收通过" in notice.value for notice in app.success)
    assert not any("流程出错" in notice.value or "实验未通过" in notice.value for notice in app.error)


def test_passing_reference_is_limited_to_selected_cifar10_experiment(app):
    app.session_state["result"] = result(94.0)
    app.run()
    assert not app.exception
    assert any("ReZero CIFAR-10" in notice.value and "独立数值验收通过" in notice.value for notice in app.success)
    assert any("不覆盖 enwiki8 或整篇论文" in caption.value for caption in app.caption)


def test_unmeasured_nonpass_does_not_invent_zero_accuracy(app):
    data = result(93.91)
    data["data"]["validation"]["metrics_comparison"]["actual"] = {}
    app.session_state["result"] = data
    app.run()
    assert not app.exception
    assert any("独立评估数值未达到固定参考" in notice.value for notice in app.warning)
    assert not any("Top-1" in caption.value for caption in app.caption)
