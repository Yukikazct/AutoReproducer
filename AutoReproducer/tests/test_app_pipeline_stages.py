"""阶段进度必须表达执行顺序、独立核验和当前任务，不能以职业卡代替。"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import frontend.history_manager as history
from frontend.backend_pipeline import ProgressStore


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOREPRO_RESUME_PROGRESS", raising=False)
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    at.session_state["docker_probe"] = (True, None)
    return at.run()


def plan(store):
    store.emit({"type": "pipeline_plan", "pipeline": "repository", "stages": [
        {"id": "review_readiness", "title": "训练前公开证据预审", "agent": "🛡️ Verifier",
         "description": "训练前检查", "status": "waiting"},
        {"id": "execute_repository", "title": "作者完整训练与测试", "agent": "⚡ CodeExecutor",
         "description": "真实训练", "status": "waiting"},
        {"id": "verify_protocol", "title": "训练后协议核验与指标复算", "agent": "🛡️ Verifier",
         "description": "训练后检查", "status": "waiting"},
    ]})


def stage(store, phase_id, agent, status, state="VALIDATE", **kwargs):
    store.emit({"type": "state", "phase_id": phase_id, "agent": agent,
                "state": state, "status": status, **kwargs})


def load(app, path):
    app.session_state["progress_file"] = str(path)
    app.run()
    assert not app.exception
    return {row["id"]: row for row in app.session_state.pipeline_stages}


def cards(app):
    return next(item.value for item in app.markdown if 'data-stage-id="' in item.value)


def test_preflight_success_does_not_complete_final_verifier(app, tmp_path):
    path = tmp_path / "training.jsonl"
    store = ProgressStore(str(path))
    plan(store)
    stage(store, "review_readiness", "🛡️ Verifier", "success", "BUILD_ENV")
    stage(store, "execute_repository", "⚡ CodeExecutor", "running", "EXECUTE_CODE")
    view = load(app, path)
    assert view["review_readiness"]["status"] == "success"
    assert view["execute_repository"]["status"] == "running"
    assert view["verify_protocol"]["status"] == "waiting"
    html = cards(app)
    assert "\n" not in html  # 空行与缩进会使浏览器把后续卡片显示成 HTML 代码。
    assert html.index('data-stage-id="review_readiness"') < html.index('data-stage-id="execute_repository"')
    assert html.index('data-stage-id="execute_repository"') < html.index('data-stage-id="verify_protocol"')
    assert 'agent-waiting" data-stage-id="verify_protocol"' in html
    assert "阶段完成: 1/3" in app.get("progress")[0].proto.text


def test_retry_clears_previous_error_and_final_check_error_stays_separate(app, tmp_path):
    path = tmp_path / "retry.jsonl"
    store = ProgressStore(str(path))
    plan(store)
    stage(store, "review_readiness", "🛡️ Verifier", "error", "BUILD_ENV", attempt=1, reason="首次引用不完整")
    stage(store, "review_readiness", "🛡️ Verifier", "running", "BUILD_ENV", attempt=2)
    view = load(app, path)
    assert view["review_readiness"]["status"] == "running"
    assert "首次引用不完整" not in cards(app)
    stage(store, "review_readiness", "🛡️ Verifier", "success", "BUILD_ENV", attempt=2)
    stage(store, "execute_repository", "⚡ CodeExecutor", "success", "EXECUTE_CODE")
    stage(store, "verify_protocol", "🛡️ Verifier", "running")
    stage(store, "verify_protocol", "🛡️ Verifier", "error", reason="协议检查异常")
    store.emit({"type": "done", "result": {"state": "ERROR", "error": "协议检查异常"}})
    view = load(app, path)
    assert view["review_readiness"]["status"] == "success"
    assert view["verify_protocol"]["status"] == "error"
    assert 'agent-error" data-stage-id="verify_protocol"' in cards(app)
    assert app.session_state.running is False
    assert app.get("progress")[0].proto.value < 100


def test_new_progress_does_not_inherit_previous_result_or_status(app, tmp_path):
    first = tmp_path / "previous.jsonl"
    store = ProgressStore(str(first))
    plan(store)
    stage(store, "verify_protocol", "🛡️ Verifier", "success")
    store.emit({"type": "log", "log": {"agent": "Verifier", "detail": "旧任务日志"}})
    store.emit({"type": "done", "result": {"state": "COMPLETED", "data": {"report": "旧任务报告"}}})
    load(app, first)
    second = tmp_path / "current.jsonl"
    store = ProgressStore(str(second))
    plan(store)
    stage(store, "execute_repository", "⚡ CodeExecutor", "running", "EXECUTE_CODE")
    view = load(app, second)
    assert view["verify_protocol"]["status"] == "waiting"
    assert "🛡️ Verifier" not in app.session_state.agent_status
    assert app.session_state.result is None
    assert app.session_state.logs == []
    assert app.session_state.running is True
    assert "旧任务报告" not in " ".join(item.value for item in app.markdown)


def test_live_log_is_scrollable_and_optimization_entry_is_reserved(app, tmp_path):
    path = tmp_path / "output.jsonl"
    store = ProgressStore(str(path))
    output = "\n".join(f"Epoch: {index}" for index in range(300))
    store.emit({"type": "repository_output", "text": output})
    load(app, path)
    assert app.text[0].value == output
    containers = app.get("flex_container")
    assert any(row.proto.height_config.pixel_height == 240 for row in containers)
    option = app.sidebar.checkbox(key="enable_optimization")
    assert option.disabled is True and option.value is False
    assert not app.sidebar.slider
