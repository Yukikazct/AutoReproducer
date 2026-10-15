"""Observe intermediate work before the responsible phase returns."""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from frontend.backend_pipeline import ProgressStore
from frontend.live_progress import resolve_progress_run


def test_backend_flushes_logs_before_a_long_phase_returns(tmp_path, monkeypatch):
    import frontend.backend_pipeline as backend

    logged, release = threading.Event(), threading.Event()
    path = tmp_path / "live.jsonl"

    class WaitingPipeline:
        def __init__(self, logger, **kwargs):
            self.logger = logger

        def run(self, request, on_event):
            on_event({"type": "pipeline_plan", "stages": [
                {"id": "read_paper", "title": "Read paper", "agent": "PaperReader"}]})
            on_event({"type": "state", "state": "READ_PAPER", "phase_id": "read_paper",
                      "agent": "PaperReader", "status": "running"})
            self.logger.log("PaperReader", "read_batch", "RUNNING", "Reading pages 4-8")
            logged.set()
            assert release.wait(10)
            on_event({"type": "state", "state": "READ_PAPER", "phase_id": "read_paper",
                      "agent": "PaperReader", "status": "success"})
            return {"state": "COMPLETED", "data": {}}

    monkeypatch.setattr(backend, "Orchestrator", WaitingPipeline)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(backend.run_pipeline_core, str(path), mock_mode=True)
        try:
            assert logged.wait(10)
            view = ProgressStore.read_snapshot(str(path))
            assert not future.done()
            assert view["running"] and not view["done"]
            assert view["logs"][-1]["detail"] == "Reading pages 4-8"
            assert view["activities"][-1]["phase_id"] == "read_paper"
            assert view["pipeline_stages"][0]["status"] == "running"
        finally:
            release.set()
        assert future.result(timeout=10)["state"] == "COMPLETED"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sum(row["type"] == "log" for row in records) == 1


def test_audit_observer_failure_preserves_execution_log(tmp_path):
    from src.audit.audit_logger import AuditLogger
    logger = AuditLogger(str(tmp_path / "logs"), str(tmp_path / "ledger"))
    logger.on_entry = Mock(side_effect=OSError("display unavailable"))
    entry = logger.log("CodeExecutor", "install_deps", "RUNNING", "Installing numpy")
    assert logger.get_summary() == [entry]
    assert "Installing numpy" in next((tmp_path / "logs").glob("*.jsonl")).read_text(encoding="utf-8")


def test_activity_includes_runtime_preparation_and_preserves_attempt_times(tmp_path):
    path = tmp_path / "timeline.jsonl"
    store = ProgressStore(str(path))
    store.emit({"type": "runtime_preparation", "status": "running", "message": "Creating Python environment"})
    store.emit({"type": "state", "phase_id": "prepare_environment", "state": "BUILD_ENV",
                "agent": "EnvBuilder", "status": "running", "at": "2026-10-13T10:00:00"})
    store.emit({"type": "log", "log": {"agent": "EnvBuilder", "action": "install_deps",
                                        "status": "RUNNING", "detail": "Installing torch"}})
    store.emit({"type": "state", "phase_id": "prepare_environment", "state": "BUILD_ENV",
                "agent": "EnvBuilder", "status": "error", "at": "2026-10-13T10:00:20"})
    failed = ProgressStore.read_snapshot(str(path))
    assert failed["activities"][0]["message"] == "Creating Python environment"
    assert failed["activities"][2]["phase_id"] == "prepare_environment"
    assert failed["pipeline_stages"][0]["started_at"] == "2026-10-13T10:00:00"
    assert failed["pipeline_stages"][0]["finished_at"] == "2026-10-13T10:00:20"
    store.emit({"type": "state", "phase_id": "prepare_environment", "state": "BUILD_ENV",
                "agent": "EnvBuilder", "status": "running", "attempt": 2, "at": "2026-10-13T10:00:25"})
    stage = ProgressStore.read_snapshot(str(path))["pipeline_stages"][0]
    assert stage["started_at"] == "2026-10-13T10:00:25"
    assert "finished_at" not in stage


def test_installer_output_belongs_to_environment_phase_not_executor_role(tmp_path):
    path = tmp_path / "install.jsonl"
    store = ProgressStore(str(path))
    for phase, agent in [("execute_repository", "CodeExecutor"), ("prepare_dependencies", "EnvBuilder")]:
        store.emit({"type": "state", "phase_id": phase, "agent": agent,
                    "status": "running", "state": "BUILD_ENV"})
    store.emit({"type": "log", "phase_id": "prepare_dependencies", "log": {
        "agent": "CodeExecutor", "action": "install_deps", "status": "RUNNING", "detail": "Installing numpy"}})
    view = ProgressStore.read_snapshot(str(path))
    assert view["activities"][-1]["phase_id"] == "prepare_dependencies"


@pytest.fixture
def live_app(tmp_path, monkeypatch):
    import frontend.backend_pipeline as backend
    import frontend.history_manager as history

    monkeypatch.delenv("AUTOREPRO_RESUME_PROGRESS", raising=False)
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "streamlit_autorefresh", None)
    start = Mock(side_effect=AssertionError("Monitoring must never start an experiment"))
    monkeypatch.setattr(backend, "run_pipeline_background", start)
    path = tmp_path / "runtime" / "progress_12345.jsonl"
    store = ProgressStore(str(path))
    store.emit({"type": "pipeline_plan", "stages": [
        {"id": "read_paper", "title": "论文解析", "agent": "PaperReader"},
        {"id": "prepare_environment", "title": "准备依赖", "agent": "EnvBuilder"}]})
    store.emit({"type": "state", "state": "READ_PAPER", "phase_id": "read_paper",
                "agent": "PaperReader", "status": "running"})
    store.emit({"type": "log", "log": {"agent": "PaperReader", "action": "read_pages",
                                        "status": "RUNNING", "detail": "正在读取论文第 4 页"}})
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    app.session_state["docker_probe"] = (True, None)
    return app, store, path, start


def test_refresh_restores_running_task_and_displays_midphase_logs(live_app):
    app, store, path, start = live_app
    app.query_params["run"] = path.name
    app.run()
    assert not app.exception
    assert app.session_state["progress_file"] == str(path)
    assert app.session_state["running"] is True
    assert "正在读取论文第 4 页" in " ".join(row.value for row in app.text)
    assert any("每 2 秒" in row.value for row in app.caption)
    store.emit({"type": "log", "log": {"agent": "PaperReader", "action": "read_pages",
                                        "status": "RUNNING", "detail": "正在读取论文第 5 页"}})
    app.button(key="refresh_live_progress").click().run()
    assert not app.exception
    assert "正在读取论文第 5 页" in " ".join(row.value for row in app.text)
    app.selectbox(key="live_phase_filter").select("prepare_environment").run()
    assert "正在读取论文第 5 页" not in " ".join(row.value for row in app.text)
    assert any("尚未产生过程记录" in row.value for row in app.info)
    start.assert_not_called()


def test_reconnect_button_recovers_older_tasks_without_a_bookmark(live_app):
    app, _, path, start = live_app
    app.run()
    app.button(key="resume_progress").click().run()
    assert not app.exception
    assert app.session_state["progress_file"] == str(path)
    # AppTest may normalize values written through st.query_params to lists.
    assert app.query_params["run"] in (path.name, [path.name])
    assert app.session_state["running"] is True
    start.assert_not_called()


def test_busy_writer_cannot_starve_the_initial_page_render(live_app, monkeypatch):
    from itertools import count
    import frontend.live_progress as monitoring

    app, _, path, start = live_app
    changes = count()
    monkeypatch.setattr(monitoring, "progress_signature", lambda path: next(changes))
    app.query_params["run"] = path.name
    app.run(timeout=10)
    assert not app.exception
    assert "正在读取论文第 4 页" in " ".join(row.value for row in app.text)
    start.assert_not_called()


@pytest.mark.parametrize("name", ["../progress_1.jsonl", "..\\progress_1.jsonl", "C:/secret.jsonl", "api-key", "progress_2.jsonl"])
def test_monitor_url_cannot_resolve_arbitrary_or_missing_files(tmp_path, name):
    assert resolve_progress_run(tmp_path, name) is None
