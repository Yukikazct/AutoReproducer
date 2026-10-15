"""Persisted installer activity stays measurable across browser reconnects."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from frontend.backend_pipeline import ProgressStore
from frontend import live_progress, progress_state


@pytest.fixture
def progress(tmp_path):
    path = tmp_path / "progress.jsonl"
    return ProgressStore(str(path)), lambda: ProgressStore.read_snapshot(str(path))


def install(store, **changes):
    event = {"type": "dependency_install_progress", "installation_id": "install-1",
             "execution_id": "run-1", "status": "running", "phase": "downloading",
             "package": "torch-2.5.1+cu121.whl", "downloaded_bytes": 1024,
             "total_bytes": 4096, "download_speed_bps": 512.0, "elapsed_s": 20.0,
             "timeout_s": 1800, "last_activity_at": 1000.0, "message": "正在下载依赖",
             "source_domain": "pypi.org"}
    store.emit({**event, **changes})


def test_reconnect_keeps_latest_package_measurements_and_history(progress):
    store, snapshot = progress
    install(store)
    first = snapshot()["dependency_installations"][0]
    assert first["downloaded_bytes"] == 1024
    install(store, downloaded_bytes=2048, last_activity_at=1001, elapsed_s=21)
    latest = snapshot()["dependency_installations"][0]
    assert latest["downloaded_bytes"] == 2048 and latest["last_activity_at"] == 1001
    assert len(latest["history"]) == 1  # Heartbeats do not duplicate log lines.
    install(store, package="numpy.whl", downloaded_bytes=128, total_bytes=None,
            message="正在下载下一个依赖")
    switched = snapshot()["dependency_installations"][0]
    assert switched["package"] == "numpy.whl" and switched["total_bytes"] is None
    assert switched["downloaded_bytes"] == 128 and len(switched["history"]) == 2


def test_actual_download_domain_replaces_primary_source_on_reconnect(progress):
    store, snapshot = progress
    install(store, phase="resolving")
    assert snapshot()["dependency_installations"][0]["source_domain"] == "pypi.org"
    install(store, source_domain="download.pytorch.org")
    row = snapshot()["dependency_installations"][0]
    assert row["source_domain"] == "download.pytorch.org"
    assert row["history"][-1]["source_domain"] == "download.pytorch.org"


@pytest.mark.parametrize("status", ["success", "error", "cancelled"])
def test_finished_installation_cannot_be_reopened_by_delayed_heartbeat(progress, status):
    store, snapshot = progress
    install(store)
    install(store, status=status, message="安装已结束")
    install(store, downloaded_bytes=3072)
    row = snapshot()["dependency_installations"][0]
    assert row["status"] == status and row["message"] == "安装已结束"


@pytest.mark.parametrize("owner_status, expected", [("success", "unconfirmed"), ("error", "interrupted")])
def test_execution_end_closes_only_its_missing_installation_results(progress, owner_status, expected):
    store, snapshot = progress
    store.emit({"type": "execution_run", "execution_id": "run-1", "status": "running"})
    install(store)
    install(store, installation_id="other", execution_id="run-2")
    store.emit({"type": "execution_run", "execution_id": "run-1", "status": owner_status})
    install(store)  # Even a later ID must not reopen the closed owner.
    install(store, installation_id="late")
    rows = snapshot()["dependency_installations"]
    assert [(row["installation_id"], row["status"]) for row in rows] == [
        ("install-1", expected), ("other", "running")]


@pytest.mark.parametrize("state, expected", [("COMPLETED", "unconfirmed"), ("ERROR", "interrupted")])
def test_pipeline_end_closes_orphan_progress_without_inventing_install_success(progress, state, expected):
    store, snapshot = progress
    install(store, execution_id="")
    store.emit({"type": "done", "result": {"state": state, "data": {}}})
    install(store, installation_id="delayed")
    view = snapshot()
    assert view["done"] and not view["running"]
    assert len(view["dependency_installations"]) == 1
    assert view["dependency_installations"][0]["status"] == expected


def test_installation_history_and_attempts_are_bounded(progress):
    store, snapshot = progress
    for index in range(55):
        install(store, message="记录 " + str(index))
    row = snapshot()["dependency_installations"][0]
    assert len(row["history"]) == progress_state.INSTALLATION_HISTORY_LIMIT
    assert row["history"][-1]["message"] == "记录 54"
    for index in range(15):
        install(store, installation_id=f"attempt-{index}", status="success")
    rows = snapshot()["dependency_installations"]
    assert len(rows) == progress_state.INSTALLATION_LIMIT
    assert rows[-1]["installation_id"] == "attempt-14"


def test_invalid_measurements_are_unknown_not_download_percentages(progress):
    store, snapshot = progress
    install(store, downloaded_bytes=True, total_bytes=-1, download_speed_bps=float("inf"),
            elapsed_s=float("nan"), last_activity_at="not a timestamp")
    row = snapshot()["dependency_installations"][0]
    assert all(row[key] is None for key in (
        "downloaded_bytes", "total_bytes", "download_speed_bps", "elapsed_s", "last_activity_at"))


class Display:
    def __init__(self):
        self.percentages, self.metrics, self.notices, self.errors = [], {}, [], []

    def container(self, **kwargs):
        return nullcontext()

    def expander(self, *args, **kwargs):
        return nullcontext()

    def columns(self, count):
        return [self] * count

    def metric(self, name, value):
        self.metrics[name] = value

    def progress(self, value, **kwargs):
        self.percentages.append(value)

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        self.notices.append(message)

    def text(self, message):
        pass

    markdown = caption = text


@pytest.mark.parametrize("downloaded,total,expected", [(1024, 4096, [0.25]), (1024, None, []),
                                                       (1024, 0, []), (8192, 4096, [])])
def test_ui_only_shows_measured_current_file_percentage(progress, monkeypatch, downloaded, total, expected):
    store, snapshot = progress
    install(store, downloaded_bytes=downloaded, total_bytes=total)
    monkeypatch.setattr(live_progress, "time", SimpleNamespace(time=lambda: 1010))
    display = Display()
    live_progress.render_dependency_progress(snapshot(), st_module=display)
    assert display.percentages == expected
    assert display.metrics["安装时限"] == "1800 秒"


def test_quiet_heartbeat_warns_about_absent_activity_without_claiming_failure(progress, monkeypatch):
    store, snapshot = progress
    install(store, last_activity_at=900)
    monkeypatch.setattr(live_progress, "time", SimpleNamespace(time=lambda: 1000))
    display = Display()
    live_progress.render_dependency_progress(snapshot(), st_module=display)
    assert any("100 秒未收到" in value and "尚不能据此判定失败" in value for value in display.notices)
    assert not display.errors
    install(store, status="error", message="安装超过总时限")
    failed = Display()
    live_progress.render_dependency_progress(snapshot(), st_module=failed)
    assert failed.errors == ["安装超过总时限"] and not failed.notices


def test_stale_monitor_record_does_not_claim_installer_is_alive(progress, monkeypatch):
    store, snapshot = progress
    install(store, last_activity_at=900)
    monkeypatch.setattr(live_progress, "time", SimpleNamespace(time=lambda: 1000))
    monkeypatch.setattr(live_progress, "elapsed_seconds", lambda *_args: 45)
    display = Display()
    live_progress.render_dependency_progress(snapshot(), st_module=display)
    assert any("监控记录已 45 秒未更新" in value for value in display.notices)
    assert not any("安装进程仍在运行" in value for value in display.notices)
    assert not display.errors
