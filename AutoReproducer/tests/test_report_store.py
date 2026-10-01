"""Unselected reports disappear; explicitly saved reports survive cleanup."""
import os
import time
from pathlib import Path

import pytest

from frontend.backend_pipeline import ProgressStore
from frontend import history_manager, report_store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(history_manager, "get_project_data_dir", lambda: tmp_path)
    return tmp_path


def result():
    return {"session_id": "20261001_090000", "report_path": "", "report_saved": False,
            "data": {"paper_title": "Example", "report": "# A complete report",
                     "execution": {"code": "print('real')", "final": {
                         "stdout": "full stdout tail", "stderr": "full stderr tail"}}}}


def done(path, data):
    store = ProgressStore(str(path))
    store.emit({"type": "done", "result": data})


def test_save_is_explicit_idempotent_and_includes_attachment(isolated):
    data = result()
    assert not (isolated / "reports").exists()
    path = Path(report_store.save_report(data))
    assert data["report_saved"] is True and path.is_file()
    assert "full stdout tail" in path.with_name(path.stem + "_execution.txt").read_text()
    assert report_store.save_report(data) == str(path)
    assert len(list((isolated / "reports").iterdir())) == 2
    report_store.discard_unsaved_report(data)
    assert path.exists() and data["data"]["report"]


def test_discard_clears_unsaved_text_and_temporary_disk_copy(isolated):
    data = result()
    progress = isolated / "runtime/progress_1.jsonl"
    done(progress, data)
    report_store.discard_unsaved_report(data, str(progress))
    assert "report" not in data["data"]
    assert not progress.exists()
    assert not (isolated / "reports").exists()


def test_expired_progress_cleanup_preserves_active_fresh_and_saved(isolated):
    data = result()
    saved = Path(report_store.save_report(data))
    old = isolated / "runtime/progress_old.jsonl"
    fresh = old.with_name("progress_fresh.jsonl")
    active = old.with_name("progress_active.jsonl")
    done(old, data)
    done(fresh, data)
    ProgressStore(str(active)).emit({"type": "state", "state": "EXECUTE_CODE"})
    for path in (old, active):
        os.utime(path, (time.time() - 7200, time.time() - 7200))
    assert report_store.cleanup_abandoned_progress() == 1
    assert not old.exists() and fresh.exists() and active.exists()
    assert saved.exists()


def test_expiration_timer_deletes_only_its_finished_copy(isolated, monkeypatch):
    progress = isolated / "runtime/progress_timer.jsonl"
    done(progress, result())
    timers = []
    class Timer:
        def __init__(self, interval, function, args):
            self.interval, self.function, self.args = interval, function, args
            timers.append(self)
        def start(self):
            pass
    monkeypatch.setattr(report_store.threading, "Timer", Timer)
    report_store.schedule_progress_cleanup(str(progress))
    timer = timers[0]
    assert timer.interval == 3600 and timer.daemon is True
    timer.function(*timer.args)
    assert not progress.exists()
    done(progress, result())
    report_store.schedule_progress_cleanup(str(progress))
    previous = timers[-1]
    os.utime(progress, ns=(progress.stat().st_atime_ns, progress.stat().st_mtime_ns + 10_000_000))
    previous.function(*previous.args)
    assert progress.exists()  # a newer run must survive the older timer


def test_deleting_saved_history_also_deletes_execution_attachment(isolated):
    data = result()
    path = Path(report_store.save_report(data))
    history_manager.delete_session(data["session_id"])
    assert not path.exists()
    assert not path.with_name(path.stem + "_execution.txt").exists()


def test_invalid_session_cannot_write_outside_report_directory(isolated):
    data = result()
    data["session_id"] = "../../outside"
    with pytest.raises(ValueError):
        report_store.save_report(data)
    assert not (isolated / "reports").exists()
