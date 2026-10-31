"""Unreadable owner records can detach monitoring without proving cleanup."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontend import task_control
from frontend.backend_pipeline import ProgressStore
from src.project_paths import task_data_scope


@pytest.fixture
def damaged_task(tmp_path, monkeypatch):
    data = tmp_path / "data"
    deps = data / "deps"
    monkeypatch.setenv("AUTOREPRO_DATA_ROOT", str(data))
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(deps))
    monkeypatch.setattr(task_control, "_tasks", {})
    monkeypatch.setattr(task_control, "_blocked_roots", {})
    progress = data / "runtime" / "progress_12345.jsonl"
    control = task_control.register_task(progress)
    store = ProgressStore(str(progress))
    store.emit({"type": "state", "state": "BUILD_ENV", "status": "running"})
    store.emit({"type": "pipeline_worker", "worker_id": "unknown-worker", "status": "running"})
    control.finish(cleanup_failed=True)
    original_record = json.loads(control.record.read_text(encoding="utf-8"))
    control.record.write_bytes(b"\x00" * 327)
    control.shared_record.write_bytes(b"\x00" * 327)
    # A fresh service must still honor the durable gate without its registry.
    monkeypatch.setattr(task_control, "_blocked_roots", {})
    sentinels = [deps / "old-packages" / "sentinel",
                 data / "training_python" / "scratch" / "old-training",
                 data / "runtime" / "old-runtime" / "wheels" / "old.whl"]
    for path in sentinels:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old environment must be preserved")
    return data, progress, control, original_record, sentinels


@pytest.mark.parametrize("local,shared", [
    (b"\x00" * 327, b"\x00" * 327), (b"{truncated", b"{truncated"),
    (None, b"\x00" * 327), (b"\x00" * 327, None),
])
def test_invalid_records_are_a_cleanup_gate_instead_of_an_unmanaged_task(damaged_task, local, shared):
    _, progress, control, _, _ = damaged_task
    for path, content in ((control.record, local), (control.shared_record, shared)):
        if content is None:
            path.unlink()
        else:
            path.write_bytes(content)
    status = task_control.task_status(progress)
    assert status["managed"] and status["record_invalid"]
    assert status["cleanup_failed"] and status["cleanup_unconfirmed"]
    assert status["isolated_restart_available"] and not status["active"]
    assert "record" not in status  # No PID/token is invented from corrupt bytes.


def test_live_local_owner_lock_refuses_detachment(damaged_task):
    data, progress, _, _, _ = damaged_task
    with task_control._OwnerLock(task_control._paths(progress)[1]).acquire(timeout=0):
        status = task_control.task_status(progress)
        assert status["record_invalid"] and status["active"]
        assert not status["isolated_restart_available"]
        with pytest.raises(RuntimeError, match="仍被持有|仍被后台持有"):
            task_control.create_isolated_restart(progress)
    assert not (data / "recovery").exists()


def test_stale_available_status_cannot_bypass_a_new_owner_lock(damaged_task, monkeypatch):
    data, progress, _, _, _ = damaged_task
    stale = task_control.task_status(progress)
    monkeypatch.setattr(task_control, "task_status", lambda path: stale)
    with task_control._OwnerLock(task_control._paths(progress)[1]).acquire(timeout=0):
        with pytest.raises(RuntimeError, match="仍被后台持有"):
            task_control.create_isolated_restart(progress)
    assert not (data / "recovery").exists()


def test_new_root_preserves_old_evidence_and_does_not_release_its_gate(damaged_task, monkeypatch):
    import src.environment_lifecycle as lifecycle

    data, progress, control, _, sentinels = damaged_task
    evidence = [progress, control.record, control.shared_record, *sentinels]
    before = {path: path.read_bytes() for path in evidence}
    cleanup = Mock(side_effect=AssertionError("unknown old processes cannot authorize cleanup"))
    monkeypatch.setattr(lifecycle, "release_abandoned_environment", cleanup)
    assert not task_control.recover_orphaned_task(progress, ProgressStore)
    isolated = Path(task_control.create_isolated_restart(progress))
    assert isolated.parent == data / "recovery"
    marker = json.loads((isolated / ".restart.json").read_text(encoding="utf-8"))
    assert marker["version"] == 1 and marker["original_cleanup"] == "unconfirmed"
    assert marker["progress_path"] == task_control._key(progress)
    assert {path: path.read_bytes() for path in evidence} == before
    snapshot = ProgressStore.read_snapshot(str(progress))
    assert snapshot["running"] and not snapshot["done"]
    with pytest.raises(RuntimeError, match="无法确认"):
        task_control.register_task(data / "another-directory" / "progress_999.jsonl")
    new_progress = isolated / "runtime" / "progress_555.jsonl"
    with task_data_scope(isolated):
        new_control = task_control.register_task(new_progress, deps_root=isolated / "deps")
        assert Path(new_control.deps_root) == isolated / "deps"
        new_control.finish()
    assert not task_control.task_status(new_progress)["active"]
    assert {path: path.read_bytes() for path in evidence} == before
    cleanup.assert_not_called()


def test_detachment_rechecks_a_restored_generation_after_taking_the_owner_lock(damaged_task, monkeypatch):
    data, progress, control, previous_record, _ = damaged_task
    stale = task_control.task_status(progress)
    monkeypatch.setattr(task_control, "task_status", lambda path: stale)
    original_acquire = task_control._OwnerLock.acquire
    changed = {**previous_record, "token": "f" * 32, "state": "cleanup_unconfirmed"}

    def acquire_then_restore(self, timeout=0):
        acquired = original_acquire(self, timeout=timeout)
        if self.path == str(task_control._paths(progress)[1]):
            task_control._write_record(control.record, changed)
            task_control._write_record(control.shared_record, changed)
        return acquired

    monkeypatch.setattr(task_control._OwnerLock, "acquire", acquire_then_restore)
    with pytest.raises(RuntimeError, match="已恢复"):
        task_control.create_isolated_restart(progress)
    assert task_control._read_task_record(progress) == changed
    assert not (data / "recovery").exists()


def test_independent_corrupt_sessions_reserve_distinct_roots(damaged_task):
    data, progress, control, _, _ = damaged_task
    other = data / "other-runtime" / "progress_777.jsonl"
    other.parent.mkdir()
    other.write_bytes(b"retained old progress")
    other_record = task_control._paths(other)[0]
    other_shared = task_control._root_paths(data / "deps", other)[2]
    other_record.write_bytes(b"{broken")
    other_shared.write_bytes(b"{broken")
    retained = {path: path.read_bytes() for path in (progress, control.record, control.shared_record,
                                                  other, other_record, other_shared)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        roots = list(pool.map(task_control.create_isolated_restart, [progress, other]))
    assert len(set(roots)) == 2
    assert all(Path(root).parent == data / "recovery" for root in roots)
    assert {path: path.read_bytes() for path in retained} == retained


@pytest.mark.skipif(os.name != "nt", reason="Windows realpath retains its prefix after a 3-to-2 error transition")
def test_reservation_stabilizes_parent_before_windows_realpath(damaged_task, monkeypatch):
    import ntpath

    data, progress, _, _, _ = damaged_task
    identifier = "a1" * 16
    expected = data / "recovery" / identifier
    control = data / "unstable-parent" / identifier
    original = ntpath._getfinalpathname
    transitions = []

    def parent_appears_after_missing_path(path):
        try:
            return original(path)
        except OSError as exc:
            target = Path(path)
            if target in (control, expected) and not target.parent.exists():
                assert exc.winerror == 3  # Missing parent, before another service creates it.
                target.parent.mkdir()
                transitions.append(target)
            raise  # The following realpath probe now gets missing leaf error 2.

    monkeypatch.setattr(ntpath, "_getfinalpathname", parent_appears_after_missing_path)
    # Establish the real OS failure deterministically instead of racing threads.
    assert not control.resolve().is_relative_to(data.resolve())
    assert transitions == [control]
    monkeypatch.setattr(task_control.uuid, "uuid4", lambda: SimpleNamespace(hex=identifier))
    assert Path(task_control.create_isolated_restart(progress)) == expected
    # A missing recovery parent could have triggered the same false rejection.
    assert transitions == [control]
    assert (expected / ".restart.json").is_file()
