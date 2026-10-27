"""Exercise UI cancellation against silent, real installer process trees."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock

from filelock import FileLock
import pytest

from frontend import backend_pipeline as backend
from frontend import task_control
from src import runtime_preparation as runtime


@pytest.fixture(autouse=True)
def isolate_failed_root_registry(monkeypatch, tmp_path):
    # Deliberate cleanup-failure tests must not block unrelated pytest cases.
    monkeypatch.setattr(task_control, "_blocked_roots", {})
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(tmp_path / "dependencies"))


def wait_for(predicate, timeout=12):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.05)
    pytest.fail("condition did not become true before timeout")


def assert_stopped(pid):
    if os.name == "nt":
        from src.process_lifecycle import _windows_api
        api = _windows_api()
        handle = api.OpenProcess(0x00100000, False, pid)
        if handle:
            try:
                assert api.WaitForSingleObject(handle, 3000) == 0
            finally:
                api.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        stat = Path(f"/proc/{pid}/stat")
        assert stat.is_file() and stat.read_text().split()[2] == "Z"


@pytest.mark.parametrize("native_owner", [False, True])
def test_owner_lock_contends_with_existing_filelock_across_processes(tmp_path, native_owner):
    lock_path = tmp_path / "owner.lock"
    owner = task_control._OwnerLock(lock_path) if native_owner else FileLock(str(lock_path))
    program = """
import sys
sys.path.insert(0, sys.argv[1])
from frontend.task_control import _OwnerLock, _OwnerLockBusy
from filelock import FileLock, Timeout
lock = FileLock(sys.argv[2]) if sys.argv[3] == 'filelock' else _OwnerLock(sys.argv[2])
try:
    with lock.acquire(timeout=0):
        print('free')
except (Timeout, _OwnerLockBusy):
    print('owned')
"""
    command = [sys.executable, "-c", program, str(Path(__file__).resolve().parents[1]),
               str(lock_path), "filelock" if native_owner else "native"]
    with owner.acquire(timeout=0):
        blocked = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert blocked.returncode == 0, blocked.stderr
        assert blocked.stdout.strip() == "owned"
    released = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert released.returncode == 0, released.stderr
    assert released.stdout.strip() == "free"


def installer_command(ready):
    return [sys.executable, "-c", (
        "import subprocess,sys,os,time,json;from pathlib import Path;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        f"Path({str(ready)!r}).write_text(json.dumps([os.getpid(),child.pid]));"
        "time.sleep(60)")]


def test_cancel_during_runtime_dependency_download_stops_tree_and_releases_lock(tmp_path, monkeypatch):
    progress, ready = tmp_path / "progress.jsonl", tmp_path / "pids.json"
    cache = tmp_path / "bootstrap"
    cache.mkdir()
    downloads = cache / "wheels"
    downloads.mkdir()
    (downloads / "incomplete.whl").write_text("partial")
    unrelated = tmp_path / "other-environment"
    unrelated.mkdir()
    (unrelated / ".ready").write_text("keep")
    lock_path = cache / "prepare.lock"
    entered = []

    def prepare(*args, cancel_check, progress_callback, **kwargs):
        entered.append(True)
        with FileLock(str(lock_path)), runtime._cleanup_failed_preparation(cache):
            progress_callback({"type": "dependency_install_progress", "installation_id": "pip-runtime",
                               "status": "running", "phase": "download", "message": "silent pip fixture"})
            runtime.run_owned_process(installer_command(ready), timeout_s=None, cancel_check=cancel_check)
            pytest.fail("a cancelled installer must not become ready")

    monkeypatch.setattr(runtime, "prepare_runtime", prepare)
    thread = backend.run_pipeline_background(str(progress), paper_title="cancellation fixture")
    wait_for(ready.is_file)
    pids = json.loads(ready.read_text())
    try:
        assert task_control.request_cancel(str(progress))
        assert task_control.task_status(str(progress))["cancelling"]
        thread.join(timeout=10)
        assert not thread.is_alive()
        for pid in pids:
            assert_stopped(pid)
    finally:
        task_control.request_cancel(str(progress))
        thread.join(timeout=10)
    snapshot = backend.ProgressStore.read_snapshot(str(progress))
    assert not snapshot["running"] and snapshot["state"] == "ERROR"
    assert snapshot["result"]["data"]["run_status"] == "interrupted"
    assert snapshot["dependency_installations"][0]["status"] == "cancelled"
    assert not downloads.exists()
    assert (unrelated / ".ready").read_text() == "keep"
    assert len(entered) == 1
    with FileLock(str(lock_path)).acquire(timeout=0):
        pass
    assert not task_control.task_status(str(progress))["active"]
    assert "silent pip fixture" in progress.read_text(encoding="utf-8")


def test_cancel_while_worker_installer_is_silent_closes_job_before_done(tmp_path, monkeypatch):
    progress, ready = tmp_path / "progress.jsonl", tmp_path / "pids.json"
    prepared = runtime.RuntimePreparation(sys.executable, True, False, "fixture", "fixture")
    monkeypatch.setattr(runtime, "prepare_runtime", Mock(return_value=prepared))
    original = runtime.run_owned_process
    cleaned = []

    def execute(argv, **kwargs):
        # This process deliberately ignores the cooperative request. The owner
        # must still stop the silent installer and its descendant within 5 s.
        request = json.loads(kwargs["input"])
        store = backend.ProgressStore(request["progress_path"], reset=False)
        store.emit({"type": "execution_run", "execution_id": "install-run", "status": "running"})
        store.emit({"type": "dependency_install_progress", "installation_id": "pip-worker",
                    "execution_id": "install-run", "status": "running", "phase": "download"})
        kwargs.pop("input")
        return original(installer_command(ready), **kwargs)

    def cleanup(_):
        for pid in json.loads(ready.read_text()):
            assert_stopped(pid)
        assert backend.ProgressStore.read_snapshot(str(progress))["running"]
        cleaned.append(True)
        return {"status": "released", "errors": []}

    monkeypatch.setattr(runtime, "run_owned_process", execute)
    monkeypatch.setattr(runtime, "release_worker_environment", cleanup)
    thread = backend.run_pipeline_background(str(progress), paper_title="silent installer")
    wait_for(ready.is_file)
    assert task_control.request_cancel(str(progress))
    thread.join(timeout=12)
    assert not thread.is_alive()
    assert cleaned == [True]
    snapshot = backend.ProgressStore.read_snapshot(str(progress))
    assert not snapshot["running"]
    assert snapshot["active_runs"] == snapshot["active_workers"] == snapshot["active_executions"] == []
    assert snapshot["dependency_installations"][0]["status"] == "cancelled"
    assert snapshot["result"]["data"]["environment_cleanup"]["status"] == "released"


def test_hung_pdf_parser_runs_in_owned_worker_and_can_be_stopped(tmp_path, monkeypatch):
    import src.pdf_input as pdf_input
    progress, ready = tmp_path / 'parse.jsonl', tmp_path / 'pids.json'
    source = tmp_path / 'paper.pdf'
    source.write_bytes(b'%PDF-parser-fixture')
    host_parser = Mock(side_effect=AssertionError('PDF parsing must be owned by the cancellable worker'))
    monkeypatch.setattr(pdf_input, 'resolve_pdf_request', host_parser)
    monkeypatch.setattr(runtime, 'prepare_runtime', Mock(return_value=runtime.RuntimePreparation(
        sys.executable, True, False, 'fixture', 'fixture')))
    script = tmp_path / 'hung_parser.py'
    script.write_text(
        "import ctypes,json,os,runpy,subprocess,sys\nfrom pathlib import Path\n"
        "import src.pdf_input as pdf_input\n"
        "def blocked(request):\n"
        "    child=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)'])\n"
        f"    Path({str(ready)!r}).write_text(json.dumps([os.getpid(),child.pid]))\n"
        "    ctypes.windll.kernel32.Sleep(60000) if os.name == 'nt' else ctypes.CDLL(None).sleep(60)\n"
        "    raise AssertionError('parser should have been stopped')\n"
        "pdf_input.resolve_pdf_request=blocked\n"
        "runpy.run_module('frontend.preset_worker',run_name='__main__')\n", encoding='utf-8')
    original = runtime.run_owned_process
    def execute(argv, **kwargs):
        program = "import sys,runpy;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2],run_name='__main__')"
        return original([sys.executable, '-c', program, str(Path(backend.__file__).parents[1]), str(script)], **kwargs)
    monkeypatch.setattr(runtime, 'run_owned_process', execute)
    cleanup = Mock(return_value={'status': 'released', 'errors': []})
    monkeypatch.setattr(runtime, 'release_worker_environment', cleanup)
    thread = backend.run_pipeline_background(str(progress), pdf_path=str(source), cleanup_pdf=False)
    try:
        wait_for(ready.is_file)
        host_parser.assert_not_called()
        assert task_control.request_cancel(progress)
        thread.join(timeout=12)
        assert not thread.is_alive()
        for pid in json.loads(ready.read_text()):
            assert_stopped(pid)
        snapshot = backend.ProgressStore.read_snapshot(str(progress))
        assert not snapshot['running']
        assert snapshot['active_workers'] == snapshot['active_runs'] == []
        assert snapshot['result']['data']['run_status'] == 'interrupted'
        assert not task_control.task_status(progress)['active']
        cleanup.assert_called_once()
        # Once cleanup is confirmed a new task can use the same progress slot.
        next_task = task_control.register_task(progress)
        next_task.finish()
    finally:
        task_control.request_cancel(progress)
        thread.join(timeout=12)


def test_old_token_and_unknown_task_cannot_cancel_new_owner(tmp_path):
    path = tmp_path / "progress.jsonl"
    old = task_control.register_task(path)
    old_token = old.token
    old.finish()
    current = task_control.register_task(path)
    try:
        current.cancel_file.write_text(old_token, encoding="ascii")
        assert not current.cancelled()
        assert not task_control.request_cancel(tmp_path / "unowned.jsonl")
        assert not current.cancelled()
        assert task_control.request_cancel(path)
        assert current.cancelled()
    finally:
        current.finish()


def test_cancellation_worker_unwinds_silent_installer_before_force_deadline(tmp_path, monkeypatch):
    progress, ready, receipt = tmp_path / "progress.jsonl", tmp_path / "pids.json", tmp_path / "cleaned.json"
    lock_path = tmp_path / "installer.lock"
    script = tmp_path / "cooperative_worker.py"
    script.write_text(
        "import json,sys\nfrom pathlib import Path\nfrom filelock import FileLock\n"
        "from src.process_lifecycle import watch_cancellation_file\n"
        "from src.runtime_preparation import run_owned_process\n"
        "from frontend.backend_pipeline import ProgressStore\n"
        "request=json.loads(sys.stdin.read())\nstore=ProgressStore(request['progress_path'],reset=False)\n"
        "try:\n"
        "    with watch_cancellation_file(request['_cancel_file'],request['_cancel_token']):\n"
        f"        with FileLock({str(lock_path)!r}):\n"
        f"            run_owned_process({installer_command(ready)!r},timeout_s=None)\n"
        "except KeyboardInterrupt:\n"
        f"    Path({str(receipt)!r}).write_text('interrupted')\n"
        "    store.emit({'type':'done','result':{'state':'ERROR','error':'cancelled','data':{'run_status':'interrupted'}}})\n"
        "    sys.exit(130)\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "prepare_runtime", Mock(return_value=runtime.RuntimePreparation(
        sys.executable, True, False, "fixture", "fixture")))
    original = runtime.run_owned_process

    def execute(argv, **kwargs):
        # Import with an explicit package path because the owned runner strips
        # PYTHONPATH from the environment, just as it does for production work.
        program = "import sys,runpy;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2],run_name='__main__')"
        return original([sys.executable, "-c", program, str(Path(backend.__file__).parents[1]), str(script)], **kwargs)

    def cleanup(_):
        assert receipt.read_text() == "interrupted"
        with FileLock(str(lock_path)).acquire(timeout=0):
            pass
        for pid in json.loads(ready.read_text()):
            assert_stopped(pid)
        return {"status": "released", "errors": []}

    monkeypatch.setattr(runtime, "run_owned_process", execute)
    monkeypatch.setattr(runtime, "release_worker_environment", cleanup)
    thread = backend.run_pipeline_background(str(progress), paper_title="cooperative installer")
    wait_for(ready.is_file)
    started = time.monotonic()
    assert task_control.request_cancel(str(progress))
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert time.monotonic() - started < 5
    assert receipt.read_text() == "interrupted"
    assert not backend.ProgressStore.read_snapshot(str(progress))["running"]


def test_cleanup_failure_blocks_reset_until_retry_succeeds(tmp_path, monkeypatch):
    progress = tmp_path / "progress.jsonl"
    control = task_control.register_task(progress)
    backend.ProgressStore(str(progress)).emit({"type": "done", "result": {
        "state": "ERROR", "error": "cleanup failed", "report_path": "existing-report.md",
        "data": {"environment_cleanup": {"status": "error", "errors": [{"reason": "busy"}]}}}})
    control.finish(cleanup_failed=True)
    assert task_control.task_status(progress)["cleanup_failed"]
    import src.environment_lifecycle as lifecycle
    cleanup = Mock(return_value={"status": "error", "errors": [{"reason": "busy"}]})
    monkeypatch.setattr(lifecycle, "release_abandoned_environment", cleanup)
    assert not task_control.recover_orphaned_task(str(progress), backend.ProgressStore)
    assert task_control.task_status(progress)["cleanup_failed"]
    cleanup.return_value = {"status": "released", "errors": []}
    assert task_control.recover_orphaned_task(str(progress), backend.ProgressStore)
    assert not task_control.task_status(progress)["cleanup_failed"]
    assert backend.ProgressStore.read_snapshot(str(progress))["result"]["report_path"] == "existing-report.md"


def test_orphaned_receipt_never_authorizes_cleanup_from_owner_pid_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.environment_lifecycle as lifecycle
    import src.process_lifecycle as process_lifecycle
    progress = tmp_path / "orphan.jsonl"
    control = task_control.register_task(progress)
    control.finish()
    record = json.loads(control.record.read_text())
    record["state"] = "running"  # The service died before writing its terminal receipt.
    control.record.write_text(json.dumps(record))
    store = backend.ProgressStore(str(progress))
    store.emit({"type": "pipeline_worker", "worker_id": "owned-worker", "status": "running"})
    store.emit({"type": "execution_run", "execution_id": "owned-run", "status": "running"})
    other = tmp_path / "other.jsonl"
    backend.ProgressStore(str(other)).emit({"type": "pipeline_worker", "worker_id": "other-worker", "status": "running"})
    other_bytes = other.read_bytes()
    cleanup = Mock(return_value={"status": "released", "errors": []})
    monkeypatch.setattr(lifecycle, "release_abandoned_environment", cleanup)
    api = SimpleNamespace(OpenProcess=Mock(return_value=123), WaitForSingleObject=Mock(return_value=258), CloseHandle=Mock())
    monkeypatch.setattr(process_lifecycle, "_windows_api", lambda: api)
    assert not task_control.recover_orphaned_task(str(progress), backend.ProgressStore)
    cleanup.assert_not_called()
    assert backend.ProgressStore.read_snapshot(str(progress))["running"]
    api.WaitForSingleObject.return_value = 0
    assert not task_control.recover_orphaned_task(str(progress), backend.ProgressStore)
    cleanup.assert_not_called()
    snapshot = backend.ProgressStore.read_snapshot(str(progress))
    assert snapshot["running"]
    assert snapshot["active_workers"] and snapshot["active_runs"]
    assert other.read_bytes() == other_bytes
    with pytest.raises(RuntimeError, match="恢复原任务"):
        task_control.register_task(tmp_path / "new.jsonl")


def test_legacy_progress_without_owner_is_never_recovered_by_guessing(tmp_path):
    progress = tmp_path / "legacy.jsonl"
    backend.ProgressStore(str(progress)).emit({"type": "pipeline_worker", "worker_id": "unknown", "status": "running"})
    before = progress.read_bytes()
    assert not task_control.recover_orphaned_task(str(progress), backend.ProgressStore)
    assert progress.read_bytes() == before


def test_preparation_cancel_retains_failed_download_cleanup_receipt(tmp_path, monkeypatch):
    progress = tmp_path / "failed-cleanup.jsonl"
    receipt = {"status": "error", "errors": [{"path": "wheels", "reason": "access denied"}]}
    monkeypatch.setattr(runtime, "_release_runtime_downloads_locked", Mock(return_value=receipt))

    def prepare(*args, **kwargs):
        with runtime._cleanup_failed_preparation(tmp_path):
            raise KeyboardInterrupt("stop during pip")

    monkeypatch.setattr(runtime, "prepare_runtime", prepare)
    thread = backend.run_pipeline_background(str(progress), paper_title="cleanup fixture")
    thread.join(timeout=10)
    assert not thread.is_alive()
    snapshot = backend.ProgressStore.read_snapshot(str(progress))
    assert snapshot["result"]["data"]["environment_cleanup"] == receipt
    assert task_control.task_status(progress)["cleanup_failed"]
    assert not task_control.task_status(progress)["cleanup_unconfirmed"]
    assert json.loads(progress.with_suffix(".control.json").read_text())["state"] == "cleanup_failed"


def test_failed_job_close_preserves_cancellation_and_blocks_cleanup_and_restart(tmp_path, monkeypatch):
    from types import SimpleNamespace
    progress = tmp_path / "failed-job.jsonl"
    close = Mock(side_effect=OSError("native close failed"))
    monkeypatch.setattr(runtime, "ProcessJob", lambda: SimpleNamespace(assign=Mock(), resume=Mock(), close=close))
    process = SimpleNamespace(pid=123, stdin=None, poll=Mock(return_value=None), kill=Mock(), wait=Mock())
    spawn = Mock(return_value=process)
    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    release = Mock()
    monkeypatch.setattr(runtime, "_release_runtime_downloads_locked", release)
    confirmed = Mock()

    def prepare(*args, **kwargs):
        checks = iter([False, True])
        with runtime._cleanup_failed_preparation(tmp_path):
            runtime.run_owned_process(["fixture-installer"], timeout_s=None,
                                      cancel_check=lambda: next(checks), on_cleanup=confirmed)

    monkeypatch.setattr(runtime, "prepare_runtime", prepare)
    thread = backend.run_pipeline_background(str(progress), paper_title="native cleanup fixture")
    thread.join(timeout=10)
    assert not thread.is_alive()
    spawn.assert_called_once()
    confirmed.assert_not_called()
    release.assert_not_called()
    status = task_control.task_status(progress)
    assert status["cleanup_unconfirmed"] and status["cleanup_failed"]
    snapshot = backend.ProgressStore.read_snapshot(str(progress))
    assert "回收未确认" in snapshot["error"]
    assert snapshot["result"]["data"]["environment_cleanup"]["process_cleanup_confirmed"] is False
    # A failed close in the still-live service must not be treated as a dead
    # Windows owner merely because the supervisor thread released its lock.
    assert not task_control.recover_orphaned_task(str(progress), backend.ProgressStore)


@pytest.mark.parametrize("forced", [False, True])
def test_posix_worker_force_stop_never_claims_nested_session_cleanup(tmp_path, monkeypatch, forced):
    progress = tmp_path / "posix-worker.jsonl"
    monkeypatch.setattr(backend, "_force_stop_contains_worker_children", lambda: False)
    monkeypatch.setattr(backend, "WORKER_CANCEL_GRACE_SECONDS", 0)
    monkeypatch.setattr(runtime, "prepare_runtime", Mock(return_value=runtime.RuntimePreparation(
        sys.executable, True, False, "fixture", "fixture")))
    release = Mock(return_value={"status": "released", "errors": []})
    monkeypatch.setattr(runtime, "release_worker_environment", release)

    def worker(argv, **kwargs):
        request = json.loads(kwargs["input"])
        store = backend.ProgressStore(request["progress_path"], reset=False)
        store.emit({"type": "execution_run", "execution_id": "nested-install", "status": "running"})
        assert task_control.request_cancel(progress)
        if forced:
            assert kwargs["cancel_check"]()
            kwargs["on_cleanup"]()  # Only the outer process group was reaped.
            raise KeyboardInterrupt("forced outer worker stop")
        store.emit({"type": "execution_run", "execution_id": "nested-install", "status": "interrupted",
                    "cleanup_confirmed": True})
        store.emit({"type": "done", "result": {"state": "ERROR", "error": "cancelled", "data": {}}})
        kwargs["on_cleanup"]()
        return subprocess.CompletedProcess(argv, 130, "", "")

    monkeypatch.setattr(runtime, "run_owned_process", worker)
    thread = backend.run_pipeline_background(str(progress), paper_title="posix cleanup fixture")
    thread.join(timeout=10)
    assert not thread.is_alive()
    snapshot = backend.ProgressStore.read_snapshot(str(progress))
    status = task_control.task_status(progress)
    if forced:
        release.assert_not_called()
        assert status["cleanup_unconfirmed"] and status["cleanup_failed"]
        assert snapshot["running"] and snapshot["active_workers"] and snapshot["active_runs"]
        assert not task_control.recover_orphaned_task(str(progress), backend.ProgressStore)
    else:
        release.assert_called_once()
        assert not status["cleanup_failed"] and not snapshot["running"]


@pytest.mark.parametrize("unconfirmed", [False, True])
def test_failed_root_blocks_same_and_new_progress_until_recovery(tmp_path, monkeypatch, unconfirmed):
    progress = tmp_path / "old.jsonl"
    control = task_control.register_task(progress)
    backend.ProgressStore(str(progress)).emit({"type": "error", "error": "cleanup fixture"})
    control.finish(cleanup_failed=True, process_cleanup_confirmed=not unconfirmed)
    saved = control.record.read_bytes()
    for target in (progress, tmp_path / "new.jsonl", tmp_path / "another-session" / "new.jsonl"):
        with pytest.raises(RuntimeError, match="清理|回收"):
            task_control.register_task(target)
    assert control.record.read_bytes() == saved
    if unconfirmed:
        assert not task_control.recover_orphaned_task(str(progress), backend.ProgressStore)
        with pytest.raises(RuntimeError, match="清理|回收"):
            task_control.register_task(tmp_path / "still-blocked.jsonl")
    else:
        import src.environment_lifecycle as lifecycle
        monkeypatch.setattr(lifecycle, "release_abandoned_environment", Mock(return_value={"status": "released", "errors": []}))
        assert task_control.recover_orphaned_task(str(progress), backend.ProgressStore)
        new = task_control.register_task(tmp_path / "now-safe.jsonl")
        new.finish()


def test_disk_failed_receipt_blocks_another_service_without_overwriting(tmp_path, monkeypatch):
    progress = tmp_path / "old.jsonl"
    control = task_control.register_task(progress)
    control.finish(cleanup_failed=True)
    saved = control.record.read_bytes()
    monkeypatch.setattr(task_control, "_blocked_roots", {})  # A fresh service's memory.
    with pytest.raises(RuntimeError, match="恢复原任务"):
        task_control.register_task(tmp_path / "new.jsonl")
    assert control.record.read_bytes() == saved
    control.record.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="无法确认"):
        task_control.register_task(tmp_path / "new.jsonl")


@pytest.mark.parametrize("process_cleanup_confirmed", [True, False])
def test_finish_write_failure_releases_owner_and_retains_durable_root_gate(
        tmp_path, monkeypatch, process_cleanup_confirmed):
    progress = tmp_path / "old-session" / "progress.jsonl"
    control = task_control.register_task(progress)
    monkeypatch.setattr(control, "_save", Mock(side_effect=PermissionError("receipt is locked")))
    try:
        with pytest.raises(PermissionError, match="receipt is locked"):
            control.finish(process_cleanup_confirmed=process_cleanup_confirmed)
        assert control.finished
        with task_control._OwnerLock(task_control._paths(progress)[1]).acquire(timeout=0):
            pass
        assert task_control._key(progress) not in task_control._tasks
        status = task_control.task_status(progress)
        assert not status["active"] and status["cleanup_failed"]
        assert status["cleanup_unconfirmed"] is (not process_cleanup_confirmed)
        monkeypatch.setattr(task_control, "_blocked_roots", {})  # A fresh service's memory.
        with pytest.raises(RuntimeError, match="恢复原任务"):
            task_control.register_task(tmp_path / "new-session" / "progress.jsonl")
    finally:
        # Keep the unpatched implementation from leaking its deliberately stuck
        # owner lock when this regression is run against the original defect.
        control.lock.release()
        task_control._tasks.pop(task_control._key(progress), None)


@pytest.mark.parametrize("state", ["running", "cleanup_failed", "cleanup_unconfirmed"])
def test_shared_root_gate_survives_a_service_exit_and_different_progress_directory(tmp_path, state):
    original = tmp_path / "old-service" / "old.jsonl"
    next_progress = tmp_path / "new-service" / "new.jsonl"
    project = str(Path(__file__).resolve().parents[1])
    program = """
import os,sys
sys.path.insert(0,sys.argv[1])
from frontend.task_control import register_task
owner=register_task(sys.argv[2])
state=sys.argv[3]
if state != 'running':
    owner.finish(cleanup_failed=True,process_cleanup_confirmed=state != 'cleanup_unconfirmed')
os._exit(0)
"""
    previous = subprocess.run([sys.executable, "-X", "utf8", "-c", program, project, str(original), state],
                              capture_output=True, text=True, timeout=10)
    assert previous.returncode == 0, previous.stderr
    with pytest.raises(RuntimeError, match="恢复原任务"):
        unexpected = task_control.register_task(next_progress)
        unexpected.finish()
    assert not next_progress.with_suffix(".control.json").exists()


def test_same_root_rejects_another_service_while_owner_is_active(tmp_path):
    original = tmp_path / "old-service" / "old.jsonl"
    control = task_control.register_task(original)
    program = """
import sys
sys.path.insert(0,sys.argv[1])
from frontend.task_control import register_task
try:
    owner=register_task(sys.argv[2])
except RuntimeError as exc:
    print(str(exc))
else:
    owner.finish()
    sys.exit(2)
"""
    try:
        attempt = subprocess.run([sys.executable, "-X", "utf8", "-c", program,
                                  str(Path(__file__).resolve().parents[1]),
                                  str(tmp_path / "new-service" / "new.jsonl")],
                                 capture_output=True, text=True, encoding="utf-8", timeout=10)
        assert attempt.returncode == 0, attempt.stderr
        assert "依赖目录" in attempt.stdout
        assert task_control.task_status(original)["active"]
    finally:
        control.finish()


def test_failed_finish_can_recover_and_start_from_another_progress_directory(tmp_path, monkeypatch):
    import src.environment_lifecycle as lifecycle
    original = tmp_path / "old-session" / "old.jsonl"
    control = task_control.register_task(original)
    backend.ProgressStore(str(original)).emit({"type": "done", "result": {
        "state": "COMPLETED", "report_path": "retained-report.md", "data": {"evidence": "retained"}}})
    monkeypatch.setattr(control, "_save", Mock(side_effect=PermissionError("locked progress receipt")))
    with pytest.raises(PermissionError):
        control.finish()
    release = Mock(return_value={"status": "in_use", "errors": []})
    monkeypatch.setattr(lifecycle, "release_abandoned_environment", release)
    assert not task_control.recover_orphaned_task(original, backend.ProgressStore)
    assert control.shared_record.exists(), "a busy environment is not a cleanup confirmation"
    release.return_value = {"status": "released", "errors": []}
    assert task_control.recover_orphaned_task(original, backend.ProgressStore)
    assert not control.shared_record.exists()
    snapshot = backend.ProgressStore.read_snapshot(str(original))
    assert snapshot["result"]["report_path"] == "retained-report.md"
    assert snapshot["result"]["data"]["evidence"] == "retained"
    new = task_control.register_task(tmp_path / "new-session" / "new.jsonl")
    new.finish()


def test_both_receipt_writes_failing_keeps_preexisting_running_gate(tmp_path, monkeypatch):
    progress = tmp_path / "original" / "progress.jsonl"
    control = task_control.register_task(progress)
    saved = control.shared_record.read_bytes()
    monkeypatch.setattr(control, "_save", Mock(side_effect=PermissionError("local receipt locked")))
    monkeypatch.setattr(control, "_save_shared", Mock(side_effect=PermissionError("shared receipt locked")))
    with pytest.raises(PermissionError, match="local receipt locked") as raised:
        control.finish()
    assert "shared receipt locked" in " ".join(raised.value.__notes__)
    assert control.shared_record.read_bytes() == saved
    with task_control._OwnerLock(task_control._paths(progress)[1]).acquire(timeout=0):
        pass
    monkeypatch.setattr(task_control, "_blocked_roots", {})
    with pytest.raises(RuntimeError, match="恢复原任务"):
        task_control.register_task(tmp_path / "different-directory" / "new.jsonl")
    assert not task_control.recover_orphaned_task(progress, backend.ProgressStore)


def test_corrupt_shared_receipt_blocks_another_directory_without_overwriting(tmp_path, monkeypatch):
    progress = tmp_path / "original" / "progress.jsonl"
    control = task_control.register_task(progress)
    control.finish(cleanup_failed=True)
    control.shared_record.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(task_control, "_blocked_roots", {})
    with pytest.raises(RuntimeError, match="无法确认"):
        task_control.register_task(tmp_path / "another-directory" / "new.jsonl")
    assert control.shared_record.read_text(encoding="utf-8") == "{broken"


def test_recovery_rechecks_generation_after_owner_locks_are_acquired(tmp_path, monkeypatch):
    import src.environment_lifecycle as lifecycle
    progress = tmp_path / "progress.jsonl"
    control = task_control.register_task(progress)
    control.finish(cleanup_failed=True)
    previous = task_control.task_status(progress)
    changed = dict(previous["record"], token="f" * 32)
    task_control._write_record(control.record, changed)
    task_control._write_record(control.shared_record, changed)
    original_record = control.record.read_bytes()
    original_shared = control.shared_record.read_bytes()
    backend.ProgressStore(str(progress)).emit({"type": "done", "result": {"state": "COMPLETED", "data": {}}})
    original_progress = progress.read_bytes()
    monkeypatch.setattr(task_control, "task_status", lambda path: previous)
    release = Mock(return_value={"status": "released", "errors": []})
    monkeypatch.setattr(lifecycle, "release_abandoned_environment", release)
    assert not task_control.recover_orphaned_task(progress, backend.ProgressStore)
    release.assert_not_called()
    assert control.record.read_bytes() == original_record
    assert control.shared_record.read_bytes() == original_shared
    assert progress.read_bytes() == original_progress


def test_registration_receipt_failure_releases_root_admission_and_created_lease(tmp_path, monkeypatch):
    original = task_control.TaskControl._save
    monkeypatch.setattr(task_control.TaskControl, "_save", Mock(side_effect=PermissionError("cannot save owner")))
    with pytest.raises(PermissionError, match="cannot save owner"):
        task_control.register_task(tmp_path / "failed" / "old.jsonl")
    monkeypatch.setattr(task_control.TaskControl, "_save", original)
    control = task_control.register_task(tmp_path / "retry" / "new.jsonl")
    control.finish()


def test_finished_control_is_idempotent_without_reintroducing_a_root_gate(tmp_path):
    control = task_control.register_task(tmp_path / "progress.jsonl")
    control.finish()
    control.finish()
    assert not control.shared_record.exists()
    replacement = task_control.register_task(tmp_path / "new-directory" / "progress.jsonl")
    replacement.finish()


def test_shared_gate_keeps_unrelated_dependency_roots_independent(tmp_path, monkeypatch):
    blocked = task_control.register_task(tmp_path / "failed.jsonl")
    blocked.finish(cleanup_failed=True)
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(tmp_path / "other-dependencies"))
    unrelated = task_control.register_task(tmp_path / "separate-session" / "new.jsonl")
    unrelated.finish()
    assert task_control.task_status(blocked.path)["cleanup_failed"]


def test_recovery_preserves_pending_worker_report_and_evidence(tmp_path, monkeypatch):
    import src.environment_lifecycle as lifecycle
    progress = tmp_path / "pending-report.jsonl"
    control = task_control.register_task(progress)
    store = backend.ProgressStore(str(progress))
    store.emit({"type": "pipeline_worker", "worker_id": "worker", "status": "running"})
    store.emit({"type": "done", "result": {"state": "COMPLETED", "report_path": "saved-report.md",
                "session_id": "saved-session", "data": {"evidence": ["saved-artifact"]}}})
    control.finish(cleanup_failed=True)
    assert backend.ProgressStore.read_snapshot(str(progress))["result"] is None
    monkeypatch.setattr(lifecycle, "release_abandoned_environment",
                        Mock(return_value={"status": "released", "errors": []}))
    assert task_control.recover_orphaned_task(progress, backend.ProgressStore)
    recovered = backend.ProgressStore.read_snapshot(str(progress))
    assert not recovered["running"] and recovered["active_workers"] == []
    assert recovered["result"]["report_path"] == "saved-report.md"
    assert recovered["result"]["session_id"] == "saved-session"
    assert recovered["result"]["data"]["evidence"] == ["saved-artifact"]


@pytest.mark.skipif(os.name != "nt", reason="Windows denies replacing an open receipt")
@pytest.mark.parametrize("locked_receipt", ["record", "shared_record", "shared_record_delete"])
def test_real_locked_receipt_finishes_ownership_and_recovers_after_unlock(tmp_path, monkeypatch, locked_receipt):
    import src.environment_lifecycle as lifecycle
    progress = tmp_path / "original" / "progress.jsonl"
    control = task_control.register_task(progress)
    target = "shared_record" if locked_receipt == "shared_record_delete" else locked_receipt
    with getattr(control, target).open("r", encoding="utf-8"):
        with pytest.raises(PermissionError):
            control.finish(cleanup_failed=locked_receipt == "shared_record")
    assert control.finished
    with task_control._OwnerLock(task_control._paths(progress)[1]).acquire(timeout=0):
        pass
    assert task_control.task_status(progress)["cleanup_failed"]
    monkeypatch.setattr(task_control, "_blocked_roots", {})
    with pytest.raises(RuntimeError, match="恢复原任务"):
        task_control.register_task(tmp_path / "another-session" / "new.jsonl")
    monkeypatch.setattr(lifecycle, "release_abandoned_environment",
                        Mock(return_value={"status": "released", "errors": []}))
    assert task_control.recover_orphaned_task(progress, backend.ProgressStore)
    replacement = task_control.register_task(tmp_path / "another-session" / "new.jsonl")
    replacement.finish()


def test_slow_runtime_download_checks_cancel_between_read1_chunks(tmp_path, monkeypatch):
    class StreamingResponse:
        reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "https://www.python.org/ftp/python/fixture.exe"

        def read(self, size):
            pytest.fail("buffer-filling read can block cancellation on a slow stream")

        def read1(self, size):
            self.reads += 1
            return b"one short network chunk"

    response = StreamingResponse()
    urlopen = Mock(return_value=response)
    monkeypatch.setattr(runtime.urllib.request, "urlopen", urlopen)
    destination = tmp_path / "python.exe"
    with pytest.raises(KeyboardInterrupt):
        runtime._download_installer(destination, cancel_check=lambda: response.reads >= 2,
                                    deadline=time.monotonic() + 60)
    assert response.reads == 2
    urlopen.assert_called_once()
    assert urlopen.call_args.kwargs["timeout"] <= 10
    assert list(tmp_path.iterdir()) == []
