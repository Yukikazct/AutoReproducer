"""A crashed Windows owner cannot publish an unobserved tree cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from frontend import backend_pipeline as backend
from frontend import task_control


@pytest.mark.skipif(os.name != "nt", reason="Windows asynchronous kill-on-job-close")
def test_crashed_owner_keeps_environment_without_tree_cleanup_proof(tmp_path, monkeypatch):
    """Even a dead real Job owner cannot authorize deleting its dependencies.

    The OS may finish descendants before this assertion on a fast machine. The
    ownership receipt still contains no observation of their exit, so recovery
    must refuse in both schedules rather than infer completion from owner PID.
    """
    from src.process_lifecycle import _windows_api
    import src.environment_lifecycle as lifecycle

    progress = tmp_path / "orphan.jsonl"
    ready = tmp_path / "children.json"
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    sentinel = dependencies / "owned-environment.txt"
    sentinel.write_text("retain until descendants are confirmed stopped", encoding="utf-8")
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(dependencies))
    store = backend.ProgressStore(str(progress))
    store.emit({"type": "pipeline_worker", "worker_id": "crashed-owner-worker", "status": "running"})
    worker = (
        "import json,os,subprocess,sys,time;from pathlib import Path;"
        "children=[subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']) for _ in range(4)];"
        f"Path({str(ready)!r}).write_text(json.dumps([os.getpid(),*[child.pid for child in children]]));"
        "time.sleep(60)"
    )
    owner_program = """
import subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from frontend.task_control import register_task
from src.process_lifecycle import ProcessJob, CREATE_SUSPENDED
control = register_task(sys.argv[2])
job = ProcessJob()
process = subprocess.Popen([sys.executable, '-c', sys.argv[3]],
                           creationflags=CREATE_SUSPENDED | subprocess.CREATE_NO_WINDOW)
job.assign(process)
job.resume(process)
time.sleep(60)
"""
    owner = subprocess.Popen([sys.executable, "-c", owner_program,
                              str(Path(__file__).resolve().parents[1]), str(progress), worker],
                             creationflags=subprocess.CREATE_NO_WINDOW)
    api = _windows_api()
    handles = []
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if ready.exists():
                try:
                    pids = json.loads(ready.read_text(encoding="utf-8"))
                    break
                except ValueError:
                    pass
            assert owner.poll() is None, "owner exited before starting its process tree"
            time.sleep(.01)
        else:
            pytest.fail("owner process tree did not become ready")
        for pid in pids:
            handle = api.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            assert handle, f"could not capture owned process {pid}"
            handles.append(handle)

        owner.kill()
        owner.wait(timeout=10)
        # A signalled Windows process handle can precede release of its file
        # locks. Wait for the owner lock to become available; this is still
        # not evidence that its descendants have exited.
        deadline = time.monotonic() + 5
        while task_control.task_status(progress).get("active") and time.monotonic() < deadline:
            time.sleep(.01)
        assert task_control.task_status(progress).get("orphaned")
        cleanup = Mock(return_value={"status": "released", "errors": []})
        monkeypatch.setattr(lifecycle, "release_abandoned_environment", cleanup)

        assert not task_control.recover_orphaned_task(str(progress), backend.ProgressStore)
        cleanup.assert_not_called()
        assert sentinel.is_file()
        assert backend.ProgressStore.read_snapshot(str(progress))["running"]
        assert task_control.task_status(progress)["orphaned"]
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=10)
        try:
            for handle in handles:
                assert api.WaitForSingleObject(handle, 10000) == 0, "owned descendant survived owner death"
        finally:
            for handle in handles:
                api.CloseHandle(handle)
