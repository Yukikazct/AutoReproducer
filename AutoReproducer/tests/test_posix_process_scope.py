"""A wedged worker and nested managed groups must exit before reset is allowed."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from src import posix_process_scope as scopes


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX session ownership")


def _ready(path, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except ValueError:
                pass
        time.sleep(.01)
    pytest.fail("worker did not publish its process ownership")


def _worker(tmp_path, *, exit_leader=False, exit_group_leader=False):
    ready = tmp_path / "children.json"
    child = (
        "import json,os,subprocess,sys,time;from pathlib import Path;"
        "grandchild=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        f"Path({str(ready)!r}).write_text(json.dumps([os.getpid(),grandchild.pid]));"
        + ("time.sleep(.3)" if exit_group_leader else "time.sleep(60)")
    )
    code = (
        "import subprocess,sys,time;"
        f"child=subprocess.Popen([sys.executable,'-c',{child!r}],process_group=0);"
        + ("time.sleep(.3)" if exit_leader else "time.sleep(60)")
    )
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    children = _ready(ready)
    return process, children


def _cleanup(process, children):
    for group in [*children, process.pid]:
        try:
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    process.wait(timeout=5)


def _stopped(pid):
    if sys.platform.startswith("linux"):
        try:
            assert scopes._linux_process_stat(pid)[2] == "Z"
        except (FileNotFoundError, ProcessLookupError):
            pass
        return
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 1 or result.stdout.strip().startswith("Z")


def test_nested_managed_groups_are_reaped_without_touching_another_session(tmp_path):
    process, children = _worker(tmp_path)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"],
                                 start_new_session=True)
    try:
        assert os.getsid(children[0]) == process.pid
        assert os.getpgid(children[0]) != process.pid
        scope = scopes.PosixSessionScope(process)
        scope.close()
        scope.close()  # Repeated finalization must not signal any recycled IDs.
        assert process.wait(timeout=5) == -signal.SIGKILL
        for pid in children:
            _stopped(pid)
        assert unrelated.poll() is None
    finally:
        _cleanup(process, children)
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_exited_but_unreaped_leader_still_anchors_nested_group_cleanup(tmp_path):
    process, children = _worker(tmp_path, exit_leader=True)
    try:
        time.sleep(.5)
        # Do not poll/wait: Darwin returns ESRCH from getsid for this zombie,
        # while its unreaped PID still prevents an unrelated session reusing it.
        assert process.returncode is None
        scope = scopes.PosixSessionScope(process)
        scope.close()
        assert process.wait(timeout=5) == 0
        for pid in children:
            _stopped(pid)
    finally:
        _cleanup(process, children)


def test_session_query_error_never_publishes_confirmed_cleanup(tmp_path, monkeypatch):
    process, children = _worker(tmp_path)
    scope = scopes.PosixSessionScope(process)
    def denied(session_id):
        raise OSError("process query denied")
    monkeypatch.setattr(scopes, "_session_members", denied)
    try:
        with pytest.raises(OSError, match="query denied"):
            scope.close()
        assert scope.closed is False
    finally:
        _cleanup(process, children)


def test_orphaned_nested_group_is_still_found_by_its_session(tmp_path):
    process, children = _worker(tmp_path, exit_group_leader=True)
    try:
        time.sleep(.5)
        scope = scopes.PosixSessionScope(process)
        scope.close()
        for pid in children:
            _stopped(pid)
    finally:
        _cleanup(process, children)


def test_unconfirmed_stop_times_out_with_owned_children_still_protected(tmp_path, monkeypatch):
    process, children = _worker(tmp_path)
    scope = scopes.PosixSessionScope(process, timeout=.1)
    monkeypatch.setattr(scope, "_signal_member", lambda *_: None)
    try:
        with pytest.raises(TimeoutError, match="could not be frozen"):
            scope.close()
        assert scope.closed is False
        assert os.getsid(children[0]) == process.pid
    finally:
        _cleanup(process, children)


def test_own_session_and_reaped_leader_cannot_authorize_signals(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(scopes.os, "killpg", lambda *args: sent.append(args))
    with pytest.raises(OSError, match="not retained"):
        scopes.PosixSessionScope(SimpleNamespace(pid=os.getsid(0), returncode=None))
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    process.wait(timeout=5)
    with pytest.raises(OSError, match="not retained"):
        scopes.PosixSessionScope(process)
    assert sent == []


def test_termination_must_wait_for_non_zombie_members_to_disappear(tmp_path, monkeypatch):
    process, children = _worker(tmp_path)
    scope = scopes.PosixSessionScope(process, timeout=.2)
    try:
        with monkeypatch.context() as replacement:
            replacement.setattr(scopes.os, "killpg", lambda *_: None)
            with pytest.raises(TimeoutError, match="still has live processes"):
                scope.close()
        assert scope.closed is False
    finally:
        _cleanup(process, children)


def test_nested_options_require_the_actual_worker_session(monkeypatch):
    monkeypatch.delenv(scopes.SESSION_ENV, raising=False)
    assert scopes.managed_process_options() == {"start_new_session": True}
    monkeypatch.setenv(scopes.SESSION_ENV, str(os.getsid(0) + 1))
    assert scopes.managed_process_options() == {"start_new_session": True}
    monkeypatch.setenv(scopes.SESSION_ENV, str(os.getsid(0)))
    assert scopes.managed_process_options() == {"start_new_session": False, "process_group": 0}


def test_failed_ps_command_cannot_be_mistaken_for_an_empty_session(monkeypatch):
    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="unavailable")
    monkeypatch.setattr(scopes.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        scopes._ps_session_members(os.getsid(0))


def test_linux_proc_parser_handles_parentheses_and_skips_unrelated_kernel_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(scopes, "_PROC_ROOT", tmp_path)
    for pid, content in ((10, "10 (odd name (nested)) S 1 10 10 0 0"),
                         (11, "11 (kernel) S 1 0 0 0 0"),
                         (12, "12 (worker) T 10 12 10 0 0")):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "stat").write_text(content)
    assert scopes._proc_session_members(10) == {10: (10, "S"), 12: (12, "T")}


def test_malformed_proc_record_is_not_an_empty_session(tmp_path, monkeypatch):
    monkeypatch.setattr(scopes, "_PROC_ROOT", tmp_path)
    directory = tmp_path / "10"
    directory.mkdir()
    (directory / "stat").write_text("not a valid process record")
    with pytest.raises((OSError, ValueError)):
        scopes._proc_session_members(10)


def test_linux_proc_permission_failure_is_not_an_empty_session(tmp_path, monkeypatch):
    monkeypatch.setattr(scopes, "_PROC_ROOT", tmp_path)
    (tmp_path / "10").mkdir()
    def denied(pid):
        raise PermissionError("owned process table unavailable")
    monkeypatch.setattr(scopes, "_linux_process_stat", denied)
    with pytest.raises(PermissionError):
        scopes._proc_session_members(10)
