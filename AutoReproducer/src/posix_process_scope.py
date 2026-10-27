"""Own a private POSIX worker session without depending on installed packages.

Application runners inside a worker use new process groups in the worker's
session. The session leader must remain an unreaped child until ``close`` has
confirmed that every non-zombie member exited.
"""
import os
import math
from pathlib import Path
import signal
import subprocess
import sys
import time


SESSION_ENV = "AUTOREPRO_WORKER_SESSION"
_PROC_ROOT = Path("/proc")


def managed_process_options():
    """Keep nested managed runners inside their worker's private session."""
    if os.name != "posix":
        return {"start_new_session": False}
    value = os.environ.get(SESSION_ENV)
    if value and value == str(os.getsid(0)):
        return {"start_new_session": False, "process_group": 0}
    return {"start_new_session": True}


def _ps_session_members(session_id):
    # Darwin's ps ``sess`` field is a hidden kernel pointer and commonly zero.
    # Query getsid for each PID instead of treating that field as a session ID.
    completed = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,stat="], capture_output=True,
        text=True, encoding="ascii", errors="strict", timeout=2, check=True)
    members = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            raise OSError("cannot parse POSIX process ownership query")
        pid, group = int(fields[0]), int(fields[1])
        if pid < 0 or group < 0 or not fields[2]:
            raise OSError("invalid POSIX process ownership query")
        if pid == 0:
            continue
        try:
            if os.getsid(pid) == session_id:
                # Re-read the group: a running process may have called setpgid
                # between the process table snapshot and its session lookup.
                actual_group = os.getpgid(pid)
                if actual_group <= 0:
                    raise OSError("owned POSIX process has no private process group")
                members[pid] = (actual_group, fields[2])
        except ProcessLookupError:
            continue
    return members


def _linux_process_stat(pid):
    raw = (_PROC_ROOT / str(pid) / "stat").read_text(encoding="utf-8", errors="surrogateescape")
    # Linux comm is parenthesized but may itself contain spaces or parentheses.
    # Fields after the final ')' start with state, ppid, pgrp and session.
    prefix, separator, tail = raw.rpartition(")")
    fields = tail.split()
    if (not separator or int(prefix.split("(", 1)[0].strip()) != pid
            or len(fields) < 4 or len(fields[0]) != 1):
        raise OSError("cannot parse Linux process ownership query")
    group, session_id = int(fields[2]), int(fields[3])
    if group < 0 or session_id < 0:
        raise OSError("invalid Linux process ownership query")
    return group, session_id, fields[0]


def _proc_session_members(session_id):
    members = {}
    for directory in _PROC_ROOT.iterdir():
        if not directory.name.isdecimal():
            continue
        pid = int(directory.name)
        try:
            group, owner_session, state = _linux_process_stat(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if owner_session == session_id:
            if group <= 0:
                raise OSError("owned Linux process has no private process group")
            members[pid] = (group, state)
    return members


def _session_members(session_id):
    if sys.platform.startswith("linux"):
        return _proc_session_members(session_id)
    return _ps_session_members(session_id)


def _leader_is_zombie(pid):
    if sys.platform.startswith("linux"):
        return _linux_process_stat(pid)[2] == "Z"
    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="], capture_output=True,
        text=True, encoding="ascii", errors="strict", timeout=2, check=True)
    return completed.stdout.strip().startswith("Z")


class PosixSessionScope:
    """Freeze a private session, kill its groups, and verify actual exit.

    Query errors, timeouts and a reaped/replaced leader remain errors. Callers
    must retain their cleanup gate when this operation does not return.
    """
    def __init__(self, process, *, timeout=5):
        if os.name != "posix":
            raise OSError("POSIX session ownership requires POSIX")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("POSIX cleanup timeout must be positive")
        self.process = process
        self.session_id = process.pid
        self.timeout = timeout
        self.closed = False
        self._leader_exited = False
        self._validate_owner()

    def _validate_owner(self):
        pid = self.session_id
        if (not isinstance(pid, int) or pid <= 0 or pid == os.getsid(0)
                or self.process.returncode is not None):
            raise OSError("POSIX worker session leader is not retained by its owner")
        try:
            private = os.getsid(pid) == pid and os.getpgid(pid) == pid
        except ProcessLookupError:
            # Darwin's getsid returns ESRCH for an unreaped zombie. The retained
            # child PID still anchors the old session and cannot have been reused.
            if not _leader_is_zombie(pid):
                raise OSError("POSIX worker session leader disappeared") from None
            self._leader_exited = True
            return
        if not private:
            raise OSError("refusing to signal a non-private POSIX worker session")

    def _members(self):
        self._validate_owner()
        return _session_members(self.session_id)

    def _signal_member(self, pid, requested):
        try:
            if os.getsid(pid) != self.session_id:
                raise OSError("POSIX worker member escaped its owned session")
            os.kill(pid, requested)
        except ProcessLookupError:
            pass

    def close(self):
        if self.closed:
            return
        self._validate_owner()
        deadline = time.monotonic() + self.timeout
        # Keep the unreaped leader as the session identity anchor. Freeze it
        # before its children so the worker cannot create or reap more runners.
        # Darwin may reject killpg for an exited leader's empty original group.
        if not self._leader_exited:
            self._signal_member(self.session_id, signal.SIGSTOP)
        stable = None
        while True:
            members = self._members()
            if self.session_id not in members and not self._leader_exited:
                raise OSError("POSIX worker session leader disappeared during cleanup")
            live = {pid: value for pid, value in members.items() if not value[1].startswith("Z")}
            for pid, (_, state) in live.items():
                if not state.startswith(("T", "t")):
                    self._signal_member(pid, signal.SIGSTOP)
            stopped = all(state.startswith(("T", "t")) for _, state in live.values())
            identity = frozenset((pid, group) for pid, (group, _) in live.items())
            if stopped and stable == identity:
                break
            stable = identity if stopped else None
            if time.monotonic() >= deadline:
                raise TimeoutError("POSIX worker session could not be frozen before cleanup")
            time.sleep(.01)

        groups = {group for group, state in members.values() if not state.startswith("Z")}
        # All live members are stopped, so they cannot fork or move groups.
        # Keep the leader's group until last to retain the ownership anchor.
        ordered = sorted(groups - {self.session_id})
        if self.session_id in groups:
            ordered.append(self.session_id)
        for group in ordered:
            self._validate_owner()
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass

        while True:
            members = self._members()
            if not any(not state.startswith("Z") for _, state in members.values()):
                self.closed = True
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("POSIX worker session still has live processes after termination")
            time.sleep(.01)
