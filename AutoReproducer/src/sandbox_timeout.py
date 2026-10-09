"""Standalone supervisor copied into Docker; requires only the standard library."""
import json
import os
import signal
import subprocess
import sys
import time

MARKER = "AUTOREPRO_PHASE_RESULT:"


class ParentWatch:
    """Track an existing parent, retaining a Windows handle to avoid PID reuse."""
    def __init__(self, pid):
        self.pid, self.handle = pid, None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            self.kernel.OpenProcess.restype = wintypes.HANDLE
            self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            self.kernel.WaitForSingleObject.restype = wintypes.DWORD
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.kernel.CloseHandle.restype = wintypes.BOOL
            self.handle = self.kernel.OpenProcess(0x00100000, False, pid)

    def alive(self):
        if os.name == "nt":
            return bool(self.handle) and self.kernel.WaitForSingleObject(self.handle, 0) == 258
        return os.getppid() == self.pid

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.handle:
            self.kernel.CloseHandle(self.handle)


def supervised(command, limit, parent_pid):
    """An orphan supervisor kills its entire experiment tree, including grandchildren."""
    with ParentWatch(parent_pid) as parent:
        if not parent.alive():
            return 130, False
        process = subprocess.Popen(command)
        deadline = time.monotonic() + limit
        while process.poll() is None:
            abandoned = not parent.alive()
            timed_out = time.monotonic() >= deadline
            if abandoned or timed_out:
                code = 130 if abandoned else 124
                print(MARKER + json.dumps({"phase": "execution", "timeout": timed_out and not abandoned,
                      "cancelled": abandoned, "seconds": limit, "returncode": code}), file=sys.stderr, flush=True)
                # RepositoryRunner starts this supervisor in a private POSIX session.
                # On Windows taskkill needs the supervisor alive to find descendants.
                if os.name == "posix":
                    os.killpg(os.getpid(), signal.SIGKILL)
                else:
                    subprocess.run(["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                                   capture_output=True, timeout=5)
                process.kill()
                process.wait(timeout=5)
                return code, timed_out and not abandoned
            time.sleep(.1)
        return process.returncode, False


def main(argv=None):
    phase, seconds, *command = list(sys.argv[1:] if argv is None else argv)
    limit = float(seconds)
    if command[:1] == ["--start-gate"]:
        gate, command = command[1], command[2:]
        deadline = time.monotonic() + 10
        while not os.path.exists(gate):
            if time.monotonic() >= deadline:
                return 125
            time.sleep(.01)
    try:
        owner = os.environ.get("AUTOREPRO_PARENT_PID")
        if owner and phase == "execution":
            code, timed_out = supervised(command, limit, int(owner))
        else:
            result = subprocess.run(command, timeout=limit)
            code = result.returncode
            timed_out = False
    except subprocess.TimeoutExpired:
        code = 124
        timed_out = True
    if code:
        # Preserve normal stdout/stderr. The host removes this structured marker
        # and explains whether installation or actual program execution failed.
        print(MARKER + json.dumps({"phase": phase, "timeout": timed_out,
                                  "seconds": limit, "returncode": code}),
              file=sys.stderr, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
