"""Bound local experiment children to their owner and propagate termination."""
import _thread
import ctypes
import os
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


CREATE_SUSPENDED = 0x00000004


@contextmanager
def termination_signals():
    """Turn catchable termination signals into the normal cancellation path."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"termination signal {signum}")
        # SIGINT/SIGHUP are registered here as well so that a CLI run unwinds the
        # same cancellation path no matter which signal arrived; signals absent on
        # this platform (SIGHUP on Windows) resolve to None and are skipped.
        for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.signal(sig, interrupt)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _windows_api():
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    for name, args, result in (
        ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        ("WaitForSingleObject", [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ("CreateToolhelp32Snapshot", [wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
        ("Process32FirstW", [wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
        ("Process32NextW", [wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
        ("Thread32First", [wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
        ("Thread32Next", [wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
        ("OpenThread", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        ("ResumeThread", [wintypes.HANDLE], wintypes.DWORD),
        ("QueryFullProcessImageNameW", [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                       ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
    ):
        function = getattr(api, name)
        function.argtypes, function.restype = args, result
    return api


def _process_image(api, pid):
    from ctypes import wintypes
    handle = api.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if api.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return os.path.normcase(os.path.abspath(buffer.value))
    finally:
        api.CloseHandle(handle)
    return None


def _is_app_execution_alias(path):
    try:
        return getattr(Path(path).lstat(), "st_reparse_tag", 0) == 0x8000001B
    except FileNotFoundError:
        return False


def validate_windows_venv():
    """Reject broker-backed venvs before they can launch unowned descendants.

    A Store AppExecutionAlias can activate the actual interpreter outside its
    redirector's assigned job, even when the redirector starts suspended. This
    cannot be fixed by altering timings or preserving a Python startup hook.
    """
    if os.name != "nt":
        return
    configuration = Path(sys.executable).parent.parent / "pyvenv.cfg"
    if not configuration.is_file():
        return
    settings = {}
    for line in configuration.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            settings[key.strip().lower()] = value.strip()
    if not settings.get("home"):
        return
    executable = "pythonw.exe" if Path(sys.executable).name.lower() == "pythonw.exe" else "python.exe"
    if _is_app_execution_alias(Path(settings["home"]) / executable):
        raise ValueError(
            "当前虚拟环境的 pyvenv.cfg home 指向 Windows Store 启动别名，"
            "无法保证训练子进程在取消或强杀时终止。请使用 python.org 发行版重建虚拟环境；"
            "本次未启动训练或安装依赖。")


def _session_processes(api):
    """Follow a verified Python launcher to the session that launched it."""
    from ctypes import wintypes
    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
    direct_parent = os.getppid()
    owners = [direct_parent]
    snapshot = api.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    if snapshot == ctypes.c_void_p(-1).value:
        return owners
    processes = {}
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        available = api.Process32FirstW(snapshot, ctypes.byref(entry))
        while available:
            processes[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile.lower())
            available = api.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        api.CloseHandle(snapshot)
    # A Windows venv python.exe can be a native redirector. Its filename alone
    # is indistinguishable from a real Python service: require this environment's
    # executable path, pyvenv.cfg, and an actual interpreter with a different image.
    venv_image = os.path.normcase(os.path.abspath(sys.executable))
    actual_image = _process_image(api, os.getpid())
    if (not (Path(sys.executable).parent.parent / "pyvenv.cfg").is_file()
            or actual_image is None or actual_image == venv_image):
        venv_image = None
    def redirector(pid):
        return bool(venv_image and pid in processes and processes[pid][1] in {"python.exe", "pythonw.exe"}
                    and _process_image(api, pid) == venv_image)
    parent = direct_parent
    while parent in processes:
        ancestor, executable = processes[parent]
        if ancestor in owners or not ancestor:
            break
        if executable not in {"py.exe", "pyw.exe"} and not redirector(parent):
            # A launching Python session may itself have a venv redirector.
            # Include that verified wrapper, then stop at this session boundary.
            if redirector(ancestor) and _process_image(api, parent) == actual_image:
                owners.append(ancestor)
            break
        # Only known launchers are traversed. Never follow an arbitrary IDE or
        # Python service through unrelated ancestors, and never terminate them.
        owners.append(ancestor)
        parent = ancestor
    return owners


class ProcessJob:
    """Windows closes the job handle even if the Python owner is force-killed."""
    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes
        class BasicLimits(ctypes.Structure):
            _fields_ = [("ProcessUserTimeLimit", ctypes.c_int64), ("JobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
        self.api = _windows_api()
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if self.handle and not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def resume(self, process):
        """Resume the pristine primary thread only after assignment to the job.

        Popen closes CreateProcess's thread handle. A CREATE_SUSPENDED process
        has not executed its redirector or spawned children, so its sole initial
        thread can be opened through the documented Toolhelp/OpenThread APIs.
        """
        if not self.handle:
            return
        from ctypes import wintypes
        class ThreadEntry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                        ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]
        snapshot = self.api.CreateToolhelp32Snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        threads = []
        try:
            entry = ThreadEntry()
            entry.dwSize = ctypes.sizeof(entry)
            available = self.api.Thread32First(snapshot, ctypes.byref(entry))
            while available:
                if entry.th32OwnerProcessID == process.pid:
                    threads.append(entry.th32ThreadID)
                available = self.api.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            self.api.CloseHandle(snapshot)
        if len(threads) != 1:
            raise OSError("cannot identify the suspended process's sole initial thread")
        handle = self.api.OpenThread(0x0002, False, threads[0])  # THREAD_SUSPEND_RESUME
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if self.api.ResumeThread(handle) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.api.CloseHandle(handle)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


@contextmanager
def watch_parent_session():
    """A closed Windows launching session cancels its still-running CLI child."""
    if os.name != "nt" or threading.current_thread() is not threading.main_thread():
        yield
        return
    api = _windows_api()
    handles = [api.OpenProcess(0x00100000, False, pid) for pid in _session_processes(api)]  # SYNCHRONIZE
    handles = [handle for handle in handles if handle]
    if not handles:
        # Session ownership is unknown; never guess that a live user cancelled.
        yield
        return
    stopped = threading.Event()
    def watch():
        while not stopped.wait(.1):
            if any(api.WaitForSingleObject(handle, 0) == 0 for handle in handles):
                _thread.interrupt_main()
                return
    watcher = threading.Thread(target=watch, name="experiment-session", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stopped.set()
        watcher.join(timeout=1)
        for handle in handles:
            api.CloseHandle(handle)
