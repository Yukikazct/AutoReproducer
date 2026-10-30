"""Prepare a compatible, owned Python runtime without modifying the host venv.

Bootstrap imports use only the standard library; dependency probes load the
declared application packages. A Store-backed host
may coordinate recovery, but every executable used for real work is checked
before it is started and is assigned to a Windows job before its first thread
is resumed. Missing app dependencies are installed only in our managed venv.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from src.process_lifecycle import CREATE_SUSPENDED, ProcessJob, validate_windows_venv
from src.posix_process_scope import PosixSessionScope, managed_process_options
from src.safety.paths import workspace_path


OWNED_PROCESS_API_VERSION = 1
SUPPORTED_VERSIONS = {(3, 11), (3, 12)}
# The ordinary installer and automatic host recovery share one production
# manifest. Resolve it from this source checkout, independently of the cwd.
APP_REQUIREMENTS = tuple(
    line.strip() for line in (Path(__file__).resolve().parents[1] / "requirements.txt").read_text("utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
)
APP_IMPORTS = tuple(
    {"Pillow": "PIL"}.get(name, name.replace("-", "_"))
    for name in (re.match(r"[A-Za-z0-9_.-]+", requirement).group() for requirement in APP_REQUIREMENTS)
)
PYTHON_RELEASE = "3.12.10"
PUBLIC_INDEXES = ("https://pypi.tuna.tsinghua.edu.cn/simple", "https://pypi.org/simple")
# Queueing, pinned wheel downloads/probes and public analysis occur before the
# experiment budget. The outer ownership timeout must allow those to finish.
PRESET_WORKER_PREPARATION_ALLOWANCE_S = 9000
_thread_locks = {}
_thread_locks_guard = threading.Lock()


class RuntimePreparationError(RuntimeError):
    """Recovery exhausted its bounded safe options."""


class ProcessCleanupError(KeyboardInterrupt):
    """Unconfirmed process cleanup must abort retries and automatic repairs."""


class RuntimePreparationTimeout(RuntimePreparationError):
    def __init__(self, message, *, stdout=None, stderr=None):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class RuntimePreparation:
    executable: str
    reused: bool
    repaired: bool
    reason: str
    fingerprint: str


def safe_runtime_diagnostic(value):
    """Keep diagnostics useful without exposing URL credentials or API keys."""
    text = str(value or "")
    def url(match):
        try:
            parsed = urllib.parse.urlsplit(match.group(0))
            return f"{parsed.scheme}://{parsed.hostname or 'redacted-host'}/[details removed]"
        except ValueError:
            return "[URL removed]"
    text = re.sub(r"https?://[^\s<>\"']+", url, text, flags=re.I)
    text = re.sub(r"(?i)(\b(?:authorization|(?:access[_-]?)?token|password|api[_-]?key)\s*[:=]\s*)"
                  r"(?:Bearer\s+|Basic\s+)?[^\s,;]+", r"\1[REDACTED]", text)
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", text)


def runtime_requirement():
    """Return why this host must delegate, or None when it is compatible."""
    if sys.implementation.name != "cpython" or sys.version_info[:2] not in SUPPORTED_VERSIONS:
        return "预设需要标准 CPython 3.11/3.12，正在自动准备兼容运行环境"
    if sys.maxsize <= 2**32:
        return "预设需要 64 位 Python，正在自动准备兼容运行环境"
    try:
        validate_windows_venv()
    except ValueError:
        return "当前 Python 来自 Windows Store，正在自动切换到可安全管理训练进程的环境"
    return _application_dependency_issue(APP_REQUIREMENTS, APP_IMPORTS)


def _application_dependency_issue(requirements, imports):
    """Check all declared bounds identically in the host and recovery probes."""
    try:
        # A missing/broken packaging install must itself trigger recovery; it
        # must never prevent this module from loading on an unprepared host.
        from packaging.requirements import Requirement
    except Exception:
        return "应用依赖缺失，正在自动修复独立运行环境"
    for value in requirements:
        try:
            requirement = Requirement(value)
            version = importlib.metadata.version(requirement.name)
            compatible = requirement.specifier.contains(version)
        except Exception:
            # Malformed metadata is an unhealthy installation too. Native
            # package imports may raise ValueError/RuntimeError/SyntaxError,
            # so recovery must not depend on one exception spelling.
            return "应用依赖缺失，正在自动修复独立运行环境"
        if not compatible:
            return "应用依赖版本不兼容，正在自动修复独立运行环境"
    try:
        for name in imports:
            importlib.import_module(name)
    except Exception:
        return "应用依赖无法加载，正在自动修复独立运行环境"
    return None


def _check(cancel_check, deadline):
    if cancel_check and cancel_check():
        raise KeyboardInterrupt("运行环境准备已取消")
    if deadline is not None and time.monotonic() >= deadline:
        raise RuntimePreparationTimeout("运行环境自动准备超时；已停止本次准备进程，可重试复用已下载的缓存")


def _clean_environment(env=None):
    environment = dict(os.environ if env is None else env)
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUTF8"] = "1"
    # PYTHONIOENCODING overrides UTF-8 mode for redirected streams. A legacy
    # Windows shell value must not disagree with our UTF-8 output decoder.
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


class _UnreapedProcessWatch:
    """Observe exit without releasing the PID that anchors an owned session."""
    def __init__(self, process):
        import select
        self.process = process
        self.queue = None
        self.exited = False
        if hasattr(select, "kqueue"):
            self.queue = select.kqueue()
            try:
                events = self.queue.control([select.kevent(
                    process.pid, filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                    fflags=select.KQ_NOTE_EXIT)], 1, 0)
                for event in events:
                    if event.flags & select.KQ_EV_ERROR:
                        raise OSError(event.data, "Could not observe the owned worker exit")
                self.exited = bool(events)
            except BaseException:
                self.queue.close()
                raise
        elif not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
            raise OSError("This POSIX host cannot retain the worker PID while observing exit")

    def running(self):
        if not self.exited:
            if self.queue is not None:
                self.exited = bool(self.queue.control(None, 1, 0))
            else:
                self.exited = os.waitid(os.P_PID, self.process.pid,
                                       os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        return not self.exited

    def close(self):
        if self.queue is not None:
            self.queue.close()


def run_owned_process(argv, *, cwd=None, env=None, input=None, timeout_s=900,
                      cancel_check=None, on_poll=None, on_cleanup=None, max_output_bytes=2*1024*1024,
                      clean_env=True, contain_session=False):
    """Run an argv with bounded polling, stdin delivery and descendant cleanup.

    Returns subprocess.CompletedProcess with UTF-8 text stdout/stderr. Input is
    delivered by a pipe (never a credential file). ``on_poll(process)`` runs
    while the process is live; ``cancel_check()`` may return True or raise
    KeyboardInterrupt. Both cancellation and timeout close the entire job.
    ``on_cleanup()`` runs only after the job is closed and the owner is reaped,
    including exceptional exits. It never confirms a failed cleanup operation.
    ``timeout_s=None`` waits without a deadline while still polling cancellation.
    ``clean_env=False`` preserves an already prepared code-execution environment.
    ``max_output_bytes=None`` retains the full output of an author program.
    ``contain_session=True`` owns every process group in a private POSIX worker
    session, including nested managed installers and repository supervisors.
    """
    if timeout_s is not None and (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                                 or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("timeout_s must be null or a positive number")
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    _check(cancel_check, deadline)
    process, job, writer = None, None, None
    session_scope, exit_watch = None, None
    session_scope_requested = contain_session and os.name == "posix"
    result_code = -1
    timeout_error = None
    payload = input.encode("utf-8") if isinstance(input, str) else input
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            job = ProcessJob()
            process = subprocess.Popen([os.fspath(arg) for arg in argv], cwd=cwd,
                                       env=_clean_environment(env) if clean_env else env,
                                       stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                                       stdout=out, stderr=err, shell=False,
                                       **({"start_new_session": True} if contain_session and os.name == "posix"
                                          else managed_process_options()),
                                       creationflags=(CREATE_SUSPENDED | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0)
            if session_scope_requested:
                session_scope = PosixSessionScope(process)
                exit_watch = _UnreapedProcessWatch(process)
            job.assign(process)
            job.resume(process)
            if payload is not None:
                def send():
                    try:
                        process.stdin.write(payload)
                        process.stdin.close()
                    except (BrokenPipeError, OSError, ValueError):
                        pass
                writer = threading.Thread(target=send, name="runtime-stdin", daemon=True)
                writer.start()
            while (exit_watch.running() if exit_watch is not None else process.poll() is None):
                _check(cancel_check, deadline)
                if on_poll:
                    on_poll(process)
                time.sleep(.1)
        except RuntimePreparationTimeout as exc:
            timeout_error = exc
        finally:
            active_failure = sys.exc_info()[1]
            try:
                if session_scope is not None:
                    session_scope.close()
                if job:
                    job.close()
                if process:
                    if os.name == "posix" and session_scope is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if process.poll() is None:
                        process.kill()
                    result_code = process.wait(timeout=5)
                    if writer:
                        writer.join(timeout=1)
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
                if session_scope_requested and (session_scope is None or not session_scope.closed):
                    raise OSError("The complete POSIX worker session cleanup was not confirmed")
                if on_cleanup:
                    on_cleanup()
            except BaseException as cleanup_failure:
                # Do not turn cancellation into an ordinary installer failure
                # that a mirror fallback or source repair might retry.
                failure = active_failure if isinstance(active_failure, (KeyboardInterrupt, SystemExit)) else ProcessCleanupError(
                    "本任务子进程回收未确认，已阻止后续安装和修复")
                failure.process_cleanup_confirmed = False
                failure.cleanup_receipt = {"status": "error", "process_cleanup_confirmed": False,
                    "errors": [{"reason": safe_runtime_diagnostic(cleanup_failure)}]}
                failure.add_note("子进程清理失败：" + safe_runtime_diagnostic(cleanup_failure))
                raise failure from cleanup_failure
            finally:
                if exit_watch is not None:
                    exit_watch.close()
        def captured(stream):
            if max_output_bytes is None:
                stream.seek(0)
                return stream.read().decode("utf-8", "replace")
            length = stream.tell()
            stream.seek(max(0, length - max_output_bytes))
            text = stream.read(max_output_bytes).decode("utf-8", "replace")
            return ("[earlier output omitted]\n" if length > max_output_bytes else "") + text
        stdout, stderr = captured(out), captured(err)
        if timeout_error is not None:
            # Capture only after closing the job and reaping its process. A
            # retry may then clean the target without a surviving installer
            # continuing to write into it, and keeps the useful partial logs.
            timeout_error.stdout, timeout_error.stderr = stdout, stderr
            raise timeout_error
        return subprocess.CompletedProcess(argv, result_code, stdout, stderr)


def _unsafe_windows_executable(executable):
    if os.name != "nt":
        return False
    executable = Path(executable)
    def alias(path):
        try:
            return getattr(path.lstat(), "st_reparse_tag", 0) == 0x8000001B
        except OSError:
            return False
    if alias(executable) or "windowsapps" in {part.lower() for part in executable.parts}:
        return True
    configuration = executable.parent.parent / "pyvenv.cfg"
    if configuration.is_file():
        for line in configuration.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip().lower() == "home":
                home = Path(value.strip())
                return alias(home / "python.exe") or "windowsapps" in {part.lower() for part in home.parts}
    return False


PROBE = ("import json,sys,sysconfig;print(json.dumps({'executable':sys.executable,"
         "'base_executable':getattr(sys,'_base_executable',sys.executable),"
         "'version':list(sys.version_info[:3]),'implementation':sys.implementation.name,"
         "'bits':64 if sys.maxsize>2**32 else 32,'platform':sysconfig.get_platform(),"
         "'prefix':sys.prefix,'base_prefix':sys.base_prefix}))")


def _probe(executable, *, cancel_check=None, deadline=None):
    executable = Path(executable)
    _check(cancel_check, deadline)
    try:
        if not executable.is_file() or _unsafe_windows_executable(executable):
            return None
        result = run_owned_process([str(executable), "-I", "-c", PROBE],
                                   timeout_s=min(10, deadline - time.monotonic()), cancel_check=cancel_check)
        info = json.loads(result.stdout)
        if (result.returncode or info.get("implementation") != "cpython"
                or tuple(info.get("version", ())[:2]) not in SUPPORTED_VERSIONS
                or info.get("bits") != 64 or _unsafe_windows_executable(info["base_executable"])):
            return None
        return info
    except (OSError, ValueError, RuntimePreparationTimeout):
        _check(cancel_check, deadline)
        return None


def _healthy(executable, project_root, *, cancel_check, deadline):
    script = ("import sys;sys.path.insert(0,sys.argv[1]);"
              "from src.process_lifecycle import validate_windows_venv;validate_windows_venv();"
              "from src.runtime_preparation import _application_dependency_issue;"
              f"issue=_application_dependency_issue({APP_REQUIREMENTS!r},{APP_IMPORTS!r});"
              "assert issue is None,issue;"
              "print('AUTOREPRO_RUNTIME_READY')")
    try:
        # -I ignores PYTHONUTF8; pass the flag explicitly for Chinese health
        # diagnostics under Windows' default locale as well as UTF-8 mode.
        result = run_owned_process([str(executable), "-I", "-X", "utf8", "-c", script, str(project_root)],
                                   timeout_s=min(30, deadline-time.monotonic()), cancel_check=cancel_check)
        return result.returncode == 0 and "AUTOREPRO_RUNTIME_READY" in result.stdout
    except (OSError, RuntimePreparationTimeout):
        _check(cancel_check, deadline)
        return False


def _candidate_paths(project_root, *, cancel_check, deadline):
    paths = [Path(sys.executable)]
    venv_bin = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    paths += [project_root / name / venv_bin for name in (".venv-py312", ".venv-py311", ".venv")]
    if os.name == "nt":
        launcher = shutil.which("py")
        if launcher and not _unsafe_windows_executable(launcher):
            try:
                result = run_owned_process([launcher, "-0p"], timeout_s=min(10, deadline-time.monotonic()), cancel_check=cancel_check)
                paths.extend(Path(match.group(1).strip()) for match in
                             re.finditer(r"(?im)^.*?([A-Z]:\\.+?python(?:w)?\.exe)\s*$", result.stdout))
            except (OSError, RuntimePreparationTimeout):
                _check(cancel_check, deadline)
        for variable in ("LOCALAPPDATA", "ProgramFiles", "ProgramW6432"):
            if os.environ.get(variable):
                root = Path(os.environ[variable])
                for version in ("Python312", "Python311"):
                    paths.extend((root / "Programs" / "Python" / version / "python.exe",
                                  root / version / "python.exe"))
        # Per-user and system python.org installations can use arbitrary paths.
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                for version in ("3.12", "3.11"):
                    try:
                        with winreg.OpenKey(hive, rf"Software\Python\PythonCore\{version}\InstallPath", 0, winreg.KEY_READ | view) as key:
                            paths.append(Path(winreg.QueryValue(key, None)) / "python.exe")
                    except OSError:
                        pass
    else:
        paths.extend(Path(path) for name in ("python3.12", "python3.11") if (path := shutil.which(name)))
    seen = set()
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            yield path


def _machine_key(project_root):
    raw = "|".join((platform.node(), os.environ.get("USERDOMAIN", ""),
                    str(Path.home()), str(project_root), sys.platform,
                    platform.machine()))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@contextmanager
def _runtime_lock(root, *, cancel_check, deadline):
    """OS file lock plus a thread lock, with cancellable bounded waiting."""
    lock_path = workspace_path(root, ".runtime.lock", "runtime lock")
    with _thread_locks_guard:
        mutex = _thread_locks.setdefault(os.path.normcase(str(lock_path)), threading.Lock())
    while not mutex.acquire(timeout=.1):
        _check(cancel_check, deadline)
    stream, acquired = None, False
    try:
        stream = lock_path.open("a+b")
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        while not acquired:
            _check(cancel_check, deadline)
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                time.sleep(.1)
        yield
    finally:
        if stream:
            if acquired:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()
        mutex.release()


def _emit(callback, stage, message, **details):
    if callback:
        callback({"type": "runtime_preparation", "stage": stage,
                  "status": "completed" if stage == "ready" else "running",
                  "message": safe_runtime_diagnostic(message), **details})


def _download_installer(destination, *, cancel_check, deadline):
    # The pinned final 3.12 bugfix installer is available from python.org.
    url = f"https://www.python.org/ftp/python/{PYTHON_RELEASE}/python-{PYTHON_RELEASE}-amd64.exe"
    partial = workspace_path(destination.parent, destination.name + ".part", "partial Python installer")
    try:
        for attempt in range(2):
            _check(cancel_check, deadline)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "AutoReproducer/runtime-preparation"})
                with urllib.request.urlopen(request, timeout=min(10, max(.1, deadline-time.monotonic()))) as response:
                    final = urllib.parse.urlsplit(response.geturl())
                    if final.scheme != "https" or final.hostname != "www.python.org":
                        raise RuntimePreparationError("Python 安装程序重定向到非官方地址，已停止自动安装")
                    total = 0
                    with partial.open("wb") as output:
                        read_chunk = getattr(response, "read1", None) or response.read
                        while True:
                            _check(cancel_check, deadline)
                            # HTTPResponse.read(n) can wait for n bytes forever
                            # on a slowly streaming peer. read1 performs one
                            # socket read, bounded by the existing 10 s timeout.
                            chunk = read_chunk(65536)
                            _check(cancel_check, deadline)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > 64*1024*1024:
                                raise RuntimePreparationError("Python 安装程序超过大小限制")
                            output.write(chunk)
                    if total < 1024*1024:
                        raise RuntimePreparationError("Python 安装程序下载不完整")
                partial.replace(destination)
                return
            except (OSError, urllib.error.URLError) as exc:
                partial.unlink(missing_ok=True)
                if attempt:
                    raise RuntimePreparationError("无法下载官方 Python 安装程序：" + safe_runtime_diagnostic(exc)) from exc
    finally:
        partial.unlink(missing_ok=True)


def _verify_installer(installer, *, cancel_check, deadline):
    powershell = shutil.which("powershell.exe")
    if not powershell:
        raise RuntimePreparationError("无法验证官方 Python 安装程序的 Windows 数字签名")
    script = ("$s=Get-AuthenticodeSignature -LiteralPath $env:AUTOREPRO_INSTALLER;"
              "if($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notmatch '(?:CN|O)=Python Software Foundation(?:,|$)'){exit 1}")
    environment = _clean_environment()
    environment["AUTOREPRO_INSTALLER"] = str(installer)
    result = run_owned_process([powershell, "-NoProfile", "-NonInteractive", "-Command", script], env=environment,
                               timeout_s=min(30, deadline-time.monotonic()), cancel_check=cancel_check)
    if result.returncode:
        raise RuntimePreparationError("官方 Python 安装程序数字签名验证失败，已停止自动安装")


def _install_python(root, *, cancel_check, deadline, progress_callback):
    if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimePreparationError("未发现兼容的标准 Python；当前系统暂不支持自动安装，请安装 64 位 CPython 3.11/3.12 后重试")
    installer = workspace_path(root, f"python-{PYTHON_RELEASE}-amd64.exe", "Python installer")
    target = workspace_path(root, "python312", "managed Python")
    _emit(progress_callback, "download_python", "未发现兼容 Python，正在下载并验证 python.org 官方安装程序")
    if not installer.is_file():
        _download_installer(installer, cancel_check=cancel_check, deadline=deadline)
    try:
        _verify_installer(installer, cancel_check=cancel_check, deadline=deadline)
    except RuntimePreparationError:
        installer.unlink(missing_ok=True)
        raise
    _emit(progress_callback, "install_python", "正在为当前用户安装独立 Python；首次准备完成后会复用缓存")
    result = run_owned_process([str(installer), "/quiet", "InstallAllUsers=0", f"TargetDir={target}",
                               "PrependPath=0", "AssociateFiles=0", "Include_launcher=0", "InstallLauncherAllUsers=0",
                               "Include_test=0", "Include_doc=0", "Include_tcltk=0", "Include_pip=1", "Shortcuts=0"],
                              timeout_s=min(300, deadline-time.monotonic()), cancel_check=cancel_check)
    candidate = target / "python.exe"
    info = _probe(candidate, cancel_check=cancel_check, deadline=deadline)
    if result.returncode not in {0, 3010} or info is None:
        raise RuntimePreparationError("自动安装官方 Python 未成功：" + safe_runtime_diagnostic(result.stderr or result.stdout or f"exit {result.returncode}"))
    return info


def _ensure_pip(executable, *, cancel_check, deadline):
    """Repair an interrupted venv creation using CPython's bundled wheels."""
    def pip_check():
        return run_owned_process([str(executable), "-I", "-m", "pip", "--version"],
                                 timeout_s=min(20, deadline-time.monotonic()), cancel_check=cancel_check)
    result = pip_check()
    if result.returncode == 0:
        return
    result = run_owned_process([str(executable), "-I", "-m", "ensurepip", "--upgrade"],
                               timeout_s=min(90, deadline-time.monotonic()), cancel_check=cancel_check)
    if result.returncode == 0 and pip_check().returncode == 0:
        return
    # Missing files can coexist with a valid dist-info. ensurepip --upgrade
    # considers that installed version satisfied; force reinstall its bundled
    # wheel, without importing the corrupt on-disk copy or contacting a server.
    bootstrap = ("import ensurepip,runpy,sys;from pathlib import Path;"
                 "wheels=sorted((Path(ensurepip.__file__).parent/'_bundled').glob('pip-*.whl'));"
                 "assert wheels,'CPython bundled pip wheel is missing';"
                 "sys.path.insert(0,str(wheels[-1]));"
                 "sys.argv=['pip','install','--no-index','--no-cache-dir','--force-reinstall',str(wheels[-1])];"
                 "runpy.run_module('pip',run_name='__main__',alter_sys=True)")
    result = run_owned_process([str(executable), "-I", "-c", bootstrap],
                               timeout_s=min(90, deadline-time.monotonic()), cancel_check=cancel_check)
    if result.returncode or pip_check().returncode:
        raise RuntimePreparationError("独立运行环境的 pip 自动修复失败：" + safe_runtime_diagnostic(result.stderr or result.stdout)[-4000:])


def _install_app_dependencies(executable, root, *, cancel_check, deadline, progress_callback, offline):
    wheels = workspace_path(root, "wheels", "runtime wheel cache")
    wheels.mkdir(exist_ok=True)
    environment = _clean_environment()
    environment["PIP_CACHE_DIR"] = str(workspace_path(root, "pip-cache", "runtime pip cache"))
    if offline:
        # --no-index alone does not disable remote find-links in a pip config.
        for name in ("PIP_FIND_LINKS", "PIP_EXTRA_INDEX_URL", "PIP_INDEX_URL"):
            environment.pop(name, None)
        environment["PIP_CONFIG_FILE"] = os.devnull
    # Preserve an explicitly configured private package source; public defaults
    # get one official fallback when the regional mirror is unreachable.
    configured = os.environ.get("AUTOREPRO_PIP_INDEX", os.environ.get("PIP_INDEX_URL", "")).strip()
    indexes = (configured,) if configured and configured.rstrip("/") not in PUBLIC_INDEXES else PUBLIC_INDEXES
    routes = (None,) if offline else indexes
    diagnostic = ""
    for attempt, index in enumerate(routes, 1):
        _check(cancel_check, deadline)
        _emit(progress_callback, "install_dependencies", "正在自动安装应用运行依赖", attempt=attempt)
        command = [str(executable), "-m", "pip", "download", "--disable-pip-version-check", "--no-input",
                   "--retries", "1", "--timeout", "15", "--find-links", str(wheels)]
        command += ["--no-index"] if offline else ["--index-url", index]
        command += ["--only-binary=:all:", "--dest", str(wheels)]
        try:
            result = run_owned_process([*command, *APP_REQUIREMENTS], env=environment,
                                       timeout_s=min(300, deadline-time.monotonic()), cancel_check=cancel_check)
        except RuntimePreparationTimeout as exc:
            _check(cancel_check, deadline)
            diagnostic = safe_runtime_diagnostic(exc)
            continue
        if result.returncode == 0:
            # Install from the completed wheel set. Cache repairs now also work
            # offline when the original installer/package index is unavailable.
            environment["PIP_CONFIG_FILE"] = os.devnull
            for name in ("PIP_FIND_LINKS", "PIP_EXTRA_INDEX_URL", "PIP_INDEX_URL"):
                environment.pop(name, None)
            result = run_owned_process([str(executable), "-m", "pip", "install", "--disable-pip-version-check",
                                        "--no-input", "--force-reinstall", "--no-index", "--find-links", str(wheels), *APP_REQUIREMENTS],
                                       env=environment, timeout_s=min(180, deadline-time.monotonic()), cancel_check=cancel_check)
            if result.returncode == 0:
                return
            raise RuntimePreparationError("缓存应用依赖安装失败：" + safe_runtime_diagnostic(result.stderr or result.stdout)[-4000:])
        diagnostic = safe_runtime_diagnostic(result.stderr or result.stdout)
        # A dependency conflict cannot be healed by sending it to another index.
        if re.search(r"resolutionimpossible|conflicting dependencies|failed building wheel|requires a different python", diagnostic, re.I):
            break
        _emit(progress_callback, "retry_dependencies", "依赖源暂时不可用，正在尝试备用源", diagnostic=diagnostic[-2000:])
    raise RuntimePreparationError("独立运行环境依赖安装失败：" + diagnostic[-4000:])


def prepare_runtime(project_root=None, *, cancel_check=None, progress_callback=None,
                    timeout_s=900, allow_download=True, offline=False):
    """Reuse a healthy runtime, repair our cache, or bootstrap official Python.

    Shared caches are serialized per machine/user/project. A health import is
    mandatory on every reuse, and the ready manifest is written only after it
    succeeds. No existing venv configuration or global package is modified.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    deadline = time.monotonic() + timeout_s
    reason = runtime_requirement() or "使用兼容的标准 Python 运行预设"
    _emit(progress_callback, "discover", reason)
    root = workspace_path(project_root, f"data/runtime/{_machine_key(project_root)}", "runtime cache")
    root.mkdir(parents=True, exist_ok=True)
    with _runtime_lock(root, cancel_check=cancel_check, deadline=deadline), _cleanup_failed_preparation(root):
        # Probe our cache before system paths so future runs are independent of
        # changes to the launching environment or the original Python install.
        environment = workspace_path(root, "app-venv", "managed app venv")
        executable = workspace_path(root, "app-venv/" + ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
                                    "managed app interpreter")
        cached = _probe(executable, cancel_check=cancel_check, deadline=deadline)
        if cached and _healthy(executable, project_root, cancel_check=cancel_check, deadline=deadline):
            _emit(progress_callback, "ready", "已验证并复用独立运行环境", executable=str(executable))
            return RuntimePreparation(str(executable), True, False, reason, _fingerprint(cached))
        standard = None
        for candidate in _candidate_paths(project_root, cancel_check=cancel_check, deadline=deadline):
            info = _probe(candidate, cancel_check=cancel_check, deadline=deadline)
            if not info:
                continue
            # Current/project app venvs can be reused; a global installation is
            # always the base of a fresh isolated app venv instead.
            if info["prefix"] != info["base_prefix"] and _healthy(candidate, project_root, cancel_check=cancel_check, deadline=deadline):
                _emit(progress_callback, "ready", "已找到并验证可复用的标准 Python 环境", executable=str(candidate))
                return RuntimePreparation(str(candidate), True, False, reason, _fingerprint(info))
            if standard is None:
                standard = info
        if standard is None:
            private_python = workspace_path(root, "python312/python.exe", "managed Python")
            standard = _probe(private_python, cancel_check=cancel_check, deadline=deadline)
        if standard is None:
            if offline or not allow_download:
                raise RuntimePreparationError("未发现可用的标准 Python；离线模式不会下载新解释器，请恢复已有运行环境缓存")
            standard = _install_python(root, cancel_check=cancel_check, deadline=deadline, progress_callback=progress_callback)
        repaired = environment.exists()
        manifest = workspace_path(root, "ready.json", "runtime manifest")
        manifest.unlink(missing_ok=True)
        if not cached:
            _emit(progress_callback, "create_environment", "正在创建可安全管理训练子进程的独立运行环境")
            # --clear is intentionally avoided: interruption leaves reusable
            # package/download caches, and no existing host env is rewritten.
            base = standard["base_executable"]
            if _unsafe_windows_executable(base):
                raise RuntimePreparationError("发现的 Python 基础解释器仍来自 Store，已拒绝启动")
            result = run_owned_process([base, "-I", "-m", "venv", str(environment)],
                                       timeout_s=min(120, deadline-time.monotonic()), cancel_check=cancel_check)
            if result.returncode or _probe(executable, cancel_check=cancel_check, deadline=deadline) is None:
                raise RuntimePreparationError("自动创建独立环境失败：" + safe_runtime_diagnostic(result.stderr or result.stdout)[-4000:])
        _ensure_pip(executable, cancel_check=cancel_check, deadline=deadline)
        _install_app_dependencies(executable, root, cancel_check=cancel_check, deadline=deadline,
                                  progress_callback=progress_callback, offline=offline)
        info = _probe(executable, cancel_check=cancel_check, deadline=deadline)
        if not info or not _healthy(executable, project_root, cancel_check=cancel_check, deadline=deadline):
            raise RuntimePreparationError("依赖安装完成但独立运行环境健康检查未通过，下次运行会继续自动修复")
        record = {"executable": str(executable), "fingerprint": _fingerprint(info),
                  "requirements": list(APP_REQUIREMENTS), "validated_at": time.time()}
        temporary = workspace_path(root, "ready.json.tmp", "runtime manifest")
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        temporary.replace(manifest)
        _emit(progress_callback, "ready", "独立运行环境已准备完成；后续复现会自动复用", executable=str(executable))
        return RuntimePreparation(str(executable), False, repaired, reason, _fingerprint(info))


def _fingerprint(info):
    return f"cpython-{info['version'][0]}{info['version'][1]}-{info['platform']}-{info['bits']}"


def release_runtime_downloads(project_root=None):
    """Release bootstrap downloads after a run; retain the application runtime.

    The preparation lock protects a concurrent runtime repair. Only the known
    download paths in this machine's managed cache are eligible for removal.
    """
    project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    root = workspace_path(project_root, f"data/runtime/{_machine_key(project_root)}", "runtime cache")
    if not root.is_dir():
        return {"status": "released", "removed": [], "freed_bytes": 0, "errors": []}
    with _runtime_lock(root, cancel_check=None, deadline=time.monotonic() + 60):
        return _release_runtime_downloads_locked(root)


def _release_runtime_downloads_locked(root):
    from src.safety.files import directory_bytes, remove_tree

    removed, errors = [], []
    for name in ("wheels", "pip-cache", f"python-{PYTHON_RELEASE}-amd64.exe",
                 f"python-{PYTHON_RELEASE}-amd64.exe.part"):
        try:
            path = workspace_path(root, name, "runtime download release")
            if path.is_dir():
                size = directory_bytes(path)
                remove_tree(path)
            elif path.is_file():
                size = path.stat().st_size
                path.unlink()
            else:
                continue
            removed.append({"path": name, "bytes": size})
        except (OSError, ValueError) as exc:
            errors.append({"path": name, "reason": str(exc)})
    return {"status": "error" if errors else "released", "removed": removed,
            "freed_bytes": sum(item["bytes"] for item in removed), "errors": errors}


@contextmanager
def _cleanup_failed_preparation(root):
    try:
        yield
    except BaseException as failure:
        # A live installer may still be writing into these folders. Keep both
        # process and filesystem ownership unresolved instead of deleting them.
        if getattr(failure, "process_cleanup_confirmed", True) is False:
            raise
        try:
            receipt = _release_runtime_downloads_locked(root)
            failure.cleanup_receipt = receipt
            if receipt["errors"]:
                failure.add_note("运行环境准备失败后的下载清理失败：" + str(receipt["errors"]))
        except BaseException as exc:
            failure.cleanup_receipt = {"status": "error", "errors": [
                {"reason": safe_runtime_diagnostic(exc)}]}
            failure.add_note("运行环境准备失败后的下载清理失败：" + str(exc))
        raise


def release_worker_environment(prepared):
    """Reclaim dependencies only after the worker's process job has exited.

    The recovery coordinator may lack filelock or other app dependencies.
    In that case the verified interpreter performs the same locked cleanup.
    """
    try:
        from src.environment_lifecycle import release_abandoned_environment
        return release_abandoned_environment()
    except ModuleNotFoundError:
        project_root = Path(__file__).resolve().parents[1]
        program = ("import sys,json;sys.path.insert(0,sys.argv[1]);"
                   "from src.environment_lifecycle import release_abandoned_environment;"
                   "print(json.dumps(release_abandoned_environment()))")
        environment = _clean_environment()
        for name in ("LLM_API_KEY", "GITHUB_TOKEN", "GH_TOKEN", "AUTOREPRO_GITHUB_TOKEN"):
            environment.pop(name, None)
        completed = run_owned_process([prepared.executable, "-I", "-c", program, str(project_root)],
                                      env=environment, timeout_s=120)
        if completed.returncode:
            raise RuntimePreparationError("工作进程环境释放失败：" + safe_runtime_diagnostic(
                completed.stderr or completed.stdout)[-1600:])
        receipt = json.loads(completed.stdout)
        if not isinstance(receipt, dict) or receipt.get("status") not in {"released", "in_use", "error"}:
            raise RuntimePreparationError("工作进程环境释放未返回有效回执")
        return receipt
