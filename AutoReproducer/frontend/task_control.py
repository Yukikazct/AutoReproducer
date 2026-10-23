"""Cancellation belongs to one progress file, never to all Python processes."""
import json
import ctypes
import errno
import os
from pathlib import Path
import threading
import uuid

_tasks = {}
_blocked_roots = {}
_guard = threading.RLock()
_local = threading.local()


def current_task():
    return getattr(_local, "control", None)


def bind_task(control):
    _local.control = control


def _key(path):
    return os.path.normcase(str(Path(path).resolve()))


def _paths(path):
    path = Path(path)
    return (path.with_suffix(".control.json"), path.with_suffix(".owner.lock"),
            path.with_suffix(".cancel"))


class _OwnerLockBusy(TimeoutError):
    pass


class _OwnerLock:
    """Nonblocking owner lock, usable before any app dependencies are installed.

    Match FileLock's native lock range (Windows byte zero / POSIX flock), so
    existing owner files and readers keep the same cross-process semantics.
    The file stays in place after release to avoid an unlink/reopen race.
    """
    def __init__(self, path):
        self.path = str(path)
        self.fd = None

    def acquire(self, timeout=0):
        if timeout != 0:
            raise ValueError("owner locks only support nonblocking acquisition")
        if self.fd is not None:
            raise RuntimeError("owner lock is already held")
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise _OwnerLockBusy(self.path) from exc
            raise
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        return self

    def release(self):
        if self.fd is None:
            return
        fd, self.fd = self.fd, None
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


class TaskControl:
    def __init__(self, path):
        from src.project_paths import get_deps_root
        self.path = str(Path(path).resolve())
        self.deps_root = str(get_deps_root().resolve())
        self.record, lock, self.cancel_file = _paths(self.path)
        self.record.parent.mkdir(parents=True, exist_ok=True)
        self.lock = _OwnerLock(lock)
        self.lock.acquire(timeout=0)
        self.event = threading.Event()
        self.token = uuid.uuid4().hex
        self.thread = None
        self.finished = False
        self.cleanup_receipt = None
        try:
            self.cancel_file.unlink(missing_ok=True)
            self._save("running")
        except BaseException:
            self.lock.release()
            raise

    def _save(self, state):
        temporary = self.record.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 1, "pid": os.getpid(),
                                         "token": self.token, "state": state,
                                         "deps_root": self.deps_root,
                                         "platform": os.name}), encoding="utf-8")
        temporary.replace(self.record)

    def cancelled(self):
        if self.event.is_set():
            return True
        try:
            requested = self.cancel_file.read_text(encoding="ascii", errors="replace") == self.token
        except OSError:
            requested = False
        if requested:
            self.event.set()
        return requested

    def finish(self, *, cleanup_failed=False, process_cleanup_confirmed=True):
        if cleanup_failed or not process_cleanup_confirmed:
            with _guard:
                _blocked_roots.setdefault(_key(self.deps_root), set()).add(_key(self.path))
        self._save("cleanup_unconfirmed" if not process_cleanup_confirmed else
                   "cleanup_failed" if cleanup_failed else "finished")
        self.finished = True
        try:
            self.cancel_file.unlink(missing_ok=True)
        except OSError:
            pass  # An old token cannot cancel a later owner; retain the receipt.
        self.lock.release()
        with _guard:
            _tasks.pop(_key(self.path), None)


def register_task(path):
    from src.project_paths import get_deps_root
    with _guard:
        if _key(path) in _tasks:
            raise RuntimeError("本任务的后台持有者仍在运行")
        root = _key(get_deps_root())
        if _blocked_roots.get(root):
            raise RuntimeError("该依赖目录仍有清理失败或进程回收未确认的任务，请先恢复原任务")
        # Persisted failures also gate another app session/service using the
        # same runtime directory. Do not overwrite an unresolved old receipt.
        for record in Path(path).resolve().parent.glob("*.control.json"):
            try:
                data = json.loads(record.read_text(encoding="utf-8"))
                if (not isinstance(data, dict) or data.get("version") != 1
                        or not isinstance(data.get("deps_root"), str)
                        or data.get("state") not in {"running", "finished", "cleanup_failed", "cleanup_unconfirmed"}):
                    raise ValueError("invalid owner record")
            except (OSError, ValueError):
                raise RuntimeError("存在无法确认清理状态的任务记录，请先恢复原任务：" + record.name) from None
            if (_key(data["deps_root"]) == root
                    and data["state"] in {"cleanup_failed", "cleanup_unconfirmed"}):
                raise RuntimeError("该依赖目录仍有未完成清理的任务，请先恢复原任务：" + record.name)
        control = TaskControl(path)
        _tasks[_key(path)] = control
        return control


def task_status(path):
    if not path:
        return {"managed": False, "active": False, "cancelling": False}
    with _guard:
        control = _tasks.get(_key(path))
    if control is not None:
        return {"managed": True, "active": not control.finished,
                "cancelling": control.cancelled()}
    record, lock_path, cancel_file = _paths(path)
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("version") != 1
                or type(data.get("pid")) is not int or data["pid"] <= 0
                or not isinstance(data.get("token"), str) or len(data["token"]) != 32):
            raise ValueError("unknown task owner")
        lock = _OwnerLock(lock_path)
        try:
            with lock.acquire(timeout=0):
                active = False
        except _OwnerLockBusy:
            active = True
        try:
            cancelling = cancel_file.read_text(encoding="ascii", errors="replace") == data["token"]
        except OSError:
            cancelling = False
        return {"managed": True, "active": active, "cancelling": cancelling,
                "orphaned": not active and data.get("state") in {"running", "cleanup_unconfirmed"},
                "cleanup_failed": data.get("state") in {"cleanup_failed", "cleanup_unconfirmed"},
                "cleanup_unconfirmed": data.get("state") == "cleanup_unconfirmed", "record": data}
    except (OSError, ValueError, TypeError):
        return {"managed": False, "active": False, "cancelling": False}


def request_cancel(path):
    """Return promptly; the supervisor confirms cleanup before allowing reset."""
    with _guard:
        control = _tasks.get(_key(path))
    if control is not None:
        control.event.set()
        return True
    status = task_status(path)
    if status["managed"] and status["active"]:
        _paths(path)[2].write_text(status["record"]["token"], encoding="ascii")
        return True
    return False


def recover_orphaned_task(path, store_type):
    """Recover a crashed Windows supervisor only after its job owner is gone.

    Legacy/foreign work without an ownership receipt is deliberately untouched.
    On Windows the OS closes the supervisor's job handles and kills descendants;
    merely finding a free lock on POSIX cannot establish that guarantee.
    """
    status = task_status(path)
    if not status.get("orphaned") and not status.get("cleanup_failed"):
        return False
    if status.get("orphaned"):
        if status["record"].get("platform") != "nt" or os.name != "nt":
            return False
        from src.process_lifecycle import _windows_api
        api = _windows_api()
        handle = api.OpenProcess(0x00100000, False, status["record"]["pid"])
        if handle:
            try:
                if api.WaitForSingleObject(handle, 0) != 0:
                    return False
            finally:
                api.CloseHandle(handle)
        elif ctypes.get_last_error() != 87:  # ERROR_INVALID_PARAMETER: no such PID
            return False  # Access denied is not proof that an owner exited.
    record, lock_path, _ = _paths(path)
    try:
        with _OwnerLock(lock_path).acquire(timeout=0):
            from src.environment_lifecycle import release_abandoned_environment, record_environment_cleanup
            if not status["record"].get("deps_root"):
                return False
            receipt = release_abandoned_environment(root=status["record"]["deps_root"])
            if receipt.get("errors"):
                return False
            store = store_type(path, reset=False)
            snapshot = store_type.read_snapshot(path)
            for scope, event, identifier in (("active_runs", "execution_run", "execution_id"),
                                              ("active_workers", "pipeline_worker", "worker_id")):
                for row in snapshot.get(scope, []):
                    store.emit({"type": event, identifier: row[identifier], "status": "interrupted",
                                "cleanup_confirmed": True, "reason": "后台持有者退出，已恢复中断状态"})
            result = dict(snapshot.get("result") or {})
            result.update(state="ERROR", error="后台进程已退出，本任务已中断")
            result["data"] = {**result.get("data", {}), "run_status": "interrupted",
                              "environment_cleanup": receipt}
            record_environment_cleanup(result, receipt)
            store.emit({"type": "done", "result": result})
            status["record"]["state"] = "finished"
            record.write_text(json.dumps(status["record"]), encoding="utf-8")
            with _guard:
                root = _key(status["record"]["deps_root"])
                blocked = _blocked_roots.get(root)
                if blocked is not None:
                    blocked.discard(_key(path))
                    if not blocked:
                        _blocked_roots.pop(root, None)
            return True
    except (OSError, ValueError):
        return False
