"""Cancellation belongs to one progress file, never to all Python processes."""
import json
import errno
import hashlib
import os
from pathlib import Path
import tempfile
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


def _root_paths(root, path=None):
    """Keep ownership outside the disposable dependency directory."""
    root = Path(root).resolve()
    directory = root.parent / ("." + root.name + ".task-control")
    name = hashlib.sha256(_key(path).encode("utf-8")).hexdigest() + ".json" if path else None
    return directory, directory / ".owner.lock", directory / name if name else None


def _write_record(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_record(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("version") != 1
            or type(data.get("pid")) is not int or data["pid"] <= 0
            or not isinstance(data.get("token"), str) or len(data["token"]) != 32
            or not isinstance(data.get("deps_root"), str) or not data["deps_root"]
            or data.get("state") not in {"running", "finished", "cleanup_failed", "cleanup_unconfirmed"}):
        raise ValueError("unknown task owner")
    return data


def _read_task_record(path):
    from src.project_paths import get_deps_root
    try:
        local = _read_record(_paths(path)[0])
    except (OSError, ValueError):
        local = None
    root = local["deps_root"] if local else get_deps_root()
    shared_path = _root_paths(root, path)[2]
    try:
        shared = _read_record(shared_path)
    except FileNotFoundError:
        if local is None:
            raise ValueError("unknown task owner") from None
        return local
    if (shared.get("progress_path") != _key(path)
            or _key(shared["deps_root"]) != _key(root)):
        raise ValueError("unknown shared task owner")
    if (local is not None and local["token"] == shared["token"]
            and shared["state"] == "running" and local["state"] != "running"):
        # A locked shared receipt can retain its initial running state after
        # the local terminal write succeeds. Keep its gate, but use the saved
        # cleanup confirmation so the same live service can retry recovery.
        return {**local, "state": "cleanup_failed" if local["state"] == "finished" else local["state"]}
    return shared


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
    def __init__(self, path, *, root_lock=None, deps_root=None):
        from src.project_paths import get_deps_root
        self.path = str(Path(path).resolve())
        self.deps_root = str(Path(deps_root if deps_root is not None else get_deps_root()).resolve())
        directory, root_lock_path, self.shared_record = _root_paths(self.deps_root, path)
        directory.mkdir(parents=True, exist_ok=True)
        self.root_lock = root_lock or _OwnerLock(root_lock_path).acquire(timeout=0)
        self.record, lock, self.cancel_file = _paths(self.path)
        self.record.parent.mkdir(parents=True, exist_ok=True)
        self.lock = _OwnerLock(lock)
        self.lock.acquire(timeout=0)
        self.event = threading.Event()
        self.token = uuid.uuid4().hex
        self.thread = None
        self.finished = False
        self._finish_guard = threading.Lock()
        self.cleanup_receipt = None
        try:
            self.cancel_file.unlink(missing_ok=True)
            self._save_shared("running")
            self._save("running")
        except BaseException:
            try:
                self.shared_record.unlink(missing_ok=True)  # No worker has started.
            finally:
                self.lock.release()
                self.root_lock.release()
            raise

    def _data(self, state):
        return {"version": 1, "pid": os.getpid(), "token": self.token,
                "state": state, "deps_root": self.deps_root, "platform": os.name,
                "progress_path": _key(self.path)}

    def _save(self, state):
        _write_record(self.record, self._data(state))

    def _save_shared(self, state):
        _write_record(self.shared_record, self._data(state))

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
        with self._finish_guard:
            if not self.finished:
                self._finish(cleanup_failed=cleanup_failed, process_cleanup_confirmed=process_cleanup_confirmed)

    def _finish(self, *, cleanup_failed, process_cleanup_confirmed):
        state = ("cleanup_unconfirmed" if not process_cleanup_confirmed else
                 "cleanup_failed" if cleanup_failed else "finished")
        try:
            if state != "finished":
                with _guard:
                    _blocked_roots.setdefault(_key(self.deps_root), set()).add(_key(self.path))
            self._save(state)
            if state == "finished":
                self.shared_record.unlink()
            else:
                self._save_shared(state)
        except BaseException as exc:
            # Even a successful process cleanup is not a published completion
            # when its receipt could not be saved. The preexisting running
            # lease remains a durable, conservative gate if this fallback fails.
            with _guard:
                _blocked_roots.setdefault(_key(self.deps_root), set()).add(_key(self.path))
            try:
                self._save_shared("cleanup_unconfirmed" if not process_cleanup_confirmed else "cleanup_failed")
            except BaseException as fallback:
                exc.add_note("共享清理凭据保存失败：" + str(fallback))
            raise
        finally:
            try:
                try:
                    self.cancel_file.unlink(missing_ok=True)
                except OSError:
                    pass  # An old token cannot cancel a later owner.
            finally:
                try:
                    self.lock.release()
                finally:
                    try:
                        self.root_lock.release()
                    finally:
                        with _guard:
                            self.finished = True
                            _tasks.pop(_key(self.path), None)


def register_task(path):
    from src.project_paths import get_deps_root
    with _guard:
        if _key(path) in _tasks:
            raise RuntimeError("本任务的后台持有者仍在运行")
        root = _key(get_deps_root())
        if _blocked_roots.get(root):
            raise RuntimeError("该依赖目录仍有清理失败或进程回收未确认的任务，请先恢复原任务")
        directory, root_lock_path, _ = _root_paths(root)
        directory.mkdir(parents=True, exist_ok=True)
        root_lock = _OwnerLock(root_lock_path)
        try:
            root_lock.acquire(timeout=0)
        except _OwnerLockBusy:
            raise RuntimeError("该依赖目录已有后台任务持有，请先停止或等待原任务完成") from None
        try:
            _check_local_receipts(path, root)
            for record in directory.iterdir():
                if record.suffix != ".json":
                    continue
                try:
                    data = _read_record(record)
                    if _key(data["deps_root"]) != root or not isinstance(data.get("progress_path"), str):
                        raise ValueError("invalid shared owner record")
                except (OSError, ValueError):
                    raise RuntimeError("存在无法确认清理状态的共享任务记录，请先恢复原任务：" + str(record)) from None
                raise RuntimeError("该依赖目录仍有未完成清理的任务，请先恢复原任务：" + data["progress_path"])
            control = TaskControl(path, root_lock=root_lock, deps_root=root)
        except BaseException:
            root_lock.release()
            raise
        _tasks[_key(path)] = control
        return control


def _check_local_receipts(path, root):
    # Older owners did not create a shared lease. Keep validating receipts
    # found beside the selected progress file during migration.
    for record in Path(path).resolve().parent.glob("*.control.json"):
        try:
            data = _read_record(record)
        except (OSError, ValueError):
            raise RuntimeError("存在无法确认清理状态的任务记录，请先恢复原任务：" + record.name) from None
        if (_key(data["deps_root"]) == root
                and data["state"] in {"running", "cleanup_failed", "cleanup_unconfirmed"}):
            raise RuntimeError("该依赖目录仍有未完成清理的任务，请先恢复原任务：" + record.name)


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
        data = _read_task_record(path)
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
    """Retry environment release only after the owner confirmed process cleanup.

    A dead Windows owner starts asynchronous job termination; its exit does not
    prove that every descendant has stopped using the environment. Running and
    cleanup-unconfirmed receipts therefore remain blocked after a service crash.
    """
    status = task_status(path)
    if status.get("active") or (not status.get("orphaned") and not status.get("cleanup_failed")):
        return False
    expected = status["record"]
    root = expected["deps_root"]
    record, lock_path, _ = _paths(path)
    directory, root_lock_path, shared_path = _root_paths(root, path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with _OwnerLock(root_lock_path).acquire(timeout=0), _OwnerLock(lock_path).acquire(timeout=0):
            # Another recovery or task generation can finish between the first
            # status read and lock acquisition. Never publish over its result.
            data = _read_task_record(path)
            if data["token"] != expected["token"] or _key(data["deps_root"]) != _key(root):
                return False
            if data["state"] != "cleanup_failed":
                return False
            from src.environment_lifecycle import release_abandoned_environment, record_environment_cleanup
            receipt = release_abandoned_environment(root=root)
            if receipt.get("errors") or receipt.get("status") == "in_use":
                return False
            store = store_type(path, reset=False)
            snapshot = store_type.read_snapshot(path, include_pending_result=True)
            for scope, event, identifier in (("active_runs", "execution_run", "execution_id"),
                                              ("active_workers", "pipeline_worker", "worker_id")):
                for row in snapshot.get(scope, []):
                    store.emit({"type": event, identifier: row[identifier], "status": "interrupted",
                                "cleanup_confirmed": True, "reason": "后台持有者退出，已恢复中断状态"})
            result = dict(snapshot.get("pending_result") or snapshot.get("result") or {})
            result.update(state="ERROR", error="后台进程已退出，本任务已中断")
            result["data"] = {**result.get("data", {}), "run_status": "interrupted",
                              "environment_cleanup": receipt}
            record_environment_cleanup(result, receipt)
            store.emit({"type": "done", "result": result})
            data["state"] = "finished"
            _write_record(record, data)
            shared_path.unlink(missing_ok=True)
            with _guard:
                blocked = _blocked_roots.get(_key(root))
                if blocked is not None:
                    blocked.discard(_key(path))
                    if not blocked:
                        _blocked_roots.pop(_key(root), None)
            return True
    except (OSError, ValueError):
        return False
