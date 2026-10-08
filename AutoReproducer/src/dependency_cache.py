"""Coordinate local dependency users and cleanup across threads and processes."""
import os
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from src.safety.paths import workspace_path


LOCK_TIMEOUT = 60
_locks = weakref.WeakValueDictionary()
_registry_lock = threading.Lock()


class DependencyCacheBusy(RuntimeError):
    pass


@contextmanager
def cache_guard(root, *, cleanup=False, timeout=None):
    """Serialize use of one cache root; cleanup never waits for active users.

    A shared lock instance is reentrant within the owning thread, so dependency
    preparation/self-healing can also safely be called on their own. Cleanup
    explicitly rejects even a same-thread active user.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = workspace_path(root, ".autorepro-cache.lock", "dependency lock")
    key = (os.getpid(), os.path.normcase(str(lock_path)))
    with _registry_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = FileLock(str(lock_path))
            _locks[key] = lock
    wait = 0 if cleanup else (LOCK_TIMEOUT if timeout is None else timeout)
    if cleanup and lock.is_locked:
        raise DependencyCacheBusy("依赖缓存正在使用，本次清理已跳过")
    try:
        lock.acquire(timeout=wait)
    except Timeout as exc:
        message = ("依赖缓存正在使用，本次清理已跳过" if cleanup else
                   f"依赖缓存正在使用，等待 {wait:g} 秒超时，请稍后重试")
        raise DependencyCacheBusy(message) from exc
    try:
        yield root
    finally:
        lock.release()
