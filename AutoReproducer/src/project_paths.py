"""Shared locations for application data and isolated dependency caches."""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_task_data_root = ContextVar("autorepro_task_data_root", default=None)


def get_task_data_root() -> Path | None:
    """Return this context's task root or an explicitly inherited task marker.

    Ordinary DATA_ROOT/DEPS_ROOT overrides do not enable task-scoped runtime
    caches. Only managed subprocesses receive this separate scope marker.
    """
    root = _task_data_root.get()
    if root is not None:
        return root
    inherited = os.environ.get("AUTOREPRO_TASK_DATA_ROOT", "").strip()
    return Path(inherited).expanduser().absolute() if inherited else None


@contextmanager
def task_data_scope(data_root):
    """Use a task's storage roots without changing the process environment.

    Nested contexts restore their caller's roots, including on cancellation.
    New threads must enter their own scope or run with an explicitly copied
    context; subprocesses receive these roots through their launch environment.
    """
    if data_root is None or not os.fspath(data_root).strip():
        raise ValueError("task data root must be an explicit nonempty path")
    root = Path(data_root).expanduser().absolute()
    token = _task_data_root.set(root)
    try:
        yield root
    finally:
        _task_data_root.reset(token)


def get_data_root(value=None) -> Path:
    """Resolve an explicit root, task scope, environment, or source default.

    Defaults never depend on the launching directory. Relative overrides are
    relative to the caller's working directory, as with ordinary CLI paths.
    Reading the environment on demand also supports embedded application use.
    """
    configured = value if value is not None else (
        get_task_data_root() or os.environ.get("AUTOREPRO_DATA_ROOT", "").strip())
    return Path(configured or PROJECT_ROOT / "data").expanduser().absolute()


def get_deps_root(value=None) -> Path:
    """Resolve an explicit cache, task cache, environment, or <data>/deps."""
    task_root = get_task_data_root()
    configured = value if value is not None else (
        task_root / "deps" if task_root is not None else os.environ.get("AUTOREPRO_DEPS_ROOT", "").strip())
    return Path(configured or get_data_root() / "deps").expanduser().absolute()
