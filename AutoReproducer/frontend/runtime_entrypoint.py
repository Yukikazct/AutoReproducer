"""Resolve owned-process capabilities without reloading an active old runtime.

Streamlit can retain ``src.runtime_preparation`` while loading a newer backend.
Load an isolated copy when its cached runner cannot own a POSIX worker session;
existing tasks keep the globals and classes of their original runtime module.
"""
import importlib
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType


OWNED_PROCESS_API_VERSION = 1
_UPGRADED_MODULE = f"src._runtime_preparation_owned_v{OWNED_PROCESS_API_VERSION}"
_load_lock = threading.RLock()


def _compatible(module: ModuleType) -> bool:
    return (
        vars(module).get("OWNED_PROCESS_API_VERSION") == OWNED_PROCESS_API_VERSION
        and all(callable(vars(module).get(name)) for name in (
            "prepare_runtime", "run_owned_process", "safe_runtime_diagnostic",
            "release_worker_environment", "ProcessCleanupError"))
    )


def load_runtime_preparation() -> ModuleType:
    """Return a runtime whose runner accepts the owned-session contract."""
    with _load_lock:
        runtime = importlib.import_module("src.runtime_preparation")
        if _compatible(runtime):
            return runtime
        upgraded = sys.modules.get(_UPGRADED_MODULE)
        if upgraded is not None and _compatible(upgraded):
            return upgraded
        source = Path(__file__).resolve().parents[1] / "src" / "runtime_preparation.py"
        spec = importlib.util.spec_from_file_location(_UPGRADED_MODULE, source)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载工作进程运行环境")
        upgraded = importlib.util.module_from_spec(spec)
        sys.modules[_UPGRADED_MODULE] = upgraded
        try:
            spec.loader.exec_module(upgraded)
            if not _compatible(upgraded):
                raise RuntimeError("工作进程运行环境版本不兼容")
        except BaseException:
            if sys.modules.get(_UPGRADED_MODULE) is upgraded:
                del sys.modules[_UPGRADED_MODULE]
            raise
        return upgraded
