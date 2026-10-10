"""Resolve the web entrypoint even when Streamlit retains an older backend.

The version describes entrypoint capabilities, including preset host recovery.
Bump it in both modules when that contract changes. Loading an upgraded copy in
its own namespace preserves the globals of any already running pipeline thread.
"""
import importlib
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType


BACKEND_API_VERSION = 4
_UPGRADED_MODULE = f"frontend._backend_pipeline_v{BACKEND_API_VERSION}"
_load_lock = threading.RLock()


def _compatible(module: ModuleType) -> bool:
    return (
        vars(module).get("BACKEND_API_VERSION") == BACKEND_API_VERSION
        and callable(vars(module).get("ProgressStore"))
        and callable(vars(module).get("run_pipeline_background"))
    )


def load_backend_pipeline() -> ModuleType:
    """Use the current backend, or isolate an upgrade from a cached old copy."""
    with _load_lock:
        backend = importlib.import_module("frontend.backend_pipeline")
        if _compatible(backend):
            return backend
        upgraded = sys.modules.get(_UPGRADED_MODULE)
        if upgraded is not None and _compatible(upgraded):
            return upgraded
        source = Path(__file__).resolve().with_name("backend_pipeline.py")
        spec = importlib.util.spec_from_file_location(_UPGRADED_MODULE, source)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载后台流水线入口")
        upgraded = importlib.util.module_from_spec(spec)
        sys.modules[_UPGRADED_MODULE] = upgraded
        try:
            spec.loader.exec_module(upgraded)
            if not _compatible(upgraded):
                raise RuntimeError("后台流水线入口版本不兼容")
        except BaseException:
            if sys.modules.get(_UPGRADED_MODULE) is upgraded:
                del sys.modules[_UPGRADED_MODULE]
            raise
        return upgraded
