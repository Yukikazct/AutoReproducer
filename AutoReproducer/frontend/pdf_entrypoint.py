"""Load current PDF evidence extraction without mutating active old readers."""
import importlib
import importlib.util
from pathlib import Path
import sys
import threading


PDF_INPUT_API_VERSION = 2
_UPGRADED_MODULE = f"src._pdf_input_v{PDF_INPUT_API_VERSION}"
_load_lock = threading.RLock()


def _compatible(module):
    return (vars(module).get("PDF_INPUT_API_VERSION") == PDF_INPUT_API_VERSION
            and callable(vars(module).get("extract_pdf_input"))
            and callable(vars(module).get("resolve_pdf_request")))


def load_pdf_input():
    with _load_lock:
        current = importlib.import_module("src.pdf_input")
        if _compatible(current):
            return current
        upgraded = sys.modules.get(_UPGRADED_MODULE)
        if upgraded is not None and _compatible(upgraded):
            return upgraded
        source = Path(__file__).resolve().parents[1] / "src" / "pdf_input.py"
        spec = importlib.util.spec_from_file_location(_UPGRADED_MODULE, source)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载 PDF 解析入口")
        upgraded = importlib.util.module_from_spec(spec)
        sys.modules[_UPGRADED_MODULE] = upgraded
        try:
            spec.loader.exec_module(upgraded)
            if not _compatible(upgraded):
                raise RuntimeError("PDF 解析入口版本不兼容")
        except BaseException:
            sys.modules.pop(_UPGRADED_MODULE, None)
            raise
        return upgraded
