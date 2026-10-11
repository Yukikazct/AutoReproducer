"""Load current PDF evidence extraction without mutating active old readers."""
import importlib
import importlib.util
import hashlib
from pathlib import Path
import sys
import threading


PDF_INPUT_API_VERSION = 3
_UPGRADED_MODULE = f"src._pdf_input_v{PDF_INPUT_API_VERSION}"
_load_lock = threading.RLock()
_project_root = Path(__file__).resolve().parents[1]


def _compatible(module):
    return (vars(module).get("PDF_INPUT_API_VERSION") == PDF_INPUT_API_VERSION
            and callable(vars(module).get("extract_pdf_input"))
            and callable(vars(module).get("resolve_pdf_request")))


def _source_fingerprint():
    return hashlib.sha256(b"".join(
        (_project_root / "src" / name).read_bytes()
        for name in ("pdf_input.py", "repository_routing.py", "repository_evidence.py")
    )).hexdigest()


def _load_private(name, source):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 PDF 解析入口")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_pdf_input():
    with _load_lock:
        current = importlib.import_module("src.pdf_input")
        fingerprint = _source_fingerprint()
        if (_compatible(current)
                and vars(current).get("PDF_SOURCE_FINGERPRINT") == fingerprint):
            return current
        name = _UPGRADED_MODULE + "_" + fingerprint[:16]
        upgraded = sys.modules.get(name)
        if upgraded is not None:
            return upgraded
        routing = _load_private("src._pdf_routing_" + fingerprint[:16],
                                _project_root / "src" / "repository_routing.py")
        upgraded = _load_private(name, _project_root / "src" / "pdf_input.py")
        try:
            if not _compatible(upgraded):
                raise RuntimeError("PDF 解析入口版本不兼容")
            # Bind only this private generation. Old workers keep their own
            # imported functions and globals until their actual run finishes.
            upgraded.extract_current_repository_links = routing.extract_current_repository_links
            upgraded.match_repository_profile = routing.match_repository_profile
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return upgraded
