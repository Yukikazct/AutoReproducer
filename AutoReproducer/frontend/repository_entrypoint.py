"""Expose new reviewed repository capabilities to an already running web app.

Active tasks retain their original modules and globals. The new ReZero route
continues in the existing owned fresh-worker boundary, rather than executing a
mixture of old cached adapters and new source files in the Streamlit process.
"""
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import threading


from src.repository_routing import REZERO_PROFILE_ID


PDF_INPUT_API_VERSION = 3
REPOSITORY_ENTRYPOINT_API_VERSION = 2
_load_lock = threading.RLock()
_project_root = Path(__file__).resolve().parents[1]


def _isolated_source(relative, prefix):
    source = _project_root / relative
    content_hash = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    name = prefix + "_" + content_hash
    with _load_lock:
        module = sys.modules.get(name)
        if module is not None:
            return module
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载当前仓库复现入口")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module


def load_repository_profiles():
    current = importlib.import_module("src.repository_profiles")
    if REZERO_PROFILE_ID in vars(current).get("PROFILE_LABELS", {}):
        return current
    return _isolated_source("src/repository_profiles.py", "src._repository_profiles_pdf_v1")


def profile_labels():
    return dict(load_repository_profiles().PROFILE_LABELS)


def get_profile(profile_id):
    # Preserve existing public injection points for old supported profiles.
    if profile_id != REZERO_PROFILE_ID:
        return importlib.import_module("src.repository_profiles").get_profile(profile_id)
    return load_repository_profiles().get_profile(profile_id)


def load_pdf_input():
    loader = _isolated_source("frontend/pdf_entrypoint.py", "frontend._current_pdf_loader")
    return loader.load_pdf_input()


def load_repository_backend():
    """Use a private current backend; never replace an old task's globals."""
    backend = _isolated_source("frontend/backend_pipeline.py", "frontend._backend_repository_v1")
    with _load_lock:
        if vars(backend).get("_repository_fresh_worker_installed"):
            return backend
        original = backend.run_pipeline_core
        signature = inspect.signature(original)

        def fresh_repository_core(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            request = dict(bound.arguments)
            generic_pdf = bool(request.get("pdf_path") and not request.get("experiment_profile"))
            if ((request.get("experiment_profile") != REZERO_PROFILE_ID and not generic_pdf)
                    or request.get("mock_mode") or request.get("_managed_runtime")):
                return original(*args, **kwargs)
            progress_path = request.pop("progress_path")
            store = backend.ProgressStore(progress_path, reset=not request.get("_append_progress"))
            return backend._run_prepared_pipeline(
                store, {"progress_path": str(Path(progress_path).resolve()), **request},
                "正在核验运行环境，并在独立后台进程加载本次 PDF 作者仓库实验。")

        backend.run_pipeline_core = fresh_repository_core
        backend._repository_fresh_worker_installed = True
    return backend


def background_entrypoint(previous_background):
    """New PDF requests get a fresh worker; active tasks retain their owner."""
    def dispatch(progress_path, **request):
        profile = request.get("experiment_profile")
        if not profile and request.get("pdf_path") and not request.get("mock_mode"):
            parser = load_pdf_input()
            try:
                resolved = parser.resolve_pdf_request(request)
            except parser.PDFInputError:
                # The established background rejection path owns invalid inputs.
                return previous_background(progress_path, **request)
            profile = resolved.get("experiment_profile")
            if profile == REZERO_PROFILE_ID:
                request = {**request, "experiment_profile": profile,
                           "pdf_resolution": resolved["pdf_resolution"]}
        generic_pdf = bool(request.get("pdf_path") and not profile)
        if (profile == REZERO_PROFILE_ID or generic_pdf) and not request.get("mock_mode"):
            return load_repository_backend().run_pipeline_background(progress_path, **request)
        return previous_background(progress_path, **request)
    return dispatch
