"""Recover a cached web entrypoint without changing an active worker's globals."""
import importlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import frontend.history_manager as history
import frontend.pipeline_entrypoint as entrypoint
import src.runtime_preparation as runtime


CURRENT_BACKEND = importlib.import_module("frontend.backend_pipeline")
APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def isolated_upgrade_cache():
    name = entrypoint._UPGRADED_MODULE
    previous = sys.modules.pop(name, None)
    try:
        yield
    finally:
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous


def cached_legacy_backend(monkeypatch):
    legacy = ModuleType("frontend.backend_pipeline")
    legacy.owner_state = {"active": True}
    exec("def run_pipeline_core():\n    return owner_state\n", vars(legacy))
    legacy.run_pipeline_background = Mock(side_effect=AssertionError("cached legacy entrypoint ran"))
    monkeypatch.setitem(sys.modules, "frontend.backend_pipeline", legacy)
    return legacy


def test_current_backend_returns_same_module_and_preserves_test_mocks(monkeypatch):
    start = Mock()
    monkeypatch.setattr(CURRENT_BACKEND, "run_pipeline_background", start)
    module = entrypoint.load_backend_pipeline()
    assert module is CURRENT_BACKEND
    assert module.run_pipeline_background is start
    assert entrypoint._UPGRADED_MODULE not in sys.modules


def test_stale_backend_upgrade_keeps_existing_worker_globals(monkeypatch):
    legacy = cached_legacy_backend(monkeypatch)
    old_core, old_start, old_globals = legacy.run_pipeline_core, legacy.run_pipeline_background, vars(legacy)
    upgraded = entrypoint.load_backend_pipeline()
    assert upgraded is not legacy and upgraded.BACKEND_API_VERSION == entrypoint.BACKEND_API_VERSION
    assert Path(upgraded.__file__).resolve() == APP.parent / "frontend" / "backend_pipeline.py"
    assert upgraded.run_pipeline_core.__globals__ is vars(upgraded)
    assert sys.modules["frontend.backend_pipeline"] is legacy
    assert legacy.run_pipeline_core is old_core and legacy.run_pipeline_background is old_start
    assert old_core.__globals__ is old_globals and old_core() is legacy.owner_state
    assert legacy.owner_state == {"active": True}
    assert entrypoint.load_backend_pipeline() is upgraded
    old_start.assert_not_called()


def test_concurrent_upgrade_loads_publish_one_complete_module(monkeypatch):
    legacy = cached_legacy_backend(monkeypatch)
    real_spec = importlib.util.spec_from_file_location
    executions = []

    def spec(name, path):
        result = real_spec(name, path)
        real_execute = result.loader.exec_module
        def execute(module):
            executions.append(module)
            time.sleep(.03)  # expose the partially registered module to other callers
            real_execute(module)
        result.loader.exec_module = execute
        return result

    monkeypatch.setattr(entrypoint.importlib.util, "spec_from_file_location", spec)
    barrier = threading.Barrier(6)
    def load():
        barrier.wait(timeout=5)
        return entrypoint.load_backend_pipeline()
    with ThreadPoolExecutor(max_workers=6) as pool:
        modules = list(pool.map(lambda _: load(), range(6)))
    assert len(executions) == 1
    assert all(module is modules[0] for module in modules)
    assert all(entrypoint._compatible(module) for module in modules)
    assert sys.modules["frontend.backend_pipeline"] is legacy


@pytest.mark.parametrize("failure", ["exception", "cancel", "incompatible"])
def test_failed_upgrade_removes_partial_alias_and_preserves_legacy(monkeypatch, failure):
    legacy = cached_legacy_backend(monkeypatch)
    error = KeyboardInterrupt("cancel") if failure == "cancel" else RuntimeError("broken backend source")
    class BrokenLoader:
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            assert sys.modules[entrypoint._UPGRADED_MODULE] is module
            module.partially_initialized = True
            if failure != "incompatible":
                raise error
    monkeypatch.setattr(entrypoint.importlib.util, "spec_from_file_location",
                        lambda name, source: importlib.util.spec_from_loader(name, BrokenLoader()))
    expected = KeyboardInterrupt if failure == "cancel" else RuntimeError
    with pytest.raises(expected):
        entrypoint.load_backend_pipeline()
    assert entrypoint._UPGRADED_MODULE not in sys.modules
    assert sys.modules["frontend.backend_pipeline"] is legacy
    assert legacy.run_pipeline_core() is legacy.owner_state
    legacy.run_pipeline_background.assert_not_called()


@pytest.mark.parametrize("profile,prepare_environment", [
    ("dlinear_etth1_reference", False),
    ("siren_camera_quick", True),
])
def test_real_app_button_with_cached_backend_reaches_safe_worker(monkeypatch, tmp_path, profile, prepare_environment):
    legacy = cached_legacy_backend(monkeypatch)
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    monkeypatch.setattr(runtime, "runtime_requirement", lambda: "Switch Store host to safe Python")
    prepared = runtime.RuntimePreparation(str(tmp_path / "safe-python.exe"), True, False,
                                          "Store host", "cpython-312")
    prepare = Mock(return_value=prepared)
    monkeypatch.setattr(runtime, "prepare_runtime", prepare)
    requests, threads, progress_paths = [], [], []

    def worker(command, **kwargs):
        payload = json.loads(kwargs["input"])
        requests.append(payload)
        assert command[0] == prepared.executable
        backend = sys.modules[entrypoint._UPGRADED_MODULE]
        store = backend.ProgressStore(payload["progress_path"], reset=False)
        result = {"state": "COMPLETED", "error": None,
                  "data": {"runtime_preparation": payload["_runtime_preparation"]},
                  "audit_logs": [], "audit_stats": {}}
        store.emit({"type": "done", "result": result})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime, "run_owned_process", worker)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["input_mode"] = "官方仓库预设"
    app.session_state["experiment_profile"] = profile
    app.session_state["docker_probe"] = (False, "test uses local execution")
    app.session_state["mock_mode"] = False
    app.session_state["repository_llm_review"] = False
    app.session_state["repository_result_review"] = False
    app.session_state["repository_prepare_only"] = False
    app.session_state["method_action"] = "准备实验环境"
    app.session_state["method_optimization"] = "off"
    app.session_state["method_llm_review"] = False
    try:
        app.run()
        assert not app.exception
        upgraded = sys.modules[entrypoint._UPGRADED_MODULE]
        host_factory = Mock(side_effect=AssertionError("Store host constructed an orchestrator"))
        monkeypatch.setattr(upgraded, "_create_orchestrator", host_factory)
        real_start = upgraded.run_pipeline_background
        def start(progress_path, **kwargs):
            path = Path(progress_path)
            assert not path.exists(), "test must not overwrite an existing progress artifact"
            progress_paths.append(path)
            thread = real_start(progress_path, **kwargs)
            threads.append(thread)
            return thread
        monkeypatch.setattr(upgraded, "run_pipeline_background", start)
        next(button for button in app.sidebar.button if "开始复现" in button.label).click().run()
        assert not app.exception
        assert len(threads) == 1
        threads[0].join(timeout=5)
        assert not threads[0].is_alive()
        app.run()
        assert not app.exception
        request, = requests
        assert request["experiment_profile"] == profile
        assert request["prepare_environment"] is prepare_environment
        assert request["prepare_only"] is False
        assert request["mock_mode"] is False and request["use_docker"] is False
        assert request["_managed_runtime"] and request["_append_progress"]
        prepare.assert_called_once()
        host_factory.assert_not_called()
        legacy.run_pipeline_background.assert_not_called()
        assert sys.modules["frontend.backend_pipeline"] is legacy
        snapshot = upgraded.ProgressStore.read_snapshot(str(progress_paths[0]))
        assert snapshot["done"] and not snapshot["running"]
        stage, = snapshot["pipeline_stages"]
        assert stage["id"] == "prepare_runtime" and stage["status"] == "success"
        assert app.session_state["result"]["data"]["runtime_preparation"]["executable"] == prepared.executable
        assert app.session_state["running"] is False
    finally:
        for thread in threads:
            thread.join(timeout=5)
        for path in progress_paths:
            path.unlink(missing_ok=True)
