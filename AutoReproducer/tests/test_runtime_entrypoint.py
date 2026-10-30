"""An upgraded backend must not call a cached runner with an old signature."""
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

from frontend import runtime_entrypoint as entrypoint


CURRENT_RUNTIME = importlib.import_module("src.runtime_preparation")


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


def cached_legacy_runtime(monkeypatch):
    legacy = ModuleType("src.runtime_preparation")
    legacy.owner_state = {"active": True}
    exec("def run_owned_process(argv):\n    return owner_state\n", vars(legacy))
    monkeypatch.setitem(sys.modules, "src.runtime_preparation", legacy)
    return legacy


def test_current_runtime_preserves_existing_module_and_test_mocks(monkeypatch):
    runner = Mock()
    monkeypatch.setattr(CURRENT_RUNTIME, "run_owned_process", runner)
    loaded = entrypoint.load_runtime_preparation()
    assert loaded is CURRENT_RUNTIME
    assert loaded.run_owned_process is runner
    assert entrypoint._UPGRADED_MODULE not in sys.modules


def test_cached_old_runtime_gets_real_owned_runner_without_changing_old_globals(monkeypatch):
    legacy = cached_legacy_runtime(monkeypatch)
    old_runner, old_globals = legacy.run_owned_process, vars(legacy)
    upgraded = entrypoint.load_runtime_preparation()
    assert upgraded is not legacy
    assert entrypoint._compatible(upgraded)
    assert Path(upgraded.__file__).resolve() == Path(CURRENT_RUNTIME.__file__).resolve()
    assert upgraded.run_owned_process.__globals__ is vars(upgraded)
    # Execute an actual process with the newly introduced argument, rather
    # than only asserting that a replacement function was imported.
    cleanup = Mock()
    completed = upgraded.run_owned_process(
        [sys.executable, "-S", "-c", "pass"], contain_session=True,
        timeout_s=10, on_cleanup=cleanup)
    assert completed.returncode == 0 and completed.stdout == completed.stderr == ""
    cleanup.assert_called_once_with()
    assert sys.modules["src.runtime_preparation"] is legacy
    assert legacy.run_owned_process is old_runner
    assert old_runner.__globals__ is old_globals
    assert old_runner([]) is legacy.owner_state
    assert legacy.owner_state == {"active": True}
    assert "OWNED_PROCESS_API_VERSION" not in vars(legacy)
    assert entrypoint.load_runtime_preparation() is upgraded


def test_prepared_backend_uses_isolated_runtime_exports_with_legacy_cached(monkeypatch, tmp_path):
    from frontend import backend_pipeline as backend

    legacy = cached_legacy_runtime(monkeypatch)
    upgraded = entrypoint.load_runtime_preparation()
    prepared = upgraded.RuntimePreparation(sys.executable, True, False, "fixture", "fixture")
    prepare = Mock(return_value=prepared)
    release = Mock(return_value={"status": "released", "errors": []})
    requests = []

    def worker(command, **kwargs):
        requests.append(json.loads(kwargs["input"]))
        assert kwargs["contain_session"] is True
        backend.ProgressStore(requests[-1]["progress_path"], reset=False).emit({
            "type": "done", "result": {"state": "COMPLETED", "data": {"fixture": True}}})
        kwargs["on_cleanup"]()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(upgraded, "prepare_runtime", prepare)
    monkeypatch.setattr(upgraded, "run_owned_process", worker)
    monkeypatch.setattr(upgraded, "release_worker_environment", release)
    progress = str(tmp_path / "progress.jsonl")
    result = backend._run_prepared_pipeline(
        backend.ProgressStore(progress), {"progress_path": progress}, "fixture")
    assert result["state"] == "COMPLETED" and result["data"]["fixture"] is True
    assert len(requests) == 1 and requests[0]["_managed_runtime"] is True
    prepare.assert_called_once()
    release.assert_called_once_with(prepared)
    assert sys.modules["src.runtime_preparation"] is legacy
    assert legacy.run_owned_process([]) is legacy.owner_state


def test_concurrent_runtime_upgrades_publish_one_complete_module(monkeypatch):
    legacy = cached_legacy_runtime(monkeypatch)
    real_spec = importlib.util.spec_from_file_location
    executions = []

    def spec(name, source):
        result = real_spec(name, source)
        real_execute = result.loader.exec_module

        def execute(module):
            executions.append(module)
            time.sleep(.03)
            real_execute(module)

        result.loader.exec_module = execute
        return result

    monkeypatch.setattr(entrypoint.importlib.util, "spec_from_file_location", spec)
    barrier = threading.Barrier(4)

    def load():
        barrier.wait(timeout=5)
        return entrypoint.load_runtime_preparation()

    with ThreadPoolExecutor(max_workers=4) as pool:
        modules = list(pool.map(lambda _: load(), range(4)))
    assert len(executions) == 1
    assert all(module is modules[0] and entrypoint._compatible(module) for module in modules)
    assert sys.modules["src.runtime_preparation"] is legacy


@pytest.mark.parametrize("failure", ["exception", "cancel", "incompatible"])
def test_failed_runtime_upgrade_preserves_legacy_and_removes_partial_alias(monkeypatch, failure):
    legacy = cached_legacy_runtime(monkeypatch)
    error = KeyboardInterrupt("cancel") if failure == "cancel" else RuntimeError("broken runtime source")

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
        entrypoint.load_runtime_preparation()
    assert entrypoint._UPGRADED_MODULE not in sys.modules
    assert sys.modules["src.runtime_preparation"] is legacy
    assert legacy.run_owned_process([]) is legacy.owner_state
