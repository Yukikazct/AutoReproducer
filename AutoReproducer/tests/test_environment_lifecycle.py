"""Reports outlive disposable dependencies; owners serialize the whole run."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

import src.agents.code_executor as executor_module
import src.environment_lifecycle as lifecycle
from src.dependency_cache import cache_guard, DependencyCacheBusy
from src.environment_lifecycle import managed_environment, release_run_environment


def payload(path, name="package.py"):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text("VALUE = 42\n", encoding="utf-8")
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / "deps"
    monkeypatch.setattr(executor_module, "DEPS_CACHE_ROOT", root)
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(root))
    monkeypatch.setenv("AUTOREPRO_DATA_ROOT", str(tmp_path / "data"))
    from src import runtime_preparation
    monkeypatch.setattr(runtime_preparation, "release_runtime_downloads", lambda: {
        "status": "released", "removed": [], "errors": [], "freed_bytes": 0})
    return root


def test_report_and_evidence_are_saved_before_release(root, tmp_path):
    stale = payload(root / "repository" / "old-runtime" / "old")
    run = tmp_path / "data" / "runs" / "example"
    run.mkdir(parents=True)
    report = run / "report.md"
    artifact = run / "figure.png"
    events = []
    for attempt in range(2):
        with managed_environment() as lease:
            assert not stale.exists()
            deps = payload(root / "repository" / "current" / "requirements")
            payload(root / "heal-missing")
            payload(root / ".pip-cache", "wheel.whl")
            artifact.write_bytes(b"retained evidence")
            report.write_text("# Finished report", encoding="utf-8")
            lease.result = {"state": "COMPLETED", "error": None,
                            "data": {"run_dir": str(run), "report": report.read_text("utf-8")}}
            (run / "result.json").write_text(json.dumps(lease.result), encoding="utf-8")
            assert deps.exists()
            events.append("report saved")
        assert all(path.name == ".autorepro-cache.lock" for path in root.iterdir())
        assert report.read_text("utf-8") == "# Finished report"
        assert artifact.read_bytes() == b"retained evidence"
        saved = json.loads((run / "result.json").read_text("utf-8"))
        assert saved["data"]["environment_cleanup"]["status"] == "released"
        assert lease.result["data"]["environment_cleanup"]["remaining"] == []
        events.append("released")
    assert events == ["report saved", "released"] * 2


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_exception_and_cancellation_release_partial_installs(root, exception):
    run_dir = root.parent / "run"
    run_dir.mkdir()
    with pytest.raises(exception):
        with managed_environment():
            lifecycle.bind_environment_run(run_dir)
            payload(root / "unfinished")
            raise exception("stopped")
    assert not (root / "unfinished").exists()
    assert json.loads((run_dir / "environment_cleanup.json").read_text("utf-8"))["status"] == "released"
    with cache_guard(root, cleanup=True):
        pass


def test_nested_entrypoints_keep_dependencies_until_outer_report(root):
    @release_run_environment
    def inner():
        payload(root / "active")
        return {"data": {}}

    with managed_environment() as lease:
        lease.result = inner()
        assert (root / "active").is_dir()
    assert not (root / "active").exists()


def test_requirements_switch_removes_old_environment_and_its_heal_packages(root):
    with managed_environment():
        first, second = root / "first", root / "second"
        lifecycle.select_dependency_environment(root, first)
        payload(first)
        payload(root / "heal-old")
        lifecycle.select_dependency_environment(root, second)
        assert not first.exists() and not (root / "heal-old").exists()
        payload(second)
        payload(root / "heal-current")
        lifecycle.select_dependency_environment(root, second)
        assert (root / "heal-current").exists()
        assert [p.name for p in root.iterdir() if p.is_dir()] == ["heal-current", "second"]


def test_release_failure_is_visible_and_blocks_the_next_run(root, monkeypatch):
    from src.safety import files
    original = files.remove_tree
    locked = root / "locked"

    def remove(path):
        if Path(path) == locked:
            raise PermissionError("file still open")
        original(path)

    monkeypatch.setattr(files, "remove_tree", remove)
    with managed_environment() as lease:
        payload(locked)
        lease.result = {"state": "COMPLETED", "error": None, "data": {}}
    assert lease.result["state"] == "ERROR"
    receipt = lease.result["data"]["environment_cleanup"]
    assert receipt["status"] == "error" and receipt["remaining"] == ["locked"]
    with pytest.raises(lifecycle.EnvironmentCleanupError):
        with managed_environment():
            pytest.fail("must not create another environment")


def test_linked_environment_preserves_external_files_and_blocks_install(root, tmp_path, require_symlinks):
    outside = payload(tmp_path / "outside")
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(lifecycle.EnvironmentCleanupError):
        with managed_environment():
            pytest.fail("linked environment is not disposable")
    assert (outside / "package.py").is_file()


def test_misconfigured_root_cannot_remove_application_data(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOREPRO_DATA_ROOT", str(tmp_path))
    keep = tmp_path / "report.md"
    keep.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="application or data"):
        with managed_environment(tmp_path):
            pass
    assert keep.read_text("utf-8") == "keep"


def test_another_process_waits_until_report_and_release_finish(root, tmp_path):
    started, entered = tmp_path / "started", tmp_path / "entered"
    script = """
import sys
from pathlib import Path
from src.environment_lifecycle import managed_environment
Path(sys.argv[2]).write_text('started')
with managed_environment(sys.argv[1]):
    assert not (Path(sys.argv[1]) / 'owner').exists()
    Path(sys.argv[3]).write_text('entered')
"""
    with managed_environment():
        payload(root / "owner")
        child = subprocess.Popen([sys.executable, "-c", script, str(root), str(started), str(entered)])
        try:
            import time
            deadline = time.monotonic() + 15
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert started.exists()
            assert not entered.exists() and child.poll() is None
            with pytest.raises(DependencyCacheBusy), cache_guard(root, cleanup=True):
                pass
        except BaseException:
            child.kill()
            child.wait(timeout=10)
            raise
    assert child.wait(timeout=20) == 0
    assert entered.is_file()


def test_actual_installs_replace_environment_and_disable_pip_cache(root, tmp_path, monkeypatch):
    agent = executor_module.CodeExecutorAgent(None, logger=Mock())
    agent.resource_events = Mock()
    agent.env_config = {"requirements_txt": "first==1"}
    calls = []

    def install(argv, **kwargs):
        assert kwargs["env"]["PIP_NO_CACHE_DIR"] == "1"
        directory = Path(argv[argv.index("--target") + 1])
        assert not any((p / "package.py").exists() for p in root.iterdir() if p.is_dir())
        payload(directory)
        calls.append(directory)
        return subprocess.CompletedProcess(argv, 0, "installed", "")

    monkeypatch.setattr(agent, "_run_dependency_command", install)
    for _ in range(2):
        with managed_environment():
            assert agent._ensure_local_deps(str(tmp_path)) is None
            previous = Path(agent._deps_dir)
            agent.env_config["requirements_txt"] = "second==1"
            assert agent._ensure_local_deps(str(tmp_path)) is None
            assert not previous.exists()
        assert not Path(agent._deps_dir).exists()
        agent.env_config["requirements_txt"] = "first==1"
    assert len(calls) == 4


def test_backend_done_and_archive_follow_release(root, tmp_path, monkeypatch):
    import frontend.backend_pipeline as backend
    from src.audit.audit_logger import AuditLogger

    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, request, on_event=None):
            payload(root / "current")
            return {"state": "COMPLETED", "error": None,
                    "data": {"report": "# Report", "paper_title": "Example"}}

    monkeypatch.setattr(backend, "Orchestrator", Pipeline, raising=False)
    monkeypatch.setattr(backend, "AuditLogger", lambda: AuditLogger(log_dir=str(tmp_path / "audit")))
    progress = tmp_path / "progress.jsonl"
    original_emit = backend.ProgressStore.emit

    def emit(store, event):
        if event["type"] == "done":
            assert not (root / "current").exists()
            assert event["result"]["data"]["environment_cleanup"]["status"] == "released"
        return original_emit(store, event)

    monkeypatch.setattr(backend.ProgressStore, "emit", emit)
    result = backend.run_pipeline_core(progress_path=str(progress), paper_title="Example", mock_mode=True)
    archive = Path(result["report_path"]).with_suffix(".result.json")
    assert json.loads(archive.read_text("utf-8"))["data"]["environment_cleanup"]["status"] == "released"


def test_crashed_owner_leaves_one_environment_until_next_owner_recovers(root, tmp_path):
    signal = tmp_path / "installed"
    script = """
import sys,time
from pathlib import Path
from src.environment_lifecycle import managed_environment
with managed_environment(sys.argv[1]):
    environment = Path(sys.argv[1]) / 'crashed'
    environment.mkdir()
    (environment / 'partial.whl').write_bytes(b'incomplete install')
    Path(sys.argv[2]).write_text('installed')
    time.sleep(60)
"""
    child = subprocess.Popen([sys.executable, "-c", script, str(root), str(signal)])
    try:
        import time
        deadline = time.monotonic() + 15
        while not signal.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert signal.is_file()
    finally:
        child.kill()
        child.wait(timeout=10)
    assert (root / "crashed").is_dir()
    with managed_environment() as lease:
        assert not (root / "crashed").exists()
        payload(root / "fresh")
        lease.result = {"data": {}}
    assert lease.result["data"]["environment_cleanup"]["before_run"]["removed"]
    assert not (root / "fresh").exists()


def test_bootstrap_download_release_keeps_application_runtime(tmp_path, monkeypatch):
    from src import runtime_preparation as runtime
    monkeypatch.setattr(runtime, "_machine_key", lambda root: "test-machine")
    root = tmp_path / "data" / "runtime" / "test-machine"
    application = payload(root / "app-venv")
    interpreter = payload(root / "python312")
    for name in ("wheels", "pip-cache"):
        payload(root / name, "download.whl")
    installer = root / f"python-{runtime.PYTHON_RELEASE}-amd64.exe"
    installer.write_bytes(b"installer")
    receipt = runtime.release_runtime_downloads(tmp_path)
    assert receipt["status"] == "released" and receipt["freed_bytes"] > 0
    assert {item["path"] for item in receipt["removed"]} == {"wheels", "pip-cache", installer.name}
    assert application.is_dir() and interpreter.is_dir()
    assert runtime.release_runtime_downloads(tmp_path)["removed"] == []


def test_bootstrap_download_failure_marks_run_failed(root, monkeypatch):
    from src import runtime_preparation as runtime
    failure = {"status": "error", "errors": [{"path": "wheels", "reason": "locked"}]}
    monkeypatch.setattr(runtime, "release_runtime_downloads", Mock(side_effect=[
        {"status": "released", "errors": []}, failure, failure]))
    with managed_environment() as lease:
        payload(root / "current")
        lease.result = {"state": "COMPLETED", "data": {}}
    assert not (root / "current").exists()
    assert lease.result["state"] == "ERROR"
    assert lease.result["data"]["environment_cleanup"]["runtime_downloads"]["status"] == "error"
    with pytest.raises(lifecycle.EnvironmentCleanupError):
        with managed_environment():
            pytest.fail("stale downloads must be released first")


def test_cleanup_error_does_not_replace_primary_interruption(root, monkeypatch):
    from src.safety import files
    monkeypatch.setattr(files, "remove_tree", Mock(side_effect=PermissionError("locked")))
    original = KeyboardInterrupt("original cancellation")
    with pytest.raises(KeyboardInterrupt) as error:
        with managed_environment():
            payload(root / "active")
            raise original
    assert error.value is original
    assert any("locked" in note for note in original.__notes__)


def test_real_offline_pip_install_executes_then_is_released(root, tmp_path, monkeypatch):
    import zipfile
    from src.runtime_preparation import run_owned_process

    wheel = tmp_path / "autorepro_lifecycle_smoke-1.0-py3-none-any.whl"
    metadata = "autorepro_lifecycle_smoke-1.0.dist-info"
    contents = {
        "autorepro_lifecycle_smoke/__init__.py": "VALUE = 42\n",
        f"{metadata}/METADATA": "Metadata-Version: 2.1\nName: autorepro-lifecycle-smoke\nVersion: 1.0\n",
        f"{metadata}/WHEEL": "Wheel-Version: 1.0\nGenerator: lifecycle-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    contents[f"{metadata}/RECORD"] = "\n".join(f"{name},," for name in [*contents, f"{metadata}/RECORD"]) + "\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setattr(executor_module, "PIP_FIND_LINKS", "")
    agent = executor_module.CodeExecutorAgent(None, logger=Mock())
    agent.resource_events = Mock()
    agent.env_config = {"requirements_txt": wheel.as_posix()}
    for _ in range(2):
        with managed_environment() as lease:
            assert agent._ensure_local_deps(str(tmp_path)) is None
            environment = Path(agent._deps_dir)
            command = [sys.executable, "-c", "import autorepro_lifecycle_smoke as pkg; print(pkg.VALUE)"]
            completed = run_owned_process(command, env=agent._exec_env(), timeout_s=30, clean_env=False)
            assert completed.returncode == 0 and completed.stdout.strip() == "42"
            lease.result = {"state": "COMPLETED", "data": {"report": "Installed and executed"}}
        assert not environment.exists()
        completed = run_owned_process(command, env=agent._exec_env(), timeout_s=30, clean_env=False)
        assert completed.returncode != 0 and "ModuleNotFoundError" in completed.stderr


def test_supervisor_reclaims_a_worker_that_exits_without_unwinding(root, tmp_path):
    from src import runtime_preparation as runtime
    program = """
import os,sys
from pathlib import Path
from src.environment_lifecycle import managed_environment
with managed_environment(sys.argv[1]):
    target = Path(sys.argv[1]) / 'unwound'
    target.mkdir()
    (target / 'installed.py').write_text('VALUE = 42')
    Path(sys.argv[2]).write_text('report saved')
    os._exit(1)
"""
    report = tmp_path / "report.md"
    completed = runtime.run_owned_process([sys.executable, "-c", program, str(root), str(report)], timeout_s=30)
    assert completed.returncode == 1 and (root / "unwound").is_dir()
    prepared = runtime.RuntimePreparation(sys.executable, True, False, "test", "test")
    receipt = runtime.release_worker_environment(prepared)
    assert receipt["status"] == "released" and not (root / "unwound").exists()
    assert report.read_text("utf-8") == "report saved"


def test_supervisor_without_filelock_delegates_cleanup_to_verified_interpreter(root, monkeypatch):
    import builtins
    from src import runtime_preparation as runtime
    payload(root / "orphan")
    original_import = builtins.__import__

    def without_filelock(name, *args, **kwargs):
        if name == "src.dependency_cache":
            raise ModuleNotFoundError("No module named filelock", name="filelock")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_filelock)
    prepared = runtime.RuntimePreparation(sys.executable, True, False, "bare host", "test")
    receipt = runtime.release_worker_environment(prepared)
    assert receipt["status"] == "released" and not (root / "orphan").exists()


def test_supervisor_never_deletes_a_new_owners_dependencies(root):
    from src import runtime_preparation as runtime
    prepared = runtime.RuntimePreparation(sys.executable, True, False, "test", "test")
    with managed_environment():
        payload(root / "new-owner")
        assert runtime.release_worker_environment(prepared)["status"] == "in_use"
        assert (root / "new-owner").is_dir()


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_failed_bootstrap_releases_downloads_before_returning(tmp_path, monkeypatch, exception):
    from src import runtime_preparation as runtime
    monkeypatch.setattr(runtime, "_machine_key", lambda root: "test-machine")
    root = tmp_path / "data" / "runtime" / "test-machine"
    failure = exception("bootstrap interrupted")

    def probe(*args, **kwargs):
        payload(root / "wheels", "package.whl")
        payload(root / "pip-cache", "package.whl")
        raise failure

    monkeypatch.setattr(runtime, "_probe", probe)
    with pytest.raises(exception) as caught:
        runtime.prepare_runtime(tmp_path)
    assert caught.value is failure
    assert not (root / "wheels").exists() and not (root / "pip-cache").exists()


def test_failed_bootstrap_cleanup_works_on_a_standard_library_only_host(tmp_path):
    from src.runtime_preparation import run_owned_process
    program = """
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from src import runtime_preparation as runtime
runtime._machine_key = lambda root: 'test-machine'
project = Path(sys.argv[2])
root = project / 'data/runtime/test-machine'
def probe(*args,**kwargs):
    (root/'wheels').mkdir(parents=True)
    (root/'wheels'/'partial.whl').write_bytes(b'partial')
    raise RuntimeError('interrupted install')
runtime._probe = probe
try:
    runtime.prepare_runtime(project)
except RuntimeError:
    pass
else:
    raise AssertionError('expected preparation failure')
assert not (root/'wheels').exists()
assert 'packaging' not in sys.modules and 'filelock' not in sys.modules
print('released without app dependencies')
"""
    result = run_owned_process([sys.executable, "-S", "-c", program,
                                str(Path(__file__).resolve().parents[1]), str(tmp_path)], timeout_s=30)
    assert result.returncode == 0, result.stderr
    assert "released without app dependencies" in result.stdout
