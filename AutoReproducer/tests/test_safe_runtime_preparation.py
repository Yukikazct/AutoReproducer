"""Automatic host recovery, cache repair and real child lifecycle regressions."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.runtime_preparation as runtime


def test_runtime_requirement_preserves_store_guard_and_triggers_recovery(monkeypatch):
    monkeypatch.setattr(runtime.sys, "version_info", (3, 12, 10))
    guard = Mock(side_effect=ValueError("Store-backed venv"))
    monkeypatch.setattr(runtime, "validate_windows_venv", guard)
    assert "Windows Store" in runtime.runtime_requirement()
    guard.assert_called_once()


def test_runtime_requirement_detects_missing_app_dependencies(monkeypatch):
    monkeypatch.setattr(runtime.sys, "version_info", (3, 12, 10))
    monkeypatch.setattr(runtime, "validate_windows_venv", lambda: None)
    monkeypatch.setattr(runtime.importlib.metadata, "version", Mock(side_effect=runtime.importlib.metadata.PackageNotFoundError("streamlit")))
    assert "依赖缺失" in runtime.runtime_requirement()


@pytest.mark.parametrize("exception", [ImportError("corrupt package"), ValueError("numpy.dtype size changed"),
                                      RuntimeError("native library unavailable"), SyntaxError("corrupt source")])
def test_runtime_requirement_detects_broken_installed_dependency(monkeypatch, exception):
    monkeypatch.setattr(runtime.sys, "version_info", (3, 12, 10))
    monkeypatch.setattr(runtime, "validate_windows_venv", lambda: None)
    monkeypatch.setattr(runtime.importlib, "import_module", Mock(side_effect=exception))
    assert "无法加载" in runtime.runtime_requirement()


@pytest.mark.parametrize("exception", [ValueError("invalid version metadata"), TypeError("invalid distribution record")])
def test_runtime_requirement_detects_corrupt_metadata(monkeypatch, exception):
    monkeypatch.setattr(runtime.sys, "version_info", (3, 12, 10))
    monkeypatch.setattr(runtime, "validate_windows_venv", lambda: None)
    monkeypatch.setattr(runtime.importlib.metadata, "version", Mock(side_effect=exception))
    assert "依赖缺失" in runtime.runtime_requirement()


def test_dependency_health_exception_handling_never_swallows_cancellation(monkeypatch):
    monkeypatch.setattr(runtime.sys, "version_info", (3, 12, 10))
    monkeypatch.setattr(runtime, "validate_windows_venv", lambda: None)
    monkeypatch.setattr(runtime.importlib, "import_module", Mock(side_effect=KeyboardInterrupt("cancel")))
    with pytest.raises(KeyboardInterrupt, match="cancel"):
        runtime.runtime_requirement()


def interpreter(path, *, venv=False):
    return {"executable": str(path), "base_executable": str(path), "version": [3, 12, 10],
            "implementation": "cpython", "bits": 64, "platform": "win-amd64",
            "prefix": "app" if venv else "base", "base_prefix": "base"}


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "runtime_requirement", lambda: "unsafe host")
    monkeypatch.setattr(runtime, "_machine_key", lambda root: "test-machine")
    base = tmp_path / "standard-python.exe"
    base.touch()
    state = {"base": base, "created": False, "healthy": False,
             "installs": 0, "downloads": 0, "venv_calls": []}
    monkeypatch.setattr(runtime, "_candidate_paths", lambda *a, **kw: iter([base]))
    def probe(path, **kw):
        runtime._check(kw.get("cancel_check"), kw["deadline"])
        if Path(path) == base:
            return interpreter(base)
        if state["created"] and "app-venv" in str(path):
            return interpreter(path, venv=True)
        return None
    def create(argv, **kw):
        assert argv[:4] == [str(base), "-I", "-m", "venv"]
        state["venv_calls"].append(argv)
        state["created"] = True
        Path(argv[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0, "", "")
    def install(*a, **kw):
        state["installs"] += 1
        state["healthy"] = True
    def download(*a, **kw):
        state["downloads"] += 1
        return interpreter(base)
    monkeypatch.setattr(runtime, "_probe", probe)
    monkeypatch.setattr(runtime, "_healthy", lambda *a, **kw: state["healthy"])
    monkeypatch.setattr(runtime, "run_owned_process", create)
    monkeypatch.setattr(runtime, "_install_app_dependencies", install)
    monkeypatch.setattr(runtime, "_ensure_pip", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "_install_python", download)
    return tmp_path, state


def test_missing_host_environment_is_created_and_reused(fake_runtime):
    project, state = fake_runtime
    events = []
    first = runtime.prepare_runtime(project, progress_callback=events.append)
    assert not first.reused and not first.repaired
    assert "app-venv" in first.executable
    assert state["installs"] == 1 and state["downloads"] == 0
    assert [event["stage"] for event in events] == ["discover", "create_environment", "ready"]
    assert events[-1]["status"] == "completed"
    manifest = project / "data/runtime/test-machine/ready.json"
    assert json.loads(manifest.read_text())["requirements"] == list(runtime.APP_REQUIREMENTS)
    second = runtime.prepare_runtime(project)
    assert second.reused and not second.repaired and second.executable == first.executable
    assert state["installs"] == 1 and len(state["venv_calls"]) == 1


def test_damaged_cached_dependencies_are_repaired_without_recreating_venv(fake_runtime):
    project, state = fake_runtime
    runtime.prepare_runtime(project)
    state["healthy"] = False
    result = runtime.prepare_runtime(project)
    assert result.repaired and not result.reused
    assert state["installs"] == 2 and len(state["venv_calls"]) == 1


def test_ready_file_is_never_trusted_without_health_check(fake_runtime, monkeypatch):
    project, state = fake_runtime
    runtime.prepare_runtime(project)
    state["healthy"] = False
    monkeypatch.setattr(runtime, "_install_app_dependencies", lambda *a, **kw: None)
    with pytest.raises(runtime.RuntimePreparationError, match="健康检查"):
        runtime.prepare_runtime(project)
    assert not (project / "data/runtime/test-machine/ready.json").exists()


def test_missing_all_interpreters_installs_official_runtime_then_isolates_apps(fake_runtime, monkeypatch):
    project, state = fake_runtime
    monkeypatch.setattr(runtime, "_candidate_paths", lambda *a, **kw: iter([]))
    result = runtime.prepare_runtime(project)
    assert state["downloads"] == 1 and state["installs"] == 1
    assert "app-venv" in result.executable


@pytest.mark.parametrize("options", [{"offline": True}, {"allow_download": False}])
def test_offline_missing_interpreter_never_downloads(fake_runtime, monkeypatch, options):
    project, state = fake_runtime
    monkeypatch.setattr(runtime, "_candidate_paths", lambda *a, **kw: iter([]))
    with pytest.raises(runtime.RuntimePreparationError, match="离线"):
        runtime.prepare_runtime(project, **options)
    assert state["downloads"] == state["installs"] == 0


def test_cancel_before_preparation_does_not_install(fake_runtime):
    project, state = fake_runtime
    with pytest.raises(KeyboardInterrupt, match="取消"):
        runtime.prepare_runtime(project, cancel_check=lambda: True)
    assert state["installs"] == state["downloads"] == 0


def test_runtime_cache_lock_serializes_threads_and_wait_can_cancel(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    def owner():
        with runtime._runtime_lock(tmp_path, cancel_check=None, deadline=time.monotonic()+5):
            entered.set()
            release.wait(2)
    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(KeyboardInterrupt):
            with runtime._runtime_lock(tmp_path, cancel_check=lambda: True, deadline=time.monotonic()+2):
                pytest.fail("concurrent preparation entered an owned cache")
    finally:
        release.set()
        thread.join(timeout=3)
    with runtime._runtime_lock(tmp_path, cancel_check=None, deadline=time.monotonic()+2):
        pass


def test_runtime_cache_lock_excludes_a_live_other_process(tmp_path):
    program = ("import sys,time;from src.runtime_preparation import _runtime_lock\n"
               "with _runtime_lock(sys.argv[1],cancel_check=None,deadline=time.monotonic()+10):\n"
               " print('locked',flush=True)\n sys.stdin.readline()\n")
    child = subprocess.Popen([sys.executable, "-c", program, str(tmp_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(runtime.RuntimePreparationTimeout):
            with runtime._runtime_lock(tmp_path, cancel_check=None, deadline=time.monotonic()+.2):
                pytest.fail("cache entered while another live process owns it")
        assert child.poll() is None
    finally:
        child.communicate("release\n", timeout=5)
    with runtime._runtime_lock(tmp_path, cancel_check=None, deadline=time.monotonic()+2):
        pass


def test_compatible_project_venv_is_reused_without_network(fake_runtime, monkeypatch):
    project, state = fake_runtime
    monkeypatch.setattr(runtime, "_probe", lambda path, **kw: interpreter(path, venv=True) if Path(path) == state["base"] else None)
    state["healthy"] = True
    result = runtime.prepare_runtime(project)
    assert result.executable == str(state["base"]) and result.reused
    assert state["downloads"] == state["installs"] == 0


def test_store_backed_candidate_is_rejected_before_spawn(tmp_path, monkeypatch):
    candidate = tmp_path / "python.exe"
    candidate.touch()
    monkeypatch.setattr(runtime, "_unsafe_windows_executable", lambda path: True)
    spawn = Mock(side_effect=AssertionError("Store alias must never be probed"))
    monkeypatch.setattr(runtime, "run_owned_process", spawn)
    assert runtime._probe(candidate, deadline=time.monotonic()+5) is None
    spawn.assert_not_called()


@pytest.mark.parametrize("exception", [OSError("unreadable pyvenv.cfg"), ValueError("malformed pyvenv.cfg")])
def test_unreadable_candidate_configuration_is_skipped_before_spawn(tmp_path, monkeypatch, exception):
    candidate = tmp_path / "python.exe"
    candidate.touch()
    monkeypatch.setattr(runtime, "_unsafe_windows_executable", Mock(side_effect=exception))
    spawn = Mock(side_effect=AssertionError("invalid candidate must never be started"))
    monkeypatch.setattr(runtime, "run_owned_process", spawn)
    assert runtime._probe(candidate, deadline=time.monotonic()+5) is None
    spawn.assert_not_called()


@pytest.mark.parametrize("version,bits,implementation", [([3, 13, 0], 64, "cpython"), ([3, 12, 1], 32, "cpython"), ([3, 12, 1], 64, "pypy")])
def test_probe_rejects_incompatible_interpreters(tmp_path, monkeypatch, version, bits, implementation):
    path = tmp_path / "python.exe"
    path.touch()
    info = interpreter(path)
    info.update(version=version, bits=bits, implementation=implementation)
    monkeypatch.setattr(runtime, "run_owned_process", lambda *a, **kw: subprocess.CompletedProcess(a, 0, json.dumps(info), ""))
    assert runtime._probe(path, deadline=time.monotonic()+5) is None


def test_mirror_unavailable_gets_one_official_fallback_and_redacts_logs(tmp_path, monkeypatch):
    calls, events = [], []
    for key in ("PIP_INDEX_URL", "AUTOREPRO_PIP_INDEX"):
        monkeypatch.delenv(key, raising=False)
    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1 if len(calls) == 1 else 0, "",
            "HTTP Error 403 https://person:password@host/path?token=abc api_key=sk-123456789012345")
    monkeypatch.setattr(runtime, "run_owned_process", run)
    runtime._install_app_dependencies("safe-python", tmp_path, cancel_check=None,
                                     deadline=time.monotonic()+10, progress_callback=events.append, offline=False)
    assert len(calls) == 3
    assert calls[1][calls[1].index("--index-url")+1] == "https://pypi.org/simple"
    assert "install" in calls[2] and "--no-index" in calls[2] and "--force-reinstall" in calls[2]
    text = json.dumps(events)
    assert "password@" not in text and "sk-123" not in text and "token=abc" not in text


def test_explicit_private_index_does_not_fall_back_and_errors_are_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOREPRO_PIP_INDEX", "https://user:secret@internal.example/simple")
    calls = []
    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, "", "failed https://user:secret@internal.example/simple?token=abc")
    monkeypatch.setattr(runtime, "run_owned_process", run)
    with pytest.raises(runtime.RuntimePreparationError) as error:
        runtime._install_app_dependencies("safe-python", tmp_path, cancel_check=None,
                                         deadline=time.monotonic()+10, progress_callback=None, offline=False)
    assert len(calls) == 1 and "secret" not in str(error.value)


def test_offline_app_repair_never_uses_index(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "run_owned_process", lambda argv, **kw: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""))
    runtime._install_app_dependencies("safe-python", tmp_path, cancel_check=None,
                                     deadline=time.monotonic()+10, progress_callback=None, offline=True)
    assert len(calls) == 2
    assert all("--no-index" in call and "--index-url" not in call for call in calls)


def test_missing_pip_is_repaired_from_bundled_ensurepip(monkeypatch):
    calls = []
    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1 if len(calls) == 1 else 0, "", "No module named pip")
    monkeypatch.setattr(runtime, "run_owned_process", run)
    runtime._ensure_pip("safe-python", cancel_check=None, deadline=time.monotonic()+5)
    assert calls == [["safe-python", "-I", "-m", "pip", "--version"],
                     ["safe-python", "-I", "-m", "ensurepip", "--upgrade"],
                     ["safe-python", "-I", "-m", "pip", "--version"]]


def test_corrupt_pip_with_intact_metadata_is_actually_repaired_offline(tmp_path):
    venv = tmp_path / "isolated-runtime"
    runtime.run_owned_process([sys.executable, "-I", "-m", "venv", str(venv)], timeout_s=45)
    executable = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    location = runtime.run_owned_process([str(executable), "-I", "-c", "import pip;print(pip.__file__)"], timeout_s=5)
    assert location.returncode == 0
    Path(location.stdout.strip()).write_text("raise ImportError('broken pip cache')\n", encoding="utf-8")
    broken = runtime.run_owned_process([str(executable), "-I", "-m", "pip", "--version"], timeout_s=5)
    assert broken.returncode != 0
    runtime._ensure_pip(executable, cancel_check=None, deadline=time.monotonic()+45)
    repaired = runtime.run_owned_process([str(executable), "-I", "-m", "pip", "--version"], timeout_s=5)
    assert repaired.returncode == 0 and "pip " in repaired.stdout


def test_offline_dependency_repair_disables_remote_pip_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PIP_FIND_LINKS", "https://must-not-contact.example/wheels")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://must-not-contact.example/simple")
    calls = []
    def run(argv, **kw):
        calls.append(kw["env"])
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(runtime, "run_owned_process", run)
    runtime._install_app_dependencies("safe-python", tmp_path, cancel_check=None,
                                     deadline=time.monotonic()+5, progress_callback=None, offline=True)
    assert calls and all("PIP_FIND_LINKS" not in env and "PIP_EXTRA_INDEX_URL" not in env
                         and env["PIP_CONFIG_FILE"] == os.devnull for env in calls)


def test_cancelled_installer_download_never_leaves_a_complete_cache_file(tmp_path, monkeypatch):
    destination = tmp_path / "python.exe"
    cancelled = {"value": False}
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def geturl(self):
            return "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
        def read(self, size):
            cancelled["value"] = True
            return b"x" * 65536
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **kw: Response())
    with pytest.raises(KeyboardInterrupt):
        runtime._download_installer(destination, cancel_check=lambda: cancelled["value"], deadline=time.monotonic()+5)
    assert not destination.exists() and not destination.with_name("python.exe.part").exists()


def test_installer_redirect_to_nonofficial_site_is_rejected(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def geturl(self):
            return "https://another.example/python.exe"
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **kw: Response())
    with pytest.raises(runtime.RuntimePreparationError, match="非官方"):
        runtime._download_installer(tmp_path / "python.exe", cancel_check=None, deadline=time.monotonic()+5)
    assert not list(tmp_path.glob("*.exe*"))


@pytest.mark.skipif(os.name != "nt", reason="Windows installer signature contract")
def test_official_install_refuses_unsigned_download_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.platform, "machine", lambda: "AMD64")
    downloaded = []
    monkeypatch.setattr(runtime, "_download_installer", lambda path, **kw: path.touch() or downloaded.append(path))
    monkeypatch.setattr(runtime, "_verify_installer", Mock(side_effect=runtime.RuntimePreparationError("signature failed")))
    execute = Mock(side_effect=AssertionError("unsigned installer started"))
    monkeypatch.setattr(runtime, "run_owned_process", execute)
    with pytest.raises(runtime.RuntimePreparationError, match="signature"):
        runtime._install_python(tmp_path, cancel_check=None, deadline=time.monotonic()+5, progress_callback=None)
    execute.assert_not_called()
    assert downloaded and not downloaded[0].exists()


def test_stdin_payload_roundtrips_without_exposing_it_in_argv():
    payload = "private-test-value-" * 20000
    result = runtime.run_owned_process([sys.executable, "-I", "-c", "import sys;print(len(sys.stdin.buffer.read()))"],
                                       input=payload, timeout_s=10)
    assert result.returncode == 0 and result.stdout.strip() == str(len(payload))
    assert payload not in str(result.args)


@pytest.mark.parametrize("mode", ["timeout", "cancel"])
def test_owned_process_cleans_descendants_on_timeout_or_cancel(tmp_path, mode):
    sentinel = tmp_path / "orphan.txt"
    child = "import time;from pathlib import Path;time.sleep(1.2);Path(" + repr(str(sentinel)) + ").write_text('orphan')"
    launcher = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c'," + repr(child) + "]);time.sleep(10)"
    started = time.monotonic()
    options = {"timeout_s": .4} if mode == "timeout" else {"timeout_s": 10, "cancel_check": lambda: time.monotonic()-started > .4}
    expected = runtime.RuntimePreparationTimeout if mode == "timeout" else KeyboardInterrupt
    with pytest.raises(expected):
        runtime.run_owned_process([sys.executable, "-I", "-c", launcher], **options)
    time.sleep(1.4)
    assert not sentinel.exists(), "a cancelled runtime child survived its owner's process job"


def test_child_descendants_are_cleaned_even_after_parent_exits(tmp_path):
    sentinel = tmp_path / "survived.txt"
    child = "import time;from pathlib import Path;time.sleep(.7);Path(" + repr(str(sentinel)) + ").write_text('orphan')"
    launcher = "import subprocess,sys;subprocess.Popen([sys.executable,'-c'," + repr(child) + "]);print('done')"
    result = runtime.run_owned_process([sys.executable, "-I", "-c", launcher], timeout_s=5)
    assert result.returncode == 0 and "done" in result.stdout
    time.sleep(.9)
    assert not sentinel.exists()


def test_owned_process_does_not_confirm_cleanup_after_native_job_close_fails(monkeypatch):
    from types import SimpleNamespace
    import src.process_lifecycle as lifecycle

    failure = OSError(6, "job handle close failed")
    monkeypatch.setattr(lifecycle.ctypes, "get_last_error", lambda: 6, raising=False)
    monkeypatch.setattr(lifecycle.ctypes, "WinError", lambda code: failure, raising=False)
    job = lifecycle.ProcessJob.__new__(lifecycle.ProcessJob)
    job.handle = 123
    job.api = SimpleNamespace(CloseHandle=Mock(return_value=0))
    job.assign, job.resume = Mock(), Mock()
    monkeypatch.setattr(runtime, "ProcessJob", lambda: job)
    process = SimpleNamespace(returncode=0, poll=Mock(return_value=0),
                              wait=Mock(), stdin=None)
    monkeypatch.setattr(runtime.subprocess, "Popen", Mock(return_value=process))
    cleaned = Mock()

    with pytest.raises(OSError) as raised:
        runtime.run_owned_process(["owned-test-process"], on_cleanup=cleaned)
    assert raised.value is failure
    assert job.handle == 123
    job.api.CloseHandle.assert_called_once_with(123)
    cleaned.assert_not_called()


@pytest.mark.parametrize('mode', ['success', 'timeout', 'cancel'])
def test_owned_cleanup_notification_occurs_after_process_is_reaped(mode):
    processes = []
    notifications = []
    started = time.monotonic()
    def observe(process):
        if not processes:
            processes.append(process)
    def cleaned():
        assert processes and processes[0].poll() is not None
        notifications.append(True)
    options = {'timeout_s': .25} if mode == 'timeout' else {'timeout_s': 5}
    if mode == 'cancel':
        options['cancel_check'] = lambda: time.monotonic() - started > .25
    delay = '.15' if mode == 'success' else '10'
    def run():
        return runtime.run_owned_process([sys.executable, '-I', '-c', 'import time;time.sleep(' + delay + ')'],
                                         on_poll=observe, on_cleanup=cleaned, **options)
    if mode == 'success':
        assert run().returncode == 0
    else:
        with pytest.raises(runtime.RuntimePreparationTimeout if mode == 'timeout' else KeyboardInterrupt):
            run()
    assert notifications == [True]
