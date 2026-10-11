"""Fault-inject cached package damage; probes are real and pip is offline."""
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.agents.code_executor as ce
from src.dependency_cache import reset_managed_environment
from src.resource_events import ResourceEventLogger
import src.repository_runner as repository
from src.runtime_platform import runtime_fingerprint


REQUIREMENTS = "native-demo==1.0"
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    ce._INSTALLED_DEPS.clear()
    monkeypatch.setattr(ce, "DEPS_CACHE_ROOT", tmp_path / "deps")
    monkeypatch.setattr(ce, "PIP_INDEX_URL", MIRROR)
    monkeypatch.setattr(ce, "PIP_FIND_LINKS", "")
    monkeypatch.setattr(ce, "ResourceEventLogger", lambda: ResourceEventLogger(tmp_path / "events.jsonl"))
    yield
    ce._INSTALLED_DEPS.clear()


def package(directory, *, version="1.0", source="VALUE = 7\n", ready=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "native_demo.py").write_text(source, encoding="utf-8")
    dist = directory / f"native_demo-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: native-demo\nVersion: {version}\n", encoding="utf-8")
    if ready:
        (directory / ".ready").write_text("ok\n", encoding="utf-8")


def executor():
    instance = ce.CodeExecutorAgent(None, logger=Mock())
    instance.env_config = {"requirements_txt": REQUIREMENTS, "dependency_health_check": True}
    return instance


def prepare(instance, tmp_path, name="prepare"):
    workdir = tmp_path / name
    workdir.mkdir()
    return instance._ensure_local_deps(str(workdir))


def fake_installer(monkeypatch, action):
    real_run = subprocess.run
    real_dependency_run = ce.CodeExecutorAgent._run_dependency_command
    calls = []

    def run(command, **kwargs):
        if command[1:4] == ["-m", "pip", "install"]:
            directory = Path(command[command.index("--target") + 1])
            manifest = Path(command[command.index("-r") + 1]).read_text(encoding="utf-8")
            calls.append({"command": command, "manifest": manifest, "directory": directory})
            return action(directory, len(calls), command)
        assert command[1:3] == ["-I", "-c"], "only real isolated health probes may run"
        return real_run(command, **kwargs)

    monkeypatch.setattr(ce.subprocess, "run", run)
    def dependency_run(instance, command, **kwargs):
        if command[1:4] == ["-m", "pip", "install"]:
            return run(command, **kwargs)
        return real_dependency_run(instance, command, **kwargs)
    monkeypatch.setattr(ce.CodeExecutorAgent, "_run_dependency_command", dependency_run)
    return calls


@pytest.mark.parametrize("damage", ["missing_metadata", "wrong_version", "native_import", "duplicate_metadata"])
def test_ready_cache_is_probed_and_cleanly_rebuilt_under_same_pins(monkeypatch, tmp_path, damage):
    directory = ce.DEPS_CACHE_ROOT / ce.reqs_digest(REQUIREMENTS)
    version = "0.9" if damage == "wrong_version" else "1.0"
    source = "raise OSError('DLL load failed: corrupt native binary')\n" if damage == "native_import" else "VALUE = 7\n"
    package(directory, version=version, source=source, ready=True)
    if damage == "missing_metadata":
        (directory / f"native_demo-{version}.dist-info" / "METADATA").unlink()
    elif damage == "duplicate_metadata":
        extra = directory / "native_demo-0.5.dist-info"
        extra.mkdir()
        (extra / "METADATA").write_text("Name: native-demo\nVersion: 0.5\n", encoding="utf-8")
    (directory / "obsolete_native.pyd").write_text("stale", encoding="utf-8")
    ce._INSTALLED_DEPS[REQUIREMENTS] = ""

    def install(target, attempt, command):
        assert not list(target.iterdir()), "rebuild must remove old binaries and dist-info"
        package(target)
        return subprocess.CompletedProcess(command, 0, "installed", "")

    calls = fake_installer(monkeypatch, install)
    instance = executor()
    assert prepare(instance, tmp_path) is None
    assert len(calls) == 1 and calls[0]["manifest"] == REQUIREMENTS
    assert instance._deps_dir == str(directory) and (directory / ".ready").is_file()
    assert not (directory / "obsolete_native.pyd").exists()
    assert [item["success"] for item in instance.env_config["dependency_health_attempts"]] == [False, True]
    assert len(instance.env_config["dependency_cache_repairs"]) == 1
    assert instance.env_config["dependency_cache_repairs"][0]["status"] == "succeeded"


def test_healthy_cache_uses_real_probe_without_pip_or_manifest_rewrite(monkeypatch, tmp_path):
    directory = ce.DEPS_CACHE_ROOT / ce.reqs_digest(REQUIREMENTS)
    package(directory, ready=True)
    calls = fake_installer(monkeypatch, lambda *args: pytest.fail("healthy cache must not invoke pip"))
    first, second = executor(), executor()
    assert prepare(first, tmp_path, "first") is None
    assert prepare(second, tmp_path, "second") is None
    assert calls == [] and first._deps_dir == second._deps_dir
    assert not (tmp_path / "second" / "requirements.txt").exists()
    assert second.env_config["dependency_health_attempts"][0]["success"]


@pytest.mark.parametrize("state", ["missing", "corrupt", "healthy"])
def test_offline_presets_only_use_healthy_cached_dependencies(monkeypatch, tmp_path, state):
    directory = ce.DEPS_CACHE_ROOT / ce.reqs_digest(REQUIREMENTS)
    if state != "missing":
        package(directory, source="raise OSError('broken DLL')\n" if state == "corrupt" else "VALUE = 7\n", ready=True)
    calls = fake_installer(monkeypatch, lambda *args: pytest.fail("offline preparation must never invoke pip"))
    instance = executor()
    instance.env_config.update(offline=True, auto_prepare=True)
    diagnostic = prepare(instance, tmp_path)
    assert calls == []
    if state == "healthy":
        assert diagnostic is None and instance._deps_dir == str(directory)
    else:
        assert "离线模式" in diagnostic and instance._deps_dir is None
        if state == "corrupt":
            assert "broken DLL" in diagnostic
            assert (directory / "native_demo.py").is_file()


def test_new_successful_pip_with_corrupt_native_package_gets_one_clean_rebuild(monkeypatch, tmp_path):
    def install(directory, attempt, command):
        assert not list(directory.iterdir())
        package(directory, source="raise OSError('invalid Win32 native image')\n" if attempt == 1 else "VALUE = 7\n")
        return subprocess.CompletedProcess(command, 0, "installed", "")

    calls = fake_installer(monkeypatch, install)
    instance = executor()
    assert prepare(instance, tmp_path) is None
    assert len(calls) == 2
    assert all(call["manifest"] == REQUIREMENTS for call in calls)
    assert [item["success"] for item in instance.env_config["dependency_install_attempts"]] == [False, True]
    assert [item["phase"] for item in instance.env_config["dependency_health_attempts"]] == ["installed", "rebuild"]


def test_repeated_native_failure_stops_without_publishing_ready_or_training(monkeypatch, tmp_path):
    def install(directory, attempt, command):
        package(directory, source="raise ImportError('native ABI incompatibility')\n")
        return subprocess.CompletedProcess(command, 0, "installed", "")

    calls = fake_installer(monkeypatch, install)
    instance = executor()
    diagnostic = prepare(instance, tmp_path)
    assert len(calls) == 2 and "native ABI incompatibility" in diagnostic
    assert "依赖健康检查失败" in diagnostic
    assert instance._deps_dir is None
    assert not (ce.DEPS_CACHE_ROOT / ce.reqs_digest(REQUIREMENTS) / ".ready").exists()
    assert instance.env_config["dependency_cache_repairs"][0]["status"] == "failed"


def test_public_source_fallback_cleans_partial_target_before_native_repair(monkeypatch, tmp_path):
    def install(directory, attempt, command):
        assert not list(directory.iterdir())
        if attempt == 1:
            (directory / "partial.dll").write_text("partial", encoding="utf-8")
            return subprocess.CompletedProcess(command, 1, "", "ConnectionError: mirror refused")
        package(directory, source="raise OSError('DLL load failed')\n" if attempt == 2 else "VALUE = 7\n")
        return subprocess.CompletedProcess(command, 0, "installed", "")

    calls = fake_installer(monkeypatch, install)
    instance = executor()
    assert prepare(instance, tmp_path) is None
    assert [call["command"][call["command"].index("-i") + 1] for call in calls] == [MIRROR, ce._OFFICIAL_PIP_INDEX, ce._OFFICIAL_PIP_INDEX]
    assert all(call["manifest"] == REQUIREMENTS for call in calls)


def test_global_package_cannot_mask_missing_cached_module(monkeypatch, tmp_path):
    directory = ce.DEPS_CACHE_ROOT / ce.reqs_digest("packaging")
    directory.mkdir(parents=True)
    metadata = directory / "packaging-99.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: packaging\nVersion: 99\n", encoding="utf-8")
    instance = executor()
    assert "outside the managed cache" in instance._check_local_dependency_health(directory, "packaging", "cached")


def test_health_probe_timeout_is_bounded_and_not_ready(monkeypatch, tmp_path):
    def install(directory, attempt, command):
        package(directory)
        return subprocess.CompletedProcess(command, 0, "installed", "")

    calls = fake_installer(monkeypatch, install)
    fake_run = ce.subprocess.run

    def run(command, **kwargs):
        if command[1:3] == ["-I", "-c"]:
            assert kwargs["timeout"] == ce.LOCAL_DEPENDENCY_HEALTH_TIMEOUT
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return fake_run(command, **kwargs)

    monkeypatch.setattr(ce.subprocess, "run", run)
    instance = executor()
    diagnostic = prepare(instance, tmp_path)
    assert "健康检查超时" in diagnostic and len(calls) == 2
    assert all(item["timed_out"] for item in instance.env_config["dependency_health_attempts"])
    assert instance._deps_dir is None


def test_supplemental_repair_is_isolated_and_preserves_primary_dependency_directory(monkeypatch, tmp_path):
    instance = executor()
    instance.deps_cache_root = ce.DEPS_CACHE_ROOT / "repository" / "abi-platform"
    instance._deps_dir = "primary"

    def install(directory, attempt, command):
        package(directory)
        return subprocess.CompletedProcess(command, 0, "installed", "")

    calls = fake_installer(monkeypatch, install)
    assert instance._heal_install_local("native_demo") is None
    repaired = calls[0]["directory"]
    assert repaired.is_relative_to(instance.deps_cache_root / "supplemental" / ce.reqs_digest(REQUIREMENTS))
    assert instance._deps_dir == "primary" and str(repaired) in instance._heal_dirs
    assert json.loads((repaired / "meta.json").read_text(encoding="utf-8"))["kind"] == "heal"
    assert instance._heal_install_local("native_demo") is None and len(calls) == 1


def test_cache_reset_rejects_root_and_outside_directory(tmp_path):
    managed = tmp_path / "managed"
    managed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="cache root"):
        reset_managed_environment(managed, managed)
    with pytest.raises(ValueError, match="escapes"):
        reset_managed_environment(managed, outside)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_cache_reset_rejects_nested_symlink_without_touching_external_files(tmp_path):
    managed, outside = tmp_path / "managed", tmp_path / "outside"
    target = managed / "cache"
    target.mkdir(parents=True)
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    try:
        (target / "link").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Host does not permit test symlinks: {exc}")
    with pytest.raises(ValueError, match="symlink or junction"):
        reset_managed_environment(managed, target)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_health_enabled_runtime_namespace_rejects_linked_ancestor(monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    ce.DEPS_CACHE_ROOT.mkdir()
    linked = ce.DEPS_CACHE_ROOT / "repository"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Host does not permit test symlinks: {exc}")
    instance = executor()
    instance.deps_cache_root = linked / "abi-platform"
    instance.deps_lock_root = ce.DEPS_CACHE_ROOT
    directory = instance.deps_cache_root / ce.reqs_digest(REQUIREMENTS)
    package(directory, ready=True)
    sentinel = directory / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    fake_installer(monkeypatch, lambda *args: pytest.fail("linked cache must not run pip"))
    diagnostic = prepare(instance, tmp_path)
    assert "路径检查失败" in diagnostic and "symlink or junction" in diagnostic
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("cached_damage", [False, True])
def test_repository_auto_prepare_installs_or_repairs_before_real_import_step(monkeypatch, tmp_path, cached_damage):
    monkeypatch.setattr(repository, "DEPS_CACHE_ROOT", ce.DEPS_CACHE_ROOT)
    directory = ce.DEPS_CACHE_ROOT / "repository" / runtime_fingerprint() / ce.reqs_digest(REQUIREMENTS)
    if cached_damage:
        package(directory, source="raise OSError('broken cached DLL')\n", ready=True)

    def install(target, attempt, command):
        assert not list(target.iterdir())
        package(target)
        return subprocess.CompletedProcess(command, 0, "installed", "")

    calls = fake_installer(monkeypatch, install)
    events = []
    runner = repository.RepositoryRunner(executor=executor())
    result = runner.run(tmp_path, [{"id": "import_check", "argv": ["python", "-c", "import native_demo; print(native_demo.VALUE)"]}],
                        {"requirements_txt": REQUIREMENTS, "require_prepared": True, "auto_prepare": True,
                         "dependency_health_check": True}, on_event=events.append)
    assert result["success"] and result["final"]["stdout"] == "7\n"
    assert len(calls) == 1 and calls[0]["manifest"] == REQUIREMENTS
    assert result["environment"]["dependencies_path"] == str(directory)
    saved = json.loads((Path(result["run_dir"]) / "environment.json").read_text(encoding="utf-8"))
    assert saved["dependency_health_attempts"] == result["environment"]["dependency_health_attempts"]
    if cached_damage:
        assert saved["dependency_cache_repairs"][0]["status"] == "succeeded"
    assert [event["status"] for event in events if event["type"] == "repository_environment"] == ["running", "success"]


def test_repository_failed_health_rebuild_retains_diagnostics_and_never_runs_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(repository, "DEPS_CACHE_ROOT", ce.DEPS_CACHE_ROOT)

    def install(target, attempt, command):
        package(target, source="raise OSError('unrepairable native DLL')\n")
        return subprocess.CompletedProcess(command, 0, "installed", "")

    fake_installer(monkeypatch, install)
    runner = repository.RepositoryRunner(executor=executor())
    run_plan = Mock(side_effect=AssertionError("failed preparation must not launch training"))
    monkeypatch.setattr(runner, "_execute", run_plan)
    result = runner.run(tmp_path, [{"id": "train", "argv": ["python", "-c", "print('NEVER')"]}],
                        {"requirements_txt": REQUIREMENTS, "auto_prepare": True, "dependency_health_check": True})
    assert not result["success"] and not result["executed"] and result["final"]["exit_code"] == -4
    run_plan.assert_not_called()
    saved = json.loads((Path(result["run_dir"]) / "environment.json").read_text(encoding="utf-8"))
    assert saved["status"] == "error" and saved["dependencies_path"] is None
    assert len(saved["dependency_install_attempts"]) == 2
    assert saved["dependency_cache_repairs"][0]["status"] == "failed"
    assert "unrepairable native DLL" in saved["dependency_health_attempts"][-1]["diagnostic"]
    execution = json.loads((Path(result["run_dir"]) / "execution.json").read_text(encoding="utf-8"))
    assert execution["environment"] == saved
