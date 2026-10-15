"""Author interpreter selection shares the disposable report/cleanup lease."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from src import author_python, environment_lifecycle as lifecycle, training_environment as training
from src.agents import code_executor


@pytest.fixture
def roots(tmp_path, monkeypatch):
    data = tmp_path / "data"
    deps = data / "deps"
    monkeypatch.setenv("AUTOREPRO_DATA_ROOT", str(data))
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(deps))
    monkeypatch.setattr(code_executor, "DEPS_CACHE_ROOT", deps)
    from src import runtime_preparation
    monkeypatch.setattr(runtime_preparation, "release_runtime_downloads", lambda: {
        "status": "released", "removed": [], "errors": [], "freed_bytes": 0})
    return data, deps


def make_runtime(root, version="3.12"):
    directory = root / "envs" / ("cpython-" + version)
    directory.mkdir(parents=True)
    executable = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    executable.parent.mkdir(exist_ok=True)
    executable.write_bytes(b"owned test interpreter")
    return executable


@pytest.mark.parametrize("exception", [None, RuntimeError, KeyboardInterrupt])
def test_real_offline_training_venv_is_released_after_report(roots, exception):
    data, deps = roots
    root = data / "training_python"
    report = data / "report.md"

    def run():
        with lifecycle.managed_environment() as lease:
            selected = author_python.select_author_python({"repository": {"files": {}}}, root, offline=True)
            executable = Path(selected["python_executable"])
            assert executable.is_relative_to(root)
            probe = subprocess.run([str(executable), "-I", "-c", "import sys;print(sys.prefix)"],
                                   capture_output=True, text=True, timeout=15, check=True)
            assert Path(probe.stdout.strip()).is_relative_to(root)
            report.write_text("# Saved before runtime release", encoding="utf-8")
            lease.result = {"state": "COMPLETED", "data": {}}
            if exception is not None:
                raise exception("cancelled")
        assert lease.result["data"]["environment_cleanup"]["training_python"]["removed"]

    if exception is None:
        run()
    else:
        with pytest.raises(exception):
            run()
    assert not root.exists()
    assert report.is_file() and Path(sys.executable).is_file()
    assert not (deps / training.JOURNAL).exists()


def test_python_conflict_removes_old_runtime_and_packages_before_replacement(roots, tmp_path, monkeypatch):
    data, deps = roots
    root = data / "training_python"
    external = tmp_path / "python3.11"
    external.touch()
    monkeypatch.setattr(author_python.shutil, "which", lambda name: str(external) if name == "python3.11" else None)
    monkeypatch.setattr(author_python, "_probe", lambda path: (3, 11) if Path(path) == external else (3, 12))
    monkeypatch.setattr(author_python, "runtime_fingerprint", lambda executable: "test-runtime")
    created = []
    def create(root, base, version):
        assert (deps / training.JOURNAL).is_file()
        assert not (deps / "old-packages").exists()
        assert not (root / "envs").exists()
        created.append(version)
        return make_runtime(root, ".".join(map(str, version)))
    monkeypatch.setattr(author_python, "_managed_venv", create)
    packet = {"repository": {"files": {}}}
    with lifecycle.managed_environment():
        selected = author_python.select_author_python(packet, root, offline=True)
        (deps / "old-packages").mkdir()
        code_executor._INSTALLED_DEPS["old"] = ""
        initial = {"success": False, "effective_env_config": {"requirements_txt": "example==1.2.0\n",
            "dependency_failure_kind": "python_incompatible", "dependency_install_attempts": [{
                "failure_kind": "python_incompatible", "diagnostic":
                "Package 'example' requires a different Python: 3.12.0 not in '<3.12'"}]}}
        environment = {"requirements_txt": "example==1.2.0\n"}
        result = author_python.retry_requested_python_conflict(packet, root, environment, selected,
            initial, lambda effective: {"success": True}, offline=True)
        assert result["success"] and environment["python_version"] == "3.11"
        assert not code_executor._INSTALLED_DEPS
        assert [path.name for path in (root / "envs").iterdir()] == ["cpython-3.11"]
    assert created == [(3, 12), (3, 11)]


def test_custom_root_journal_recovers_after_abandoned_process(roots, tmp_path):
    data, deps = roots
    custom = tmp_path / "custom-runs" / "training_python"
    deps.mkdir(parents=True)
    make_runtime(custom)
    report = custom.parent / "report.md"
    report.write_text("retained", encoding="utf-8")
    (deps / training.JOURNAL).write_text(json.dumps({"version": 1, "roots": [str(custom)]}), encoding="utf-8")
    # A pre-journal default training environment is migrated at the same time.
    make_runtime(data / "training_python", "3.11")
    receipt = lifecycle.release_abandoned_environment(deps)
    assert receipt["status"] == "released"
    assert len(receipt["training_python"]["removed"]) == 2
    assert not custom.exists() and not (data / "training_python").exists()
    assert report.read_text("utf-8") == "retained"


def test_explicit_dependency_lease_owns_a_custom_training_root(roots, tmp_path):
    data, deps = roots
    explicit_deps = tmp_path / "explicit-deps"
    custom = tmp_path / "custom-data" / "training_python"
    with lifecycle.managed_environment(explicit_deps):
        training.prepare_training_environment(custom)
        make_runtime(custom)
        assert (explicit_deps / training.JOURNAL).is_file()
    assert not custom.exists() and not (explicit_deps / training.JOURNAL).exists()


def test_crashed_training_owner_recovers_custom_runtime_without_unwinding(roots, tmp_path):
    data, deps = roots
    custom = tmp_path / "worker-data" / "training_python"
    script = "\n".join([
        "import os,sys",
        "from src.environment_lifecycle import managed_environment",
        "from src.training_environment import prepare_training_environment",
        "with managed_environment():",
        "    root=prepare_training_environment(sys.argv[1])",
        "    (root/'interpreters').mkdir()",
        "    (root/'interpreters'/'partial-download').write_bytes(b'owned')",
        "    os._exit(9)",
    ])
    worker = subprocess.run([sys.executable, "-c", script, str(custom)], capture_output=True,
                            text=True, timeout=20)
    assert worker.returncode == 9, worker.stderr
    assert custom.exists() and (deps / training.JOURNAL).exists()
    with lifecycle.managed_environment():
        assert not custom.exists()
    assert not (deps / training.JOURNAL).exists()


def test_training_release_failure_blocks_replacement_and_persists_error_state(roots, monkeypatch):
    data, deps = roots
    root = data / "training_python"
    original = training.remove_tree
    def remove(path):
        if Path(path) == root:
            raise PermissionError("interpreter still open")
        original(path)
    monkeypatch.setattr(training, "remove_tree", remove)
    run_dir = data / "runs" / "example"
    run_dir.mkdir(parents=True)
    with lifecycle.managed_environment() as lease:
        training.prepare_training_environment(root)
        make_runtime(root)
        lease.result = {"state": "COMPLETED", "error": None, "data": {"run_dir": str(run_dir)}}
        (run_dir / "result.json").write_text(json.dumps(lease.result), encoding="utf-8")
        with pytest.raises(lifecycle.EnvironmentCleanupError):
            training.prepare_training_environment(root)
        assert root.exists()
    saved = json.loads((run_dir / "result.json").read_text("utf-8"))
    assert saved["state"] == lease.result["state"] == "ERROR"
    assert saved["data"]["environment_cleanup"]["training_python"]["status"] == "error"
    assert (deps / training.JOURNAL).exists()
    with pytest.raises(lifecycle.EnvironmentCleanupError):
        with lifecycle.managed_environment():
            pytest.fail("a second environment must not start")


@pytest.mark.parametrize("linked_ancestor", [False, True])
def test_linked_training_root_is_rejected_without_touching_external_files(roots, tmp_path, require_symlinks, linked_ancestor):
    data, deps = roots
    external = tmp_path / "external"
    make_runtime(external / "training_python")
    if linked_ancestor:
        link = data / "alias"
        data.mkdir(parents=True)
        link.symlink_to(external, target_is_directory=True)
        target = link / "training_python"
    else:
        target = data / "training_python"
        data.mkdir(parents=True)
        target.symlink_to(external / "training_python", target_is_directory=True)
    if linked_ancestor:
        with lifecycle.managed_environment():
            with pytest.raises(ValueError, match="symlink"):
                training.prepare_training_environment(target)
    else:
        with pytest.raises(lifecycle.EnvironmentCleanupError):
            with lifecycle.managed_environment():
                pass
    assert (external / "training_python" / "envs").is_dir()


def test_internal_venv_links_are_unlinked_without_deleting_base_python(roots, tmp_path, require_symlinks):
    data, deps = roots
    root = data / "training_python"
    external = tmp_path / "host-python"
    external.write_bytes(b"retain host interpreter")
    with lifecycle.managed_environment():
        training.prepare_training_environment(root)
        directory = root / "envs" / "cpython-3.12" / "bin"
        directory.mkdir(parents=True)
        (directory / "python").symlink_to(external)
    assert not root.exists() and external.read_bytes() == b"retain host interpreter"


def test_training_install_processes_own_scratch_and_disable_shared_caches(roots, monkeypatch):
    data, deps = roots
    root = data / "training_python"
    calls = []
    def execute(command, **kwargs):
        environment = kwargs["env"]
        assert Path(environment["TEMP"]).is_relative_to(root)
        assert Path(environment["UV_CACHE_DIR"]).is_relative_to(root)
        assert environment["PIP_NO_CACHE_DIR"] == "1"
        assert "GITHUB_TOKEN" not in environment
        calls.append(command)
        binary = root / "tools" / "uv" / "Scripts" / "uv.exe"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.touch()
        return Mock(returncode=1 if len(calls) == 2 else 0)
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setattr(author_python.shutil, "which", lambda name: None)
    monkeypatch.setattr(author_python, "run_owned_process", execute)
    with lifecycle.managed_environment():
        training.prepare_training_environment(root)
        author_python._uv_command(root, offline=False)
        author_python._ensure_pip("owned-python", root)
    assert len(calls) == 4 and calls[2][1:3] == ["-m", "ensurepip"] and not root.exists()


def test_shared_training_root_cannot_be_deleted_by_another_dependency_owner(roots, tmp_path):
    data, deps = roots
    root = data / "training_python"
    with lifecycle.managed_environment():
        training.prepare_training_environment(root)
        make_runtime(root)
        script = ("import json,sys;from src.training_environment import release_training_environments;"
                  "print(json.dumps(release_training_environments(sys.argv[1])))")
        competitor = tmp_path / "other-deps"
        competitor.mkdir()
        result = subprocess.run([sys.executable, "-c", script, str(competitor)],
                                capture_output=True, text=True, timeout=15, check=True)
        receipt = json.loads(result.stdout)
        assert receipt["status"] == "error" and "another active run" in receipt["errors"][0]["reason"]
        assert root.exists()


def test_nested_distinct_dependency_owner_preserves_outer_training_runtime(roots, tmp_path):
    data, deps = roots
    root = data / "training_python"
    with lifecycle.managed_environment():
        training.prepare_training_environment(root)
        make_runtime(root)
        with pytest.raises(lifecycle.EnvironmentCleanupError):
            with lifecycle.managed_environment(tmp_path / "nested-deps"):
                pytest.fail("nested owner must not take the live interpreter")
        assert root.exists()
