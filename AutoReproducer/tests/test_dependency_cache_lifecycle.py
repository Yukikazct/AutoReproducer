"""Cache use/cleanup races without installing packages or running training."""
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import Mock

import pytest

import frontend.history_manager as history
import src.agents.code_executor as ce
import src.dependency_cache as cache
import src.repository_runner as repository


@pytest.fixture
def root(tmp_path, monkeypatch):
    path = tmp_path / "deps"
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(path))
    monkeypatch.setattr(ce, "DEPS_CACHE_ROOT", path)
    monkeypatch.setattr(repository, "DEPS_CACHE_ROOT", path)
    ce._INSTALLED_DEPS.clear()
    yield path
    ce._INSTALLED_DEPS.clear()


def environment(root, name="legacy", *, recent=False):
    path = root / name
    path.mkdir(parents=True)
    (path / ".ready").write_text("ok", encoding="utf-8")
    (path / "package.py").write_text("VALUE = 1", encoding="utf-8")
    (path / "meta.json").write_text(json.dumps({
        "kind": "reqs", "last_used": datetime.now().isoformat() if recent else "2000-01-01T00:00:00",
    }), encoding="utf-8")
    return path


def test_nested_environments_use_leaf_recency_and_cannot_delete_namespaces(root):
    hot = environment(root, "repository/runtime/hot", recent=True)
    cold = environment(root, "repository/runtime/cold")
    legacy = environment(root, recent=True)
    os.utime(root / "repository", (1000, 1000))
    os.utime(hot.parent, (1000, 1000))
    assert {item["name"] for item in history.list_deps_cache()} == {
        "legacy", "repository/runtime/hot", "repository/runtime/cold"}
    assert history.delete_deps_cache(["repository", "repository/runtime", ".", ".."]) == (0, 0)
    assert history.cleanup_deps_cache(30)[0] == 1
    assert hot.is_dir() and legacy.is_dir() and not cold.exists()
    assert history.delete_deps_cache(["repository/runtime/hot"])[0] == 1
    assert legacy.exists() and hot.parent.exists()


def test_cache_symlinks_are_never_listed_or_deleted(root, tmp_path):
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    try:
        (root / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Host does not permit test symlinks: {exc}")
    assert history.list_deps_cache() == []
    assert history.delete_deps_cache(["linked", str(outside), "../outside"]) == (0, 0)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_malformed_metadata_does_not_prevent_exit_refresh_or_lock_release(root):
    path = environment(root)
    executor = ce.CodeExecutorAgent(None, logger=Mock())
    executor._deps_dir = str(path)
    with executor.dependency_scope():
        (path / "meta.json").write_text("[]", encoding="utf-8")
    assert ce.read_deps_meta(path)["last_used"]
    assert history.delete_deps_cache([path.name])[0] == 1


def test_active_process_blocks_cleanup_and_execution_then_crash_releases_lock(root, tmp_path, monkeypatch):
    environment(root)
    program = (
        "import sys\nfrom src.dependency_cache import cache_guard\n"
        "with cache_guard(sys.argv[1], timeout=5):\n"
        " print('locked', flush=True)\n sys.stdin.readline()\n")
    process = subprocess.Popen([sys.executable, "-c", program, str(root)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8")
    try:
        assert process.stdout.readline().strip() == "locked"
        notices = []
        assert history.delete_deps_cache(["legacy"], on_skip=notices.append) == (0, 0)
        assert history.cleanup_deps_cache(0, on_skip=notices.append) == (0, 0)
        assert len(notices) == 2 and all("正在使用" in notice for notice in notices)
        monkeypatch.setattr(cache, "LOCK_TIMEOUT", 0.02)
        executor = ce.CodeExecutorAgent(None, logger=Mock())
        execute = Mock(side_effect=AssertionError("Busy cache must not start code"))
        monkeypatch.setattr(executor, "_run_local_script", execute)
        result = executor.execute_in_workspace("print('blocked')", str(tmp_path / "work"))
        assert result["exit_code"] == -4 and result["executed"] is False
        assert "超时" in result["stderr"]
        execute.assert_not_called()
        runner = repository.RepositoryRunner(executor=executor)
        result = runner.run(tmp_path, [{"id": "run", "argv": ["python", "-c", "print(1)"]}], {})
        assert result["final"]["exit_code"] == -4 and not result["executed"]
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
    assert history.delete_deps_cache(["legacy"])[0] == 1


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "exception"])
def test_local_use_prevents_same_thread_cleanup_and_refreshes_on_exit(root, tmp_path, monkeypatch, outcome):
    reqs = "test-package==1"
    path = environment(root, ce.reqs_digest(reqs))
    executor = ce.CodeExecutorAgent(None, logger=Mock())
    executor.env_config = {"requirements_txt": reqs}
    notices = []

    def run(*args):
        (path / "meta.json").write_text('{"last_used":"2000-01-01T00:00:00"}', encoding="utf-8")
        assert history.delete_deps_cache([path.name], on_skip=notices.append) == (0, 0)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired("test", 1)
        if outcome == "exception":
            raise RuntimeError("expected execution failure")
        return {"success": outcome == "success", "exit_code": 0 if outcome == "success" else 1,
                "stdout": "", "stderr": ""}

    monkeypatch.setattr(executor, "_run_local_script", run)
    result = executor.execute_in_workspace("print('small test')", str(tmp_path / "work"))
    assert result["success"] is (outcome == "success")
    assert notices and "正在使用" in notices[0]
    assert ce.read_deps_meta(path)["last_used"] != "2000-01-01T00:00:00"
    assert history.delete_deps_cache([path.name])[0] == 1


def test_repository_scope_covers_steps_and_refreshes_nested_environment(root, tmp_path):
    runtime = repository.runtime_fingerprint()
    reqs = "test-package==1"
    identifier = f"repository/{runtime}/{ce.reqs_digest(reqs)}"
    path = environment(root, identifier)
    notices = []

    def event(item):
        if item.get("type") == "repository_step" and item.get("status") == "running":
            (path / "meta.json").write_text('{"last_used":"2000-01-01T00:00:00"}', encoding="utf-8")
            assert history.delete_deps_cache([identifier], on_skip=notices.append) == (0, 0)

    runner = repository.RepositoryRunner(executor=ce.CodeExecutorAgent(None, logger=Mock()))
    result = runner.run(tmp_path, [{"id": "run", "argv": ["python", "-c", "print('OK')"]}],
                        {"requirements_txt": reqs}, on_event=event)
    assert result["success"] and notices and not result["artifact_warnings"]
    assert ce.read_deps_meta(path)["last_used"] != "2000-01-01T00:00:00"
    assert history.delete_deps_cache([identifier])[0] == 1


def test_concurrent_installers_wait_for_ready_and_install_once(root, tmp_path, monkeypatch):
    installing, release, waiting = threading.Event(), threading.Event(), threading.Event()
    calls = []
    original_guard = ce.cache_guard

    @contextmanager
    def observed_guard(path, **kwargs):
        if threading.current_thread().name.endswith("_1"):
            waiting.set()
        with original_guard(path, **kwargs):
            yield

    def pip(cmd, **kwargs):
        calls.append(cmd)
        installing.set()
        assert release.wait(5)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ce, "cache_guard", observed_guard)
    monkeypatch.setattr(ce.subprocess, "run", pip)
    first, second = [ce.CodeExecutorAgent(None, logger=Mock()) for _ in range(2)]
    for index, executor in enumerate((first, second)):
        executor.env_config = {"requirements_txt": "test-package==1"}
        (tmp_path / f"prepare{index}").mkdir()
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first._ensure_local_deps, str(tmp_path / "prepare0"))
        try:
            assert installing.wait(5)
            b = pool.submit(second._ensure_local_deps, str(tmp_path / "prepare1"))
            assert waiting.wait(5)
            assert not b.done()
            assert not list(root.glob("*/.ready"))
        finally:
            release.set()
        assert a.result(timeout=5) is None and b.result(timeout=5) is None
    assert len(calls) == 1 and first._deps_dir == second._deps_dir


def test_self_heal_install_is_protected_and_cleanup_rechecks_current_recency(root, monkeypatch):
    executor = ce.CodeExecutorAgent(None, logger=Mock())
    notices = []

    def pip(cmd, **kwargs):
        assert history.cleanup_deps_cache(0, on_skip=notices.append) == (0, 0)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ce.subprocess, "run", pip)
    assert executor._heal_install_local("requests") is None
    assert notices and "正在使用" in notices[0]
    stale = environment(root, "was-cold")
    history.list_deps_cache()
    ce.touch_deps_meta(stale)
    assert history.cleanup_deps_cache(30) == (0, 0)
    assert history.delete_deps_cache(["heal-requests", "was-cold"])[0] == 2
