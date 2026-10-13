"""Configured storage roots stay consistent across writers, readers and cleanup."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from frontend import history_manager as history
from src.audit.audit_logger import AuditLogger
from src.project_paths import PROJECT_ROOT, get_data_root, get_deps_root
from src.resource_events import ResourceEventLogger


def test_default_paths_do_not_drift_with_launch_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOREPRO_DATA_ROOT", raising=False)
    monkeypatch.delenv("AUTOREPRO_DEPS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_data_root() == PROJECT_ROOT / "data"
    assert get_deps_root() == PROJECT_ROOT / "data" / "deps"
    assert history.get_project_data_dir() == get_data_root()


def test_worker_startup_uses_configured_unicode_data_root(tmp_path):
    directory = tmp_path / "数据 工作目录"
    directory.mkdir()
    root = directory / "运行 数据"
    environment = dict(os.environ, AUTOREPRO_DATA_ROOT=str(root), PYTHONIOENCODING="utf-8")
    environment.pop("AUTOREPRO_DEPS_ROOT", None)
    script = """import json,sys
sys.path.insert(0, sys.argv[1])
from src.agents.code_executor import DEPS_CACHE_ROOT
from src.execution_artifacts import ARTIFACT_ROOT, default_report_path
from frontend.history_manager import get_project_data_dir
print(json.dumps([str(p) for p in (DEPS_CACHE_ROOT, ARTIFACT_ROOT,
                                 default_report_path(), get_project_data_dir())]))
"""
    result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", script, str(PROJECT_ROOT)],
                            cwd=directory, env=environment, capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(root / "deps"), str(root / "reports" / "artifacts"),
                                       str(root / "reports" / "report.md"), str(root)]


def test_configured_root_is_shared_by_audit_history_resources_and_events(tmp_path, monkeypatch):
    from src.resource_manager import ResourceManager

    root = tmp_path / "自定义数据"
    monkeypatch.setenv("AUTOREPRO_DATA_ROOT", str(root))
    monkeypatch.delenv("AUTOREPRO_DEPS_ROOT", raising=False)
    assert get_data_root() == root
    assert get_deps_root() == root / "deps"
    assert ResourceManager().data_root == root
    logger = AuditLogger()
    logger.log_experiment("FINISH", "finished", result={"state": "COMPLETED"})
    assert history.list_sessions()[0]["session_id"] == logger.session_id
    events = ResourceEventLogger()
    event = events.emit("code", "paper", "fetch", "completed")
    assert history.list_resource_events() == [event]
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(tmp_path / "packages"))
    assert get_deps_root() == tmp_path / "packages"
    assert get_deps_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_explicit_inventory_root_does_not_change_process_environment(tmp_path, monkeypatch):
    configured, selected = tmp_path / "configured", tmp_path / "selected"
    monkeypatch.setenv("AUTOREPRO_DATA_ROOT", str(configured))
    monkeypatch.delenv("AUTOREPRO_DEPS_ROOT", raising=False)
    environment = selected / "deps" / "sample"
    environment.mkdir(parents=True)
    (environment / ".ready").write_text("ready", encoding="utf-8")
    (environment / "meta.json").write_text("{}", encoding="utf-8")
    event = ResourceEventLogger(str(selected / "resource_events.jsonl")).emit(
        "code", "paper", "fetch", "completed")
    assert history.list_resource_inventory(data_root=selected)[0]["id"] == "sample"
    assert history.list_resource_events(data_root=selected) == [event]
    assert get_data_root() == configured


def test_event_tail_streams_and_tolerates_incomplete_append(tmp_path, monkeypatch):
    logger = ResourceEventLogger(str(tmp_path / "resource_events.jsonl"))
    logger.path.write_text('\n'.join([json.dumps({"i": i}) for i in range(1000)]) +
                           '\n{"incomplete":', encoding="utf-8")
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("loaded full event log"))
    assert logger.list(limit=3) == [{"i": 998}, {"i": 999}]
    assert logger.list(limit=0) == []


def test_failed_history_delete_does_not_claim_freed_space(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    path = tmp_path / "reports" / "report.md"
    path.parent.mkdir()
    path.write_text("retained", encoding="utf-8")
    def denied(*args, **kwargs):
        raise PermissionError("file is busy")
    monkeypatch.setattr(Path, "unlink", denied)
    assert history._unlink_files([path]) == (0, 0)
    assert path.read_text(encoding="utf-8") == "retained"


def test_history_scan_and_cleanup_do_not_follow_links(tmp_path, monkeypatch, require_symlinks):
    root, outside = tmp_path / "data", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    keep = outside / "keep.md"
    keep.write_text("retained", encoding="utf-8")
    (root / "reports").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(history, "get_project_data_dir", lambda: root)
    assert history.get_storage_stats()["reports"] == {"files": 0, "bytes": 0}
    assert history.clear_sessions() == (0, 0)
    assert keep.read_text(encoding="utf-8") == "retained"


def test_inventory_skips_malformed_manifest_shapes(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "get_project_data_dir", lambda: tmp_path)
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(tmp_path / "deps"))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for i, payload in enumerate([[], None, {"resources": []}, {"resources": {"code": 12}}]):
        (manifests / f"{i}.json").write_text(json.dumps(payload), encoding="utf-8")
    assert history.list_resource_inventory() == []


def test_history_cleanup_preserves_base_referenced_with_windows_separators(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(tmp_path))
    base = tmp_path / "repository" / "runtime" / "base"
    child = tmp_path / "repository" / "runtime" / "child"
    for directory, metadata in [(base, {}), (child, {"base_environment": r"repository\runtime\base"})]:
        directory.mkdir(parents=True)
        (directory / ".ready").write_text("ready", encoding="utf-8")
        (directory / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert history.delete_deps_cache(["repository/runtime/base"]) == (0, 0)
    assert base.is_dir()
