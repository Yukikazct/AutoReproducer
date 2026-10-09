"""Storage UI snapshots reuse size scans without changing displayed totals."""
from collections import Counter
import json
import os

import pytest

from frontend import history_manager as history


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize("external_deps", [False, True])
def test_snapshot_matches_public_views_and_scans_shared_payload_once(tmp_path, monkeypatch, external_deps):
    base = tmp_path / "data"
    monkeypatch.setattr(history, "get_project_data_dir", lambda: base)
    deps = tmp_path / "external-deps" if external_deps else base / "deps"
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(deps))
    environment = deps / "repository" / "runtime" / "hash"
    payload = environment / "package" / "nested"
    write(payload / "module.py", "a = 1\n")
    write(environment / ".ready", "ready")
    write(environment / "meta.json", json.dumps({"kind": "reqs", "requirements": "package==1",
                                                "last_used": "2026-10-09T10:00:00"}))
    (environment / "package-1.dist-info").mkdir()
    source = base / "repos" / "author" / "source"
    write(source / "model.py", "model")
    dataset = base / "datasets" / "observations.csv"
    write(dataset, "x,y\n1,2\n")
    write(base / "reports" / "report.md", "报告")
    write(base / "runs" / "not-in-storage-categories.bin", "excluded")
    write(base / "manifests" / "paper.json", json.dumps({"paper_id": "paper", "resources": {
        "code": str(source), "dataset": str(dataset)}}))
    expected = {"storage": history.get_storage_stats(), "deps_items": history.list_deps_cache(),
                "inventory": history.list_resource_inventory()}

    visited = Counter()
    original_scandir = os.scandir
    def scandir(path):
        visited[os.path.normcase(os.path.abspath(path))] += 1
        return original_scandir(path)
    monkeypatch.setattr(history.os, "scandir", scandir)
    monkeypatch.setattr(history, "_dir_size", lambda path: pytest.fail("snapshot used an unshared recursive scan"))

    assert history.collect_storage_snapshot() == expected
    # A deep dependency payload is shared by stats, the cache list and inventory.
    assert visited[os.path.normcase(str(payload))] == 1
    # The repository payload is shared by aggregate stats and its manifest row.
    assert visited[os.path.normcase(str(source))] == 1
    assert visited[os.path.normcase(str(base / "runs"))] == 0


def test_snapshots_refresh_in_memory_without_writing_a_cache(tmp_path, monkeypatch):
    base = tmp_path / "data"
    monkeypatch.setattr(history, "get_project_data_dir", lambda: base)
    monkeypatch.setenv("AUTOREPRO_DEPS_ROOT", str(base / "deps"))
    path = base / "datasets" / "sample.txt"
    write(path, "abc")
    first = history.collect_storage_snapshot()
    path.write_text("abcdef", encoding="utf-8")
    second = history.collect_storage_snapshot()
    assert first["storage"]["datasets"] == {"files": 1, "bytes": 3}
    assert second["storage"]["datasets"] == {"files": 1, "bytes": 6}
    assert first["deps_items"] == second["deps_items"] == []
    assert list(base.rglob("*")) == [path.parent, path]
