"""Legacy resource fetching must use isolated exports of real fixed Git objects."""
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.resource_manager import ResourceManager


def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], stderr=subprocess.PIPE,
    ).decode().strip()


@pytest.fixture()
def source(tmp_path):
    repo = tmp_path / "作者源码 with spaces"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.name", "Repository Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "core.autocrlf", "false")
    (repo / "model.py").write_bytes(b"VERSION = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "first version")
    first = git(repo, "rev-parse", "HEAD")
    git(repo, "tag", "-a", "v1", "-m", "first version")
    (repo / "model.py").write_bytes(b"VERSION = 2\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "second version")
    return repo, first, git(repo, "rev-parse", "HEAD")


def test_different_commits_and_reruns_have_independent_clean_workspaces(tmp_path, source):
    repo, first, second = source
    manager = ResourceManager(data_root=tmp_path / "data")
    one = manager.fetch_code("paper", str(repo), revision=first)
    assert one["commit"] == first
    assert (Path(one["path"]) / "model.py").read_bytes() == b"VERSION = 1\n"
    (Path(one["path"]) / "model.py").write_bytes(b"LOCAL REPAIR\n")
    (Path(one["path"]) / "result.json").write_text("old results", encoding="utf-8")
    (Path(one["cache_path"]) / "model.py").write_bytes(b"DIRTY CACHE\n")

    two = manager.fetch_code("paper", str(repo), revision=second)
    replay = manager.fetch_code("paper", str(repo), revision=first)
    assert len({one["path"], two["path"], replay["path"]}) == 3
    assert one["cache_path"] != two["cache_path"]
    assert one["cache_path"] == replay["cache_path"]
    assert replay["state"] == "cached"
    assert (Path(two["path"]) / "model.py").read_bytes() == b"VERSION = 2\n"
    assert (Path(replay["path"]) / "model.py").read_bytes() == b"VERSION = 1\n"
    assert not (Path(replay["path"]) / "result.json").exists()
    assert not (Path(replay["path"]) / ".git").exists()
    assert (Path(one["path"]) / "model.py").read_bytes() == b"LOCAL REPAIR\n"
    assert replay["files"]["model.py"] == hashlib.sha256(b"VERSION = 1\n").hexdigest()


def test_unavailable_revision_never_returns_head_or_overwrites_prior_source(tmp_path, source):
    repo, first, _ = source
    manager = ResourceManager(data_root=tmp_path / "data")
    good = manager.fetch_code("paper", str(repo), revision=first)
    marker = Path(good["path"]) / ".autorepro-repo-source.json"
    original = marker.read_bytes()
    failed = manager.fetch_code("paper", str(repo), revision="0" * 40)
    assert failed["state"] == "pin-failed"
    assert failed["path"] == failed["commit"] == ""
    assert marker.read_bytes() == original


@pytest.mark.parametrize("revision", ["", "v1", "HEAD"])
def test_unpinned_and_named_references_record_the_resolved_commit(tmp_path, source, revision):
    repo, first, second = source
    manager = ResourceManager(data_root=tmp_path / "data")
    prepared = manager.fetch_code("paper", str(repo), revision=revision)
    expected = first if revision == "v1" else second
    assert prepared["commit"] == prepared["resolved_sha"] == expected
    marker = json.loads((Path(prepared["path"]) / ".autorepro-repo-source.json").read_text(encoding="utf-8"))
    assert marker["revision"] == revision
    assert marker["commit"] == expected
    assert marker["archive_sha256"] == prepared["archive_sha256"]


def test_legacy_directory_and_another_repository_are_not_reused(tmp_path, source):
    repo, first, _ = source
    manager = ResourceManager(data_root=tmp_path / "data")
    legacy = manager._repo_dir("paper")
    legacy.mkdir()
    (legacy / "model.py").write_bytes(b"UNVERIFIED LEGACY\n")
    prepared = manager.fetch_code("paper", str(repo), revision=first)
    assert (legacy / "model.py").read_bytes() == b"UNVERIFIED LEGACY\n"
    other = tmp_path / "another source"
    subprocess.run(["git", "clone", str(repo), str(other)], check=True, capture_output=True)
    other_prepared = manager.fetch_code("paper", str(other), revision=first)
    assert prepared["cache_path"] != other_prepared["cache_path"]
    assert (Path(other_prepared["path"]) / "model.py").read_bytes() == b"VERSION = 1\n"


def test_explicit_workspace_is_not_overwritten_or_reported_as_a_cache_hit(tmp_path, source):
    repo, first, second = source
    manager = ResourceManager(data_root=tmp_path / "data")
    target = tmp_path / "workspace"
    prepared = manager.fetch_code("paper", str(repo), target=str(target), revision=first)
    before = (target / ".autorepro-repo-source.json").read_bytes()
    failed = manager.fetch_code("paper", str(repo), target=str(target), revision=second)
    assert prepared["commit"] == first
    assert failed["path"] == ""
    assert (target / "model.py").read_bytes() == b"VERSION = 1\n"
    assert (target / ".autorepro-repo-source.json").read_bytes() == before


def test_manifest_preserves_exact_workspace_and_source_hashes(tmp_path, source):
    repo, first, _ = source
    manager = ResourceManager(data_root=tmp_path / "data")
    prepared = manager.fetch_code("paper", str(repo), revision=first)
    manifest = manager.build_manifest("paper", code_url=str(repo), code_snapshot=prepared)
    manager.save_manifest(manifest)
    saved = manager.get_manifest("paper")
    assert saved["resources"]["code"] == prepared["path"]
    assert saved["repository"]["resolved_sha"] == first
    assert saved["repository"]["files"] == prepared["files"]


def test_restore_rebases_the_isolated_workspace_and_preserves_source_evidence(tmp_path, source):
    repo, first, _ = source
    manager = ResourceManager(data_root=tmp_path / "data")
    prepared = manager.fetch_code("paper", str(repo), revision=first)
    manager.save_manifest(manager.build_manifest("paper", code_url=str(repo), code_snapshot=prepared))
    archive = manager.archive("paper")
    restored = ResourceManager(data_root=tmp_path / "restored")
    restored.restore(archive["archive"])
    manifest = restored.get_manifest("paper")
    path = Path(manifest["resources"]["code"])
    assert path.is_relative_to(restored.data_root)
    assert manifest["repository"]["path"] == str(path)
    assert manifest["repository"]["resolved_sha"] == first
    assert manifest["repository"]["files"] == prepared["files"]
    assert (path / "model.py").read_bytes() == b"VERSION = 1\n"


def test_wrong_cache_origin_is_not_treated_as_the_requested_repository(tmp_path, source):
    repo, first, _ = source
    manager = ResourceManager(data_root=tmp_path / "data")
    prepared = manager.fetch_code("paper", str(repo), revision=first)
    cache = Path(prepared["cache_path"])
    git(cache, "remote", "set-url", "origin", str(tmp_path / "wrong source"))
    replay = manager.fetch_code("paper", str(repo), revision=first)
    assert replay["state"] == "cloned"
    assert replay["commit"] == first
    assert git(Path(replay["cache_path"]), "remote", "get-url", "origin") == str(repo)
    assert list(cache.parent.glob(".invalid-*")), "preserve the old cache for diagnosis"


def test_ambiguous_branch_and_tag_do_not_choose_an_arbitrary_version(tmp_path, source):
    repo, _, second = source
    git(repo, "branch", "v1", second)
    manager = ResourceManager(data_root=tmp_path / "data")
    ambiguous = manager.fetch_code("paper", str(repo), revision="v1")
    assert ambiguous["state"] == "pin-failed"
    assert ambiguous["path"] == ""
    explicit = manager.fetch_code("paper", str(repo), revision="refs/heads/v1")
    assert explicit["commit"] == second


def test_report_exposes_source_version_without_claiming_repository_execution(tmp_path, source):
    from src.audit.audit_logger import AuditLogger
    from src.agents.report_generator import ReportGeneratorAgent

    repo, first, _ = source
    manager = ResourceManager(data_root=tmp_path / "data")
    prepared = manager.fetch_code("paper", str(repo), revision=first)
    report = ReportGeneratorAgent(AuditLogger()).run(
        {"repository": prepared}, report_path=str(tmp_path / "report.md"))["report"]
    assert first in report
    assert prepared["archive_sha256"] in report
    assert prepared["path"] in report
    assert "实际执行来源以执行记录为准" in report


def test_source_record_write_failure_does_not_publish_a_ready_workspace(tmp_path, source, monkeypatch):
    repo, first, _ = source
    manager = ResourceManager(data_root=tmp_path / "data")

    def fail(*args, **kwargs):
        raise OSError("source record cannot be written")

    monkeypatch.setattr(manager, "_record_repo_source", fail)
    prepared = manager.fetch_code("paper", str(repo), revision=first)
    assert prepared["state"] == "pin-failed"
    assert prepared["path"] == prepared["commit"] == ""
    assert "resolved_sha" not in prepared


@pytest.mark.parametrize("failure", ["result", "exception"])
def test_orchestrator_stops_before_environment_and_execution_when_pin_fails(tmp_path, monkeypatch, failure):
    from src.audit.audit_logger import AuditLogger
    from src.llm.llm_client import LLMClient
    from src.orchestrator import Orchestrator

    manager = ResourceManager(data_root=tmp_path / "data")
    if failure == "result":
        manager.fetch_code = Mock(return_value={"state": "pin-failed", "path": "", "commit": "", "detail": "missing commit"})
    else:
        manager.fetch_code = Mock(side_effect=RuntimeError("missing commit"))
    orchestrator = Orchestrator(llm_client=LLMClient(mock_mode=True), mock_mode=False,
                                logger=AuditLogger(), resource_manager=manager)
    monkeypatch.setattr(orchestrator.agents["reader"], "run", Mock(return_value={"paper_info": {"title": "Pinned source fixture"}}))
    monkeypatch.setattr(orchestrator.agents["finder"], "run", Mock(return_value={"resources": {
        "code_repo_url": "https://github.com/owner/repo",
        "repo_discovery": {"selected_repo": "https://github.com/owner/repo", "pinned_revision": "0" * 40},
    }}))
    builder = Mock(side_effect=AssertionError("must not prepare environment"))
    executor = Mock(side_effect=AssertionError("must not execute fallback code"))
    monkeypatch.setattr(orchestrator.agents["builder"], "run", builder)
    monkeypatch.setattr(orchestrator.agents["executor"], "run", executor)
    result = orchestrator.run({"paper_title": "Pinned source fixture"})
    assert result["state"] == "ERROR"
    assert "固定" in result["error"]
    builder.assert_not_called()
    executor.assert_not_called()
