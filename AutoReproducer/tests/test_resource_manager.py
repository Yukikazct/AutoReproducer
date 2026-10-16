"""ResourceManager 存储管理单元测试（P0-1）。

覆盖：
1. manifest 生成/保存/读取（字段与 2026-09-09 存量格式兼容）；
2. 存量 manifest JSON 可被读取（legacy 兼容）；
3. fetch_code 幂等（缓存命中不重复克隆，占位 URL 不下载）；
4. fetch_dataset 合成冒烟集降级（离线流程验证）；
5. 权重下载降级（离线跳过不伪造、本地路径复制）；
6. cleanup 清理 L0 并记录 cleaned_at（兼容存量 test_cleanup.json）；
7. archive / restore L1 温存储往返；
8. 配额统计与 enforce_quota 建议归档；
9. stats 按论文聚合。

运行: python -m pytest tests/test_resource_manager.py -v
"""
import json
import os
import zipfile
from pathlib import Path

import pytest

sys_path_fix = Path(__file__).parent.parent
import sys  # noqa: E402
if str(sys_path_fix) not in sys.path:
    sys.path.insert(0, str(sys_path_fix))

from src.resource_manager import ResourceManager, _fmt_bytes  # noqa: E402


@pytest.fixture()
def mgr(tmp_path):
    """每个用例独立临时数据根，配额 1MB 便于触发超限。"""
    return ResourceManager(data_root=str(tmp_path / "data"),
                           quota_bytes=1024 * 1024)


# ---------------- 1. manifest 往返 ----------------

def test_manifest_roundtrip(mgr, tmp_path):
    pid = "abc123def456"
    repo = mgr._repo_dir(pid)
    (repo / "train.py").parent.mkdir(parents=True, exist_ok=True)
    (repo / "train.py").write_text("print(1)\n", encoding="utf-8")

    manifest = mgr.build_manifest(
        paper_id=pid, paper_title="Test Paper",
        code_url="https://github.com/example/repo",
        dataset_name="CIFAR-10")
    path = mgr.save_manifest(manifest)

    assert Path(path).is_file()
    loaded = mgr.get_manifest(pid)
    assert loaded["paper_id"] == pid
    assert loaded["paper_title"] == "Test Paper"
    assert loaded["resources"]["code"].endswith("repos" + os.sep + pid)
    assert loaded["resources"]["dataset"] == ""
    assert loaded["resources"]["weights"] == ""
    # 存量格式字段齐全
    for key in ("paper_id", "created_at", "resources", "paper_title",
                "code_url", "dataset_name"):
        assert key in loaded
    # 幂等可重复保存
    mgr.save_manifest(loaded)
    again = mgr.get_manifest(pid)
    assert again["paper_id"] == pid


def test_loads_legacy_manifest(mgr, tmp_path):
    """存量 2026-09-09 manifest JSON 直接可读（兼容性）。"""
    legacy = {
        "paper_id": "c539dcd696de",
        "created_at": "2026-09-09T17:23:48.734092",
        "resources": {"code": "", "dataset": "", "weights": ""},
        "paper_title": "ResNet: Deep Residual Learning",
        "code_url": "https://github.com/example/repo",
        "dataset_name": "https://example.com/dataset",
    }
    cfg = _write_legacy(mgr, legacy)
    loaded = mgr.get_manifest("c539dcd696de")
    assert loaded is not None
    assert loaded["resources"]["dataset"] == ""
    assert loaded["paper_title"] == legacy["paper_title"]
    assert len(mgr.list_manifests()) == 1
    assert cfg.unlink() or True


def _write_legacy(mgr, data: dict) -> Path:
    path = mgr._manifest_path(data["paper_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_paper_id_generation():
    pid = ResourceManager.paper_id_for("Some Title")
    assert len(pid) == 12
    assert pid == ResourceManager.paper_id_for("Some Title")
    assert ResourceManager.paper_id_for("A", corpus_key="corpus_1") == "corpus_1"


# ---------------- 2. fetch_code 懒加载 ----------------

def test_fetch_code_placeholder_skipped(mgr):
    """example.com 占位 URL 不触发克隆。"""
    info = mgr.fetch_code("abc123def456", "https://github.com/example/repo")
    assert info["state"] == "placeholder-skip"
    assert info["path"] == ""


def test_fetch_code_reuses_source_cache_but_not_the_execution_workspace(mgr, monkeypatch):
    from src import repository_reproduction as reproduction

    destinations = []

    def export(root, profile, destination, **kwargs):
        destinations.append(destination)
        destination.mkdir(parents=True)
        return {"url": profile["repository"]["url"], "path": str(destination),
                "resolved_sha": "a" * 40, "cache_path": str(root / "repository_cache" / "fixed"),
                "acquisition": "cloned" if len(destinations) == 1 else "cached"}

    monkeypatch.setattr(reproduction, "export_repository", export)
    monkeypatch.setattr(reproduction, "resolve_repository_revision", lambda *args: "a" * 40)
    first = mgr.fetch_code("paper", "https://github.com/real/repo")
    second = mgr.fetch_code("paper", "https://github.com/real/repo")
    assert first["state"] == "cloned"
    assert second["state"] == "cached"
    assert first["cache_path"] == second["cache_path"]
    assert first["path"] != second["path"]


def test_fetch_code_clone_failure_honest(mgr, monkeypatch):
    from src import repository_reproduction as reproduction

    def fail(*args):
        raise RuntimeError("repository not found")

    monkeypatch.setattr(reproduction, "resolve_repository_revision", fail)
    info = mgr.fetch_code("paper", "https://github.com/real/repo")
    assert info["state"] == "clone-failed"
    assert "repository not found" in info["detail"]
    assert info["path"] == info["commit"] == ""


# ---------------- 3. fetch_dataset 冒烟降级 ----------------

def test_fetch_dataset_smoke_offline(mgr):
    pid = "abc123def456"
    info = mgr.fetch_dataset(pid, "CIFAR-10", level="smoke")
    assert info["state"] == "smoke-synth"
    assert info["level"] == "smoke"
    assert info["rows"] == 8
    smoke = Path(info["path"])
    assert (smoke / "samples.csv").is_file()
    assert (smoke / "dataset_info.json").is_file()
    lines = (smoke / "samples.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 9          # 1 表头 + 8 样本


def test_fetch_dataset_idempotent_cached(mgr):
    pid = "abc123def456"
    mgr.fetch_dataset(pid, "CIFAR-10", level="smoke")
    info2 = mgr.fetch_dataset(pid, "CIFAR-10", level="smoke")
    assert info2["state"] == "cached"


def test_fetch_dataset_no_name(mgr):
    info = mgr.fetch_dataset("abc123def456", "")
    assert info["state"] == "skipped"


# ---------------- 4. 权重懒加载 ----------------

def test_fetch_weights_none(mgr):
    info = mgr.fetch_weights("abc123def456", "")
    assert info["state"] == "none"
    info2 = mgr.fetch_weights("abc123def456", "none")
    assert info2["state"] == "none"


def test_fetch_weights_local_copy(mgr, tmp_path):
    src = tmp_path / "pretrained.bin"
    src.write_bytes(b"\x00\x01weight")
    info = mgr.fetch_weights("abc123def456", str(src))
    assert info["state"] == "copied"
    assert Path(info["path"]).is_file()
    assert Path(info["path"]).read_bytes() == b"\x00\x01weight"


def test_fetch_weights_http_offline(mgr, monkeypatch):
    """URL 下载依赖 curl，失败/离线时诚实跳过，不伪造文件。"""
    pid = "abc123def456"

    def fake_run(cmd, **kw):
        return type("P", (), {"returncode": 3, "stderr": "404", "stdout": ""})()

    monkeypatch.setattr("src.resource_manager.subprocess.run", fake_run)
    info = mgr.fetch_weights(pid, "https://example.com/w.bin")
    assert info["state"] == "skipped"
    assert info["path"] == ""
    assert "不伪造" in info["detail"]


# ---------------- 5. cleanup ----------------

def test_cleanup_marks_cleaned_at(mgr):
    pid = "abc123def456"
    repo = mgr._repo_dir(pid)
    (repo / "run.py").parent.mkdir(parents=True, exist_ok=True)
    (repo / "run.py").write_text("print(1)\n", encoding="utf-8")
    mgr.fetch_dataset(pid, "MNIST", level="smoke")
    mgr.build_manifest(pid, paper_title="T")
    mgr.save_manifest(mgr.build_manifest(pid, paper_title="T"))
    assert mgr._repo_dir(pid).exists()

    result = mgr.cleanup(pid)
    assert not mgr._repo_dir(pid).exists()
    assert not mgr._dataset_dir(pid).exists()
    assert len(result["removed"]) == 2
    manifest = mgr.get_manifest(pid)
    assert manifest["cleaned_at"], "清理后应记录 cleaned_at"
    assert manifest["resources"]["code"] == ""
    assert manifest["resources"]["dataset"] == ""


# ---------------- 6. archive / restore ----------------

def test_archive_restore_roundtrip(mgr, tmp_path):
    pid = "abc123def456"
    repo = mgr._repo_dir(pid)
    (repo / "run.py").parent.mkdir(parents=True, exist_ok=True)
    (repo / "run.py").write_text("print(1)\n", encoding="utf-8")
    mgr.fetch_dataset(pid, "MNIST", level="smoke")
    mgr.save_manifest(mgr.build_manifest(pid, paper_title="T"))

    archive = mgr.archive(pid, dest_dir=str(tmp_path / "l1"))
    assert archive["entries"] >= 2
    assert Path(archive["archive"]).is_file()
    with zipfile.ZipFile(archive["archive"]) as zf:
        names = zf.namelist()
        assert any("run.py" in n for n in names)
        assert "manifest.json" in names

    # 清理后从 L1 恢复
    mgr.cleanup(pid)
    assert not mgr._repo_dir(pid).exists()
    res = mgr.restore(archive["archive"])
    assert res["ok"] is True
    assert mgr._repo_dir(pid).exists()
    assert (mgr._repo_dir(pid) / "run.py").is_file()
    assert (mgr._dataset_dir(pid) / "dataset_smoke" /
            "samples.csv").is_file()
    assert mgr.get_manifest(pid) is not None


def test_restore_missing_archive(mgr):
    res = mgr.restore("C:/no/such/archive.zip")
    assert res["ok"] is False


# ---------------- 7. 配额守护 ----------------

def test_quota_usage_counts_bytes(mgr):
    pid = "abc123def456"
    repo = mgr._repo_dir(pid)
    (repo / "big.bin").parent.mkdir(parents=True, exist_ok=True)
    (repo / "big.bin").write_bytes(b"\x00" * 2048)
    usage = mgr.quota_usage()
    assert usage["bytes"] >= 2048
    assert usage["quota"] == 1024 * 1024
    assert usage["percent"] > 0
    assert "repos" in usage["per_root"]
    assert _fmt_bytes(1024) == "1.0KB"


def test_enforce_quota_suggests_within_limit(mgr):
    pid = "abc123def456"
    repo = mgr._repo_dir(pid)
    (repo / "small.txt").parent.mkdir(parents=True, exist_ok=True)
    (repo / "small.txt").write_text("tiny", encoding="utf-8")
    assert mgr.enforce_quota() == []


def test_enforce_quota_suggests_cleanup(mgr, tmp_path):
    # 小配额（1KB），写入 3KB -> 超限
    tiny = ResourceManager(data_root=str(tmp_path / "d2"),
                           quota_bytes=1024)
    for i in range(3):
        p = tiny._repo_dir(f"pid{i:012x}")
        (p / f"f{i}.bin").parent.mkdir(parents=True, exist_ok=True)
        (p / f"f{i}.bin").write_bytes(b"\x00" * 1024)
    suggestions = tiny.enforce_quota(dry_run=True)
    assert suggestions, "超限应产生建议"
    assert all(s["dry_run"] is True for s in suggestions)
    assert all(s["reason"].startswith("L0 配额超限") for s in suggestions)
    # 建议按 last_used 升序（旧 -> 新）
    times = [s["last_used"] for s in suggestions]
    assert times == sorted(times)


# ---------------- 8. 统计 ----------------

def test_stats_aggregates_papers(mgr):
    pid = "abc123def456"
    repo = mgr._repo_dir(pid)
    (repo / "a.txt").parent.mkdir(parents=True, exist_ok=True)
    (repo / "a.txt").write_text("hello", encoding="utf-8")
    mgr.save_manifest(mgr.build_manifest(pid, paper_title="T"))
    stats = mgr.stats()
    assert stats["manifest_count"] == 1
    assert len(stats["papers"]) == 1
    assert stats["papers"][0]["paper_id"] == pid
    assert stats["papers"][0]["title"] == "T"
    assert stats["papers"][0]["code_bytes"] > 0
    assert stats["usage"]["bytes"] > 0


@pytest.mark.parametrize("paper_id", [
    "../outside", "a/b", "a\\b", "/tmp/outside", "..", "",
    "CON", "nul.txt", "COM1", "LPT¹.csv", "trailing.", "trailing ", "name?",
])
def test_managed_paths_reject_invalid_ids(mgr, paper_id):
    with pytest.raises(ValueError):
        mgr.cleanup(paper_id)
    with pytest.raises(ValueError):
        mgr.save_manifest({"paper_id": paper_id})


def test_manifest_failure_keeps_previous_complete_document(mgr, monkeypatch):
    mgr.save_manifest({"paper_id": "paper", "paper_title": "previous"})

    def fail_replace(*args):
        raise OSError("disk full")

    monkeypatch.setattr("src.resource_manager.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        mgr.save_manifest({"paper_id": "paper", "paper_title": "next"})
    assert mgr.get_manifest("paper")["paper_title"] == "previous"
    remaining = {path.name for path in mgr.manifests_root.iterdir()}
    assert "paper.json" in remaining
    assert remaining <= {"paper.json", "paper.lock"}  # Windows releases remove lock files.


def test_concurrent_manifest_readers_see_complete_documents(mgr):
    from concurrent.futures import ThreadPoolExecutor

    mgr.save_manifest({"paper_id": "paper", "value": "initial"})

    def write(value):
        for _ in range(20):
            mgr.save_manifest({"paper_id": "paper", "value": value * 1000})

    with ThreadPoolExecutor(max_workers=3) as executor:
        writers = [executor.submit(write, value) for value in ("a", "b")]
        for _ in range(200):
            loaded = mgr.get_manifest("paper")
            assert loaded and loaded["value"] in ("initial", "a" * 1000, "b" * 1000)
        for writer in writers:
            writer.result()


def test_manifest_reader_waits_for_another_process_writer(mgr):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    mgr.save_manifest({"paper_id": "paper", "value": "initial"})
    path = mgr._manifest_path("paper")
    script = (
        "import json,sys;from pathlib import Path;from filelock import FileLock;"
        "p=Path(sys.argv[1]);lock=FileLock(str(p.with_suffix('.lock')));"
        "lock.acquire();print('locked',flush=True);sys.stdin.readline();"
        "p.write_text(json.dumps({'paper_id':'paper','value':'new'}),encoding='utf-8');"
        "lock.release()"
    )
    process = subprocess.Popen([sys.executable, "-u", "-c", script, str(path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(mgr.get_manifest, "paper")
            try:
                from concurrent.futures import TimeoutError
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.1)
            finally:
                process.stdin.write("publish\n")
                process.stdin.flush()
            assert pending.result(timeout=5)["value"] == "new"
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def test_archive_failure_preserves_previous_archive(mgr, monkeypatch):
    repo = mgr._repo_dir("paper")
    repo.mkdir()
    (repo / "file.txt").write_text("version one")
    archive = Path(mgr.archive("paper")["archive"])
    previous = archive.read_bytes()

    def fail_write(*args):
        raise OSError("archive disk full")

    monkeypatch.setattr(zipfile.ZipFile, "write", fail_write)
    with pytest.raises(OSError, match="disk full"):
        mgr.archive("paper")
    assert archive.read_bytes() == previous
    assert list(mgr.archive_root.iterdir()) == [archive]


def test_archive_rejects_destination_inside_resources(mgr):
    repo = mgr._repo_dir("paper")
    repo.mkdir()
    with pytest.raises(ValueError, match="资源内部"):
        mgr.archive("paper", dest_dir=str(repo / "archives"))


def test_archive_cleaned_resources_does_not_replace_previous_backup(mgr):
    repo = mgr._repo_dir("paper")
    repo.mkdir()
    (repo / "file.txt").write_text("backup content")
    mgr.save_manifest(mgr.build_manifest("paper"))
    archive = Path(mgr.archive("paper")["archive"])
    previous = archive.read_bytes()
    mgr.cleanup("paper")
    with pytest.raises(ValueError, match="保留已有归档"):
        mgr.archive("paper")
    assert archive.read_bytes() == previous


def test_archive_rejects_symlinks_without_touching_target(mgr, tmp_path, require_symlinks):
    external = tmp_path / "external.txt"
    external.write_text("external")
    repo = mgr._repo_dir("paper")
    repo.mkdir()
    (repo / "link.txt").symlink_to(external)
    with pytest.raises(ValueError, match="符号链接"):
        mgr.archive("paper")
    assert external.read_text() == "external"
    assert not (mgr.archive_root / "paper.zip").exists()


def test_cleanup_rejects_symlink_cache_root(mgr, tmp_path, require_symlinks):
    outside = tmp_path / "outside" / "paper"
    outside.mkdir(parents=True)
    (outside / "keep.txt").write_text("keep")
    mgr.repos_root.rmdir()
    mgr.repos_root.symlink_to(outside.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        mgr.cleanup("paper")
    assert (outside / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("unsafe", [
    "repos/paper/../../escaped.txt", "repos/paper/../../../escaped.txt",
    "/repos/paper/escaped.txt", "repos/paper/..\\escaped.txt",
    "repos/other/file.txt",
])
def test_restore_validates_entire_archive_before_modifying_l0(mgr, tmp_path, unsafe):
    repo = mgr._repo_dir("paper")
    repo.mkdir()
    (repo / "file.txt").write_text("original")
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("repos/paper/file.txt", "replacement")
        zf.writestr(unsafe, "unwanted")
    result = mgr.restore(str(archive))
    assert not result["ok"]
    assert (repo / "file.txt").read_text() == "original"
    assert not (tmp_path / "escaped.txt").exists()
    assert not list(mgr.data_root.glob(".restore-*"))


def test_restore_rejects_existing_symlink_parent(mgr, tmp_path, require_symlinks):
    outside = tmp_path / "outside"
    outside.mkdir()
    repo = mgr._repo_dir("paper")
    repo.mkdir()
    (repo / "redirect").symlink_to(outside, target_is_directory=True)
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("repos/paper/redirect/unwanted.txt", "unwanted")
    assert not mgr.restore(str(archive))["ok"]
    assert not list(outside.iterdir())


def test_restore_corrupt_payload_keeps_existing_files(mgr, tmp_path):
    repo = mgr._repo_dir("paper")
    repo.mkdir()
    (repo / "file.txt").write_text("original")
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("repos/paper/file.txt", "replacement")
        zf.writestr("repos/paper/other.txt", "corrupt-me")
    archive.write_bytes(archive.read_bytes().replace(b"corrupt-me", b"broken-now"))
    result = mgr.restore(str(archive))
    assert not result["ok"]
    assert (repo / "file.txt").read_text() == "original"
    assert not (repo / "other.txt").exists()


def test_restore_rebases_manifest_and_preserves_empty_dirs_and_permissions(mgr, tmp_path):
    repo = mgr._repo_dir("paper")
    (repo / "empty").mkdir(parents=True)
    script = repo / "train.sh"
    script.write_text("#!/bin/sh\ntrue\n")
    script.chmod(0o755)
    mgr.fetch_dataset("paper", "MNIST")
    weights = repo / "weights" / "model.bin"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    manifest = mgr.build_manifest("paper")
    manifest["cleaned_at"] = "old timestamp"
    mgr.save_manifest(manifest)
    archive = mgr.archive("paper")
    new_manager = ResourceManager(data_root=str(tmp_path / "new-root"))
    result = new_manager.restore(archive["archive"], paper_id="paper")
    assert result["ok"]
    new_repo = new_manager._repo_dir("paper")
    assert (new_repo / "empty").is_dir()
    if os.name != "nt":
        assert (new_repo / "train.sh").stat().st_mode & 0o111
    restored_manifest = new_manager.get_manifest("paper")
    assert "cleaned_at" not in restored_manifest
    assert restored_manifest["resources"] == {
        "code": str(new_repo),
        "dataset": str(new_manager._dataset_dir("paper") / "dataset_smoke"),
        "weights": str(new_repo / "weights" / "model.bin"),
    }
    assert not new_manager.restore(archive["archive"], paper_id="another")["ok"]


def test_cleanup_reports_failures_and_retains_live_manifest_paths(mgr, monkeypatch):
    repo = mgr._repo_dir("paper")
    repo.mkdir()
    (repo / "file.txt").write_text("keep")
    mgr.fetch_dataset("paper", "MNIST")
    mgr.save_manifest(mgr.build_manifest("paper"))
    original_rmtree = __import__("shutil").rmtree

    def fail_repo(path, *args, **kwargs):
        if Path(path).samefile(repo):
            raise PermissionError("busy repository")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("src.resource_manager.shutil.rmtree", fail_repo)
    result = mgr.cleanup("paper")
    assert not result["ok"]
    assert str(repo) not in result["removed"]
    assert "busy repository" in result["errors"][0]
    manifest = mgr.get_manifest("paper")
    assert manifest["resources"]["code"] == str(repo)
    assert manifest["resources"]["dataset"] == ""
    assert not manifest.get("cleaned_at")


def test_cleanup_can_remove_manifest(mgr):
    mgr.save_manifest(mgr.build_manifest("paper"))
    assert mgr.cleanup("paper", keep_manifest=False)["ok"]
    assert mgr.get_manifest("paper") is None


def test_zero_quota_is_respected(tmp_path):
    manager = ResourceManager(data_root=str(tmp_path), quota_bytes=0)
    assert manager.quota_usage()["quota"] == 0


@pytest.mark.parametrize("alias", [
    "repos/paper/FILE.txt", "repos/paper/file.txt.", "repos/paper/file.txt ",
    "repos/paper/CON", "repos/paper/LPT1.log", "repos/paper/aux.txt",
    "repos/paper/name:stream", "repos/paper/file?.txt",
])
def test_restore_rejects_windows_names_and_aliases_before_publication(mgr, tmp_path, alias):
    repo = mgr._repo_dir("paper")
    repo.mkdir()
    original = repo / "file.txt"
    original.write_text("original")
    archive = tmp_path / "windows-alias.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("repos/paper/file.txt", "replacement")
        zf.writestr(alias, "aliased content")
    result = mgr.restore(str(archive))
    assert not result["ok"]
    assert result["restored"] == []
    assert original.read_text() == "original"


@pytest.mark.parametrize("first,second", [
    ("repos/paper/dir/a.txt", "repos/paper/DIR/b.txt"),
    ("repos/paper/caf\u00e9.txt", "repos/paper/cafe\u0301.txt"),
])
def test_restore_rejects_parent_and_unicode_aliases(mgr, tmp_path, first, second):
    archive = tmp_path / "aliases.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(first, "one")
        zf.writestr(second, "two")
    assert not mgr.restore(str(archive))["ok"]
    assert not mgr._repo_dir("paper").exists()


def test_manifest_retries_only_transient_windows_sharing_errors(mgr, monkeypatch):
    original_replace = os.replace
    attempts = []
    sleeps = []

    def locked_then_available(source, destination):
        attempts.append(source)
        if len(attempts) < 3:
            error = PermissionError("Windows sharing conflict")
            error.winerror = 32
            raise error
        return original_replace(source, destination)

    monkeypatch.setattr("src.resource_manager.os.replace", locked_then_available)
    monkeypatch.setattr("src.resource_manager.time.sleep", sleeps.append)
    mgr.save_manifest({"paper_id": "paper", "paper_title": "saved"})
    assert mgr.get_manifest("paper")["paper_title"] == "saved"
    assert len(attempts) == 3
    assert sum(sleeps) <= 0.5


def test_windows_sharing_retries_are_bounded(mgr, monkeypatch):
    mgr.save_manifest({"paper_id": "paper", "paper_title": "previous"})
    sleeps = []

    def locked(*args):
        error = PermissionError("permanently locked")
        error.winerror = 32
        raise error

    monkeypatch.setattr("src.resource_manager.os.replace", locked)
    monkeypatch.setattr("src.resource_manager.time.sleep", sleeps.append)
    with pytest.raises(PermissionError, match="permanently locked"):
        mgr.save_manifest({"paper_id": "paper", "paper_title": "next"})
    assert mgr.get_manifest("paper")["paper_title"] == "previous"
    assert len(sleeps) == 10
    assert sum(sleeps) <= 0.5


@pytest.mark.skipif(os.name != "nt", reason="Requires native Windows file sharing semantics")
@pytest.mark.parametrize("operation", ["manifest", "archive"])
def test_windows_locked_destination_preserves_previous_file(mgr, operation):
    import ctypes
    from ctypes import wintypes

    repo = mgr._repo_dir("paper")
    repo.mkdir()
    (repo / "file.txt").write_text("original")
    manifest = mgr.build_manifest("paper")
    if operation == "manifest":
        destination = Path(mgr.save_manifest(manifest))
        publish = lambda: mgr.save_manifest({**manifest, "paper_title": "new"})
    else:
        destination = Path(mgr.archive("paper")["archive"])
        publish = lambda: mgr.archive("paper")
    original = destination.read_bytes()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # Read/write sharing is allowed; replacement/deletion explicitly is not.
    handle = kernel.CreateFileW(str(destination), 0x80000000, 0x1 | 0x2,
                                None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        with pytest.raises(OSError):
            publish()
        assert destination.read_bytes() == original
        assert not list(destination.parent.glob("*.tmp"))
    finally:
        kernel.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="Requires native Windows junction support")
def test_windows_junction_is_not_traversed_by_cleanup_or_restore(mgr, tmp_path):
    import subprocess

    outside = tmp_path / "outside"
    outside.mkdir()
    keep = outside / "keep.txt"
    keep.write_text("keep")
    repo = mgr.repos_root / "paper"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(repo), str(outside)],
                   check=True, capture_output=True)
    try:
        with pytest.raises(ValueError):
            mgr.cleanup("paper")
        with pytest.raises(ValueError):
            mgr.archive("paper")
        archive = tmp_path / "junction.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("repos/paper/keep.txt", "replacement")
        assert not mgr.restore(str(archive))["ok"]
        assert keep.read_text() == "keep"
    finally:
        repo.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="Requires native Windows chmod behavior")
def test_windows_restores_posix_readonly_modes_as_writable_files(mgr, tmp_path):
    archive = tmp_path / "posix-mode.zip"
    member = zipfile.ZipInfo("repos/paper/train.sh")
    member.create_system = 3
    member.external_attr = (0o100555 << 16)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, "#!/bin/sh\ntrue\n")
    assert mgr.restore(str(archive))["ok"]
    assert mgr.cleanup("paper")["ok"]
