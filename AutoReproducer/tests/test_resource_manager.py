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


def test_fetch_code_idempotent(mgr, monkeypatch):
    """已缓存时重复 fetch 不克隆。"""
    pid = "abc123def456"
    repo = mgr._repo_dir(pid)
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "model.py").write_text("x=1\n", encoding="utf-8")
    calls = []

    def fake_clone(cmd, **kw):
        calls.append(cmd)
        return type("P", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr("src.resource_manager.subprocess.run", fake_clone)
    info = mgr.fetch_code(pid, "https://github.com/real/repo")
    assert info["state"] == "cached"
    assert not calls, "缓存命中不应再次克隆"


def test_fetch_code_clone_failure_honest(mgr, monkeypatch):
    """克隆失败时诚实标注 clone-failed，不伪造成功。"""
    pid = "abc123def456"

    def fake_clone(cmd, **kw):
        return type("P", (), {
            "returncode": 128,
            "stderr": "fatal: repository not found",
            "stdout": ""})()

    monkeypatch.setattr("src.resource_manager.subprocess.run", fake_clone)
    info = mgr.fetch_code(pid, "https://github.com/real/repo")
    assert info["state"] == "clone-failed"
    assert "repository not found" in info["detail"]
    assert info["path"] == ""


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