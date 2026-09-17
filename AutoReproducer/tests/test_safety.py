"""安全网模块测试：补丁白名单策略 + 工作区快照回滚。

运行: python -m pytest tests/test_safety.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety.patch_policy import PatchPolicy
from src.safety.workspace_snapshot import (
    snapshot_workspace,
    restore_snapshot,
    changed_files,
)


# ---------------- PatchPolicy 白名单 ----------------

def test_patch_policy_allows_editable_file():
    policy = PatchPolicy(editable=["src/model.py", "train.py"])
    ok, reason = policy.check("src/model.py")
    assert ok
    assert "允许" in reason
    assert policy.editable == ("src/model.py", "train.py")


def test_patch_policy_protected_overrides_editable():
    """protected 兜底优先于 editable：即使登记进白名单也不得越界。"""
    policy = PatchPolicy(editable=["data/train.csv", ".git/config"])
    assert policy.check("data/train.csv")[0] is False
    assert policy.check(".git/config")[0] is False


def test_patch_policy_rejects_unknown_path():
    policy = PatchPolicy(editable=["src/model.py"])
    assert policy.check("src/other.py")[0] is False


def test_patch_policy_rejects_path_traversal():
    policy = PatchPolicy(editable=["src/model.py"])
    assert policy.check("../secret.txt")[0] is False
    assert policy.check("src/../../etc/passwd")[0] is False


def test_patch_policy_rejects_absolute_path():
    policy = PatchPolicy(editable=["src/model.py"])
    assert policy.check("/etc/passwd")[0] is False
    assert policy.check("C:/Windows/system32/x.dll")[0] is False


def test_patch_policy_normalizes_backslash_and_dot():
    policy = PatchPolicy(editable=["src/model.py"])
    assert policy.check("src\\model.py")[0] is True
    assert policy.check("./src/model.py")[0] is True


def test_patch_policy_default_protected_prefixes():
    policy = PatchPolicy(editable=["anything.py"])
    for path in (".git/config", "data/train.csv", "tests/test_x.py",
                 "checkpoints/best.pt", "__pycache__/x.pyc"):
        assert policy.check(path)[0] is False, path


def test_patch_policy_nested_protected_dir():
    """受保护目录在任意层级都应判 protected（不只首段）。"""
    policy = PatchPolicy(editable=["run.py"])
    for path in ("data/config.py", "src/data/config.py",
                 "tests/unit/test_x.py", "pkg/checkpoints/best.pt"):
        assert policy.classify(path) == "protected", path


def test_patch_policy_protects_secret_files():
    """密钥/凭据文件（含后缀 .pem/.key）一律 protected。"""
    policy = PatchPolicy(editable=["run.py"])
    for path in (".env", "config/.env", "id_rsa", "server.key", "ca.pem",
                 "credentials.json"):
        assert policy.classify(path) == "protected", path
    # 正常文件不受影响
    assert policy.classify("run.py") == "editable"


# ---------------- WorkspaceSnapshot 快照与回滚 ----------------

@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text(
        "def train(): pass\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text(
        "ref: refs/heads/main\n", encoding="utf-8")
    return tmp_path


def test_snapshot_records_sha256_and_content(workspace):
    snap = snapshot_workspace(workspace)
    assert "src/model.py" in snap
    assert snap["src/model.py"]["sha256"]
    assert snap["src/model.py"]["content"].startswith(b"def train")


def test_snapshot_skips_git_and_cache_dirs(workspace):
    snap = snapshot_workspace(workspace)
    assert ".git/HEAD" not in snap
    assert "__pycache__" not in snap


def test_restore_reverts_modified_file(workspace):
    snap = snapshot_workspace(workspace)
    (workspace / "src" / "model.py").write_text(
        "def broken(): 语法错误\n", encoding="utf-8")
    restored, removed = restore_snapshot(workspace, snap)
    assert "src/model.py" in restored
    assert removed == []
    assert (workspace / "src" / "model.py").read_text(
        encoding="utf-8") == "def train(): pass\n"


def test_restore_rebuilds_deleted_file(workspace):
    snap = snapshot_workspace(workspace)
    (workspace / "src" / "model.py").unlink()
    restored, _ = restore_snapshot(workspace, snap)
    assert "src/model.py" in restored
    assert (workspace / "src" / "model.py").exists()


def test_restore_removes_new_files(workspace):
    snap = snapshot_workspace(workspace)
    new_file = workspace / "src" / "temp_patch.py"
    new_file.write_text("x = 1\n", encoding="utf-8")
    cached = workspace / "src" / "tmp" / "cache.bin"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"junk")
    restored, removed = restore_snapshot(workspace, snap)
    assert "src/temp_patch.py" in removed
    assert "src/tmp/cache.bin" in removed
    assert not new_file.exists()
    assert not cached.exists()
    assert not (workspace / "src" / "tmp").exists()   # 空目录被清理
    assert restored == []


def test_restore_keeps_untouched_files(workspace):
    snap = snapshot_workspace(workspace)
    restored, removed = restore_snapshot(workspace, snap)
    assert restored == []
    assert removed == []
    assert (workspace / "data" / "train.csv").read_text(
        encoding="utf-8") == "x,y\n1,2\n"


def test_changed_files_reports_all_diffs(workspace):
    before = snapshot_workspace(workspace)
    (workspace / "src" / "model.py").write_text("changed\n", encoding="utf-8")
    (workspace / "new.py").write_text("n\n", encoding="utf-8")
    (workspace / "data" / "train.csv").unlink()
    after = snapshot_workspace(workspace)
    changed = changed_files(before, after)
    assert "src/model.py" in changed
    assert "new.py" in changed
    assert "data/train.csv" in changed