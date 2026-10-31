"""融合快照指纹测试：全工作区 SHA-256 指纹 + 运行中篡改检测钩子。

运行: python -m pytest tests/test_workspace_snapshot_fusion.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety.workspace_snapshot import (
    SnapshotGuard,
    changed_files,
    fingerprint_workspace,
    snapshot_workspace,
    restore_snapshot,
    workspace_fingerprint,
)


def _make_workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("import torch\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "README.txt").write_text("keep\n", encoding="utf-8")
    return tmp_path


# ---------------- 全工作区指纹 ----------------

def test_fingerprint_stable_for_same_content(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    fp1 = fingerprint_workspace(ws)
    fp2 = fingerprint_workspace(ws)
    assert fp1 == fp2
    assert len(fp1) == 64                     # SHA-256 hex


def test_fingerprint_changes_on_modify(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    before = fingerprint_workspace(ws)
    (ws / "src" / "model.py").write_text("import torch.nn\n", encoding="utf-8")
    after = fingerprint_workspace(ws)
    assert before != after


def test_fingerprint_changes_on_add_and_delete(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    base = fingerprint_workspace(ws)
    (ws / "extra.py").write_text("x = 1\n", encoding="utf-8")
    assert fingerprint_workspace(ws) != base
    (ws / "extra.py").unlink()
    (ws / "src" / "model.py").unlink()
    assert fingerprint_workspace(ws) != base


def test_snapshot_restores_and_removes_files_beyond_windows_max_path(tmp_path):
    from src.safety.paths import filesystem_path
    root = _make_workspace(tmp_path)
    relative = Path('nested') / ('a' * 100) / ('b' * 100) / 'kept.txt'
    original = Path(filesystem_path(root / relative))
    original.parent.mkdir(parents=True)
    original.write_bytes(b'original input')
    baseline = snapshot_workspace(root)
    assert baseline[relative.as_posix()]['content'] == b'original input'
    original.write_bytes(b'changed candidate')
    extra = original.with_name('candidate-output.txt')
    extra.write_bytes(b'failed output')
    restored, removed = restore_snapshot(root, baseline)
    assert relative.as_posix() in restored
    assert relative.with_name('candidate-output.txt').as_posix() in removed
    assert original.read_bytes() == b'original input'
    assert not extra.exists()
    assert snapshot_workspace(root) == baseline


def test_fingerprint_order_independent():
    """指纹仅依赖 路径+内容哈希 集合，与存储顺序无关。"""
    snap1 = {"a.py": {"sha256": "h1"}, "b.py": {"sha256": "h2"}}
    snap2 = {"b.py": {"sha256": "h2"}, "a.py": {"sha256": "h1"}}
    assert workspace_fingerprint(snap1) == workspace_fingerprint(snap2)


def test_fingerprint_discriminates_content_via_sha256():
    snap1 = {"a.py": {"sha256": "h1"}}
    snap2 = {"a.py": {"sha256": "h2"}}
    assert workspace_fingerprint(snap1) != workspace_fingerprint(snap2)


def test_snapshot_fingerprint_matches_live_fingerprint(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    snap = snapshot_workspace(ws)
    assert workspace_fingerprint(snap) == fingerprint_workspace(ws)


# ---------------- SnapshotGuard 篡改检测钩子 ----------------

def test_guard_capture_requires_before_verify(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    guard = SnapshotGuard(ws)
    with pytest.raises(RuntimeError, match="capture"):
        guard.verify()
    with pytest.raises(RuntimeError, match="capture"):
        _ = guard.baseline_fingerprint


def test_guard_verify_ok_when_untouched(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    guard = SnapshotGuard(ws)
    guard.capture()
    ok, changed = guard.verify()
    assert ok is True
    assert changed == []


def test_guard_detects_modify_and_add(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    guard = SnapshotGuard(ws)
    guard.capture()
    (ws / "src" / "model.py").write_text("tampered\n", encoding="utf-8")
    (ws / "new_file.txt").write_text("n", encoding="utf-8")
    ok, changed = guard.verify()
    assert ok is False
    assert "src/model.py" in changed
    assert "new_file.txt" in changed


def test_guard_detects_delete(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    guard = SnapshotGuard(ws)
    guard.capture()
    (ws / "data" / "README.txt").unlink()
    ok, changed = guard.verify()
    assert ok is False
    assert "data/README.txt" in changed


def test_guard_rollback_restores_baseline(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    guard = SnapshotGuard(ws)
    guard.capture()

    # 篡改：改内容 + 新增 + 删除
    (ws / "src" / "model.py").write_text("tampered\n", encoding="utf-8")
    (ws / "intruder.txt").write_text("evil", encoding="utf-8")
    (ws / "data" / "README.txt").unlink()

    restored, removed = guard.rollback()
    assert "src/model.py" in restored
    assert "data/README.txt" in restored
    assert "intruder.txt" in removed

    # 回滚后指纹回到基线
    assert fingerprint_workspace(ws) == guard.baseline_fingerprint
    ok, changed = guard.verify()
    assert ok is True and changed == []


def test_guard_rollback_twice_is_idempotent(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    guard = SnapshotGuard(ws)
    guard.capture()
    (ws / "src" / "model.py").write_text("x\n", encoding="utf-8")
    guard.rollback()
    r2, rm2 = guard.rollback()
    assert r2 == [] and rm2 == []           # 已还原，二次回滚无操作


def test_guard_works_with_custom_skip(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    (ws / "cache").mkdir()
    (ws / "cache" / "tmp.bin").write_bytes(b"\x01\x02")
    guard = SnapshotGuard(ws, skip_dirs=["cache"])
    guard.capture()
    (ws / "cache" / "tmp.bin").write_bytes(b"\xff\xff")   # skip 目录内变化不检测
    ok, changed = guard.verify()
    assert ok is True and changed == []


def test_guard_rollback_uses_skip_consistently(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    (ws / "cache").mkdir()
    guard = SnapshotGuard(ws, skip_dirs=["cache"])
    guard.capture()
    (ws / "cache" / "c.txt").write_text("t", encoding="utf-8")   # 新增于 skip 目录
    restored, removed = guard.rollback()
    assert (ws / "cache" / "c.txt").exists()      # skip 目录内不清理
    assert removed == []


def test_restore_validates_every_entry_before_changing_any_file(tmp_path):
    import hashlib

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "model.py"
    target.write_bytes(b"changed\n")
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")
    content = b"original\n"
    entry = {"sha256": hashlib.sha256(content).hexdigest(), "content": content}

    with pytest.raises(ValueError):
        restore_snapshot(workspace, {"model.py": entry, "../outside.py": entry})

    assert target.read_bytes() == b"changed\n"
    assert outside.read_bytes() == b"outside\n"


def test_restore_rejects_corrupt_snapshot_before_writing(tmp_path):
    target = tmp_path / "model.py"
    target.write_bytes(b"original\n")
    baseline = snapshot_workspace(tmp_path)
    target.write_bytes(b"changed\n")
    baseline["model.py"]["content"] = b"corrupt\n"

    with pytest.raises(ValueError, match="snapshot"):
        restore_snapshot(tmp_path, baseline)

    assert target.read_bytes() == b"changed\n"


def test_snapshot_skip_names_apply_to_relative_directories(tmp_path):
    workspace = tmp_path / ".git" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "model.py").write_bytes(b"original\n")

    assert set(snapshot_workspace(workspace)) == {"model.py"}


def test_snapshot_rejects_file_symlink(tmp_path, require_symlinks):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")
    (workspace / "model.py").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink|junction"):
        snapshot_workspace(workspace)


def test_restore_rejects_swapped_link_before_changing_any_file(tmp_path, require_symlinks):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = workspace / "model.py"
    model.write_bytes(b"original\n")
    other = workspace / "other.py"
    other.write_bytes(b"other\n")
    baseline = snapshot_workspace(workspace)
    other.write_bytes(b"changed\n")
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")
    model.unlink()
    model.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink|junction"):
        restore_snapshot(workspace, baseline)

    assert outside.read_bytes() == b"outside\n"
    assert other.read_bytes() == b"changed\n"


def test_restore_does_not_write_through_hardlink(tmp_path):
    import os

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = workspace / "model.py"
    model.write_bytes(b"original\n")
    baseline = snapshot_workspace(workspace)
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")
    model.unlink()
    os.link(outside, model)

    restore_snapshot(workspace, baseline)

    assert model.read_bytes() == b"original\n"
    assert outside.read_bytes() == b"outside\n"


def test_restore_empty_snapshot_keeps_workspace_root(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "candidate.py").write_bytes(b"candidate\n")
    restore_snapshot(workspace, {})
    assert workspace.is_dir()
    assert list(workspace.iterdir()) == []


def test_restore_rejects_noncanonical_snapshot_keys(tmp_path):
    target = tmp_path / "model.py"
    target.write_bytes(b"original\n")
    entry = snapshot_workspace(tmp_path)["model.py"]
    target.write_bytes(b"changed\n")
    with pytest.raises(ValueError):
        restore_snapshot(tmp_path, {"./model.py": entry})
    assert target.read_bytes() == b"changed\n"
