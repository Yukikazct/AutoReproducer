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