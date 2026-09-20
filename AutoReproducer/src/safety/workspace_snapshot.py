"""WorkspaceSnapshot - 工作区快照与回滚（安全网 A3-2，借鉴 ScholarAgent 的 workspace_snapshot 设计）。

真实优化执行（或任何破坏性尝试）前，对目标工作区做一次快照：
每条文件记录 {相对路径: {"sha256": 指纹, "content": 内容副本}}。

执行失败、验收不通过或补丁被拒绝时，调用 :func:`restore_snapshot` 全量还原:

- 快照存在但当前缺失 / 内容不符的文件 -> 重建为快照版本；
- 快照时刻之后新增的文件 -> 删除。

保证论文原始仓库在任意失败路径下都能恢复原状；快照跳过 .git 等
内部目录，不复制版本控制数据。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

# 快照条目的类型别名：{rel_path: {"sha256": str, "content": bytes}}
Snapshot = Dict[str, Dict[str, object]]

_DEFAULT_SKIP_DIRS: Tuple[str, ...] = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
)
_DEFAULT_SKIP_FILES: Tuple[str, ...] = (".DS_Store",)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_workspace_files(root: Path, skip_dirs: Optional[Iterable[str]] = None,
                         skip_files: Optional[Iterable[str]] = None) -> Iterable[Path]:
    """遍历工作区内的普通文件（跳过 .git、缓存目录与临时文件）。"""
    skip_dirs = set(skip_dirs or _DEFAULT_SKIP_DIRS)
    skip_files = set(skip_files or _DEFAULT_SKIP_FILES)
    root = Path(root)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(d in path.parts for d in skip_dirs):
            continue
        if path.name in skip_files:
            continue
        yield path


def snapshot_workspace(root: Path, skip_dirs: Optional[Iterable[str]] = None,
                       skip_files: Optional[Iterable[str]] = None) -> Snapshot:
    """对工作区生成 {相对路径: {"sha256": 指纹, "content": 内容}} 快照。"""
    root = Path(root)
    snap: Snapshot = {}
    for f in iter_workspace_files(root, skip_dirs=skip_dirs, skip_files=skip_files):
        rel = f.relative_to(root).as_posix()
        data = f.read_bytes()
        snap[rel] = {"sha256": _sha256_bytes(data), "content": data}
    return snap


def changed_files(before: Snapshot, after: Snapshot) -> List[str]:
    """对比两个快照，返回新增 / 删除 / 内容变化的相对路径（字典序）。"""
    changed = []
    for rel in sorted(set(before) | set(after)):
        if before.get(rel, {}).get("sha256") != after.get(rel, {}).get("sha256"):
            changed.append(rel)
    return changed


def workspace_fingerprint(snapshot: Mapping[str, Mapping[str, object]]) -> str:
    """对整份快照生成全局 SHA-256 指纹。

    仅基于 {相对路径, sha256} 组合（不含内容副本），路径字典序拼接：
    同一工作区内容 => 同一指纹；任一文件增删改 => 指纹变化。
    """
    h = hashlib.sha256()
    for _rel in sorted(snapshot):
        rel = str(_rel)
        sel_hash = snapshot[rel]["sha256"]
        assert isinstance(sel_hash, str), "快照条目 sha256 必须为字符串"
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(sel_hash.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def fingerprint_workspace(root: Path, skip_dirs: Optional[Iterable[str]] = None,
                          skip_files: Optional[Iterable[str]] = None) -> str:
    """对当前工作区直接计算全量指纹（不保留内容副本，轻量探测用）。"""
    return workspace_fingerprint(
        snapshot_workspace(root, skip_dirs=skip_dirs, skip_files=skip_files))


class SnapshotGuard:
    """运行中篡改检测钩子（安全网 A3-3，融合 ScholarAgent 指纹校验）。

    用法::

        guard = SnapshotGuard(workspace)
        baseline = guard.capture()          # 执行前记录基线指纹
        ... 执行优化 / 复现循环 ...
        ok, changed = guard.verify()        # 运行中任何时点探测篡改
        if not ok:
            restored, removed = guard.rollback()   # 中止恢复：还原基线

    verify() 逐文件比对当前状态与基线指纹，返回 (是否一致, 变更文件列表)；
    指纹不一致（文件被外部进程/提权样例改写）时由调用方决定中止并
    rollback() 恢复原始工作区（等价于 restore_snapshot(baseline)）。
    """

    def __init__(self, root: Path, skip_dirs: Optional[Iterable[str]] = None,
                 skip_files: Optional[Iterable[str]] = None) -> None:
        self._root = Path(root)
        self._skip_dirs = skip_dirs
        self._skip_files = skip_files
        self._baseline: Optional[Snapshot] = None
        self._baseline_fingerprint: Optional[str] = None

    # ---------------- 基线 ----------------

    def capture(self) -> str:
        """记录当前工作区为基线，返回基线指纹。"""
        snap = snapshot_workspace(self._root, skip_dirs=self._skip_dirs,
                                  skip_files=self._skip_files)
        self._baseline = snap
        self._baseline_fingerprint = workspace_fingerprint(snap)
        return self._baseline_fingerprint

    @property
    def baseline_fingerprint(self) -> str:
        if self._baseline_fingerprint is None:
            raise RuntimeError("SnapshotGuard.capture() 必须先调用")
        return self._baseline_fingerprint

    @property
    def baseline_snapshot(self) -> Snapshot:
        if self._baseline is None:
            raise RuntimeError("SnapshotGuard.capture() 必须先调用")
        return self._baseline

    # ---------------- 运行中检测 ----------------

    def current_fingerprint(self) -> str:
        """不保留内容副本的轻量当前指纹。"""
        return fingerprint_workspace(self._root, skip_dirs=self._skip_dirs,
                                     skip_files=self._skip_files)

    def verify(self) -> Tuple[bool, List[str]]:
        """比对当前工作区与基线。返回 (是否一致, 变更文件列表)。"""
        baseline = self.baseline_snapshot
        current = snapshot_workspace(self._root, skip_dirs=self._skip_dirs,
                                     skip_files=self._skip_files)
        changed = changed_files(baseline, current)
        return (not changed, changed)

    # ---------------- 中止恢复 ----------------

    def rollback(self) -> Tuple[List[str], List[str]]:
        """以基线为准还原工作区，返回 (已恢复文件列表, 已删除的新增文件列表)。

        等价于 :func:`restore_snapshot` + 基线快照：快照中缺失或内容不符的
        文件重建为基线版本，快照后新增的文件删除，保证原始仓库在篡改 /
        失败路径下完全还原。
        """
        restored, removed = restore_snapshot(
            self._root, self.baseline_snapshot,
            skip_dirs=self._skip_dirs, skip_files=self._skip_files)
        return restored, removed


def restore_snapshot(root: Path, snapshot: Snapshot,
                     skip_dirs: Optional[Iterable[str]] = None,
                     skip_files: Optional[Iterable[str]] = None) -> Tuple[List[str], List[str]]:
    """按快照还原工作区。返回 (已恢复文件列表, 已删除的新增文件列表)。

    - 快照中存在、但当前缺失或内容不符 -> 重建为快照版本（记入 restored）；
    - 快照后新增的文件 -> 删除（记入 removed）。
    """
    root = Path(root)
    restored: List[str] = []
    removed: List[str] = []

    # 1) 还原快照内文件
    for rel, entry in snapshot.items():
        target = root / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(entry["content"])  # type: ignore[arg-type]
            restored.append(rel)
        elif isinstance(entry["content"], bytes) and \
                _sha256_bytes(target.read_bytes()) != entry["sha256"]:
            target.write_bytes(entry["content"])
            restored.append(rel)

    # 2) 删除快照之后新增的文件（先收集再删除，避免遍历中目录变化）
    current = list(iter_workspace_files(root, skip_dirs=skip_dirs,
                                        skip_files=skip_files))
    for f in current:
        rel = f.relative_to(root).as_posix()
        if rel not in snapshot:
            f.unlink()
            removed.append(rel)

    for d in sorted({p.parent for p in current}, key=lambda v: -len(v.parts)):
        try:
            d.rmdir()          # 清理可能变空的目录（非递归，失败忽略）
        except OSError:
            pass

    return restored, removed