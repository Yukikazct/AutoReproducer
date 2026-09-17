"""PatchPolicy - 补丁白名单策略（安全网 A3-1，借鉴 ScholarAgent 的 patch_policy 设计）。

任何 LLM 产物（优化补丁、自动改写）在修改工作区文件前，必须先通过本策略判定：

- editable 白名单：显式登记允许修改的相对路径（可多条）；
- protected 兜底规则：.git / data / tests / __pycache__ 等目录与 .gitignore
  等关键文件一律拒绝——即使被登记进 editable 也不得越界，
  防止优化过程破坏论文原始仓库与评测数据；
- 路径归一化（统一正斜杠、剥离 ./）后判定；绝对路径、空路径、.. 穿越
  一律视为非法，直接拒绝。

典型用法::

    policy = PatchPolicy(editable=["src/model.py", "train.py"])
    ok, reason = policy.check("src/model.py")   # (True, 允许说明)
    ok, reason = policy.check("data/train.csv") # (False, 受保护说明)
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

# 默认受保护目录前缀（命中顶层或任意层级即拒绝修改）
DEFAULT_PROTECTED_PREFIXES: Tuple[str, ...] = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".github",
    ".idea",
    ".vscode",
    "data",
    "dataset",
    "datasets",
    "checkpoint",
    "checkpoints",
    "tests",
)

# 默认受保护文件：无论位于哪个目录都不允许修改
DEFAULT_PROTECTED_FILES: Tuple[str, ...] = (
    ".gitignore",
    "poetry.lock",
    "Pipfile.lock",
    ".env",
    ".env.local",
    "id_rsa",
    "id_ed25519",
    "id_dsa",
    "credentials.json",
    "credentials.yml",
    "credentials.yaml",
)

# 密钥/凭据类文件后缀（任意文件名命中，如 server.key / ca.pem）
DEFAULT_PROTECTED_SUFFIXES: Tuple[str, ...] = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
)


class PatchPolicy:
    """补丁白名单策略：只放行 editable 白名单内且未被 protected 兜底拦截的路径。"""

    def __init__(self, editable: Optional[Iterable[str]] = None,
                 protected_prefixes: Optional[Iterable[str]] = None,
                 protected_files: Optional[Iterable[str]] = None,
                 protected_suffixes: Optional[Iterable[str]] = None) -> None:
        self._editable = {self._normalize(p) for p in (editable or [])
                          if self._normalize(p)}
        self._protected_prefixes = tuple(
            protected_prefixes or DEFAULT_PROTECTED_PREFIXES)
        self._protected_files = tuple(protected_files or DEFAULT_PROTECTED_FILES)
        self._protected_suffixes = tuple(
            protected_suffixes or DEFAULT_PROTECTED_SUFFIXES)

    @staticmethod
    def _normalize(path: object) -> str:
        """归一化为相对 POSIX 路径；非法路径（绝对/空/穿越）返回空字符串。"""
        p = str(path).replace("\\", "/").strip()
        if not p or p.startswith("//") or p[:1] == "/":
            return ""          # 空路径或 POSIX 绝对路径
        if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
            return ""          # Windows 盘符绝对路径（C:/...）
        while p.startswith("./"):     # 只剥 ./ 前缀，保留 .git 等点开头目录
            p = p[2:]
        if not p or p.startswith("/"):
            return ""
        if ".." in p.split("/"):      # 任何层级的 .. 穿越都拒绝
            return ""
        return p

    def classify(self, path: object) -> str:
        """返回路径类别：editable / protected / unknown。"""
        norm = self._normalize(path)
        if not norm:
            return "unknown"
        if any(norm == f or norm.endswith("/" + f) for f in self._protected_files):
            return "protected"
        if any(norm.endswith(s) for s in self._protected_suffixes):
            return "protected"
        # protected 目录前缀：任意层级命中即拒绝（不只首段），
        # 例如 src/data/x 与 data/x 一样受保护。
        if any(seg in self._protected_prefixes for seg in norm.split("/")):
            return "protected"
        return "editable" if norm in self._editable else "unknown"

    def check(self, path: object) -> Tuple[bool, str]:
        """判定补丁能否修改 path。返回 (是否允许, 说明)。"""
        norm = self._normalize(path)
        if not norm:
            return False, f"非法路径(空/绝对/..穿越),拒绝修改: {path}"
        kind = self.classify(norm)
        if kind == "editable":
            return True, f"在 editable 白名单内,允许修改: {norm}"
        if kind == "protected":
            return False, f"受保护路径(数据/评测/版本控制),禁止修改: {norm}"
        return False, f"不在 editable 白名单内,禁止修改: {norm}"

    @property
    def editable(self) -> Tuple[str, ...]:
        """当前可修改路径清单（只读）。"""
        return tuple(sorted(self._editable))