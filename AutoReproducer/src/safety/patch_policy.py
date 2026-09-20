"""PatchPolicy - 补丁安全策略（安全网，融合 ScholarAgent patch_policy 完备规则）。

任何 LLM 产物（优化补丁、自动改写）在修改工作区文件前，必须先通过本策略判定。
在既有 editable 白名单 + protected 兜底规则之上，融合 ScholarAgent 的
完备防作弊规则（backend/app/research/patch_policy.py，MIT）：

- 目录段禁改模式：eval / test / data / benchmark / gold / label / spec /
  verifier 等任意层目录段命中即拒绝 —— 防止 LLM 补丁改写评测、数据集、
  基准真相或验证器；
- 数据与权重扩展名：.csv / .jsonl / .parquet / .pkl / .pt / .safetensors /
  .bin 等一律只读；
- 文件名模式：test_*.py / *_eval.py / dataset_utils.py 等文件名即拒绝；
- 单文件体积预算：默认 96KB 以上文件不进入补丁范围（防大文件整体改写）；
- 结构化决策：decide() 返回 PatchDecision（path/allowed/rule/reason），
  filter_patches() 支持批量 LLM 补丁切片（去重 + 每候选 ≤3 文件），
  便于 TrialLedger 记录与报告消费。

强制层序（禁止规则无条件优先于 editable 白名单）：
    非法路径 → protected(文件/第一段) → 目录段模式 → 文件名模式 →
    扩展名 → 体积预算 → editable 白名单

典型用法::

    policy = PatchPolicy(editable=["src/model.py", "train.py"])
    ok, reason = policy.check("src/model.py")     # (True, 允许说明)
    ok, reason = policy.check("data/train.csv")   # (False, 受保护说明)
    ok, reason = policy.check("test_model.py")    # (False, 评测文件名)
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, List, Optional, Tuple

# ---------------------------------------------------------------- 常量

# 默认受保护目录前缀（命中顶层即拒绝修改，保留原语义）
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

# 评测/测试/数据基础设施目录段（任意层级命中即拒绝，来自 ScholarAgent）
DENIED_SEGMENT_PATTERNS: Tuple[str, ...] = (
    "eval",
    "evals",
    "evaluation",
    "evaluator",
    "evaluators",
    "grader",
    "grading",
    "judge",
    "verifier",
    "checker",
    "benchmark",
    "benchmarks",
    "test",
    "tests",
    "testing",
    "spec",
    "specs",
    "dataset",
    "datasets",
    "data",
    "gold",
    "ground_truth",
    "groundtruth",
    "label",
    "labels",
    "annotation",
    "annotations",
)

# 数据/权重/二进制扩展名：一律只读（来自 ScholarAgent）
DENIED_EXTENSIONS: Tuple[str, ...] = (
    ".csv",
    ".tsv",
    ".jsonl",
    ".parquet",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".arrow",
    ".h5",
    ".hdf5",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
)

# 文件名即评测/测试/数据基础设施（前缀或后缀）
DENIED_STEM_RE = re.compile(
    r"^(test|tests|eval|evals|evaluat(e|ion|or)|dataset|benchmark|"
    r"grading|verif(y|ier)|gold|ground[_-]?truth|label[s]?)[_.-]",
    re.IGNORECASE,
)
DENIED_STEM_SUFFIX_RE = re.compile(
    r"[_.-](test|tests|eval|evals|evaluator|evaluation|dataset|benchmark)$",
    re.IGNORECASE,
)

# 单文件体积预算（超过即拒绝入补丁）
MAX_PATCHABLE_FILE_BYTES: int = 96 * 1024

# 单候选补丁最多允许修改的文件数（拒绝列表不在此列）
MAX_PATCH_FILES_PER_CANDIDATE: int = 3


# ---------------------------------------------------------------- 决策值

@dataclass(frozen=True)
class PatchDecision:
    """结构化补丁决策：path / allowed / rule(命中的规则名) / reason。"""

    path: str
    allowed: bool
    rule: str
    reason: str

    def to_dict(self) -> dict:
        return {"path": self.path, "allowed": self.allowed,
                "rule": self.rule, "reason": self.reason}


# ---------------------------------------------------------------- 策略主体

class PatchPolicy:
    """补丁白名单策略：只放行 editable 白名单内且未被强制层拦截的路径。"""

    def __init__(self, editable: Optional[Iterable[str]] = None,
                 protected_prefixes: Optional[Iterable[str]] = None,
                 protected_files: Optional[Iterable[str]] = None,
                 protected_suffixes: Optional[Iterable[str]] = None,
                 workspace: Optional[str] = None,
                 max_patchable_file_bytes: int = MAX_PATCHABLE_FILE_BYTES) -> None:
        self._editable = {self._normalize(p) for p in (editable or [])
                          if self._normalize(p)}
        self._protected_prefixes = tuple(
            protected_prefixes or DEFAULT_PROTECTED_PREFIXES)
        self._protected_files = tuple(protected_files or DEFAULT_PROTECTED_FILES)
        self._protected_suffixes = tuple(
            protected_suffixes or DEFAULT_PROTECTED_SUFFIXES)
        # 可选工作区根：提供后启用单文件体积预算校验（相对路径解析到真实文件）
        self._workspace = workspace
        self.max_patchable_file_bytes = int(max_patchable_file_bytes)

    # ---------------- 基础工具 ----------------

    @staticmethod
    def _normalize(path: object) -> str:
        """归一化为相对 POSIX 路径；非法路径(绝对/空/穿越)返回空字符串。"""
        p = str(path).replace("\\", "/").strip()
        if not p or p.startswith("//") or p[:1] == "/":
            return ""          # 空路径或 POSIX 绝对路径
        if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
            return ""          # Windows 盘符绝对路径(C:/...)
        while p.startswith("./"):     # 只剥 ./ 前缀，保留 .git 等点开头目录
            p = p[2:]
        if not p or p.startswith("/"):
            return ""
        if ".." in p.split("/"):      # 任何层级的 .. 穿越都拒绝
            return ""
        return p

    def _with_workspace(self, workspace: Optional[str]) -> "PatchPolicy":
        """返回绑定工作区根的同策略副本（体积预算校验启用）。"""
        return PatchPolicy(
            editable=self._editable,
            protected_prefixes=self._protected_prefixes,
            protected_files=self._protected_files,
            protected_suffixes=self._protected_suffixes,
            workspace=workspace,
            max_patchable_file_bytes=self.max_patchable_file_bytes,
        )

    # ---------------- 判定 ----------------

    def classify(self, path: object) -> str:
        """返回路径类别：editable / protected / unknown（兼容旧接口）。"""
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

    def decide(self, path: object,
               workspace: Optional[str] = None) -> PatchDecision:
        """结构化判定补丁能否修改 path（本策略的完整决策入口）。"""
        norm = self._normalize(path)
        if not norm:
            return PatchDecision(str(path), False, "illegal",
                                 f"非法路径(空/绝对/穿越),拒绝修改: {path}")

        # 1) protected 文件（精确/尾段/glob）
        if any(_glob_match([f], norm) or norm.endswith("/" + f)
               for f in self._protected_files):
            return PatchDecision(norm, False, "protected",
                                 f"受保护文件,禁止修改: {norm}")

        # 1b) 密钥/凭据类后缀（.pem/.key/.p12/... 一律只读，不因在白名单而放行）
        if any(norm.lower().endswith(s) for s in self._protected_suffixes):
            return PatchDecision(norm, False, "protected",
                                 f"密钥/凭据类文件,禁止修改: {norm}")

        rel = PurePosixPath(norm)
        parts = [p.lower() for p in rel.parts]

        # 2) protected 第一段（原语义）
        if any(part in self._protected_prefixes for part in parts[:1]):
            return PatchDecision(norm, False, "protected",
                                 f"受保护路径(数据/评测/版本控制),禁止修改: {norm}")

        # 3) 任意目录段（除最后一段）命中评测/数据基础设施
        for seg in parts[:-1]:
            if seg in DENIED_SEGMENT_PATTERNS:
                return PatchDecision(norm, False, "denied_segment",
                                     f"目录 '{seg}' 为评测/测试/数据基础设施,禁止修改: {norm}")

        # 4) 文件名段本身是评测词根
        if parts[-1] in DENIED_SEGMENT_PATTERNS:
            return PatchDecision(norm, False, "denied_filename",
                                 f"文件名属于评测/测试/数据基础设施,禁止修改: {norm}")

        # 5) 文件名模式（test_*.py / *_eval.py / dataset_utils.py 等）
        stem = rel.stem
        if DENIED_STEM_RE.match(stem) or DENIED_STEM_SUFFIX_RE.search(stem):
            return PatchDecision(norm, False, "denied_filename",
                                 f"文件名标识评测/测试/数据基础设施,禁止修改: {norm}")

        # 6) 数据/权重/二进制扩展名
        if rel.name.lower().endswith(DENIED_EXTENSIONS):
            return PatchDecision(norm, False, "denied_extension",
                                 f"数据/权重/二进制文件只读,禁止修改: {norm}")

        # 7) 单文件体积预算（workspace 可解析出真实文件时生效）
        root = self._workspace or workspace
        if root is not None:
            try:
                from pathlib import Path
                target = (Path(root) / norm).resolve()
                target.relative_to(Path(root).resolve())
                if target.is_file() and \
                        target.stat().st_size > self.max_patchable_file_bytes:
                    return PatchDecision(
                        norm, False, "size_budget",
                        f"文件超过补丁体积预算({self.max_patchable_file_bytes} B),"
                        f"禁止修改: {norm}")
            except (ValueError, OSError):
                pass          # 无法解析真实路径时跳过体积校验

        # 8) editable 白名单
        if self._editable:
            if _glob_match(self._editable, norm):
                return PatchDecision(norm, True, "editable",
                                     f"在 editable 白名单内,允许修改: {norm}")
            return PatchDecision(norm, False, "unknown",
                                 f"不在 editable 白名单内,禁止修改: {norm}")

        # 9) 无白名单：默认允许（仍受上面全部强制层约束）
        return PatchDecision(norm, True, "default_allow",
                             f"无白名单限制,默认允许修改: {norm}")

    def check(self, path: object,
              workspace: Optional[str] = None) -> Tuple[bool, str]:
        """判定补丁能否修改 path，返回 (是否允许, 说明)（兼容接口）。"""
        d = self.decide(path, workspace=workspace)
        if d.allowed:
            return True, d.reason
        return False, d.reason

    def filter_patches(self,
                       patches: Iterable[dict],
                       workspace: Optional[str] = None,
                       ) -> Tuple[List[dict], List[PatchDecision]]:
        """批量 LLM 补丁切片：返回 (放行的补丁列表, 拒绝的决策列表)。

        - 每个 path 只允许一次（重复丢弃）；
        - 同一候选最多放行 MAX_PATCH_FILES_PER_CANDIDATE 个文件；
        - 拒绝项携带结构化 PatchDecision（rule/reason），供账本记录。
        """
        allowed: List[dict] = []
        rejected: List[PatchDecision] = []
        seen: set = set()
        for patch in patches or []:
            path = str(patch.get("path", "")).strip()
            decision = self.decide(path, workspace=workspace)
            if not decision.allowed:
                rejected.append(decision)
                continue
            if decision.path in seen:
                continue
            seen.add(decision.path)
            enriched = dict(patch)
            enriched["path"] = decision.path
            allowed.append(enriched)
        allowed = allowed[:MAX_PATCH_FILES_PER_CANDIDATE]
        return allowed, rejected

    # ---------------- 只读属性 ----------------

    @property
    def editable(self) -> Tuple[str, ...]:
        """当前可修改路径清单（只读）。"""
        return tuple(sorted(self._editable))


def _glob_match(patterns: Iterable[str], path: str) -> bool:
    """glob 匹配：支持 fnmatch 通配或「目录前缀 + /」两种语义。"""
    for pattern in patterns or []:
        if fnmatch.fnmatch(path, pattern) or path.startswith(pattern + "/"):
            return True
    return False