"""TrialLedger - 优化试验账本（融合 ScholarAgent research/ledger.py 设计）。

职责：把每一轮优化 trial 的 Keep/Reject 决策完整、可审计地落盘，
供报告生成与经验库消费：

- 每条记录保留 candidate（方向/参数）/ evaluation（指标/得分/有效性）/
  kept 决策 / reason 理由 / restored（工作区是否已回滚）/ applied_files
  （Keep 补丁落盘文件）等结构化字段，与 ScholarAgent record_trial 对齐；
- JSONL 追加持久化，字段白名单 + 线程安全 + 损坏行跳过 + 上限截断
  （与 ExperienceStore 同套路），默认路径 data/experience/trials.jsonl；
- best() 按 repo 取最优 trial；kept()/rejected() 分类查询；
  validated_trials() 只返回"真实验证过"的 Keep 决策，可直接播种
  ExperienceStore 与 BeamUCT 先验（P1-⑥ 消费）。
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.experience.experience_store import DEFAULT_EXPERIENCE_PATH

# 账本默认位置：<experience_dir>/trials.jsonl
DEFAULT_LEDGER_PATH = DEFAULT_EXPERIENCE_PATH.parent / "trials.jsonl"
LEDGER_VERSION = "autorepro.ledger/v1"

MAX_LEDGER_RECORDS = int(os.environ.get("AUTOREPRO_LEDGER_MAX_RECORDS", "5000"))

# record_trial 落盘字段白名单
_TRIAL_KEYS = (
    "trial_id", "version", "repo", "candidate", "evaluation", "kept",
    "reason", "restored", "applied_files", "rejected_files", "timestamp",
)


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


def _as_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        return [value]
    return []


class TrialLedger:
    """JSONL 试验账本：record_trial / all / kept / rejected / best / summary。"""

    def __init__(self, path: Optional[str | Path] = None,
                 max_records: int = MAX_LEDGER_RECORDS) -> None:
        self.path = Path(path) if path else DEFAULT_LEDGER_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 写入

    def record_trial(self, *, repo: str,
                     candidate: Optional[Dict[str, Any]] = None,
                     evaluation: Optional[Dict[str, Any]] = None,
                     kept: bool,
                     reason: str = "",
                     restored: bool = False,
                     applied_files: Any = None,
                     rejected_files: Any = None,
                     trial_id: Optional[str] = None) -> Dict[str, Any]:
        """记录一次 trial 的 Keep/Reject 决策，返回落盘后的完整记录。

        candidate: {direction/arm, params, ...} 与 ScholarAgent 结构兼容；
        evaluation: {valid, score, metric, success, stdout_tail, ...}；
        kept=True 表示 Keep（真实验证的改进），否则 Reject。
        """
        record = {
            "trial_id": str(trial_id) if trial_id else uuid.uuid4().hex[:12],
            "version": LEDGER_VERSION,
            "repo": str(repo),
            "candidate": candidate if isinstance(candidate, dict) else {},
            "evaluation": evaluation if isinstance(evaluation, dict) else {},
            "kept": bool(kept),
            "reason": str(reason),
            "restored": bool(restored),
            "applied_files": _as_str_list(applied_files),
            "rejected_files": _as_str_list(rejected_files),
            "timestamp": _now_iso(),
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    record, ensure_ascii=False, sort_keys=True,
                    default=str) + "\n")
            self._truncate_if_needed()
        return record

    def _truncate_if_needed(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as handle:
                lines = handle.readlines()
            if len(lines) > self.max_records:
                with open(self.path, "w", encoding="utf-8") as handle:
                    handle.writelines(lines[-self.max_records:])
        except OSError:
            pass

    # ---------------------------------------------------------------- 读取

    def all(self, repo: Optional[str] = None) -> List[Dict[str, Any]]:
        """读取全部台账；repo 过滤可选。损坏行跳过。"""
        if not self.path.is_file():
            return []
        records: List[Dict[str, Any]] = []
        with open(self.path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if repo is not None and record.get("repo") != repo:
                    continue
                records.append(record)
        return records

    def count(self) -> int:
        try:
            with open(self.path, encoding="utf-8", errors="replace") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError:
            return 0

    def kept(self, repo: Optional[str] = None) -> List[Dict[str, Any]]:
        return [r for r in self.all(repo=repo) if r.get("kept")]

    def rejected(self, repo: Optional[str] = None) -> List[Dict[str, Any]]:
        return [r for r in self.all(repo=repo) if not r.get("kept")]

    def valid_keep_trials(self, repo: Optional[str] = None
                          ) -> List[Dict[str, Any]]:
        """真实验证过的 Keep 决策：evaluation.valid=True 且 kept=True。

        这是播撒经验库 / BeamUCT 先验的唯一合法来源（防止未经验证的
        猜测污染经验；与 ExperienceStore 的 validated 标记同口径）。
        """
        out = []
        for r in self.all(repo=repo):
            if not r.get("kept"):
                continue
            ev = r.get("evaluation", {}) if isinstance(
                r.get("evaluation"), dict) else {}
            if ev.get("valid") is True:
                out.append(r)
        return out

    def best(self, repo: str) -> Optional[Dict[str, Any]]:
        """repo 内 valid Keep 中 evaluation.score 最高的 trial。"""
        scored = [r for r in self.valid_keep_trials(repo=repo)
                  if isinstance(r.get("evaluation", {}).get("score"),
                                 (int, float))]
        if not scored:
            return None

        def _score(r: Dict[str, Any]) -> float:
            return float(r["evaluation"]["score"])

        return max(scored, key=_score)

    def summary(self, repo: Optional[str] = None) -> Dict[str, Any]:
        """汇总账本：总 trial 数 / Keep 数 / Reject 数 / valid Keep 数。"""
        records = self.all(repo=repo)
        kept = sum(1 for r in records if r.get("kept"))
        valid_keep = sum(
            1 for r in records
            if r.get("kept") and isinstance(r.get("evaluation"), dict)
            and r.get("evaluation", {}).get("valid") is True)
        return {
            "trials_total": len(records),
            "kept": kept,
            "rejected": len(records) - kept,
            "valid_keep": valid_keep,
        }

    # ---------------------------------------------------------------- 维护

    def clear(self) -> None:
        with self._lock:
            self.path.write_text("", encoding="utf-8")