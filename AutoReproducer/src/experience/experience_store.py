"""ExperienceStore - 经验库（融合 ScholarAgent optimizer/experience_store 设计）。

职责：把每一轮"真实验证过的"优化经历（方向 / 参数 / 指标 / 得分）追加到
JSONL 文件，供后续任务与 BeamUCT 先验播种复用：

- 只有 validated=True 的记录才算"经验"（真实评估而非猜测）；
- 按 repo 聚合，支持 summarize()（按方向统计 best/mean/count）与
  best()（按方向模式取最优记录）；
- 上限截断：超 MAX_RECORDS 只保留最新记录，历史可回收；
- 线程安全：append 持锁写入，多 Agent 并发追加不丢行；
- 路径约定：默认 data/experience/experience.jsonl（AUTOREPRO_DATA_ROOT 可覆盖），
  与环境变量 AUTOREPRO_DATA_ROOT 对齐三层存储的 L0 数据根。
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.resource_manager import DATA_ROOT

# 经验库默认位置：<data_root>/experience/experience.jsonl
EXPERIENCE_DIR = Path(os.environ.get(
    "AUTOREPRO_EXPERIENCE_DIR", str(DATA_ROOT / "experience")))
DEFAULT_EXPERIENCE_PATH = EXPERIENCE_DIR / "experience.jsonl"

# 单文件最大保留记录数（超出截断，保留最新）
MAX_RECORDS = int(os.environ.get("AUTOREPRO_EXPERIENCE_MAX_RECORDS", "10000"))

# 记录规范化字段集合（append 时按白名单落盘，防脏键污染）
_RECORD_KEYS = (
    "repo", "direction", "params", "metric", "score",
    "direction_mode", "validated", "context", "timestamp",
    "trial_id", "decision",
)


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


class ExperienceStore:
    """JSONL 经验库：append / all / best / summarize / clear。"""

    def __init__(self, path: Optional[str | Path] = None,
                 max_records: int = MAX_RECORDS) -> None:
        self.path = Path(path) if path else DEFAULT_EXPERIENCE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 写入

    def append(self, record: Dict[str, Any]) -> None:
        """追加一条经验记录（白名单字段规范化后落盘）。"""
        payload = {
            "repo": str(record.get("repo", "")),
            "direction": str(record.get("direction", "")),
            "params": record.get("params", {}) if isinstance(
                record.get("params"), dict) else {},
            "metric": str(record.get("metric", "")),
            "score": record.get("score"),
            "direction_mode": str(record.get("direction_mode", "maximize")),
            "validated": bool(record.get("validated", False)),
            "context": record.get("context", {}) if isinstance(
                record.get("context"), dict) else {},
            "timestamp": str(record.get("timestamp") or _now_iso()),
        }
        for extra in ("trial_id", "decision"):
            if record.get(extra) is not None:
                payload[extra] = record[extra]
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    payload, ensure_ascii=False, sort_keys=True,
                    default=str) + "\n")
            self._truncate_if_needed()

    def _truncate_if_needed(self) -> None:
        """行数超上限时保留最新 max_records 条（OSError 静默降级）。"""
        try:
            with open(self.path, encoding="utf-8") as handle:
                lines = handle.readlines()
            if len(lines) > self.max_records:
                with open(self.path, "w", encoding="utf-8") as handle:
                    handle.writelines(lines[-self.max_records:])
        except OSError:
            pass                            # 无法读取时不截断，下次再试

    # ---------------------------------------------------------------- 读取

    def all(self, repo: Optional[str] = None,
            validated_only: bool = True) -> List[Dict[str, Any]]:
        """读取记录；repo 过滤 + validated 过滤（默认只要真实验证过的）。"""
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
                    continue                # 半行写入（进程被杀）跳过
                if repo is not None and record.get("repo") != repo:
                    continue
                if validated_only and not record.get("validated"):
                    continue
                records.append(record)
        return records

    def count(self) -> int:
        try:
            with open(self.path, encoding="utf-8", errors="replace") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError:
            return 0

    def best(self, repo: str,
             direction_mode: str = "maximize") -> Optional[Dict[str, Any]]:
        """返回 repo 的 validated 最优记录（无得分记录时返回 None）。"""
        scored = [r for r in self.all(repo=repo)
                  if isinstance(r.get("score"), (int, float))]
        if not scored:
            return None
        if direction_mode == "minimize":
            return min(scored, key=lambda r: float(r["score"]))
        return max(scored, key=lambda r: float(r["score"]))

    def summarize(self, repo: Optional[str] = None) -> Dict[str, Any]:
        """按 direction 聚合 validated 记录：count / best / mean。"""
        records = self.all(repo=repo)
        by_direction: Dict[str, Dict[str, Any]] = {}
        for record in records:
            name = str(record.get("direction", ""))
            entry = by_direction.setdefault(
                name, {"count": 0, "scores": []})
            entry["count"] += 1
            if isinstance(record.get("score"), (int, float)):
                entry["scores"].append(float(record["score"]))
        return {
            "record_count": len(records),
            "directions": {
                name: {
                    "count": e["count"],
                    "best": max(e["scores"]) if e["scores"] else None,
                    "mean": (sum(e["scores"]) / len(e["scores"])
                             if e["scores"] else None),
                }
                for name, e in sorted(by_direction.items())
            },
        }

    # ---------------------------------------------------------------- 维护

    def clear(self) -> None:
        with self._lock:
            self.path.write_text("", encoding="utf-8")