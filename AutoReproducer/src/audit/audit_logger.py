"""审计日志模块 - 记录所有 Agent 的决策与执行过程（可审计追踪）。

对齐方案「创新点四：可审计的完整实验追踪」：
- audit 日志：所有 Agent 输入输出 / 决策依据以 JSON Lines 写入 data/logs/；
- experiment_ledger：面向研究过程的完整实验账本（含每次尝试与结果）写入
  data/experiment_ledger/，支持按 session_id 时间轴回放。

P1-⑫ 用量计量：
- plan 级计量：begin_plan/end_plan 界定一个计划（如复现流水线阶段），
  record_llm_usage/record_sandbox_exec 把 LLM token 与容器耗时归入当前 plan，
  互斥 plan 各自成账，便于按阶段核算资源消耗；
- 无 plan 上下文时归入 "unattributed" 桶（与 ScholarAgent usage.py 语义一致）。
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 项目根目录（src/audit/audit_logger.py -> parents[2] 为仓库内 AutoReproducer 包根）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AuditLogger:
    """审计日志记录器：追踪每一步的决策依据，并落盘为可回放的账本。

    兼做用量计量器（P1-⑫）：以 plan 为键累计 LLM token / LLM 调用耗时 /
    容器执行次数与耗时，供审计统计与资源核算。
    """

    def __init__(self, log_dir: Optional[str] = None,
                 ledger_dir: Optional[str] = None):
        # 默认落在项目根 data/ 下，避免随 CWD 漂移
        base = _PROJECT_ROOT / "data"
        self.log_dir = Path(log_dir) if log_dir else (base / "logs")
        self.ledger_dir = Path(ledger_dir) if ledger_dir else (base / "experiment_ledger")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entries: List[Dict[str, Any]] = []
        self.llm_calls = 0
        self._start_time = time.time()
        # ---- P1-⑫ 用量计量 ----
        # plan 上下文栈（begin_plan 入栈，end_plan 出栈，支持嵌套）
        self._plan_stack: List[str] = []
        # plan_id -> {"calls","prompt_tokens","completion_tokens",
        #             "total_tokens","llm_seconds","exec_calls","exec_seconds",
        #             "models": {model: calls}}
        self._plan_usage: Dict[str, Dict[str, Any]] = {}
        # LLMClient usage_hook 入口（等同 record_llm_usage，便于直接绑定）
        self.usage_hook = self.record_llm_usage

    def log(self, agent: str, action: str, status: str, detail: str,
            data: Optional[dict] = None) -> dict:
        """记录一条审计日志。"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(time.time() - self._start_time, 2),
            "agent": agent,
            "action": action,
            "status": status,
            "detail": detail,
            "data": data or {},
        }
        self.entries.append(entry)
        self._flush(entry)
        return entry

    def log_experiment(self, phase: str, decision: str,
                       inputs: Optional[dict] = None,
                       outputs: Optional[dict] = None,
                       result: Optional[dict] = None) -> dict:
        """写入一条实验账本记录（Ledger）：可覆盖 Agent 输入输出与尝试结果。"""
        record = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "decision": decision,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "result": result or {},
        }
        ledger_file = self.ledger_dir / f"ledger_{self.session_id}.jsonl"
        with open(ledger_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def replay(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """按时间轴回放某次实验的完整账本（默认当前会话）。

        返回按写入顺序排列的 ledger 记录，供「任意时间点状态回放与审计」。
        """
        sid = session_id or self.session_id
        ledger_file = self.ledger_dir / f"ledger_{sid}.jsonl"
        records: List[Dict[str, Any]] = []
        if not ledger_file.exists():
            return records
        for line in ledger_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def add_llm_calls(self, count: int) -> None:
        """累计 LLM 调用次数（预算统计口径）。"""
        self.llm_calls += max(0, int(count))

    # ---------------- P1-⑫ 用量计量（plan 级） ----------------

    @property
    def current_plan(self) -> str:
        """当前生效的 plan id（无 plan 上下文时为空串）。"""
        return self._plan_stack[-1] if self._plan_stack else ""

    def begin_plan(self, plan_id: str) -> None:
        """进入一个 plan（入栈并初始化该 plan 的用量桶）。"""
        plan_id = plan_id or "unattributed"
        self._plan_stack.append(plan_id)
        self._plan_usage.setdefault(plan_id, {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_seconds": 0.0,
            "exec_calls": 0,
            "exec_seconds": 0.0,
            "models": {},
        })

    def end_plan(self, plan_id: Optional[str] = None) -> Dict[str, Any]:
        """结束当前 plan（可指定校验），返回该 plan 的用量快照副本。

        栈为空或 plan_id 与栈顶不符时返回空 dict（不抛异常，计量非关键路径）。
        """
        if not self._plan_stack:
            return {}
        top = self._plan_stack[-1]
        pid = plan_id or top
        if pid != top:
            return {}
        self._plan_stack.pop()
        return self.plan_snapshot(pid)

    def plan_snapshot(self, plan_id: str) -> Dict[str, Any]:
        """返回指定 plan 的用量快照（不含嵌套子 plan 归并）。"""
        bucket = self._plan_usage.get(plan_id or "unattributed") or {}
        return {
            "plan_id": plan_id or "unattributed",
            "calls": bucket.get("calls", 0),
            "prompt_tokens": bucket.get("prompt_tokens", 0),
            "completion_tokens": bucket.get("completion_tokens", 0),
            "total_tokens": bucket.get("total_tokens", 0),
            "llm_seconds": round(bucket.get("llm_seconds", 0.0), 2),
            "exec_calls": bucket.get("exec_calls", 0),
            "exec_seconds": round(bucket.get("exec_seconds", 0.0), 2),
            "models": dict(bucket.get("models", {})),
        }

    def plans_usage(self) -> Dict[str, Dict[str, Any]]:
        """返回全部 plan 的用量快照（plan_id -> snapshot）。"""
        return {pid: self.plan_snapshot(pid) for pid in self._plan_usage}

    def record_llm_usage(self, model: str = "", prompt_tokens: int = 0,
                         completion_tokens: int = 0,
                         duration_seconds: float = 0.0, calls: int = 1) -> None:
        """把一次 LLM 调用用量归入当前 plan（LLMClient.usage_hook 入口）。

        无 plan 上下文时归入 "unattributed" 桶；用法与 ScholarAgent
        usage.record_llm 对齐（calls/prompt/completion/total/llm_seconds/model）。
        """
        plan_id = self.current_plan or "unattributed"
        bucket = self._plan_usage.setdefault(plan_id, {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_seconds": 0.0,
            "exec_calls": 0,
            "exec_seconds": 0.0,
            "models": {},
        })
        p = max(0, int(prompt_tokens or 0))
        c = max(0, int(completion_tokens or 0))
        bucket["calls"] += max(0, int(calls or 0))
        bucket["prompt_tokens"] += p
        bucket["completion_tokens"] += c
        bucket["total_tokens"] += p + c
        bucket["llm_seconds"] += max(0.0, float(duration_seconds or 0.0))
        bucket["models"][model or "unknown"] = \
            bucket["models"].get(model or "unknown", 0) + max(0, int(calls or 0))

    def record_sandbox_exec(self, duration_seconds: float = 0.0,
                            calls: int = 1) -> None:
        """把一次容器/沙箱执行的耗时归入当前 plan（容器耗时维度）。"""
        plan_id = self.current_plan or "unattributed"
        bucket = self._plan_usage.setdefault(plan_id, {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_seconds": 0.0,
            "exec_calls": 0,
            "exec_seconds": 0.0,
            "models": {},
        })
        bucket["exec_calls"] += max(0, int(calls or 0))
        bucket["exec_seconds"] += max(0.0, float(duration_seconds or 0.0))

    def _flush(self, entry: dict) -> None:
        """将单条日志追加写入 JSON Lines 文件。"""
        log_file = self.log_dir / f"session_{self.session_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_summary(self) -> List[Dict[str, Any]]:
        """获取当前会话的所有日志。"""
        return self.entries

    def get_stats(self) -> dict:
        """获取统计信息（含预算口径的 LLM 调用数与 plan 级用量计量）。"""
        total = len(self.entries)
        errors = sum(1 for e in self.entries if e["status"] == "ERROR")
        success = sum(1 for e in self.entries if e["status"] == "SUCCESS")
        stats = {
            "total_steps": total,
            "errors": errors,
            "success": success,
            "duration_sec": round(time.time() - self._start_time, 2),
            "llm_calls": self.llm_calls,
        }
        # P1-⑫ 用量计量：全 plan 合计 + 逐 plan 明细（对齐 ScholarAgent
        # usage.snapshot 语义，汇总覆盖 unattributed 桶）
        plans = self.plans_usage()
        stats["plans"] = plans
        llm_tot = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "total_tokens": 0, "llm_seconds": 0.0,
                   "exec_calls": 0, "exec_seconds": 0.0,
                   "models": {}}
        for snap in plans.values():
            for k in ("calls", "prompt_tokens", "completion_tokens",
                      "total_tokens"):
                llm_tot[k] += snap[k]
            llm_tot["llm_seconds"] += snap["llm_seconds"]
            llm_tot["exec_calls"] += snap["exec_calls"]
            llm_tot["exec_seconds"] += snap["exec_seconds"]
            for model, cnt in snap["models"].items():
                llm_tot["models"][model] = llm_tot["models"].get(model, 0) + cnt
        stats["usage"] = {
            "llm_calls": llm_tot["calls"],
            "prompt_tokens": llm_tot["prompt_tokens"],
            "completion_tokens": llm_tot["completion_tokens"],
            "total_tokens": llm_tot["total_tokens"],
            "llm_seconds": round(llm_tot["llm_seconds"], 2),
            "container_exec_calls": llm_tot["exec_calls"],
            "container_exec_seconds": round(llm_tot["exec_seconds"], 2),
            "models": llm_tot["models"],
        }
        return stats