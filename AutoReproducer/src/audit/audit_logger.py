"""审计日志模块 - 记录所有Agent的决策和执行过程"""
import json
import time
from datetime import datetime
from pathlib import Path


class AuditLogger:
    """审计日志记录器，追踪每一步的决策依据"""

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entries = []
        self._start_time = time.time()

    def log(self, agent: str, action: str, status: str, detail: str, data: dict = None):
        """记录一条审计日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(time.time() - self._start_time, 2),
            "agent": agent,
            "action": action,
            "status": status,
            "detail": detail,
            "data": data or {}
        }
        self.entries.append(entry)
        self._flush(entry)
        return entry

    def _flush(self, entry: dict):
        """将单条日志写入文件"""
        log_file = self.log_dir / f"session_{self.session_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_summary(self) -> list:
        """获取当前会话的所有日志"""
        return self.entries

    def get_stats(self) -> dict:
        """获取统计信息"""
        total = len(self.entries)
        errors = sum(1 for e in self.entries if e["status"] == "ERROR")
        success = sum(1 for e in self.entries if e["status"] == "SUCCESS")
        return {
            "total_steps": total,
            "errors": errors,
            "success": success,
            "duration_sec": round(time.time() - self._start_time, 2)
        }
