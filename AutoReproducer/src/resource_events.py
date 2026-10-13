"""可重建的资源操作事件账本。

事件以 JSONL 保存，原始磁盘状态仍是最终事实；该模块只负责让下载/安装
过程可观测，并且不写入凭据或完整环境变量。
"""
import json
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from src.project_paths import get_data_root


def _data_root() -> Path:
    return get_data_root()


def read_resource_events(path, limit=500) -> list:
    """Read a bounded tail without loading the entire event ledger into memory."""
    if limit <= 0:
        return []
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as stream:
            lines = deque(stream, maxlen=limit)
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


class ResourceEventLogger:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else _data_root() / "resource_events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, resource_type: str, resource_id: str, operation: str,
             state: str, **details: Any) -> Dict[str, Any]:
        event = {
            "event_id": uuid.uuid4().hex,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "monotonic": time.monotonic(),
            "resource_type": resource_type,
            "resource_id": resource_id,
            "operation": operation,
            "state": state,
            **{k: v for k, v in details.items() if v is not None},
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def list(self, limit: int = 500) -> list:
        return read_resource_events(self.path, limit)
