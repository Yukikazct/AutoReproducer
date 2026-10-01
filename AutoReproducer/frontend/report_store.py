"""Reports remain in the current session until the user explicitly saves them."""
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from frontend import history_manager

TEMP_REPORT_TTL = 3600


def save_report(result):
    """Save this report and its execution attachment once; return its path."""
    existing = result.get("report_path") or ""
    if existing and Path(existing).is_file():
        return existing
    data = result.get("data") or {}
    report = data.get("report") or ""
    if not report:
        raise ValueError("当前没有可保存的报告")
    title = ((data.get("paper_info") or {}).get("title") or data.get("paper_title") or "report")[:30]
    title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_") or "report"
    sid = result.get("session_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not isinstance(sid, str) or not re.fullmatch(r"\d{8}_\d{6}", sid):
        raise ValueError("报告会话编号无效")
    directory = history_manager.get_project_data_dir() / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{title}_{sid}.md"
    attachment = directory / f"{title}_{sid}_execution.txt"
    execution = data.get("execution") or {}
    final = execution.get("final") or {}
    lines = ["# 完整执行输出（未被截断）", "", "## 代码", "```python",
             execution.get("code") or "（无）", "```", "", "## 标准输出", "```",
             final.get("stdout") or "（无输出）", "```", "", "## 错误输出", "```",
             final.get("stderr") or "（无）", "```", ""]
    saved = report + f"\n\n---\n\n> 完整执行输出附件: `{attachment.name}`（同一目录，未截断）\n"
    try:
        attachment.write_text("\n".join(lines), encoding="utf-8")
        target.write_text(saved, encoding="utf-8")
    except OSError:
        # A failed save must not leave a new orphan attachment.
        if not target.exists():
            attachment.unlink(missing_ok=True)
        raise
    result.update(report_path=str(target), report_saved=True)
    data["report"] = saved
    return str(target)


def delete_finished_progress(path, expected_mtime=None):
    """Destroy the temporary report copy after consumption or expiration."""
    if not path:
        return False
    from frontend.backend_pipeline import ProgressStore
    target = Path(path)
    try:
        if expected_mtime is not None and target.stat().st_mtime_ns != expected_mtime:
            return False  # Never destroy a newer run that reused this path.
        view = ProgressStore.read_snapshot(str(target))
        if view["running"]:
            return False
        target.unlink()
        return True
    except OSError:
        return False


def schedule_progress_cleanup(path, ttl_seconds=TEMP_REPORT_TTL):
    """Abandoned browser sessions also expire without another page visit."""
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError:
        return
    timer = threading.Timer(ttl_seconds, delete_finished_progress, args=(path, mtime))
    timer.daemon = True
    timer.start()


def cleanup_abandoned_progress(ttl_seconds=TEMP_REPORT_TTL):
    """Reclaim expired terminal copies after a server restart."""
    directory = history_manager.get_project_data_dir() / "runtime"
    removed = 0
    for path in directory.glob("progress_*.jsonl"):
        try:
            if path.stat().st_mtime < time.time() - ttl_seconds:
                removed += int(delete_finished_progress(str(path)))
        except OSError:
            continue
    return removed


def discard_unsaved_report(result, progress_path=""):
    """Reset/replacement drops unsaved text; saved reports remain on disk."""
    if result and not result.get("report_path"):
        (result.get("data") or {}).pop("report", None)
    delete_finished_progress(progress_path)
