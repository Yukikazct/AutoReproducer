"""Recover abandoned running records only after their OS lock is released."""
import json
from pathlib import Path

from filelock import FileLock, Timeout

from src.method_adapters import write_json


def persist_interrupted_run(run_dir, reason, *, data=None, optimization=None):
    """Make the final run record and saved baseline report visibly incomplete."""
    run_dir = Path(run_dir)
    if data is None:
        result_path = run_dir / "result.json"
        data = json.loads(result_path.read_text(encoding="utf-8")).get("data", {}) if result_path.is_file() else {}
        validation_path = run_dir / "validation.json"
        if "validation" not in data and validation_path.is_file():
            data["validation"] = json.loads(validation_path.read_text(encoding="utf-8"))
    data.update(run_status="interrupted", run_dir=str(run_dir.resolve()), report_path=str((run_dir / "report.md").resolve()))
    if optimization is not None:
        data["optimization"] = optimization
    report_path = run_dir / "report.md"
    previous = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    banner = "> **实验已中断（interrupted）**"
    if not previous.startswith(banner):
        explanation = "已有记录已保留，本次运行未完成。"
        if optimization is not None or (run_dir / "optimization.json").is_file():
            explanation += "优化进度见 [optimization.json](optimization.json)，基线完成不代表优化确认完成。"
        if previous:
            explanation += "下方为中断前保存的报告。"
        previous = f"{banner}：{reason}。\n> {explanation}\n\n" + previous
    data["report"] = previous
    report_path.write_text(previous, encoding="utf-8")
    write_json(run_dir / "result.json", {"state": "INTERRUPTED", "data": data, "error": reason})


def recover_interrupted_optimizations(root):
    """Return recovered paths; active studies and completed results are untouched.

    A run owns .optimization.lock for the full study. The OS releases the lock
    after force termination; no PID reuse or elapsed-time guess is involved.
    Existing artifacts remain evidence of their individual completed stages.
    """
    recovered = []
    for path in Path(root).glob("repository_*/optimization.json"):
        try:
            with FileLock(str(path.parent / ".optimization.lock"), timeout=0):
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("status") != "running":
                    continue
                reason = "实验所有者已退出，恢复未完成的运行记录；留出确认未完成，不宣称已验证提升"
                for record_path in path.parent.glob("trials/*/*.json"):
                    if record_path.name not in {"trial.json", "holdout.json"}:
                        continue
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    if record.get("status") == "running":
                        record.update(status="interrupted", error=reason)
                        write_json(record_path, record)
                result.update(status="interrupted", optimized=False, reason=reason, recovered=True)
                for name in ("trials", "confirmation_training", "confirmation"):
                    for record in result.get(name, []):
                        if record.get("status") == "running":
                            record["status"] = "interrupted"
                persist_interrupted_run(path.parent, reason, optimization=result)
                write_json(path, result)
                recovered.append(str(path))
        except (Timeout, OSError, ValueError):
            # An active run, unreadable file or legacy partial JSON is not proof
            # of successful completion and must not be overwritten by inference.
            continue
    return recovered
