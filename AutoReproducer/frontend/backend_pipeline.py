"""后台复现流水线执行器 + 进度事件存储（无 Streamlit 依赖，便于单元测试）。

用户点击前端"开始复现"后，Streamlit 主线程启动后台线程执行
run_pipeline_core()，把每个阶段的进度事件实时追加写入进度文件
(JSON Lines)。前端通过 ProgressStore.read_snapshot() 轮询聚合，
实现"后台运行 + 用户实时看到当前进度"：

事件类型：
  state    -> {"type": "state", "state": <FSM状态>, "agent": <显示名>,
               "phase_id": <本轮阶段ID>, "status": <执行状态>}
  pipeline_plan -> {"type": "pipeline_plan", "stages": <有序阶段列表>}
  log      -> {"type": "log", "log": <审计日志条目>}
  done     -> {"type": "done", "result": <完整流水线结果>}
  error    -> {"type": "error", "error": <后台线程捕获的异常消息>}

read_snapshot() 聚合以上事件为前端可直接渲染的视图：
  {state, agent_status, pipeline_stages, logs, result, done, error, running, updated_at}
"""
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.llm.llm_client import LLMClient
from src.audit.audit_logger import AuditLogger
from src.project_paths import get_data_root


# Contract checked by the web loader before starting a new background thread.
BACKEND_API_VERSION = 11
WORKER_CANCEL_GRACE_SECONDS = 5


def _force_stop_contains_worker_children():
    # Windows jobs contain nested sessions. A POSIX process group does not
    # contain inner runners that deliberately start their own session.
    return os.name == "nt"


def __getattr__(name):
    # Recovery must remain importable when the host is missing app dependencies
    # such as filelock. Preserve the exported class for subclassing and callers
    # that explicitly request it, while deferring the full agent graph import.
    if name != "Orchestrator":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from src.orchestrator import Orchestrator as implementation
    globals()[name] = implementation
    return implementation


def _create_orchestrator(*args, **kwargs):
    implementation = globals().get("Orchestrator") or __getattr__("Orchestrator")
    return implementation(*args, **kwargs)

# ---------------- 流水线 Agent 定义（与前端卡片一致） ----------------

AGENTS = [
    ("READ_PAPER", "📖 PaperReader", "reader"),
    ("FIND_RESOURCES", "🔍 ResourceFinder", "finder"),
    ("BUILD_ENV", "🔧 EnvBuilder", "builder"),
    ("EXECUTE_CODE", "⚡ CodeExecutor", "executor"),
]

OPTIMIZER_NAME = "🧪 Optimizer"
REPORTER_NAME = "📝 ReportGenerator"
VERIFIER_NAME = "🛡️ Verifier"


# ---------------- 进度事件存储 ----------------

class ProgressStore:
    """将流水线进度事件写入 JSON Lines 文件，供前端轮询读取。"""

    def __init__(self, path: str, *, reset: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 清空旧内容（每次复现从零开始）
        if reset:
            self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()
        self._start_ts = time.time()

    def emit(self, event: Dict[str, Any]) -> None:
        """追加一条进度事件（线程安全）。"""
        record = {
            "ts": round(time.time() - self._start_ts, 2),
            "at": datetime.now().isoformat(),
            **event,
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def read_snapshot(path: str, *, include_pending_result: bool = False) -> Dict[str, Any]:
        from frontend.progress_state import read_progress_snapshot
        return read_progress_snapshot(path, include_pending_result=include_pending_result)


def _write_report_copy(path: Path, raw: bytes) -> None:
    from src.safety.paths import filesystem_path
    path = Path(filesystem_path(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(raw)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sync_run_report(result: dict) -> None:
    """Publish the final web report and its portable files beside the run evidence."""
    data = result["data"]
    if not data.get("run_dir"):
        return
    from src.execution_artifacts import verified_report_image_bytes
    from src.report_revisions import sanitize_report_result
    from src.safety.paths import workspace_path

    run_dir = Path(data["run_dir"])
    report = Path(result["report_path"])
    from src.source_repair_evidence import report_evidence_bytes
    repair_evidence = report_evidence_bytes(report.read_text(encoding='utf-8'), report)
    assets = report.with_name(report.stem + "_assets")
    if assets.exists():
        for image in assets.iterdir():
            relative = assets.name + "/" + image.name
            raw = repair_evidence.get(relative)
            if raw is None:
                raw = verified_report_image_bytes(relative, report)
            if raw is None:
                raise ValueError("报告图片已改变或无法验证: " + image.name)
            target = workspace_path(run_dir, relative, "run report image")
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_report_copy(target, raw)
    attachment = report.with_name(report.stem + "_execution.txt")
    _write_report_copy(workspace_path(run_dir, attachment.name, "run output attachment"),
                       attachment.read_bytes())
    _write_report_copy(workspace_path(run_dir, "report.md", "run report"), report.read_bytes())
    _write_report_copy(workspace_path(run_dir, "result.json", "run result"),
                       json.dumps(sanitize_report_result(result), ensure_ascii=False, indent=2).encode("utf-8"))


def _record_pipeline_finish(result, *, logger=None):
    """Append the final lifecycle outcome, retaining earlier experiment records."""
    if logger is None:
        session_id = result.get("session_id") or ""
        if not re.fullmatch(r"\d{8}_\d{6}", session_id):
            return
        logger = AuditLogger()
        logger.session_id = session_id
    data = result.get("data") or {}
    stats = result.get("audit_stats") or {}
    logger.log_experiment(
        "FINISH", "流水线与运行环境收尾完成",
        inputs={"paper_title": data.get("paper_title", "")},
        outputs={"title": (data.get("paper_info") or {}).get("title") or data.get("paper_title", "")},
        result={"state": result.get("state"), "error": result.get("error"),
                "environment_cleanup": (data.get("environment_cleanup") or {}).get("status"),
                "duration_sec": stats.get("duration_sec", 0), "llm_calls": stats.get("llm_calls", 0)})


# ---------------- 后台流水线执行 ----------------

def _emit_state(store: ProgressStore, state: str, agent: str,
                status: str) -> None:
    store.emit({"type": "state", "state": state,
                "agent": agent, "status": status})


def _reject_pdf_input(store: ProgressStore, reason: str) -> Dict[str, Any]:
    """Reject unreadable evidence before resource search or code generation."""
    store.emit({"type": "pipeline_plan", "stages": [
        {"id": "read_paper", "agent": "📖 PaperReader", "title": "论文解析", "status": "waiting"}]})
    store.emit({"type": "state", "state": "READ_PAPER", "agent": "📖 PaperReader",
                "phase_id": "read_paper", "status": "error", "reason": reason})
    result = {"state": "ERROR", "error": reason,
              "data": {"pdf_input": {"status": "rejected", "reason": reason}},
              "audit_logs": [], "audit_stats": {}, "report_path": "", "session_id": ""}
    store.emit({"type": "done", "result": result})
    return result


def run_pipeline_core(progress_path: str,
                      paper_title: str = "", pdf_path: str = "",
                      corpus_paper: Optional[str] = None,
                      model_name: str = "", base_url: str = "",
                      api_key: str = "", mock_mode: bool = False,
                      max_trials: int = 10,
                      use_docker: bool = False,
                      workspace_dir: Optional[str] = None,
                      experiment_profile: Optional[str] = None,
                      use_llm_review: bool = False,
                      allow_result_summary_review: bool = False,
                      enable_optimization: bool = False,
                      prepare_environment: bool = False,
                      optimization_mode: str = "off",
                      max_candidates: int = 3,
                      budget_seconds: int = 7200,
                      prepare_only: bool = False,
                      offline: bool = False,
                      title_resolution: Optional[dict] = None,
                      pdf_resolution: Optional[dict] = None,
                      preferred_repo_url: str = "",
                      experiment_request: Optional[dict] = None,
                      author_execution_only: bool = True,
                      strict_validation: bool = False,
                      source_repair: bool = False,
                      source_repair_auto: bool = False,
                      source_repair_editable: Optional[List[str]] = None,
                      _managed_runtime: bool = False,
                      _append_progress: bool = False,
                      _runtime_preparation: Optional[dict] = None) -> Dict[str, Any]:
    """后台执行论文阅读、仓库发现、环境准备、代码运行与报告流程。

    与前端解耦：不触碰 st.session_state，进度实时写入 progress_path；
    返回最终 result（与 app.py 旧 run_pipeline 结构一致，供 done 事件与
    前端结果摘要复用）。临时 PDF 由调用方负责清理。
    """
    # Old sessions and API callers may still send the removed review controls.
    # Normalize before worker handoff as well as before direct orchestration.
    use_llm_review = allow_result_summary_review = enable_optimization = False
    optimization_mode = "off"
    author_execution_only = True
    # Disabled repair controls must not reuse paths from an earlier PDF task.
    source_repair_editable = source_repair_editable if source_repair is True else []
    source_repair_auto = source_repair_auto if source_repair is True else False
    if source_repair_auto is True and source_repair_editable is None:
        source_repair_editable = []
    store = ProgressStore(progress_path, reset=not _append_progress)
    from src.experiment_request import normalize_experiment_request
    try:
        experiment_request = normalize_experiment_request(experiment_request)
    except ValueError as exc:
        return _reject_pdf_input(store, str(exc))
    from src.title_routing import resolve_title_request
    resolved = {"paper_title": paper_title, "pdf_path": pdf_path,
                "preferred_repo_url": preferred_repo_url,
                "experiment_request": experiment_request,
                "corpus_paper": corpus_paper, "mock_mode": mock_mode,
                "author_execution_only": author_execution_only,
                "strict_validation": strict_validation,
                "source_repair": source_repair,
                "source_repair_auto": source_repair_auto,
                "source_repair_editable": source_repair_editable,
                "experiment_profile": experiment_profile}
    pdf_parser_reason = None
    if pdf_path:
        from src.pdf_input import resolve_pdf_request, PDFInputError, PDFParserUnavailable
        try:
            resolved = resolve_pdf_request(resolved)
            paper_title = resolved.get("paper_title", paper_title)
            pdf_resolution = resolved.get("pdf_resolution")
        except PDFParserUnavailable as exc:
            if not _managed_runtime and not mock_mode and not use_docker:
                pdf_parser_reason = str(exc)
            else:
                return _reject_pdf_input(store, str(exc))
        except PDFInputError as exc:
            return _reject_pdf_input(store, str(exc))
    if not (strict_validation is True and not pdf_path and not experiment_profile):
        resolved = resolve_title_request(resolved)
    experiment_profile = resolved.get("experiment_profile")
    from src.source_repair import normalize_source_repair_request
    try:
        normalize_source_repair_request({**resolved, 'source_repair': source_repair,
            'source_repair_auto': source_repair_auto,
            'source_repair_editable': source_repair_editable, 'strict_validation': strict_validation})
    except ValueError as exc:
        return _reject_pdf_input(store, str(exc))
    title_resolution = title_resolution or resolved.get("title_resolution")
    if title_resolution:
        store.emit({"type": "title_resolution", **title_resolution})
    if pdf_resolution:
        store.emit({"type": "pdf_resolution", **pdf_resolution})
    if not mock_mode and not use_docker and not _managed_runtime:
        from src.runtime_preparation import runtime_requirement
        from frontend.task_control import current_task
        reason = pdf_parser_reason or runtime_requirement()
        if not reason and current_task() is not None:
            reason = "正在核验可中断的独立后台运行环境"
        if reason:
            request = {
                "progress_path": str(Path(progress_path).resolve()),
                "paper_title": paper_title, "pdf_path": pdf_path,
                "preferred_repo_url": preferred_repo_url,
                "experiment_request": experiment_request,
                "corpus_paper": corpus_paper, "model_name": model_name,
                "author_execution_only": author_execution_only,
                "strict_validation": strict_validation,
                "source_repair": source_repair,
                "source_repair_auto": source_repair_auto,
                "source_repair_editable": source_repair_editable,
                "base_url": base_url, "api_key": api_key, "mock_mode": mock_mode,
                "max_trials": max_trials, "use_docker": use_docker,
                "workspace_dir": workspace_dir, "experiment_profile": experiment_profile,
                "use_llm_review": use_llm_review,
                "allow_result_summary_review": allow_result_summary_review,
                "enable_optimization": enable_optimization,
                "prepare_environment": prepare_environment, "optimization_mode": optimization_mode,
                "max_candidates": max_candidates, "budget_seconds": budget_seconds,
                "prepare_only": prepare_only, "offline": offline,
                "title_resolution": title_resolution,
                "pdf_resolution": pdf_resolution,
            }
            return _run_prepared_pipeline(store, request, reason)

    pipeline_logger = None

    def execute():
        nonlocal pdf_path, pipeline_logger
        state: Dict[str, Any] = {"current": "INIT"}
        running: Dict[str, Any] = {"value": True}
        pdf_path = pdf_path or ""

        def _current() -> str:
            return state["current"]

        def _set_current(s: str) -> None:
            state["current"] = s

        def _mark_result(result: Dict[str, Any]) -> Dict[str, Any]:
            running["value"] = False
            return result

        logger = pipeline_logger = AuditLogger()
        error_msg: Optional[str] = None
        emitted = 0
        log_emit_lock = threading.RLock()

        def _emit_new_logs() -> None:
            """只把新增的审计日志追加进进度文件，避免每次全量重发造成重复。"""
            nonlocal emitted
            with log_emit_lock:
                entries = list(logger.get_summary())
                for entry in entries[emitted:]:
                    store.emit({"type": "log", "phase_id": state.get("phase_id", ""), "log": entry})
                    emitted += 1

        # Flush on every audit entry, including downloads, LLM requests and repair
        # attempts inside a long phase. State events remain a compatibility fallback.
        logger.on_entry = lambda entry: _emit_new_logs()

        llm = LLMClient(
            mock_mode=mock_mode,
            model="" if mock_mode else model_name,
            base_url=base_url,
            api_key=api_key,
        )
        orchestrator = _create_orchestrator(llm_client=llm, mock_mode=mock_mode,
                                    logger=logger, max_trials=max_trials,
                                    use_docker=use_docker,
                                    workspace_dir=workspace_dir)

        display_names = {"PaperReader": "📖 PaperReader", "ResourceFinder": "🔍 ResourceFinder",
                         "EnvBuilder": "🔧 EnvBuilder", "CodeExecutor": "⚡ CodeExecutor",
                         "ResultValidator": "✅ ResultValidator", "Optimizer": OPTIMIZER_NAME,
                         "ReportGenerator": REPORTER_NAME, "Verifier": VERIFIER_NAME}

        def _on_event(event):
            if event.get("type") == "state":
                _set_current(event["state"])
                if event.get("phase_id"):
                    state["phase_id"] = event["phase_id"]
                store.emit({**event, "agent": display_names.get(event.get("agent"), event.get("agent", ""))})
            elif event.get("type") in {"repository_step", "repository_output", "execution_step", "execution_output", "execution_run", "repository_environment", "dependency_install_progress"}:
                store.emit(event)
            elif event.get("type") == "pipeline_plan":
                store.emit({**event, "stages": [
                    {**stage, "agent": display_names.get(stage.get("agent"), stage.get("agent", ""))}
                    for stage in event.get("stages", [])]})
            _emit_new_logs()

        try:
            outcome = orchestrator.run({
                "paper_title": paper_title, "pdf_path": pdf_path,
                "preferred_repo_url": preferred_repo_url,
                "experiment_request": experiment_request,
                "corpus_paper": corpus_paper, "experiment_profile": experiment_profile,
                "author_execution_only": author_execution_only,
                "strict_validation": strict_validation,
                "source_repair": source_repair,
                "source_repair_auto": source_repair_auto,
                "source_repair_editable": source_repair_editable,
                "prepare_only": prepare_only, "offline": offline,
                "use_llm_review": use_llm_review,
                "allow_result_summary_review": allow_result_summary_review,
                "enable_optimization": enable_optimization,
                "prepare_environment": prepare_environment, "optimization_mode": optimization_mode,
                "max_candidates": max_candidates, "budget_seconds": budget_seconds,
                "title_resolution": title_resolution,
                "pdf_resolution": pdf_resolution,
            }, on_event=_on_event)
            data = outcome["data"]
            error_msg = outcome.get("error")
            _set_current(outcome["state"])
        except Exception as exc:
            error_msg = f"流水线异常: {exc}"
            logger.log("Orchestrator", "run", "ERROR", error_msg)
            data = getattr(orchestrator, "data", {})
            _set_current("ERROR")
        if _runtime_preparation:
            data["runtime_preparation"] = _runtime_preparation
        if title_resolution:
            data["title_resolution"] = title_resolution
        if pdf_resolution:
            data["pdf_resolution"] = pdf_resolution
        _emit_new_logs()

    # 报告落盘（供历史记录与下载）
        report_path = ""
        report_text = data.get("report", "")
        if report_text:
            try:
                reports_dir = get_data_root() / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                # 使用流水线 session_id 作为文件名时间戳，与 ledger 文件名保持一致
                ts = logger.session_id
                paper_title_safe = (data.get("paper_info", {}).get("title", "") or data.get("paper_title", "") or "report")[:30]
                # 去除非法字符
                paper_title_safe = "".join(c for c in paper_title_safe if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
                report_file = reports_dir / f"{paper_title_safe}_{ts}.md"
                from src.agents.report_generator import ReportGeneratorAgent
                data["report_path"] = str(report_file.resolve())
                report_text = ReportGeneratorAgent(logger).run(data, report_path=report_file)["report"]
                report_file.write_text(report_text, encoding="utf-8")
                report_path = str(report_file)

                # 完整执行输出附件：报告现已**全文内嵌** code/stdout/stderr，
                # 附件保留作为可直接下载/归档的纯文本旁路（终端里 wc/grep、
                # 或想脱离 Markdown 单独留档时用）。
                execution_raw = data.get("execution", {}) or {}
                final_raw = execution_raw.get("final", {}) or {}
                attach_lines = ["# 完整执行输出（未被截断）",
                                "", "## 生成代码", "```python",
                                execution_raw.get("code", "（无）"), "```",
                                "", "## 标准输出 (full)", "```",
                                final_raw.get("stdout", "（无输出）"), "```",
                                "", "## 诊断输出 (stderr)", "```",
                                final_raw.get("stderr", "（无）"), "```", ""]
                attach_file = reports_dir / f"{paper_title_safe}_{ts}_execution.txt"
                attach_file.write_text("\n".join(attach_lines), encoding="utf-8")
                # 在报告末尾补充附件说明，保证用户知道完整输出的位置
                report_text += (f"\n\n---\n\n> 📎 **完整执行输出附件**: "
                                f"`{attach_file.name}`（同一目录下，未被截断）\n")
                report_file.write_text(report_text, encoding="utf-8")
                data.update(report=report_text, report_length=len(report_text))
            except Exception as exc:
                reason = "最终报告保存失败: " + str(exc)
                error_msg = "; ".join(filter(None, [error_msg, reason]))
                data["report_publication"] = {"status": "error", "reason": reason}
                _set_current("ERROR")
                logger.log("ReportGenerator", "publish_report", "ERROR", reason)

        stats = logger.get_stats()
        result = {
            "state": _current(),
            "error": error_msg,
            "data": data,
            "audit_logs": logger.get_summary(),
            "audit_stats": stats,
            "report_path": report_path,
            "session_id": logger.session_id,
        }
        if report_path:
            # Progress files are disposable; retain the structured inputs alongside
            # the report so history can regenerate it without rerunning the paper.
            def publication_error(exc):
                reason = "最终报告与运行归档未完整保存: " + str(exc)
                result["state"] = "ERROR"
                result["error"] = "; ".join(filter(None, [result.get("error"), reason]))
                data["report_publication"] = {"status": "error", "reason": reason}
                _set_current("ERROR")
                store.emit({"type": "report_archive", "status": "error", "reason": reason})

            try:
                _sync_run_report(result)
            except (OSError, TypeError, ValueError) as exc:
                publication_error(exc)
            temporary = None
            try:
                from src.report_revisions import sanitize_report_result
                archive = Path(report_path).with_suffix(".result.json")
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=archive.parent,
                                                 suffix=".tmp", delete=False) as output:
                    temporary = Path(output.name)
                    json.dump(sanitize_report_result(result), output, ensure_ascii=False, indent=2)
                os.replace(temporary, archive)
            except (OSError, TypeError, ValueError) as exc:
                publication_error(exc)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return _mark_result(result)

    # Runtime recovery above may launch a different interpreter. Only its worker
    # takes this lease, avoiding a parent/child deadlock on the dependency lock.
    from src.environment_lifecycle import managed_environment
    environment = None
    try:
        with managed_environment() as environment:
            environment.result = execute()
    except BaseException as exc:
        # Direct callers need the same persisted failed outcome as the background
        # wrapper when cleanup/report publication raises after execution ended.
        result = environment.result if environment is not None else None
        if isinstance(result, dict) and result.get("state") == "ERROR":
            try:
                _record_pipeline_finish(result, logger=pipeline_logger)
            except Exception:
                pass  # Keep the original cancellation/publication exception.
            store.emit({"type": "done", "result": result})
        else:
            from src.runtime_preparation import safe_runtime_diagnostic
            store.emit({"type": "error", "error": safe_runtime_diagnostic(exc)})
        raise
    result = environment.result
    _record_pipeline_finish(result, logger=pipeline_logger)
    _emit_state(store, result["state"], "", "error" if result["state"] == "ERROR" else "success")
    store.emit({"type": "done", "result": result})
    return result


def _run_prepared_pipeline(store: ProgressStore, request: dict, reason: str, *, cancel_check=None,
                           cancel_file=None, cancel_token=None) -> dict:
    """Recover the host, then continue this reproduction in an owned interpreter.

    Only the request pipe carries credentials. The existing progress file remains
    the single source of truth for the page throughout recovery and training.
    """
    from src.runtime_preparation import (
        prepare_runtime, run_owned_process, safe_runtime_diagnostic, release_worker_environment,
    )
    from frontend.task_control import current_task
    control = current_task()
    if control is not None and cancel_check is None:
        cancel_check, cancel_file, cancel_token = control.cancelled, control.cancel_file, control.token
    stage = {"id": "prepare_runtime", "agent": "🔧 EnvBuilder",
             "title": "自动准备运行环境", "description": reason, "status": "waiting"}
    store.emit({"type": "pipeline_plan", "pipeline": "repository" if request.get("experiment_profile") else "generic", "stages": [stage]})
    def state(status, message):
        store.emit({"type": "state", "state": "BUILD_ENV", "agent": stage["agent"],
                    "phase_id": stage["id"], "status": status, "reason": message})
    state("running", reason)
    metadata = None
    prepared = None
    supervisor_cleanup = None
    worker_id = "worker_" + uuid.uuid4().hex
    worker_started = False
    worker_cleaned = False
    worker_finished = False
    forced_cleanup_unconfirmed = False

    def cancel_installations():
        snapshot = ProgressStore.read_snapshot(str(store.path))
        for installation in snapshot.get("dependency_installations", []):
            if installation.get("status") == "running":
                store.emit({"type": "dependency_install_progress", **installation,
                            "status": "cancelled", "message": "用户停止任务，安装进程已结束"})

    def confirmed_cleanup():
        nonlocal worker_cleaned
        worker_cleaned = not forced_cleanup_unconfirmed

    def finish_worker(status, error_message=None):
        nonlocal worker_finished, supervisor_cleanup
        if not worker_started or worker_finished:
            return
        # Only the process owner can confirm cleanup after its retained job has
        # closed. A stale child's log cannot prove that its processes exited.
        if worker_cleaned:
            if status == "interrupted":
                cancel_installations()
            try:
                supervisor_cleanup = release_worker_environment(prepared)
            except Exception as exc:
                supervisor_cleanup = {"status": "error", "errors": [
                    {"reason": safe_runtime_diagnostic(exc)}]}
            if control is not None:
                control.cleanup_receipt = supervisor_cleanup
            if supervisor_cleanup.get("errors"):
                status = "error"
                store.emit({"type": "environment_cleanup", "status": "error",
                            "reason": "工作进程环境自动释放失败：" + str(supervisor_cleanup["errors"])})
            snapshot = ProgressStore.read_snapshot(str(store.path))
            for run in snapshot.get("active_runs", []):
                store.emit({"type": "execution_run", "execution_id": run["execution_id"],
                            "phase_id": run.get("phase_id", ""), "status": "interrupted",
                            "cleanup_confirmed": True,
                            "reason": "持有者进程已退出并清理，但未返回本轮执行终态"})
        try:
            if worker_cleaned:
                snapshot = ProgressStore.read_snapshot(str(store.path), include_pending_result=True)
                result = snapshot.get("pending_result")
                if isinstance(result, dict):
                    # Keep the owned scope open until every saved copy includes
                    # the supervisor receipt. Closing it first publishes the
                    # child's earlier done while these files are still changing.
                    from src.environment_lifecycle import record_environment_cleanup
                    if error_message:
                        result["state"] = "ERROR"
                        result["error"] = "; ".join(filter(None, [result.get("error"), error_message]))
                        if cancel_check and cancel_check():
                            result.setdefault("data", {})["run_status"] = "interrupted"
                    elif status != "success" and result.get("state") != "ERROR":
                        result.update(state="ERROR", error="后台工作进程未正常结束，不能确认流水线成功")
                    receipt = result.setdefault("data", {}).get("environment_cleanup")
                    if receipt is None:
                        receipt = supervisor_cleanup
                    else:
                        receipt["supervisor_cleanup"] = supervisor_cleanup
                        if supervisor_cleanup.get("errors"):
                            receipt["status"] = "error"
                            receipt.setdefault("errors", []).extend(supervisor_cleanup["errors"])
                    record_environment_cleanup(result, receipt)
                    _record_pipeline_finish(result)
                    store.emit({"type": "done", "result": result})
        except Exception as exc:
            status = "error"
            # Invalidate the child's pending completion before closing its
            # scope, even when publishing the final receipts itself fails.
            store.emit({"type": "error", "error": safe_runtime_diagnostic(exc)})
            raise
        finally:
            store.emit({"type": "pipeline_worker", "worker_id": worker_id,
                        "status": status, "cleanup_confirmed": worker_cleaned})
            worker_finished = worker_cleaned

    effective_key = request.get("api_key") or os.environ.get("LLM_API_KEY", "")
    try:
        prepared = prepare_runtime(Path(__file__).resolve().parents[1],
                                   offline=request.get("offline", False),
                                   cancel_check=cancel_check,
                                   progress_callback=store.emit)
        metadata = asdict(prepared)
        state("success", "兼容运行环境已就绪，继续本次复现")
        store.emit({"type": "runtime_preparation", "status": "success", **metadata})
        payload = {**request, "api_key": effective_key,
                   "_managed_runtime": True, "_append_progress": True,
                   "_runtime_preparation": metadata}
        if cancel_file:
            payload.update(_cancel_file=str(cancel_file), _cancel_token=cancel_token)
        cancellation_started = None

        def worker_cancelled():
            # Give the worker time to unwind its installer and save interruption
            # records. A wedged native call is still bounded by the owned job.
            nonlocal cancellation_started, forced_cleanup_unconfirmed
            if not cancel_check or not cancel_check():
                return False
            if cancellation_started is None:
                cancellation_started = time.monotonic()
                if cancel_file:
                    Path(cancel_file).write_text(cancel_token, encoding="ascii")
            force_stop = not cancel_file or time.monotonic() - cancellation_started >= WORKER_CANCEL_GRACE_SECONDS
            if force_stop and not _force_stop_contains_worker_children():
                forced_cleanup_unconfirmed = True
            return force_stop
        worker_env = os.environ.copy()
        worker_env.pop("LLM_API_KEY", None)
        worker_started = True
        store.emit({"type": "pipeline_worker", "worker_id": worker_id, "status": "running"})
        completed = run_owned_process(
            [prepared.executable, "-X", "utf8", "-m", "frontend.preset_worker"],
            cwd=Path(__file__).resolve().parents[1], input=json.dumps(payload), env=worker_env,
            # The worker owns training as well as preparation. A legacy budget
            # must never stop an author command that is still running.
            timeout_s=None,
            cancel_check=worker_cancelled,
            on_cleanup=confirmed_cleanup,
        )
        # Normal return also carries the run_owned_process cleanup contract.
        confirmed_cleanup()
        if not worker_cleaned:
            from src.runtime_preparation import ProcessCleanupError
            raise ProcessCleanupError("外层进程退出不足以确认独立会话中的子进程已回收")
        cancelled = bool(cancel_check and cancel_check())
        finish_worker("interrupted" if cancelled else "success" if completed.returncode == 0 else "error",
                      "用户已停止本任务" if cancelled else None)
        snapshot = ProgressStore.read_snapshot(str(store.path))
        if snapshot.get("done") and isinstance(snapshot.get("result"), dict):
            return snapshot["result"]
        if snapshot.get("error"):
            raise RuntimeError(snapshot["error"])
        diagnostic = safe_runtime_diagnostic(completed.stderr or completed.stdout)
        raise RuntimeError(f"安全运行环境进程未返回完整结果（退出码 {completed.returncode}）：{diagnostic[-1600:]}")
    except (Exception, KeyboardInterrupt) as exc:
        cancelled = isinstance(exc, KeyboardInterrupt) or bool(cancel_check and cancel_check())
        unconfirmed = forced_cleanup_unconfirmed or getattr(exc, "process_cleanup_confirmed", True) is False
        diagnostic = safe_runtime_diagnostic(exc)
        cancellation_message = "用户已停止本任务" + ("：" + diagnostic if diagnostic and diagnostic != "用户已停止本任务" else "")
        message = ("本任务子进程回收未确认，已阻止新任务；请停止原服务后恢复" if unconfirmed else
                   cancellation_message if cancelled else diagnostic)
        failure_receipt = getattr(exc, "cleanup_receipt", None)
        if forced_cleanup_unconfirmed:
            failure_receipt = {"status": "error", "process_cleanup_confirmed": False,
                "errors": [{"reason": "POSIX 强制停止仅回收外层进程组，独立会话中的后代进程回收未确认"}]}
            exc.process_cleanup_confirmed = False
            exc.cleanup_receipt = failure_receipt
        if isinstance(failure_receipt, dict):
            supervisor_cleanup = failure_receipt
            if control is not None:
                control.cleanup_receipt = failure_receipt
        if cancelled and not worker_started and not unconfirmed:
            cancel_installations()
        if effective_key:
            message = message.replace(effective_key, "[REDACTED]")
        finish_worker("interrupted" if isinstance(exc, KeyboardInterrupt) else "error", message)
        snapshot = ProgressStore.read_snapshot(str(store.path))
        if snapshot.get("done") and isinstance(snapshot.get("result"), dict):
            # A child may have already saved its report before its owner was
            # interrupted. Its final receipt and retained evidence still belong
            # to this failed run; do not replace them with a bootstrap stub.
            if isinstance(exc, KeyboardInterrupt):
                raise
            return snapshot["result"]
        if metadata is None:
            state("error", message)
        else:
            store.emit({"type": "error", "error": message})
        result = {"state": "ERROR", "error": message,
                  "data": {"runtime_preparation": metadata or {"status": "failed", "reason": message}},
                  "audit_logs": [], "audit_stats": {}, "report_path": "", "session_id": ""}
        if supervisor_cleanup is not None:
            result["data"]["environment_cleanup"] = supervisor_cleanup
        if cancelled:
            result["data"]["run_status"] = "interrupted"
        store.emit({"type": "done", "result": result})
        if isinstance(exc, KeyboardInterrupt):
            raise
        return result


def run_pipeline_background(progress_path: str, *,
                            paper_title: str = "", pdf_path: str = "",
                            corpus_paper: Optional[str] = None,
                            model_name: str = "", base_url: str = "",
                            api_key: str = "", mock_mode: bool = False,
                            max_trials: int = 10,
                            use_docker: bool = False,
                            workspace_dir: Optional[str] = None,
                            experiment_profile: Optional[str] = None,
                            use_llm_review: bool = False,
                            allow_result_summary_review: bool = False,
                            enable_optimization: bool = False,
                            prepare_environment: bool = False,
                            optimization_mode: str = "off",
                            max_candidates: int = 3,
                            budget_seconds: int = 7200,
                            prepare_only: bool = False,
                            offline: bool = False,
                            title_resolution: Optional[dict] = None,
                            pdf_resolution: Optional[dict] = None,
                            preferred_repo_url: str = "",
                            experiment_request: Optional[dict] = None,
                            author_execution_only: bool = True,
                            strict_validation: bool = False,
                            source_repair: bool = False,
                            source_repair_auto: bool = False,
                            source_repair_editable: Optional[List[str]] = None,
                            cleanup_pdf: bool = True,
                            on_done=None) -> threading.Thread:
    """启动后台线程执行流水线；返回守护线程句柄。

    线程内捕获一切异常并写入 error 事件；临时 PDF 默认在线程结束时删除
    （cleanup_pdf）。on_done(result) 在线程收尾时回调（可选）。
    """
    from frontend.task_control import register_task, bind_task
    control = register_task(progress_path)
    try:
        store = ProgressStore(progress_path)
    except BaseException:
        control.finish()
        raise

    def _worker() -> None:
        bind_task(control)
        try:
            request = dict(
                progress_path=str(Path(progress_path).resolve()), paper_title=paper_title, pdf_path=pdf_path,
                preferred_repo_url=preferred_repo_url,
                experiment_request=experiment_request,
                author_execution_only=author_execution_only,
                strict_validation=strict_validation,
                source_repair=source_repair,
                source_repair_auto=source_repair_auto if source_repair is True else False,
                source_repair_editable=source_repair_editable if source_repair is True else [],
                corpus_paper=corpus_paper, model_name=model_name,
                base_url=base_url, api_key=api_key, mock_mode=mock_mode,
                max_trials=max_trials, use_docker=use_docker,
                workspace_dir=workspace_dir, experiment_profile=experiment_profile,
                prepare_only=prepare_only, offline=offline, use_llm_review=use_llm_review,
                allow_result_summary_review=allow_result_summary_review,
                enable_optimization=enable_optimization, prepare_environment=prepare_environment,
                optimization_mode=optimization_mode, max_candidates=max_candidates, budget_seconds=budget_seconds,
                title_resolution=title_resolution, pdf_resolution=pdf_resolution)
            if control.cancelled():
                raise KeyboardInterrupt("用户已停止本任务")
            # Keep PDF/request validation before the handoff. The bound task
            # forces real work into an owned process even on a healthy host.
            result = run_pipeline_core(**request)
            if on_done:
                try:
                    on_done(result)
                except Exception:
                    pass
        except (Exception, KeyboardInterrupt) as e:
            from src.runtime_preparation import safe_runtime_diagnostic
            diagnostic = safe_runtime_diagnostic(e)
            cancellation_message = "用户已停止本任务" + ("：" + diagnostic if diagnostic and diagnostic != "用户已停止本任务" else "")
            message = ("本任务子进程回收未确认，已阻止新任务；请停止原服务后恢复"
                       if getattr(e, "process_cleanup_confirmed", True) is False else
                       cancellation_message if isinstance(e, KeyboardInterrupt) else diagnostic)
            if api_key:
                message = message.replace(api_key, "[REDACTED]")
            store.emit({"type": "error", "error": message})
            if on_done:
                try:
                    on_done(None)
                except Exception:
                    pass
        finally:
            if cleanup_pdf and pdf_path:
                try:
                    if os.path.exists(pdf_path):
                        os.unlink(pdf_path)
                except OSError:
                    pass
            snapshot = ProgressStore.read_snapshot(progress_path)
            receipt = ((snapshot.get("result") or {}).get("data") or {}).get("environment_cleanup") or {}
            receipt = control.cleanup_receipt or receipt
            control.finish(cleanup_failed=bool(receipt.get("errors")),
                           process_cleanup_confirmed=receipt.get("process_cleanup_confirmed", True))
            bind_task(None)

    t = threading.Thread(target=_worker, name="autorepro-pipeline",
                         daemon=True)
    control.thread = t
    try:
        t.start()
    except BaseException:
        control.finish()
        raise
    return t
